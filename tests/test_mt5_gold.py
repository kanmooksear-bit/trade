"""XAUUSD / MetaTrader 5 behaviour, tested against a fake MetaTrader5 module
(the real package only runs on Windows next to an MT5 terminal)."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autotrader import mt5_connector
from autotrader.backtest import run_backtest
from autotrader.broker import MT5Broker, PaperBroker, build_broker
from autotrader.config import load_config
from autotrader.data import DataSource, load_csv, synthetic_ohlcv
from autotrader.engine import Order, TradingEngine
from autotrader.journal import Journal
from autotrader.report import performance
from autotrader.strategies import STRATEGY_CLASSES

PRESET = Path(__file__).resolve().parent.parent / "config.xauusd.example.yaml"


class FakeMT5:
    TIMEFRAME_H1, TIMEFRAME_D1 = 16385, 16408
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE = 10009
    DEAL_ENTRY_IN, DEAL_ENTRY_OUT, DEAL_ENTRY_OUT_BY = 0, 1, 3
    DEAL_REASON_SL, DEAL_REASON_TP, DEAL_REASON_SO = 4, 5, 6

    def __init__(self):
        self.requests = []
        self.positions = {}
        self.deals = {}
        self.position_deals = {}
        self.next_ticket = 1000
        self.bid, self.ask = 2650.10, 2650.45
        self.rates_calls = []
        self.reject_sltp = False

    def symbol_info(self, symbol):
        return SimpleNamespace(visible=True, digits=2, point=0.01, trade_contract_size=100.0, volume_min=0.01,
                               volume_max=50.0, volume_step=0.01, filling_mode=2, trade_stops_level=0)

    def symbol_select(self, symbol, enable):
        return True

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self.bid, ask=self.ask, time=1_760_000_000)

    def terminal_info(self):
        return SimpleNamespace(trade_allowed=True)

    def account_info(self):
        return SimpleNamespace(login=123, balance=1000.0, equity=1000.0, margin_free=900.0, currency="USD")

    def last_error(self):
        return (0, "ok")

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        self.rates_calls.append((symbol, timeframe, start_pos, count))
        t0 = 1_700_000_000
        dtype = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"),
                 ("tick_volume", "i8"), ("spread", "i4"), ("real_volume", "i8")]
        rows = [(t0 + 3600 * i, 2600 + i, 2601 + i, 2599 + i, 2600.5 + i, 100 + i, 30, 0) for i in range(count)]
        return np.array(rows, dtype=dtype)

    def order_send(self, request):
        self.requests.append(request)
        if request["action"] == self.TRADE_ACTION_SLTP and self.reject_sltp:
            return SimpleNamespace(retcode=10016, deal=0, order=0, volume=0, price=0, comment="Invalid stops")
        self.next_ticket += 1
        ticket = self.next_ticket
        if request["action"] == self.TRADE_ACTION_DEAL:
            self.deals[ticket] = SimpleNamespace(commission=-3.0, fee=0.0, swap=0.0)
            if "position" in request:
                self.positions.pop(request["position"], None)
                self.deals[ticket].swap = -1.5
            else:
                self.positions[ticket] = SimpleNamespace(ticket=ticket, volume=request["volume"], magic=request["magic"])
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, deal=ticket, order=ticket, volume=request.get("volume"),
                               price=request.get("price"), comment="done")

    def positions_get(self, ticket=None, symbol=None):
        if ticket is not None:
            return tuple(p for t, p in self.positions.items() if t == ticket)
        return tuple(self.positions.values())

    def history_deals_get(self, ticket=None, position=None):
        if ticket is not None:
            return (self.deals[ticket],) if ticket in self.deals else ()
        return self.position_deals.get(position, ())


@pytest.fixture
def mt5():
    fake = FakeMT5()
    mt5_connector.set_module(fake)
    yield fake
    mt5_connector.set_module(None)


@pytest.fixture
def gold_cfg():
    return load_config(PRESET, {
        "data": {"source": "synthetic", "symbols": ["XAUUSD"]},
        "storage": {"db_path": ":memory:"}})


def test_mt5_data_skips_forming_bar_and_maps_timeframe(mt5, gold_cfg):
    df = DataSource({**gold_cfg, "data": {**gold_cfg["data"], "source": "mt5"}}).fetch("XAUUSD", bars=50)
    assert mt5.rates_calls == [("XAUUSD", FakeMT5.TIMEFRAME_H1, 1, 50)]
    assert len(df) == 50 and list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_mt5_open_sends_lots_and_server_side_stops(mt5, gold_cfg):
    broker = MT5Broker(gold_cfg["broker"], {"cash": 1000.0}, gold_cfg)
    assert broker.normalize_qty("XAUUSD", 2.57) == pytest.approx(2.0)  # 0.0257 lot -> 0.02 lot
    assert broker.normalize_qty("XAUUSD", 0.5) == 0.0  # below 0.01 lot
    fill = broker.open("XAUUSD", 1, 3.0, 0.0, stop_dist=12.3456, tp_dist=30.0)
    req = mt5.requests[-1]
    assert req["volume"] == 0.03 and req["type"] == FakeMT5.ORDER_TYPE_BUY and req["price"] == 2650.45
    assert req["sl"] == 2638.10 and req["tp"] == 2680.45  # rounded to 2 digits
    assert req["type_filling"] == FakeMT5.ORDER_FILLING_IOC and req["magic"] == gold_cfg["broker"]["magic"]
    assert fill.ticket in mt5.positions and fill.qty == pytest.approx(3.0) and fill.fee == pytest.approx(3.0)


def test_mt5_close_modify_and_reconcile(mt5, gold_cfg):
    broker = MT5Broker(gold_cfg["broker"], {"cash": 1000.0}, gold_cfg)
    fill = broker.open("XAUUSD", -1, 2.0, 0.0, stop_dist=10.0, tp_dist=None)
    assert broker.modify("XAUUSD", fill.ticket, 2655.123, None)
    assert mt5.requests[-1]["action"] == FakeMT5.TRADE_ACTION_SLTP and mt5.requests[-1]["sl"] == 2655.12
    closing = broker.close("XAUUSD", -1, 2.0, 0.0, fill.ticket)
    req = mt5.requests[-1]
    assert req["position"] == fill.ticket and req["type"] == FakeMT5.ORDER_TYPE_BUY and req["price"] == 2650.45
    assert closing.fee == pytest.approx(4.5)  # commission 3 + swap 1.5

    second = broker.open("XAUUSD", 1, 1.0, 0.0, stop_dist=10.0, tp_dist=None)
    del mt5.positions[second.ticket]  # the server hit the stop-loss
    mt5.position_deals[second.ticket] = (
        SimpleNamespace(entry=FakeMT5.DEAL_ENTRY_IN, price=2650.45, volume=0.01, commission=-3.0, fee=0.0, swap=0.0,
                        reason=0, time=1),
        SimpleNamespace(entry=FakeMT5.DEAL_ENTRY_OUT, price=2640.45, volume=0.01, commission=0.0, fee=0.0, swap=-0.5,
                        reason=FakeMT5.DEAL_REASON_SL, time=1_760_003_600))
    pos = SimpleNamespace(trade_id=7, ticket=second.ticket, stop_moved=False)
    [closed] = broker.reconcile([pos])
    assert (closed.trade_id, closed.price, closed.reason, closed.fee) == (7, 2640.45, "stop", 0.5)


def test_engine_leaves_stops_to_mt5_and_records_server_exit(mt5, gold_cfg):
    cfg = {**gold_cfg, "mode": "live", "broker": {**gold_cfg["broker"], "type": "mt5"}}
    engine = TradingEngine(cfg, Journal(":memory:"))
    assert isinstance(engine.broker, MT5Broker)
    params = {**STRATEGY_CLASSES["trend"].default_params(), "trail_atr": 1.0, "breakeven_at_r": 0.0}
    engine.fill_orders([Order("entry", "XAUUSD", 1, qty=1.0, stop_dist=10.0, tp_dist=None, strategy="trend",
                              regime="trend_up", features={"atr": 5.0}, params=params)],
                       {"XAUUSD": 2650.0}, "t0", {"XAUUSD": "2025-01-01 00:00:00+00:00"})
    pos = engine.positions["XAUUSD"]
    bar = pd.Series({"open": 2650.0, "high": 2662.0, "low": 2630.0, "close": 2660.0})
    assert not engine._process_bar(pos, "2025-01-01 01:00:00+00:00", bar, 5.0)  # bar crossed SL: MT5 decides
    assert pos.stop == 2657.0 and mt5.requests[-1]["action"] == FakeMT5.TRADE_ACTION_SLTP  # trailing sent to server
    del mt5.positions[pos.ticket]
    mt5.position_deals[pos.ticket] = (SimpleNamespace(entry=FakeMT5.DEAL_ENTRY_OUT, price=2657.0, volume=0.01,
                                                      commission=0.0, fee=0.0, swap=0.0,
                                                      reason=FakeMT5.DEAL_REASON_SL, time=1_760_000_000),)
    engine.check_stops({}, "t2")
    trade = engine.journal.closed_trades()[0]
    assert "XAUUSD" not in engine.positions and trade["exit_reason"] == "trail" and trade["exit_price"] == 2657.0


def test_rejected_trailing_stop_beyond_price_closes_at_market(mt5, gold_cfg):
    cfg = {**gold_cfg, "mode": "live", "broker": {**gold_cfg["broker"], "type": "mt5"}}
    engine = TradingEngine(cfg, Journal(":memory:"))
    params = {**STRATEGY_CLASSES["trend"].default_params(), "trail_atr": 1.0, "breakeven_at_r": 0.0}
    engine.fill_orders([Order("entry", "XAUUSD", 1, qty=1.0, stop_dist=10.0, tp_dist=None, strategy="trend",
                              regime="trend_up", features={"atr": 5.0}, params=params)],
                       {"XAUUSD": 2650.0}, "t0", {"XAUUSD": "2025-01-01 00:00:00+00:00"})
    pos = engine.positions["XAUUSD"]
    mt5.reject_sltp = True
    spike_and_drop = pd.Series({"open": 2650.0, "high": 2670.0, "low": 2648.0, "close": 2660.0})  # trail 2665
    assert engine._process_bar(pos, "2025-01-01 01:00:00+00:00", spike_and_drop, 5.0)
    trade = engine.journal.closed_trades()[0]
    assert trade["exit_reason"] == "trail" and mt5.requests[-1]["position"] == pos.ticket


def test_real_money_brokers_need_explicit_live_mode(mt5, gold_cfg):
    with pytest.raises(RuntimeError):
        build_broker({**gold_cfg, "broker": {**gold_cfg["broker"], "type": "mt5"}}, {})


def test_paper_broker_models_cfd_costs(gold_cfg):
    b = PaperBroker({**gold_cfg["broker"], "commission_per_lot": 3.5}, {})
    fill = b.open("XAUUSD", 1, 2.0, 2650.0, stop_dist=10.0, tp_dist=20.0)
    assert fill.price == pytest.approx(2650.25, abs=0.01)  # half the 0.35 spread + tiny slippage, 2 digits
    assert fill.price == round(fill.price, 2) and fill.sl == round(fill.price - 10.0, 2)
    assert fill.fee == pytest.approx(3.5 * 0.02)


def test_min_lot_rule_and_probe_cap(gold_cfg):
    engine = TradingEngine(gold_cfg, Journal(":memory:"), broker=PaperBroker(gold_cfg["broker"], {}))
    reading = SimpleNamespace(features={"atr": 6.0}, regime="trend_up")
    signal = SimpleNamespace(params=STRATEGY_CLASSES["trend"].default_params(), direction=1, strategy="trend",
                             strength=0.8, reason="")
    cand = SimpleNamespace(symbol="XAUUSD", signal=signal, reading=reading, weight=1.0, score=0.8)
    # stop 2.5 ATR = 15 USD: 1% of 1000 is 0.0067 lot -> 0.01 lot allowed (1.5% risk <= 2%)
    order = engine._entry_order(cand, 1000.0, {"XAUUSD": 2650.0}, 0.0, forced=False)
    assert order.qty == pytest.approx(1.0)
    # a forced probe may not exceed the normal 1% risk
    assert engine._entry_order(cand, 1000.0, {"XAUUSD": 2650.0}, 0.0, forced=True) is None
    reading.features["atr"] = 20.0  # 0.01 lot would risk 5%: skipped, and the log says how much capital is needed
    assert engine._entry_order(cand, 1000.0, {"XAUUSD": 2650.0}, 0.0, forced=False) is None
    assert any("ต้องมีทุนราว 5,000" in e for e in engine.events)


def test_intraday_daily_rule_waits_for_probe_hour(gold_cfg):
    cfg = load_config(PRESET, {
        "data": {"source": "synthetic"}, "broker": {"starting_cash": 20_000},
        "risk": {"max_drawdown": 0.9}, "storage": {"db_path": ":memory:"}})
    data = {"XAUUSD": synthetic_ohlcv("XAUUSD", start="2024-01-01", end="2024-03-15", start_price=2000,
                                      timeframe="1h", vol_scale=0.6, weekdays_only=True, digits=2)}
    result = run_backtest(cfg, data)
    trades = result.journal.closed_trades()
    forced = [t for t in trades if t["forced"]]
    assert forced, "expected some forced daily trades"
    probe_hour = cfg["schedule"]["probe_after_hour"]
    assert all(pd.Timestamp(t["entry_time"]).hour >= probe_hour for t in forced)
    per_day = Counter(pd.Timestamp(t["entry_time"]).date() for t in forced)
    assert max(per_day.values()) == 1
    stats = performance(result.journal, 20_000)
    assert stats["days_with_entry"] >= 0.8 * stats["days"]
    assert all(t["entry_price"] == round(t["entry_price"], 2) for t in trades)


def test_load_metatrader_csv_export(tmp_path):
    path = tmp_path / "XAUUSD_H1.csv"
    path.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                    "2024.01.02\t01:00:00\t2063.12\t2065.50\t2061.01\t2064.88\t1532\t0\t25\n"
                    "2024.01.02\t02:00:00\t2064.88\t2066.00\t2063.00\t2063.50\t1200\t0\t25\n")
    df = load_csv(path)
    assert str(df.index[0]) == "2024-01-02 01:00:00+00:00"
    assert df["close"].tolist() == [2064.88, 2063.5] and df["volume"].tolist() == [1532, 1200]


def test_performance_uses_time_not_bar_count():
    j = Journal(":memory:")
    for i, eq in enumerate(np.linspace(1000, 1100, 24 * 365 + 1)):  # one year of hourly points, +10%
        j.record_equity(str(pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)), eq, eq, 0)
    assert performance(j, 1000.0)["cagr"] == pytest.approx(0.10, abs=0.002)
