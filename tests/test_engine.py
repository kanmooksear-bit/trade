import json

import pandas as pd
import pytest

from autotrader.backtest import WINDOW, run_backtest
from autotrader.broker import PaperBroker
from autotrader.config import load_config
from autotrader.data import synthetic_ohlcv
from autotrader.engine import Order, TradingEngine
from autotrader.journal import Journal
from autotrader.llm_review import ClaudeReviewer
from autotrader.strategies import STRATEGY_CLASSES


def no_cost_cfg(**overrides):
    return load_config(overrides={"broker": {"fee_rate": 0.0, "slippage_bps": 0.0}, **overrides})


def engine_with_position(direction=1, entry=100.0, stop_dist=5.0, tp_dist=10.0, **param_overrides):
    cfg = no_cost_cfg()
    engine = TradingEngine(cfg, Journal(":memory:"), broker=PaperBroker(cfg["broker"], {}))
    params = {**STRATEGY_CLASSES["trend"].default_params(), "trail_atr": 0.0, "breakeven_at_r": 0.0,
              **param_overrides}
    order = Order("entry", "X", direction, qty=10, stop_dist=stop_dist, tp_dist=tp_dist, strategy="trend",
                  regime="trend_up", features={"atr": 2.0}, params=params)
    engine.fill_orders([order], {"X": entry}, "2024-01-01 00:00:00+00:00", {"X": "2023-12-31 00:00:00+00:00"})
    return engine, engine.positions["X"]


def bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def last_trade(engine):
    return engine.journal.closed_trades(limit=1)[0]


def test_stop_fills_at_stop_price():
    engine, pos = engine_with_position()
    assert engine._process_bar(pos, "2024-01-02", bar(99, 101, 94, 96), 2.0)
    t = last_trade(engine)
    assert t["exit_price"] == 95.0 and t["exit_reason"] == "stop" and t["r_multiple"] == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open():
    engine, pos = engine_with_position()
    engine._process_bar(pos, "2024-01-02", bar(90, 92, 88, 91), 2.0)
    assert last_trade(engine)["exit_price"] == 90.0
    assert last_trade(engine)["r_multiple"] == pytest.approx(-2.0)


def test_target_and_short_side():
    engine, pos = engine_with_position(direction=-1)
    engine._process_bar(pos, "2024-01-02", bar(99, 100, 89, 92), 2.0)
    t = last_trade(engine)
    assert t["exit_price"] == 90.0 and t["exit_reason"] == "target" and t["pnl"] == pytest.approx(100.0)


def test_breakeven_and_trailing_only_tighten():
    engine, pos = engine_with_position(tp_dist=None, breakeven_at_r=1.0, trail_atr=2.0)
    engine._process_bar(pos, "2024-01-02", bar(100, 106, 99, 105), 2.0)  # +1.2R reached
    assert pos.stop == pytest.approx(102.0)  # trail = 106 - 2*2 beats breakeven 100
    engine._process_bar(pos, "2024-01-03", bar(104, 104.5, 103, 103.5), 2.0)
    assert pos.stop == pytest.approx(102.0)  # never loosens
    engine._process_bar(pos, "2024-01-04", bar(103, 103, 101, 101.5), 2.0)
    t = last_trade(engine)
    assert t["exit_reason"] == "trail" and t["pnl"] > 0


def test_cash_ledger_matches_trade_pnl():
    cfg = load_config()
    data = {s: synthetic_ohlcv(s, start="2021-01-01", end="2022-12-31") for s in ["AAA", "BBB"]}
    result = run_backtest(cfg, data)
    trades = result.journal.closed_trades()
    open_fees = sum(p.fees for p in result.engine.positions.values())
    expected_cash = cfg["broker"]["starting_cash"] + sum(t["pnl"] for t in trades) - open_fees
    assert result.engine.broker.cash == pytest.approx(expected_cash)
    assert result.stats["trades"] > 50
    assert all(t["direction"] == 1 for t in trades)  # allow_short is off by default


def test_trade_every_day_rule_enters_almost_daily():
    cfg = load_config()
    data = {s: synthetic_ohlcv(s, start="2021-01-01", end="2022-06-30") for s in ["A", "B", "C", "D", "E"]}
    stats = run_backtest(cfg, data).stats
    assert stats["days_with_entry"] >= 0.9 * stats["days"]
    assert stats["forced_trades"] > 0


def test_no_forced_trades_when_rule_disabled():
    cfg = load_config(overrides={"schedule": {"trade_every_day": False}})
    data = {"AAA": synthetic_ohlcv("AAA", start="2021-01-01", end="2022-06-30")}
    result = run_backtest(cfg, data)
    assert result.stats["forced_trades"] == 0
    assert result.stats["days_with_entry"] < result.stats["days"] / 2


def test_daily_restarts_match_continuous_run(tmp_path):
    """A once-a-day process that reloads all state from SQLite must behave exactly like one long run."""
    cfg = load_config(overrides={"broker": {"allow_short": True}})
    data = {s: synthetic_ohlcv(s, start="2021-01-01", end="2022-03-31") for s in ["AAA", "BBB"]}
    continuous = run_backtest(cfg, data, warmup=250)

    db = str(tmp_path / "bot.db")
    index = data["AAA"].index
    pending = []
    for i in range(250, len(index)):
        journal = Journal(db)
        engine = TradingEngine(cfg, journal)  # broker rebuilt from saved state
        ts, prev = index[i], index[i - 1]
        window = {s: df.iloc[max(0, i - WINDOW + 1): i + 1] for s, df in data.items()}
        opens = {s: float(df["open"].iloc[i]) for s, df in data.items()}
        engine.fill_orders(pending, opens, str(ts), {s: str(prev) for s in data})
        pending = engine.run_cycle(window, str(ts))
        engine.save()
        journal.close()

    journal = Journal(db)
    restarted = [(t["symbol"], t["entry_time"], round(t["pnl"], 8)) for t in journal.closed_trades()]
    original = [(t["symbol"], t["entry_time"], round(t["pnl"], 8)) for t in continuous.journal.closed_trades()]
    assert restarted == original and len(original) > 20
    assert journal.get_state("params") == json.loads(json.dumps(continuous.engine.store.snapshot()))


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [FakeBlock(text)]
        self.stop_reason = stop_reason


def fake_reviewer(response):
    reviewer = ClaudeReviewer.__new__(ClaudeReviewer)
    reviewer.model, reviewer.effort = "claude-opus-5-5", "medium"
    reviewer.client = type("C", (), {})()
    reviewer.client.beta = type("B", (), {})()
    reviewer.client.beta.messages = FakeMessages(response)
    return reviewer


def test_llm_review_parses_and_filters_suggestions():
    body = {"root_cause": "stop inside noise", "avoidable": True, "analysis": "สต็อปแคบ",
            "suggestions": [{"target": "trend", "param": "stop_atr", "value": 3.0, "reason": "noise"},
                            {"target": "trend", "param": "made_up", "value": 1.0, "reason": "x"},
                            {"target": "breakout", "param": "stop_atr", "value": 3.0, "reason": "other strategy"}]}
    reviewer = fake_reviewer(FakeResponse(json.dumps(body)))
    params = {"trend": {"stop_atr": {"value": 2.5}}, "breakout": {"stop_atr": {"value": 2.0}},
              "regime": {}, "risk": {}}
    out = reviewer.review({"strategy": "trend", "id": 1}, [], params)
    assert [s["param"] for s in out["suggestions"]] == ["stop_atr"]
    assert out["suggestions"][0]["target"] == "trend"
    sent = reviewer.client.beta.messages.kwargs
    assert sent["model"] == "claude-opus-5-5" and sent["output_config"]["format"]["type"] == "json_schema"


def test_llm_refusal_returns_none():
    reviewer = fake_reviewer(FakeResponse("", stop_reason="refusal"))
    assert reviewer.review({"strategy": "trend"}, [], {"trend": {}}) is None
