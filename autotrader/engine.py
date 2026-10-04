"""The trading cycle shared by backtests, paper trading and live trading.

One cycle runs per closed bar (daily, hourly, ... as configured):

1. detect the market regime of every symbol
2. walk open positions through the new bar(s): stop / target / trailing / breakeven
3. learn from closed trades: selector stats, risk streak, adjustment probation,
   and a post-mortem + parameter adjustment for every losing trade
4. hindsight reviews of earlier stop-outs
5. close-based exits (time stop, regime change, signal reversal)
6. rank every (symbol, strategy) signal by strength x regime weight and enter the best
7. if nothing was entered yet today, place a small forced "probe" trade (trade-every-day rule)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

from .adapter import Adapter, ParamStore
from .broker import ExternalClose, build_broker
from .data import timeframe_seconds
from .exits import bar_exit, tightened_stop, update_extremes
from .sessions import in_window, local_time
from .journal import Journal
from . import whatif
from .regime import (HIGH_VOL, REGIME_DEFAULTS, REGIME_SPACE, REGIME_TH, RegimeDetector, RegimeReading, classify,
                     compute_features)
from .review import LossReviewer, summarize
from .risk import RISK_PARAM_DEFAULTS, RISK_PARAM_SPACE, RiskManager
from .selector import DEFAULT_AFFINITY, StrategySelector
from .strategies import STRATEGY_CLASSES, Signal, build_strategies

MIN_BARS = 100


@dataclass
class Position:
    trade_id: int
    symbol: str
    strategy: str
    regime: str
    direction: int
    qty: float
    entry_price: float
    entry_time: str
    stop: float
    take_profit: float | None
    risk_per_unit: float
    max_hold: int
    trail_atr: float
    breakeven_at_r: float
    exit_on_regime_change: bool
    entry_atr: float
    forced: bool
    last_checked: str
    bars_held: int = 0
    best: float = 0.0
    worst: float = 0.0
    fees: float = 0.0
    stop_moved: bool = False
    ticket: int | None = None  # broker reference (MT5 position ticket)


@dataclass
class Order:
    kind: str  # entry | exit
    symbol: str
    direction: int
    qty: float = 0.0
    stop_dist: float = 0.0
    tp_dist: float | None = None
    strategy: str = ""
    regime: str = ""
    forced: bool = False
    strength: float = 0.0
    score: float = 0.0
    reason: str = ""
    features: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    trade_id: int | None = None
    exit_reason: str = ""


@dataclass
class Candidate:
    symbol: str
    signal: Signal
    reading: RegimeReading
    weight: float
    score: float


def _side(d: int) -> str:
    return "Long" if d > 0 else "Short"


class TradingEngine:
    def __init__(self, cfg: dict, journal: Journal, broker=None, llm_reviewer=None,
                 log: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.journal = journal
        self.log = log
        get = journal.get_state
        self.cycle = int(get("cycle", 0))
        saved = get("params", {}) or {}
        self.strategies = build_strategies(saved)
        self.regime_params = {**REGIME_DEFAULTS, **saved.get("regime", {})}
        self.risk_params = {**RISK_PARAM_DEFAULTS, **saved.get("risk", {})}
        self.store = ParamStore()
        for name, strat in self.strategies.items():
            self.store.register(name, strat.params, STRATEGY_CLASSES[name].param_space())
        self.store.register("regime", self.regime_params, REGIME_SPACE)
        self.store.register("risk", self.risk_params, RISK_PARAM_SPACE)
        self.detector = RegimeDetector(self.regime_params, get("regime_state", {}))
        self.selector = StrategySelector(cfg["learning"], get("selector", {}))
        self.risk = RiskManager(cfg["risk"], self.risk_params, get("risk", {}))
        self.adapter = Adapter(self.store, journal, cfg["learning"], get("adapter", {}))
        self.broker = broker if broker is not None else build_broker(cfg, get("broker", {}))
        saved_kind = get("broker_kind", None)
        if saved_kind and saved_kind != type(self.broker).__name__:
            raise RuntimeError(
                f"ฐานข้อมูลนี้เป็นของโหมด {saved_kind} แต่ตอนนี้ใช้ {type(self.broker).__name__} — "
                "ห้ามใช้ไฟล์ร่วมกัน (เงินทุน/ไม้ค้างจะปนกัน) ให้ตั้ง storage.db_path เป็นไฟล์ใหม่")
        self.positions: dict[str, Position] = {p["symbol"]: Position(**p) for p in get("positions", [])}
        self.hindsight_queue: list[dict] = get("hindsight", [])
        self.closed_queue: list[int] = get("closed_queue", [])
        self.last_equity: float | None = get("last_equity", None)
        self.last_entry_day: str | None = get("last_entry_day", None)
        self.tf_seconds = timeframe_seconds(cfg["data"].get("timeframe", "1d"))
        self.digits = cfg["broker"].get("digits")
        self.reviewer = LossReviewer()
        self.llm = llm_reviewer
        self.readings: dict[str, RegimeReading] = {
            s: RegimeReading(st["reading"]["regime"], st["reading"]["raw"], st["reading"]["features"])
            for s, st in self.detector.state.items() if st.get("reading")}
        self.events: list[str] = []
        self._llm_calls = 0
        # what-if testing: long history (set by the backtester, else the cycle's data) + per-bar feature cache
        self.long_history: dict[str, pd.DataFrame] | None = None
        self._hist: dict[str, pd.DataFrame] = {}
        self.feat_cache: dict[str, dict[str, tuple[dict, str]]] = {}
        self._cost_fn = whatif.cost_function(cfg["broker"])

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        j = self.journal
        j.set_state("cycle", self.cycle)
        j.set_state("params", self.store.snapshot())
        j.set_state("regime_state", self.detector.state)
        j.set_state("selector", self.selector.stats)
        j.set_state("risk", self.risk.state)
        j.set_state("adapter", self.adapter.state)
        j.set_state("broker", self.broker.state)
        j.set_state("positions", [asdict(p) for p in self.positions.values()])
        j.set_state("hindsight", self.hindsight_queue)
        j.set_state("closed_queue", self.closed_queue)
        j.set_state("last_equity", self.last_equity)
        j.set_state("last_entry_day", self.last_entry_day)
        j.set_state("broker_kind", type(self.broker).__name__)
        j.commit()

    def px(self, price: float | None) -> str:
        if price is None:
            return "-"
        return f"{price:.{int(self.digits)}f}" if self.digits is not None else f"{price:.6g}"

    def _event(self, msg: str) -> None:
        self.events.append(msg)
        if self.log:
            self.log(msg)

    def pop_events(self) -> list[str]:
        out, self.events = self.events, []
        return out

    # ------------------------------------------------------------ accounting
    def unrealized(self, prices: dict[str, float]) -> float:
        return sum(p.direction * (prices.get(s, p.entry_price) - p.entry_price) * p.qty
                   for s, p in self.positions.items())

    def gross_exposure(self, prices: dict[str, float]) -> float:
        return sum(abs(p.qty * prices.get(s, p.entry_price)) for s, p in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.broker.equity(self.unrealized(prices))

    # ------------------------------------------------------------ open/close
    def _open(self, o: Order, price: float, time: str, last_bar: str) -> None:
        if o.symbol in self.positions or o.qty <= 0:
            return
        try:
            fill = self.broker.open(o.symbol, o.direction, o.qty, price, o.stop_dist, o.tp_dist,
                                    comment=f"at {o.strategy}{' probe' if o.forced else ''}")
        except Exception as exc:  # noqa: BLE001 - broker errors must not kill the cycle
            self._event(f"⚠️ ส่งคำสั่งเปิด {o.symbol} ไม่สำเร็จ: {exc}")
            return
        entry = fill.price
        stop = fill.sl if fill.sl is not None else entry - o.direction * o.stop_dist
        tp = fill.tp if fill.tp is not None else (entry + o.direction * o.tp_dist if o.tp_dist else None)
        risk_per_unit = abs(entry - stop) or o.stop_dist
        trade_id = self.journal.open_trade(
            symbol=o.symbol, strategy=o.strategy, regime=o.regime, direction=o.direction, qty=fill.qty,
            entry_time=time, entry_price=entry, stop=stop, take_profit=tp, forced=int(o.forced),
            strength=o.strength, score=o.score, entry_features=o.features, params=o.params,
            signal_reason=o.reason, fees=fill.fee, status="open")
        p = o.params
        self.positions[o.symbol] = Position(
            trade_id=trade_id, symbol=o.symbol, strategy=o.strategy, regime=o.regime, direction=o.direction,
            qty=fill.qty, entry_price=entry, entry_time=time, stop=stop, take_profit=tp,
            risk_per_unit=risk_per_unit, max_hold=int(p["max_hold"]), trail_atr=float(p["trail_atr"]),
            breakeven_at_r=float(p["breakeven_at_r"]), exit_on_regime_change=bool(int(p["exit_on_regime_change"])),
            entry_atr=float(o.features.get("atr", o.stop_dist)), forced=o.forced, last_checked=last_bar,
            best=entry, worst=entry, fees=fill.fee, ticket=fill.ticket)
        tag = " [ไม้บังคับรายวัน]" if o.forced else ""
        size = f"{self.broker.lots(fill.qty):.2f} lot" if self.broker.contract_size != 1 else f"x {fill.qty:.6g}"
        self._event(f"🟢 เปิด {_side(o.direction)} {o.symbol} @ {self.px(entry)} {size} ด้วย {o.strategy} "
                    f"({REGIME_TH.get(o.regime, o.regime)}, score {o.score:.2f}){tag} SL {self.px(stop)}"
                    + (f" TP {self.px(tp)}" if tp else ""))

    def _close(self, pos: Position, price: float, time: str, reason: str, exit_features: dict) -> None:
        try:
            fill = self.broker.close(pos.symbol, pos.direction, pos.qty, price, pos.ticket)
        except Exception as exc:  # noqa: BLE001
            self._event(f"⚠️ ส่งคำสั่งปิด {pos.symbol} ไม่สำเร็จ: {exc} (จะลองใหม่รอบหน้า)")
            return
        self._finalize_close(pos, fill.price, fill.fee, time, reason, exit_features)

    def _finalize_close(self, pos: Position, price: float, fee: float, time: str, reason: str,
                        exit_features: dict) -> None:
        gross = pos.direction * (price - pos.entry_price) * pos.qty
        self.broker.realize(gross)
        fees = pos.fees + fee
        pnl = gross - fees
        risk_amount = pos.risk_per_unit * pos.qty
        r = pnl / risk_amount if risk_amount else 0.0
        mfe_r = pos.direction * (pos.best - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
        mae_r = pos.direction * (pos.worst - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
        reading = self.readings.get(pos.symbol)
        self.journal.close_trade(
            pos.trade_id, exit_time=time, exit_price=price, exit_reason=reason,
            regime_exit=reading.regime if reading else pos.regime, gross_pnl=gross, fees=fees, pnl=pnl,
            r_multiple=r, bars_held=pos.bars_held, mfe_r=mfe_r, mae_r=mae_r, exit_features=exit_features)
        del self.positions[pos.symbol]
        self.closed_queue.append(pos.trade_id)
        icon = "✅" if pnl > 0 else "🔴"
        self._event(f"{icon} ปิด {_side(pos.direction)} {pos.symbol} @ {self.px(price)} ({reason}) "
                    f"P&L {pnl:+.2f} ({r:+.2f}R) หลังถือ {pos.bars_held} แท่ง")

    def _reconcile(self, time: str) -> None:
        """Record positions the broker closed by itself (server-side SL/TP)."""
        closed: list[ExternalClose] = self.broker.reconcile(list(self.positions.values()))
        by_id = {p.trade_id: p for p in self.positions.values()}
        for c in closed:
            pos = by_id.get(c.trade_id)
            if pos is not None:
                self._finalize_close(pos, c.price, c.fee, c.time, c.reason, self._exit_features(pos))

    def _exit_features(self, pos: Position, bar: pd.Series | None = None) -> dict:
        reading = self.readings.get(pos.symbol)
        f = dict(reading.features) if reading else {}
        if bar is not None and pos.entry_atr:
            f["bar_range_entry_atr"] = float(bar["high"] - bar["low"]) / pos.entry_atr
        f["regime"] = reading.regime if reading else pos.regime
        return f

    def fill_orders(self, orders: list[Order], prices: dict[str, float], time: str,
                    last_bar: dict[str, str]) -> None:
        """Execute orders: at the next bar's open (backtest) or right now (live)."""
        for o in [o for o in orders if o.kind == "exit"]:
            pos = self.positions.get(o.symbol)
            if pos and pos.trade_id == o.trade_id and o.symbol in prices:
                self._close(pos, prices[o.symbol], time, o.exit_reason, o.features)
        for o in [o for o in orders if o.kind == "entry"]:
            if o.symbol in prices:
                self._open(o, prices[o.symbol], time, last_bar.get(o.symbol, time))
        self._learn_closed(time)

    # --------------------------------------------------------- bar handling
    def _process_bar(self, pos: Position, ts, bar: pd.Series, atr_now: float) -> bool:
        o, h, l = float(bar["open"]), float(bar["high"]), float(bar["low"])
        pos.bars_held += 1
        pos.last_checked = str(ts)
        update_extremes(pos, h, l)
        # with server-side SL/TP (MT5) the broker executes them; _reconcile picks those exits up
        hit = None if self.broker.server_side_stops else bar_exit(pos, o, h, l)
        if hit is not None:
            self._close(pos, hit[0], str(ts), hit[1], self._exit_features(pos, bar))
            return True
        new_stop = tightened_stop(pos, atr_now, self.digits)
        if new_stop != pos.stop:
            if self.broker.modify(pos.symbol, pos.ticket, new_stop, pos.take_profit):
                pos.stop = new_stop
                pos.stop_moved = True
            elif pos.direction * (float(bar["close"]) - new_stop) <= 0:
                # price already closed beyond the new stop (the server refuses such a SL): exit now
                pos.stop_moved = True
                self._close(pos, float(bar["close"]), str(ts), "trail", self._exit_features(pos, bar))
                return True
            else:
                self._event(f"⚠️ เลื่อน SL ของ {pos.symbol} ไป {self.px(new_stop)} ไม่สำเร็จ (ใช้ SL เดิม)")
        return False

    def check_stops(self, prices: dict[str, float], time: str) -> None:
        """Check between bar closes (daemon mode)."""
        if self.broker.server_side_stops:
            self._reconcile(time)
            self._learn_closed(time)
            return
        for pos in list(self.positions.values()):
            p = prices.get(pos.symbol)
            if p is None:
                continue
            d = pos.direction
            if d * (p - pos.stop) <= 0:
                self._close(pos, p, time, "trail" if pos.stop_moved else "stop", self._exit_features(pos))
            elif pos.take_profit and d * (p - pos.take_profit) >= 0:
                self._close(pos, p, time, "target", self._exit_features(pos))
        self._learn_closed(time)

    # -------------------------------------------------------------- learning
    def _whatif_histories(self) -> list[whatif.History]:
        bars = int(self.cfg["learning"].get("whatif_bars", 1000))
        lookback = int(self.regime_params["vol_lookback"])
        out = []
        for sym, df in self._hist.items():
            start = max(MIN_BARS, len(df) - bars)
            if len(df) - start < 50:
                continue
            cache = self.feat_cache.setdefault(sym, {})
            feats, regimes = [], []
            for j in range(start, len(df)):
                key = str(df.index[j])
                if key not in cache:  # bars the engine never saw (warm-up, or a fresh process)
                    f = compute_features(df.iloc[max(0, j - 299): j + 1], lookback)
                    cache[key] = (f, classify(f, self.regime_params))
                feats.append(cache[key][0])
                regimes.append(cache[key][1])
            keep = {str(t) for t in df.index[start:]}
            for key in [k for k in cache if k not in keep]:
                del cache[key]
            out.append(whatif.History(df, start, feats, regimes))
        return out

    def _validator(self):
        return self._whatif if self._hist else None

    def _whatif(self, strategy: str, changes: dict) -> tuple[bool, str]:
        if strategy not in self.strategies:
            return True, ""
        histories = self._whatif_histories()
        if not histories:
            return False, "ไม่มีข้อมูลย้อนหลังให้ทดสอบ"
        return whatif.evaluate(strategy, dict(self.strategies[strategy].params), changes, histories,
                               self._cost_fn, self.cfg)

    def _learn_closed(self, time: str) -> None:
        queue, self.closed_queue = self.closed_queue, []
        for tid in queue:
            t = self.journal.trade(tid)
            if not t:
                continue
            r = float(t["r_multiple"] or 0.0)
            notes: list[str] = []
            if not t["forced"]:
                notes.append(self.selector.record(t["regime"], t["strategy"], r, self.cycle))
                notes.append(self.risk.on_trade_closed(r))
                notes.extend(self.adapter.on_trade_closed(t, time))
            notes = [n for n in notes if n]
            if float(t["pnl"] or 0) >= 0:
                for n in notes:
                    self._event(f"   • {n}")
                continue
            history = [x for x in self.journal.closed_trades(strategy=t["strategy"], limit=60)
                       if not x["forced"] and x["id"] != tid][:30]
            diagnoses = self.reviewer.review(t, history)
            llm_out = self._llm_review(t, diagnoses)
            notes.extend(self.adapter.submit(t, diagnoses, time, self._validator()))
            if llm_out and self.cfg["llm_review"].get("apply_suggestions"):
                for s in llm_out.get("suggestions", []):
                    msg = self.adapter.apply_suggestion(s["target"], s["param"], float(s["value"]),
                                                        s.get("reason", ""), time, self._validator())
                    if msg:
                        notes.append(f"(AI) {msg}")
            summary = summarize(t, diagnoses, notes)
            if llm_out and llm_out.get("analysis"):
                summary += f"\n  🤖 AI: {llm_out['analysis']}"
            self.journal.add_review(tid, time, "loss", [d.to_dict() for d in diagnoses], summary, llm_out)
            self._event("🔍 ทบทวนไม้ขาดทุน\n" + summary)
            if t["exit_reason"] == "stop" and not t["forced"]:
                self.hindsight_queue.append({"trade_id": tid, "symbol": t["symbol"], "exit_time": t["exit_time"],
                                             "waited": 0})

    def _llm_review(self, trade: dict, diagnoses) -> dict | None:
        llm_cfg = self.cfg["llm_review"]
        if not self.llm or not llm_cfg.get("enabled") or self._llm_calls >= int(llm_cfg.get("max_reviews_per_day", 5)):
            return None
        self._llm_calls += 1
        try:
            return self.llm.review(trade, [d.to_dict() for d in diagnoses], self.store.describe())
        except Exception as exc:  # noqa: BLE001 - AI review is optional
            self._event(f"⚠️ AI review ล้มเหลว: {exc}")
            return None

    def _run_hindsight(self, data: dict[str, pd.DataFrame], time: str) -> None:
        k = int(self.cfg["learning"].get("hindsight_bars", 5))
        remaining = []
        for h in self.hindsight_queue:
            df = data.get(h["symbol"])
            h["waited"] += 1
            if df is None:
                if h["waited"] < 3 * k:
                    remaining.append(h)
                continue
            after = df[df.index > pd.Timestamp(h["exit_time"])]
            if len(after) < k:
                if h["waited"] < 3 * k:
                    remaining.append(h)
                continue
            trade = self.journal.trade(h["trade_id"])
            diagnoses = self.reviewer.hindsight(trade, after.iloc[:k]) if trade else []
            if diagnoses:
                notes = self.adapter.submit(trade, diagnoses, time, self._validator())
                summary = summarize(trade, diagnoses, notes)
                self.journal.add_review(trade["id"], time, "hindsight", [d.to_dict() for d in diagnoses], summary)
                self._event("🔁 ทบทวนย้อนหลัง\n" + summary)
        self.hindsight_queue = remaining

    # ------------------------------------------------------------ the cycle
    def run_cycle(self, data: dict[str, pd.DataFrame], time: str) -> list[Order]:
        """Process the latest closed bar of every symbol; returns orders to execute."""
        self.cycle += 1
        self._llm_calls = 0
        data = {s: df for s, df in data.items() if df is not None and len(df) >= MIN_BARS}
        closes = {s: float(df["close"].iloc[-1]) for s, df in data.items()}
        self.readings = {s: self.detector.detect(s, df) for s, df in data.items()}
        for s, df in data.items():
            self.feat_cache.setdefault(s, {})[str(df.index[-1])] = (self.readings[s].features,
                                                                   self.readings[s].regime)
        self._hist = self.long_history if self.long_history is not None else data
        # the decision happens when the latest bar closes; "day" and probe timing use that bar clock
        decision = max(df.index[-1] for df in data.values()) + pd.Timedelta(seconds=self.tf_seconds) \
            if data else pd.Timestamp(time)
        day = str(decision)[:10]
        self.risk.start_day(day, self.last_equity if self.last_equity is not None else self.equity(closes))
        self._reconcile(time)

        exits_now = 0
        for sym, pos in list(self.positions.items()):
            df = data.get(sym)
            if df is None:
                continue
            new_bars = df[df.index > pd.Timestamp(pos.last_checked)]
            atr_now = self.readings[sym].features["atr"]
            for ts, bar in new_bars.iterrows():
                if self._process_bar(pos, ts, bar, atr_now):
                    exits_now += 1
                    break
        self._learn_closed(time)
        self._run_hindsight(data, time)
        for msg in self.adapter.process_pending(self._whatif, time):
            self._event(f"🧪 {msg}")

        equity = self.equity(closes)
        self.risk.update_equity(equity)
        orders: list[Order] = []

        candidates = self._candidates(data)
        min_score = float(self.cfg["risk"]["min_score"])
        exiting: set[str] = set()
        hours = self.cfg["schedule"].get("trading_hours") or {}
        local = local_time(decision, self.cfg["data"].get("server_tz", "utc"), hours.get("utc_offset", 7)) \
            if hours else decision
        session_open = in_window(local, hours["start"], hours["end"]) if hours else True
        for sym, pos in self.positions.items():
            reading = self.readings.get(sym)
            if reading is None:
                continue
            reason = ""
            if not session_open and hours.get("close_at_end", True):
                reason = "session_end"
            elif pos.bars_held >= pos.max_hold:
                reason = "time"
            elif pos.exit_on_regime_change and reading.regime != pos.regime \
                    and DEFAULT_AFFINITY.get(reading.regime, {}).get(pos.strategy, 0.3) < 0.5:
                reason = "regime_change"
            else:
                opposite = [c for c in candidates if c.symbol == sym and c.signal.setup
                            and c.signal.direction == -pos.direction and c.score >= min_score]
                if opposite:
                    reason = "reversal"
            if reason:
                orders.append(Order("exit", sym, pos.direction, qty=pos.qty, trade_id=pos.trade_id,
                                    exit_reason=reason, features=self._exit_features(pos)))
                exiting.add(sym)

        can_open, why = self.risk.can_open(equity)
        entries = 0
        if not session_open:
            pass  # outside trading hours: manage exits only
        elif not can_open:
            self._event(f"⛔ ไม่เปิดไม้ใหม่: {why}")
        else:
            max_open = int(self.cfg["risk"]["max_open_positions"])
            normal_open = sum(1 for s, p in self.positions.items() if not p.forced and s not in exiting)
            taken = set(exiting) | {s for s, p in self.positions.items() if not p.forced}
            pending_notional = 0.0
            for c in candidates:
                if normal_open + entries >= max_open:
                    break
                if not c.signal.setup or c.score < min_score or c.symbol in taken:
                    continue
                order = self._entry_order(c, equity, closes, pending_notional, forced=False)
                if order:
                    probe = self.positions.get(c.symbol)
                    if probe is not None:  # a forced probe holds this symbol: swap it for the real setup
                        orders.append(Order("exit", c.symbol, probe.direction, qty=probe.qty,
                                            trade_id=probe.trade_id, exit_reason="replaced",
                                            features=self._exit_features(probe)))
                        exiting.add(c.symbol)
                    orders.append(order)
                    taken.add(c.symbol)
                    pending_notional += order.qty * closes[c.symbol]
                    entries += 1
            sched = self.cfg["schedule"]
            probe_due = self.tf_seconds >= 86400 or local.hour >= int(sched.get("probe_after_hour", 0))
            if entries == 0 and sched.get("trade_every_day", True) and self.last_entry_day != day and probe_due:
                entries += self._forced_entry(candidates, equity, closes, taken, exiting, pending_notional, orders,
                                              day)
        if entries:
            self.last_entry_day = day

        self.last_equity = equity
        self.journal.record_equity(str(time), equity, self.broker.cash, len(self.positions))
        forced = sum(1 for o in orders if o.kind == "entry" and o.forced)
        regimes = ", ".join(f"{s}:{REGIME_TH.get(r.regime, r.regime)}" for s, r in self.readings.items())
        self.journal.log_day(day, entries, exits_now + len(exiting), forced, regimes)
        return orders

    def _candidates(self, data: dict[str, pd.DataFrame]) -> list[Candidate]:
        allow_short = bool(self.cfg["broker"].get("allow_short", False))
        out: list[Candidate] = []
        for sym, reading in self.readings.items():
            for strat in self.strategies.values():
                sig = strat.generate(data[sym], reading.features)
                if sig.direction == 0 or (sig.direction < 0 and not allow_short):
                    continue
                w = self.selector.weight(reading.regime, strat.name, self.cycle)
                if w <= 0:
                    continue
                out.append(Candidate(sym, sig, reading, w, sig.strength * w))
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def _entry_order(self, c: Candidate, equity: float, closes: dict[str, float], pending_notional: float,
                     forced: bool) -> Order | None:
        f = c.reading.features
        p = dict(c.signal.params)
        if forced:
            p["max_hold"] = min(int(p["max_hold"]), int(self.cfg["schedule"].get("probe_max_hold", 2)))
        stop_dist = float(p["stop_atr"]) * f["atr"]
        tp_dist = float(p["tp_atr"]) * f["atr"] if float(p["tp_atr"]) > 0 else None
        qty, why = self.risk.size(
            equity, closes[c.symbol], stop_dist, high_vol=c.reading.regime == HIGH_VOL, forced=forced,
            score=c.score, gross_exposure=self.gross_exposure(closes) + pending_notional,
            min_notional=float(self.cfg["broker"].get("min_notional", 10.0)))
        if qty <= 0:
            self._event(f"ข้าม {c.symbol} ({c.signal.strategy}): {why}")
            return None
        tradable = self.broker.normalize_qty(c.symbol, qty)
        if tradable <= 0:  # below the broker's minimum lot
            min_q = self.broker.min_qty(c.symbol)
            min_risk = min_q * stop_dist
            # a forced probe must never risk more than a normal trade
            ceiling = float(self.cfg["risk"]["risk_per_trade"]) if forced else \
                float(self.cfg["risk"].get("max_min_lot_risk", 0.02))
            if min_q > 0 and min_risk <= ceiling * equity:
                tradable = min_q
            else:
                need = min_risk / float(self.cfg["risk"]["risk_per_trade"])
                self._event(f"ข้าม {c.symbol} ({c.signal.strategy}): lot ขั้นต่ำเสี่ยง {min_risk:,.2f} "
                            f"({min_risk / equity:.1%} ของทุน) เกินเพดาน {ceiling:.0%} — ต้องมีทุนราว {need:,.0f} "
                            "หรือใช้ timeframe ที่เล็กลง (สต็อปแคบลง)")
                return None
        qty = tradable
        features = {**f, "regime": c.reading.regime, "weight": c.weight}
        return Order("entry", c.symbol, c.signal.direction, qty=qty, stop_dist=stop_dist, tp_dist=tp_dist,
                     strategy=c.signal.strategy, regime=c.reading.regime, forced=forced,
                     strength=c.signal.strength, score=c.score, reason=c.signal.reason, features=features, params=p)

    def _note_once(self, day: str, msg: str) -> None:
        if getattr(self, "_noted_day", None) != (day, msg):
            self._noted_day = (day, msg)
            self._event(msg)

    def _forced_entry(self, candidates, equity, closes, taken, exiting, pending_notional, orders, day) -> int:
        probes_open = sum(1 for s, p in self.positions.items() if p.forced and s not in exiting)
        if probes_open >= int(self.cfg["schedule"].get("probe_slots", 2)):
            self._note_once(day, "ℹ️ วันนี้ยังไม่มีไม้ใหม่: ช่องไม้บังคับเต็ม (กฎความเสี่ยงสำคัญกว่ากฎเทรดทุกวัน)")
            return 0
        for c in candidates:
            if c.symbol in taken or c.symbol in exiting or c.symbol in self.positions:
                continue
            order = self._entry_order(c, equity, closes, pending_notional, forced=True)
            if order:
                orders.append(order)
                return 1
        free = [s for s in self.readings if s not in self.positions and s not in taken and s not in exiting]
        if not free:
            why = "ทุก symbol มีสถานะเปิดอยู่แล้ว (ถือไม้เดิมต่อ)"
        elif not self.cfg["broker"].get("allow_short", False):
            why = "สัญญาณที่เหลือเป็นขาลงทั้งหมด แต่ปิดการ short ไว้ (allow_short: false)"
        else:
            why = "กลยุทธ์ที่เหมาะกับสภาวะนี้ติด cooldown ทั้งหมด"
        self._note_once(day, f"ℹ️ วันนี้ยังไม่มีไม้ใหม่: {why}")
        return 0
