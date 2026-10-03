"""The daily trading cycle shared by backtests, paper trading and live trading.

One cycle (run once per closed daily bar):

1. detect the market regime of every symbol
2. walk open positions through the new bar(s): stop / target / trailing / breakeven
3. learn from closed trades: selector stats, risk streak, adjustment probation,
   and a post-mortem + parameter adjustment for every losing trade
4. hindsight reviews of earlier stop-outs
5. close-based exits (time stop, regime change, signal reversal)
6. rank every (symbol, strategy) signal by strength x regime weight and enter the best
7. if nothing was entered, place a small forced "probe" trade (trade-every-day rule)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import pandas as pd

from .adapter import Adapter, ParamStore
from .broker import build_broker
from .journal import Journal
from .regime import HIGH_VOL, REGIME_DEFAULTS, REGIME_SPACE, REGIME_TH, RegimeDetector, RegimeReading
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
        self.positions: dict[str, Position] = {p["symbol"]: Position(**p) for p in get("positions", [])}
        self.hindsight_queue: list[dict] = get("hindsight", [])
        self.closed_queue: list[int] = get("closed_queue", [])
        self.last_equity: float | None = get("last_equity", None)
        self.reviewer = LossReviewer()
        self.llm = llm_reviewer
        self.readings: dict[str, RegimeReading] = {
            s: RegimeReading(st["reading"]["regime"], st["reading"]["raw"], st["reading"]["features"])
            for s, st in self.detector.state.items() if st.get("reading")}
        self.events: list[str] = []
        self._llm_calls = 0

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
        j.commit()

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
            fill = self.broker.execute(o.symbol, o.direction, o.qty, price)
        except Exception as exc:  # noqa: BLE001 - exchange errors must not kill the cycle
            self._event(f"⚠️ ส่งคำสั่งเปิด {o.symbol} ไม่สำเร็จ: {exc}")
            return
        entry = fill.price
        stop = entry - o.direction * o.stop_dist
        tp = entry + o.direction * o.tp_dist if o.tp_dist else None
        trade_id = self.journal.open_trade(
            symbol=o.symbol, strategy=o.strategy, regime=o.regime, direction=o.direction, qty=fill.qty,
            entry_time=time, entry_price=entry, stop=stop, take_profit=tp, forced=int(o.forced),
            strength=o.strength, score=o.score, entry_features=o.features, params=o.params,
            signal_reason=o.reason, fees=fill.fee, status="open")
        p = o.params
        self.positions[o.symbol] = Position(
            trade_id=trade_id, symbol=o.symbol, strategy=o.strategy, regime=o.regime, direction=o.direction,
            qty=fill.qty, entry_price=entry, entry_time=time, stop=stop, take_profit=tp,
            risk_per_unit=o.stop_dist, max_hold=int(p["max_hold"]), trail_atr=float(p["trail_atr"]),
            breakeven_at_r=float(p["breakeven_at_r"]), exit_on_regime_change=bool(int(p["exit_on_regime_change"])),
            entry_atr=float(o.features.get("atr", o.stop_dist)), forced=o.forced, last_checked=last_bar,
            best=entry, worst=entry, fees=fill.fee)
        tag = " [ไม้บังคับรายวัน]" if o.forced else ""
        self._event(f"🟢 เปิด {_side(o.direction)} {o.symbol} @ {entry:.4g} x {fill.qty:.6g} ด้วย {o.strategy} "
                    f"({REGIME_TH.get(o.regime, o.regime)}, score {o.score:.2f}){tag} SL {stop:.4g}"
                    + (f" TP {tp:.4g}" if tp else ""))

    def _close(self, pos: Position, price: float, time: str, reason: str, exit_features: dict) -> None:
        try:
            fill = self.broker.execute(pos.symbol, -pos.direction, pos.qty, price)
        except Exception as exc:  # noqa: BLE001
            self._event(f"⚠️ ส่งคำสั่งปิด {pos.symbol} ไม่สำเร็จ: {exc} (จะลองใหม่รอบหน้า)")
            return
        gross = pos.direction * (fill.price - pos.entry_price) * pos.qty
        self.broker.realize(gross)
        fees = pos.fees + fill.fee
        pnl = gross - fees
        risk_amount = pos.risk_per_unit * pos.qty
        r = pnl / risk_amount if risk_amount else 0.0
        mfe_r = pos.direction * (pos.best - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
        mae_r = pos.direction * (pos.worst - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
        reading = self.readings.get(pos.symbol)
        self.journal.close_trade(
            pos.trade_id, exit_time=time, exit_price=fill.price, exit_reason=reason,
            regime_exit=reading.regime if reading else pos.regime, gross_pnl=gross, fees=fees, pnl=pnl,
            r_multiple=r, bars_held=pos.bars_held, mfe_r=mfe_r, mae_r=mae_r, exit_features=exit_features)
        del self.positions[pos.symbol]
        self.closed_queue.append(pos.trade_id)
        icon = "✅" if pnl > 0 else "🔴"
        self._event(f"{icon} ปิด {_side(pos.direction)} {pos.symbol} @ {fill.price:.4g} ({reason}) "
                    f"P&L {pnl:+.2f} ({r:+.2f}R) หลังถือ {pos.bars_held} แท่ง")

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
        d = pos.direction
        pos.bars_held += 1
        pos.last_checked = str(ts)
        pos.best = max(pos.best, h) if d > 0 else min(pos.best, l)
        pos.worst = min(pos.worst, l) if d > 0 else max(pos.worst, h)
        exit_price, reason = None, ""
        stop_label = "trail" if pos.stop_moved else "stop"
        if d > 0:
            if o <= pos.stop:
                exit_price, reason = o, stop_label
            elif l <= pos.stop:
                exit_price, reason = pos.stop, stop_label
            elif pos.take_profit and o >= pos.take_profit:
                exit_price, reason = o, "target"
            elif pos.take_profit and h >= pos.take_profit:
                exit_price, reason = pos.take_profit, "target"
        else:
            if o >= pos.stop:
                exit_price, reason = o, stop_label
            elif h >= pos.stop:
                exit_price, reason = pos.stop, stop_label
            elif pos.take_profit and o <= pos.take_profit:
                exit_price, reason = o, "target"
            elif pos.take_profit and l <= pos.take_profit:
                exit_price, reason = pos.take_profit, "target"
        if exit_price is not None:
            self._close(pos, exit_price, str(ts), reason, self._exit_features(pos, bar))
            return True
        # end of bar: breakeven and trailing stop only tighten, never loosen
        best_r = d * (pos.best - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
        new_stop = pos.stop
        if pos.breakeven_at_r > 0 and best_r >= pos.breakeven_at_r:
            new_stop = max(new_stop, pos.entry_price) if d > 0 else min(new_stop, pos.entry_price)
        if pos.trail_atr > 0:
            trail = pos.best - d * pos.trail_atr * atr_now
            new_stop = max(new_stop, trail) if d > 0 else min(new_stop, trail)
        if new_stop != pos.stop:
            pos.stop = new_stop
            pos.stop_moved = True
        return False

    def check_stops(self, prices: dict[str, float], time: str) -> None:
        """Intraday check with live prices (daemon mode) between daily cycles."""
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
            notes.extend(self.adapter.submit(t, diagnoses, time))
            if llm_out and self.cfg["llm_review"].get("apply_suggestions"):
                for s in llm_out.get("suggestions", []):
                    msg = self.adapter.apply_suggestion(s["target"], s["param"], float(s["value"]),
                                                        s.get("reason", ""), time)
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
                notes = self.adapter.submit(trade, diagnoses, time)
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
        day = str(time)[:10]
        self.risk.start_day(day, self.last_equity if self.last_equity is not None else self.equity(closes))

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

        equity = self.equity(closes)
        self.risk.update_equity(equity)
        orders: list[Order] = []

        candidates = self._candidates(data)
        min_score = float(self.cfg["risk"]["min_score"])
        exiting: set[str] = set()
        for sym, pos in self.positions.items():
            reading = self.readings.get(sym)
            if reading is None:
                continue
            reason = ""
            if pos.bars_held >= pos.max_hold:
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
        if not can_open:
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
            if entries == 0 and self.cfg["schedule"].get("trade_every_day", True):
                entries += self._forced_entry(candidates, equity, closes, taken, exiting, pending_notional, orders)

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
        features = {**f, "regime": c.reading.regime, "weight": c.weight}
        return Order("entry", c.symbol, c.signal.direction, qty=qty, stop_dist=stop_dist, tp_dist=tp_dist,
                     strategy=c.signal.strategy, regime=c.reading.regime, forced=forced,
                     strength=c.signal.strength, score=c.score, reason=c.signal.reason, features=features, params=p)

    def _forced_entry(self, candidates, equity, closes, taken, exiting, pending_notional, orders) -> int:
        probes_open = sum(1 for s, p in self.positions.items() if p.forced and s not in exiting)
        if probes_open >= int(self.cfg["schedule"].get("probe_slots", 2)):
            self._event("ℹ️ วันนี้ไม่มีไม้ใหม่: ช่องไม้บังคับเต็ม (กฎความเสี่ยงสำคัญกว่ากฎเทรดทุกวัน)")
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
            why = "ทุกเหรียญมีสถานะเปิดอยู่แล้ว (เพิ่ม symbols เพื่อให้เทรดได้ทุกวัน)"
        elif not self.cfg["broker"].get("allow_short", False):
            why = "สัญญาณที่เหลือเป็นขาลงทั้งหมด แต่ปิดการ short ไว้ (allow_short: false)"
        else:
            why = "กลยุทธ์ที่เหมาะกับสภาวะนี้ติด cooldown ทั้งหมด"
        self._event(f"ℹ️ วันนี้ไม่มีไม้ใหม่: {why}")
        return 0
