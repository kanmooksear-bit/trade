"""What-if testing: before a parameter change suggested by a loss review is
adopted, replay the strategy on recent history with the current and the
proposed parameters and keep the change only if it would clearly have done
better.

The replay uses the same signal code and exit rules (``exits.py``) as live
trading: decide at a bar's close, fill at the next open, stops/targets inside
bars, breakeven/trailing at closes, time and regime-change exits, and
round-trip trading costs. It ignores portfolio limits and the learned
selector weights (the regime prior is used), so it measures the strategy
itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .exits import bar_exit, tightened_stop, update_extremes
from .selector import DEFAULT_AFFINITY
from .strategies import STRATEGY_CLASSES

SIGNAL_WINDOW = 220  # bars handed to a strategy per signal (enough for its indicators)


@dataclass
class SimResult:
    trades: int = 0
    sum_r: float = 0.0

    @property
    def avg_r(self) -> float:
        return self.sum_r / self.trades if self.trades else 0.0


@dataclass
class _SimPos:
    direction: int
    entry_price: float
    stop: float
    take_profit: float | None
    risk_per_unit: float
    breakeven_at_r: float
    trail_atr: float
    max_hold: int
    regime: str
    exit_on_regime_change: bool
    best: float = 0.0
    worst: float = 0.0
    bars_held: int = 0
    stop_moved: bool = False


@dataclass
class History:
    """One symbol's bars; ``features``/``regimes`` describe each close from ``start`` on
    (earlier bars only provide indicator look-back)."""
    df: pd.DataFrame
    start: int
    features: list[dict]
    regimes: list[str]


def simulate(strategy: str, params: dict, histories: list[History], cost_fn: Callable[[float], float],
             allow_short: bool, min_score: float) -> SimResult:
    strat = STRATEGY_CLASSES[strategy](params)
    p = strat.params
    res = SimResult()

    def close(pos: _SimPos, price: float) -> None:
        r = (pos.direction * (price - pos.entry_price) - cost_fn(pos.entry_price)) / pos.risk_per_unit
        res.trades += 1
        res.sum_r += r

    for h in histories:
        df, base = h.df, h.start
        o, hi, lo = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low"))
        n = len(df)
        pos: _SimPos | None = None
        pending_entry: tuple | None = None
        exit_next = False
        for i in range(base, n):
            f, regime_i = h.features[i - base], h.regimes[i - base]
            if exit_next and pos is not None:  # close-based exit decided last bar, filled at this open
                close(pos, o[i])
                pos, exit_next = None, False
            if pending_entry is not None:
                d, stop_dist, tp_dist, regime = pending_entry
                pos = _SimPos(d, o[i], o[i] - d * stop_dist, o[i] + d * tp_dist if tp_dist else None, stop_dist,
                              float(p["breakeven_at_r"]), float(p["trail_atr"]), int(p["max_hold"]), regime,
                              bool(int(p["exit_on_regime_change"])), best=o[i], worst=o[i])
                pending_entry = None
            if pos is not None:
                pos.bars_held += 1
                update_extremes(pos, hi[i], lo[i])
                hit = bar_exit(pos, o[i], hi[i], lo[i])
                if hit is not None:
                    close(pos, hit[0])
                    pos = None
                else:
                    new_stop = tightened_stop(pos, f["atr"])
                    if new_stop != pos.stop:
                        pos.stop, pos.stop_moved = new_stop, True
                    regime_bad = pos.exit_on_regime_change and regime_i != pos.regime and \
                        DEFAULT_AFFINITY.get(regime_i, {}).get(strategy, 0.3) < 0.5
                    if (pos.bars_held >= pos.max_hold or regime_bad) and i < n - 1:
                        exit_next = True
                continue
            if i >= n - 1:
                continue
            window = df.iloc[max(0, i - SIGNAL_WINDOW + 1): i + 1]
            sig = strat.generate(window, f)
            if not sig.setup or (sig.direction < 0 and not allow_short):
                continue
            if sig.strength * DEFAULT_AFFINITY.get(regime_i, {}).get(strategy, 0.3) < min_score:
                continue
            tp_dist = float(p["tp_atr"]) * f["atr"] if float(p["tp_atr"]) > 0 else None
            pending_entry = (sig.direction, float(p["stop_atr"]) * f["atr"], tp_dist, regime_i)
        if pos is not None:  # mark open trade to the last close
            close(pos, float(df["close"].iloc[-1]))
    return res


def _halves(histories: list[History]) -> tuple[list[History], list[History]]:
    first, second = [], []
    for h in histories:
        mid = h.start + (len(h.df) - h.start) // 2
        k = mid - h.start
        first.append(History(h.df.iloc[:mid], h.start, h.features[:k], h.regimes[:k]))
        second.append(History(h.df, mid, h.features[k:], h.regimes[k:]))
    return first, second


def evaluate(strategy: str, base_params: dict, changes: dict, histories: list[History],
             cost_fn: Callable[[float], float], cfg: dict) -> tuple[bool, str]:
    """Return (adopt, Thai explanation) for changing ``changes`` on ``strategy``.

    Adopt only if the change adds at least ``whatif_min_gain_r`` over the whole
    replay *and* is not worse in either half of it (one lucky stretch must not
    justify a change)."""
    allow_short = bool(cfg["broker"].get("allow_short", False))
    min_score = float(cfg["risk"]["min_score"])
    learning = cfg["learning"]
    new_params = {**base_params, **changes}

    def run(params, hs):
        return simulate(strategy, params, hs, cost_fn, allow_short, min_score)

    old, new = run(base_params, histories), run(new_params, histories)
    bars = sum(len(h.df) - h.start for h in histories)
    text = (f"ทดสอบย้อนหลัง {bars} แท่ง: ค่าเดิม {old.trades} ไม้ {old.sum_r:+.1f}R → "
            f"ค่าใหม่ {new.trades} ไม้ {new.sum_r:+.1f}R")
    if max(old.trades, new.trades) < int(learning.get("whatif_min_trades", 8)):
        return False, text + " (ไม้น้อยเกินไปที่จะสรุป)"
    gain = new.sum_r - old.sum_r
    needed = max(float(learning.get("whatif_min_gain_r", 1.0)), 0.1 * abs(old.sum_r))
    if new.trades < 0.4 * old.trades:
        return False, text + " (ไม้ลดลงมากเกินไป)"
    if gain < needed:
        return False, text + f" (ดีขึ้นไม่ถึง {needed:.1f}R)"
    first, second = _halves(histories)
    g1 = run(new_params, first).sum_r - run(base_params, first).sum_r
    g2 = run(new_params, second).sum_r - run(base_params, second).sum_r
    if g1 < 0 or g2 < 0:
        return False, text + f" (ดีขึ้นไม่สม่ำเสมอ: ครึ่งแรก {g1:+.1f}R ครึ่งหลัง {g2:+.1f}R)"
    return True, text + f" (ครึ่งแรก {g1:+.1f}R ครึ่งหลัง {g2:+.1f}R)"


def cost_function(broker_cfg: dict) -> Callable[[float], float]:
    """Round-trip trading cost per unit, in price units."""
    spread = float(broker_cfg.get("spread", 0.0))
    slip = float(broker_cfg.get("slippage_bps", 0.0)) / 10_000.0
    fee = float(broker_cfg.get("fee_rate", 0.0))
    commission = 2 * float(broker_cfg.get("commission_per_lot", 0.0)) / max(float(broker_cfg.get("contract_size", 1.0)),
                                                                           1e-12)
    return lambda price: spread + 2 * slip * price + 2 * fee * price + commission

