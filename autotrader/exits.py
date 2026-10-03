"""Exit rules shared by the live engine and the what-if simulator, so a
parameter change is tested with exactly the logic that will trade it.

Works on any object with: direction, stop, take_profit, entry_price,
risk_per_unit, best, worst, breakeven_at_r, trail_atr, stop_moved.
"""
from __future__ import annotations


def update_extremes(pos, high: float, low: float) -> None:
    if pos.direction > 0:
        pos.best, pos.worst = max(pos.best, high), min(pos.worst, low)
    else:
        pos.best, pos.worst = min(pos.best, low), max(pos.worst, high)


def bar_exit(pos, o: float, h: float, l: float) -> tuple[float, str] | None:
    """Stop / target hit inside a bar. Gaps fill at the open; if both levels
    are inside the same bar the stop is assumed first (conservative)."""
    stop_label = "trail" if pos.stop_moved else "stop"
    tp = pos.take_profit
    if pos.direction > 0:
        if o <= pos.stop:
            return o, stop_label
        if l <= pos.stop:
            return pos.stop, stop_label
        if tp and o >= tp:
            return o, "target"
        if tp and h >= tp:
            return tp, "target"
    else:
        if o >= pos.stop:
            return o, stop_label
        if h >= pos.stop:
            return pos.stop, stop_label
        if tp and o <= tp:
            return o, "target"
        if tp and l <= tp:
            return tp, "target"
    return None


def tightened_stop(pos, atr_now: float, digits: int | None = None) -> float:
    """Breakeven and trailing stop at the bar close; only ever tightens."""
    d = pos.direction
    best_r = d * (pos.best - pos.entry_price) / pos.risk_per_unit if pos.risk_per_unit else 0.0
    new_stop = pos.stop
    if pos.breakeven_at_r > 0 and best_r >= pos.breakeven_at_r:
        new_stop = max(new_stop, pos.entry_price) if d > 0 else min(new_stop, pos.entry_price)
    if pos.trail_atr > 0:
        trail = pos.best - d * pos.trail_atr * atr_now
        new_stop = max(new_stop, trail) if d > 0 else min(new_stop, trail)
    return round(new_stop, int(digits)) if digits is not None else new_stop
