"""Trading strategies.

Every strategy always returns a directional *bias* (so a forced daily trade
can still pick a side) plus ``setup=True`` only when its full entry rules are
met. All tunable numbers live in ``params`` with bounds in ``space`` so the
adapter can move them after losing trades without leaving sane ranges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

# Parameters shared by every strategy (exit management + entry filters).
COMMON_DEFAULTS = {
    "stop_atr": 2.0,
    "tp_atr": 4.0,
    "max_hold": 12,
    "min_strength": 0.45,
    "trend_filter": 0,  # 1 = only trade in the direction of the 200 EMA
    "max_extension_atr": 3.0,  # skip entries this many ATR away from EMA20 (chasing)
    "breakeven_at_r": 1.5,  # move stop to entry once trade is this many R in profit
    "trail_atr": 0.0,  # 0 = no trailing stop
    "exit_on_regime_change": 0,
}

COMMON_SPACE = {
    "stop_atr": (1.0, 4.5, 0.25),
    "tp_atr": (1.0, 8.0, 0.25),
    "max_hold": (2, 30, 1),
    "min_strength": (0.30, 0.85, 0.05),
    "trend_filter": (0, 1, 1),
    "max_extension_atr": (1.0, 5.0, 0.25),
    "breakeven_at_r": (0.5, 3.0, 0.25),
    "trail_atr": (0.0, 5.0, 0.25),
    "exit_on_regime_change": (0, 1, 1),
}


@dataclass
class Signal:
    strategy: str
    direction: int  # +1 long, -1 short, 0 none
    strength: float  # 0..1
    setup: bool  # full entry conditions met
    reason: str
    params: dict = field(default_factory=dict)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


class Strategy:
    name = "base"
    title_th = ""
    defaults: dict = {}
    space: dict = {}

    def __init__(self, params: dict | None = None):
        self.params = {**COMMON_DEFAULTS, **self.defaults, **(params or {})}

    @classmethod
    def param_space(cls) -> dict:
        return {**COMMON_SPACE, **cls.space}

    @classmethod
    def default_params(cls) -> dict:
        return {**COMMON_DEFAULTS, **cls.defaults}

    def generate(self, df: pd.DataFrame, feats: dict) -> Signal:
        direction, strength, setup, reason = self._raw(df, feats)
        return self._finalize(direction, strength, setup, reason, feats)

    def _raw(self, df: pd.DataFrame, feats: dict):  # pragma: no cover - interface
        raise NotImplementedError

    def _finalize(self, direction, strength, setup, reason, feats) -> Signal:
        p = self.params
        notes = [reason]
        if direction != 0 and setup:
            if int(p["trend_filter"]) and feats["htf_trend"] != 0 and direction != feats["htf_trend"]:
                setup, strength = False, strength * 0.5
                notes.append("สวนเทรนด์ใหญ่ (EMA200) จึงไม่เข้า")
            if direction * feats["extension"] > p["max_extension_atr"]:
                setup, strength = False, strength * 0.6
                notes.append(f"ราคาวิ่งไกลจาก EMA20 {abs(feats['extension']):.1f} ATR (ไล่ราคา)")
            if strength < p["min_strength"]:
                setup = False
                notes.append(f"สัญญาณอ่อน {strength:.2f} < {p['min_strength']:.2f}")
        if not setup:
            strength = min(strength, 0.4)
        return Signal(self.name, int(direction), _clip01(strength), bool(setup and direction != 0),
                      "; ".join(n for n in notes if n), dict(p))


class TrendFollowing(Strategy):
    name = "trend"
    title_th = "ตามเทรนด์"
    defaults = {"ema_fast": 20, "ema_slow": 50, "adx_min": 20.0, "stop_atr": 2.5, "tp_atr": 6.0,
                "max_hold": 20, "trail_atr": 3.0, "exit_on_regime_change": 1}
    space = {"ema_fast": (5, 40, 1), "ema_slow": (20, 120, 5), "adx_min": (12.0, 35.0, 1.0)}

    def _raw(self, df, feats):
        p = self.params
        close = df["close"].to_numpy(dtype=float)
        fast = float(ind.ema(close, p["ema_fast"])[-1])
        slow = float(ind.ema(close, max(p["ema_slow"], p["ema_fast"] + 5))[-1])
        c = feats["close"]
        bias = int(np.sign(fast - slow))
        aligned = (bias > 0 and c > fast) or (bias < 0 and c < fast)
        strength = 0.6 * _clip01((feats["adx"] - 12.0) / 25.0) + 0.4 * _clip01(feats["er"] / 0.6)
        setup = bool(aligned and feats["adx"] >= p["adx_min"])
        reason = f"EMA{int(p['ema_fast'])}{'>' if bias > 0 else '<'}EMA{int(p['ema_slow'])}, ADX {feats['adx']:.0f}"
        return bias, strength, setup, reason


class MeanReversion(Strategy):
    name = "mean_reversion"
    title_th = "สวนกลับค่าเฉลี่ย"
    defaults = {"bb_period": 20, "bb_std": 2.0, "rsi_low": 30.0, "rsi_high": 70.0, "stop_atr": 1.75,
                "tp_atr": 2.5, "max_hold": 7, "breakeven_at_r": 1.0}
    space = {"bb_period": (10, 40, 1), "bb_std": (1.5, 3.0, 0.1), "rsi_low": (15.0, 40.0, 1.0),
             "rsi_high": (60.0, 85.0, 1.0)}

    def _raw(self, df, feats):
        p = self.params
        n = int(p["bb_period"])
        window = df["close"].to_numpy(dtype=float)[-n:]
        mid = float(window.mean())
        sd = float(window.std(ddof=1)) if len(window) > 1 else 0.0
        upper, lower = mid + float(p["bb_std"]) * sd, mid - float(p["bb_std"]) * sd
        c = feats["close"]
        sd = sd or feats["atr"]
        z = (c - mid) / sd
        r = feats["rsi"]
        bias = int(-np.sign(z)) if abs(z) > 0.25 else 0
        long_ok = c < lower and r < p["rsi_low"]
        short_ok = c > upper and r > p["rsi_high"]
        rsi_ext = (p["rsi_low"] - r) / p["rsi_low"] if bias > 0 else (r - p["rsi_high"]) / (100 - p["rsi_high"])
        strength = 0.5 * _clip01((abs(z) - 1.0) / 1.5) + 0.5 * _clip01(0.5 + rsi_ext)
        setup = bool(long_ok or short_ok)
        reason = f"z-score {z:+.2f}, RSI {r:.0f}"
        return bias, strength, setup, reason


class Breakout(Strategy):
    name = "breakout"
    title_th = "เบรกเอาท์"
    defaults = {"lookback": 20, "volume_mult": 1.2, "confirm_atr": 0.0, "stop_atr": 2.0, "tp_atr": 5.0,
                "max_hold": 12, "trail_atr": 2.5}
    space = {"lookback": (10, 60, 1), "volume_mult": (0.8, 2.5, 0.1), "confirm_atr": (0.0, 1.0, 0.1)}

    def _raw(self, df, feats):
        p = self.params
        n = int(p["lookback"])
        hi = float(df["high"].iloc[-n - 1:-1].max())
        lo = float(df["low"].iloc[-n - 1:-1].min())
        c, a = feats["close"], feats["atr"]
        conf = float(p["confirm_atr"]) * a
        mid = (hi + lo) / 2.0
        bias = int(np.sign(c - mid))
        broke_up = c > hi + conf
        broke_down = c < lo - conf
        vr = feats["volume_ratio"]
        vol_ok = vr >= p["volume_mult"]
        strength = 0.45 + 0.3 * _clip01(vr / (2 * p["volume_mult"])) + 0.25 * _clip01(1.0 - feats["bw_pct"])
        if not (broke_up or broke_down):
            strength *= 0.5
        setup = bool((broke_up or broke_down) and vol_ok)
        reason = f"ช่อง {int(p['lookback'])} แท่ง [{lo:.6g}, {hi:.6g}], volume x{vr:.2f}"
        return bias, strength, setup, reason


class Momentum(Strategy):
    name = "momentum"
    title_th = "โมเมนตัม"
    defaults = {"roc_period": 10, "threshold_atr": 2.0, "stop_atr": 2.5, "tp_atr": 3.5, "max_hold": 6,
                "rsi_exhaustion": 82.0}
    space = {"roc_period": (3, 30, 1), "threshold_atr": (0.5, 5.0, 0.25), "rsi_exhaustion": (70.0, 95.0, 1.0)}

    def _raw(self, df, feats):
        p = self.params
        n = int(p["roc_period"])
        close = df["close"].to_numpy(dtype=float)
        if len(close) <= n:
            return 0, 0.0, False, "ข้อมูลไม่พอ"
        move = (close[-1] - close[-1 - n]) / feats["atr"]
        bias = int(np.sign(move))
        strength = _clip01(abs(move) / (2.0 * p["threshold_atr"]))
        exhausted = (bias > 0 and feats["rsi"] > p["rsi_exhaustion"]) or (
            bias < 0 and feats["rsi"] < 100 - p["rsi_exhaustion"])
        if exhausted:
            strength *= 0.5
        setup = bool(abs(move) >= p["threshold_atr"] and not exhausted)
        reason = f"เคลื่อนที่ {move:+.1f} ATR ใน {n} แท่ง, RSI {feats['rsi']:.0f}"
        return bias, strength, setup, reason


STRATEGY_CLASSES = {cls.name: cls for cls in (TrendFollowing, MeanReversion, Breakout, Momentum)}


def build_strategies(saved_params: dict | None = None) -> dict[str, Strategy]:
    saved_params = saved_params or {}
    return {name: cls(saved_params.get(name)) for name, cls in STRATEGY_CLASSES.items()}
