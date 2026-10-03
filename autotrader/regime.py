"""Market regime detection.

Each bar is classified as one of five regimes from trend strength (ADX,
efficiency ratio, EMA slope), volatility percentile and Bollinger-band width.
A regime only becomes "stable" after it has been seen ``confirm_bars`` times
in a row, so the strategy selector does not flip-flop on noise; a jump into
HIGH_VOL is accepted immediately because it is a risk event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind

TREND_UP = "trend_up"
TREND_DOWN = "trend_down"
RANGE = "range"
SQUEEZE = "squeeze"
HIGH_VOL = "high_vol"
REGIMES = (TREND_UP, TREND_DOWN, RANGE, SQUEEZE, HIGH_VOL)

REGIME_TH = {
    TREND_UP: "ขาขึ้น",
    TREND_DOWN: "ขาลง",
    RANGE: "ไซด์เวย์",
    SQUEEZE: "บีบตัว (รอเบรก)",
    HIGH_VOL: "ผันผวนสูง",
}

REGIME_DEFAULTS = {
    "adx_trend": 20.0,
    "er_trend": 0.20,
    "high_vol_pct": 0.90,
    "high_vol_ratio": 1.3,  # and at least this multiple of the median volatility
    "squeeze_pct": 0.20,
    "vol_lookback": 120,
    "confirm_bars": 2,
}

REGIME_SPACE = {
    "adx_trend": (15.0, 35.0, 1.0),
    "er_trend": (0.10, 0.45, 0.025),
    "high_vol_pct": (0.75, 0.97, 0.02),
    "high_vol_ratio": (1.05, 2.0, 0.05),
    "squeeze_pct": (0.05, 0.35, 0.025),
    "vol_lookback": (60, 250, 10),
    "confirm_bars": (1, 5, 1),
}


@dataclass
class RegimeReading:
    regime: str
    raw: str
    features: dict = field(default_factory=dict)


def compute_features(df: pd.DataFrame, vol_lookback: int = 120) -> dict:
    o, h, l, close = (df[c].to_numpy(dtype=float) for c in ("open", "high", "low", "close"))
    volume = df["volume"].to_numpy(dtype=float) if "volume" in df else np.zeros(len(df))
    atr_s = ind.atr(h, l, close, 14)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    atr_pct = ind.wilder(ind.true_range(h, l, close) / prev_close, 14)  # scale-free, no lag bias in trends
    lookback_atr = atr_pct[-int(vol_lookback):]
    ema20 = ind.ema(close, 20)
    ema50 = ind.ema(close, 50)
    ema200 = ind.ema(close, 200)
    _, _, _, _, bb_width = ind.bollinger(close, 20, 2.0)
    c = float(close[-1])
    a = float(atr_s[-1]) or c * 1e-4
    slope_bars = 10
    slope50 = (ema50[-1] - ema50[-1 - slope_bars]) / a if len(close) > slope_bars else 0.0
    return {
        "close": c,
        "open": float(o[-1]),
        "high": float(h[-1]),
        "low": float(l[-1]),
        "atr": a,
        "atr_pct": float(atr_pct[-1]),
        "adx": float(ind.adx(h, l, close, 14)[-1]),
        "er": float(ind.efficiency_ratio(close, 20)[-1]),
        "rsi": float(ind.rsi(close, 14)[-1]),
        "ema20": float(ema20[-1]),
        "ema50": float(ema50[-1]),
        "ema200": float(ema200[-1]),
        "slope50": float(slope50),
        "vol_pct": ind.percentile_rank(atr_pct, vol_lookback),
        "vol_ratio": float(atr_pct[-1] / np.median(lookback_atr)) if np.median(lookback_atr) > 0 else 1.0,
        "bw_pct": ind.percentile_rank(bb_width, vol_lookback),
        "extension": (c - float(ema20[-1])) / a,
        "htf_trend": int(np.sign(c - float(ema200[-1]))),
        "volume_ratio": ind.volume_ratio(volume, 20),
        "bar_range_atr": (float(h[-1]) - float(l[-1])) / a,
    }


def classify(features: dict, p: dict) -> str:
    f = features
    if f["vol_pct"] >= p["high_vol_pct"] and f.get("vol_ratio", 2.0) >= p["high_vol_ratio"]:
        return HIGH_VOL
    if f["adx"] >= p["adx_trend"] and f["er"] >= p["er_trend"]:
        if f["close"] > f["ema50"] and f["slope50"] > 0:
            return TREND_UP
        if f["close"] < f["ema50"] and f["slope50"] < 0:
            return TREND_DOWN
    if f["bw_pct"] <= p["squeeze_pct"]:
        return SQUEEZE
    return RANGE


class RegimeDetector:
    def __init__(self, params: dict, state: dict | None = None):
        self.params = params
        # per symbol: {"stable", "candidate", "count", "last_bar", "reading"}
        self.state: dict = state if state is not None else {}

    def detect(self, symbol: str, df: pd.DataFrame) -> RegimeReading:
        bar = str(df.index[-1])
        st = self.state.get(symbol)
        if st and st.get("last_bar") == bar and st.get("reading"):
            r = st["reading"]  # same bar seen twice (e.g. two runs in one day): do not double count
            return RegimeReading(r["regime"], r["raw"], r["features"])
        feats = compute_features(df, int(self.params["vol_lookback"]))
        raw = classify(feats, self.params)
        if not st:
            st = {"stable": raw, "candidate": raw, "count": 0}
        if raw == st["candidate"]:
            st["count"] += 1
        else:
            st["candidate"], st["count"] = raw, 1
        if raw == HIGH_VOL or st["count"] >= int(self.params["confirm_bars"]):
            st["stable"] = raw
        feats["regime_raw"] = raw
        reading = RegimeReading(st["stable"], raw, feats)
        st["last_bar"] = bar
        st["reading"] = {"regime": reading.regime, "raw": raw, "features": feats}
        self.state[symbol] = st
        return reading
