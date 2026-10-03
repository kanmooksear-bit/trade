import numpy as np

from autotrader.regime import (HIGH_VOL, RANGE, REGIME_DEFAULTS, TREND_DOWN, TREND_UP, RegimeDetector,
                               classify, compute_features)
from tests.conftest import make_ohlcv


def _trend(rng, drift, n=300):
    return 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.004, n)))


def test_uptrend_and_downtrend_detected(rng):
    up = compute_features(make_ohlcv(_trend(rng, 0.006)))
    down = compute_features(make_ohlcv(_trend(rng, -0.006)))
    assert classify(up, REGIME_DEFAULTS) == TREND_UP
    assert classify(down, REGIME_DEFAULTS) == TREND_DOWN


def test_choppy_market_is_range(rng):
    closes = 100 + 3 * np.sin(np.arange(300) / 3) + rng.normal(0, 0.5, 300)
    f = compute_features(make_ohlcv(closes, spread=0.004))
    assert classify(f, {**REGIME_DEFAULTS, "squeeze_pct": 0.0, "high_vol_pct": 1.01}) == RANGE


def test_volatility_spike_is_high_vol(rng):
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 280)))
    wild = calm[-1] * np.exp(np.cumsum(rng.normal(0, 0.08, 20)))
    df = make_ohlcv(np.concatenate([calm, wild]), spread=0.002)
    df.iloc[-20:, df.columns.get_loc("high")] *= 1.08
    df.iloc[-20:, df.columns.get_loc("low")] *= 0.92
    assert classify(compute_features(df), REGIME_DEFAULTS) == HIGH_VOL


def test_regime_needs_confirmation_and_ignores_repeated_bar(rng):
    det = RegimeDetector({**REGIME_DEFAULTS, "confirm_bars": 2})
    df = make_ohlcv(_trend(rng, 0.006))
    det.state["X"] = {"stable": RANGE, "candidate": RANGE, "count": 5}
    first = det.detect("X", df.iloc[:-1])
    assert first.raw == TREND_UP and first.regime == RANGE  # seen once: not yet confirmed
    again = det.detect("X", df.iloc[:-1])  # same bar again (second run on the same day)
    assert again.regime == RANGE and det.state["X"]["count"] == 1
    second = det.detect("X", df)
    assert second.regime == TREND_UP
