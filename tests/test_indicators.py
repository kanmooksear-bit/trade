import numpy as np
import pandas as pd

from autotrader import indicators as ind


def test_ema_matches_pandas(rng):
    x = rng.normal(100, 5, 300)
    expected = pd.Series(x).ewm(span=20, adjust=False).mean().to_numpy()
    assert np.allclose(ind.ema(x, 20), expected)


def test_wilder_atr_matches_pandas(rng):
    close = 100 + np.cumsum(rng.normal(0, 1, 200))
    high, low = close + 1.0, close - 1.0
    prev = pd.Series(close).shift(1)
    tr = pd.concat([pd.Series(high - low), (pd.Series(high) - prev).abs(), (pd.Series(low) - prev).abs()],
                   axis=1).max(axis=1).fillna(2.0)
    expected = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    assert np.allclose(ind.atr(high, low, close, 14), expected)


def test_rsi_bounds_and_extremes():
    up = np.arange(1, 100, dtype=float)
    assert ind.rsi(up, 14)[-1] == 100.0
    assert ind.rsi(up[::-1], 14)[-1] < 1.0
    noisy = 100 + np.sin(np.arange(200))
    r = ind.rsi(noisy, 14)
    assert np.all((r >= 0) & (r <= 100))


def test_efficiency_ratio_straight_line_is_one():
    assert ind.efficiency_ratio(np.linspace(1, 50, 60), 20)[-1] == 1.0
    zigzag = np.tile([1.0, 2.0], 40)
    assert ind.efficiency_ratio(zigzag, 20)[-1] < 0.1


def test_adx_high_in_trend_low_in_chop(rng):
    trend = np.linspace(100, 200, 200)
    chop = 100 + rng.normal(0, 1, 200)
    adx_trend = ind.adx(trend * 1.005, trend * 0.995, trend, 14)[-1]
    adx_chop = ind.adx(chop + 1, chop - 1, chop, 14)[-1]
    assert adx_trend > 40 > adx_chop


def test_donchian_excludes_current_bar():
    high = np.array([1, 2, 3, 10], dtype=float)
    hh, _ = ind.donchian(high, high, 3)
    assert hh[-1] == 3.0


def test_bollinger_and_rolling_early_values():
    x = np.arange(1, 31, dtype=float)
    mid, upper, lower, std, width = ind.bollinger(x, 20, 2.0)
    assert mid[-1] == np.mean(x[-20:])
    assert np.isclose(std[-1], np.std(x[-20:], ddof=1))
    assert mid[0] == 1.0 and std[0] == 0.0
    assert np.all(upper >= lower)


def test_percentile_rank():
    assert ind.percentile_rank([1, 2, 3, 4, 5], 5) == 1.0
    assert ind.percentile_rank([5, 4, 3, 2, 1], 5) == 0.0
