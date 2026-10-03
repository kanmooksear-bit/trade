"""Technical indicators on numpy arrays (fast enough to recompute every bar).

All functions accept array-likes (numpy arrays or pandas Series) and return
numpy arrays aligned with the input.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def ewm(x, alpha: float) -> np.ndarray:
    """Recursive exponential smoothing seeded with the first value (adjust=False)."""
    x = _arr(x)
    out = np.empty_like(x)
    if not len(x):
        return out
    acc = x[0]
    keep = 1.0 - alpha
    for i, v in enumerate(x.tolist()):
        acc = alpha * v + keep * acc
        out[i] = acc
    out[0] = x[0]
    return out


def ema(x, n: int) -> np.ndarray:
    return ewm(x, 2.0 / (max(int(n), 1) + 1.0))


def wilder(x, n: int) -> np.ndarray:
    return ewm(x, 1.0 / max(int(n), 1))


def rolling(x, n: int, fn) -> np.ndarray:
    """Apply ``fn(window_matrix, axis=1)`` over trailing windows; early values use what is available."""
    x = _arr(x)
    n = max(int(n), 1)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    k = min(n, len(x))
    for i in range(k - 1):
        out[i] = fn(x[: i + 1][None, :], axis=1)[0]
    out[k - 1:] = fn(sliding_window_view(x, k), axis=1)
    return out


def sma(x, n: int) -> np.ndarray:
    return rolling(x, n, np.mean)


def true_range(high, low, close) -> np.ndarray:
    high, low, close = _arr(high), _arr(low), _arr(close)
    prev = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))


def atr(high, low, close, n: int = 14) -> np.ndarray:
    return wilder(true_range(high, low, close), n)


def rsi(close, n: int = 14) -> np.ndarray:
    delta = np.diff(_arr(close), prepend=_arr(close)[0])
    gain = wilder(np.clip(delta, 0.0, None), n)
    loss = wilder(np.clip(-delta, 0.0, None), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + gain / loss)
    return np.where(loss > 0, out, np.where(gain > 0, 100.0, 50.0))


def adx(high, low, close, n: int = 14) -> np.ndarray:
    high, low = _arr(high), _arr(low)
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr_n = wilder(true_range(high, low, close), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * wilder(plus_dm, n) / tr_n
        minus_di = 100.0 * wilder(minus_dm, n) / tr_n
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    return wilder(np.nan_to_num(dx, nan=0.0, posinf=0.0), n)


def bollinger(close, n: int = 20, k: float = 2.0):
    close = _arr(close)
    mid = sma(close, n)
    std = rolling(close, max(int(n), 2), lambda w, axis: np.std(w, axis=axis, ddof=1) if w.shape[1] > 1
                  else np.zeros(w.shape[0]))
    upper, lower = mid + k * std, mid - k * std
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.nan_to_num((upper - lower) / mid)
    return mid, upper, lower, std, width


def efficiency_ratio(close, n: int = 20) -> np.ndarray:
    """Kaufman efficiency ratio: 1 = straight line, 0 = pure noise."""
    close = _arr(close)
    out = np.zeros_like(close)
    if len(close) <= n:
        return out
    change = np.abs(close[n:] - close[:-n])
    steps = np.abs(np.diff(close))
    csum = np.concatenate([[0.0], np.cumsum(steps)])
    vol = csum[n:] - csum[:-n]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[n:] = np.nan_to_num(change / vol)
    return out


def donchian(high, low, n: int = 20):
    """Highest high / lowest low of the previous n bars (current bar excluded)."""
    high, low = _arr(high), _arr(low)
    hh = rolling(high, n, np.max)
    ll = rolling(low, n, np.min)
    hh = np.concatenate([[high[0]], hh[:-1]])
    ll = np.concatenate([[low[0]], ll[:-1]])
    return hh, ll


def percentile_rank(x, lookback: int) -> float:
    """Percentile (0..1) of the last value within the trailing window."""
    w = _arr(x)[-int(lookback):]
    w = w[~np.isnan(w)]
    if len(w) < 2:
        return 0.5
    last = w[-1]
    return float(((w < last).sum() + 0.5 * ((w == last).sum() - 1)) / (len(w) - 1))


def volume_ratio(volume, n: int = 20) -> float:
    v = _arr(volume)
    if len(v) < 2 or v[-n:].sum() <= 0:
        return 1.0
    avg = v[-n - 1:-1].mean()
    return float(v[-1] / avg) if avg > 0 else 1.0
