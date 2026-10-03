"""Market data sources. All return a DataFrame indexed by UTC timestamp with
columns open, high, low, close, volume, containing only *closed* bars."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TIMEFRAME_SECONDS = {"1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
COLUMNS = ["open", "high", "low", "close", "volume"]


def drop_incomplete(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the last bar if it is still forming."""
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    seconds = TIMEFRAME_SECONDS.get(timeframe, 86400)
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    if last + pd.Timedelta(seconds=seconds) > now:
        return df.iloc[:-1]
    return df


def synthetic_ohlcv(symbol: str, start: str = "2019-01-01", end: str | None = None, seed: int = 7,
                    start_price: float = 100.0) -> pd.DataFrame:
    """Regime-switching random walk (trend up/down, range, squeeze, high vol).

    Deterministic per (symbol, seed) and anchored at ``start`` so a daily run
    sees the same history plus one new bar each day.
    """
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), end_ts, freq="D")
    n = len(index)
    h = int(hashlib.sha256(f"{symbol}:{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    regimes = {
        "up": (0.004, 0.018), "down": (-0.004, 0.022), "range": (0.0, 0.014),
        "squeeze": (0.0, 0.006), "wild": (0.0, 0.045),
    }
    names = list(regimes)
    probs = [0.24, 0.18, 0.30, 0.13, 0.15]
    closes = np.empty(n)
    vols = np.empty(n)
    price, anchor = start_price, start_price
    i = 0
    while i < n:
        name = rng.choice(names, p=probs)
        length = int(rng.integers(25, 110))
        drift, vol = regimes[name]
        anchor = price
        for _ in range(min(length, n - i)):
            shock = rng.standard_t(5) * vol / np.sqrt(5 / 3)
            if name in ("range", "squeeze"):
                ret = 0.08 * np.log(anchor / price) + shock
            else:
                ret = drift + shock
            price = max(price * np.exp(ret), 1e-6)
            closes[i], vols[i] = price, vol
            i += 1
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1] * np.exp(rng.normal(0, vols[1:] * 0.25))
    body_hi = np.maximum(opens, closes)
    body_lo = np.minimum(opens, closes)
    highs = body_hi * np.exp(np.abs(rng.normal(0, vols * 0.6)))
    lows = body_lo * np.exp(-np.abs(rng.normal(0, vols * 0.6)))
    move = np.abs(np.log(closes / opens))
    volume = 1_000_000 * np.exp(rng.normal(0, 0.3, n)) * (1 + 25 * move)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
                        index=index)


class DataSource:
    def __init__(self, cfg: dict):
        self.cfg = cfg["data"]
        self.timeframe = self.cfg.get("timeframe", "1d")
        self._exchange = None

    def fetch(self, symbol: str, bars: int | None = None) -> pd.DataFrame:
        bars = int(bars or self.cfg.get("history_bars", 400))
        source = self.cfg.get("source", "synthetic")
        if source == "synthetic":
            df = synthetic_ohlcv(symbol, seed=int(self.cfg.get("synthetic_seed", 7)))
        elif source == "ccxt":
            df = self._fetch_ccxt(symbol, bars + 1)
        elif source == "yfinance":
            df = self._fetch_yfinance(symbol, bars + 1)
        elif source == "csv":
            df = load_csv(Path(self.cfg.get("csv_dir", "data/csv")) / f"{symbol.replace('/', '_')}.csv")
        else:
            raise ValueError(f"unknown data source: {source}")
        df = drop_incomplete(df, self.timeframe)
        return df.iloc[-bars:]

    def _fetch_ccxt(self, symbol: str, limit: int) -> pd.DataFrame:
        import ccxt  # optional dependency

        if self._exchange is None:
            self._exchange = getattr(ccxt, self.cfg.get("exchange", "binance"))({"enableRateLimit": True})
        step_ms = TIMEFRAME_SECONDS.get(self.timeframe, 86400) * 1000
        since = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - limit * step_ms
        rows: list = []
        while len(rows) < limit:  # exchanges cap candles per request, so page forward in time
            batch = self._exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=min(1000, limit))
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 2:
                break
            since = batch[-1][0] + step_ms
        df = pd.DataFrame(rows, columns=["ts", *COLUMNS]).drop_duplicates("ts")
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df.astype(float)

    def _fetch_yfinance(self, symbol: str, limit: int) -> pd.DataFrame:
        import yfinance as yf  # optional dependency

        interval = {"1d": "1d", "1h": "1h", "1w": "1wk"}.get(self.timeframe, "1d")
        days = int(limit * TIMEFRAME_SECONDS.get(self.timeframe, 86400) / 86400 * 1.6) + 10  # weekends/holidays
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).date().isoformat()
        raw = yf.Ticker(symbol).history(start=start, interval=interval, auto_adjust=True)
        df = raw.rename(columns=str.lower)[COLUMNS]
        df.index = pd.to_datetime(df.index, utc=True)
        return df.astype(float)

    def latest_price(self, symbol: str, df: pd.DataFrame) -> float:
        """Price used to fill orders right now (live/paper runs)."""
        if self.cfg.get("source") == "ccxt" and self._exchange is not None:
            try:
                return float(self._exchange.fetch_ticker(symbol)["last"])
            except Exception:  # noqa: BLE001 - fall back to the last close
                pass
        return float(df["close"].iloc[-1])


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in ("timestamp", "date", "time", "datetime") if c in df.columns)
    df.index = pd.to_datetime(df.pop(time_col), utc=True)
    if "volume" not in df:
        df["volume"] = 0.0
    return df[COLUMNS].astype(float).sort_index()
