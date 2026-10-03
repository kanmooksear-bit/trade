"""Market data sources. All return a DataFrame indexed by timestamp with
columns open, high, low, close, volume, containing only *closed* bars.

Sources: synthetic (offline testing), ccxt (crypto), yfinance (stocks / GC=F),
csv (incl. MetaTrader history exports) and mt5 (live terminal, e.g. HFM).
MT5 bars keep the broker's server clock, exactly as shown in the terminal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .mt5_connector import connect as mt5_connect
from .mt5_connector import ensure_symbol, normalize_timeframe, timeframe_const

TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
                     "1w": 604800}
COLUMNS = ["open", "high", "low", "close", "volume"]


def timeframe_seconds(tf: str) -> int:
    return TIMEFRAME_SECONDS.get(normalize_timeframe(tf), 86400)


def drop_incomplete(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the last bar if it is still forming."""
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="UTC")
    last = df.index[-1]
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    if last + pd.Timedelta(seconds=timeframe_seconds(timeframe)) > now:
        return df.iloc[:-1]
    return df


def synthetic_ohlcv(symbol: str, start: str = "2019-01-01", end: str | None = None, seed: int = 7,
                    start_price: float = 100.0, timeframe: str = "1d", vol_scale: float = 1.0,
                    weekdays_only: bool = False, digits: int | None = None) -> pd.DataFrame:
    """Regime-switching random walk (trend up/down, range, squeeze, high vol).

    Deterministic per (symbol, seed) and anchored at ``start`` so a daily run
    sees the same history plus new bars. Volatility and regime lengths are
    expressed per day and scaled to the bar size.
    """
    tf = timeframe_seconds(timeframe)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").floor(f"{tf}s") - pd.Timedelta(seconds=tf)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), end_ts, freq=f"{tf}s")
    if weekdays_only:
        index = index[index.dayofweek < 5]
    n = len(index)
    bars_per_day = max(1.0, 86400 / tf)
    h = int(hashlib.sha256(f"{symbol}:{seed}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    regimes = {  # (drift per day, volatility per day)
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
        length = int(rng.integers(25, 110) * bars_per_day)
        drift, vol = regimes[name]
        drift = drift * vol_scale / bars_per_day
        vol = vol * vol_scale / np.sqrt(bars_per_day)
        pull = 0.08 / bars_per_day
        anchor = price
        for _ in range(min(length, n - i)):
            shock = rng.standard_t(5) * vol / np.sqrt(5 / 3)
            ret = pull * np.log(anchor / price) + shock if name in ("range", "squeeze") else drift + shock
            price = max(price * np.exp(ret), 1e-6)
            closes[i], vols[i] = price, vol
            i += 1
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1] * np.exp(rng.normal(0, vols[1:] * 0.25))
    highs = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0, vols * 0.6)))
    lows = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0, vols * 0.6)))
    move = np.abs(np.log(closes / opens))
    volume = 1_000_000 * np.exp(rng.normal(0, 0.3, n)) * (1 + 25 * move / max(vol_scale, 1e-9))
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
                      index=index)
    if digits is not None:
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].round(digits)
    return df


class DataSource:
    def __init__(self, cfg: dict):
        self.full_cfg = cfg
        self.cfg = cfg["data"]
        self.timeframe = normalize_timeframe(self.cfg.get("timeframe", "1d"))
        self.source = self.cfg.get("source", "synthetic")
        self._exchange = None

    def fetch(self, symbol: str, bars: int | None = None) -> pd.DataFrame:
        bars = int(bars or self.cfg.get("history_bars", 400))
        if self.source == "synthetic":
            s = self.cfg.get("synthetic", {})
            df = synthetic_ohlcv(symbol, seed=int(self.cfg.get("synthetic_seed", 7)), timeframe=self.timeframe,
                                 start=s.get("start", "2019-01-01"), start_price=float(s.get("start_price", 100.0)),
                                 vol_scale=float(s.get("vol_scale", 1.0)),
                                 weekdays_only=bool(s.get("weekdays_only", False)), digits=s.get("digits"))
        elif self.source == "ccxt":
            df = self._fetch_ccxt(symbol, bars + 1)
        elif self.source == "yfinance":
            df = self._fetch_yfinance(symbol, bars + 1)
        elif self.source == "csv":
            files = self.cfg.get("csv_files") or {}
            path = files.get(symbol) or Path(self.cfg.get("csv_dir", "data/csv")) / f"{symbol.replace('/', '_')}.csv"
            df = load_csv(Path(path))
        elif self.source == "mt5":
            return self._fetch_mt5(symbol, bars)  # already excludes the forming bar
        else:
            raise ValueError(f"unknown data source: {self.source}")
        if self.source in ("ccxt", "yfinance", "synthetic"):
            df = drop_incomplete(df, self.timeframe)
        return df.iloc[-bars:]

    def _fetch_mt5(self, symbol: str, bars: int) -> pd.DataFrame:
        mt5 = mt5_connect(self.full_cfg)
        ensure_symbol(mt5, symbol)
        # start_pos=1 skips the bar that is still forming
        rates = mt5.copy_rates_from_pos(symbol, timeframe_const(mt5, self.timeframe), 1, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 ไม่ส่งข้อมูล {symbol}: {mt5.last_error()}")
        df = pd.DataFrame({
            "open": rates["open"], "high": rates["high"], "low": rates["low"], "close": rates["close"],
            "volume": rates["tick_volume"],
        }, index=pd.to_datetime(rates["time"], unit="s", utc=True))
        return df.astype(float)

    def _fetch_ccxt(self, symbol: str, limit: int) -> pd.DataFrame:
        import ccxt  # optional dependency

        if self._exchange is None:
            self._exchange = getattr(ccxt, self.cfg.get("exchange", "binance"))({"enableRateLimit": True})
        step_ms = timeframe_seconds(self.timeframe) * 1000
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

        interval = {"1d": "1d", "1h": "1h", "1w": "1wk", "15m": "15m", "30m": "30m"}.get(self.timeframe, "1d")
        days = int(limit * timeframe_seconds(self.timeframe) / 86400 * 1.6) + 10  # weekends/holidays
        start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).date().isoformat()
        raw = yf.Ticker(symbol).history(start=start, interval=interval, auto_adjust=True)
        df = raw.rename(columns=str.lower)[COLUMNS]
        df.index = pd.to_datetime(df.index, utc=True)
        return df.astype(float)

    def latest_price(self, symbol: str, df: pd.DataFrame) -> float:
        """Price used to value positions and fill paper orders right now."""
        if self.source == "ccxt" and self._exchange is not None:
            try:
                return float(self._exchange.fetch_ticker(symbol)["last"])
            except Exception:  # noqa: BLE001 - fall back to the last close
                pass
        if self.source == "mt5":
            tick = mt5_connect(self.full_cfg).symbol_info_tick(symbol)
            if tick is not None and tick.bid > 0:
                return float(tick.bid)  # MT5 charts are built from bid prices
        return float(df["close"].iloc[-1])

    def now(self) -> pd.Timestamp:
        """Current time on the same clock as the bars (broker server time for MT5)."""
        if self.source == "mt5":
            mt5 = mt5_connect(self.full_cfg)
            symbols = self.cfg.get("symbols") or []
            tick = mt5.symbol_info_tick(symbols[0]) if symbols else None
            if tick is not None and tick.time:
                return pd.Timestamp(int(tick.time), unit="s", tz="UTC")
        return pd.Timestamp.now(tz="UTC")


def load_csv(path: Path) -> pd.DataFrame:
    """Generic OHLCV CSV, or a MetaTrader export (<DATE> <TIME> <OPEN> ... tab separated)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]
    mt5_date = (r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3")  # MetaTrader writes 2024.01.31
    if "date" in df.columns and "time" in df.columns:
        stamp = df.pop("date").astype(str).str.replace(*mt5_date, regex=True) + " " + df.pop("time").astype(str)
    else:
        name = next((c for c in ("timestamp", "datetime", "date", "time") if c in df.columns), df.columns[0])
        col = df.pop(name)
        if pd.api.types.is_numeric_dtype(col):
            stamp = pd.to_datetime(col, unit="ms" if col.max() > 1e11 else "s", utc=True)
        else:
            stamp = col.astype(str).str.replace(*mt5_date, regex=True)
    df.index = pd.to_datetime(stamp, utc=True)
    if "volume" not in df:
        vol = df.get("vol")
        tick = df.get("tickvol")
        df["volume"] = vol if vol is not None and float(vol.sum()) > 0 else (tick if tick is not None else 0.0)
    return df[COLUMNS].astype(float).sort_index()
