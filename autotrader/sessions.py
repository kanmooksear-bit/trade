"""Trading hours.

Bar timestamps are on the data source's clock: UTC for ccxt / synthetic data,
and the broker's server clock for MetaTrader 5. HFM (like most MT5 brokers)
runs its server at GMT+2, and GMT+3 while the USA is on daylight-saving time,
so the gold day starts at the New York close.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd


def _nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def us_dst(ts_utc: pd.Timestamp) -> bool:
    """USA daylight-saving time: 2nd Sunday of March 07:00 UTC to 1st Sunday of November 06:00 UTC."""
    naive = ts_utc.tz_convert("UTC").tz_localize(None).to_pydatetime() if ts_utc.tzinfo else ts_utc.to_pydatetime()
    start = _nth_sunday(naive.year, 3, 2) + timedelta(hours=7)
    end = _nth_sunday(naive.year, 11, 1) + timedelta(hours=6)
    return start <= naive < end


def server_offset_hours(ts: pd.Timestamp, server_tz) -> float:
    """Hours to add to UTC to get the data clock. ``server_tz``: 'utc', 'mt5_gmt2_us_dst', or a number."""
    if server_tz in (None, "", "utc", "UTC"):
        return 0.0
    if server_tz == "mt5_gmt2_us_dst":
        # server-clock ts is ~2-3h ahead of UTC; testing DST on (ts - 2h) is accurate except within an hour of a switch
        return 3.0 if us_dst(ts - pd.Timedelta(hours=2)) else 2.0
    return float(server_tz)


def local_time(bar_clock_ts: pd.Timestamp, server_tz, local_offset_hours: float) -> pd.Timestamp:
    """Convert a timestamp on the data clock to the trader's local clock."""
    utc = bar_clock_ts - pd.Timedelta(hours=server_offset_hours(bar_clock_ts, server_tz))
    return utc + pd.Timedelta(hours=float(local_offset_hours))


def parse_hhmm(text: str) -> time:
    h, m = str(text).split(":")
    return time(int(h), int(m))


def in_window(local_ts: pd.Timestamp, start: str, end: str) -> bool:
    """True when ``start <= local time < end`` (windows crossing midnight are supported)."""
    t, s, e = local_ts.time(), parse_hhmm(start), parse_hhmm(end)
    return s <= t < e if s <= e else (t >= s or t < e)
