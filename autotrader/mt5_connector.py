"""Connection to a MetaTrader 5 terminal (e.g. HFM) via the official
``MetaTrader5`` Python package.

The package only runs on Windows, next to an installed MT5 terminal that is
logged in to the trading account with "Algo Trading" enabled. Credentials are
read from environment variables, never from the config file.
"""
from __future__ import annotations

import os

_module = None

MT5_TIMEFRAMES = {
    "1m": "TIMEFRAME_M1", "5m": "TIMEFRAME_M5", "15m": "TIMEFRAME_M15", "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1", "4h": "TIMEFRAME_H4", "1d": "TIMEFRAME_D1",
}


def set_module(module) -> None:
    """Inject a module (tests use a fake MT5)."""
    global _module
    _module = module


def connect(cfg: dict):
    """Initialise the terminal once per process and return the MetaTrader5 module."""
    global _module
    if _module is not None:
        return _module
    try:
        import MetaTrader5 as mt5  # Windows-only optional dependency
    except ImportError as exc:
        raise RuntimeError("ต้องติดตั้ง MetaTrader5 (pip install MetaTrader5) และรันบน Windows "
                           "ที่เปิดโปรแกรม MT5 ของ HFM ไว้") from exc
    m = cfg.get("mt5", {})
    args = [m["path"]] if m.get("path") else []  # terminal path is positional in the MT5 API
    kwargs = {}
    login = os.environ.get(m.get("login_env", "MT5_LOGIN"), "")
    if login:
        kwargs["login"] = int(login)
        kwargs["password"] = os.environ.get(m.get("password_env", "MT5_PASSWORD"), "")
        kwargs["server"] = m.get("server") or os.environ.get(m.get("server_env", "MT5_SERVER"), "")
    if not mt5.initialize(*args, **kwargs):
        raise RuntimeError(f"เชื่อมต่อ MT5 ไม่ได้: {mt5.last_error()} — เปิดโปรแกรม MT5 และล็อกอินไว้หรือยัง?")
    _module = mt5
    return mt5


def timeframe_const(mt5, timeframe: str):
    name = MT5_TIMEFRAMES.get(normalize_timeframe(timeframe))
    if not name:
        raise ValueError(f"MT5 ไม่รองรับ timeframe {timeframe}")
    return getattr(mt5, name)


def normalize_timeframe(tf: str) -> str:
    """Accept both '1h' and MT5-style 'H1'."""
    t = tf.strip()
    mt5_style = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1w"}
    return mt5_style.get(t.upper(), t.lower())


def ensure_symbol(mt5, symbol: str):
    info = mt5.symbol_info(symbol)
    if info is not None and not info.visible:
        mt5.symbol_select(symbol, True)
        info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"ไม่พบสัญลักษณ์ {symbol} ใน MT5 — ดูชื่อที่ถูกต้องใน Market Watch (บางบัญชีมีตัวต่อท้าย)")
    return info
