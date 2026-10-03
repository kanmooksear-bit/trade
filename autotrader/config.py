"""Configuration: built-in defaults deep-merged with an optional YAML file."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "mode": "paper",  # paper | live
    "data": {
        "source": "synthetic",  # synthetic | ccxt | yfinance | csv
        "exchange": "binance",
        "timeframe": "1d",
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"],
        "history_bars": 1200,  # must cover learning.whatif_bars
        "csv_dir": "data/csv",
        "csv_files": {},  # optional {symbol: path}, e.g. an MT5 history export
        "synthetic_seed": 7,
        "synthetic": {"start": "2019-01-01", "start_price": 100.0, "vol_scale": 1.0, "weekdays_only": False,
                      "digits": None},
    },
    "broker": {
        "type": "paper",  # paper | ccxt | mt5
        "exchange": "binance",
        "sandbox": True,
        "starting_cash": 10_000.0,
        "fee_rate": 0.001,
        "slippage_bps": 5.0,
        "allow_short": False,
        "min_notional": 10.0,
        # CFD / MT5 style costs and sizes (crypto defaults: no lots, % fee)
        "spread": 0.0,  # in price units, e.g. 0.30 = 30 points on 2-digit XAUUSD
        "commission_per_lot": 0.0,  # per side
        "contract_size": 1.0,  # units per lot (XAUUSD: 100 oz)
        "volume_min": 0.0,  # lots
        "volume_step": 0.0,  # lots, 0 = any size
        "digits": None,  # price decimals (XAUUSD at HFM: 2)
        "magic": 26100301,  # MT5 magic number marking this bot's orders
        "deviation_points": 30,  # MT5 max slippage in points
        "api_key_env": "EXCHANGE_API_KEY",
        "api_secret_env": "EXCHANGE_API_SECRET",
    },
    "risk": {
        "risk_per_trade": 0.01,  # fraction of equity lost if the stop is hit
        "max_position_pct": 0.30,
        "max_gross_exposure": 1.0,
        "max_open_positions": 3,
        "daily_loss_limit": 0.03,
        "max_drawdown": 0.25,  # kill switch: halt new entries until reset
        "losing_streak_reduce_after": 3,
        "min_risk_multiplier": 0.25,
        "min_score": 0.35,  # candidate score needed for a normal (non-forced) entry
        "max_min_lot_risk": 0.02,  # if 1% risk is below the minimum lot, allow the minimum lot up to this risk
    },
    "schedule": {
        "trade_every_day": True,
        "probe_slots": 2,  # extra slots reserved for forced daily trades (on top of max_open_positions)
        "probe_max_hold": 2,  # forced trades are closed after this many bars
        "probe_after_hour": 0,  # intraday: only force a trade after this hour (bar clock) if none yet today
        "stop_check_minutes": 60,
        "poll_seconds": 60,  # daemon: how often to look for a newly closed bar
    },
    "mt5": {
        "path": "",  # terminal64.exe path if MT5 is not found automatically
        "server": "",  # e.g. the HFM server name shown in the MT5 login window
        "login_env": "MT5_LOGIN",
        "password_env": "MT5_PASSWORD",
        "server_env": "MT5_SERVER",
    },
    "learning": {
        "enabled": True,
        "evidence_required": 2,  # same diagnosis this many times before a change is considered
        "whatif": True,  # replay a proposed change on recent history; adopt only if clearly better
        "whatif_bars": 1000,  # bars per symbol used for that replay
        "whatif_min_gain_r": 1.0,  # required improvement in total R over the replay
        "whatif_min_trades": 8,
        "probation_trades": 10,  # live trades used to judge whether an adopted change helped
        "revert_tolerance": 0.2,  # revert if average R after the change is this much worse (R-multiple)
        "hindsight_bars": 5,
        "selector_alpha": 0.2,
        "selector_learning_rate": 0.6,
        "cooldown_after_losses": 3,
        "cooldown_bars": 10,
    },
    "llm_review": {
        "enabled": False,
        "model": "claude-opus-5-5",
        "effort": "medium",
        "apply_suggestions": False,
        "max_reviews_per_day": 5,
    },
    "notify": {"telegram_bot_token_env": "TELEGRAM_BOT_TOKEN", "telegram_chat_id_env": "TELEGRAM_CHAT_ID"},
    "storage": {"db_path": "data/autotrader.db"},
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as fh:
                cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
        else:
            raise FileNotFoundError(f"config file not found: {p}")
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
