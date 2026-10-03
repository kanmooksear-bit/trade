"""Command line interface.

    python -m autotrader backtest [--days 1500] [--no-learning] [--no-daily]
    python -m autotrader run            # process the newest closed bar (paper or live)
    python -m autotrader check          # stop/target check between bars
    python -m autotrader daemon         # run forever: a cycle per closed bar + stop checks
    python -m autotrader sizing         # is the capital enough for the broker's minimum lot?
    python -m autotrader report         # performance, loss reviews, adjustments
    python -m autotrader params         # current (self-tuned) parameters
    python -m autotrader reset-halt     # re-enable trading after the drawdown kill switch
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .backtest import run_backtest
from .config import load_config
from .data import DataSource, synthetic_ohlcv, timeframe_seconds
from .engine import TradingEngine
from .journal import Journal
from .report import format_report
from .runner import check_stops, daemon, params_text, run_cycle_once, sizing_text


def _load_backtest_data(cfg: dict, days: int) -> dict[str, pd.DataFrame]:
    symbols = cfg["data"]["symbols"]
    tf = timeframe_seconds(cfg["data"]["timeframe"])
    warmup_days = 300 * tf / 86400
    if cfg["data"]["source"] == "synthetic":
        syn = cfg["data"].get("synthetic", {})
        end = pd.Timestamp.now(tz="UTC").floor(f"{tf}s") - pd.Timedelta(seconds=tf)
        start = end - pd.Timedelta(days=days + warmup_days)
        return {s: synthetic_ohlcv(s, start=str(start), end=str(end), seed=int(cfg["data"]["synthetic_seed"]),
                                   timeframe=cfg["data"]["timeframe"],
                                   start_price=float(syn.get("start_price", 100.0)),
                                   vol_scale=float(syn.get("vol_scale", 1.0)),
                                   weekdays_only=bool(syn.get("weekdays_only", False)), digits=syn.get("digits"))
                for s in symbols}
    source = DataSource(cfg)
    bars = int(days * 86400 / tf) + 300
    return {s: source.fetch(s, bars=bars) for s in symbols}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autotrader", description="Adaptive auto-trader")
    parser.add_argument("--config", "-c", default=None, help="YAML config (default: built-in defaults)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bt = sub.add_parser("backtest", help="walk-forward backtest with live learning")
    bt.add_argument("--days", type=int, default=1500)
    bt.add_argument("--source", choices=["synthetic", "ccxt", "yfinance", "csv"])
    bt.add_argument("--no-learning", action="store_true", help="disable reviews/adjustments (for comparison)")
    bt.add_argument("--no-daily", action="store_true", help="disable the trade-every-day rule (for comparison)")
    bt.add_argument("--save-db", default=None, help="keep the backtest journal in this SQLite file")
    bt.add_argument("--verbose", "-v", action="store_true", help="print every trade and review")
    run = sub.add_parser("run", help="process the newest closed bar now")
    run.add_argument("--force", action="store_true", help="run even if this bar was already processed")
    sub.add_parser("check", help="stop / target check between bars")
    sub.add_parser("sizing", help="check capital vs. minimum lot size")
    sub.add_parser("daemon", help="run forever")
    rep = sub.add_parser("report", help="performance and loss reviews")
    rep.add_argument("--reviews", type=int, default=5)
    sub.add_parser("params", help="show current self-tuned parameters")
    sub.add_parser("reset-halt", help="re-enable trading after the drawdown kill switch")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if getattr(args, "source", None):
        overrides.setdefault("data", {})["source"] = args.source
    if getattr(args, "no_learning", False):
        overrides["learning"] = {"enabled": False}
    if getattr(args, "no_daily", False):
        overrides["schedule"] = {"trade_every_day": False}
    cfg = load_config(args.config, overrides)

    if args.cmd == "backtest":
        data = _load_backtest_data(cfg, args.days)
        log = print if args.verbose else None
        result = run_backtest(cfg, data, warmup=250, log=log, db_path=args.save_db or ":memory:")
        print(format_report(result.journal, float(cfg["broker"]["starting_cash"])))
        return 0
    if args.cmd == "run":
        print(run_cycle_once(cfg, print, force=args.force)[0])
        return 0
    if args.cmd == "sizing":
        print(sizing_text(cfg))
        return 0
    if args.cmd == "check":
        check_stops(cfg, print)
        return 0
    if args.cmd == "daemon":
        daemon(cfg, print)
        return 0
    journal = Journal(cfg["storage"]["db_path"])
    try:
        if args.cmd == "report":
            print(format_report(journal, float(cfg["broker"]["starting_cash"]), recent_reviews=args.reviews))
        elif args.cmd == "params":
            print(params_text(journal))
        elif args.cmd == "reset-halt":
            engine = TradingEngine(cfg, journal)
            eq = journal.get_state("last_equity") or engine.broker.cash
            engine.risk.reset_halt(float(eq))
            engine.save()
            print("ปลดล็อกแล้ว — บอทจะเปิดไม้ใหม่ได้ในรอบถัดไป")
    finally:
        journal.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
