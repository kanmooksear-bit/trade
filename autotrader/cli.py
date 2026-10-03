"""Command line interface.

    python -m autotrader backtest [--days 1500] [--no-learning] [--no-daily]
    python -m autotrader run            # one daily cycle (paper or live)
    python -m autotrader check          # intraday stop/target check
    python -m autotrader daemon         # run forever: daily cycle + periodic stop checks
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
from .data import DataSource, synthetic_ohlcv
from .engine import TradingEngine
from .journal import Journal
from .report import format_report
from .runner import check_stops, daemon, params_text, run_daily


def _load_backtest_data(cfg: dict, days: int) -> dict[str, pd.DataFrame]:
    symbols = cfg["data"]["symbols"]
    if cfg["data"]["source"] == "synthetic":
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days + 300)
        return {s: synthetic_ohlcv(s, start=str(start.date()), end=str(end.date()),
                                   seed=int(cfg["data"]["synthetic_seed"])) for s in symbols}
    source = DataSource(cfg)
    return {s: source.fetch(s, bars=days + 300) for s in symbols}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autotrader", description="Adaptive daily auto-trader")
    parser.add_argument("--config", "-c", default=None, help="YAML config (default: built-in defaults)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    bt = sub.add_parser("backtest", help="walk-forward backtest with live learning")
    bt.add_argument("--days", type=int, default=1500)
    bt.add_argument("--source", choices=["synthetic", "ccxt", "yfinance", "csv"])
    bt.add_argument("--no-learning", action="store_true", help="disable reviews/adjustments (for comparison)")
    bt.add_argument("--no-daily", action="store_true", help="disable the trade-every-day rule (for comparison)")
    bt.add_argument("--save-db", default=None, help="keep the backtest journal in this SQLite file")
    bt.add_argument("--verbose", "-v", action="store_true", help="print every trade and review")
    run = sub.add_parser("run", help="run one daily cycle now")
    run.add_argument("--force", action="store_true", help="run even if this bar was already processed")
    sub.add_parser("check", help="intraday stop / target check")
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
        print(run_daily(cfg, print, force=args.force))
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
