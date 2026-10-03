"""Walk-forward backtest that runs the exact same engine as live trading:
decisions at each bar's close, fills at the next bar's open, learning on the fly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .broker import PaperBroker
from .engine import TradingEngine
from .journal import Journal
from .report import performance

WINDOW = 300


@dataclass
class BacktestResult:
    journal: Journal
    engine: TradingEngine
    stats: dict


def run_backtest(cfg: dict, data: dict[str, pd.DataFrame], warmup: int = 250,
                 log: Callable[[str], None] | None = None, db_path: str = ":memory:") -> BacktestResult:
    journal = Journal(db_path)
    broker = PaperBroker(cfg["broker"], {})
    engine = TradingEngine(cfg, journal, broker=broker, log=log)
    index = sorted(set().union(*(df.index for df in data.values())))
    positions = {s: pd.Series(range(len(df)), index=df.index) for s, df in data.items()}
    long_window = int(cfg["learning"].get("whatif_bars", 1000)) + WINDOW
    pending = []
    for i in range(max(warmup, 1), len(index)):
        ts, prev = index[i], index[i - 1]
        window, opens, longer = {}, {}, {}
        for s, df in data.items():
            if ts not in positions[s].index:
                continue
            j = int(positions[s][ts])
            window[s] = df.iloc[max(0, j - WINDOW + 1): j + 1]
            longer[s] = df.iloc[max(0, j - long_window + 1): j + 1]
            opens[s] = float(df["open"].iloc[j])
        engine.fill_orders(pending, opens, str(ts), {s: str(prev) for s in opens})
        engine.long_history = longer
        pending = engine.run_cycle(window, str(ts))
    engine.save()
    return BacktestResult(journal, engine, performance(journal, float(cfg["broker"]["starting_cash"])))
