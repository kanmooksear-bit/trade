"""SQLite journal: trades, loss reviews, parameter adjustments, equity curve
and the bot's persistent state (so a once-a-day run picks up where it left off)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT, strategy TEXT, regime TEXT, direction INTEGER, qty REAL,
    entry_time TEXT, entry_price REAL, stop REAL, take_profit REAL,
    exit_time TEXT, exit_price REAL, exit_reason TEXT, regime_exit TEXT,
    gross_pnl REAL, fees REAL, pnl REAL, r_multiple REAL,
    forced INTEGER, strength REAL, score REAL, bars_held INTEGER, mfe_r REAL, mae_r REAL,
    entry_features TEXT, exit_features TEXT, params TEXT, signal_reason TEXT,
    status TEXT DEFAULT 'open'
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER, time TEXT, kind TEXT, diagnoses TEXT, summary TEXT, llm TEXT
);
CREATE TABLE IF NOT EXISTS adjustments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT, target TEXT, param TEXT, old REAL, new REAL, diagnosis TEXT, reason TEXT,
    status TEXT, baseline_r REAL, trades_after INTEGER DEFAULT 0, sum_r_after REAL DEFAULT 0,
    resolved_time TEXT
);
CREATE TABLE IF NOT EXISTS equity (time TEXT PRIMARY KEY, equity REAL, cash REAL, open_positions INTEGER);
CREATE TABLE IF NOT EXISTS daily_log (
    date TEXT PRIMARY KEY, entries INTEGER, exits INTEGER, forced INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_COLS = ("entry_features", "exit_features", "params", "diagnoses", "llm")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=float)


class Journal:
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for col in JSON_COLS:
            if col in d and isinstance(d[col], str):
                try:
                    d[col] = json.loads(d[col])
                except json.JSONDecodeError:
                    pass
        return d

    # ----- key/value state -------------------------------------------------
    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT OR REPLACE INTO kv(key, value) VALUES(?, ?)", (key, _dumps(value)))

    # ----- trades ----------------------------------------------------------
    def open_trade(self, **fields: Any) -> int:
        for col in ("entry_features", "params"):
            fields[col] = _dumps(fields.get(col, {}))
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self.conn.execute(f"INSERT INTO trades({cols}) VALUES({marks})", tuple(fields.values()))
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, **fields: Any) -> None:
        fields["exit_features"] = _dumps(fields.get("exit_features", {}))
        fields["status"] = "closed"
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE trades SET {sets} WHERE id=?", (*fields.values(), trade_id))

    def trade(self, trade_id: int) -> dict | None:
        return self._row(self.conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone())

    def closed_trades(self, strategy: str | None = None, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM trades WHERE status='closed'"
        args: list = []
        if strategy:
            q += " AND strategy=?"
            args.append(strategy)
        q += " ORDER BY id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [self._row(r) for r in self.conn.execute(q, args).fetchall()]

    # ----- reviews ---------------------------------------------------------
    def add_review(self, trade_id: int, time: str, kind: str, diagnoses: list, summary: str,
                   llm: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO reviews(trade_id, time, kind, diagnoses, summary, llm) VALUES(?,?,?,?,?,?)",
            (trade_id, time, kind, _dumps(diagnoses), summary, _dumps(llm) if llm else None))

    def reviews(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM reviews ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(r) for r in rows]

    # ----- adjustments -----------------------------------------------------
    def add_adjustment(self, **fields: Any) -> int:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        cur = self.conn.execute(f"INSERT INTO adjustments({cols}) VALUES({marks})", tuple(fields.values()))
        return int(cur.lastrowid)

    def adjustments(self, status: str | None = None, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM adjustments"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY id DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    def update_adjustment(self, adj_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE adjustments SET {sets} WHERE id=?", (*fields.values(), adj_id))

    # ----- equity / daily log ----------------------------------------------
    def record_equity(self, time: str, equity: float, cash: float, open_positions: int) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES(?,?,?,?)", (time, equity, cash, open_positions))

    def equity_curve(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM equity ORDER BY time").fetchall()]

    def log_day(self, date: str, entries: int, exits: int, forced: int, notes: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO daily_log VALUES(?,?,?,?,?)", (date, entries, exits, forced, notes))

    def daily_log(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM daily_log ORDER BY date").fetchall()]
