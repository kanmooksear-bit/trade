"""Performance statistics and human-readable (Thai) reports from a journal."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .journal import Journal
from .regime import REGIME_TH


def performance(journal: Journal, starting_cash: float) -> dict:
    curve = journal.equity_curve()
    trades = list(reversed(journal.closed_trades()))
    eq = np.array([c["equity"] for c in curve]) if curve else np.array([starting_cash])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((peak - eq) / peak).max()) if len(eq) else 0.0
    total = float(eq[-1] / starting_cash - 1.0)
    cagr, sharpe = 0.0, 0.0
    if len(curve) > 1:
        # works for any bar size: measure time from timestamps and use end-of-day equity for Sharpe
        series = pd.Series(eq, index=pd.to_datetime([c["time"] for c in curve], utc=True, format="ISO8601"))
        years = max((series.index[-1] - series.index[0]).total_seconds() / (365.25 * 86400), 1 / 365.25)
        cagr = float((eq[-1] / starting_cash) ** (1 / years) - 1.0) if eq[-1] > 0 else -1.0
        daily = series.groupby(series.index.date).last()
        rets = daily.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * np.sqrt(len(daily) / years))
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.array([])
    wins = pnl[pnl > 0].sum() if len(pnl) else 0.0
    losses = -pnl[pnl < 0].sum() if len(pnl) else 0.0
    log = journal.daily_log()
    forced = [t for t in trades if t["forced"]]
    adj = Counter(a["status"] for a in journal.adjustments())
    return {
        "final_equity": float(eq[-1]),
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "avg_r": float(np.mean([t["r_multiple"] for t in trades])) if trades else 0.0,
        "profit_factor": float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0,
        "days": len(log),
        "days_with_entry": sum(1 for d in log if d["entries"] > 0),
        "forced_trades": len(forced),
        "forced_pnl": float(sum(t["pnl"] for t in forced)),
        "adjustments": dict(adj),
        "loss_reviews": sum(1 for _ in journal.conn.execute("SELECT 1 FROM reviews WHERE kind='loss'")),
    }


def breakdown(journal: Journal, key: str) -> list[dict]:
    groups: dict[str, list] = defaultdict(list)
    for t in journal.closed_trades():
        groups[t[key]].append(t)
    rows = []
    for k, ts in sorted(groups.items()):
        r = [t["r_multiple"] for t in ts]
        rows.append({key: k, "trades": len(ts), "win_rate": sum(x > 0 for x in r) / len(r),
                     "avg_r": sum(r) / len(r), "pnl": sum(t["pnl"] for t in ts)})
    return rows


def diagnosis_counts(journal: Journal) -> Counter:
    c: Counter = Counter()
    for row in journal.conn.execute("SELECT diagnoses FROM reviews"):
        for d in json.loads(row["diagnoses"] or "[]"):
            c[f"{d['code']} ({d['title']})"] += 1
    return c


def format_report(journal: Journal, starting_cash: float, recent_reviews: int = 3) -> str:
    p = performance(journal, starting_cash)
    lines = [
        "═══ สรุปผล ═══",
        f"เงินทุนเริ่มต้น {starting_cash:,.2f} → {p['final_equity']:,.2f}  ({p['total_return']:+.1%}, "
        f"CAGR {p['cagr']:+.1%})",
        f"Max drawdown {p['max_drawdown']:.1%} | Sharpe {p['sharpe']:.2f}",
        f"จำนวนไม้ {p['trades']} | ชนะ {p['win_rate']:.0%} | เฉลี่ย {p['avg_r']:+.2f}R | "
        f"Profit factor {p['profit_factor']:.2f}",
        f"วันที่มีการเปิดไม้ใหม่ {p['days_with_entry']}/{p['days']} วัน | ไม้บังคับรายวัน {p['forced_trades']} ไม้ "
        f"(P&L {p['forced_pnl']:+.2f})",
        f"ทบทวนไม้ขาดทุน {p['loss_reviews']} ครั้ง | การปรับพารามิเตอร์: {p['adjustments'] or 'ไม่มี'}",
        "",
        "─── แยกตามกลยุทธ์ ───",
    ]
    for row in breakdown(journal, "strategy"):
        lines.append(f"  {row['strategy']:<15} {row['trades']:>4} ไม้  ชนะ {row['win_rate']:.0%}  "
                     f"{row['avg_r']:+.2f}R  P&L {row['pnl']:+,.2f}")
    lines.append("─── แยกตามสภาวะตลาดตอนเข้า ───")
    for row in breakdown(journal, "regime"):
        lines.append(f"  {REGIME_TH.get(row['regime'], row['regime']):<15} {row['trades']:>4} ไม้  "
                     f"ชนะ {row['win_rate']:.0%}  {row['avg_r']:+.2f}R  P&L {row['pnl']:+,.2f}")
    counts = diagnosis_counts(journal)
    if counts:
        lines.append("─── สาเหตุที่ขาดทุนบ่อย ───")
        for name, n in counts.most_common(8):
            lines.append(f"  {n:>4} × {name}")
    adjustments = journal.adjustments(limit=8)
    if adjustments:
        lines.append("─── การปรับล่าสุด ───")
        for a in adjustments:
            lines.append(f"  [{a['status']}] {a['target']}.{a['param']}: {a['old']} → {a['new']} ({a['diagnosis']})")
    reviews = journal.reviews(limit=recent_reviews)
    if reviews:
        lines.append("─── ตัวอย่างการทบทวนล่าสุด ───")
        for r in reviews:
            lines.append(r["summary"])
    return "\n".join(lines)
