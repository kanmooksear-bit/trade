"""Paper/live operation: one daily cycle, intraday stop checks, and a daemon loop."""
from __future__ import annotations

import json
import os
import time as _time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from .data import DataSource
from .engine import TradingEngine
from .journal import Journal
from .llm_review import build_llm_reviewer
from .regime import REGIME_TH


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")


def notify(cfg: dict, text: str, log=print) -> None:
    """Send a message to Telegram if TELEGRAM_BOT_TOKEN / chat id are configured."""
    n = cfg.get("notify", {})
    token = os.environ.get(n.get("telegram_bot_token_env") or "TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get(n.get("telegram_chat_id_env") or "TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return
    body = urllib.parse.urlencode({"chat_id": chat, "text": text[:4000]}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", body, timeout=15)
    except Exception as exc:  # noqa: BLE001 - notifications must never stop trading
        log(f"⚠️ ส่ง Telegram ไม่สำเร็จ: {exc}")


def _fetch_all(cfg: dict, source: DataSource, log) -> dict[str, pd.DataFrame]:
    data = {}
    for sym in cfg["data"]["symbols"]:
        try:
            data[sym] = source.fetch(sym)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the others
            log(f"⚠️ ดึงข้อมูล {sym} ไม่ได้: {exc}")
    return data


def status_text(engine: TradingEngine, prices: dict[str, float]) -> str:
    eq = engine.equity(prices)
    lines = [f"💼 Equity {eq:,.2f} | เงินสด {engine.broker.cash:,.2f} | DD {engine.risk.drawdown(eq):.1%} | "
             f"risk x{engine.risk.state['risk_multiplier']:.2f}"]
    if engine.risk.state.get("halted"):
        lines.append(f"⛔ HALTED: {engine.risk.state['halt_reason']}")
    for sym, r in engine.readings.items():
        lines.append(f"  {sym}: {REGIME_TH.get(r.regime, r.regime)}")
    for p in engine.positions.values():
        px = prices.get(p.symbol, p.entry_price)
        upnl = p.direction * (px - p.entry_price) * p.qty
        lines.append(f"  📌 {p.symbol} {'Long' if p.direction > 0 else 'Short'} {p.strategy} "
                     f"เข้า {p.entry_price:.4g} ตอนนี้ {px:.4g} ({upnl:+.2f}) SL {p.stop:.4g}"
                     + (" [บังคับ]" if p.forced else ""))
    return "\n".join(lines)


def run_daily(cfg: dict, log=print, force: bool = False) -> str:
    journal = Journal(cfg["storage"]["db_path"])
    try:
        engine = TradingEngine(cfg, journal, llm_reviewer=build_llm_reviewer(cfg, log))
        source = DataSource(cfg)
        data = _fetch_all(cfg, source, log)
        if not data:
            raise RuntimeError("ไม่มีข้อมูลราคาเลย — ตรวจการเชื่อมต่อหรือรายชื่อ symbols")
        latest_bar = max(str(df.index[-1]) for df in data.values())
        if journal.get_state("last_cycle_bar") == latest_bar and not force:
            return f"แท่ง {latest_bar} ประมวลผลไปแล้ว (ใช้ --force ถ้าต้องการรันซ้ำ)"
        now = _now()
        orders = engine.run_cycle(data, now)
        prices = {s: source.latest_price(s, df) for s, df in data.items()}
        engine.fill_orders(orders, prices, now, {s: str(df.index[-1]) for s, df in data.items()})
        journal.set_state("last_cycle_bar", latest_bar)
        engine.save()
        text = f"📅 รอบประจำวัน {now[:10]} (แท่งล่าสุด {latest_bar[:10]})\n" + "\n".join(engine.pop_events()) \
            + "\n" + status_text(engine, prices)
        notify(cfg, text, log)
        return text
    finally:
        journal.close()


def check_stops(cfg: dict, log=print) -> None:
    journal = Journal(cfg["storage"]["db_path"])
    try:
        engine = TradingEngine(cfg, journal, llm_reviewer=build_llm_reviewer(cfg, log), log=log)
        if not engine.positions:
            return
        source = DataSource(cfg)
        prices = {}
        for sym in list(engine.positions):
            try:
                df = source.fetch(sym, bars=5)
                prices[sym] = source.latest_price(sym, df)
            except Exception as exc:  # noqa: BLE001
                log(f"⚠️ ดึงราคา {sym} ไม่ได้: {exc}")
        engine.check_stops(prices, _now())
        engine.save()
        events = engine.pop_events()
        if events:
            notify(cfg, "\n".join(events), log)
    finally:
        journal.close()


def daemon(cfg: dict, log=print) -> None:
    """Run the daily cycle at schedule.daily_run_utc and check stops in between."""
    hh, mm = (int(x) for x in cfg["schedule"]["daily_run_utc"].split(":"))
    interval = max(1, int(cfg["schedule"]["stop_check_minutes"])) * 60
    last_day = None
    log(f"เริ่ม daemon: รันรอบประจำวันเวลา {hh:02d}:{mm:02d} UTC, เช็คสต็อปทุก {interval // 60} นาที")
    while True:
        now = datetime.now(timezone.utc)
        try:
            if (now.hour, now.minute) >= (hh, mm) and last_day != now.date():
                log(run_daily(cfg, log))
                last_day = now.date()
            else:
                check_stops(cfg, log)
        except Exception as exc:  # noqa: BLE001 - keep the daemon alive, report the problem
            log(f"⚠️ error: {exc}")
            notify(cfg, f"⚠️ autotrader error: {exc}", log)
        _time.sleep(interval)


def params_text(journal: Journal) -> str:
    return json.dumps(journal.get_state("params", {}), ensure_ascii=False, indent=2)
