"""Paper/live operation: one cycle per closed bar, stop checks, and a daemon loop."""
from __future__ import annotations

import json
import os
import time as _time
import urllib.parse
import urllib.request

import pandas as pd

from .broker import PaperBroker
from .data import DataSource
from .engine import TradingEngine
from .journal import Journal
from .llm_review import build_llm_reviewer
from .mt5_connector import connect as mt5_connect
from .mt5_connector import ensure_symbol
from .regime import REGIME_TH, compute_features
from .strategies import STRATEGY_CLASSES


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
    account = getattr(engine.broker, "account_text", None)
    if account:
        lines.append(account())
    if engine.risk.state.get("halted"):
        lines.append(f"⛔ HALTED: {engine.risk.state['halt_reason']}")
    for sym, r in engine.readings.items():
        lines.append(f"  {sym}: {REGIME_TH.get(r.regime, r.regime)}")
    for p in engine.positions.values():
        px = prices.get(p.symbol, p.entry_price)
        upnl = p.direction * (px - p.entry_price) * p.qty
        size = f"{engine.broker.lots(p.qty):.2f} lot " if engine.broker.contract_size != 1 else ""
        lines.append(f"  📌 {p.symbol} {'Long' if p.direction > 0 else 'Short'} {size}{p.strategy} "
                     f"เข้า {engine.px(p.entry_price)} ตอนนี้ {engine.px(px)} ({upnl:+.2f}) SL {engine.px(p.stop)}"
                     + (f" TP {engine.px(p.take_profit)}" if p.take_profit else "")
                     + (" [บังคับ]" if p.forced else ""))
    return "\n".join(lines)


def run_cycle_once(cfg: dict, log=print, force: bool = False) -> tuple[str, bool]:
    """Process the newest closed bar if it has not been processed yet. Returns (text, ran)."""
    journal = Journal(cfg["storage"]["db_path"])
    try:
        source = DataSource(cfg)
        symbols = cfg["data"]["symbols"]
        if not force and symbols:  # cheap check before downloading full history
            try:
                probe = source.fetch(symbols[0], bars=3)
                if journal.get_state("last_cycle_bar") == str(probe.index[-1]):
                    return f"แท่ง {probe.index[-1]} ประมวลผลไปแล้ว (ใช้ --force ถ้าต้องการรันซ้ำ)", False
            except Exception:  # noqa: BLE001 - fall through to the full fetch, which reports errors
                pass
        engine = TradingEngine(cfg, journal, llm_reviewer=build_llm_reviewer(cfg, log))
        data = _fetch_all(cfg, source, log)
        if not data:
            raise RuntimeError("ไม่มีข้อมูลราคาเลย — ตรวจการเชื่อมต่อหรือรายชื่อ symbols")
        latest_bar = max(str(df.index[-1]) for df in data.values())
        if journal.get_state("last_cycle_bar") == latest_bar and not force:
            return f"แท่ง {latest_bar} ประมวลผลไปแล้ว (ใช้ --force ถ้าต้องการรันซ้ำ)", False
        now = source.now().isoformat(timespec="seconds")
        orders = engine.run_cycle(data, now)
        prices = {s: source.latest_price(s, df) for s, df in data.items()}
        engine.fill_orders(orders, prices, now, {s: str(df.index[-1]) for s, df in data.items()})
        journal.set_state("last_cycle_bar", latest_bar)
        events = engine.pop_events()
        new_day = journal.get_state("last_summary_day") != now[:10]
        if new_day:
            journal.set_state("last_summary_day", now[:10])
        engine.save()
        text = f"🕐 รอบแท่ง {latest_bar[:16]} (เวลา {now[:16]})\n" + "\n".join(events) \
            + ("\n" if events else "") + status_text(engine, prices)
        if events or new_day:  # hourly bars would otherwise flood Telegram
            notify(cfg, text, log)
        return text, True
    finally:
        journal.close()


def run_daily(cfg: dict, log=print, force: bool = False) -> str:
    return run_cycle_once(cfg, log, force)[0]


def check_stops(cfg: dict, log=print) -> None:
    journal = Journal(cfg["storage"]["db_path"])
    try:
        engine = TradingEngine(cfg, journal, llm_reviewer=build_llm_reviewer(cfg, log), log=log)
        if not engine.positions:
            return
        source = DataSource(cfg)
        prices = {}
        if not engine.broker.server_side_stops:  # MT5 keeps SL/TP on the server: only reconcile
            for sym in list(engine.positions):
                try:
                    df = source.fetch(sym, bars=5)
                    prices[sym] = source.latest_price(sym, df)
                except Exception as exc:  # noqa: BLE001
                    log(f"⚠️ ดึงราคา {sym} ไม่ได้: {exc}")
        engine.check_stops(prices, source.now().isoformat(timespec="seconds"))
        engine.save()
        events = engine.pop_events()
        if events:
            notify(cfg, "\n".join(events), log)
    finally:
        journal.close()


def daemon(cfg: dict, log=print) -> None:
    """Run a cycle whenever a new bar closes; check stops in between."""
    poll = max(5, int(cfg["schedule"].get("poll_seconds", 60)))
    stop_every = max(1, int(cfg["schedule"]["stop_check_minutes"])) * 60
    last_stop_check = 0.0
    log(f"เริ่ม daemon: timeframe {cfg['data']['timeframe']}, เช็คแท่งใหม่ทุก {poll} วินาที, "
        f"เช็คสถานะ/สต็อปทุก {stop_every // 60} นาที")
    while True:
        try:
            text, ran = run_cycle_once(cfg, log)
            if ran:
                log(text)
            elif _time.time() - last_stop_check >= stop_every:
                check_stops(cfg, log)
                last_stop_check = _time.time()
        except Exception as exc:  # noqa: BLE001 - keep the daemon alive, report the problem
            log(f"⚠️ error: {exc}")
            notify(cfg, f"⚠️ autotrader error: {exc}", log)
        _time.sleep(poll)


def sizing_text(cfg: dict) -> str:
    """Can the account trade the configured symbols at the configured risk with the broker's minimum lot?"""
    source = DataSource(cfg)
    broker = PaperBroker(cfg["broker"], {})
    capital = float(cfg["broker"]["starting_cash"])
    risk_pct = float(cfg["risk"]["risk_per_trade"])
    ceiling = float(cfg["risk"].get("max_min_lot_risk", 0.02))
    lines = [f"ทุนที่ให้บอทใช้ {capital:,.2f} | เสี่ยงต่อไม้ {risk_pct:.1%} = {capital * risk_pct:,.2f} | "
             f"ยอมใช้ lot ขั้นต่ำได้ถ้าเสี่ยงไม่เกิน {ceiling:.0%}"]
    for sym in cfg["data"]["symbols"]:
        df = source.fetch(sym)
        if source.source == "mt5":
            info = ensure_symbol(mt5_connect(cfg), sym)
            broker.contract_size, broker.volume_min, broker.volume_step = (
                float(info.trade_contract_size), float(info.volume_min), float(info.volume_step))
        atr = compute_features(df)["atr"]
        lines.append(f"\n{sym} timeframe {cfg['data']['timeframe']}: ATR {atr:,.2f} | "
                     f"1 lot = {broker.contract_size:g} หน่วย | lot ขั้นต่ำ {broker.volume_min:g}")
        for name, cls in STRATEGY_CLASSES.items():
            stop = float(cls.default_params()["stop_atr"]) * atr
            qty = broker.normalize_qty(sym, capital * risk_pct / stop)
            min_risk = broker.min_qty(sym) * stop
            if qty > 0:
                verdict = f"✅ เปิดได้ {broker.lots(qty):.2f} lot"
            elif broker.min_qty(sym) and min_risk <= ceiling * capital:
                verdict = f"⚠️ ใช้ lot ขั้นต่ำ (เสี่ยง {min_risk / capital:.1%})"
            else:
                verdict = f"❌ เล็กเกินไป ต้องมีทุนราว {min_risk / risk_pct:,.0f}"
            lines.append(f"  {name:<15} สต็อป ≈ {stop:,.2f}  lot ขั้นต่ำเสี่ยง {min_risk:,.2f}  {verdict}")
    return "\n".join(lines)


def params_text(journal: Journal) -> str:
    return json.dumps(journal.get_state("params", {}), ensure_ascii=False, indent=2)
