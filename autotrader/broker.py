"""Order execution.

* ``PaperBroker`` - simulated fills (also used by the backtester). Models
  percentage fees, slippage, a fixed spread, commission per lot and lot sizes,
  so it can mimic both crypto exchanges and CFD brokers such as HFM.
* ``CCXTBroker`` - market orders on crypto exchanges through ccxt.
* ``MT5Broker`` - market orders through a MetaTrader 5 terminal with stop-loss
  and take-profit placed on the broker's server.

Quantities are always in *units* of the asset (ounces for XAUUSD); brokers
that trade in lots convert with ``contract_size`` (100 oz per lot for gold).
Accounting is margin-style: cash changes only by realised P&L and costs,
equity = cash + unrealised P&L. ``starting_cash`` is the capital allocated to
the bot, so it never sizes from the whole account.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pandas as pd

from .mt5_connector import connect as mt5_connect
from .mt5_connector import ensure_symbol


@dataclass
class Fill:
    price: float
    qty: float
    fee: float
    ticket: int | None = None
    sl: float | None = None
    tp: float | None = None


@dataclass
class ExternalClose:
    """A position the broker closed by itself (server-side SL/TP, stop-out, manual close)."""
    trade_id: int
    price: float
    time: str
    reason: str
    fee: float


class PaperBroker:
    server_side_stops = False

    def __init__(self, cfg: dict, state: dict | None = None):
        self.fee_rate = float(cfg.get("fee_rate", 0.001))
        self.slippage = float(cfg.get("slippage_bps", 5.0)) / 10_000.0
        self.spread = float(cfg.get("spread", 0.0))
        self.commission_per_lot = float(cfg.get("commission_per_lot", 0.0))
        self.contract_size = float(cfg.get("contract_size", 1.0))
        self.volume_min = float(cfg.get("volume_min", 0.0))
        self.volume_step = float(cfg.get("volume_step", 0.0))
        self.digits = cfg.get("digits")
        self.state = state if state is not None else {}
        self.state.setdefault("cash", float(cfg.get("starting_cash", 10_000.0)))

    @property
    def cash(self) -> float:
        return float(self.state["cash"])

    def round_price(self, price: float) -> float:
        return round(price, int(self.digits)) if self.digits is not None else price

    # ---- sizing helpers -------------------------------------------------
    def normalize_qty(self, symbol: str, qty: float) -> float:
        """Round down to a tradable size; 0 if below the minimum lot."""
        if self.volume_step <= 0:
            return qty
        lots = math.floor(qty / self.contract_size / self.volume_step + 1e-9) * self.volume_step
        if lots + 1e-12 < self.volume_min:
            return 0.0
        return round(lots, 8) * self.contract_size

    def min_qty(self, symbol: str) -> float:
        return self.volume_min * self.contract_size

    def lots(self, qty: float) -> float:
        return qty / self.contract_size

    # ---- execution ------------------------------------------------------
    def _fill(self, side: int, qty: float, ref_price: float) -> Fill:
        price = self.round_price(ref_price * (1.0 + side * self.slippage) + side * self.spread / 2.0)
        fee = abs(price * qty) * self.fee_rate + self.commission_per_lot * abs(qty) / self.contract_size
        self.state["cash"] = self.cash - fee
        return Fill(price, qty, fee)

    def open(self, symbol: str, direction: int, qty: float, ref_price: float, stop_dist: float,
             tp_dist: float | None, comment: str = "") -> Fill:
        fill = self._fill(direction, qty, ref_price)
        fill.sl = self.round_price(fill.price - direction * stop_dist)
        fill.tp = self.round_price(fill.price + direction * tp_dist) if tp_dist else None
        return fill

    def close(self, symbol: str, direction: int, qty: float, ref_price: float, ticket: int | None) -> Fill:
        return self._fill(-direction, qty, ref_price)

    def modify(self, symbol: str, ticket: int | None, sl: float, tp: float | None) -> bool:
        return True

    def reconcile(self, positions: list) -> list[ExternalClose]:
        return []

    def realize(self, pnl: float) -> None:
        self.state["cash"] = self.cash + pnl

    def equity(self, unrealized: float) -> float:
        return self.cash + unrealized


class CCXTBroker(PaperBroker):
    """Market orders on a crypto exchange through ccxt (spot by default).

    Use ``sandbox: true`` (exchange testnet) until you trust the bot.
    """

    def __init__(self, cfg: dict, state: dict | None = None):
        super().__init__(cfg, state)
        import ccxt  # optional dependency

        key = os.environ.get(cfg.get("api_key_env", "EXCHANGE_API_KEY"), "")
        secret = os.environ.get(cfg.get("api_secret_env", "EXCHANGE_API_SECRET"), "")
        if not key or not secret:
            raise RuntimeError("ตั้ง environment variable ของ API key/secret ก่อนใช้โหมด live")
        exchange_cls = getattr(ccxt, cfg.get("exchange", "binance"))
        self.exchange = exchange_cls({"apiKey": key, "secret": secret, "enableRateLimit": True})
        if cfg.get("sandbox", True):
            self.exchange.set_sandbox_mode(True)
        self.exchange.load_markets()
        self.allow_short = bool(cfg.get("allow_short", False))

    def _execute(self, symbol: str, side: int, qty: float, ref_price: float) -> Fill:
        if side < 0 and not self.allow_short:
            # spot sell: never sell more than we hold (fees may have been taken in the base asset)
            base = self.exchange.market(symbol)["base"]
            free = float(self.exchange.fetch_balance().get("free", {}).get(base, 0.0) or 0.0)
            qty = min(qty, free)
        amount = float(self.exchange.amount_to_precision(symbol, qty))
        order = self.exchange.create_order(symbol, "market", "buy" if side > 0 else "sell", amount)
        price = float(order.get("average") or order.get("price") or ref_price)
        filled = float(order.get("filled") or amount)
        quote = self.exchange.market(symbol)["quote"]
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        if fee and fee_info.get("currency") not in (None, quote):
            fee *= price  # fee charged in the base asset
        if not fee:
            fee = abs(price * filled) * self.fee_rate
        self.state["cash"] = self.cash - fee
        return Fill(price, filled, fee)

    def open(self, symbol, direction, qty, ref_price, stop_dist, tp_dist, comment="") -> Fill:
        fill = self._execute(symbol, direction, qty, ref_price)
        fill.sl = fill.price - direction * stop_dist
        fill.tp = fill.price + direction * tp_dist if tp_dist else None
        return fill

    def close(self, symbol, direction, qty, ref_price, ticket) -> Fill:
        return self._execute(symbol, -direction, qty, ref_price)


class MT5Broker(PaperBroker):
    """MetaTrader 5 (e.g. HFM). Stop-loss / take-profit live on the broker's
    server, so they still trigger if this program or the PC is offline."""

    server_side_stops = True

    def __init__(self, cfg: dict, state: dict | None = None, full_cfg: dict | None = None):
        super().__init__(cfg, state)
        self.mt5 = mt5_connect(full_cfg or {})
        self.magic = int(cfg.get("magic", 26100301))
        self.deviation = int(cfg.get("deviation_points", 30))
        terminal = self.mt5.terminal_info()
        if terminal is not None and not getattr(terminal, "trade_allowed", True):
            raise RuntimeError("MT5 ยังไม่เปิด Algo Trading — กดปุ่ม 'Algo Trading' ในโปรแกรม MT5 ก่อน")

    # ---- symbol specifics -------------------------------------------------
    def _info(self, symbol: str):
        return ensure_symbol(self.mt5, symbol)

    def normalize_qty(self, symbol: str, qty: float) -> float:
        info = self._info(symbol)
        step = info.volume_step or 0.01
        lots = math.floor(qty / info.trade_contract_size / step + 1e-9) * step
        lots = min(lots, info.volume_max)
        if lots + 1e-12 < info.volume_min:
            return 0.0
        return round(lots, 8) * info.trade_contract_size

    def min_qty(self, symbol: str) -> float:
        info = self._info(symbol)
        return info.volume_min * info.trade_contract_size

    def _lots(self, symbol: str, qty: float) -> float:
        info = self._info(symbol)
        step = info.volume_step or 0.01
        decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
        return round(qty / info.trade_contract_size, decimals)

    def _filling(self, info) -> int:
        mode = int(getattr(info, "filling_mode", 0))
        if mode & 1:  # SYMBOL_FILLING_FOK
            return self.mt5.ORDER_FILLING_FOK
        if mode & 2:  # SYMBOL_FILLING_IOC
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request: dict):
        result = self.mt5.order_send(request)
        if result is None:
            raise RuntimeError(f"order_send ล้มเหลว: {self.mt5.last_error()}")
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"MT5 ปฏิเสธคำสั่ง (retcode {result.retcode}): {result.comment}")
        return result

    def _deal_costs(self, deal_ticket: int) -> float | None:
        deals = self.mt5.history_deals_get(ticket=deal_ticket) if deal_ticket else None
        if not deals:
            return None
        d = deals[0]
        return -(float(d.commission) + float(getattr(d, "fee", 0.0)))

    # ---- execution --------------------------------------------------------
    def open(self, symbol, direction, qty, ref_price, stop_dist, tp_dist, comment="") -> Fill:
        info = self._info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        price = float(tick.ask if direction > 0 else tick.bid)
        min_dist = float(info.trade_stops_level) * float(info.point)
        stop_dist = max(stop_dist, min_dist)
        sl = round(price - direction * stop_dist, info.digits)
        tp = round(price + direction * max(tp_dist, min_dist), info.digits) if tp_dist else 0.0
        lots = self._lots(symbol, qty)
        result = self._send({
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY if direction > 0 else self.mt5.ORDER_TYPE_SELL,
            "price": price, "sl": sl, "tp": tp, "deviation": self.deviation, "magic": self.magic,
            "comment": (comment or "autotrader")[:31], "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        })
        fill_price = float(result.price or price)
        filled = float(result.volume or lots) * info.trade_contract_size
        fee = self._deal_costs(result.deal)
        if fee is None:
            fee = self.commission_per_lot * filled / info.trade_contract_size
        self.state["cash"] = self.cash - fee
        # in MT5 the position ticket equals the ticket of the order that opened it
        return Fill(fill_price, filled, fee, ticket=int(result.order), sl=sl, tp=tp or None)

    def close(self, symbol, direction, qty, ref_price, ticket) -> Fill:
        info = self._info(symbol)
        positions = self.mt5.positions_get(ticket=ticket) if ticket else None
        if ticket and not positions:
            raise RuntimeError(f"ไม่พบสถานะ #{ticket} ใน MT5 (อาจถูกปิดไปแล้ว)")
        volume = float(positions[0].volume) if positions else self._lots(symbol, qty)
        tick = self.mt5.symbol_info_tick(symbol)
        price = float(tick.bid if direction > 0 else tick.ask)
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
            "type": self.mt5.ORDER_TYPE_SELL if direction > 0 else self.mt5.ORDER_TYPE_BUY,
            "price": price, "deviation": self.deviation, "magic": self.magic, "comment": "autotrader close",
            "type_time": self.mt5.ORDER_TIME_GTC, "type_filling": self._filling(info),
        }
        if ticket:
            request["position"] = int(ticket)
        result = self._send(request)
        deals = self.mt5.history_deals_get(ticket=result.deal) if result.deal else None
        fee = -(float(deals[0].commission) + float(getattr(deals[0], "fee", 0.0)) + float(deals[0].swap)) \
            if deals else self.commission_per_lot * volume
        self.state["cash"] = self.cash - fee
        return Fill(float(result.price or price), volume * info.trade_contract_size, fee, ticket=ticket)

    def modify(self, symbol: str, ticket: int | None, sl: float, tp: float | None) -> bool:
        if not ticket:
            return False
        info = self._info(symbol)
        try:
            self._send({"action": self.mt5.TRADE_ACTION_SLTP, "symbol": symbol, "position": int(ticket),
                        "sl": round(sl, info.digits), "tp": round(tp, info.digits) if tp else 0.0})
            return True
        except RuntimeError:
            return False

    def reconcile(self, positions: list) -> list[ExternalClose]:
        """Find bot positions that MT5 closed on its own and how they closed."""
        tracked = [p for p in positions if p.ticket]
        if not tracked:
            return []
        live = self.mt5.positions_get()
        if live is None:  # connection problem: do not assume anything was closed
            return []
        open_tickets = {int(p.ticket) for p in live}
        closed: list[ExternalClose] = []
        for pos in tracked:
            if int(pos.ticket) in open_tickets:
                continue
            deals = self.mt5.history_deals_get(position=int(pos.ticket)) or ()
            mt5 = self.mt5
            out_entries = (getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_OUT_BY", 3))
            outs = [d for d in deals if d.entry in out_entries]
            if not outs:
                continue
            last = outs[-1]
            vol = sum(float(d.volume) for d in outs) or 1.0
            price = sum(float(d.price) * float(d.volume) for d in outs) / vol
            fee = -sum(float(d.commission) + float(getattr(d, "fee", 0.0)) + float(d.swap) for d in outs)
            reason = {getattr(mt5, "DEAL_REASON_SL", 4): "trail" if pos.stop_moved else "stop",
                      getattr(mt5, "DEAL_REASON_TP", 5): "target",
                      getattr(mt5, "DEAL_REASON_SO", 6): "stop_out"}.get(last.reason, "manual")
            when = str(pd.Timestamp(int(last.time), unit="s", tz="UTC"))  # broker server clock
            self.state["cash"] = self.cash - fee
            closed.append(ExternalClose(pos.trade_id, price, when, reason, fee))
        return closed

    def account_text(self) -> str:
        acc = self.mt5.account_info()
        if acc is None:
            return ""
        return (f"บัญชี MT5 #{acc.login}: balance {acc.balance:,.2f} equity {acc.equity:,.2f} "
                f"free margin {acc.margin_free:,.2f} {acc.currency}")


def build_broker(cfg: dict, state: dict | None):
    kind = cfg["broker"].get("type", "paper")
    if kind == "paper":
        return PaperBroker(cfg["broker"], state)
    if kind in ("ccxt", "mt5"):
        if cfg.get("mode") != "live":
            raise RuntimeError(f"broker.type={kind} ส่งคำสั่งจริง ต้องตั้ง mode: live อย่างชัดเจน")
        if kind == "ccxt":
            return CCXTBroker(cfg["broker"], state)
        return MT5Broker(cfg["broker"], state, cfg)
    raise ValueError(f"unknown broker type: {kind}")
