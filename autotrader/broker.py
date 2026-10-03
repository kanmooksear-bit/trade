"""Order execution: a paper broker (default, also used by the backtester) and
a ccxt broker for real exchanges.

Accounting is margin-style for both long and short: cash only changes by
realised P&L and fees, equity = cash + unrealised P&L of open positions.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Fill:
    price: float
    qty: float
    fee: float


class PaperBroker:
    def __init__(self, cfg: dict, state: dict | None = None):
        self.fee_rate = float(cfg.get("fee_rate", 0.001))
        self.slippage = float(cfg.get("slippage_bps", 5.0)) / 10_000.0
        self.state = state if state is not None else {}
        self.state.setdefault("cash", float(cfg.get("starting_cash", 10_000.0)))

    @property
    def cash(self) -> float:
        return float(self.state["cash"])

    def execute(self, symbol: str, side: int, qty: float, ref_price: float) -> Fill:
        price = ref_price * (1.0 + side * self.slippage)
        fee = abs(price * qty) * self.fee_rate
        self.state["cash"] = self.cash - fee
        return Fill(price, qty, fee)

    def realize(self, pnl: float) -> None:
        self.state["cash"] = self.cash + pnl

    def equity(self, unrealized: float) -> float:
        return self.cash + unrealized


class CCXTBroker(PaperBroker):
    """Market orders on a real exchange through ccxt (spot by default).

    The bot keeps its own ledger of the capital you allocate to it
    (``broker.starting_cash``), so it never sizes trades from the whole account.
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

    def execute(self, symbol: str, side: int, qty: float, ref_price: float) -> Fill:
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


def build_broker(cfg: dict, state: dict | None):
    kind = cfg["broker"].get("type", "paper")
    if kind == "paper":
        return PaperBroker(cfg["broker"], state)
    if kind == "ccxt":
        if cfg.get("mode") != "live":
            raise RuntimeError("broker.type=ccxt ต้องตั้ง mode: live อย่างชัดเจน")
        return CCXTBroker(cfg["broker"], state)
    raise ValueError(f"unknown broker type: {kind}")
