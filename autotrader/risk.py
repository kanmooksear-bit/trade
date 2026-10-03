"""Risk management: position sizing, daily loss limit, drawdown kill switch,
and automatic de-risking during losing streaks."""
from __future__ import annotations

RISK_PARAM_DEFAULTS = {
    "high_vol_size_mult": 0.5,
    "probe_risk_mult": 0.3,
}

RISK_PARAM_SPACE = {
    "high_vol_size_mult": (0.2, 1.0, 0.05),
    "probe_risk_mult": (0.1, 0.6, 0.05),
}


class RiskManager:
    def __init__(self, cfg: dict, params: dict, state: dict | None = None):
        self.cfg = cfg
        self.params = params
        self.state = state if state is not None else {}
        self.state.setdefault("peak_equity", None)
        self.state.setdefault("day", None)
        self.state.setdefault("day_start_equity", None)
        self.state.setdefault("halted", False)
        self.state.setdefault("halt_reason", "")
        self.state.setdefault("consecutive_losses", 0)
        self.state.setdefault("risk_multiplier", 1.0)

    # ----- equity tracking -------------------------------------------------
    def start_day(self, day: str, equity: float) -> None:
        if self.state["day"] != day:
            self.state["day"] = day
            self.state["day_start_equity"] = equity

    def update_equity(self, equity: float) -> None:
        peak = self.state["peak_equity"]
        if peak is None or equity > peak:
            self.state["peak_equity"] = peak = equity
        dd = 1.0 - equity / peak if peak else 0.0
        if dd >= float(self.cfg["max_drawdown"]) and not self.state["halted"]:
            self.state["halted"] = True
            self.state["halt_reason"] = (f"Drawdown {dd:.1%} เกินเพดาน {float(self.cfg['max_drawdown']):.0%} "
                                         "— หยุดเปิดไม้ใหม่จนกว่าจะสั่ง reset-halt")

    def drawdown(self, equity: float) -> float:
        peak = self.state["peak_equity"] or equity
        return 1.0 - equity / peak if peak else 0.0

    def reset_halt(self, equity: float) -> None:
        self.state["halted"] = False
        self.state["halt_reason"] = ""
        self.state["peak_equity"] = equity

    def can_open(self, equity: float) -> tuple[bool, str]:
        if self.state["halted"]:
            return False, self.state["halt_reason"]
        start = self.state["day_start_equity"]
        if start:
            day_loss = 1.0 - equity / start
            if day_loss >= float(self.cfg["daily_loss_limit"]):
                return False, f"ขาดทุนวันนี้ {day_loss:.1%} ถึงลิมิตรายวัน {float(self.cfg['daily_loss_limit']):.0%}"
        return True, ""

    # ----- streak handling -------------------------------------------------
    def on_trade_closed(self, r_multiple: float) -> str | None:
        floor = float(self.cfg["min_risk_multiplier"])
        if r_multiple > 0:
            self.state["consecutive_losses"] = 0
            before = self.state["risk_multiplier"]
            self.state["risk_multiplier"] = min(1.0, before + 0.25)
            if before < 1.0:
                return f"ชนะแล้ว เพิ่มขนาดความเสี่ยงกลับเป็น x{self.state['risk_multiplier']:.2f}"
            return None
        self.state["consecutive_losses"] += 1
        n = self.state["consecutive_losses"]
        if n >= int(self.cfg["losing_streak_reduce_after"]):
            before = self.state["risk_multiplier"]
            self.state["risk_multiplier"] = max(floor, before * 0.5)
            if self.state["risk_multiplier"] < before:
                return f"แพ้ติดกัน {n} ไม้ ลดขนาดความเสี่ยงเหลือ x{self.state['risk_multiplier']:.2f}"
        return None

    # ----- sizing ----------------------------------------------------------
    def size(self, equity: float, price: float, stop_distance: float, *, high_vol: bool, forced: bool,
             score: float, gross_exposure: float, min_notional: float) -> tuple[float, str]:
        if stop_distance <= 0 or price <= 0 or equity <= 0:
            return 0.0, "ข้อมูลราคาไม่ถูกต้อง"
        mult = self.state["risk_multiplier"] * (0.6 + 0.4 * min(1.0, max(0.0, score)))
        if high_vol:
            mult *= float(self.params["high_vol_size_mult"])
        if forced:
            mult *= float(self.params["probe_risk_mult"])
        risk_amount = equity * float(self.cfg["risk_per_trade"]) * mult
        qty = risk_amount / stop_distance
        qty = min(qty, equity * float(self.cfg["max_position_pct"]) / price)
        room = equity * float(self.cfg["max_gross_exposure"]) - gross_exposure
        if room <= 0:
            return 0.0, "ใช้ exposure เต็มเพดานแล้ว"
        qty = min(qty, room / price)
        if qty * price < min_notional:
            return 0.0, f"มูลค่าไม้ {qty * price:.2f} ต่ำกว่าขั้นต่ำ {min_notional}"
        return qty, ""
