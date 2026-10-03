"""Post-mortem of losing trades.

Every losing trade is examined against the market data captured at entry and
exit. Each finding (``Diagnosis``) explains the cause in Thai and proposes a
bounded parameter change. A second "hindsight" review runs a few bars after
the exit to check whether the stop was just noise (price later went our way).

Not every loss is a mistake: if nothing specific is found the trade is
classified as normal variance and no parameter is touched, which keeps the
bot from overfitting to random noise.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .regime import REGIME_TH
from .selector import DEFAULT_AFFINITY


@dataclass
class Adjust:
    target: str  # strategy name | "regime" | "risk"
    param: str
    op: str  # add | set | mul
    value: float


@dataclass
class Diagnosis:
    code: str
    title: str
    detail: str
    adjustments: list[Adjust] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _side(direction: int) -> str:
    return "Long" if direction > 0 else "Short"


def compare(history: list[dict], pred, min_n: int = 4, margin: float = 0.15) -> tuple[bool, str]:
    """Do past trades matching ``pred`` really do worse than the others?

    Returns (act, evidence). With too little history the single-trade rule is
    trusted (the adapter still waits for repeated evidence before acting).
    """
    hit = [float(t["r_multiple"] or 0) for t in history if pred(t)]
    rest = [float(t["r_multiple"] or 0) for t in history if not pred(t)]
    if len(hit) < min_n or len(rest) < min_n:
        return True, f"(ข้อมูลย้อนหลังยังน้อย: {len(hit)}/{len(rest)} ไม้)"
    a, b = sum(hit) / len(hit), sum(rest) / len(rest)
    return a < b - margin, f"(ย้อนหลัง {len(hit) + len(rest)} ไม้: กลุ่มนี้เฉลี่ย {a:+.2f}R เทียบกับ {b:+.2f}R)"


class LossReviewer:
    def review(self, trade: dict, history: list[dict] | None = None) -> list[Diagnosis]:
        """``history``: recent closed non-forced trades of the same strategy."""
        t = trade
        ef = t.get("entry_features") or {}
        xf = t.get("exit_features") or {}
        p = t.get("params") or {}
        strat = t["strategy"]
        d = int(t["direction"])
        r = float(t["r_multiple"] or 0.0)
        out: list[Diagnosis] = []
        forced = bool(t.get("forced"))

        if forced:
            out.append(Diagnosis(
                "FORCED_PROBE", "ไม้บังคับเทรดรายวัน",
                "เปิดเพราะกฎ 'ต้องเทรดทุกวัน' ทั้งที่ไม่มีสัญญาณครบเงื่อนไข — ลดขนาดไม้บังคับลง",
                [Adjust("risk", "probe_risk_mult", "add", -0.05)]))

        # --- market-driven causes (apply to forced trades too) --------------
        atr_in, atr_out = float(ef.get("atr") or 0), float(xf.get("atr") or 0)
        shock_bar = float(xf.get("bar_range_entry_atr") or 0)
        if atr_in > 0 and (atr_out / atr_in >= 1.4 or shock_bar >= 2.5):
            out.append(Diagnosis(
                "VOLATILITY_SHOCK", "ความผันผวนพุ่งกะทันหัน",
                f"ATR เปลี่ยนจาก {atr_in:.4g} เป็น {atr_out:.4g} (x{atr_out / atr_in:.2f}), "
                f"แท่งวันออกกว้าง {shock_bar:.1f} เท่าของ ATR ตอนเข้า — ให้จับสภาวะผันผวนเร็วขึ้นและลดขนาดไม้",
                [Adjust("regime", "high_vol_pct", "add", -0.02), Adjust("regime", "high_vol_ratio", "add", -0.05),
                 Adjust("risk", "high_vol_size_mult", "add", -0.05)]))

        if r < -1.3:
            out.append(Diagnosis(
                "GAP_THROUGH_STOP", "ราคากระโดดข้ามจุดตัดขาดทุน",
                f"เสีย {r:.2f}R มากกว่าที่วางแผน (1R) เพราะราคาเปิดข้ามสต็อป — เป็นความเสี่ยงที่ควบคุมได้ด้วยขนาดไม้เท่านั้น",
                [Adjust("risk", "high_vol_size_mult", "add", -0.05)]))

        if not forced:
            out.extend(self._setup_causes(t, ef, p, strat, d, r, history or []))

        if not out:
            out.append(Diagnosis(
                "NORMAL_VARIANCE", "ขาดทุนตามปกติของระบบ",
                f"สัญญาณครบเงื่อนไข, เสีย {r:.2f}R ตามแผน ไม่พบความผิดพลาดเฉพาะ — "
                "ไม่ปรับพารามิเตอร์ (กัน overfit) แต่ตัวเลือกกลยุทธ์จะนับผลนี้ไว้"))
        return out

    def _setup_causes(self, t, ef, p, strat, d, r, history) -> list[Diagnosis]:
        out: list[Diagnosis] = []
        regime_in, regime_out = t.get("regime"), t.get("regime_exit")
        if regime_out and regime_in and regime_out != regime_in \
                and DEFAULT_AFFINITY.get(regime_out, {}).get(strat, 0.3) < 0.5:
            adj = ([Adjust(strat, "exit_on_regime_change", "set", 1)] if not int(p.get("exit_on_regime_change", 0))
                   else [Adjust("regime", "confirm_bars", "add", -1)])
            out.append(Diagnosis(
                "REGIME_SHIFT", "สภาวะตลาดเปลี่ยนระหว่างถือ",
                f"เข้าตอน{REGIME_TH.get(regime_in, regime_in)} แต่ออกตอน{REGIME_TH.get(regime_out, regime_out)} "
                f"ซึ่งกลยุทธ์ {strat} ไม่ถนัด — ให้ออกเร็วขึ้นเมื่อสภาวะเปลี่ยน", adj))

        htf = int(ef.get("htf_trend") or 0)
        if htf and d != htf and not int(p.get("trend_filter", 0)):
            act, ev = compare(history, lambda x: int((x.get("entry_features") or {}).get("htf_trend") or 0)
                              not in (0, int(x["direction"])))
            out.append(Diagnosis(
                "COUNTER_TREND", "เทรดสวนเทรนด์ใหญ่",
                f"เปิด {_side(d)} ขณะที่ราคาอยู่{'เหนือ' if htf > 0 else 'ใต้'} EMA200 {ev} — "
                + (f"เปิดตัวกรองเทรนด์ใหญ่ให้ {strat}" if act else "แต่สถิติไม่ได้แย่กว่า จึงยังไม่ปรับ"),
                [Adjust(strat, "trend_filter", "set", 1)] if act else []))

        ext = d * float(ef.get("extension") or 0)
        if ext > 2.0:
            act, ev = compare(history, lambda x: int(x["direction"]) * float(
                (x.get("entry_features") or {}).get("extension") or 0) > 2.0)
            out.append(Diagnosis(
                "CHASED_ENTRY", "ไล่ราคา (เข้าช้าเกินไป)",
                f"ราคาตอนเข้าห่าง EMA20 ไป {ext:.1f} ATR {ev} — "
                + ("ลดระยะสูงสุดที่ยอมไล่" if act else "แต่สถิติไม่ได้แย่กว่า จึงยังไม่ปรับ"),
                [Adjust(strat, "max_extension_atr", "add", -0.25)] if act else []))

        strength = float(t.get("strength") or 0)
        floor = float(p.get("min_strength", 0.45))
        if strength < floor + 0.1:
            act, ev = compare(history, lambda x: float(x.get("strength") or 0)
                              < float((x.get("params") or {}).get("min_strength", 0.45)) + 0.1)
            out.append(Diagnosis(
                "WEAK_SIGNAL", "สัญญาณอ่อน",
                f"ความแรงสัญญาณ {strength:.2f} เกือบต่ำสุดที่ยอมรับ ({floor:.2f}) {ev} — "
                + ("คัดสัญญาณให้เข้มขึ้น" if act else "แต่สถิติไม่ได้แย่กว่า จึงยังไม่ปรับ"),
                [Adjust(strat, "min_strength", "add", 0.05)] if act else []))

        bars = int(t.get("bars_held") or 0)
        if strat == "breakout" and t.get("exit_reason") == "stop" and bars <= 3:
            out.append(Diagnosis(
                "FALSE_BREAKOUT", "เบรกหลอก",
                f"โดนสต็อปภายใน {bars} แท่งหลังเบรก — ต้องการการยืนยันมากขึ้น (ระยะเบรก + volume)",
                [Adjust(strat, "confirm_atr", "add", 0.1), Adjust(strat, "volume_mult", "add", 0.1)]))

        mfe = float(t.get("mfe_r") or 0)
        if mfe >= 1.0 and r < 0:
            adj = [Adjust(strat, "breakeven_at_r", "add", -0.25)]
            adj.append(Adjust(strat, "trail_atr", "add", -0.25) if float(p.get("trail_atr", 0)) > 0
                       else Adjust(strat, "trail_atr", "set", 3.0))
            out.append(Diagnosis(
                "GAVE_BACK_PROFIT", "เคยกำไรแล้วปล่อยกลับมาขาดทุน",
                f"เคยกำไรสูงสุด {mfe:.2f}R แต่จบที่ {r:.2f}R — เลื่อนจุดบังทุน/trailing stop ให้เร็วขึ้น", adj))

        gross, net = float(t.get("gross_pnl") or 0), float(t.get("pnl") or 0)
        if gross > 0 >= net:
            out.append(Diagnosis(
                "COSTS", "ค่าธรรมเนียม/สลิปเพจกินกำไร",
                f"ก่อนค่าธรรมเนียมกำไร {gross:.2f} แต่สุทธิ {net:.2f} — ตั้งเป้ากำไรให้ไกลขึ้น",
                [Adjust(strat, "tp_atr", "add", 0.25)]))

        if t.get("exit_reason") == "time":
            out.append(Diagnosis(
                "STALLED", "ถือนานแต่ราคาไม่ไปไหน",
                f"ถือครบ {bars} แท่งแล้วยังไม่ถึงเป้า — ลดเวลาถือสูงสุด",
                [Adjust(strat, "max_hold", "add", -1)]))
        return out

    def hindsight(self, trade: dict, after: pd.DataFrame) -> list[Diagnosis]:
        """Called once ``after`` holds the bars following the exit."""
        if trade.get("exit_reason") != "stop" or after is None or after.empty:
            return []
        d = int(trade["direction"])
        entry = float(trade["entry_price"])
        risk = abs(entry - float(trade["stop"])) if trade.get("stop") else 0.0
        if risk <= 0:
            return []
        tp = trade.get("take_profit")
        target_dist = max(abs(float(tp) - entry) if tp else 0.0, 1.5 * risk)
        target = entry + d * target_dist
        hit = (after["high"].max() >= target) if d > 0 else (after["low"].min() <= target)
        if not hit:
            return []
        strat = trade["strategy"]
        return [Diagnosis(
            "STOP_TOO_TIGHT", "สต็อปแคบเกินไป (โดนสะบัด)",
            f"หลังโดนสต็อป {len(after)} แท่ง ราคากลับไปถึง {target:.4g} ในทิศที่เราคาดไว้ — ขยายระยะสต็อปของ {strat}",
            [Adjust(strat, "stop_atr", "add", 0.25)])]


def summarize(trade: dict, diagnoses: list[Diagnosis], notes: list[str] | None = None) -> str:
    head = (f"ไม้ #{trade['id']} {trade['symbol']} {trade['strategy']} {_side(int(trade['direction']))} "
            f"ขาดทุน {float(trade['pnl']):.2f} ({float(trade['r_multiple']):.2f}R) "
            f"ออกด้วย {trade['exit_reason']} หลังถือ {trade['bars_held']} แท่ง")
    lines = [head] + [f"  {i}. {dg.title}: {dg.detail}" for i, dg in enumerate(diagnoses, 1)]
    lines += [f"  • {n}" for n in (notes or []) if n]
    return "\n".join(lines)
