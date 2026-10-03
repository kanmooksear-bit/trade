"""Turns loss diagnoses into bounded parameter changes, and undoes changes
that did not help.

* A diagnosis must repeat ``evidence_required`` times for the same strategy
  before a parameter moves (one unlucky trade is not a pattern).
* Every change is clamped to the parameter's allowed range and step.
* Every change goes on probation: after ``probation_trades`` further trades
  the average R is compared with the baseline before the change, and the
  change is reverted if results got worse.
"""
from __future__ import annotations

from .journal import Journal
from .review import Adjust, Diagnosis


class ParamStore:
    """Uniform get/set over strategy, regime and risk parameter dicts."""

    def __init__(self):
        self._params: dict[str, dict] = {}
        self._space: dict[str, dict] = {}

    def register(self, target: str, params: dict, space: dict) -> None:
        self._params[target] = params
        self._space[target] = space

    def targets(self) -> list[str]:
        return list(self._params)

    def has(self, target: str, param: str) -> bool:
        return target in self._space and param in self._space[target]

    def get(self, target: str, param: str) -> float:
        return self._params[target][param]

    def space(self, target: str, param: str) -> tuple:
        return self._space[target][param]

    def clamp(self, target: str, param: str, value: float) -> float:
        lo, hi, step = self._space[target][param]
        value = min(hi, max(lo, value))
        if step:
            value = lo + round((value - lo) / step) * step
            value = min(hi, max(lo, value))
        if isinstance(lo, int) and isinstance(step, int):
            return int(round(value))
        return round(float(value), 6)

    def set(self, target: str, param: str, value: float) -> float:
        value = self.clamp(target, param, value)
        self._params[target][param] = value
        return value

    def snapshot(self) -> dict:
        return {t: dict(p) for t, p in self._params.items()}

    def describe(self) -> dict:
        return {t: {k: {"value": self._params[t][k], "min": s[0], "max": s[1], "step": s[2]}
                    for k, s in self._space[t].items()} for t in self._params}


class Adapter:
    def __init__(self, store: ParamStore, journal: Journal, cfg: dict, state: dict | None = None):
        self.store = store
        self.journal = journal
        self.enabled = bool(cfg.get("enabled", True))
        self.evidence_required = max(1, int(cfg.get("evidence_required", 2)))
        self.probation_trades = max(1, int(cfg.get("probation_trades", 6)))
        self.tolerance = float(cfg.get("revert_tolerance", 0.05))
        self.state = state if state is not None else {}
        self.state.setdefault("evidence", {})

    # ------------------------------------------------------------------
    def _baseline(self, target: str) -> float:
        strategy = None if target in ("regime", "risk") else target
        trades = [t for t in self.journal.closed_trades(strategy=strategy, limit=40) if not t["forced"]][:20]
        if not trades:
            return 0.0
        return sum(float(t["r_multiple"] or 0) for t in trades) / len(trades)

    def _apply(self, adj: Adjust, diagnosis: str, reason: str, time: str) -> str | None:
        if not self.store.has(adj.target, adj.param):
            return None
        old = self.store.get(adj.target, adj.param)
        if adj.op == "add":
            proposed = old + adj.value
        elif adj.op == "mul":
            proposed = old * adj.value
        else:
            proposed = adj.value
        new = self.store.clamp(adj.target, adj.param, proposed)
        if new == old:
            return None  # already at the limit
        baseline = self._baseline(adj.target)
        self.store.set(adj.target, adj.param, new)
        self.journal.add_adjustment(time=time, target=adj.target, param=adj.param, old=old, new=new,
                                    diagnosis=diagnosis, reason=reason, status="probation",
                                    baseline_r=baseline)
        return f"ปรับ {adj.target}.{adj.param}: {old} → {new} ({diagnosis})"

    def submit(self, trade: dict, diagnoses: list[Diagnosis], time: str) -> list[str]:
        if not self.enabled:
            return []
        messages: list[str] = []
        ev = self.state["evidence"]
        for dg in diagnoses:
            if not dg.adjustments:
                continue
            key = f"{trade['strategy']}|{dg.code}"
            ev[key] = ev.get(key, 0) + 1
            if ev[key] < self.evidence_required:
                messages.append(f"เก็บหลักฐาน {dg.code} ของ {trade['strategy']} ({ev[key]}/{self.evidence_required}) ยังไม่ปรับ")
                continue
            ev[key] = 0
            for adj in dg.adjustments:
                msg = self._apply(adj, dg.code, dg.title, time)
                if msg:
                    messages.append(msg)
        return messages

    def apply_suggestion(self, target: str, param: str, value: float, reason: str, time: str) -> str | None:
        """External (LLM) suggestion: moved at most two steps toward ``value``."""
        if not self.enabled or not self.store.has(target, param):
            return None
        _, _, step = self.store.space(target, param)
        old = self.store.get(target, param)
        limit = 2 * (step or abs(old) * 0.1 or 1)
        value = old + max(-limit, min(limit, value - old))
        return self._apply(Adjust(target, param, "set", value), "LLM", reason, time)

    def on_trade_closed(self, trade: dict, time: str) -> list[str]:
        if not self.enabled:
            return []
        messages: list[str] = []
        r = float(trade["r_multiple"] or 0)
        for adj in self.journal.adjustments(status="probation"):
            if adj["target"] not in (trade["strategy"], "regime", "risk"):
                continue
            n = int(adj["trades_after"]) + 1
            total = float(adj["sum_r_after"]) + r
            if n < self.probation_trades:
                self.journal.update_adjustment(adj["id"], trades_after=n, sum_r_after=total)
                continue
            after = total / n
            label = f"{adj['target']}.{adj['param']} {adj['old']}→{adj['new']}"
            if after < float(adj["baseline_r"]) - self.tolerance:
                if self.store.has(adj["target"], adj["param"]) and \
                        self.store.get(adj["target"], adj["param"]) == adj["new"]:
                    self.store.set(adj["target"], adj["param"], adj["old"])
                status = "reverted"
                messages.append(f"ยกเลิกการปรับ {label}: หลังปรับเฉลี่ย {after:.2f}R แย่กว่าก่อนปรับ "
                                f"{float(adj['baseline_r']):.2f}R")
            else:
                status = "kept"
                messages.append(f"คงการปรับ {label}: หลังปรับเฉลี่ย {after:.2f}R "
                                f"(ก่อนปรับ {float(adj['baseline_r']):.2f}R)")
            self.journal.update_adjustment(adj["id"], trades_after=n, sum_r_after=total, status=status,
                                           resolved_time=time)
        return messages
