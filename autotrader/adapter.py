"""Turns loss diagnoses into bounded parameter changes, and undoes changes
that did not help.

* A diagnosis must repeat ``evidence_required`` times for the same strategy
  before a change is even considered (one unlucky trade is not a pattern).
* The change is then replayed on recent history (``whatif.py``) and adopted
  only if it would clearly have done better.
* Every change is clamped to the parameter's allowed range and step.
* Every change goes on probation: after ``probation_trades`` further trades
  the average R is compared with the baseline before the change, and the
  change is reverted if results got clearly worse.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from .journal import Journal
from .review import Adjust, Diagnosis

# validator(strategy, {param: new_value}) -> (adopt, explanation)
Validator = Callable[[str, dict], tuple]


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
        self.probation_trades = max(1, int(cfg.get("probation_trades", 10)))
        self.tolerance = float(cfg.get("revert_tolerance", 0.2))
        self.whatif = bool(cfg.get("whatif", True))
        self.state = state if state is not None else {}
        self.state.setdefault("evidence", {})
        self.state.setdefault("pending", [])

    # ------------------------------------------------------------------
    def _baseline(self, target: str) -> float:
        strategy = None if target in ("regime", "risk") else target
        trades = [t for t in self.journal.closed_trades(strategy=strategy, limit=40) if not t["forced"]][:20]
        if not trades:
            return 0.0
        return sum(float(t["r_multiple"] or 0) for t in trades) / len(trades)

    def _proposed(self, adj: Adjust) -> tuple[float, float] | None:
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
        return None if new == old else (old, new)

    def _apply(self, adj: Adjust, diagnosis: str, reason: str, time: str) -> str | None:
        change = self._proposed(adj)
        if change is None:
            return None  # unknown parameter or already at its limit
        old, new = change
        baseline = self._baseline(adj.target)
        self.store.set(adj.target, adj.param, new)
        self.journal.add_adjustment(time=time, target=adj.target, param=adj.param, old=old, new=new,
                                    diagnosis=diagnosis, reason=reason, status="probation",
                                    baseline_r=baseline)
        return f"ปรับ {adj.target}.{adj.param}: {old} → {new} ({diagnosis})"

    def _validate_and_apply(self, strategy: str, code: str, title: str, adjustments: list[Adjust], time: str,
                            validator: Validator | None) -> list[str]:
        changes = {}
        for adj in adjustments:
            proposed = self._proposed(adj)
            if proposed is not None:
                changes[(adj.target, adj.param)] = proposed[1]
        if not changes:
            return []
        reason = title
        strategy_changes = {param: v for (target, param), v in changes.items() if target == strategy}
        if self.whatif and validator is not None and strategy_changes:
            adopt, text = validator(strategy, strategy_changes)
            if not adopt:
                return [f"ไม่ปรับตาม {code}: {text}"]
            reason = f"{title} — {text}"
        out = []
        for adj in adjustments:
            msg = self._apply(adj, code, reason, time)
            if msg:
                out.append(msg + (f" [{reason.split(' — ', 1)[1]}]" if " — " in reason else ""))
        return out

    def submit(self, trade: dict, diagnoses: list[Diagnosis], time: str,
               validator: Validator | None = None) -> list[str]:
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
            needs_data = self.whatif and any(a.target == trade["strategy"] for a in dg.adjustments)
            if needs_data and validator is None:  # no market data in this call: test it in the next cycle
                self.state["pending"].append({"strategy": trade["strategy"], "code": dg.code, "title": dg.title,
                                              "adjustments": [asdict(a) for a in dg.adjustments]})
                messages.append(f"{dg.code}: จะทดสอบย้อนหลังก่อนปรับในรอบถัดไป")
                continue
            messages.extend(self._validate_and_apply(trade["strategy"], dg.code, dg.title, dg.adjustments, time,
                                                     validator))
        return messages

    def process_pending(self, validator: Validator, time: str) -> list[str]:
        pending, self.state["pending"] = self.state["pending"], []
        out: list[str] = []
        for item in pending:
            adjustments = [Adjust(**a) for a in item["adjustments"]]
            out.extend(self._validate_and_apply(item["strategy"], item["code"], item["title"], adjustments, time,
                                                validator))
        return out

    def apply_suggestion(self, target: str, param: str, value: float, reason: str, time: str,
                         validator: Validator | None = None) -> str | None:
        """External (LLM) suggestion: moved at most two steps toward ``value``, then what-if tested."""
        if not self.enabled or not self.store.has(target, param):
            return None
        _, _, step = self.store.space(target, param)
        old = self.store.get(target, param)
        limit = 2 * (step or abs(old) * 0.1 or 1)
        value = old + max(-limit, min(limit, value - old))
        msgs = self._validate_and_apply(target, "LLM", reason, [Adjust(target, param, "set", value)], time,
                                        validator)
        return "; ".join(msgs) or None

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
