"""Optional AI post-mortem of losing trades with Claude.

The rule-based reviewer always runs; this adds a written analysis (in Thai)
and, if ``llm_review.apply_suggestions`` is on, parameter suggestions that the
adapter still clamps to the allowed ranges and to at most two steps per change.
Needs ``pip install anthropic`` and an ANTHROPIC_API_KEY (or ``ant auth login``).
"""
from __future__ import annotations

import json

SYSTEM = (
    "You are a trading-risk analyst reviewing one losing trade from an automated, regime-switching "
    "trading bot. Explain the most likely root cause using only the data provided, separating avoidable "
    "mistakes from normal variance (a valid setup that simply lost). Suggest parameter changes only when "
    "the evidence supports them; an empty list is a good answer for normal variance. Every suggestion must "
    "use a target and param that appear in the parameter table and stay within its min/max. "
    "Write 'analysis' in Thai, at most four sentences."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause": {"type": "string"},
        "avoidable": {"type": "boolean"},
        "analysis": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "param": {"type": "string"},
                    "value": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["target", "param", "value", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["root_cause", "avoidable", "analysis", "suggestions"],
    "additionalProperties": False,
}

TRADE_FIELDS = ("id", "symbol", "strategy", "regime", "regime_exit", "direction", "entry_price", "exit_price",
                "stop", "take_profit", "exit_reason", "bars_held", "pnl", "r_multiple", "mfe_r", "mae_r",
                "strength", "score", "forced", "signal_reason", "entry_features", "exit_features")


class ClaudeReviewer:
    def __init__(self, cfg: dict):
        import anthropic  # optional dependency

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = cfg.get("model", "claude-opus-5-5")
        self.effort = cfg.get("effort", "medium")

    def review(self, trade: dict, diagnoses: list[dict], params: dict) -> dict | None:
        strategy = trade["strategy"]
        relevant = {k: v for k, v in params.items() if k in (strategy, "regime", "risk")}
        payload = {
            "trade": {k: trade.get(k) for k in TRADE_FIELDS},
            "rule_based_diagnoses": [{k: d[k] for k in ("code", "title", "detail")} for d in diagnoses],
            "parameter_table": relevant,
        }
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=float)}],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            return None
        out = json.loads(text)
        out["suggestions"] = [s for s in out.get("suggestions", [])
                              if s.get("param") in relevant.get(s.get("target"), {})]
        return out


def build_llm_reviewer(cfg: dict, log=print):
    if not cfg["llm_review"].get("enabled"):
        return None
    try:
        return ClaudeReviewer(cfg["llm_review"])
    except Exception as exc:  # noqa: BLE001 - missing package/credentials: keep trading rule-based
        log(f"⚠️ ปิด AI review: {exc}")
        return None
