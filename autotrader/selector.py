"""Strategy selector: which strategy to trust in which market regime.

Starts from a prior affinity table (trend strategies in trends, mean
reversion in ranges, ...) and learns from realised R-multiples per
(regime, strategy) pair. A pair that loses several times in a row is put on
cooldown so the bot stops repeating the same mistake in the same conditions.
"""
from __future__ import annotations

from .regime import HIGH_VOL, RANGE, SQUEEZE, TREND_DOWN, TREND_UP

DEFAULT_AFFINITY = {
    TREND_UP: {"trend": 1.0, "breakout": 0.6, "momentum": 0.5, "mean_reversion": 0.15},
    TREND_DOWN: {"trend": 1.0, "breakout": 0.6, "momentum": 0.5, "mean_reversion": 0.15},
    RANGE: {"mean_reversion": 1.0, "breakout": 0.35, "momentum": 0.2, "trend": 0.15},
    SQUEEZE: {"breakout": 1.0, "mean_reversion": 0.45, "momentum": 0.3, "trend": 0.3},
    HIGH_VOL: {"momentum": 0.7, "mean_reversion": 0.4, "trend": 0.35, "breakout": 0.25},
}


class StrategySelector:
    def __init__(self, cfg: dict, state: dict | None = None):
        self.alpha = float(cfg.get("selector_alpha", 0.2))
        self.lr = float(cfg.get("selector_learning_rate", 0.6))
        self.cooldown_after = int(cfg.get("cooldown_after_losses", 3))
        self.cooldown_bars = int(cfg.get("cooldown_bars", 10))
        self.learning = bool(cfg.get("enabled", True))
        self.stats: dict = state if state is not None else {}

    @staticmethod
    def key(regime: str, strategy: str) -> str:
        return f"{regime}|{strategy}"

    def _stat(self, regime: str, strategy: str) -> dict:
        return self.stats.setdefault(self.key(regime, strategy), {
            "n": 0, "wins": 0, "ewma_r": 0.0, "sum_r": 0.0, "loss_streak": 0, "cooldown_until": -1})

    def affinity(self, regime: str, strategy: str) -> float:
        return DEFAULT_AFFINITY.get(regime, {}).get(strategy, 0.3)

    def in_cooldown(self, regime: str, strategy: str, cycle: int) -> bool:
        return self._stat(regime, strategy)["cooldown_until"] > cycle

    def weight(self, regime: str, strategy: str, cycle: int) -> float:
        prior = self.affinity(regime, strategy)
        if not self.learning:
            return prior
        st = self._stat(regime, strategy)
        if st["cooldown_until"] > cycle:
            return 0.0
        shrink = st["n"] / (st["n"] + 5.0)  # trust learned edge more as evidence grows
        w = prior + self.lr * shrink * st["ewma_r"]
        return float(min(2.0, max(0.0, w)))

    def record(self, regime: str, strategy: str, r_multiple: float, cycle: int) -> str | None:
        st = self._stat(regime, strategy)
        st["n"] += 1
        st["sum_r"] += r_multiple
        st["ewma_r"] = r_multiple if st["n"] == 1 else (1 - self.alpha) * st["ewma_r"] + self.alpha * r_multiple
        if r_multiple > 0:
            st["wins"] += 1
            st["loss_streak"] = 0
            return None
        st["loss_streak"] += 1
        if self.learning and st["loss_streak"] >= self.cooldown_after:
            st["cooldown_until"] = cycle + self.cooldown_bars
            st["loss_streak"] = 0
            return (f"พักกลยุทธ์ {strategy} ในสภาวะ {regime} {self.cooldown_bars} รอบ "
                    f"(แพ้ติดกัน {self.cooldown_after} ครั้ง)")
        return None

    def table(self) -> list[dict]:
        rows = []
        for key, st in sorted(self.stats.items()):
            regime, strategy = key.split("|", 1)
            if st["n"] == 0:
                continue
            rows.append({"regime": regime, "strategy": strategy, "trades": st["n"],
                         "win_rate": st["wins"] / st["n"], "avg_r": st["sum_r"] / st["n"],
                         "ewma_r": st["ewma_r"], "cooldown_until": st["cooldown_until"]})
        return rows
