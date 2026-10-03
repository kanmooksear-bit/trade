import pytest

from autotrader.risk import RISK_PARAM_DEFAULTS, RiskManager
from autotrader.selector import StrategySelector


def make_risk(cfg):
    return RiskManager(cfg["risk"], dict(RISK_PARAM_DEFAULTS))


def test_position_size_risks_one_percent(cfg):
    risk = make_risk(cfg)
    qty, _ = risk.size(10_000, 100.0, 5.0, high_vol=False, forced=False, score=1.0, gross_exposure=0,
                       min_notional=10)
    assert qty * 5.0 == pytest.approx(100.0)  # 1% of equity lost at the stop


def test_size_caps_and_multipliers(cfg):
    risk = make_risk(cfg)
    capped, _ = risk.size(10_000, 100.0, 0.01, high_vol=False, forced=False, score=1.0, gross_exposure=0,
                          min_notional=10)
    assert capped * 100.0 == pytest.approx(10_000 * cfg["risk"]["max_position_pct"])
    normal, _ = risk.size(10_000, 100.0, 5.0, high_vol=False, forced=False, score=1.0, gross_exposure=0,
                          min_notional=10)
    probe, _ = risk.size(10_000, 100.0, 5.0, high_vol=True, forced=True, score=1.0, gross_exposure=0,
                         min_notional=10)
    assert probe == pytest.approx(normal * 0.5 * 0.3)
    full, why = risk.size(10_000, 100.0, 5.0, high_vol=False, forced=False, score=1.0, gross_exposure=10_000,
                          min_notional=10)
    assert full == 0 and why


def test_daily_loss_limit_and_drawdown_kill_switch(cfg):
    risk = make_risk(cfg)
    risk.start_day("2024-01-01", 10_000)
    risk.update_equity(10_000)
    assert risk.can_open(9_800)[0]
    assert not risk.can_open(9_600)[0]  # -4% today
    risk.start_day("2024-01-02", 9_600)
    assert risk.can_open(9_600)[0]
    risk.update_equity(7_400)  # -26% from peak
    ok, why = risk.can_open(7_400)
    assert not ok and "Drawdown" in why
    risk.reset_halt(7_400)
    assert not risk.can_open(7_400)[0]  # still blocked by today's loss limit
    risk.start_day("2024-01-03", 7_400)
    assert risk.can_open(7_400)[0]


def test_losing_streak_halves_risk_and_wins_restore(cfg):
    risk = make_risk(cfg)
    for _ in range(3):
        risk.on_trade_closed(-1.0)
    assert risk.state["risk_multiplier"] == 0.5
    risk.on_trade_closed(-1.0)
    risk.on_trade_closed(-1.0)
    assert risk.state["risk_multiplier"] == cfg["risk"]["min_risk_multiplier"]
    risk.on_trade_closed(2.0)
    assert risk.state["risk_multiplier"] == 0.5


def test_selector_learns_and_cools_down(cfg):
    sel = StrategySelector(cfg["learning"])
    prior = sel.weight("range", "mean_reversion", 0)
    for _ in range(4):
        sel.record("range", "mean_reversion", 1.5, 0)
    assert sel.weight("range", "mean_reversion", 0) > prior
    msg = None
    for _ in range(3):
        msg = sel.record("trend_up", "breakout", -1.0, cycle=10)
    assert msg and sel.weight("trend_up", "breakout", 11) == 0.0
    assert sel.weight("trend_up", "breakout", 10 + cfg["learning"]["cooldown_bars"]) > 0


def test_selector_without_learning_uses_prior_only(cfg):
    sel = StrategySelector({**cfg["learning"], "enabled": False})
    for _ in range(5):
        sel.record("range", "mean_reversion", -1.0, 0)
    assert sel.weight("range", "mean_reversion", 1) == sel.affinity("range", "mean_reversion")
