import pandas as pd
import pytest

from autotrader.adapter import Adapter, ParamStore
from autotrader.journal import Journal
from autotrader.review import Adjust, Diagnosis, LossReviewer, summarize
from autotrader.strategies import STRATEGY_CLASSES


def base_trade(**kw):
    t = {
        "id": 1, "symbol": "BTC/USDT", "strategy": "trend", "regime": "trend_up", "regime_exit": "trend_up",
        "direction": 1, "entry_price": 100.0, "exit_price": 95.0, "stop": 95.0, "take_profit": 110.0,
        "exit_reason": "stop", "bars_held": 5, "gross_pnl": -50.0, "pnl": -51.0, "r_multiple": -1.0,
        "mfe_r": 0.3, "mae_r": -1.0, "strength": 0.9, "forced": 0,
        "entry_features": {"atr": 2.0, "htf_trend": 1, "extension": 0.5},
        "exit_features": {"atr": 2.1, "bar_range_entry_atr": 1.2},
        "params": dict(STRATEGY_CLASSES["trend"].default_params()),
    }
    t.update(kw)
    return t


def codes(diags):
    return {d.code for d in diags}


def test_clean_loss_is_normal_variance():
    assert codes(LossReviewer().review(base_trade())) == {"NORMAL_VARIANCE"}


@pytest.mark.parametrize("changes, code", [
    ({"regime_exit": "range"}, "REGIME_SHIFT"),
    ({"exit_features": {"atr": 3.5, "bar_range_entry_atr": 1.0}}, "VOLATILITY_SHOCK"),
    ({"entry_features": {"atr": 2.0, "htf_trend": -1, "extension": 0.5}}, "COUNTER_TREND"),
    ({"entry_features": {"atr": 2.0, "htf_trend": 1, "extension": 2.6}}, "CHASED_ENTRY"),
    ({"strength": 0.5}, "WEAK_SIGNAL"),
    ({"mfe_r": 1.4, "r_multiple": -0.3}, "GAVE_BACK_PROFIT"),
    ({"gross_pnl": 2.0, "pnl": -1.0}, "COSTS"),
    ({"exit_reason": "time", "r_multiple": -0.2}, "STALLED"),
    ({"r_multiple": -1.8}, "GAP_THROUGH_STOP"),
    ({"strategy": "breakout", "bars_held": 2,
      "params": dict(STRATEGY_CLASSES["breakout"].default_params())}, "FALSE_BREAKOUT"),
])
def test_specific_causes_detected(changes, code):
    diags = LossReviewer().review(base_trade(**changes))
    assert code in codes(diags)
    assert "NORMAL_VARIANCE" not in codes(diags)
    assert all(d.adjustments for d in diags if d.code == code)


def test_forced_trade_does_not_tune_strategy_entry_rules():
    diags = LossReviewer().review(base_trade(forced=1, strength=0.2,
                                             entry_features={"atr": 2.0, "htf_trend": -1, "extension": 3.0}))
    assert codes(diags) == {"FORCED_PROBE"}


def test_history_blocks_adjustment_when_group_is_not_worse():
    t = base_trade(entry_features={"atr": 2.0, "htf_trend": -1, "extension": 0.5})
    history = ([{"direction": 1, "r_multiple": 0.8, "entry_features": {"htf_trend": -1}}] * 5 +
               [{"direction": 1, "r_multiple": -0.2, "entry_features": {"htf_trend": 1}}] * 5)
    [ct] = [d for d in LossReviewer().review(t, history) if d.code == "COUNTER_TREND"]
    assert ct.adjustments == []
    worse = ([{"direction": 1, "r_multiple": -0.9, "entry_features": {"htf_trend": -1}}] * 5 +
             [{"direction": 1, "r_multiple": 0.6, "entry_features": {"htf_trend": 1}}] * 5)
    [ct] = [d for d in LossReviewer().review(t, worse) if d.code == "COUNTER_TREND"]
    assert ct.adjustments


def test_hindsight_flags_stop_too_tight():
    after = pd.DataFrame({"open": [96, 100], "high": [99, 112], "low": [94, 99], "close": [98, 111]})
    assert codes(LossReviewer().hindsight(base_trade(), after)) == {"STOP_TOO_TIGHT"}
    flat = pd.DataFrame({"open": [96, 95], "high": [97, 96], "low": [90, 88], "close": [92, 89]})
    assert LossReviewer().hindsight(base_trade(), flat) == []


def test_summary_is_readable():
    text = summarize(base_trade(), LossReviewer().review(base_trade(regime_exit="range")), ["note"])
    assert "BTC/USDT" in text and "REGIME" not in text and "สภาวะตลาดเปลี่ยน" in text


# --------------------------------------------------------------------------- adapter
def make_adapter(**learning):
    journal = Journal(":memory:")
    params = {"stop_atr": 2.0, "flag": 0}
    store = ParamStore()
    store.register("trend", params, {"stop_atr": (1.0, 3.0, 0.25), "flag": (0, 1, 1)})
    cfg = {"enabled": True, "evidence_required": 2, "probation_trades": 3, "revert_tolerance": 0.05, **learning}
    return Adapter(store, journal, cfg), store, params, journal


def diag(value=0.5):
    return [Diagnosis("STOP_TOO_TIGHT", "t", "d", [Adjust("trend", "stop_atr", "add", value)])]


def closed(journal, r, n=1):
    for _ in range(n):
        tid = journal.open_trade(symbol="X", strategy="trend", direction=1, forced=0)
        journal.close_trade(tid, r_multiple=r, pnl=r)


def test_adjustment_requires_repeated_evidence_and_respects_bounds():
    adapter, store, params, _ = make_adapter()
    trade = {"strategy": "trend"}
    assert adapter.submit(trade, diag(), "t1")[0].startswith("เก็บหลักฐาน")
    assert params["stop_atr"] == 2.0
    adapter.submit(trade, diag(), "t2")
    assert params["stop_atr"] == 2.5
    for _ in range(2):
        adapter.submit(trade, diag(5.0), "t3")
    assert params["stop_atr"] == 3.0  # clamped to max
    assert store.clamp("trend", "flag", 0.7) == 1 and isinstance(store.clamp("trend", "flag", 0.7), int)


def test_adjustment_reverted_when_results_get_worse():
    adapter, _, params, journal = make_adapter(evidence_required=1)
    closed(journal, 0.5, n=5)  # baseline +0.5R
    adapter.submit({"strategy": "trend"}, diag(), "t")
    assert params["stop_atr"] == 2.5
    for _ in range(3):
        adapter.on_trade_closed({"strategy": "trend", "r_multiple": -1.0}, "t")
    assert params["stop_atr"] == 2.0
    assert journal.adjustments()[0]["status"] == "reverted"


def test_adjustment_kept_when_results_hold_up():
    adapter, _, params, journal = make_adapter(evidence_required=1)
    closed(journal, -0.5, n=5)
    adapter.submit({"strategy": "trend"}, diag(), "t")
    for _ in range(3):
        adapter.on_trade_closed({"strategy": "trend", "r_multiple": 0.4}, "t")
    assert params["stop_atr"] == 2.5
    assert journal.adjustments()[0]["status"] == "kept"


def test_learning_disabled_changes_nothing():
    adapter, _, params, _ = make_adapter(enabled=False, evidence_required=1)
    assert adapter.submit({"strategy": "trend"}, diag(), "t") == []
    assert params["stop_atr"] == 2.0


def test_llm_suggestion_limited_to_two_steps():
    adapter, _, params, _ = make_adapter()
    adapter.apply_suggestion("trend", "stop_atr", 3.0, "wide", "t")
    assert params["stop_atr"] == 2.5
    assert adapter.apply_suggestion("trend", "unknown", 1.0, "x", "t") is None
