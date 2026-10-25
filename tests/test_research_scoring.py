"""Composite scoring: ranking, penalties and the honest NO TRADE verdict."""

from __future__ import annotations

import pytest

from tradingagents.research.contracts import Metrics
from tradingagents.research.scoring import (
    NO_TRADE,
    ScoreWeights,
    compare_metrics,
    no_trade_explanation,
    rank_candidates,
    regime_stability,
    score_strategy,
    select_champion,
)


def _metrics(**overrides) -> Metrics:
    payload = {
        "trades": 120, "wins": 55, "losses": 65, "win_rate": 0.458,
        "net_pnl": 12_400.0, "gross_pnl": 18_000.0, "total_cost": 5_600.0,
        "expectancy": 103.3, "expectancy_r": 0.32, "profit_factor": 1.55,
        "avg_win": 420.0, "avg_loss": -190.0, "max_drawdown": 9_000.0,
        "max_drawdown_pct": 0.09, "sharpe": 1.4, "sortino": 2.1,
        "total_return_pct": 12.4, "exposure_pct": 0.35, "avg_bars_held": 4.2,
        "cost_share_of_gross": 0.31,
        "by_regime": {"trend_up": {"trades": 30, "expectancy": 0.4},
                      "range": {"trades": 20, "expectancy": 0.1},
                      "high_vol": {"trades": 8, "expectancy": -0.2},
                      "thin": {"trades": 2, "expectancy": -1.0}},
    }
    payload.update(overrides)
    return Metrics(**payload)


@pytest.mark.unit
def test_a_strong_result_scores_well_and_states_its_components():
    score = score_strategy(_metrics(), oos_metrics=_metrics(trades=60, expectancy_r=0.24),
                           walk_forward_positive_share=0.75,
                           sensitivity={"fragile": 0.1, "profitable_share": 0.9})
    assert score.total > 55
    assert score.verdict != NO_TRADE
    assert set(score.components) == set(ScoreWeights().as_dict())
    assert not score.penalties


@pytest.mark.unit
def test_the_score_is_bounded_and_never_exceeds_one_hundred():
    perfect = _metrics(trades=800, expectancy_r=5.0, profit_factor=9.0, sharpe=8.0,
                       max_drawdown_pct=0.0, cost_share_of_gross=0.0,
                       by_regime={"trend_up": {"trades": 100, "expectancy": 1.0}})
    score = score_strategy(perfect, oos_metrics=perfect, walk_forward_positive_share=1.0,
                           sensitivity={"fragile": 0.0})
    assert 0 <= score.total <= 100


@pytest.mark.unit
def test_negative_out_of_sample_expectancy_forces_no_trade():
    score = score_strategy(_metrics(), oos_metrics=_metrics(expectancy_r=-0.12))
    assert score.verdict == NO_TRADE
    assert score.penalties["negative_oos"] >= 50
    assert any("out-of-sample" in reason for reason in score.reasons)


@pytest.mark.unit
def test_penalties_name_each_weakness_they_discount_for():
    score = score_strategy(_metrics(trades=12, max_drawdown_pct=0.42,
                                    cost_share_of_gross=0.7),
                           sensitivity={"fragile": 0.7})
    assert {"thin_sample", "deep_drawdown", "cost_dominated",
            "fragile_parameters"} <= set(score.penalties)
    text = " ".join(score.reasons)
    assert "trades" in text and "drawdown" in text and "costs" in text


@pytest.mark.unit
def test_regime_stability_ignores_regimes_with_too_few_trades():
    metrics = _metrics(by_regime={"trend_up": {"trades": 40, "expectancy": 0.3},
                                  "range": {"trades": 30, "expectancy": -0.1},
                                  "thin": {"trades": 3, "expectancy": 5.0}})
    assert regime_stability(metrics, min_trades_per_regime=5) == pytest.approx(0.5)
    assert regime_stability(_metrics(by_regime={})) == 0.0


@pytest.mark.unit
def test_ranking_is_deterministic_and_ties_break_by_name():
    first = score_strategy(_metrics())
    second = score_strategy(_metrics(expectancy_r=0.10))
    # Two identical scores: the order must still be reproducible.
    ranked = rank_candidates({"b": first, "a": first})
    assert [name for name, _ in ranked] == ["a", "b"]
    assert ranked[0][1].total == ranked[1][1].total
    best = rank_candidates({"a": first, "b": second})
    assert best[0][0] == "a"


@pytest.mark.unit
def test_no_candidate_above_the_floor_yields_an_explained_no_trade():
    weak = score_strategy(_metrics(expectancy_r=0.02, profit_factor=1.02, sharpe=0.05),
                          oos_metrics=_metrics(expectancy_r=-0.01))
    name, score = select_champion({"weak": weak}, minimum_score=40.0)
    assert name is None
    assert score.verdict == NO_TRADE
    assert any("scored" in reason for reason in score.reasons)
    text = no_trade_explanation(score.reasons)
    assert text.startswith(NO_TRADE) and len(text) > len(NO_TRADE) + 5


@pytest.mark.unit
def test_a_high_score_below_the_floor_is_still_refused():
    decent = score_strategy(_metrics(expectancy_r=0.05, profit_factor=1.05, sharpe=0.2,
                                     max_drawdown_pct=0.2))
    assert decent.total < 40
    assert select_champion({"decent": decent})[0] is None


@pytest.mark.unit
def test_the_champion_is_the_best_tradeable_candidate_not_the_first():
    good = score_strategy(_metrics(), oos_metrics=_metrics(expectancy_r=0.2))
    bad = score_strategy(_metrics(expectancy_r=-0.3, profit_factor=0.6))
    name, score = select_champion({"bad": bad, "good": good}, minimum_score=40.0)
    assert name == "good"
    assert score.tradeable


@pytest.mark.unit
def test_metric_comparison_reports_the_difference_not_just_two_numbers():
    comparison = compare_metrics(_metrics(), _metrics(expectancy_r=0.42, trades=95))
    assert comparison["expectancy_r"]["difference"] == pytest.approx(0.10, abs=1e-6)
    assert comparison["trades"]["difference"] == -25
    assert "baseline" in comparison["profit_factor"]
