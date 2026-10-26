"""Ranking a strategy on more than a good-looking equity curve.

Win rate and gross profit are deliberately absent as ranking inputs. A strategy
that wins 90% of the time while losing everything in the tenth trade is not a
strategy; a strategy with a 30% win rate and +0.4R expectancy may be an excellent
one. The score combines the measurements that survive scrutiny:

* net expectancy in R (after costs),
* profit factor,
* risk-adjusted return (Sharpe, with Sortino as a supporting figure),
* maximum drawdown,
* out-of-sample retention of the in-sample edge,
* walk-forward consistency,
* stability across regimes,
* sample size, and the uncertainty that comes with a small one,
* parameter sensitivity,
* cost share of gross profit.

The score is a ranking device, not a verdict: :func:`acceptance_gates` decides
promotion. A high score with a failed gate is still a rejection, and when no
candidate clears the bar the answer is :data:`NO_TRADE`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tradingagents.research.contracts import Metrics

#: Returned as the verdict when nothing is good enough to trade.
NO_TRADE = "NO TRADE"


@dataclass(frozen=True)
class ScoreWeights:
    """Weights for the composite score; they sum to 1 after normalisation."""

    expectancy_r: float = 0.20
    profit_factor: float = 0.14
    sharpe: float = 0.12
    drawdown: float = 0.14
    out_of_sample: float = 0.10
    walk_forward: float = 0.10
    regime_stability: float = 0.06
    sample: float = 0.06
    sensitivity: float = 0.05
    cost: float = 0.03

    def as_dict(self) -> dict[str, float]:
        total = sum(self.__dict__.values())
        return {key: round(value / total, 4) for key, value in self.__dict__.items()}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scale(value: float | None, low: float, high: float) -> float:
    if value is None:
        return 0.0
    if high == low:
        return 0.0
    return _clip((value - low) / (high - low))


@dataclass
class StrategyScore:
    """A composite score with its components and every penalty that reduced it."""

    total: float
    components: dict[str, float]
    penalties: dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def tradeable(self) -> bool:
        return self.verdict != NO_TRADE

    def as_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 4), "components": self.components,
                "penalties": self.penalties, "verdict": self.verdict,
                "reasons": self.reasons}


def regime_stability(metrics: Metrics, min_trades_per_regime: int = 5) -> float:
    """Share of evaluated regimes with a positive expectancy.

    Regimes with fewer than ``min_trades_per_regime`` trades are ignored rather
    than counted as wins: two lucky trades in a quiet market say nothing.
    """
    evaluated = {name: stats for name, stats in (metrics.by_regime or {}).items()
                 if stats.get("trades", 0) >= min_trades_per_regime}
    if not evaluated:
        return 0.0
    positive = sum(1 for stats in evaluated.values() if stats.get("expectancy", 0) > 0)
    return positive / len(evaluated)


def score_strategy(
    metrics: Metrics,
    *,
    oos_metrics: Metrics | None = None,
    walk_forward_positive_share: float | None = None,
    sensitivity: dict[str, float] | None = None,
    weights: ScoreWeights | None = None,
    minimum_trades: int = 40,
    target_trades: int = 500,
) -> StrategyScore:
    """Score one evaluated strategy. Every component is stated in the result."""
    weights = weights or ScoreWeights()
    normalised = weights.as_dict()

    components = {
        "expectancy_r": _scale(metrics.expectancy_r, -0.2, 0.6),
        "profit_factor": _scale(metrics.profit_factor, 1.0, 2.0),
        "sharpe": _scale(metrics.sharpe, 0.0, 2.0),
        "drawdown": 1 - _clip(metrics.max_drawdown_pct / 0.30),
        "out_of_sample": _scale(
            oos_metrics.expectancy_r if oos_metrics is not None else None, -0.1, 0.5),
        "walk_forward": _clip(walk_forward_positive_share
                              if walk_forward_positive_share is not None else 0.0),
        "regime_stability": regime_stability(metrics),
        "sample": _clip(metrics.trades / target_trades) if target_trades else 1.0,
        "sensitivity": 1 - _clip((sensitivity or {}).get("fragile", 1.0)),
        "cost": 1 - _clip((metrics.cost_share_of_gross or 1.0) / 0.5),
    }
    total = sum(normalised[name] * components[name] for name in normalised) * 100

    penalties: dict[str, float] = {}
    reasons: list[str] = []
    if metrics.trades < minimum_trades:
        penalties["thin_sample"] = 15.0
        reasons.append(f"only {metrics.trades} trades: the score is discounted for uncertainty")
    if oos_metrics is not None and oos_metrics.expectancy_r <= 0:
        penalties["negative_oos"] = 60.0
        reasons.append("out-of-sample expectancy is not positive")
    if metrics.max_drawdown_pct > 0.25:
        penalties["deep_drawdown"] = 10.0
        reasons.append(f"drawdown of {metrics.max_drawdown_pct:.1%} is worse than the 25% guardrail")
    if (sensitivity or {}).get("fragile", 0) >= 0.5:
        penalties["fragile_parameters"] = 10.0
        reasons.append("the result depends on the exact parameters chosen")
    if (metrics.cost_share_of_gross or 0) > 0.5:
        penalties["cost_dominated"] = 10.0
        reasons.append("transaction costs consume more than half of gross profit")

    total -= sum(penalties.values())
    total = max(0.0, min(100.0, total))

    verdict = NO_TRADE if total <= 0 or penalties.get("negative_oos") else "candidate"
    if verdict == NO_TRADE and not reasons:
        reasons.append("the composite score is zero or negative")
    return StrategyScore(total=round(total, 2), components={k: round(v, 4) for k, v in
                                                            components.items()},
                         penalties=penalties, verdict=verdict, reasons=reasons)


def rank_candidates(scores: dict[str, StrategyScore]) -> list[tuple[str, StrategyScore]]:
    """Candidates best-first. Ties are broken by name so the order is reproducible."""
    return sorted(scores.items(), key=lambda item: (-item[1].total, item[0]))


def select_champion(scores: dict[str, StrategyScore],
                    minimum_score: float = 40.0) -> tuple[str | None, StrategyScore]:
    """The best tradeable candidate, or ``(None, NO_TRADE score)`` with the reason.

    A candidate below ``minimum_score`` is not selected however it ranks among its
    peers: being the least bad of ten untested ideas is not an achievement.
    """
    ranked = rank_candidates(scores)
    for name, score in ranked:
        if score.tradeable and score.total >= minimum_score:
            return name, score
    best_name = ranked[0][0] if ranked else "none"
    best = ranked[0][1] if ranked else StrategyScore(0.0, {}, verdict=NO_TRADE)
    reasons = [f"best candidate {best_name} scored {best.total:.1f} < {minimum_score:.0f}"]
    reasons.extend(best.reasons)
    return None, StrategyScore(total=best.total, components=best.components,
                               penalties=best.penalties, verdict=NO_TRADE,
                               reasons=reasons)


def compare_metrics(baseline: Metrics, candidate: Metrics,
                    keys: Iterable[str] | None = None) -> dict[str, dict[str, float]]:
    """Baseline versus candidate, field by field, with the difference spelled out."""
    fields = list(keys or ("trades", "expectancy", "expectancy_r", "profit_factor",
                           "max_drawdown_pct", "sharpe", "sortino", "net_pnl",
                           "total_cost", "cost_share_of_gross"))
    comparison: dict[str, dict[str, float]] = {}
    for key in fields:
        base_value = getattr(baseline, key, None)
        candidate_value = getattr(candidate, key, None)
        if base_value is None or candidate_value is None:
            continue
        comparison[key] = {"baseline": round(float(base_value), 6),
                           "candidate": round(float(candidate_value), 6),
                           "difference": round(float(candidate_value) - float(base_value), 6)}
    return comparison


def no_trade_explanation(reasons: Iterable[str]) -> str:
    """The one-line "why we are not trading" string the dashboard shows."""
    joined = "; ".join(reason for reason in reasons if reason)
    return f"{NO_TRADE}: {joined}" if joined else f"{NO_TRADE}: no candidate cleared the gates"
