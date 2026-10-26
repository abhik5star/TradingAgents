"""Statistics that decide whether a result means anything.

The rules of this module:

* Every test states its assumptions and its sample size. Nothing returns a bare
  "significant".
* Anything that could be produced by chance gets a distribution: permutation
  tests for "is A different from B", bootstrap intervals for "how uncertain is
  this expectancy", Monte Carlo reshuffling for "how bad could the drawdown have
  been".
* Multiple comparisons are corrected, because a lab that tries two hundred ideas
  will always find one that looks good.
* Win rate is never used to rank anything on its own. Expectancy, profit factor,
  risk-adjusted return, drawdown and out-of-sample stability are.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents.research.contracts import Metrics, TradeRecord

TRADING_DAYS = 252


def _as_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray([float(v) for v in values], dtype=float)
    return array[np.isfinite(array)]


def mean(values: Sequence[float]) -> float:
    array = _as_array(values)
    return float(array.mean()) if len(array) else 0.0


def standard_deviation(values: Sequence[float]) -> float:
    array = _as_array(values)
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def max_drawdown(equity: Sequence[float]) -> tuple[float, float]:
    """Peak-to-trough drawdown in currency and as a fraction of the peak."""
    curve = np.asarray([float(v) for v in equity], dtype=float)
    if len(curve) == 0:
        return 0.0, 0.0
    peaks = np.maximum.accumulate(curve)
    drawdowns = peaks - curve
    worst = float(drawdowns.max()) if len(drawdowns) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(peaks > 0, drawdowns / peaks, 0.0)
    return worst, float(fraction.max())


def sharpe_ratio(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float | None:
    array = _as_array(returns)
    if len(array) < 2:
        return None
    deviation = array.std(ddof=1)
    if deviation == 0:
        return None
    return float(array.mean() / deviation * math.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float | None:
    array = _as_array(returns)
    if len(array) < 2:
        return None
    downside = array[array < 0]
    if len(downside) == 0:
        return None
    deviation = math.sqrt(float((downside ** 2).mean()))
    if deviation == 0:
        return None
    return float(array.mean() / deviation * math.sqrt(periods_per_year))


def profit_factor(pnls: Sequence[float]) -> float | None:
    array = _as_array(pnls)
    gains = array[array > 0].sum()
    losses = -array[array < 0].sum()
    if losses == 0:
        return None if gains == 0 else float("inf")
    return float(gains / losses)


def expectancy(pnls: Sequence[float]) -> float:
    return mean(pnls)


def expectancy_r(r_multiples: Sequence[float]) -> float:
    return mean(r_multiples)


def _breakdown(trades: Sequence[TradeRecord], attribute: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        groups.setdefault(str(getattr(trade, attribute)), []).append(trade)
    breakdown: dict[str, dict[str, float]] = {}
    for key, group in groups.items():
        pnls = [t.net_pnl for t in group]
        breakdown[key] = {
            "trades": float(len(group)),
            "net_pnl": round(float(sum(pnls)), 4),
            "expectancy": round(float(np.mean(pnls)), 4),
            "win_rate": round(float(np.mean([p > 0 for p in pnls])), 4),
            "profit_factor": round(profit_factor(pnls) or 0.0, 4),
        }
    return breakdown


def compute_metrics(trades: Sequence[TradeRecord], equity_curve: Sequence[float],
                    initial_equity: float, bars_in_market: int = 0, total_bars: int = 0,
                    periods_per_year: int = TRADING_DAYS) -> Metrics:
    """Every headline number, from the trade list and the realised equity curve."""
    metrics = Metrics()
    metrics.trades = len(trades)
    if not trades:
        return metrics
    pnls = np.asarray([t.net_pnl for t in trades], dtype=float)
    gross = np.asarray([t.gross_pnl for t in trades], dtype=float)
    costs = np.asarray([t.cost for t in trades], dtype=float)
    r_multiples = np.asarray([t.r_multiple for t in trades], dtype=float)

    metrics.wins = int((pnls > 0).sum())
    metrics.losses = int((pnls < 0).sum())
    metrics.win_rate = metrics.wins / metrics.trades
    metrics.net_pnl = float(pnls.sum())
    metrics.gross_pnl = float(gross.sum())
    metrics.total_cost = float(costs.sum())
    metrics.expectancy = float(pnls.mean())
    metrics.expectancy_r = float(r_multiples.mean())
    metrics.profit_factor = profit_factor(pnls)
    metrics.avg_win = float(pnls[pnls > 0].mean()) if metrics.wins else 0.0
    metrics.avg_loss = float(pnls[pnls < 0].mean()) if metrics.losses else 0.0
    worst, worst_fraction = max_drawdown(equity_curve)
    metrics.max_drawdown = float(worst)
    metrics.max_drawdown_pct = float(worst_fraction)
    if initial_equity:
        metrics.total_return_pct = float(equity_curve[-1] / initial_equity - 1) if len(equity_curve) else 0.0
    metrics.exposure_pct = float(bars_in_market / total_bars) if total_bars else 0.0
    metrics.avg_bars_held = float(np.mean([t.bars_held for t in trades]))

    # Daily returns from the realised curve, so Sharpe and Sortino are comparable
    # across strategies with different trade counts.
    if len(equity_curve) > 2:
        curve = pd.Series(equity_curve, dtype=float)
        daily = curve.pct_change().dropna()
        metrics.sharpe = sharpe_ratio(daily.to_numpy(), periods_per_year)
        metrics.sortino = sortino_ratio(daily.to_numpy(), periods_per_year)
    if gross.sum() > 0:
        metrics.cost_share_of_gross = float(costs.sum() / gross.sum())
    metrics.by_regime = _breakdown(trades, "regime")
    metrics.by_session = _breakdown(trades, "session")
    return metrics


# --------------------------------------------------------------------------------------
# Hypothesis testing without scipy
# --------------------------------------------------------------------------------------


@dataclass
class TestResult:
    """A difference between two samples, with the test that produced it."""

    statistic: float
    p_value: float
    method: str
    sample_a: int
    sample_b: int
    effect_size: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict[str, float | str | int]:
        return {"statistic": round(self.statistic, 6), "p_value": round(self.p_value, 6),
                "method": self.method, "sample_a": self.sample_a, "sample_b": self.sample_b,
                "effect_size": round(self.effect_size, 6)}


def welch_t(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t statistic and its degrees of freedom (no distributional p-value)."""
    array_a, array_b = _as_array(a), _as_array(b)
    if len(array_a) < 2 or len(array_b) < 2:
        return 0.0, 0.0
    variance_a, variance_b = array_a.var(ddof=1), array_b.var(ddof=1)
    denominator = math.sqrt(variance_a / len(array_a) + variance_b / len(array_b))
    if denominator == 0:
        return 0.0, float(len(array_a) + len(array_b) - 2)
    statistic = (array_a.mean() - array_b.mean()) / denominator
    numerator = (variance_a / len(array_a) + variance_b / len(array_b)) ** 2
    denominator_df = ((variance_a / len(array_a)) ** 2 / max(1, len(array_a) - 1)
                      + (variance_b / len(array_b)) ** 2 / max(1, len(array_b) - 1))
    degrees = numerator / denominator_df if denominator_df else 1.0
    return float(statistic), float(degrees)


def permutation_test(a: Sequence[float], b: Sequence[float], n_permutations: int = 2000,
                     seed: int = 13) -> TestResult:
    """Two-sided permutation test on the difference in means.

    Distribution-free: it only assumes the two samples are exchangeable under the
    null, which is the right assumption for "were these trades different from
    those trades?" and needs no normality.
    """
    array_a, array_b = _as_array(a), _as_array(b)
    if len(array_a) == 0 or len(array_b) == 0:
        return TestResult(0.0, 1.0, "permutation", len(array_a), len(array_b), 0.0)
    observed = float(array_a.mean() - array_b.mean())
    pooled = np.concatenate([array_a, array_b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        difference = shuffled[: len(array_a)].mean() - shuffled[len(array_a):].mean()
        if abs(difference) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    pooled_sd = math.sqrt(((len(array_a) - 1) * array_a.var(ddof=1)
                          + (len(array_b) - 1) * array_b.var(ddof=1))
                          / max(1, len(array_a) + len(array_b) - 2))
    effect = observed / pooled_sd if pooled_sd else 0.0
    return TestResult(observed, float(p_value), "permutation", len(array_a), len(array_b),
                      float(effect))


def bootstrap_ci(values: Sequence[float], statistic: Callable[[np.ndarray], float] = np.mean,
                 n_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 17) -> tuple[float, float]:
    """Percentile bootstrap interval of ``statistic`` over the sample."""
    array = _as_array(values)
    if len(array) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    draws = np.array([statistic(rng.choice(array, size=len(array), replace=True))
                      for _ in range(n_bootstrap)])
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def monte_carlo_drawdown(pnls: Sequence[float], initial_equity: float, n_simulations: int = 2000,
                         seed: int = 19) -> dict[str, float]:
    """Distribution of maximum drawdown when the same trades arrive in another order.

    Trade order is not a strategy choice, so a drawdown that only survives one
    ordering is luck, not risk control.
    """
    array = _as_array(pnls)
    if len(array) == 0:
        return {"median": 0.0, "p95": 0.0, "worst": 0.0}
    rng = np.random.default_rng(seed)
    drawdowns = np.empty(n_simulations)
    for i in range(n_simulations):
        shuffled = rng.permutation(array)
        curve = initial_equity + np.concatenate([[0.0], np.cumsum(shuffled)])
        drawdowns[i] = max_drawdown(curve)[0]
    return {"median": float(np.median(drawdowns)),
            "p95": float(np.quantile(drawdowns, 0.95)),
            "worst": float(drawdowns.max())}


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> dict[str, object]:
    """Benjamini-Hochberg false-discovery control over a family of experiments.

    Returns the adjusted q-values and the rejected set, so an experiment record can
    say "3 of 40 candidates survived FDR control at 5%" rather than pretending the
    best of forty is a discovery.
    """
    values = np.asarray([float(p) for p in p_values], dtype=float)
    n = len(values)
    if n == 0:
        return {"q_values": [], "rejected": [], "alpha": alpha, "tests": 0}
    order = np.argsort(values)
    ranked = values[order]
    q_values = np.empty(n)
    running = 1.0
    for position in range(n - 1, -1, -1):
        running = min(running, ranked[position] * n / (position + 1))
        q_values[order[position]] = running
    rejected = [bool(q <= alpha) for q in q_values]
    return {"q_values": [round(float(q), 6) for q in q_values],
            "rejected": rejected, "alpha": alpha, "tests": n,
            "discoveries": int(sum(rejected))}


def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> dict[str, object]:
    """The conservative correction: ``alpha`` divided by the number of tests."""
    values = [float(p) for p in p_values]
    if not values:
        return {"threshold": alpha, "rejected": [], "tests": 0}
    threshold = alpha / len(values)
    return {"threshold": threshold, "rejected": [p <= threshold for p in values],
            "tests": len(values)}


def adjusted_p_value(p_value: float, family_size: int) -> float:
    """Sidak-style adjustment for a single p-value drawn from ``family_size`` tests."""
    if family_size <= 1:
        return float(p_value)
    return float(1 - (1 - float(p_value)) ** family_size)


# --------------------------------------------------------------------------------------
# Stability, sensitivity and distribution shape
# --------------------------------------------------------------------------------------


def parameter_stability(expectancies: Sequence[float]) -> dict[str, float]:
    """How much a result depends on the exact parameters.

    A robust strategy keeps a positive expectancy across neighbouring parameters.
    ``fragile`` is flagged when fewer than two thirds of the variants are
    profitable or the spread is wider than the average edge itself.
    """
    values = _as_array(expectancies)
    if len(values) == 0:
        return {"variants": 0, "profitable_share": 0.0, "spread": 0.0, "fragile": 1.0}
    profitable = float((values > 0).mean())
    spread = float(values.max() - values.min())
    centre = float(values.mean())
    fragile = bool(profitable < 2 / 3 or (centre > 0 and spread > abs(centre) * 2) or centre <= 0)
    return {"variants": int(len(values)), "profitable_share": round(profitable, 4),
            "mean_expectancy": round(centre, 6), "spread": round(spread, 6),
            "best": round(float(values.max()), 6), "worst": round(float(values.min()), 6),
            "fragile": float(fragile)}


def distribution_summary(returns: Sequence[float]) -> dict[str, float]:
    """Mean, deviation, skew, excess kurtosis and the 5% tail of a return sample."""
    array = _as_array(returns)
    if len(array) < 3:
        return {"n": int(len(array)), "mean": 0.0, "std": 0.0, "skew": 0.0,
                "excess_kurtosis": 0.0, "var_5pct": 0.0}
    deviation = array.std(ddof=1)
    if deviation == 0:
        return {"n": int(len(array)), "mean": float(array.mean()), "std": 0.0, "skew": 0.0,
                "excess_kurtosis": 0.0, "var_5pct": float(array.min())}
    centred = array - array.mean()
    skew = float((centred ** 3).mean() / deviation ** 3)
    kurtosis = float((centred ** 4).mean() / deviation ** 4 - 3)
    return {"n": int(len(array)), "mean": float(array.mean()), "std": float(deviation),
            "skew": round(skew, 4), "excess_kurtosis": round(kurtosis, 4),
            "var_5pct": float(np.quantile(array, 0.05))}


def spearman_correlation(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation, used for feature importance without a linearity assumption."""
    left, right = _as_array(a), _as_array(b)
    if len(left) < 3 or len(left) != len(right):
        return None
    ranks_left = pd.Series(left).rank().to_numpy()
    ranks_right = pd.Series(right).rank().to_numpy()
    if ranks_left.std() == 0 or ranks_right.std() == 0:
        return None
    return float(np.corrcoef(ranks_left, ranks_right)[0, 1])


def feature_importance(features: pd.DataFrame, target: pd.Series,
                       min_observations: int = 100) -> dict[str, float]:
    """Univariate Spearman importance of every feature against a forward target.

    Univariate on purpose: it answers "does this measurement carry information?"
    before a model is allowed to combine them, and it cannot manufacture an edge
    by fitting noise across hundreds of columns the way a tree ensemble can.
    """
    joined = features.join(target.rename("__target"), how="inner").dropna(subset=["__target"])
    if len(joined) < min_observations:
        return {}
    importance: dict[str, float] = {}
    for column in joined.columns:
        if column == "__target":
            continue
        values = joined[column]
        if values.nunique() < 2:
            continue
        correlation = spearman_correlation(values.to_numpy(), joined["__target"].to_numpy())
        if correlation is not None:
            importance[column] = round(correlation, 4)
    return dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))


def sample_adequacy(trades: int, minimum: int = 500) -> dict[str, object]:
    """Whether a sample is large enough for the claim being made about it.

    ``minimum`` is a configurable target, not proof: 500 trades do not make a
    strategy profitable, and 40 trades of a rare setup can still be worth
    reporting — but not worth promoting.
    """
    return {"trades": int(trades), "target": int(minimum), "adequate": bool(trades >= minimum),
            "shortfall": max(0, int(minimum) - int(trades)),
            "note": ("sample target met; this is not evidence of profitability"
                     if trades >= minimum else
                     "sample below target: treat every metric as provisional")}


def annualised_return(equity_curve: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate of a realised curve (0 when it is not computable)."""
    curve = _as_array(equity_curve)
    if len(curve) < 2 or curve[0] <= 0 or curve[-1] <= 0:
        return 0.0
    years = len(curve) / periods_per_year
    if years <= 0:
        return 0.0
    return float((curve[-1] / curve[0]) ** (1 / years) - 1)


def rolling_expectancy(pnls: Sequence[float], window: int) -> list[float]:
    """Expectancy over rolling trade windows, for a stability chart."""
    array = _as_array(pnls)
    if len(array) < window:
        return []
    series = pd.Series(array)
    return [round(float(v), 6) for v in series.rolling(window).mean().dropna().to_list()]
