"""Validation gates: chronological splits, walk-forward, look-ahead probing, acceptance.

A candidate strategy is not accepted because a backtest looked good. It is
accepted only when every gate below passes on data the search never touched:

* chronological train / validation / **untouched test** windows with a purge gap
  (so overlapping labels cannot leak) and an embargo after each boundary,
* out-of-sample expectancy, profit factor, drawdown against the baseline and
  cost share,
* a permutation test on the out-of-sample difference, corrected for the number of
  experiments in the family,
* walk-forward consistency across folds,
* parameter sensitivity, and a cost-stress rerun at multiplied friction,
* a look-ahead probe that rebuilds the features on truncated data and compares,
* a holdout-usage budget, so the same final window is not optimised against
  indefinitely,
* data quality and sample size.

Every gate returns what was required, what was measured and whether it passed, so
a rejection is explainable rather than mysterious.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.research.contracts import GateResult, Metrics, stable_hash, utc_now
from tradingagents.research.engine import BacktestResult, EngineConfig, run_backtest
from tradingagents.research.features import FeatureTable, build_feature_table
from tradingagents.research.quant import (
    adjusted_p_value,
    benjamini_hochberg,
    parameter_stability,
    permutation_test,
)
from tradingagents.research.strategies import StrategySpec, parameter_variants


@dataclass(frozen=True)
class ChronologicalSplit:
    """Three consecutive windows of one table, with the gaps that stop leakage."""

    train: tuple[pd.Timestamp, pd.Timestamp]
    validation: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]
    purge_bars: int = 5
    embargo_bars: int = 2

    def label_bounds(self) -> dict[str, str]:
        return {name: f"{bounds[0].date()}..{bounds[1].date()}"
                for name, bounds in (("train", self.train), ("validation", self.validation),
                                     ("test", self.test))}

    def as_dict(self) -> dict[str, Any]:
        return {"train": self.label_bounds()["train"],
                "validation": self.label_bounds()["validation"],
                "test": self.label_bounds()["test"],
                "purge_bars": self.purge_bars,
                "embargo_bars": self.embargo_bars}

    def subset(self, table: FeatureTable, segment: str) -> FeatureTable:
        """The bars of one segment, cut with the purge/embargo gaps applied.

        ``train`` stops ``purge_bars`` before its nominal end, ``validation``
        starts ``embargo_bars`` after it and stops short of the test window by the
        purge, and ``test`` is preceded by both gaps. That is what makes the test
        window untouched rather than merely later.
        """
        bounds = {"train": self.train, "validation": self.validation, "test": self.test}[segment]
        window = table.bars.loc[bounds[0]:bounds[1]]
        if window.empty:
            raise ValueError(f"the {segment} window has no bars")
        return table.slice(str(window.index[0]), str(window.index[-1]))


def chronological_split(index: pd.DatetimeIndex, train_fraction: float = 0.5,
                        validation_fraction: float = 0.25, purge_bars: int = 5,
                        embargo_bars: int = 2) -> ChronologicalSplit:
    """Split a bar index into train / validation / test with purge and embargo gaps.

    ``purge_bars`` removes bars between segments whose forward-looking labels
    could overlap the next segment; ``embargo_bars`` additionally buffers the
    boundary so a rolling feature computed near the end of one segment cannot be
    reused as a decision input on the other side.
    """
    total = len(index)
    if total < 60:
        raise ValueError(f"at least 60 bars are needed to split a sample, got {total}")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 0.95:
        raise ValueError("leave a meaningful untouched test window")
    gap = int(purge_bars + embargo_bars)
    train_end = int(total * train_fraction) - purge_bars
    validation_start = train_end + gap
    validation_end = validation_start + int(total * validation_fraction) - gap
    test_start = validation_end + gap
    if train_end <= 10 or validation_end <= validation_start or test_start >= total - 5:
        raise ValueError("the requested split leaves a window too small to evaluate")
    return ChronologicalSplit(
        train=(index[10], index[train_end]),
        validation=(index[validation_start], index[validation_end]),
        test=(index[test_start], index[total - 1]),
        purge_bars=purge_bars, embargo_bars=embargo_bars,
    )


# --------------------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    index: int
    train_bounds: str
    test_bounds: str
    metrics: Metrics
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"fold": self.index, "train": self.train_bounds, "test": self.test_bounds,
                "metrics": self.metrics.as_dict(), "notes": self.notes}


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    positive_share: float
    mean_expectancy_r: float
    consistency_note: str

    @property
    def folds_run(self) -> int:
        return len(self.folds)

    def as_dict(self) -> dict[str, Any]:
        return {"folds": [f.as_dict() for f in self.folds],
                "positive_share": round(self.positive_share, 4),
                "mean_expectancy_r": round(self.mean_expectancy_r, 6),
                "consistency_note": self.consistency_note}


def walk_forward(spec: StrategySpec, table: FeatureTable, config: EngineConfig | None = None,
                 folds: int = 4, train_fraction: float = 0.6,
                 purge_bars: int = 5) -> WalkForwardResult:
    """Roll forward through the sample: train on the past, test on the next block.

    The rules are fixed, so "training" here means the *in-sample reference* window
    a researcher would have looked at before deciding to trade forward. Each fold's
    test block is only ever evaluated, never re-used to pick a parameter.
    """
    index = table.bars.index
    total = len(index)
    if folds < 2:
        raise ValueError("walk-forward needs at least two folds to say anything")
    initial_window = int(total * train_fraction)
    usable = total - initial_window
    if usable < folds * 10:
        raise ValueError(f"{total} bars cannot fill {folds} walk-forward folds")

    step = usable // folds
    results: list[WalkForwardFold] = []
    for fold in range(folds):
        train_end = initial_window + fold * step - purge_bars
        test_start = train_end + purge_bars + 1
        test_end = min(total - 1, test_start + step - 1)
        if train_end <= 10 or test_end <= test_start:
            continue
        window = table.slice(str(index[test_start]), str(index[test_end]))
        run = run_backtest(spec, window, replace(config) if config else EngineConfig())
        results.append(WalkForwardFold(
            index=fold + 1,
            train_bounds=f"{index[0].date()}..{index[train_end].date()}",
            test_bounds=f"{index[test_start].date()}..{index[test_end].date()}",
            metrics=run.metrics,
            notes=run.notes,
        ))
    if not results:
        return WalkForwardResult([], 0.0, 0.0, "no fold produced a testable window")
    expectancies = [f.metrics.expectancy_r for f in results]
    positive = float(np.mean([e > 0 for e in expectancies]))
    note = ("walk-forward expectancy is consistently positive across folds"
            if positive >= 0.75 else
            "walk-forward expectancy is inconsistent; the edge does not survive time")
    return WalkForwardResult(results, positive, float(np.mean(expectancies)), note)


# --------------------------------------------------------------------------------------
# Look-ahead probe
# --------------------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """The result of rebuilding features on truncated data and comparing."""

    ok: bool
    probes: int
    differences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "probes": self.probes, "differences": self.differences}


def probe_lookahead(data: Any, primary_timeframe: str | None, context_timeframe: str | None = None,
                    confirm_timeframe: str | None = None, probes: int = 3,
                    seed: int = 23, tolerance: float = 1e-9) -> LeakageReport:
    """Detect future information by rebuilding features with the future removed.

    For a probe bar *k*, the full feature table's row *k* must equal the row *k*
    produced from data truncated at *k*. Any difference — however small — means
    some feature recipe used a bar that had not printed yet, and the whole feature
    set is reported as leaking rather than being silently tolerated.
    """
    full = build_feature_table(data, primary_timeframe, context_timeframe, confirm_timeframe)
    frames = dict(data.frames.items())
    n = len(full.bars)
    candidates = list(range(max(60, n // 4), n - 10))
    if not candidates:
        return LeakageReport(ok=True, probes=0,
                             differences=[] if n else ["too few bars to probe"])
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(candidates, size=min(probes, len(candidates)), replace=False))
    differences: list[str] = []
    for k in chosen:
        cut = full.bars.index[k]
        truncated_frames = {tf: frame.loc[:cut] for tf, frame in frames.items()}
        from tradingagents.research.data import MarketData

        truncated = MarketData(ticker=data.ticker, frames=truncated_frames,
                               asset_type=data.asset_type, exchange=data.exchange,
                               provenance={"probe": True})
        rebuilt = build_feature_table(truncated, primary_timeframe, context_timeframe,
                                      confirm_timeframe)
        if cut not in rebuilt.features.index:
            continue
        left = full.features.loc[cut]
        right = rebuilt.features.loc[cut]
        for column in full.features.columns:
            a, b = left[column], right[column]
            if isinstance(a, str) or isinstance(b, str):
                if str(a) != str(b):
                    differences.append(f"{cut}: {column} differs ({a!r} vs {b!r}) after truncation")
                continue
            try:
                a_value, b_value = float(a), float(b)
            except (TypeError, ValueError):
                continue
            both_nan = np.isnan(a_value) and np.isnan(b_value)
            if both_nan:
                continue
            if np.isnan(a_value) != np.isnan(b_value) or abs(a_value - b_value) > tolerance:
                differences.append(
                    f"{cut}: {column} is {a_value:.6g} in the full table and "
                    f"{b_value:.6g} with the future removed")
    return LeakageReport(ok=not differences, probes=len(chosen), differences=differences)


# --------------------------------------------------------------------------------------
# Holdout discipline
# --------------------------------------------------------------------------------------


class HoldoutLedger:
    """Counts how often each test window has been evaluated.

    Repeatedly inspecting one holdout turns it into a training set. The ledger is
    persisted, so the count survives restarts and is reported to the user.
    """

    def __init__(self, store: Any):
        self.store = store

    def uses(self, fingerprint: str, window: str) -> int:
        document = self.store.load_holdouts()
        return int(document.get("windows", {}).get(f"{fingerprint}:{window}", 0))

    def record(self, fingerprint: str, window: str, experiment_id: str,
               max_uses: int) -> tuple[int, bool]:
        document = self.store.load_holdouts()
        windows = document.setdefault("windows", {})
        key = f"{fingerprint}:{window}"
        uses = int(windows.get(key, 0)) + 1
        windows[key] = uses
        document.setdefault("history", []).append({
            "key": key, "experiment": experiment_id, "at": utc_now(), "use": uses})
        self.store.save_holdouts(document)
        return uses, uses <= max_uses


# --------------------------------------------------------------------------------------
# Acceptance criteria and gates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriteria:
    """What must be true before a candidate may replace the baseline.

    ``target_trades`` is the 500-trade validation *target* from the brief, kept as
    a configurable ambition rather than a pass mark: reaching 500 trades does not
    demonstrate an edge, and a rare setup may legitimately need a different
    policy. ``min_trades`` is the hard floor.
    """

    min_trades: int = 40
    min_oos_trades: int = 25
    target_trades: int = 500
    min_expectancy_r: float = 0.0
    min_profit_factor: float = 1.1
    max_drawdown_vs_baseline: float = 1.15
    max_cost_share_of_gross: float = 0.5
    require_cost_stress_pass: bool = True
    max_p_value: float = 0.05
    min_walk_forward_positive_share: float = 0.6
    min_walk_forward_folds: int = 3
    max_holdout_uses: int = 3
    max_fragile_parameter_variants: float = 0.34
    require_beats_baseline: bool = True

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _baseline_comparison(candidate: BacktestResult, baseline: BacktestResult) -> GateResult:
    candidate_expectancy = candidate.metrics.expectancy_r
    baseline_expectancy = baseline.metrics.expectancy_r
    passed = candidate_expectancy > baseline_expectancy
    return GateResult(
        "beats_baseline", passed,
        required=f"out-of-sample expectancy > {baseline_expectancy:+.3f}R (baseline)",
        measured=f"{candidate_expectancy:+.3f}R",
        detail=("the candidate must earn more per unit of risk than the version it would replace"
                if passed else "the candidate does not improve on the baseline's expectancy"),
    )


def acceptance_gates(
    candidate: BacktestResult,
    baseline: BacktestResult,
    *,
    oos: BacktestResult,
    oos_baseline: BacktestResult,
    stress: BacktestResult | None,
    walk: WalkForwardResult | None,
    sensitivity: dict[str, float] | None,
    criteria: AcceptanceCriteria,
    family_size: int = 1,
    holdout_uses: int = 0,
    data_quality_ok: bool = True,
    leakage: LeakageReport | None = None,
) -> list[GateResult]:
    """Run every acceptance gate and return the full, auditable list.

    ``family_size`` is how many candidates the search evaluated before this one;
    the permutation p-value is adjusted for it, so the "best of forty" is not
    reported as a discovery.
    """
    gates: list[GateResult] = []

    gates.append(GateResult(
        "data_quality", data_quality_ok,
        required="market data passes its quality gate",
        measured="passed" if data_quality_ok else "failed",
        detail="a run on broken data cannot promote a strategy"))

    if leakage is not None:
        gates.append(GateResult(
            "no_lookahead",
            leakage.ok,
            required="features identical when the future is removed",
            measured=f"{leakage.probes} probes, {len(leakage.differences)} differences",
            detail="; ".join(leakage.differences[:3]) or "no future information detected",
        ))

    gates.append(GateResult(
        "sample_size",
        candidate.metrics.trades >= criteria.min_trades,
        required=f"at least {criteria.min_trades} trades (target {criteria.target_trades})",
        measured=f"{candidate.metrics.trades} trades",
        detail=("sample target met; 500 trades is an ambition, not proof of profitability"
                if candidate.metrics.trades >= criteria.target_trades else
                "below the validation target; metrics are provisional"),
    ))
    gates.append(GateResult(
        "oos_sample",
        oos.metrics.trades >= criteria.min_oos_trades,
        required=f"at least {criteria.min_oos_trades} out-of-sample trades",
        measured=f"{oos.metrics.trades} trades",
        detail="out-of-sample evidence is thin" if oos.metrics.trades < criteria.min_oos_trades else "",
    ))
    gates.append(GateResult(
        "oos_expectancy",
        oos.metrics.expectancy_r > criteria.min_expectancy_r,
        required=f"out-of-sample expectancy > {criteria.min_expectancy_r:+.2f}R",
        measured=f"{oos.metrics.expectancy_r:+.3f}R",
        detail="after costs, on data the search did not touch",
    ))
    profit_factor = oos.metrics.profit_factor
    gates.append(GateResult(
        "oos_profit_factor",
        profit_factor is not None and profit_factor >= criteria.min_profit_factor,
        required=f"out-of-sample profit factor >= {criteria.min_profit_factor:.2f}",
        measured=f"{profit_factor:.2f}" if profit_factor is not None else "not computable",
        detail="profit factor below the floor or undefined (no losing trades)",
    ))
    if criteria.require_beats_baseline:
        gates.append(_baseline_comparison(oos, oos_baseline))
    gates.append(GateResult(
        "drawdown_vs_baseline",
        oos.metrics.max_drawdown_pct <= max(oos_baseline.metrics.max_drawdown_pct
                                            * criteria.max_drawdown_vs_baseline,
                                            oos_baseline.metrics.max_drawdown_pct + 0.02),
        required=f"drawdown <= {criteria.max_drawdown_vs_baseline:.2f}x the baseline's and not "
                 "materially worse",
        measured=f"{oos.metrics.max_drawdown_pct:.2%} vs "
                 f"{oos_baseline.metrics.max_drawdown_pct:.2%}",
        detail="a bigger drawdown for a marginal edge is not an improvement",
    ))
    cost_share = oos.metrics.cost_share_of_gross
    gates.append(GateResult(
        "cost_share",
        cost_share is not None and cost_share <= criteria.max_cost_share_of_gross,
        required=f"costs <= {criteria.max_cost_share_of_gross:.0%} of gross profit",
        measured=f"{cost_share:.0%}" if cost_share is not None else "not computable",
        detail="costs should not eat the edge",
    ))
    if criteria.require_cost_stress_pass:
        passed = stress is not None and stress.metrics.expectancy_r > 0
        gates.append(GateResult(
            "cost_stress", passed,
            required="positive expectancy with doubled friction",
            measured=f"{stress.metrics.expectancy_r:+.3f}R" if stress else "not run",
            detail="a strategy that only works on cheap fills is not deployable",
        ))
    if walk is not None:
        gates.append(GateResult(
            "walk_forward_consistency",
            walk.folds_run >= criteria.min_walk_forward_folds
            and walk.positive_share >= criteria.min_walk_forward_positive_share,
            required=f">= {criteria.min_walk_forward_positive_share:.0%} of "
                     f">= {criteria.min_walk_forward_folds} folds positive",
            measured=f"{walk.positive_share:.0%} of {walk.folds_run} folds, "
                     f"mean {walk.mean_expectancy_r:+.3f}R",
            detail=walk.consistency_note,
        ))
    if sensitivity is not None:
        fragile_share = float(sensitivity.get("fragile", 1.0))
        gates.append(GateResult(
            "parameter_sensitivity",
            fragile_share <= criteria.max_fragile_parameter_variants,
            required=f"less than {criteria.max_fragile_parameter_variants:.0%} fragile variants",
            measured=f"{fragile_share:.0%} fragile across "
                     f"{sensitivity.get('variants', 0)} variants",
            detail=f"profitable share {sensitivity.get('profitable_share', 0):.0%}, "
                   f"expectancy spread {sensitivity.get('spread', 0):.4f}",
        ))
    gates.append(GateResult(
        "holdout_budget",
        holdout_uses <= criteria.max_holdout_uses,
        required=f"the untouched test window looked at <= {criteria.max_holdout_uses} times",
        measured=f"{holdout_uses} use(s)",
        detail="repeated inspection turns the holdout into a training set",
    ))

    # Statistical and practical meaning, corrected for the search family.
    trade_pnls = [t.r_multiple for t in oos.trades]
    baseline_pnls = [t.r_multiple for t in oos_baseline.trades]
    test = permutation_test(trade_pnls, baseline_pnls, n_permutations=800, seed=29)
    adjusted = adjusted_p_value(test.p_value, max(1, family_size))
    gates.append(GateResult(
        "statistical_significance",
        adjusted <= criteria.max_p_value,
        required=f"family-adjusted permutation p <= {criteria.max_p_value:.2f} "
                 f"(family of {max(1, family_size)})",
        measured=f"p={adjusted:.4f} (raw {test.p_value:.4f}), effect size "
                 f"{test.effect_size:+.2f}",
        detail="a candidate that is not distinguishable from the baseline is not an improvement",
    ))
    return gates


def gates_passed(gates: Iterable[GateResult], require_blocking: bool = True) -> bool:
    relevant = [g for g in gates if g.blocking or not require_blocking]
    return all(g.passed for g in relevant)


def multiple_testing_summary(p_values: Iterable[float], alpha: float = 0.05) -> dict[str, Any]:
    """Apply FDR control to a batch of experiments and describe what survived."""
    summary = benjamini_hochberg(list(p_values), alpha=alpha)
    summary["note"] = ("experiments are corrected as one family; a single good-looking "
                       "result among many is expected by chance")
    return summary


def cross_validate_windows(table: FeatureTable, windows: list[tuple[str, str]]) -> str:
    """A short fingerprint of which windows were evaluated, for the audit trail."""
    return stable_hash([[str(a), str(b)] for a, b in windows])[:12]


def parameter_sensitivity(spec: StrategySpec, table: FeatureTable, config: EngineConfig | None,
                          grid: dict[str, tuple[float, ...]] | None = None,
                          oos_only: ChronologicalSplit | None = None) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Rerun the strategy across a small declared parameter grid on the test window.

    Sensitivity is measured out-of-sample only. Sweeping parameters over the whole
    sample and then reporting the best combination is the multiple-testing error
    this gate exists to catch.
    """
    target = oos_only.subset(table, "test") if oos_only is not None else table
    variants = parameter_variants(spec, grid)
    detail: list[dict[str, Any]] = []
    for variant in variants:
        run = run_backtest(variant, target, config)
        detail.append({"exits": variant.exits.as_dict(),
                       "expectancy_r": round(run.metrics.expectancy_r, 6),
                       "trades": run.metrics.trades})
    stability = parameter_stability([d["expectancy_r"] for d in detail])
    return stability, detail
