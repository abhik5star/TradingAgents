"""The research loop: observe → hypothesise → test → validate → decide → record.

One pass of :class:`HypothesisLoop` does exactly what the brief describes:

1. **Observe** the current baseline's trades and autopsies, per regime, session and
   timeframe context.
2. **Identify** the largest measured weakness, not the most interesting story.
3. **Form** a hypothesis with a stated reason, evidence, data range, rules
   involved, expected benefit, possible downside, baseline version, experiment
   parameters, split, cost assumptions, acceptance criteria and confidence limits.
4. **Test** the proposed rule change on the training window, **validate** it on the
   untouched test window, walk it forward, stress its costs and sweep its
   parameters.
5. **Decide** accept, reject or continue research, against pre-declared gates.
6. **Record** the outcome — including failures — and retrieve prior rejections
   before proposing the same experiment again.

Two properties make it a research system rather than a story generator:

* the *idea generator* is separated from the *evidence*. The default generator is
  deterministic diagnosis; an LLM can be plugged in as an
  :class:`IdeaProvider` and it can only emit a :class:`RuleChange` that goes
  through the same gates;
* a hypothesis that fails is stored, counted and cited. The loop can — and often
  will — conclude that nothing helped and that the correct output is ``NO TRADE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import pandas as pd

from tradingagents.research.autopsy import autopsy_batch, recurring_weaknesses
from tradingagents.research.contracts import (
    CostModel,
    DeploymentStage,
    ExperimentStatus,
    jsonable,
    stable_hash,
    utc_now,
)
from tradingagents.research.data import MarketData, load_market_data
from tradingagents.research.engine import BacktestResult, EngineConfig, run_backtest
from tradingagents.research.features import FeatureTable, build_feature_table
from tradingagents.research.quant import permutation_test
from tradingagents.research.risk import RiskPolicy
from tradingagents.research.scoring import NO_TRADE, compare_metrics, score_strategy
from tradingagents.research.store import ResearchStore, new_id
from tradingagents.research.strategies import (
    RuleChange,
    StrategySpec,
    apply_rule_change,
    baseline_strategy,
)
from tradingagents.research.validation import (
    AcceptanceCriteria,
    HoldoutLedger,
    acceptance_gates,
    chronological_split,
    gates_passed,
    parameter_sensitivity,
    probe_lookahead,
    walk_forward,
)

# --------------------------------------------------------------------------------------
# Diagnostics: what the evidence says is wrong
# --------------------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """A measured weakness, with the rule change that would address it."""

    kind: str
    title: str
    evidence: list[str]
    severity: float
    change: RuleChange
    downside: str

    @property
    def signature(self) -> str:
        return stable_hash({"kind": self.kind, "change": self.change.as_dict()})[:12]

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "title": self.title, "evidence": self.evidence,
                "severity": round(self.severity, 4), "change": self.change.as_dict(),
                "downside": self.downside, "signature": self.signature}


def _feature_at(trade: Any, table: FeatureTable, column: str) -> float | None:
    if column not in table.features.columns:
        return None
    try:
        stamp = pd.Timestamp(trade.entry_time)
    except (TypeError, ValueError):
        return None
    try:
        row = table.features[column].asof(stamp)
    except (TypeError, ValueError):
        return None
    try:
        value = float(row)
    except (TypeError, ValueError):
        return None
    return value if pd.notna(value) else None


def diagnose(result: BacktestResult, table: FeatureTable, *,
             min_trades: int = 8) -> list[Diagnostic]:
    """Rank measured weaknesses by the losses they account for.

    Each diagnostic carries the smallest rule change that addresses it. Regime and
    session filters are built from what was measured positive, so the change is a
    statement about this data rather than a preference.
    """
    diagnostics: list[Diagnostic] = []
    trades = result.trades
    losses = [t for t in trades if t.net_pnl < 0]
    if not trades:
        return diagnostics

    # 1. A regime that keeps losing.
    regimes = {name: stats for name, stats in (result.metrics.by_regime or {}).items()
               if stats["trades"] >= min_trades}
    negative = {name: stats for name, stats in regimes.items() if stats["expectancy"] < 0}
    if negative:
        worst = min(negative.items(), key=lambda item: item[1]["expectancy"] * item[1]["trades"])
        keep = tuple(sorted(name for name, stats in regimes.items() if stats["expectancy"] > 0))
        if keep:
            diagnostics.append(Diagnostic(
                kind="regime_filter",
                title=f"Entries in the {worst[0]} regime lose money",
                evidence=[f"{int(worst[1]['trades'])} trades, expectancy "
                          f"{worst[1]['expectancy']:+.2f}, net {worst[1]['net_pnl']:+.0f}",
                          f"positive regimes: {', '.join(keep)}"],
                severity=abs(worst[1]["expectancy"] * worst[1]["trades"]),
                change=RuleChange("add_regime_filter", {"regimes": list(keep)},
                                  rationale=f"stop trading the {worst[0]} regime"),
                downside=("the filter removes trades that may be profitable in the future; "
                          "it also reduces the sample available to validate the strategy"),
            ))

    # 2. A session that keeps losing.
    sessions = {name: stats for name, stats in (result.metrics.by_session or {}).items()
                if stats["trades"] >= min_trades}
    negative_sessions = {name: stats for name, stats in sessions.items()
                         if stats["expectancy"] < 0}
    if negative_sessions:
        worst = min(negative_sessions.items(),
                    key=lambda item: item[1]["expectancy"] * item[1]["trades"])
        keep = tuple(float({"asia": 0, "london": 1, "london_ny_overlap": 2,
                            "new_york": 3, "off_hours": 4}.get(name, 4))
                     for name, stats in sessions.items() if stats["expectancy"] > 0)
        if keep:
            diagnostics.append(Diagnostic(
                kind="session_filter",
                title=f"Trades in the {worst[0]} session lose money",
                evidence=[f"{int(worst[1]['trades'])} trades, expectancy "
                          f"{worst[1]['expectancy']:+.2f}"],
                severity=abs(worst[1]["expectancy"] * worst[1]["trades"]),
                change=RuleChange("add_session_filter", {"sessions": list(keep)},
                                  rationale=f"avoid the {worst[0]} session"),
                downside="session filters cut trade count and can overfit to one sample period",
            ))

    # 3. Entries taken while the higher timeframe disagreed.
    if "mtf_conflict" in table.features.columns and losses:
        conflict_losses = sum(1 for t in losses if _feature_at(t, table, "mtf_conflict") == 1.0)
        share = conflict_losses / len(losses)
        if share >= 0.4:
            diagnostics.append(Diagnostic(
                kind="timeframe_conflict",
                title="Most losing trades were taken while the timeframes disagreed",
                evidence=[f"{conflict_losses}/{len(losses)} losing trades had "
                          "mtf_conflict set at entry"],
                severity=float(conflict_losses),
                change=RuleChange("require_htf_alignment", {"direction": "up"},
                                  rationale="require higher-timeframe agreement before entry"),
                downside=("alignment cuts many trades and lags turning points; some of the "
                          "removed trades will be winners"),
            ))

    # 4. Low-ADX entries: the example from the brief.
    if "regime_adx" in table.features.columns and losses:
        low_adx = [t for t in losses if (_feature_at(t, table, "regime_adx") or 100) < 20]
        share = len(low_adx) / len(losses)
        if share >= 0.35:
            diagnostics.append(Diagnostic(
                kind="low_adx",
                title="Losing trades cluster in low-ADX conditions",
                evidence=[f"{len(low_adx)}/{len(losses)} losing trades entered with ADX below 20"],
                severity=float(len(low_adx)),
                change=RuleChange("add_filter",
                                  {"condition": {"left": "regime_adx", "op": ">=",
                                                 "right": 20.0, "label": "ADX at least 20"}},
                                  rationale="avoid trend entries in a directionless market"),
                downside="a hard ADX floor misses the early part of new trends",
            ))

    # 5. Profitable excursions given back.
    if losses:
        gave_back = [t for t in losses if t.mfe_r >= 1.0]
        share = len(gave_back) / len(losses)
        if share >= 0.3:
            diagnostics.append(Diagnostic(
                kind="give_back",
                title="Losing trades had at least 1R of favourable movement before failing",
                evidence=[f"{len(gave_back)}/{len(losses)} losers reached +1R first",
                          f"median mfe {pd.Series([t.mfe_r for t in gave_back]).median():.2f}R"],
                severity=float(len(gave_back)) * 2,
                change=RuleChange("add_trailing_stop", {"trailing_atr": 1.0,
                                                        "trail_after_r": 1.0},
                                  rationale="protect gains once the trade has moved 1R"),
                downside=("a trailing stop exits earlier in ordinary pullbacks and can "
                          "reduce expectancy more than it saves"),
            ))

    # 6. Entries with no follow-through at all.
    if losses:
        no_follow = [t for t in losses if t.mfe_r < 0.25 and t.mae_r >= 0.8]
        if len(no_follow) / len(losses) >= 0.3:
            diagnostics.append(Diagnostic(
                kind="premature_entry",
                title="A large share of losses moved against the entry immediately",
                evidence=[f"{len(no_follow)}/{len(losses)} losers never reached +0.25R"],
                severity=float(len(no_follow)) * 1.5,
                change=RuleChange("add_filter",
                                  {"condition": {"left": "close_location", "op": ">=",
                                                 "right": 0.5,
                                                 "label": "close in the upper half of its bar"}},
                                  rationale="buy only when the bar closes strong"),
                downside="confirmation conditions reduce trade count and add entry lag",
            ))

    # 7. Friction dominating the result.
    share = result.metrics.cost_share_of_gross or 0.0
    if share >= 0.4:
        diagnostics.append(Diagnostic(
            kind="costs",
            title="Transaction costs consume a large share of gross profit",
            evidence=[f"costs are {share:.0%} of gross profit",
                      f"total cost {result.metrics.total_cost:+.0f}"],
            severity=share * 100,
            change=RuleChange("reduce_risk", {"factor": 0.5},
                              rationale="halve exposure while the edge is inside the friction"),
            downside="smaller positions do not fix a cost problem; this only limits the damage",
        ))
    return sorted(diagnostics, key=lambda d: -d.severity)


# --------------------------------------------------------------------------------------
# Idea providers
# --------------------------------------------------------------------------------------


class IdeaProvider(Protocol):
    """Anything that can turn diagnostics into proposed rule changes."""

    name: str

    def propose(self, diagnostics: list[Diagnostic], history: list[dict[str, Any]]
                ) -> list[tuple[Diagnostic, str]]:
        """Return ``(diagnostic, reconsider_reason)`` pairs worth testing next."""
        ...


class DiagnosticIdeaProvider:
    """The default: propose the measured weaknesses, most costly first.

    A previously rejected diagnostic is skipped unless a reconsideration reason is
    supplied, so the loop does not re-run the same failed experiment forever.
    """

    name = "diagnostic"

    def propose(self, diagnostics: list[Diagnostic], history: list[dict[str, Any]]
                ) -> list[tuple[Diagnostic, str]]:
        rejected = {record.get("signature") for record in history
                    if record.get("decision") == "rejected"}
        proposals: list[tuple[Diagnostic, str]] = []
        for diagnostic in diagnostics:
            if diagnostic.signature in rejected:
                continue
            proposals.append((diagnostic, ""))
        return proposals


class LLMIdeaProvider:
    """Adapter for an LLM that proposes *ideas only*.

    ``complete`` is a callable taking a prompt and returning text that parses as a
    JSON list of ``{"kind": ..., "params": {...}, "rationale": ...}`` objects. Anything
    that is not a declared :class:`RuleChange` kind is discarded. The provider
    cannot run an experiment, alter the risk policy, approve a version or promote
    anything: it produces candidates for the same gates as every other idea.
    """

    name = "llm"

    def __init__(self, complete, diagnostics_hint: bool = True):
        self.complete = complete
        self.diagnostics_hint = diagnostics_hint

    def propose(self, diagnostics: list[Diagnostic], history: list[dict[str, Any]]
                ) -> list[tuple[Diagnostic, str]]:
        import json

        payload = {
            "diagnostics": [d.as_dict() for d in diagnostics[:5]],
            "rejected": [{"title": record.get("title"), "reason": record.get("explanation")}
                         for record in history if record.get("decision") == "rejected"][:10],
            "allowed_kinds": list(RuleChange.KINDS),
        }
        prompt = (
            "You are proposing ONE testable rule change for a trading strategy research "
            "engine. Reply with a JSON list of objects with keys kind, params, rationale. "
            "Only use the allowed kinds. Do not claim results; the engine will test the "
            "idea and may reject it.\n\n" + json.dumps(payload, indent=2, default=str))
        try:
            text = self.complete(prompt)
            ideas = json.loads(text)
        except Exception:  # a broken idea generator must not stop research
            return []
        proposals: list[tuple[Diagnostic, str]] = []
        for idea in ideas if isinstance(ideas, list) else []:
            try:
                change = RuleChange(kind=idea["kind"], params=idea.get("params", {}),
                                    rationale=str(idea.get("rationale", ""))[:400])
            except (KeyError, ValueError, TypeError):
                continue
            diagnostic = Diagnostic(
                kind=f"llm_{change.kind}", title=change.rationale or change.describe(),
                evidence=["proposed by an LLM from the diagnostics above; not evidence"],
                severity=0.0, change=change,
                downside="the idea is unvalidated until the engine's gates say otherwise")
            proposals.append((diagnostic, "LLM proposal; still subject to every gate"))
        return proposals


# --------------------------------------------------------------------------------------
# Hypothesis record
# --------------------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """A complete, auditable experiment specification and its result."""

    hypothesis_id: str
    ticker: str
    title: str
    reason: str
    evidence: list[str]
    data_range: str
    features: list[str]
    rules_involved: list[str]
    expected_benefit: str
    possible_downside: str
    baseline_version: str
    experiment_parameters: dict[str, Any]
    split: dict[str, Any]
    cost_assumptions: dict[str, Any]
    acceptance_criteria: dict[str, Any]
    status: str = ExperimentStatus.RESEARCH.value
    signature: str = ""
    change: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    decision: str = "pending"
    explanation: str = ""
    confidence: str = ""
    candidate_version: str | None = None
    reconsidered_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__.copy())


def _expected_benefit(change: RuleChange) -> str:
    return {
        "add_regime_filter": "fewer losing trades in the regime that does not pay",
        "add_session_filter": "avoid the hours where the edge is negative",
        "require_htf_alignment": "higher-quality entries with less immediate adverse movement",
        "add_trailing_stop": "keep more of the favourable excursion already reached",
        "add_filter": "exclude the trades the evidence shows losing",
        "add_confirmation": "delay entry until confirmation, reducing premature entries",
        "reduce_risk": "limit the damage while the edge is inside transaction costs",
    }.get(change.kind, "improve risk-adjusted expectancy without increasing risk")


def _confidence(accepted: bool, gates: list[Any], trades: int) -> str:
    if not accepted:
        return ("the evidence is sufficient to reject this change for this sample; it is not "
                "proof the idea can never work")
    failed = [g.name for g in gates if not g.passed]
    base = ("accepted on out-of-sample evidence with a clean walk-forward"
            if not failed else f"accepted with caveats: {', '.join(failed)}")
    return f"{base}; {trades} out-of-sample trades is a limited sample and not a guarantee"


# --------------------------------------------------------------------------------------
# Paper trading
# --------------------------------------------------------------------------------------


@dataclass
class PaperRecord:
    """The paper-trading stage of a candidate: forward data, no capital at risk."""

    status: str
    trades: int = 0
    expectancy_r: float = 0.0
    net_pnl: float = 0.0
    started: str = ""
    ended: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__.copy())


def paper_trade(spec: StrategySpec, table: FeatureTable, after: pd.Timestamp,
                config: EngineConfig, *, min_bars: int = 30,
                min_trades: int = 5) -> PaperRecord:
    """Trade the candidate forward, on bars after the validation window.

    If the loaded data does not extend past the test window there is nothing to
    paper trade yet, and the record says so. It does not pretend a backtest was
    paper trading.
    """
    forward = table.bars.loc[table.bars.index > after]
    if len(forward) < min_bars:
        return PaperRecord(status="pending", started=str(after.date()),
                           note=(f"only {len(forward)} bars after the test window; re-run once "
                                 "more data has printed"))
    result = run_backtest(spec, table.slice(str(forward.index[0]), str(forward.index[-1])),
                          config)
    passed = result.metrics.trades >= min_trades and result.metrics.expectancy_r > 0
    return PaperRecord(
        status="passed" if passed else "failed",
        trades=result.metrics.trades,
        expectancy_r=round(result.metrics.expectancy_r, 6),
        net_pnl=round(result.metrics.net_pnl, 4),
        started=str(forward.index[0].date()), ended=str(forward.index[-1].date()),
        note=("paper trading supported the candidate" if passed else
              "paper trading did not support the candidate: fewer than the required trades "
              "or negative expectancy"),
    )


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


@dataclass
class LoopPlan:
    """Everything a research session declares before it looks at a result."""

    ticker: str
    timeframes: list[str] = field(default_factory=lambda: ["1d"])
    start: str = ""
    end: str = ""
    asset_type: str = "stock"
    initial_equity: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel)
    risk_policy: RiskPolicy = field(default_factory=RiskPolicy)
    criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    train_fraction: float = 0.5
    validation_fraction: float = 0.25
    purge_bars: int = 5
    embargo_bars: int = 2
    walk_forward_folds: int = 4
    max_hypotheses: int = 3
    seed: int = 7

    def as_dict(self) -> dict[str, Any]:
        return jsonable({"ticker": self.ticker, "timeframes": self.timeframes,
                         "start": self.start, "end": self.end,
                         "asset_type": self.asset_type, "costs": self.costs.as_dict(),
                         "criteria": self.criteria.as_dict(),
                         "train_fraction": self.train_fraction,
                         "validation_fraction": self.validation_fraction,
                         "purge_bars": self.purge_bars, "embargo_bars": self.embargo_bars,
                         "max_hypotheses": self.max_hypotheses})


@dataclass
class LoopSummary:
    """What one research session did, including what it refused to accept."""

    ticker: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    versions: list[str] = field(default_factory=list)
    champion: str | None = None
    no_trade_reason: str = ""
    data_quality: dict[str, Any] = field(default_factory=dict)
    leakage: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    best: dict[str, Any] | None = None
    weaknesses: list[dict[str, Any]] = field(default_factory=list)
    drift_alert: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return sum(1 for h in self.hypotheses if h.decision == "accepted")

    @property
    def rejected(self) -> int:
        return sum(1 for h in self.hypotheses if h.decision == "rejected")

    def as_dict(self) -> dict[str, Any]:
        return jsonable({
            "ticker": self.ticker,
            "hypotheses": [h.as_dict() for h in self.hypotheses],
            "versions": self.versions, "champion": self.champion,
            "no_trade_reason": self.no_trade_reason,
            "data_quality": self.data_quality, "leakage": self.leakage,
            "baseline": self.baseline, "best": self.best,
            "weaknesses": self.weaknesses, "drift_alert": self.drift_alert,
            "notes": self.notes,
            "accepted": self.accepted, "rejected": self.rejected,
        })


class HypothesisLoop:
    """The continuous research loop, one session at a time."""

    def __init__(self, store: ResearchStore, registry: Any, plan: LoopPlan,
                 provider: IdeaProvider | None = None,
                 table: FeatureTable | None = None,
                 data: MarketData | None = None,
                 config: EngineConfig | None = None):
        self.store = store
        self.registry = registry
        self.plan = plan
        self.provider = provider or DiagnosticIdeaProvider()
        self.table = table
        self.data = data
        self.config = config or EngineConfig(
            initial_equity=plan.initial_equity, costs=plan.costs,
            risk_policy=plan.risk_policy)
        self.holdouts = HoldoutLedger(store)

    # ---------- prepare ----------

    def _load(self) -> FeatureTable:
        if self.table is not None:
            return self.table
        if not self.plan.start or not self.plan.end:
            raise ValueError("a research plan needs a start and end date, or a prepared table")
        self.data = load_market_data(self.plan.ticker, self.plan.timeframes,
                                     self.plan.start, self.plan.end,
                                     asset_type=self.plan.asset_type)
        self.table = build_feature_table(self.data, self.plan.timeframes[0],
                                         self.plan.timeframes[1] if len(self.plan.timeframes) > 1
                                         else None,
                                         self.plan.timeframes[2] if len(self.plan.timeframes) > 2
                                         else None)
        return self.table

    # ---------- the loop ----------

    def run(self) -> LoopSummary:
        """Observe, hypothesise, test, decide, record. Never promotes a version to live."""
        summary = LoopSummary(ticker=self.plan.ticker)
        table = self._load()
        summary.data_quality = {tf: report.as_dict() for tf, report in table.data.quality.items()}
        data_ok = table.data.ok
        if not data_ok:
            reason = "data quality gate failed: " + "; ".join(table.data.blocking_issues())
            summary.no_trade_reason = reason
            summary.notes.append(reason)
            self.store.append("risk_event", {"id": new_id("RSK"), "kind": "research_blocked",
                                             "reason": reason})
            return summary

        leakage = probe_lookahead(table.data, table.timeframe, table.context_timeframe,
                                  table.confirm_timeframe, probes=3)
        summary.leakage = leakage.as_dict()
        if not leakage.ok:
            summary.no_trade_reason = ("look-ahead detected in the feature table; results "
                                       "cannot be trusted until it is fixed")
            summary.notes.append(summary.no_trade_reason)
            return summary

        split = chronological_split(table.bars.index, self.plan.train_fraction,
                                    self.plan.validation_fraction, self.plan.purge_bars,
                                    self.plan.embargo_bars)

        # The champion is the baseline; seed V01 on the first session. It records the
        # same split as every candidate, so the baseline can be paper traded forward
        # from the end of its own out-of-sample window like any other version.
        champion = self.registry.champion_version()
        if champion is None:
            champion = self.registry.register(
                baseline_strategy(table.timeframe, table.context_timeframe),
                status=ExperimentStatus.PAPER_TESTING,
                stage=DeploymentStage.AWAITING_APPROVAL,
                dataset={**table.summary(), "split": split.as_dict()},
                provenance={"seeded_as_baseline": True})
            self.store.append("deployment", {"id": new_id("DEP"), "kind": "baseline_seeded",
                                             "version": champion.version})
        baseline_spec = champion.spec
        baseline_spec.validate_columns(table.features.columns)
        train_val = table.slice(str(split.train[0]), str(split.validation[1]))
        test_table = split.subset(table, "test")

        baseline_run = run_backtest(baseline_spec, train_val, self.config)
        baseline_oos = run_backtest(baseline_spec, test_table, self.config)
        summary.baseline = {"version": champion.version,
                            "in_sample": baseline_run.metrics.as_dict(),
                            "out_of_sample": baseline_oos.metrics.as_dict(),
                            "blocked_entries": len(baseline_run.blocked)}
        autopsies = autopsy_batch(baseline_run.trades, policy=self.plan.risk_policy)
        summary.weaknesses = recurring_weaknesses(autopsies)
        diagnostics = diagnose(baseline_run, train_val)

        history = [record for record in self.store.read("hypothesis")
                   if record.get("ticker") == self.plan.ticker]
        proposals = self.provider.propose(diagnostics, history)
        if not proposals:
            summary.no_trade_reason = self._no_trade_reason(
                "no measured weakness remains to test, and the baseline has not earned "
                "promotion on its own evidence", baseline_oos, table)
            self.store.append("experiment", {
                "id": new_id("EXP"), "ticker": self.plan.ticker, "kind": "no_hypothesis",
                "baseline": champion.version, "reason": summary.no_trade_reason,
                "accepted": False})
            summary.champion = champion.version
            return summary

        scores: dict[str, Any] = {}
        family = len([record for record in self.store.read("experiment")
                      if record.get("ticker") == self.plan.ticker])
        for diagnostic, reconsider in proposals[: self.plan.max_hypotheses]:
            hypothesis = self._build_hypothesis(diagnostic, reconsider, champion, table, split)
            self.store.append("hypothesis", hypothesis.as_dict())
            try:
                candidate, outcome = self._run_experiment(
                    hypothesis, baseline_spec, table, split, train_val, test_table,
                    baseline_oos, family, baseline_run)
            except Exception as exc:  # a broken experiment is recorded, not hidden
                hypothesis.status = ExperimentStatus.REJECTED.value
                hypothesis.decision = "rejected"
                hypothesis.explanation = f"the experiment could not be run: {exc}"
                hypothesis.finished_at = utc_now()
                self.store.append("hypothesis", hypothesis.as_dict())
                self.store.error("hypothesis_loop", str(exc),
                                 hypothesis_id=hypothesis.hypothesis_id)
                summary.hypotheses.append(hypothesis)
                family += 1
                continue
            family += 1
            hypothesis.results = outcome["results"]
            hypothesis.decision = outcome["decision"]
            hypothesis.explanation = outcome["explanation"]
            hypothesis.confidence = outcome["confidence"]
            hypothesis.status = outcome["status"]
            hypothesis.candidate_version = outcome["version"]
            hypothesis.finished_at = utc_now()
            self.store.append("hypothesis", hypothesis.as_dict())
            summary.hypotheses.append(hypothesis)
            if outcome["version"]:
                summary.versions.append(outcome["version"])
            if outcome["decision"] == "accepted":
                scores[outcome["version"]] = outcome["score"]
            self.store.append("experiment", {
                "id": hypothesis.hypothesis_id.replace("H-", "EXP-"),
                "ticker": self.plan.ticker, "hypothesis_id": hypothesis.hypothesis_id,
                "candidate_version": outcome["version"],
                "baseline_version": champion.version,
                "decision": outcome["decision"],
                "gates": outcome["gates"], "score": outcome["score"].as_dict(),
                "results": outcome["results"], "accepted": outcome["decision"] == "accepted",
            })

        current = self.registry.champion_version()
        summary.champion = current.version if current else None
        summary.best = self._best(summary)
        if not summary.versions:
            summary.no_trade_reason = self._no_trade_reason(
                "every proposed change was rejected; the baseline does not clear the gates "
                "on this sample either", baseline_oos, table)
        return summary

    # ---------- one experiment ----------

    def _build_hypothesis(self, diagnostic: Diagnostic, reconsider: str, champion: Any,
                          table: FeatureTable, split: Any) -> Hypothesis:
        features = sorted({*(c.left for c in champion.spec.long_entry),
                           *(c.left for c in champion.spec.short_entry)})
        signature = diagnostic.signature
        return Hypothesis(
            hypothesis_id=new_id("H"),
            ticker=table.ticker,
            title=diagnostic.title,
            reason=(f"{diagnostic.title}. The baseline {champion.version} was measured on "
                    f"{split.train[0].date()}..{split.validation[1].date()}; the weakness "
                    "accounts for the largest share of measured losses."),
            evidence=diagnostic.evidence,
            data_range=f"{split.train[0].date()}..{split.test[1].date()} "
                       f"({table.ticker} {table.timeframe})",
            features=features,
            rules_involved=[f"{champion.version}: " + c.label for c in
                            (*champion.spec.long_entry, *champion.spec.short_entry)],
            expected_benefit=_expected_benefit(diagnostic.change),
            possible_downside=diagnostic.downside,
            baseline_version=champion.version,
            experiment_parameters={"change": diagnostic.change.as_dict(),
                                   "engine": {"initial_equity": self.config.initial_equity,
                                              "max_bars_in_trade": self.config.max_bars_in_trade}},
            split=split.as_dict(),
            cost_assumptions=self.config.costs.as_dict(),
            acceptance_criteria=self.plan.criteria.as_dict(),
            signature=signature,
            change=diagnostic.change.as_dict(),
            reconsidered_reason=reconsider,
        )

    def _run_experiment(self, hypothesis: Hypothesis, baseline_spec: StrategySpec,
                        table: FeatureTable, split: Any, train_val: FeatureTable,
                        test_table: FeatureTable, baseline_oos: BacktestResult,
                        family: int, baseline_run: BacktestResult) -> tuple[StrategySpec, dict]:
        change = RuleChange(hypothesis.change["kind"], hypothesis.change.get("params", {}),
                            hypothesis.change.get("rationale", ""))
        candidate = apply_rule_change(baseline_spec, change)
        version = self.registry.next_version_name()
        candidate = candidate.with_version(version, name=f"{baseline_spec.name} + {change.kind}",
                                           description=hypothesis.reason)
        candidate.validate_columns(test_table.features.columns)
        hypothesis.status = ExperimentStatus.BACKTESTING.value

        in_sample = run_backtest(candidate, train_val, self.config)
        hypothesis.status = ExperimentStatus.VALIDATION.value
        oos = run_backtest(candidate, test_table, self.config)
        stress = run_backtest(candidate, test_table,
                              replace(self.config, costs=self.config.costs.stressed(2.0)))
        try:
            walk = walk_forward(candidate, train_val, self.config, folds=self.plan.walk_forward_folds)
        except ValueError:
            walk = None
        sensitivity, sensitivity_detail = parameter_sensitivity(
            candidate, table, self.config, oos_only=split)
        uses, within_budget = self.holdouts.record(
            table.version, f"{split.test[0]}..{split.test[1]}", hypothesis.hypothesis_id,
            self.plan.criteria.max_holdout_uses)

        trade_test = permutation_test([t.r_multiple for t in oos.trades],
                                      [t.r_multiple for t in baseline_oos.trades],
                                      n_permutations=600, seed=self.plan.seed)
        gates = acceptance_gates(
            in_sample, baseline_run, oos=oos, oos_baseline=baseline_oos, stress=stress,
            walk=walk, sensitivity=sensitivity, criteria=self.plan.criteria,
            family_size=max(1, family), holdout_uses=uses, data_quality_ok=table.data.ok,
        )
        accepted = gates_passed(gates) and within_budget
        score = score_strategy(
            oos.metrics, oos_metrics=oos.metrics,
            walk_forward_positive_share=walk.positive_share if walk else None,
            sensitivity=sensitivity, minimum_trades=self.plan.criteria.min_trades,
            target_trades=self.plan.criteria.target_trades)

        paper = paper_trade(candidate, table, split.test[1], self.config)

        results = {
            "in_sample": in_sample.as_dict(), "out_of_sample": oos.as_dict(),
            "baseline_out_of_sample": baseline_oos.as_dict(),
            "comparison": compare_metrics(baseline_oos.metrics, oos.metrics),
            "cost_stress": stress.metrics.as_dict() if stress else None,
            "walk_forward": walk.as_dict() if walk else None,
            "sensitivity": sensitivity, "sensitivity_detail": sensitivity_detail,
            "permutation": trade_test.as_dict(),
            "gates": [g.as_dict() for g in gates],
            "holdout_uses": uses, "holdout_within_budget": within_budget,
            "paper": paper.as_dict(),
            "score": score.as_dict(),
            "entry_counts": {"signals": oos.signals, "entries": oos.entries,
                             "blocked": len(oos.blocked)},
        }
        status = ExperimentStatus.PAPER_TESTING if accepted else ExperimentStatus.REJECTED
        stage = (DeploymentStage.PAPER_TESTED if paper.status == "passed"
                 else DeploymentStage.ROBUSTNESS if accepted
                 else DeploymentStage.REJECTED)
        self.registry.register(
            candidate, experiment_id=hypothesis.hypothesis_id,
            parent_version=baseline_spec.version,
            status=status, stage=stage,
            dataset={**table.summary(), "split": split.as_dict()},
            backtest=in_sample.as_dict(), out_of_sample=oos.as_dict(),
            walk_forward=(walk.as_dict() if walk else {}),
            changes_from_parent=[f"{change.kind}: {change.describe()}"],
            provenance={"hypothesis_id": hypothesis.hypothesis_id,
                        "diagnostic": hypothesis.signature,
                        "paper": paper.as_dict()},
        )
        if accepted:
            self.registry.update(version, paper=paper.as_dict())
        self.store.write_artifact(hypothesis.hypothesis_id, "trades.json", oos.trades_frame())
        self.store.write_artifact(hypothesis.hypothesis_id, "equity.json",
                                  {str(k): float(v) for k, v in oos.equity.items()})
        self.store.write_artifact(hypothesis.hypothesis_id, "metrics.json", results)

        failed = [g.name for g in gates if not g.passed]
        if accepted:
            explanation = (
                f"accepted for paper trading: the candidate improved out-of-sample expectancy "
                f"from {baseline_oos.metrics.expectancy_r:+.3f}R to {oos.metrics.expectancy_r:+.3f}R "
                f"with all gates passing. Paper trading is {paper.status}. Promotion to champion "
                "still requires explicit approval.")
        else:
            explanation = (
                "rejected: " + "; ".join(f"{g.name} ({g.measured})" for g in gates if not g.passed)
                + ("" if within_budget else "; the holdout budget was exhausted")) if failed else (
                "rejected: the candidate cleared the individual gates but failed the holdout "
                "budget")
        return candidate, {
            "results": results, "decision": "accepted" if accepted else "rejected",
            "explanation": explanation, "status": status.value,
            "confidence": _confidence(accepted, gates, oos.metrics.trades),
            "version": version if accepted else None, "gates": [g.as_dict() for g in gates],
            "score": score,
        }

    # ---------- reporting ----------

    def _best(self, summary: LoopSummary) -> dict[str, Any] | None:
        accepted = [h for h in summary.hypotheses if h.decision == "accepted"]
        if not accepted:
            return None
        best = max(accepted, key=lambda h: (h.results.get("score") or {}).get("total", 0))
        return {"hypothesis_id": best.hypothesis_id, "version": best.candidate_version,
                "title": best.title,
                "score": (best.results.get("score") or {}).get("total"),
                "oos_expectancy_r": ((best.results.get("out_of_sample") or {})
                                     .get("metrics") or {}).get("expectancy_r")}

    def _no_trade_reason(self, reason: str, oos: BacktestResult, table: FeatureTable) -> str:
        baseline_score = score_strategy(oos.metrics, oos_metrics=oos.metrics)
        detail = (f"baseline {oos.spec.version} out-of-sample expectancy "
                  f"{oos.metrics.expectancy_r:+.3f}R over {oos.metrics.trades} trades, "
                  f"composite score {baseline_score.total:.1f}")
        if table.data.warnings():
            detail += "; data warnings: " + "; ".join(table.data.warnings())
        return f"{NO_TRADE}: {reason}. {detail}."


def run_hypothesis_loop(plan: LoopPlan, *, store: ResearchStore | None = None,
                        registry: Any = None, provider: IdeaProvider | None = None,
                        table: FeatureTable | None = None,
                        data: MarketData | None = None,
                        config: EngineConfig | None = None) -> LoopSummary:
    """Convenience entry point: build the store and registry if not supplied."""
    from tradingagents.research.versions import StrategyRegistry

    store = store or ResearchStore()
    registry = registry or StrategyRegistry(store)
    loop = HypothesisLoop(store, registry, plan, provider=provider, table=table, data=data,
                          config=config)
    return loop.run()


def recent_hypotheses(store: ResearchStore, ticker: str | None = None,
                      limit: int = 20) -> list[dict[str, Any]]:
    """The newest hypothesis state per idea, newest first, for the dashboard and CLI.

    A hypothesis is appended twice — when it is opened and when it is decided — so
    the append-only log holds both events. The dashboard wants one row per idea,
    resolved to its latest state, while the log keeps the full history.
    """
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in store.read("hypothesis"):
        key = str(record.get("hypothesis_id") or record.get("id") or len(order))
        if key not in latest:
            order.append(key)
        latest[key] = record
    records = [latest[key] for key in order]
    if ticker:
        records = [r for r in records if r.get("ticker") == ticker]
    return list(reversed(records))[:limit]
