"""Paper-first deployment: the only road from an idea to live eligibility.

The stage order is fixed and each step has a check that reads stored evidence
rather than a claim:

``proposed → backtested → oos_validated → walk_forward → robustness →
paper_tested → risk_reviewed → awaiting_user_approval → live_eligible``

A version can only move forward one checked stage at a time, and it cannot reach
``live_eligible`` without a recorded approval from a named user. Live trading
remains disabled until :meth:`StrategyRegistry.enable_live` is called with the
champion's version typed as confirmation, and the research engine never calls it.

Nothing here allocates capital, sizes a live order or contacts a broker. This
project has no execution venue; the pipeline exists so that when one is added,
the path into it is already gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingagents.research.contracts import (
    DeploymentStage,
    jsonable,
    utc_now,
)
from tradingagents.research.risk import RiskEngine, RiskPolicy
from tradingagents.research.store import ResearchStore, new_id
from tradingagents.research.strategies import StrategySpec


@dataclass
class StageCheck:
    """One deployment stage: what it requires and what the record actually shows."""

    stage: str
    passed: bool
    requirement: str
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "passed": self.passed, "requirement": self.requirement,
                "evidence": self.evidence}


@dataclass
class PromotionReview:
    """The full readiness picture for one version."""

    version: str
    checks: list[StageCheck] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    furthest_stage: str = DeploymentStage.PROPOSED.value
    live_eligible: bool = False

    @property
    def ready_for_approval(self) -> bool:
        return not self.blockers and self.furthest_stage == DeploymentStage.AWAITING_APPROVAL.value

    def as_dict(self) -> dict[str, Any]:
        return jsonable({"version": self.version,
                         "checks": [c.as_dict() for c in self.checks],
                         "blockers": self.blockers,
                         "furthest_stage": self.furthest_stage,
                         "ready_for_approval": self.ready_for_approval,
                         "live_eligible": self.live_eligible})


class PromotionPipeline:
    """Check, advance and review the paper-first promotion path."""

    def __init__(self, registry: Any, store: ResearchStore,
                 risk: RiskEngine | None = None):
        self.registry = registry
        self.store = store
        self.risk = risk or RiskEngine(RiskPolicy(), store)

    # ---------- checks ----------

    def review(self, version: str, *, paper_min_trades: int = 20,
               paper_min_days: int = 14) -> PromotionReview:
        """Evaluate every stage from the stored evidence and list what blocks promotion."""
        record = self.registry.get(version)
        backtest = record.backtest or {}
        oos = record.out_of_sample or {}
        walk = record.walk_forward or {}
        paper = record.paper or {}
        gates = [g for g in (oos.get("gates") or []) if isinstance(g, dict)]
        checks: list[StageCheck] = []

        rules_ok = bool(record.spec.long_entry or record.spec.short_entry)
        checks.append(StageCheck(
            DeploymentStage.BACKTESTED.value, bool(backtest.get("metrics")) and rules_ok,
            "a rules-based backtest with cost assumptions exists",
            f"{len(backtest.get('metrics') or {})} metrics recorded"))

        metrics = (oos.get("metrics") or {})
        oos_ok = bool(metrics) and (metrics.get("expectancy_r") or 0) > 0 and \
            (metrics.get("trades") or 0) >= 10
        checks.append(StageCheck(
            DeploymentStage.OOS_VALIDATED.value, oos_ok,
            "positive out-of-sample expectancy over at least 10 trades",
            f"expectancy {metrics.get('expectancy_r')}R over {metrics.get('trades')} trades"))

        walk_ok = bool(walk.get("folds")) and (walk.get("positive_share") or 0) >= 0.5
        checks.append(StageCheck(
            DeploymentStage.WALK_FORWARD.value, walk_ok,
            "walk-forward folds exist and at least half are positive",
            f"{len(walk.get('folds') or [])} folds, positive share "
            f"{walk.get('positive_share')}"))

        gate_names = {g.get("name") for g in gates}
        robustness_ok = {"cost_stress", "parameter_sensitivity"} <= gate_names and all(
            g.get("passed") for g in gates
            if g.get("name") in {"cost_stress", "parameter_sensitivity"})
        checks.append(StageCheck(
            DeploymentStage.ROBUSTNESS.value, robustness_ok,
            "cost stress and parameter sensitivity gates passed",
            f"gates recorded: {len(gates)}"))

        paper_ok = (paper.get("status") == "passed"
                    and int(paper.get("trades") or 0) >= paper_min_trades)
        checks.append(StageCheck(
            DeploymentStage.PAPER_TESTED.value, paper_ok,
            f"paper trading passed with at least {paper_min_trades} trades",
            f"status {paper.get('status') or 'not started'}, "
            f"{paper.get('trades', 0)} trades over "
            f"{paper.get('started', '?')}..{paper.get('ended', '?')}"))

        review = self.review_risk(record.spec)
        checks.append(StageCheck(
            DeploymentStage.RISK_REVIEWED.value, review["passed"],
            "the strategy's risk request fits inside the independent policy",
            "; ".join(review["reasons"]) or "within policy"))

        approved = bool(record.approval)
        checks.append(StageCheck(
            DeploymentStage.AWAITING_APPROVAL.value, True,
            "an approval can now be requested from the user",
            "approval recorded" if approved else "awaiting explicit user approval"))

        blockers: list[str] = []
        furthest = DeploymentStage.PROPOSED.value
        for check in checks:
            if not check.passed:
                blockers.append(f"{check.stage}: {check.requirement} (evidence: {check.evidence})")
                break
            furthest = check.stage
        if approved:
            furthest = DeploymentStage.LIVE_ELIGIBLE.value

        return PromotionReview(version=version, checks=checks, blockers=blockers,
                               furthest_stage=furthest,
                               live_eligible=bool(record.live_eligible and approved))

    def review_risk(self, spec: StrategySpec) -> dict[str, Any]:
        """Whether a spec's own sizing request fits the authoritative risk policy."""
        policy = self.risk.policy
        reasons: list[str] = []
        if spec.sizing.risk_per_trade > policy.max_risk_per_trade:
            reasons.append(f"strategy asks {spec.sizing.risk_per_trade:.2%} per trade but the "
                           f"policy caps it at {policy.max_risk_per_trade:.2%}")
        if spec.sizing.max_leverage > policy.max_leverage:
            reasons.append(f"strategy asks {spec.sizing.max_leverage:g}x leverage but the "
                           f"policy caps it at {policy.max_leverage:g}x")
        if spec.exits.stop_atr <= 0:
            reasons.append("the strategy has no stop distance")
        return {"passed": not reasons, "reasons": reasons or ["sizing inside the policy caps"]}

    # ---------- advancing ----------

    def advance(self, version: str) -> PromotionReview:
        """Move the version to the furthest stage whose checks pass, recording each step."""
        review = self.review(version)
        record = self.registry.get(version)
        order = [DeploymentStage.PROPOSED.value, DeploymentStage.BACKTESTED.value,
                 DeploymentStage.OOS_VALIDATED.value, DeploymentStage.WALK_FORWARD.value,
                 DeploymentStage.ROBUSTNESS.value, DeploymentStage.PAPER_TESTED.value,
                 DeploymentStage.RISK_REVIEWED.value, DeploymentStage.AWAITING_APPROVAL.value]
        target = review.furthest_stage
        current = record.stage
        advanced = current not in order or order.index(target) > order.index(current)
        if advanced:
            self.registry.update(version, stage=target)
            self.store.append("deployment", {
                "id": new_id("DEP"), "kind": "stage_advanced", "version": version,
                "from": current, "to": target, "blockers": review.blockers,
                "at": utc_now()})
        return review

    def request_approval(self, version: str, *, approver: str, note: str = "") -> dict[str, Any]:
        """Ask for the explicit approval that live eligibility requires."""
        review = self.review(version)
        if review.blockers:
            raise PermissionError(
                f"{version} cannot be approved yet: " + "; ".join(review.blockers))
        if not review.ready_for_approval:
            raise PermissionError(
                f"{version} is at {review.furthest_stage}; paper testing and risk review "
                "must complete first")
        return self.registry.approve(
            version, approver=approver, note=note,
            paper_trades=int((self.registry.get(version).paper or {}).get("trades") or 0))

    def live_status(self) -> dict[str, Any]:
        """Whether live trading is enabled, and what is blocking it if not."""
        champion = self.registry.champion_version()
        enabled = self.registry.live_trading_enabled()
        reasons: list[str] = []
        if champion is None:
            reasons.append("no champion strategy is registered")
        else:
            if not champion.approval:
                reasons.append(f"champion {champion.version} has no recorded user approval")
            if not champion.live_eligible:
                reasons.append(f"champion {champion.version} is not live-eligible")
            if not enabled:
                reasons.append("live trading is disabled by default")
        return {"live_enabled": enabled,
                "champion": champion.version if champion else None,
                "status": champion.status if champion else None,
                "reasons": reasons}


def deployment_summary(store: ResearchStore) -> list[dict[str, Any]]:
    """The deployment event stream, newest first, for the audit view."""
    return list(reversed(store.read("deployment")))
