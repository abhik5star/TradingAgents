"""Immutable strategy versions, comparison, promotion and rollback.

A version is written once: its rules, parameters, dataset fingerprint, feature
version, experiment id, backtest and out-of-sample results, paper-trading record
and the change from its parent. Nothing overwrites those numbers. What can change
is the *status* — Research → Backtesting → Validation → Paper Testing → Approved /
Rejected — and a status change is appended to the deployment stream with who
asked and why, so the history of a decision is visible after the fact.

Only one version is the champion at a time. Promoting a version requires an
accepted experiment, and making a version live-eligible requires an explicit user
approval record. Live trading stays disabled by default and is never enabled by a
research run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tradingagents.research.contracts import (
    DeploymentStage,
    ExperimentStatus,
    jsonable,
    stable_hash,
    utc_now,
)
from tradingagents.research.store import ResearchStore, new_id
from tradingagents.research.strategies import StrategySpec


class VersionError(RuntimeError):
    """Raised when a version operation would violate immutability or approval rules."""


@dataclass
class StrategyVersion:
    """One immutable snapshot of a strategy, plus its mutable lifecycle metadata."""

    version: str
    spec: StrategySpec
    parent_version: str | None = None
    status: str = ExperimentStatus.RESEARCH.value
    stage: str = DeploymentStage.PROPOSED.value
    created_at: str = field(default_factory=utc_now)
    experiment_id: str | None = None
    dataset: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    backtest: dict[str, Any] = field(default_factory=dict)
    out_of_sample: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    paper: dict[str, Any] = field(default_factory=dict)
    risk_review: dict[str, Any] = field(default_factory=dict)
    changes_from_parent: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    status_history: list[dict[str, Any]] = field(default_factory=list)
    approval: dict[str, Any] | None = None
    live_eligible: bool = False

    def as_dict(self) -> dict[str, Any]:
        return jsonable({**self.__dict__, "spec": self.spec.as_dict(),
                         "signature": self.spec.signature()})

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyVersion:
        spec = StrategySpec.from_dict(payload["spec"])
        fields = {f for f in cls.__dataclass_fields__ if f != "spec"}
        data = {k: v for k, v in payload.items() if k in fields}
        return cls(spec=spec, **data)


class StrategyRegistry:
    """The version store and the promotion rules around it."""

    def __init__(self, store: ResearchStore):
        self.store = store

    # ---------- reading ----------

    def _document(self) -> dict[str, Any]:
        document = self.store.load_registry()
        document.setdefault("versions", [])
        document.setdefault("approvals", [])
        return document

    def versions(self) -> list[StrategyVersion]:
        return [StrategyVersion.from_dict(payload) for payload in self._document()["versions"]]

    def version_names(self) -> list[str]:
        return [version.version for version in self.versions()]

    def get(self, version: str) -> StrategyVersion:
        for candidate in self.versions():
            if candidate.version == version:
                return candidate
        raise VersionError(f"no strategy version {version!r}; known versions: "
                           f"{self.version_names() or 'none'}")

    def champion_version(self) -> StrategyVersion | None:
        name = self._document().get("champion")
        if not name:
            return None
        try:
            return self.get(name)
        except VersionError:
            return None

    def next_version_name(self) -> str:
        highest = 0
        for version in self.versions():
            digits = "".join(ch for ch in version.version if ch.isdigit())
            if version.version.upper().startswith("V") and digits:
                highest = max(highest, int(digits))
        return f"V{highest + 1:02d}"

    # ---------- writing ----------

    def register(self, spec: StrategySpec, *, experiment_id: str | None = None,
                 parent_version: str | None = None,
                 status: ExperimentStatus | str = ExperimentStatus.RESEARCH,
                 stage: DeploymentStage | str = DeploymentStage.PROPOSED,
                 dataset: dict[str, Any] | None = None,
                 backtest: dict[str, Any] | None = None,
                 out_of_sample: dict[str, Any] | None = None,
                 walk_forward: dict[str, Any] | None = None,
                 changes_from_parent: list[str] | None = None,
                 model_version: str | None = None,
                 provenance: dict[str, Any] | None = None) -> StrategyVersion:
        """Write a new version. An existing version name is never overwritten."""
        document = self._document()
        if any(entry["version"] == spec.version for entry in document["versions"]):
            raise VersionError(
                f"version {spec.version} already exists and is immutable; "
                f"register the change as {self.next_version_name()}")
        record = StrategyVersion(
            version=spec.version, spec=spec, parent_version=parent_version,
            status=ExperimentStatus(status).value, stage=DeploymentStage(stage).value,
            experiment_id=experiment_id, dataset=dataset or {},
            backtest=backtest or {}, out_of_sample=out_of_sample or {},
            walk_forward=walk_forward or {}, changes_from_parent=list(changes_from_parent or []),
            model_version=model_version, provenance=provenance or {},
            status_history=[{"status": ExperimentStatus(status).value, "at": utc_now(),
                             "by": "research_engine"}],
        )
        document["versions"].append(record.as_dict())
        if not document.get("champion"):
            document["champion"] = record.version
        self.store.save_registry(document)
        self._deploy_event("version_registered", record.version,
                           {"experiment_id": experiment_id, "parent": parent_version})
        return record

    def update(self, version: str, **changes: Any) -> StrategyVersion:
        """Update lifecycle metadata only; results and rules are write-once.

        Guarded explicitly: rules, dataset fingerprints and measured results are
        in ``IMMUTABLE_FIELDS`` and cannot be passed here even by mistake.
        """
        immutable = {"version", "spec", "dataset", "backtest", "out_of_sample",
                     "walk_forward", "created_at", "parent_version", "experiment_id"}
        blocked = immutable & set(changes)
        if blocked:
            raise VersionError(f"refusing to overwrite immutable field(s): {sorted(blocked)}")
        actor = str(changes.pop("_by", "research_engine"))
        document = self._document()
        for entry in document["versions"]:
            if entry["version"] != version:
                continue
            for key, value in changes.items():
                if key not in StrategyVersion.__dataclass_fields__:
                    raise VersionError(f"unknown version field {key!r}")
                entry[key] = jsonable(value)
            if "status" in changes:
                entry.setdefault("status_history", []).append(
                    {"status": changes["status"], "at": utc_now(), "by": actor})
            self.store.save_registry(document)
            return StrategyVersion.from_dict(entry)
        raise VersionError(f"no strategy version {version!r}")

    def set_status(self, version: str, status: ExperimentStatus | str, *,
                   reason: str = "", by: str = "research_engine") -> StrategyVersion:
        record = self.update(version, status=ExperimentStatus(status).value, _by=by)
        self._deploy_event("status_changed", version,
                           {"status": record.status, "reason": reason, "by": by})
        return record

    def promote(self, version: str, *, experiment_id: str | None = None,
                require_accepted: bool = True) -> StrategyVersion:
        """Make ``version`` the champion, keeping the previous one intact for rollback."""
        record = self.get(version)
        if require_accepted and record.status not in (ExperimentStatus.APPROVED.value,
                                                      ExperimentStatus.PAPER_TESTING.value):
            raise VersionError(
                f"version {version} is {record.status}; only an accepted or paper-tested "
                "version can become champion")
        document = self._document()
        previous = document.get("champion")
        document["champion"] = version
        document.setdefault("history", []).append({
            "kind": "promotion", "from": previous, "to": version,
            "experiment_id": experiment_id, "at": utc_now(), "by": "user_approval"})
        self.store.save_registry(document)
        self._deploy_event("promoted", version,
                           {"previous": previous, "experiment_id": experiment_id})
        return self.get(version)

    def rollback(self, to_version: str, *, reason: str, by: str = "user",
                 from_version: str | None = None) -> StrategyVersion:
        """Return the champion pointer to an earlier version, recording why.

        The rolled-back version keeps every number it produced; only its status
        changes. The previous champion is never deleted, so a bad rollback can be
        reversed by promoting it again.
        """
        target = self.get(to_version)
        document = self._document()
        current = from_version or document.get("champion")
        if current and current != to_version:
            try:
                record = self.get(current)
                history = record.status_history + [{"status": ExperimentStatus.ROLLED_BACK.value,
                                                    "at": utc_now(), "by": by, "reason": reason}]
                for entry in document["versions"]:
                    if entry["version"] == current:
                        entry["status"] = ExperimentStatus.ROLLED_BACK.value
                        entry["live_eligible"] = False
                        entry["status_history"] = jsonable(history)
            except VersionError:
                current = None
        document["champion"] = to_version
        # A rollback is a safety action: live permission is withdrawn immediately and
        # has to be granted again deliberately with the new champion's version typed.
        live_was_enabled = bool(document.get("live_enabled"))
        document["live_enabled"] = False
        document["live_disabled_at"] = utc_now()
        document.setdefault("history", []).append({
            "kind": "rollback", "from": current, "to": to_version, "reason": reason,
            "at": utc_now(), "by": by, "live_disabled": live_was_enabled})
        self.store.save_registry(document)
        self._deploy_event("rolled_back", to_version,
                           {"from": current, "reason": reason, "by": by,
                            "live_disabled": live_was_enabled})
        return target

    def approve(self, version: str, *, approver: str, note: str = "",
                paper_days: int | None = None, paper_trades: int | None = None) -> dict[str, Any]:
        """Record an explicit human approval. Required before live eligibility."""
        record = self.get(version)
        if record.status not in (ExperimentStatus.PAPER_TESTING.value,
                                 ExperimentStatus.APPROVED.value):
            raise VersionError(
                f"version {version} is {record.status}; it must complete paper testing before "
                "a user may approve it for live eligibility")
        if not approver or not approver.strip():
            raise VersionError("an approval must name who gave it")
        approval = {"approval_id": new_id("APR"), "version": version, "approver": approver.strip(),
                    "note": note, "at": utc_now(), "paper_days": paper_days,
                    "paper_trades": paper_trades}
        document = self._document()
        document.setdefault("approvals", []).append(approval)
        for entry in document["versions"]:
            if entry["version"] == version:
                entry["approval"] = approval
                entry["status"] = ExperimentStatus.APPROVED.value
                entry["stage"] = DeploymentStage.LIVE_ELIGIBLE.value
                entry["live_eligible"] = True
                entry.setdefault("status_history", []).append(
                    {"status": ExperimentStatus.APPROVED.value, "at": utc_now(), "by": approver})
        self.store.save_registry(document)
        self._deploy_event("approved", version, approval)
        return approval

    def live_trading_enabled(self) -> bool:
        """Live trading is off unless explicitly enabled *and* an approval exists."""
        document = self._document()
        if not document.get("live_enabled"):
            return False
        champion = document.get("champion")
        approvals = [a for a in document.get("approvals", []) if a.get("version") == champion]
        record = self.champion_version()
        return bool(approvals and record is not None and record.live_eligible)

    def enable_live(self, *, approver: str, confirmation: str) -> dict[str, Any]:
        """Turn live eligibility on for the approved champion; refuses without approval.

        ``confirmation`` must be the champion's version name. That is not security,
        it is a deliberate speed bump: enabling live trading should require typing
        the version you mean.
        """
        record = self.champion_version()
        if record is None:
            raise VersionError("there is no champion strategy to enable")
        if not record.live_eligible or not record.approval:
            raise VersionError(
                f"champion {record.version} has no recorded user approval; live trading stays off")
        if confirmation.strip().upper() != record.version.upper():
            raise VersionError(
                f"confirmation {confirmation!r} does not match the champion {record.version}")
        document = self._document()
        document["live_enabled"] = True
        document["live_enabled_at"] = utc_now()
        document["live_enabled_by"] = approver
        self.store.save_registry(document)
        self._deploy_event("live_enabled", record.version, {"by": approver})
        return {"live_enabled": True, "version": record.version, "by": approver}

    def disable_live(self, *, by: str = "user", reason: str = "") -> None:
        document = self._document()
        document["live_enabled"] = False
        document["live_disabled_at"] = utc_now()
        self.store.save_registry(document)
        self._deploy_event("live_disabled", document.get("champion"), {"by": by,
                                                                       "reason": reason})

    # ---------- comparison ----------

    def compare(self, baseline: str, candidate: str) -> dict[str, Any]:
        """Baseline versus candidate on the same fields, including what changed in the rules."""
        left, right = self.get(baseline), self.get(candidate)
        changes = compare_specs(left.spec, right.spec)

        def metric_pair(field_name: str) -> dict[str, Any]:
            a = (left.backtest.get("metrics") or {}).get(field_name)
            b = (right.backtest.get("metrics") or {}).get(field_name)
            a_oos = (left.out_of_sample.get("metrics") or {}).get(field_name)
            b_oos = (right.out_of_sample.get("metrics") or {}).get(field_name)
            return {
                "backtest": {"baseline": a, "candidate": b,
                             "difference": (b - a) if isinstance(a, (int, float))
                             and isinstance(b, (int, float)) else None},
                "out_of_sample": {"baseline": a_oos, "candidate": b_oos,
                                  "difference": (b_oos - a_oos) if isinstance(a_oos, (int, float))
                                  and isinstance(b_oos, (int, float)) else None},
            }

        return {
            "baseline": baseline, "candidate": candidate,
            "rule_changes": changes,
            "metrics": {name: metric_pair(name) for name in
                        ("trades", "expectancy_r", "profit_factor", "max_drawdown_pct",
                         "sharpe", "net_pnl", "cost_share_of_gross")},
            "paper": {"baseline": left.paper, "candidate": right.paper},
            "status": {"baseline": left.status, "candidate": right.status},
            "dataset": {"baseline": left.dataset, "candidate": right.dataset},
            "signature": {"baseline": left.spec.signature(),
                          "candidate": right.spec.signature()},
        }

    def _deploy_event(self, kind: str, version: str | None, payload: dict[str, Any]) -> None:
        self.store.append("deployment", {"id": new_id("DEP"), "kind": kind,
                                         "version": version, "payload": payload})


def compare_specs(baseline: StrategySpec, candidate: StrategySpec) -> list[str]:
    """Human-readable difference between two rule sets."""
    changes: list[str] = []
    for label, attribute in (("long entry", "long_entry"), ("short entry", "short_entry"),
                             ("filter", "filters")):
        left = {c.label or c.describe() for c in getattr(baseline, attribute)}
        right = {c.label or c.describe() for c in getattr(candidate, attribute)}
        added, removed = sorted(right - left), sorted(left - right)
        if added:
            changes.append(f"{label}: added " + "; ".join(added))
        if removed:
            changes.append(f"{label}: removed " + "; ".join(removed))
    if baseline.exits != candidate.exits:
        changes.append(f"exits: {baseline.exits.as_dict()} -> {candidate.exits.as_dict()}")
    if baseline.sizing != candidate.sizing:
        changes.append(f"sizing: {baseline.sizing.as_dict()} -> {candidate.sizing.as_dict()}")
    if sorted(baseline.allowed_regimes) != sorted(candidate.allowed_regimes):
        changes.append(f"regimes: {list(baseline.allowed_regimes)} -> "
                       f"{list(candidate.allowed_regimes)}")
    for label, attribute in (("timeframe", "timeframe"), ("context", "context_timeframe"),
                             ("confirmation", "confirm_timeframe")):
        if getattr(baseline, attribute) != getattr(candidate, attribute):
            changes.append(f"{label}: {getattr(baseline, attribute)} -> "
                           f"{getattr(candidate, attribute)}")
    if not changes:
        changes.append("no rule difference: only the recorded evidence differs")
    return changes


def spec_fingerprint(spec: StrategySpec) -> str:
    """The rule signature plus version, for reporting a version's exact content."""
    return stable_hash({"signature": spec.signature(), "version": spec.version})[:12]
