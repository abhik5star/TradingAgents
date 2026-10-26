"""Trade autopsy: why this trade happened, and whether it was a mistake.

The distinction the brief insists on is the one most systems get wrong: a losing
trade is not automatically a mistake, and a winning trade does not validate a
strategy. So every autopsy starts from the evidence and asks a fixed set of
questions:

* did the entry have the follow-through the rules claimed?
* was the regime classified correctly, and were the timeframes aligned?
* was the setup actually confirmed by the rules the strategy declares?
* were costs, spread and slippage a material part of the result?
* was the stop appropriate for the volatility that was measured at entry?
* was the planned risk inside the risk budget?
* did the trade exit the way the tested strategy says it exits?
* is this one loss ordinary variance, or the same weakness repeating?

Findings are classified with :class:`~tradingagents.research.contracts.MistakeCategory`.
When the evidence is insufficient the verdict is ``unknown`` rather than a guess,
and a loss with nothing wrong in it is recorded as ``normal_statistical_loss``.
A weakness is only "repeatable" once it appears in more than one trade.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tradingagents.research.contracts import MistakeCategory, TradeRecord

#: Exit reasons that are part of a tested strategy. Anything else is an unplanned exit.
PLANNED_EXITS = ("stop", "stop_gap", "target", "trailing_stop", "time_stop",
                 "partial_full", "end_of_data")

#: Categories that describe what a trade *was*, not a fault in it. Ordinary variance is
#: not a weakness, and "unknown" means the evidence is missing; neither may be counted
#: towards a repeatable weakness, nor listed as one on the dashboard.
NON_DEFECTS = frozenset({MistakeCategory.NORMAL_LOSS.value, MistakeCategory.UNKNOWN.value})

#: A stop closer than this fraction of the entry ATR is too tight for measured volatility.
TIGHT_STOP_ATR = 0.75
#: A stop wider than this many ATRs risks more than the setup justifies.
WIDE_STOP_ATR = 4.0
#: Costs above this share of the gross move make the result an execution story.
COST_MATERIALITY = 0.30


@dataclass
class Finding:
    """One answer to one autopsy question, with the evidence behind it."""

    category: MistakeCategory
    question: str
    description: str
    evidence: str
    severity: str = "info"

    def as_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["category"] = self.category.value
        return payload


@dataclass
class AutopsyRecord:
    """The structured post-mortem of a single trade."""

    trade_id: str
    version: str
    ticker: str
    is_loss: bool
    verdict: MistakeCategory
    findings: list[Finding] = field(default_factory=list)
    repeatable: bool = False
    note: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def categories(self) -> list[str]:
        return sorted({finding.category.value for finding in self.findings})

    def as_dict(self) -> dict[str, Any]:
        return {"trade_id": self.trade_id, "version": self.version, "ticker": self.ticker,
                "is_loss": self.is_loss, "verdict": self.verdict.value,
                "findings": [f.as_dict() for f in self.findings],
                "repeatable": self.repeatable, "note": self.note,
                "categories": self.categories(), "context": self.context}


def _cost_share(trade: TradeRecord) -> float | None:
    gross_move = abs(trade.exit_price - trade.entry_price) * abs(trade.quantity)
    if gross_move <= 0:
        return None
    return trade.cost / gross_move


def autopsy_trade(
    trade: TradeRecord,
    *,
    policy: Any = None,
    atr_at_entry: float | None = None,
    equity_at_entry: float | None = None,
    htf_conflict: bool | None = None,
    regime_allowed: bool | None = None,
    data_quality_ok: bool = True,
    model_probability: float | None = None,
) -> AutopsyRecord:
    """Diagnose one trade. Missing evidence produces an ``unknown`` finding, not a guess."""
    findings: list[Finding] = []
    is_loss = trade.net_pnl < 0
    atr = atr_at_entry if atr_at_entry is not None else trade.context.get("atr_at_entry")
    stop_distance = abs(trade.entry_price - trade.stop_price)
    stop_atr = (stop_distance / atr) if atr and atr > 0 else None

    if not data_quality_ok:
        findings.append(Finding(
            MistakeCategory.DATA_QUALITY, "was the input data trustworthy?",
            "the trade was evaluated on data that failed its quality gate",
            "data quality gate failed", "serious"))
    if not trade.signals:
        findings.append(Finding(
            MistakeCategory.STRATEGY_LOGIC, "did the rules that fired get recorded?",
            "no rule evidence was stored for this entry, so the trade cannot be audited",
            "signals list empty", "warning"))
    if trade.exit_reason not in PLANNED_EXITS:
        findings.append(Finding(
            MistakeCategory.STRATEGY_LOGIC, "did the exit match the tested strategy?",
            f"the position closed with '{trade.exit_reason}', which is not part of the tested exits",
            f"exit_reason={trade.exit_reason}", "warning"))
    if htf_conflict:
        findings.append(Finding(
            MistakeCategory.REGIME_CLASSIFICATION, "were the timeframes aligned?",
            "the higher and trading timeframes disagreed at entry",
            "mtf_conflict was true at the signal bar", "serious"))
    if trade.regime == "unstable":
        findings.append(Finding(
            MistakeCategory.REGIME_CLASSIFICATION, "was the regime classified?",
            "the entry happened in an unclassified/unstable regime",
            "regime=unstable", "warning"))
    if regime_allowed is False:
        findings.append(Finding(
            MistakeCategory.STRATEGY_LOGIC, "was the setup allowed in this regime?",
            "the trade was taken in a regime the strategy does not declare",
            f"regime={trade.regime} not in the allowed list", "serious"))
    if is_loss and trade.mfe_r >= 1.0:
        findings.append(Finding(
            MistakeCategory.STRATEGY_LOGIC, "was the exit consistent with the plan?",
            f"price moved {trade.mfe_r:.2f}R in favour before the loss: the exit gave back a "
            "profitable position",
            f"mfe_r={trade.mfe_r:.2f}, exit={trade.exit_reason}", "warning"))
    if is_loss and trade.mfe_r < 0.25 and trade.mae_r >= 0.8:
        findings.append(Finding(
            MistakeCategory.STRATEGY_LOGIC, "was the entry late or premature?",
            "the trade moved against the entry almost immediately with no favourable excursion",
            f"mfe_r={trade.mfe_r:.2f}, mae_r={trade.mae_r:.2f}", "warning"))
    share = _cost_share(trade)
    if share is not None and share >= COST_MATERIALITY:
        findings.append(Finding(
            MistakeCategory.EXECUTION, "were costs material?",
            f"costs were {share:.0%} of the gross price move",
            f"cost={trade.cost:.4f}, gross_move={abs(trade.exit_price - trade.entry_price) * abs(trade.quantity):.4f}",
            "warning"))
    if stop_atr is not None and stop_atr < TIGHT_STOP_ATR:
        findings.append(Finding(
            MistakeCategory.RISK_MANAGEMENT, "was the stop appropriate for volatility?",
            f"the stop sat {stop_atr:.2f} ATR from entry, inside normal noise",
            f"stop_distance={stop_distance:.4f}, atr={float(atr):.4f}", "warning"))
    if stop_atr is not None and stop_atr > WIDE_STOP_ATR:
        findings.append(Finding(
            MistakeCategory.RISK_MANAGEMENT, "was the stop appropriate for the setup?",
            f"the stop sat {stop_atr:.2f} ATR from entry, wider than the setup justifies",
            f"stop_distance={stop_distance:.4f}, atr={float(atr):.4f}", "warning"))
    if (policy is not None and equity_at_entry and equity_at_entry > 0
            and trade.planned_risk > policy.max_risk_per_trade * equity_at_entry * 1.0001):
        findings.append(Finding(
            MistakeCategory.RISK_MANAGEMENT, "was position sizing inside the risk budget?",
            f"planned risk {trade.planned_risk:,.2f} exceeded the "
            f"{policy.max_risk_per_trade:.2%} per-trade cap",
            f"equity_at_entry={equity_at_entry:,.2f}", "serious"))
    if model_probability is not None and model_probability < 0.5 and is_loss:
        findings.append(Finding(
            MistakeCategory.MODEL_PREDICTION, "did the model support this trade?",
            f"the model gave the setup a {model_probability:.0%} probability and the trade lost",
            f"model_probability={model_probability:.4f}", "info"))

    if not findings:
        if is_loss:
            verdict = MistakeCategory.NORMAL_LOSS
            findings.append(Finding(
                MistakeCategory.NORMAL_LOSS, "was this loss a mistake?",
                "a loss with no rule, regime, execution or risk violation: ordinary variance",
                f"net_pnl={trade.net_pnl:.4f}, r_multiple={trade.r_multiple:+.2f}R", "info"))
            note = ("A single losing trade is not evidence against a strategy. Only the "
                    "distribution of outcomes, on out-of-sample data, decides that.")
        else:
            verdict = MistakeCategory.UNKNOWN
            findings.append(Finding(
                MistakeCategory.UNKNOWN, "was the win earned by the plan?",
                "a profitable trade with no recorded rule evidence proves nothing",
                "no findings and no rule evidence", "info"))
            note = "A winning trade does not validate a strategy. It is one observation."
    else:
        seriousness = {"serious": 3, "warning": 2, "info": 1}
        verdict = max(findings, key=lambda f: seriousness.get(f.severity, 0)).category
        note = ("This trade is explained by the findings above; whether the cause is "
                "repeatable is decided across the batch.")

    return AutopsyRecord(trade_id=trade.trade_id, version=trade.version, ticker=trade.ticker,
                         is_loss=is_loss, verdict=verdict, findings=findings, note=note,
                         context={"regime": trade.regime, "session": trade.session,
                                  "exit_reason": trade.exit_reason,
                                  "r_multiple": round(trade.r_multiple, 4)})


def _defects(record: AutopsyRecord) -> list[str]:
    """The categories that are actual faults, excluding ordinary variance and unknowns."""
    return [category for category in record.categories() if category not in NON_DEFECTS]


def autopsy_batch(trades: Iterable[TradeRecord], *, policy: Any = None,
                  min_occurrences: int = 3, **context: Any) -> list[AutopsyRecord]:
    """Autopsy every trade, then flag the categories that recur.

    A category seen once is a story; seen ``min_occurrences`` times among losses it
    is a repeatable weakness and gets ``repeatable=True``.
    """
    records = [autopsy_trade(trade, policy=policy, **context) for trade in trades]
    counts: dict[str, int] = {}
    for record in records:
        if record.is_loss:
            for category in _defects(record):
                counts[category] = counts.get(category, 0) + 1
    for record in records:
        if record.is_loss:
            repeated = [category for category in _defects(record)
                        if counts.get(category, 0) >= min_occurrences]
            record.repeatable = bool(repeated)
            if repeated:
                record.note = (f"recurs across {counts[repeated[0]]} losing trades: "
                               f"{', '.join(repeated)}. This is a repeatable weakness, not variance.")
    return records


def recurring_weaknesses(records: Iterable[AutopsyRecord]) -> list[dict[str, Any]]:
    """Losses grouped by category, worst first, with an example and a suggested action."""
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.is_loss:
            continue
        for finding in record.findings:
            category = finding.category.value
            if category in NON_DEFECTS:
                continue
            entry = grouped.setdefault(category, {"category": category, "trades": 0,
                                                  "example": finding.description,
                                                  "repeatable": False})
            entry["trades"] += 1
            entry["repeatable"] = entry["repeatable"] or record.repeatable
    ordered = sorted(grouped.values(), key=lambda entry: (-entry["trades"], entry["category"]))
    for entry in ordered:
        entry["action"] = _ACTION.get(entry["category"], "Review this category before the next run.")
    return ordered


_ACTION = {
    MistakeCategory.STRATEGY_LOGIC.value:
        "Check the entry trigger and exit rules: the documented plan and the executed plan differ.",
    MistakeCategory.DATA_QUALITY.value:
        "Fix or exclude the data source; results from this series cannot be trusted.",
    MistakeCategory.REGIME_CLASSIFICATION.value:
        "Test a regime filter, and require timeframe agreement before entry.",
    MistakeCategory.EXECUTION.value:
        "Re-run with higher slippage and spread assumptions; the edge is inside the friction.",
    MistakeCategory.RISK_MANAGEMENT.value:
        "Size stops from measured volatility and keep planned risk inside the per-trade cap.",
    MistakeCategory.MODEL_PREDICTION.value:
        "Retrain or recalibrate the model, and check for drift before using its probabilities.",
    MistakeCategory.NORMAL_LOSS.value:
        "No action: losses of this size are within the strategy's expected distribution.",
    MistakeCategory.UNKNOWN.value:
        "Collect more evidence: record rule signals and the regime at entry for every trade.",
}
