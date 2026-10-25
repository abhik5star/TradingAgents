"""The Strategy Intelligence view: one snapshot, two renderers.

The snapshot is assembled from the stored evidence only — version registry,
hypothesis records, experiment records, autopsies, risk events and model events.
Nothing is re-run to draw it, so the dashboard cannot accidentally become another
experiment, and it shows the same numbers the audit trail holds.

:func:`render_console` prints it as Rich panels in the CLI, and
:func:`render_section_html` / :func:`write_dashboard_html` render it into the
existing report theme (the same tokens and typography as the analysis report).

Statuses are the seven from the brief: Research, Backtesting, Validation, Paper
Testing, Approved, Rejected, Rolled Back. ``NO TRADE`` is shown whenever the last
run found nothing worth trading, with the reason that produced it.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from typing import Any

from tradingagents.research.contracts import jsonable, utc_now
from tradingagents.research.scoring import NO_TRADE
from tradingagents.research.store import ResearchStore
from tradingagents.research.versions import StrategyRegistry


def _latest_by(records: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The last write for each key, in first-seen order.

    Hypothesis and experiment records are appended twice (on creation and on
    completion); the dashboard must show the completed one, not both.
    """
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = str(record.get(key) or record.get("id") or "")
        if identifier:
            latest[identifier] = record
    return list(latest.values())


def _metric(metrics: dict[str, Any] | None, name: str, default: Any = None) -> Any:
    return (metrics or {}).get(name, default)


def _format_metric(value: Any, suffix: str = "", decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:,.{decimals}f}{suffix}"
    return str(value)


def build_snapshot(store: ResearchStore, summary: Any | None = None,
                   ticker: str | None = None) -> dict[str, Any]:
    """Assemble the dashboard document from stored evidence (plus a fresh loop summary)."""
    registry = StrategyRegistry(store)
    versions = registry.versions()
    champion = registry.champion_version()
    hypotheses = _latest_by(store.read("hypothesis"), "hypothesis_id")
    experiments = [record for record in _latest_by(store.read("experiment"), "id")
                   if record.get("kind") != "no_hypothesis"]
    autopsies = _latest_by(store.read("autopsy"), "trade_id")
    risk_events = store.read("risk_event")
    ml_events = store.read("ml_event")
    if ticker:
        hypotheses = [r for r in hypotheses if r.get("ticker") == ticker]
        experiments = [r for r in experiments if r.get("ticker") == ticker]

    completed = [record for record in experiments
                 if record.get("decision") in ("accepted", "rejected")]
    accepted = [record for record in completed if record.get("decision") == "accepted"]
    rejected = [record for record in completed if record.get("decision") == "rejected"]
    latest_hypothesis = hypotheses[-1] if hypotheses else None
    active = next((record for record in reversed(hypotheses)
                   if record.get("status") in ("Research", "Backtesting", "Validation")), None)

    latest_experiment = completed[-1] if completed else None
    comparison = ((latest_experiment or {}).get("results") or {}).get("comparison") or {}
    oos_metrics = (((latest_experiment or {}).get("results") or {}).get("out_of_sample")
                   or {}).get("metrics")
    baseline_metrics = (((latest_experiment or {}).get("results") or {})
                        .get("baseline_out_of_sample") or {}).get("metrics")
    gates = ((latest_experiment or {}).get("results") or {}).get("gates") or []
    failed_gates = [gate for gate in gates if not gate.get("passed")]

    weakness_counts: dict[str, dict[str, Any]] = {}
    for record in autopsies:
        if not record.get("is_loss"):
            continue
        for finding in record.get("findings", []):
            category = finding.get("category", "unknown")
            entry = weakness_counts.setdefault(category, {"category": category, "trades": 0,
                                                          "repeatable": False})
            entry["trades"] += 1
            entry["repeatable"] = entry["repeatable"] or bool(record.get("repeatable"))
    weaknesses = sorted(weakness_counts.values(), key=lambda item: -item["trades"])

    paper_candidates = [version for version in versions
                        if version.status == "Paper Testing" or version.paper]
    latest_paper = (paper_candidates[-1].paper if paper_candidates else {}) or {}
    approvals = list(store.load_registry().get("approvals") or [])
    drift_alerts = [event for event in ml_events if event.get("kind") == "drift_alert"]
    last_training = next((event for event in reversed(ml_events)
                          if event.get("kind") == "model_registered"), None)

    no_trade_reason = ""
    if summary is not None:
        no_trade_reason = getattr(summary, "no_trade_reason", "") or ""
    if not no_trade_reason and latest_experiment and latest_experiment.get("decision") == "rejected":
        no_trade_reason = (f"{NO_TRADE}: the latest candidate was rejected — "
                           + "; ".join(f"{gate.get('name')} ({gate.get('measured')})"
                                       for gate in failed_gates) or "gates failed")

    status_counts: dict[str, int] = {}
    for version in versions:
        status_counts[version.status] = status_counts.get(version.status, 0) + 1

    snapshot = {
        "generated_at": utc_now(),
        "ticker": ticker or getattr(summary, "ticker", None),
        "champion": None if champion is None else {
            "version": champion.version,
            "name": champion.spec.name,
            "status": champion.status,
            "stage": champion.stage,
            "rules": champion.spec.describe(),
            "signature": champion.spec.signature(),
            "out_of_sample_expectancy_r": _metric(
                (champion.out_of_sample or {}).get("metrics"), "expectancy_r"),
            "paper": champion.paper or {},
            "approval": champion.approval,
        },
        "live_enabled": registry.live_trading_enabled(),
        "active_experiment": None if active is None else {
            "hypothesis_id": active.get("hypothesis_id"),
            "title": active.get("title"),
            "status": active.get("status"),
        },
        "latest_hypothesis": None if latest_hypothesis is None else {
            "hypothesis_id": latest_hypothesis.get("hypothesis_id"),
            "title": latest_hypothesis.get("title"),
            "decision": latest_hypothesis.get("decision"),
            "status": latest_hypothesis.get("status"),
            "explanation": latest_hypothesis.get("explanation"),
            "confidence": latest_hypothesis.get("confidence"),
            "reason": latest_hypothesis.get("reason"),
            "evidence": latest_hypothesis.get("evidence") or [],
            "expected_benefit": latest_hypothesis.get("expected_benefit"),
            "possible_downside": latest_hypothesis.get("possible_downside"),
            "created_at": latest_hypothesis.get("created_at"),
        },
        "counts": {
            "experiments_completed": len(completed),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "hypotheses": len(hypotheses),
            "versions": len(versions),
            "statuses": status_counts,
        },
        "baseline_vs_candidate": {
            "baseline_version": (latest_experiment or {}).get("baseline_version"),
            "candidate_version": (latest_experiment or {}).get("candidate_version"),
            "decision": (latest_experiment or {}).get("decision"),
            "metrics": comparison,
        },
        "out_of_sample": {"candidate": oos_metrics, "baseline": baseline_metrics},
        "paper_testing": {"status": latest_paper.get("status", "not started"),
                          "trades": latest_paper.get("trades", 0),
                          "expectancy_r": latest_paper.get("expectancy_r"),
                          "started": latest_paper.get("started"),
                          "ended": latest_paper.get("ended"),
                          "note": latest_paper.get("note", "")},
        "mistakes": weaknesses,
        "risk": {
            "blocks": len([event for event in risk_events
                           if event.get("kind") == "entry_blocked"]),
            "latest": (risk_events[-1] if risk_events else None),
            "policy": store.load_registry().get("risk_policy"),
        },
        "drift_alerts": drift_alerts[-3:],
        "last_training": last_training,
        "last_run": (latest_experiment or {}).get("recorded_at")
        or (latest_hypothesis or {}).get("finished_at"),
        "data_quality": (getattr(summary, "data_quality", None) or {}),
        "leakage": (getattr(summary, "leakage", None) or {}),
        "approvals": approvals[-5:],
        "holdout_windows": store.load_holdouts().get("windows", {}),
        "no_trade_reason": no_trade_reason,
        "notes": list(getattr(summary, "notes", []) or []),
    }
    if summary is not None and getattr(summary, "best", None):
        snapshot["best_candidate"] = summary.best
    return jsonable(snapshot)


def publish_snapshot(store: ResearchStore, summary: Any | None = None,
                     ticker: str | None = None) -> dict[str, Any]:
    """Build the snapshot and persist it, so the dashboard survives a restart."""
    snapshot = build_snapshot(store, summary=summary, ticker=ticker)
    store.write_snapshot(snapshot)
    return snapshot


# --------------------------------------------------------------------------------------
# Console rendering
# --------------------------------------------------------------------------------------


def render_console(snapshot: dict[str, Any], console: Any = None) -> None:
    """Print the snapshot as Rich panels, in the existing CLI's voice."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = console or Console()
    champion = snapshot.get("champion") or {}
    counts = snapshot.get("counts") or {}
    heading = "Strategy Intelligence"
    if snapshot.get("ticker"):
        heading += f" · {snapshot['ticker']}"

    champion_lines = []
    if champion:
        champion_lines.append(f"[bold]{escape(champion['version'])}[/bold] "
                              f"{escape(champion['name'])}")
        champion_lines.append(f"status: {champion['status']} · stage: {champion['stage']}")
        if champion.get("out_of_sample_expectancy_r") is not None:
            champion_lines.append("out-of-sample expectancy: "
                                  f"{champion['out_of_sample_expectancy_r']:+.3f}R")
        if champion.get("approval"):
            champion_lines.append(f"approved by {escape(str(champion['approval'].get('approver')))}")
    else:
        champion_lines.append("[yellow]no champion strategy registered yet[/yellow]")
    live = "enabled" if snapshot.get("live_enabled") else "disabled (default)"
    champion_lines.append(f"live trading: {live}")
    console.print(Panel("\n".join(champion_lines), title=heading, border_style="green"))

    research = Table(box=None, show_header=False, pad_edge=False)
    research.add_column("field", style="dim")
    research.add_column("value")
    active = snapshot.get("active_experiment")
    research.add_row("active experiment",
                     f"{active['hypothesis_id']} · {active['title']}" if active else "none")
    latest = snapshot.get("latest_hypothesis")
    if latest:
        research.add_row("latest hypothesis", escape(str(latest["title"])))
        research.add_row("decision", f"{latest['decision']} ({latest['status']})")
    research.add_row("experiments", f"{counts.get('experiments_completed', 0)} completed · "
                                   f"{counts.get('accepted', 0)} accepted · "
                                   f"{counts.get('rejected', 0)} rejected")
    research.add_row("versions", str(counts.get("versions", 0)))
    research.add_row("paper testing",
                     f"{snapshot['paper_testing']['status']} · "
                     f"{snapshot['paper_testing']['trades']} trades")
    research.add_row("last run", str(snapshot.get("last_run") or "never"))
    console.print(Panel(research, title="Research", border_style="cyan"))

    oos = snapshot.get("out_of_sample") or {}
    performance = Table(box=None, show_header=True, header_style="bold")
    for column in ("sample", "trades", "expectancy", "profit factor", "max drawdown"):
        performance.add_column(column)
    for label, metrics in (("baseline", oos.get("baseline")), ("candidate", oos.get("candidate"))):
        if not metrics:
            continue
        performance.add_row(label, str(metrics.get("trades", 0)),
                            f"{_format_metric(metrics.get('expectancy_r'), 'R', 3)}",
                            _format_metric(metrics.get("profit_factor")),
                            f"{(metrics.get('max_drawdown_pct') or 0):.1%}")
    if performance.row_count:
        console.print(Panel(performance, title="Out-of-sample", border_style="blue"))

    health = Table(box=None, show_header=False, pad_edge=False)
    health.add_column("field", style="dim")
    health.add_column("value")
    mistakes = snapshot.get("mistakes") or []
    health.add_row("recurring weaknesses", ", ".join(
        f"{entry['category']} ({entry['trades']})" for entry in mistakes[:4]) or "none recorded")
    drift = snapshot.get("drift_alerts") or []
    health.add_row("drift alerts", f"{len(drift)} recent" if drift else "none")
    health.add_row("last training",
                   str((snapshot.get("last_training") or {}).get("recorded_at") or "never"))
    health.add_row("risk blocks", str((snapshot.get("risk") or {}).get("blocks", 0)))
    quality = snapshot.get("data_quality") or {}
    if quality:
        failed = [tf for tf, report in quality.items() if not report.get("ok")]
        health.add_row("data quality", "blocked: " + ", ".join(failed) if failed else "passed")
    leakage = snapshot.get("leakage") or {}
    if leakage:
        health.add_row("look-ahead probe",
                       "passed" if leakage.get("ok") else "FAILED: " + ", ".join(
                           leakage.get("differences", [])[:2]))
    console.print(Panel(health, title="Health", border_style="yellow"))

    reason = snapshot.get("no_trade_reason")
    if reason:
        console.print(Panel(Text(reason), title="Decision", border_style="red"))
    else:
        console.print(Panel("Trading decisions follow the gates; nothing is promoted silently.",
                            title="Decision", border_style="red"))


# --------------------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------------------


def _table_rows(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    champion = snapshot.get("champion") or {}
    counts = snapshot.get("counts") or {}
    latest = snapshot.get("latest_hypothesis") or {}
    oos = snapshot.get("out_of_sample") or {}
    paper = snapshot.get("paper_testing") or {}
    rows = [
        ("Champion", f"{champion.get('version', '—')} · {champion.get('name', 'none')}"),
        ("Status", f"{champion.get('status', '—')} · {champion.get('stage', '—')}"),
        ("Experiments", f"{counts.get('experiments_completed', 0)} completed · "
                        f"{counts.get('accepted', 0)} accepted · "
                        f"{counts.get('rejected', 0)} rejected"),
        ("Active research", (snapshot.get("active_experiment") or {}).get("title", "none")),
        ("Latest hypothesis", latest.get("title", "none")),
        ("Hypothesis decision", f"{latest.get('decision', '—')} ({latest.get('status', '—')})"),
        ("Baseline out-of-sample", _format_metric(
            _metric(oos.get("baseline"), "expectancy_r"), "R", 3)),
        ("Candidate out-of-sample", _format_metric(
            _metric(oos.get("candidate"), "expectancy_r"), "R", 3)),
        ("Paper testing", f"{paper.get('status', 'not started')} · "
                          f"{paper.get('trades', 0)} trades"),
        ("Risk blocks", str((snapshot.get("risk") or {}).get("blocks", 0))),
        ("Model drift alerts", str(len(snapshot.get("drift_alerts") or []))),
        ("Last successful run", str(snapshot.get("last_run") or "never")),
        ("Live trading", "enabled" if snapshot.get("live_enabled") else "disabled"),
    ]
    return [(label, str(value)) for label, value in rows]


def render_section_html(snapshot: dict[str, Any]) -> str:
    """The Strategy Intelligence block for the existing HTML report.

    Everything is escaped and the page's content-security policy still forbids
    scripts, so a model-written hypothesis cannot inject markup into the report.
    """
    rows = "".join(
        f'<div><dt>{escape(label)}</dt><dd>{escape(value)}</dd></div>'
        for label, value in _table_rows(snapshot))
    latest = snapshot.get("latest_hypothesis") or {}
    detail = []
    if latest.get("evidence"):
        detail.append("<p><strong>Evidence:</strong> "
                      + escape("; ".join(str(item) for item in latest["evidence"])) + "</p>")
    if latest.get("explanation"):
        detail.append(f"<p><strong>Decision:</strong> {escape(str(latest['explanation']))}</p>")
    if latest.get("confidence"):
        detail.append(f"<p><strong>Confidence:</strong> {escape(str(latest['confidence']))}</p>")
    mistakes = snapshot.get("mistakes") or []
    if mistakes:
        items = "".join(
            f"<li>{escape(str(entry['category']))} — {entry['trades']} losing trade(s)"
            f"{' · repeatable' if entry.get('repeatable') else ''}</li>"
            for entry in mistakes[:5])
        detail.append(f"<p><strong>Recurring weaknesses:</strong></p><ul>{items}</ul>")
    reason = snapshot.get("no_trade_reason")
    if reason:
        detail.append(f'<p class="note"><strong>{escape(reason)}</strong></p>')
    return (
        '<section id="strategy-intelligence"><h2>Strategy Intelligence</h2>'
        f'<dl class="run">{rows}</dl>' + "".join(detail) + "</section>")


def write_dashboard_html(snapshot: dict[str, Any], path: str) -> str:
    """Write a standalone page using the same styles as the analysis report."""
    from tradingagents.report_html import _CSS  # same theme, not a second one

    body = render_section_html(snapshot)
    title = "Strategy Intelligence: TradingAgents"
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>{escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page" id="top">
<header class="masthead"><h1>Strategy Intelligence</h1>
<p class="date">Evidence as of {escape(str(snapshot.get('generated_at', '')))}</p></header>
<main>
{body}
</main>
<footer class="credit">Generated by the TradingAgents research engine. No strategy is
promoted, sized or made live from this page.</footer>
</div>
</body>
</html>
"""
    from pathlib import Path

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return str(target)
