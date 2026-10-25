"""``tradingagents research`` — the Strategy Intelligence commands.

These commands sit beside the existing workflow (run an analysis, run a backtest);
nothing about the default run changes. They operate on the research store under
``<results_dir>/research`` and only ever read stored evidence or run a declared
experiment.

The order a user is meant to follow:

``research init`` → ``research observe`` → ``research run`` → ``research status``
→ ``research approve`` (after paper testing) → ``research promote``.

Live trading is never enabled by any command except ``research live --enable``,
which requires a recorded human approval and the champion's version typed as
confirmation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from cli.display import console
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.research.contracts import ExperimentStatus, Timeframe
from tradingagents.research.dashboard import publish_snapshot, render_console, write_dashboard_html
from tradingagents.research.data import CsvDataSource, MarketData, YahooDataSource, load_market_data
from tradingagents.research.engine import EngineConfig, run_backtest
from tradingagents.research.features import FeatureTable, build_feature_table
from tradingagents.research.hypothesis import HypothesisLoop, LoopPlan, recent_hypotheses
from tradingagents.research.ml import (
    ModelRegistry,
    auc_significance,
    build_dataset,
    ml_adds_value,
    train_with_oof,
)
from tradingagents.research.pipeline import PromotionPipeline
from tradingagents.research.risk import RiskEngine, RiskPolicy, load_policy, save_policy
from tradingagents.research.scoring import NO_TRADE
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import baseline_strategy
from tradingagents.research.validation import AcceptanceCriteria, HoldoutLedger
from tradingagents.research.versions import StrategyRegistry

research_app = typer.Typer(
    name="research",
    help="Strategy Intelligence: hypothesis loop, validation gates, versions, autopsies.",
    no_args_is_help=True,
)

_DEFAULT_TIMEFRAMES = ("1d",)
#: Execution assumptions come from config, so a run's costs are reproducible from
#: the config that produced it rather than from whatever the CLI defaults were that day.
_COSTS = DEFAULT_CONFIG.get("research_costs") or {
    "commission_bps": 2.0, "slippage_bps": 2.0, "spread_bps": 2.0,
    "tax_bps_sell": 0.0, "funding_bps_per_day": 0.0}


def _store(research_dir: str | None) -> ResearchStore:
    return ResearchStore(DEFAULT_CONFIG, root=research_dir) if research_dir else ResearchStore(
        DEFAULT_CONFIG)


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def _policy(store: ResearchStore) -> RiskPolicy:
    return load_policy(store)


def _load_table(ticker: str, timeframe: str, context: str | None, confirm: str | None,
                start: str, end: str, csv_dir: str | None,
                asset_type: str) -> tuple[FeatureTable, MarketData]:
    """Load and feature-build one instrument, with every data problem reported."""
    timeframes = [timeframe] + [tf for tf in (context, confirm) if tf]
    source = CsvDataSource(csv_dir) if csv_dir else YahooDataSource(
        Path(DEFAULT_CONFIG["data_cache_dir"]) / "research")
    data = load_market_data(ticker, timeframes, start, end, asset_type=asset_type, source=source)
    quality = Table(title=f"Data quality · {ticker}", box=None, header_style="bold")
    quality.add_column("timeframe")
    quality.add_column("bars", justify="right")
    quality.add_column("coverage", justify="right")
    quality.add_column("status")
    quality.add_column("notes")
    for name, report in sorted(data.quality.items()):
        status = "[green]ok[/green]" if report.ok else "[red]blocked[/red]"
        notes = "; ".join(report.blocking or report.warnings)[:90]
        quality.add_row(name, str(report.bars),
                        f"{report.coverage:.0%}" if report.coverage is not None else "n/a",
                        status, notes)
    console.print(quality)
    if data.blocking_issues():
        _fail("data quality gate failed; no trades or experiments will be run on this series")
    table = build_feature_table(data, timeframe, context, confirm)
    console.print(f"[dim]{len(table.bars)} bars · {len(table.features.columns)} features · "
                  f"feature version {table.version}[/dim]")
    for note in table.notes:
        console.print(f"[yellow]{note}[/yellow]")
    return table, data


def _runtime(store: ResearchStore) -> tuple[RiskPolicy, RiskEngine]:
    policy = _policy(store)
    return policy, RiskEngine(policy, store)


# --------------------------------------------------------------------------------------
# init / status / observe
# --------------------------------------------------------------------------------------


@research_app.command("init")
def init(
    research_dir: str = typer.Option(None, "--research-dir", help="Where the research memory lives"),
    timeframe: str = typer.Option("1d", "--timeframe", help="Trading timeframe for the baseline"),
    context: str = typer.Option(None, "--context", help="Higher timeframe used for context"),
) -> None:
    """Create the research store and seed the baseline version (V01)."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    if registry.champion_version() is not None:
        champion = registry.champion_version()
        console.print(f"[yellow]Research memory already initialised; champion is "
                      f"{champion.version} ({champion.status}).[/yellow]")
    else:
        spec = baseline_strategy(Timeframe.parse(timeframe).value,
                                 Timeframe.parse(context).value if context else None)
        registry.register(spec, status=ExperimentStatus.RESEARCH,
                          stage="proposed", provenance={"seeded_as_baseline": True})
        console.print(f"[green]Seeded baseline {spec.version} and set it as the reference "
                      "champion. It has no measured edge yet: run `research run` to test "
                      "ideas against it.[/green]")
    console.print(f"Memory: {store.root}")
    console.print("Live trading: disabled (default).")


@research_app.command("status")
def status(
    research_dir: str = typer.Option(None, "--research-dir"),
    ticker: str = typer.Option(None, "--ticker", help="Limit to one instrument"),
    snapshot_only: bool = typer.Option(False, "--cached", help="Show the last saved snapshot"),
) -> None:
    """Show the Strategy Intelligence dashboard."""
    store = _store(research_dir)
    snapshot = store.read_snapshot() if snapshot_only else None
    snapshot = snapshot or publish_snapshot(store, ticker=ticker)
    render_console(snapshot, console)


@research_app.command("observe")
def observe(
    ticker: str = typer.Argument(..., help="Instrument, e.g. NVDA, RELIANCE.NS, BTC-USD"),
    start: str = typer.Option(..., "--start", help="First bar date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last bar date, YYYY-MM-DD"),
    timeframe: str = typer.Option("1d", "--timeframe"),
    context: str = typer.Option(None, "--context", help="Higher timeframe, e.g. 4h"),
    confirm: str = typer.Option(None, "--confirm", help="Lower timeframe for confirmation"),
    csv_dir: str = typer.Option(None, "--csv", help="Read bars from CSV files instead of Yahoo"),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Measure the baseline and list the weaknesses the next hypothesis would address."""
    from tradingagents.research.autopsy import autopsy_batch, recurring_weaknesses
    from tradingagents.research.hypothesis import diagnose

    store = _store(research_dir)
    table, _ = _load_table(ticker, timeframe, context, confirm, start, end, csv_dir, asset_type)
    policy, _ = _runtime(store)
    config = EngineConfig(costs=LoopPlan(ticker=ticker).costs, risk_policy=policy)
    spec = baseline_strategy(table.timeframe, table.context_timeframe)
    spec.validate_columns(table.features.columns)
    result = run_backtest(spec, table, config)
    console.print(f"\n[bold]{result.summary_line()}[/bold]")
    if result.metrics.profit_factor is None:
        console.print("[dim]profit factor is undefined with no losing trades[/dim]")
    autopsies = autopsy_batch(result.trades, policy=policy)
    weaknesses = recurring_weaknesses(autopsies)
    if weaknesses:
        table_output = Table(title="Recurring weaknesses", box=None, header_style="bold")
        for column in ("category", "trades", "repeatable", "suggested action"):
            table_output.add_column(column)
        for entry in weaknesses:
            table_output.add_row(entry["category"], str(entry["trades"]),
                                 "yes" if entry["repeatable"] else "no", entry["action"][:70])
        console.print(table_output)
    diagnostics = diagnose(result, table)
    if diagnostics:
        listing = Table(title="Testable hypotheses", box=None, header_style="bold")
        listing.add_column("#", justify="right")
        listing.add_column("weakness")
        listing.add_column("evidence")
        listing.add_column("proposed change")
        for index, diagnostic in enumerate(diagnostics, 1):
            listing.add_row(str(index), diagnostic.title[:60],
                            "; ".join(diagnostic.evidence)[:70], diagnostic.change.describe())
        console.print(listing)
        console.print("[dim]`research run` tests these in order and records every "
                      "decision, including rejections.[/dim]")
    else:
        console.print("[yellow]No weakness crossed the reporting thresholds; the baseline is "
                      "either sound or the sample is too small to say.[/yellow]")


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


@research_app.command("run")
def run(
    ticker: str = typer.Argument(..., help="Instrument to research"),
    start: str = typer.Option(..., "--start", help="First bar date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last bar date, YYYY-MM-DD"),
    timeframe: str = typer.Option("1d", "--timeframe"),
    context: str = typer.Option(None, "--context"),
    confirm: str = typer.Option(None, "--confirm"),
    csv_dir: str = typer.Option(None, "--csv"),
    asset_type: str = typer.Option("stock", "--asset-type"),
    hypotheses: int = typer.Option(DEFAULT_CONFIG.get("research_max_hypotheses", 2),
                                    "--hypotheses", help="How many hypotheses to test this run"),
    commission_bps: float = typer.Option(_COSTS["commission_bps"], "--commission-bps"),
    slippage_bps: float = typer.Option(_COSTS["slippage_bps"], "--slippage-bps"),
    spread_bps: float = typer.Option(_COSTS["spread_bps"], "--spread-bps"),
    tax_bps_sell: float = typer.Option(_COSTS["tax_bps_sell"], "--tax-bps-sell"),
    funding_bps_per_day: float = typer.Option(_COSTS["funding_bps_per_day"],
                                              "--funding-bps-per-day"),
    min_trades: int = typer.Option(DEFAULT_CONFIG.get("research_min_trades", 40),
                                   "--min-trades",
                                   help="Minimum trades to accept a candidate"),
    research_dir: str = typer.Option(None, "--research-dir"),
    html: str = typer.Option(None, "--html", help="Write the dashboard page to this path"),
) -> None:
    """Run the hypothesis loop over one instrument and record every decision."""
    from tradingagents.research.contracts import CostModel

    store = _store(research_dir)
    registry = StrategyRegistry(store)
    policy, risk = _runtime(store)
    table, data = _load_table(ticker, timeframe, context, confirm, start, end, csv_dir, asset_type)
    costs = CostModel(commission_bps=commission_bps, slippage_bps=slippage_bps,
                      spread_bps=spread_bps, tax_bps_sell=tax_bps_sell,
                      funding_bps_per_day=funding_bps_per_day)
    plan = LoopPlan(
        ticker=ticker, timeframes=[timeframe] + [tf for tf in (context, confirm) if tf],
        start=start, end=end, asset_type=asset_type, costs=costs, risk_policy=policy,
        criteria=AcceptanceCriteria(
            min_trades=min_trades,
            min_oos_trades=max(10, int(DEFAULT_CONFIG.get("research_min_oos_trades", 20))),
            max_holdout_uses=int(DEFAULT_CONFIG.get("research_holdout_budget", 3))),
        max_hypotheses=hypotheses,
        walk_forward_folds=int(DEFAULT_CONFIG.get("research_walk_forward_folds", 4)),
    )
    config = EngineConfig(initial_equity=plan.initial_equity, costs=costs,
                          risk_policy=policy)
    loop = HypothesisLoop(store, registry, plan, table=table, data=data, config=config)
    summary = loop.run()
    snapshot = publish_snapshot(store, summary=summary, ticker=ticker)

    for hypothesis in summary.hypotheses:
        colour = "green" if hypothesis.decision == "accepted" else "yellow"
        console.print(f"[{colour}]{hypothesis.hypothesis_id} {hypothesis.title} → "
                      f"{hypothesis.decision}[/{colour}]")
        console.print(f"  {hypothesis.explanation}")
        if hypothesis.candidate_version:
            console.print(f"  candidate version: {hypothesis.candidate_version}")
    if summary.best:
        console.print(f"\n[bold]Best candidate:[/bold] {summary.best}")
    if summary.no_trade_reason:
        console.print(f"\n[bold red]{summary.no_trade_reason}[/bold red]")
    decision = ("candidate accepted for paper testing" if summary.best else
                snapshot.get("no_trade_reason") or NO_TRADE)
    console.print(f"\nDecision: {decision}")
    console.print("Promotion is not automatic: run `research status`, then `research approve` "
                  "after paper testing.")
    if html:
        console.print(f"Dashboard written to {write_dashboard_html(snapshot, html)}")
    _ = risk  # the risk engine is exercised inside the loop; kept here for symmetry


# --------------------------------------------------------------------------------------
# versions, approvals, rollback
# --------------------------------------------------------------------------------------


@research_app.command("versions")
def versions(research_dir: str = typer.Option(None, "--research-dir")) -> None:
    """List strategy versions, their status and the champion."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    records = registry.versions()
    if not records:
        console.print("[yellow]No versions yet. Run `research init`.[/yellow]")
        return
    champion = registry.champion_version()
    table = Table(box=None, header_style="bold")
    for column in ("version", "name", "status", "stage", "OOS expectancy", "trades", "champion"):
        table.add_column(column)
    for record in records:
        metrics = (record.out_of_sample or {}).get("metrics") or {}
        table.add_row(
            record.version, record.spec.name[:32], record.status, record.stage,
            f"{metrics.get('expectancy_r', 0):+.3f}R" if metrics else "—",
            str(metrics.get("trades", "—")),
            "yes" if champion and record.version == champion.version else "")
    console.print(table)


@research_app.command("show")
def show(
    version: str = typer.Argument(..., help="Version to inspect, e.g. V02"),
    research_dir: str = typer.Option(None, "--research-dir"),
    json_output: bool = typer.Option(False, "--json", help="Print the stored record as JSON"),
) -> None:
    """Show one strategy version: its rules, evidence, status and approvals."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    try:
        record = registry.get(version)
    except Exception as exc:
        _fail(str(exc))
        return
    if json_output:
        console.print_json(json.dumps(record.as_dict(), default=str))
        return
    console.print(f"[bold]{record.version}[/bold] {record.spec.name} · {record.status} · "
                  f"{record.stage}")
    console.print(record.spec.describe())
    if record.changes_from_parent:
        console.print("\n[bold]Changes from parent[/bold]")
        for change in record.changes_from_parent:
            console.print(f"- {change}")
    for label, payload in (("In-sample", record.backtest), ("Out-of-sample", record.out_of_sample)):
        metrics = (payload or {}).get("metrics") or {}
        if metrics:
            console.print(f"\n[bold]{label}[/bold]: {metrics.get('trades', 0)} trades, "
                          f"expectancy {metrics.get('expectancy_r', 0):+.3f}R, "
                          f"PF {metrics.get('profit_factor')}, "
                          f"max DD {metrics.get('max_drawdown_pct', 0):.1%}")
    if record.walk_forward:
        console.print(f"\n[bold]Walk-forward[/bold]: {record.walk_forward.get('positive_share', 0):.0%} "
                      f"of {len(record.walk_forward.get('folds', []))} folds positive")
    if record.paper:
        console.print(f"[bold]Paper[/bold]: {record.paper.get('status')} · "
                      f"{record.paper.get('trades', 0)} trades · "
                      f"{record.paper.get('expectancy_r')}R")
    if record.approval:
        console.print(f"[green]Approved by {record.approval.get('approver')} "
                      f"at {record.approval.get('at')}[/green]")
    review = PromotionPipeline(registry, store).review(record.version)
    if review.blockers:
        console.print("\n[yellow]Blockers for promotion[/yellow]")
        for blocker in review.blockers:
            console.print(f"- {blocker}")
    else:
        console.print(f"\n[green]No blockers; furthest stage {review.furthest_stage}[/green]")


@research_app.command("compare")
def compare(
    baseline: str = typer.Argument(..., help="Baseline version"),
    candidate: str = typer.Argument(..., help="Candidate version"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Compare two versions: rules, in-sample, out-of-sample, paper, status."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    try:
        comparison = registry.compare(baseline, candidate)
    except Exception as exc:
        _fail(str(exc))
        return
    console.print(f"[bold]{baseline} → {candidate}[/bold]")
    for change in comparison["rule_changes"]:
        console.print(f"- {change}")
    table = Table(box=None, header_style="bold")
    for column in ("metric", "baseline", "candidate", "difference"):
        table.add_column(column)
    for name, payload in comparison["metrics"].items():
        table.add_row(name, str(payload["backtest"]["baseline"]),
                      str(payload["backtest"]["candidate"]),
                      str(payload["backtest"]["difference"]))
    console.print(table)
    console.print("[dim]Out-of-sample figures are in the stored records: `research show V0x`.[/dim]")


@research_app.command("promote")
def promote(
    version: str = typer.Argument(..., help="Version to make champion"),
    research_dir: str = typer.Option(None, "--research-dir"),
    force: bool = typer.Option(False, "--force", help="Promote even if gates have not passed"),
) -> None:
    """Make a paper-tested version the champion (keeps the previous one for rollback)."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    pipeline = PromotionPipeline(registry, store, _runtime(store)[1])
    try:
        review = pipeline.review(version)
        if review.blockers and not force:
            console.print("[red]Refusing to promote: " + "; ".join(review.blockers) + "[/red]")
            raise typer.Exit(code=1)
        registry.promote(version, require_accepted=not force)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(str(exc))
        return
    console.print(f"[green]{version} is now the champion.[/green] The previous champion is "
                  "retained; `research rollback <version>` restores it.")


@research_app.command("rollback")
def rollback(
    version: str = typer.Argument(..., help="Version to restore as champion"),
    reason: str = typer.Option(..., "--reason", help="Why the current version is being replaced"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Restore an earlier version as champion and record the reason."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    try:
        registry.rollback(version, reason=reason)
    except Exception as exc:
        _fail(str(exc))
        return
    console.print(f"[green]Rolled back to {version}.[/green] The rolled-back version keeps its "
                  "results and status history.")


@research_app.command("approve")
def approve(
    version: str = typer.Argument(..., help="Version that has completed paper testing"),
    approver: str = typer.Option(..., "--approver", help="Who is approving"),
    note: str = typer.Option("", "--note", help="Why"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Record explicit user approval: the only route to live eligibility."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    pipeline = PromotionPipeline(registry, store, _runtime(store)[1])
    try:
        approval = pipeline.request_approval(version, approver=approver, note=note)
    except Exception as exc:
        _fail(str(exc))
        return
    console.print(f"[green]Approved {version} ({approval['approval_id']}).[/green]")
    console.print("Live trading is still disabled. Enable it deliberately with "
                  f"`research live --enable --confirm {version} --approver {approver}`.")


@research_app.command("live")
def live(
    enable: bool = typer.Option(False, "--enable/--disable", help="Enable or disable live trading"),
    confirm: str = typer.Option("", "--confirm", help="Type the champion's version to confirm"),
    approver: str = typer.Option("", "--approver", help="Who is enabling it"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Enable or disable live eligibility. Disabled by default; never automatic."""
    store = _store(research_dir)
    registry = StrategyRegistry(store)
    pipeline = PromotionPipeline(registry, store, _runtime(store)[1])
    if not enable:
        registry.disable_live(by=approver or "user", reason="explicit disable")
        console.print("[yellow]Live trading disabled.[/yellow]")
        return
    try:
        result = registry.enable_live(approver=approver or "user", confirmation=confirm)
    except Exception as exc:
        _fail(str(exc))
        return
    console.print(f"[green]Live eligibility enabled for {result['version']}.[/green]")
    console.print("[yellow]This platform has no execution venue wired to it. Eligibility is a "
                  "record, not an order path; the risk engine still governs every entry.[/yellow]")
    console.print(json.dumps(pipeline.live_status(), indent=2))


# --------------------------------------------------------------------------------------
# machine learning
# --------------------------------------------------------------------------------------


@research_app.command("ml-train")
def ml_train(
    ticker: str = typer.Argument(...),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    timeframe: str = typer.Option("1d", "--timeframe"),
    context: str = typer.Option(None, "--context"),
    confirm: str = typer.Option(None, "--confirm"),
    csv_dir: str = typer.Option(None, "--csv"),
    horizon: int = typer.Option(5, "--horizon", help="Bars ahead for the direction label"),
    folds: int = typer.Option(4, "--folds"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Train a model on historical features, evaluated walk-forward only."""
    store = _store(research_dir)
    table, _ = _load_table(ticker, timeframe, context, confirm, start, end, csv_dir, "stock")
    try:
        dataset = build_dataset(table, horizon=horizon)
        outcome = train_with_oof(dataset, folds=folds)
    except Exception as exc:
        _fail(f"cannot train on this data: {exc}")
        return
    registry = ModelRegistry(store)
    record = registry.register(outcome)
    probabilities = outcome.oof_probabilities.dropna()
    labels = dataset.y.loc[probabilities.index]
    significance = auc_significance(probabilities.to_numpy(), labels.to_numpy())
    verdict = ml_adds_value(outcome, improvement_test=lambda: significance["p_value"])
    console.print(f"[bold]Model {record['model_id']}[/bold] registered as candidate.")
    console.print(f"Out-of-fold AUC {outcome.oof_metrics.get('auc')} "
                  f"(permutation p={significance['p_value']:.3f}) · "
                  f"Brier {outcome.oof_metrics.get('brier')} · "
                  f"calibration error {outcome.oof_metrics.get('calibration_error')}")
    console.print(f"In-sample AUC {outcome.in_sample_metrics.get('auc')} (reported for contrast "
                  "only; never used)")
    top = list(outcome.importances.items())[:5]
    console.print("Top features by permutation importance: "
                  + ", ".join(f"{name} {value:+.4f}" for name, value in top))
    if verdict.adds_value:
        console.print("[green]The model discriminates out of fold on unseen rows and is "
                      "calibrated enough to be considered. Promotion is still explicit: "
                      "`research ml-promote <id>`.[/green]")
    else:
        console.print("[yellow]The model is not used: " + "; ".join(verdict.reasons) + "[/yellow]")
    console.print("Use `research ml-status` to see the registry and drift checks.")


@research_app.command("ml-promote")
def ml_promote(
    model_id: str = typer.Argument(..., help="Model id from `research ml-status`"),
    reason: str = typer.Option(..., "--reason", help="Why this model should be champion"),
    research_dir: str = typer.Option(None, "--research-dir"),
) -> None:
    """Promote a registered model to champion, if its out-of-fold evidence allows it."""
    store = _store(research_dir)
    registry = ModelRegistry(store)
    try:
        record = registry.promote(model_id, reason=reason)
    except ValueError as exc:
        _fail(f"Refusing to promote {model_id}: {exc}")
        return
    console.print(f"[green]{record['model_id']} is now the champion model "
                  f"(out-of-fold AUC {record['oof'].get('auc')}).[/green]")
    console.print("A champion model can only reduce position size; it cannot alter risk limits.")


@research_app.command("ml-status")
def ml_status(research_dir: str = typer.Option(None, "--research-dir")) -> None:
    """Show registered models, the champion and retraining recommendations."""
    store = _store(research_dir)
    registry = ModelRegistry(store)
    records = registry.models()
    if not records:
        console.print("[yellow]No models registered. `research ml-train` trains one when the "
                      "data supports it.[/yellow]")
        return
    table = Table(box=None, header_style="bold")
    for column in ("model", "status", "OOF AUC", "calibration error", "features", "created"):
        table.add_column(column)
    champion_record = registry.champion() or {}
    for record in records:
        oof = record.get("oof") or {}
        status = ("champion" if record["model_id"] == champion_record.get("model_id")
                  else record.get("status", ""))
        table.add_row(record["model_id"], status,
                      str(oof.get("auc")), str(oof.get("calibration_error")),
                      str(len((record.get("artifact") or {}).get("features", []))),
                      str(record.get("created_at", ""))[:19])
    console.print(table)
    champion = registry.champion()
    console.print(f"Champion: {champion['model_id'] if champion else 'none (candidates must be promoted explicitly)'}")
    decision = registry.should_retrain(feature_version=str(
        (champion or {}).get("feature_version", "")))
    console.print(("Retrain recommended: " + "; ".join(decision["reasons"]))
                  if decision["retrain"] else "No retrain is due.")


# --------------------------------------------------------------------------------------
# risk policy
# --------------------------------------------------------------------------------------


@research_app.command("risk")
def risk_command(
    set_: Annotated[list[str] | None, typer.Option(
        "--set", help="key=value; only tightening is accepted")] = None,
    research_dir: Annotated[str | None, typer.Option("--research-dir")] = None,
) -> None:
    """Show the risk policy, or tighten it. Loosening a limit is refused."""
    store = _store(research_dir)
    policy = _policy(store)
    engine = RiskEngine(policy, store)
    if set_:
        changes: dict[str, object] = {}
        for item in set_:
            key, _, raw = item.partition("=")
            key = key.strip()
            if key not in RiskPolicy.__dataclass_fields__:
                _fail(f"unknown policy field {key!r}")
                return
            current = getattr(policy, key)
            try:
                if isinstance(current, bool):
                    changes[key] = raw.strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    changes[key] = int(raw)
                elif isinstance(current, float):
                    changes[key] = float(raw)
                else:
                    changes[key] = raw
            except ValueError:
                _fail(f"{key} expects a value like {current!r}")
                return
        proposed = RiskPolicy(**{**policy.as_dict(), **changes})
        review = engine.apply_policy_change(proposed)
        if not review.allowed:
            _fail("; ".join(review.reasons))
            return
        save_policy(store, proposed)
        policy = proposed
        console.print("[green]Policy tightened.[/green]")
    table = Table(title="Risk policy", box=None, header_style="bold")
    table.add_column("limit")
    table.add_column("value", justify="right")
    for key, value in policy.as_dict().items():
        table.add_row(key, str(value))
    console.print(table)
    console.print("[dim]The risk engine is authoritative: strategies and models can only "
                  "reduce size, never raise it.[/dim]")


# --------------------------------------------------------------------------------------
# dashboard page
# --------------------------------------------------------------------------------------


@research_app.command("dashboard")
def dashboard(
    research_dir: str = typer.Option(None, "--research-dir"),
    html: str = typer.Option(None, "--html", help="Write the standalone dashboard page here"),
    ticker: str = typer.Option(None, "--ticker"),
) -> None:
    """Render the Strategy Intelligence dashboard, optionally as a standalone page."""
    store = _store(research_dir)
    snapshot = publish_snapshot(store, ticker=ticker)
    render_console(snapshot, console)
    if html:
        console.print(f"Written: {write_dashboard_html(snapshot, html)}")


@research_app.command("hypotheses")
def hypotheses(
    research_dir: str = typer.Option(None, "--research-dir"),
    ticker: str = typer.Option(None, "--ticker"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """List recent hypotheses, including the rejected ones and why."""
    store = _store(research_dir)
    records = recent_hypotheses(store, ticker=ticker, limit=limit)
    if not records:
        console.print("[yellow]No hypotheses recorded yet.[/yellow]")
        return
    table = Table(box=None, header_style="bold")
    for column in ("hypothesis", "title", "decision", "status", "explanation"):
        table.add_column(column)
    for record in records:
        table.add_row(str(record.get("hypothesis_id")), str(record.get("title"))[:50],
                      str(record.get("decision")), str(record.get("status")),
                      str(record.get("explanation"))[:80])
    console.print(table)


@research_app.command("holdouts")
def holdouts(research_dir: str = typer.Option(None, "--research-dir")) -> None:
    """Show how often each untouched test window has been evaluated."""
    store = _store(research_dir)
    document = store.load_holdouts()
    windows = document.get("windows", {})
    if not windows:
        console.print("No holdout window has been used yet.")
        return
    table = Table(box=None, header_style="bold")
    table.add_column("dataset:window")
    table.add_column("uses", justify="right")
    for key, uses in sorted(windows.items()):
        table.add_row(key, str(uses))
    console.print(table)
    console.print("[dim]The acceptance gate refuses promotion once a window exceeds its "
                  "budget.[/dim]")
    _ = HoldoutLedger
