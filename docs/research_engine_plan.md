# Strategy Intelligence: audit and implementation plan

This document records what the repository already had, what was missing, and how the
research engine was added **without** rebuilding anything that already worked. It is the
plan the implementation followed; the outcome is in
[`research_engine_report.md`](research_engine_report.md).

## 1. Audit of the existing project

The brief described a "Clowdbot Trader" with a dashboard, a risk engine and a strategy
registry. This repository does not contain any of that: it is TradingAgents (LangGraph
multi-agent analysis, a Typer CLI, a Markdown/HTML report, a decision-quality backtest and
a memory log). Nothing named `clowdbot` exists in the tree. The audit therefore mapped every
capability in the brief onto what actually exists here.

| Capability in the brief | Where it lives today | State | Decision |
| --- | --- | --- | --- |
| OHLCV acquisition | `tradingagents/dataflows/vendors/yahoo/ohlcv.py`, cache in `tradingagents/dataflows/files.py` | working | **reuse** through a thin adapter (`research/data.py`) that also accepts CSV |
| Atomic / locked writes | `files.py::replace_file`, `files.py::locked` | working | **reuse** for every registry and memory write |
| Decision-quality evaluation | `tradingagents/backtest.py` | working, explicitly *not* a portfolio simulator | **leave untouched**; the file itself forbids growing into one |
| Portfolio bookkeeping | `tradingagents/portfolio.py` | working, a standing book for a single analysis | **not extended**; no fills, no cash ledger |
| Outcome tracking / lessons | `memory/log.py`, `memory/reflection.py`, `memory/settlement.py` | working, markdown decision memory | **not duplicated**; the research memory is a separate append-only JSONL store, because the two records answer different questions (a rating's outcome vs a rule set's out-of-sample evidence) |
| Multi-agent debate and rating | `tradingagents/graph/`, `agents/` | working | **not touched** |
| Rich CLI | `cli/main.py`, `cli/display.py` | working | **extended** with a `research` sub-app; no existing command changed |
| HTML report | `tradingagents/report_html.py` (+ `reporting.py`) | working, self-contained, CSP `default-src 'none'` | **extended** with an optional section; theme and CSS untouched |
| Configuration | `tradingagents/default_config.py` | working, env-overridable | **extended** with `research_*` keys |
| Cost-aware backtester (stops, targets, partials, sizing, halt rules) | — | **missing** | new `research/engine.py` |
| Technical / SMC / pattern / regime features | partial: `stockstats` is used inside analyst tools | **partial** | new `research/indicators.py`, `smc.py`, `patterns.py`, `regimes.py` — pandas/NumPy only |
| Independent, authoritative risk engine | — | **missing** (the "risk debate" is prose from an LLM) | new `research/risk.py`, frozen policy, tightening-only |
| Strategy version registry, promotion, rollback | — | **missing** | new `research/versions.py`, `pipeline.py` |
| Statistical validation, walk-forward, purged splits, holdout ledger | — | **partial** (no purging, no ledger) | new `research/validation.py`, `quant.py` |
| ML with honest out-of-fold evaluation | — | **missing** | new `research/ml.py` (NumPy only) |
| Trade autopsy | — | **missing** | new `research/autopsy.py` |
| Research loop (hypothesis → experiment → decide → record) | — | **missing** | new `research/hypothesis.py` |
| Composite scoring with an explicit NO TRADE | — | **missing** | new `research/scoring.py` |
| Strategy Intelligence dashboard | — | **missing** | new `research/dashboard.py` + HTML section |

Two rules came out of the audit:

1. **Do not extend `tradingagents/backtest.py`.** Its docstring states it must not grow into a
   portfolio simulator. The cost-aware engine therefore lives in `research/engine.py`, and the
   two do not share state or semantics.
2. **Do not add a second design system.** The dashboard reuses `report_html._CSS` and the
   existing `<dl class="run">` layout, so the report still looks like one product.

## 2. Design

```
tradingagents/research/
  contracts.py    frozen dataclasses: CostModel, Metrics, TradeRecord, GateResult, enums
  store.py        append-only JSONL streams + atomic JSON documents + secret scrubbing
  data.py         market data loading, quality gates, resampling, sessions, exchanges
  indicators.py   trend, momentum, volatility, volume, Fibonacci, opening range
  smc.py          swings, BOS/CHoCH, sweeps, FVG, order blocks, premium/discount
  patterns.py     classic patterns, ranges, compression, gaps
  orderflow.py    volume proxies, explicitly labelled; spread/slippage/fill quality
  regimes.py      regime classification + multi-timeframe alignment
  features.py     one causal feature table (primary + context + confirmation)
  strategies.py   declarative rule DSL, candidate generation, duplicate detection
  quant.py        bootstrap, permutation tests, deflated significance, Monte Carlo
  engine.py       event-driven backtester: costs, stops, targets, partials, halts
  risk.py         independent policy, order intents, halts, tightening-only changes
  validation.py   chronological splits with purge/embargo, walk-forward, leakage probe, gates
  scoring.py      composite score, ranking, NO TRADE verdict
  autopsy.py      per-trade post-mortem with categories and repeatability
  versions.py     immutable V01… registry, promotion, approval, rollback
  pipeline.py     paper-first stage checks and the approval gate
  ml.py           NumPy logistic regression, k-means, OOF folds, calibration, drift
  hypothesis.py   observe → hypothesise → test → decide → record loop
  dashboard.py    snapshot, Rich console view, HTML section, standalone page
```

### Data flow

```
Yahoo / CSV ──► MarketData (quality gate) ──► FeatureTable (causal, versioned)
                                                     │
                                    StrategySpec ────► engine.run_backtest
                                                     │            │
                                        risk.RiskEngine (authoritative gate)
                                                     │
                              validation gates ──► scoring ──► NO TRADE | candidate
                                                     │
                              versions.V01… ◄── hypothesis loop ──► store (JSONL)
                                                     │
                                    pipeline (paper → risk review → approval)
                                                     │
                                        dashboard (Rich + HTML section)
```

### Decisions that shape behaviour

* **Nothing is computed from a future bar.** Rules are evaluated on a closed bar and fill at
  the next bar's open; features use only trailing windows; a coarser context frame is joined
  with `merge_asof` on the *time its bar became known*. `validation.probe_lookahead`
  rebuilds the feature table on truncated history and compares rows, so any leak becomes a
  failing test rather than a nice-looking equity curve.
* **The risk engine is not advisory.** Strategies declare a sizing *request*; `RiskEngine`
  applies the policy cap and can only ever shrink it. Policy changes are reviewed by
  `weakened_by` and refused unless every field is equal or tighter.
* **A rejection is a result.** Every hypothesis, including the ones that fail, is written to
  the log with the gate that failed and the number that failed it. The loop records
  `NO TRADE` — with the measured reason — rather than promoting the least bad idea.
* **Paper first, approval second, live never by default.** `live_eligible` requires a named
  user's approval after paper testing; enabling live requires typing the champion's version
  and is refused if the champion is not approved. A rollback immediately withdraws live
  permission.
* **ML is optional and cheap to audit.** NumPy logistic regression with stored coefficients,
  means and scales; expanding-window folds with a purge gap of one label horizon; AUC,
  Brier, calibration error and permutation p-value all reported together. A model that cannot
  be calibrated does not size positions, and registration never makes a model champion.

## 3. Implementation phases

Each phase ended with its own tests and left the tree green.

| Phase | Work | Verified by |
| --- | --- | --- |
| 1 | `contracts`, `store`, `data` (+ quality gates, resampling, sessions) | `tests/test_research_data.py`, `test_research_store.py` |
| 2 | `indicators`, `smc`, `patterns`, `orderflow`, `regimes`, `features` | `tests/test_research_features.py` |
| 3 | `strategies`, `engine`, `risk` | `tests/test_research_engine.py`, `test_research_risk.py` |
| 4 | `quant`, `validation`, `scoring`, `autopsy` | `test_research_validation.py`, `test_research_scoring.py`, `test_research_autopsy.py` |
| 5 | `versions`, `pipeline` | `test_research_versions.py`, `test_research_deployment.py` |
| 6 | `ml` | `test_research_ml.py` |
| 7 | `hypothesis` loop | `test_research_hypothesis.py` |
| 8 | `dashboard`, CLI, report hook, config keys | `test_research_dashboard.py`, CLI smoke |
| 9 | Counterfactual look-ahead and cost-stress guardrails | `test_research_guardrails.py` |

## 4. Map of the brief onto the implementation

| Requirement | Implemented as |
| --- | --- |
| Continuous hypothesis loop | `hypothesis.HypothesisLoop` (observe → diagnose → hypothesise → test → validate → decide → record) |
| Modular ML/AI layer | `ml.py` + `IdeaProvider` protocol (diagnostic or LLM), separate from the strategy engine |
| Strategy lab | `strategies.generate_candidates`, `duplicate_candidates`, `apply_rule_change` |
| Composite scoring and NO TRADE | `scoring.score_strategy`, `select_champion`, `no_trade_explanation` |
| Trade autopsies by mistake category | `autopsy.autopsy_trade/autopsy_batch/recurring_weaknesses` |
| Immutable strategy versions V01… | `versions.StrategyRegistry` (no overwrite, status history, rollback) |
| Anti-overfitting gates | `validation.acceptance_gates` (sample, OOS, walk-forward, cost stress, sensitivity, holdout budget, multiple testing, leakage) |
| Independent risk engine | `risk.RiskEngine` + frozen `RiskPolicy`, tightening-only review |
| Paper first, explicit approval, live disabled | `pipeline.PromotionPipeline`, `versions.enable_live`, `research live` |
| Persistent audit memory | `store.ResearchStore` JSONL streams + `dashboard` snapshot |
| Strategy Intelligence dashboard | Rich console view + HTML section in the existing report |
| Integration, not a rewrite | new package only; existing modules extended by optional parameters |

## 5. Non-goals

* No broker integration, no order routing, no capital allocation.
* No fabricated microstructure: order-flow features are labelled proxies and the dashboard
  says so.
* No claim that a backtest predicts future returns; every gate is evidence about the past.
