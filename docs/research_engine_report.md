# Strategy Intelligence: implementation report

Date: 2026-10-25 · Branch: `arena/b249c1bb-tradingagents` · Base: `e8f5503` (v0.7.0)

This report covers the research engine added to TradingAgents: what was built, how it
integrates with what was already here, the evidence that it works, and — just as
importantly — what it does **not** do.

The plan behind it, including the audit of the existing repository, is in
[`research_engine_plan.md`](research_engine_plan.md).

---

## 1. What was added

A new package, `tradingagents/research/` (22 modules, ~8,500 lines), a CLI sub-app
`tradingagents research` (`cli/research.py`), and an optional Strategy Intelligence section in
the existing HTML report. Existing modules were extended by optional parameters only; no
existing command, agent, prompt, data vendor path or report style was replaced.

| Layer | Module | What it does |
| --- | --- | --- |
| Evidence | `contracts.py` | Frozen dataclasses: `CostModel`, `Metrics`, `TradeRecord`, `GateResult`, `DataQualityReport`, enums, stable hashing |
| | `store.py` | Append-only JSONL streams + atomic JSON documents, credential scrubbing, corrupt-line tolerance |
| Data | `data.py` | Declared windows, resampling, exchange sessions/kill zones, quality gates, CSV or Yahoo sources |
| Features | `indicators.py` | EMA/SMA/VWAP/RSI/StochRSI/MACD/ATR/ADX/Supertrend/Bollinger/Fibonacci/opening range |
| | `smc.py` | Confirmed swings, BOS/CHoCH, liquidity sweeps, fair value gaps, order blocks, premium/discount, prior-day/week extremes |
| | `patterns.py` | Double top/bottom, head & shoulders, flags, triangles, ranges, compression, gaps |
| | `orderflow.py` | Volume/flow **proxies**, labelled as proxies; spread, slippage and fill-quality statistics |
| | `regimes.py` | Regime classification and multi-timeframe alignment |
| | `features.py` | One causal, versioned feature table (primary + coarser context + finer confirmation) |
| Research | `strategies.py` | Declarative rule DSL, candidate generation, duplicate detection, rule changes |
| | `engine.py` | Event-driven backtester: next-open fills, costs, stops/targets/trailing/partials, sizing, halts |
| | `risk.py` | Independent frozen policy, order intents, halts, tightening-only review |
| | `validation.py` | Purged/embargoed chronological splits, walk-forward, look-ahead probe, holdout ledger, acceptance gates |
| | `quant.py` | Bootstrap, permutation tests, multiple-testing control, Monte Carlo, trade statistics |
| | `scoring.py` | Composite score with penalties and an explicit `NO TRADE` |
| | `autopsy.py` | Per-trade post-mortem with categories and repeatability |
| | `versions.py` | Immutable V01… registry, comparison, promotion, approval, rollback |
| | `pipeline.py` | Paper-first promotion stages and the approval gate |
| | `ml.py` | NumPy logistic regression, k-means, out-of-fold evaluation, calibration, drift, model registry |
| | `hypothesis.py` | The loop: observe → diagnose → hypothesise → test → validate → decide → record |
| `paper.py` | Forward paper trading: sessions, history, verdicts |
| `server.py` | The browser dashboard (snapshot + paper-trade form, no other actions) |
| Delivery | `dashboard.py` | Snapshot, Rich console view, HTML section, standalone page |
| | `cli/research.py` | `init, observe, run, status, versions, show, compare, promote, rollback, approve, live, risk, paper, serve, dashboard, hypotheses, holdouts, ml-train, ml-status, ml-promote` |

Integration points touched in existing files:

| File | Change |
| --- | --- |
| `cli/main.py` | registers `research_app` as the `research` sub-app (3 lines + import) |
| `tradingagents/report_html.py` | `render_report(..., research_section=None)` appends a contents entry and a section |
| `tradingagents/reporting.py` | `write_report_tree(..., research_section=None)` forwards it |
| `tradingagents/default_config.py` | `research_*` keys (store path, thresholds, costs, LLM-idea switch, live flag) + env overrides |
| `tradingagents/dataflows/vendors/yahoo/ohlcv.py` | new `load_ohlcv_window(symbol, start, end, interval=…)` so research never imports a vendor library |

---

## 2. How the pieces fit

```
Yahoo / CSV ──► MarketData (quality gate) ──► FeatureTable (causal, versioned)
                                                    │
                                  StrategySpec ─────► engine.run_backtest
                                                    │      │
                                       risk.RiskEngine (authoritative, can only shrink)
                                                    │
                          validation gates ──► scoring ──► NO TRADE | candidate version
                                                    │
                            versions V01… ◄── hypothesis loop ──► store (JSONL evidence)
                                                    │
                                   pipeline: backtest → OOS → walk-forward → robustness
                                             → paper → risk review → user approval
                                                    │
                                        dashboard: Rich console + HTML section
```

The loop writes **two** records per hypothesis: one when it is opened and one when it is
decided, with the gate that failed and the number that failed it. `NO TRADE` is a first-class
outcome, not an error, and it always carries its measured reason.

---

## 3. Verification

### Test suite

```
$ .venv/bin/python -m pytest -q
1575 passed, 1 skipped, 1 deselected, 100 subtests passed    (58.8 s)
```

Baseline before this work was `1361 passed, 1 skipped`. The addition is **194 tests** in 14
new modules:

| Test module | Covers |
| --- | --- |
| `test_research_data.py` (14) | normalisation, resampling, sessions, quality gates, CSV source, Yahoo delegation |
| `test_research_features.py` (28) | indicators, SMC/ICT, patterns, regimes, feature assembly, genuine MTF causality |
| `test_research_engine.py` (14) | next-open fill, costs vs gross, stop-before-target, gaps, sizing, risk gate, halts, partials, short side |
| `test_research_risk.py` (22) | refusal to loosen every limit, emergency stop, halts, exposure and frequency caps, cap-not-approve sizing |
| `test_research_store.py` (11) | append-only streams, secret scrubbing, corrupt lines, artifacts, registry documents |
| `test_research_validation.py` (17) | purged splits, walk-forward, planted-leak detection, holdout budget, every gate both ways |
| `test_research_scoring.py` (10) | composite scoring, penalties, ranking, `NO TRADE` selection |
| `test_research_autopsy.py` (15) | variance vs mistake, stop sizing, exit discipline, repeatability |
| `test_research_versions.py` (11) | immutability, status history, promotion, approval, rollback, live gating |
| `test_research_deployment.py` (11) | stage chain, paper-first refusal, risk review, rollback withdrawing live, emergency stop |
| `test_research_ml.py` (14) | label tail dropped, purged folds, OOF-only reporting, calibration, drift, registration ≠ promotion |
| `test_research_hypothesis.py` (14) | full session, rejections with gates, no repeated experiment, LLM can only propose, data-quality stop |
| `test_research_dashboard.py` (8) | snapshot honesty, console panels, HTML escaping, report integration |
| `test_research_guardrails.py` (5) | the whole feature table and the backtest decisions are unchanged when the future is truncated |
| `test_research_paper.py` (11) | forward windows, pending vs failed, append-only sessions, status transitions, cost share, dashboard wiring |
| `test_research_server.py` (7) | page rendering, form token, redirect message, JSON snapshot, escaping, refused sessions |

```
$ .venv/bin/ruff check .
All checks passed!
```

### Counterfactual look-ahead test (the important one)

`tests/test_research_guardrails.py` builds the feature table twice — once on the full history,
once truncated 200 bars earlier — and requires every shared column to be identical on every
bar both runs could see. It repeats this for a 4h-context-on-1h-primary table (where an
as-of merge is easiest to get wrong) and for the backtest's trades. A planted leak (a
whole-sample z-score) is included as a positive control and is caught.

### CLI, end to end, on a CSV fixture

```
$ python -m cli.main research init --research-dir /tmp/res            # seeds V01
$ python -m cli.main research run SYNTH --start 2022-01-03 --end 2024-09-06 \
      --csv /tmp/csv --research-dir /tmp/res --hypotheses 2 --html /tmp/dash.html
$ python -m cli.main research status --research-dir /tmp/res
$ python -m cli.main research versions | hypotheses | risk | holdouts | live | ml-status
$ python -m cli.main research promote V02 --research-dir /tmp/res     # refused, see below
$ python -m cli.main research rollback V01 --reason test              # works
```

Observed on a 700-bar synthetic fixture (test-only data, 99 warm-up bars for `sma_200`):

* 148 feature columns, feature version `83d9251e79c1`, data-quality `ok` at 100% coverage;
* the loop tested one hypothesis and **rejected** it, citing `sample_size (6 trades)`,
  `oos_sample (2 trades)`, `oos_expectancy (-1.031R)`, `beats_baseline`, `cost_stress`,
  `walk_forward_consistency (50% of 4 folds)`, `parameter_sensitivity (100% fragile across
  16 variants)` and `statistical_significance (p=1.000)`;
* the verdict was `NO TRADE: every proposed change was rejected; the baseline does not clear
  the gates on this sample either`;
* `promote` and `approve` refused with the failing stage and the measured value;
* `rollback` moved the champion back to V01 and withdrew live eligibility;
* `ml-train` registered a model with out-of-fold AUC 0.573 (permutation p=0.005), Brier 0.410,
  calibration error 0.407, and **refused to use it** because probabilities that miscalibrated
  cannot size positions;
* the HTML dashboard rendered from the same snapshot and the report embedded the section.

This is the behaviour I wanted to see: the engine says no, and says why, instead of
producing a nice-looking equity curve.

### Safety properties, as tests rather than as prose

* Every loosening of a risk limit is refused and named (`test_research_risk`).
* A model that cannot discriminate never becomes champion; registration is not promotion.
* A rejected hypothesis is retrievable, and a second session does not re-run it.
* `live_eligible` requires paper testing **and** a named approver; enabling live additionally
  requires the champion's version typed as confirmation.
* A rollback withdraws live permission immediately.
* The research engine never imports `yfinance`, never contacts a broker, and has no order
  path at all — `test_layering.py` still passes.

---

## 3b. Paper trading and the browser dashboard

`tradingagents/research/paper.py` and `tradingagents/research/server.py` were added after the
first round of this report, for the one question the engine could not answer: *what does the
version do on bars that arrived after the decision?*

| Piece | What it does |
| --- | --- |
| `paper.run_paper_session` | Trades a version forward from the end of its recorded out-of-sample window, costs and risk policy included, and appends a session record |
| `paper.paper_history` / `latest_paper` | Read the append-only `paper_sessions.jsonl` stream |
| `research paper` | CLI: run a session, `--history` to list them |
| `research serve` | Browser dashboard (standard-library HTTP) with the snapshot and a paper-trade form |
| Dashboard section | A "Paper trading" panel in the console view and two new rows in the HTML section |

Properties enforced by `tests/test_research_paper.py` (11 tests) and
`tests/test_research_server.py` (7 tests):

* a session starts strictly **after** the version's out-of-sample window and says how many
  forward bars it saw; a window with too few bars is `pending`, never a fake zero-trade result;
* a session records `start_source`: `version_split` when it came from the recorded split,
  `declared` when a person supplied the date — and the note says plainly that a declared window
  overlapping the selection sample is not a forward test;
* sessions are append-only and the version keeps the latest as its `paper` record, which is
  exactly what `PromotionPipeline.review` reads before an approval can be requested;
* a passing session moves the version into `Paper Testing`; a failing one leaves it unapproved;
* the dashboard cannot approve, promote, roll back, change the risk policy or enable live
  trading — there is no route for it — and a POST without the per-process form token is
  refused (403). Everything from the store is HTML-escaped.

Observed on the preview fixture (1700 synthetic daily bars, research window 2015-01-02 →
2020-06-30, forward window 2020-07-01 → 2021-07-08): `V01 paper passed: 14 trades,
expectancy +0.678R, net +4,811.51, max DD 1.6%` — recorded, shown in the dashboard, and still
*not* approvable, because the promotion review found earlier stages empty. That is the
intended shape of the answer.

## 4. Known limitations (read these before trusting a number)

1. **No live trading, no broker, no execution.** There is no venue in this repository. The
   engine computes what it would have done and refuses to pretend otherwise.
2. **Daily history is the practical limit for a 40-trade floor.** On five years of daily bars
   the baseline took ~8 trades; meeting `min_trades=40` out of sample needs a longer, or
   intraday, series. The gate refuses rather than lowering the bar — the threshold is
   configurable (`research_min_trades`), the evidence requirement is not.
3. **Order flow is a proxy.** Without tick or depth data, "CVD" is derived from candle volume
   and shape. Every such column is prefixed `proxy_` and the dashboard labels it.
4. **Synthetic fixtures are for tests only.** No dashboard or CLI number can be produced from
   them; `test_research_data.py` asserts the store/CLI paths require a declared source.
5. **The LLM idea provider is a seam, not a shipped feature.** `research_llm_ideas` defaults to
   False and the provider only emits declarative rule changes, which go through the same gates
   as any other idea. It cannot size, approve or promote.
6. **Multiple testing is controlled, not eliminated.** `quant.multiple_testing` adjusts the
   p-value for the number of experiments in the family; searching a thousand rule changes will
   still eventually find one that looks good.
7. **Walk-forward folds use expanding windows of the same rules.** Parameters are not re-fitted
   per fold; what varies is the sample. Parameter sensitivity is measured separately.
8. **The regime classifier is rule-based** (ADX/ATR percentiles, structure, compression), not a
   fitted model, and it says `unstable` when the evidence conflicts.
9. **Single instrument, single position.** The risk policy caps open positions at one; there is
   no portfolio construction here, and the existing `tradingagents/portfolio.py` is untouched.
10. **Results are about the past.** Every gate is evidence about a historical sample; none of
    it is a forecast, and the `NO TRADE` message says so.

---

## 5. How to use it

```bash
# 1. Seed the research memory (V01 baseline, live disabled)
python -m cli.main research init

# 2. Look at what the baseline actually does before hypothesising
python -m cli.main research observe --ticker AAPL --start 2015-01-01 --end 2025-01-01

# 3. Run a session: it will accept, reject, or say NO TRADE — and record which
python -m cli.main research run AAPL --start 2015-01-01 --end 2025-01-01 \
    --hypotheses 3 --html results/research/dashboard.html

# 4. Read the evidence
python -m cli.main research status            # Strategy Intelligence dashboard
python -m cli.main research hypotheses        # every attempt, with the gate that failed
python -m cli.main research versions          # V01.., status, stage, champion
python -m cli.main research risk              # current policy; only tightening is accepted

# 5. Only after paper testing and a user decision
python -m cli.main research approve V02 --approver you
python -m cli.main research live --enable V02  # requires the typed confirmation
```

The HTML section appears in the normal report when the research snapshot exists; the same
snapshot can be written as a standalone page (`research dashboard --html …`) that reuses the
report's CSS rather than introducing a second theme.

---

## 6. Files

Added:

```
tradingagents/research/{__init__,contracts,store,data,indicators,smc,patterns,orderflow,
                        regimes,features,strategies,quant,engine,risk,validation,scoring,
                        autopsy,versions,ml,hypothesis,pipeline,dashboard}.py
cli/research.py
tests/{research_fixtures,test_research_paper,test_research_server,test_research_data,test_research_features,test_research_engine,
       test_research_risk,test_research_store,test_research_validation,test_research_scoring,
       test_research_autopsy,test_research_versions,test_research_deployment,
       test_research_ml,test_research_hypothesis,test_research_dashboard,
       test_research_guardrails}.py
docs/research_engine_plan.md
docs/research_engine_report.md
```

Modified: `cli/main.py`, `tradingagents/report_html.py`, `tradingagents/reporting.py`,
`tradingagents/default_config.py`,
`tradingagents/dataflows/vendors/yahoo/ohlcv.py`.

Not modified: the agent graph, prompts, analysts, debate, risk debate, memory log,
`tradingagents/backtest.py`, `tradingagents/portfolio.py`, the Markdown report, and the
existing report CSS.
