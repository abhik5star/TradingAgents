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
| `feeds.py` | Free-feed registry, standard-library WebSocket client, message parsers, candle builder, feed hub |
| `livepage.py` + `assets/*.js` | The 3D market view: server-rendered HTML plus hand-written WebGL (scene math, renderer, page controller) |
| `server.py` | The browser dashboard (snapshot + paper-trade form, no other actions) and the live view routes |
| Delivery | `dashboard.py` | Snapshot, Rich console view, HTML section, standalone page |
| | `cli/research.py` | `init, observe, run, status, versions, show, compare, promote, rollback, approve, live, risk, paper, serve, feeds, dashboard, hypotheses, holdouts, ml-train, ml-status, ml-promote` |

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
1647 passed, 1 skipped, 1 deselected, 100 subtests passed    (60.3 s)
```

Baseline before this work was `1361 passed, 1 skipped`. The addition is **286 tests** across
18 new modules:

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
| `test_research_server.py` (10) | page rendering, form token, redirect message, JSON snapshot, escaping, refused sessions, `/feeds` and `/api/feeds`, no store writes |
| `test_research_feeds.py` (43) | WebSocket handshake/masking/fragmentation/ping/close over a socket pair, provider parsers, candle rolling, replay determinism, poll labels, hub snapshot, CLI feed builder |
| `test_design_board.py` (7) | the design file is self-contained, its layers resolve, its scripts parse, and its palette matches the CSS the app ships |
| `test_research_livepage.py` (19) | page carries the numbers without JS, no external references, boot payload escaping, sparklines, Node syntax + geometry + a full stub-GL render |

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

## 3c. Free feeds and the 3D market view

The last piece answers a different question from the rest of the engine: *what is the market
doing right now, and can I see it without lying to myself about where the numbers come from?*

| Piece | What it does |
| --- | --- |
| `feeds.PROVIDERS` | The registry of routes: `binance`, `coinbase`, `kraken` (keyless public WebSockets), `generic` (a broker endpoint you supply), `poll` (delayed quotes from the data layer), `replay` (simulated) |
| `feeds.WebSocket` | RFC 6455 client in the standard library: handshake and accept-key check, masking, continuation frames, ping/pong, close |
| `feeds.BarSeries` | Ticks → minute candles, plus the candle being formed, change, volume and tick count |
| `feeds.FeedHub` | One thread per feed, one lock, one `state()` snapshot for the API |
| `livepage.render_live_page` | The server-rendered page: legend, KPI cards, tape table, sparklines, provider and broker tables, boot payload |
| `assets/scene-math.js` | Column-major matrices, orbit camera, box geometry for candles — DOM-free, so the tests run it under Node |
| `assets/scene.js` | WebGL 1 renderer: two shader programs (solid candles with a directional light and depth fog, blended grid lines), pointer orbit, wheel zoom |
| `assets/boot.js` | Polls `/api/feeds`, rebuilds geometry only when a candle actually changed, floats price tags over the newest candle by projecting them with the scene's own camera, drives the legend and the table, and survives a failed poll |

**What is actually free.** Crypto is genuinely free and keyless: Binance `stream.binance.com`,
Coinbase `ws-feed.exchange.coinbase.com` and Kraken `ws.kraken.com/v2` all publish documented
public streams with no account, no key and sub-second latency. Indian equities are not: NSE and
BSE license their real-time data through brokers. Zerodha Kite (binary ticker), Upstox
(protobuf market feed) and Angel One SmartStream all require an API key from your own funded
account, and Kite Connect is a paid plan. There is no keyless NSE WebSocket, and any page
claiming otherwise is either delayed or unofficial. `research feeds` prints this as a table so
the answer to "is there a free Indian real-time feed?" is something you can point at, and the
dashboard offers the two honest routes: `generic:` with your broker's endpoint and key, or
`poll:` with delayed quotes labelled *delayed* on every panel.

**Why the scene is hand-written.** The page is served by a local process whose only job is to
show you your own evidence; it must work with the network switched off, which rules out Three.js
from a CDN and every icon font. So the renderer is ~250 lines of WebGL in the package
(`assets/*.js`, shipped as package data), the shaders are inline, and the geometry is built by a
DOM-free module that the test suite executes under Node. The tests there are not smoke tests:
one renders a frame against a stub GL context and asserts two draws (grid lines, then triangles),
that the triangle count equals the geometry's vertex count, and that every uploaded matrix is a
finite 4×4.

**What the scene shows.** Each symbol is a track along Z; each candle is a lit box with a
wick column and a faint tick on the floor, so a high/low range reads as depth rather than as a
line. The camera orbits a centred target, the geometry is re-centred on the origin when the
symbol count or candle count changes, and the fit is computed from the widest track
(`radius = width * 0.8 + 1.8`, clamped) so three symbols and one symbol both frame sensibly. A
`projectPoint` call turns world coordinates into CSS pixels, which is what keeps a live price
tag pinned to the newest close while the camera moves.

**Degradation, in order.** No JavaScript → the server-rendered table, cards and sparklines are
already in the HTML. No WebGL → the stage shows an explanation over the table. No feed →
"no feeds configured" rather than an empty chart. Unreachable endpoint → the row says
`unavailable` with the error. Simulation → `simulated` in the legend, the card, the table and
the API.

**A connected feed that says nothing useful.** A broker feed that accepts the socket, exchanges
heartbeats and never sends a field this parser recognises would otherwise sit there as a flat
line. After twenty messages without a tick the feed records
`20 messages arrived but none carried a recognisable price; check the subscribe payload and the
field names this provider uses`, and retracts it the moment a price arrives. That message
survives a disconnect, because losing the connection should not erase the reason the feed was
already useless.

**One protocol limit, stated rather than hidden.** `generic` reads JSON text frames — Angel
One's SmartStream JSON, or any broker that publishes `last_price`/`last_traded_quantity`.
Zerodha's Kite ticker and Upstox v3 send binary protobuf, which this parser cannot read; they
need a decoder before they can be attached. The provider table says so, and the silent-feed
warning above is what you would see if you tried.

Observed in this sandbox, which blocks outbound WebSocket and TLS to exchanges (verified by
handshake failure to both Binance and Coinbase before writing the code): three `replay:` tracks
rendered and ticked at 8 bars/s with `simulated` on every label, and `poll:RELIANCE.NS` reported
`VendorUnavailableError: Yahoo Finance request failed: SSLError` in its row — exactly the honest
answer. The `binance:`/`coinbase:` paths are exercised in the suite against a socket pair and a
stub transport, not against the live venue, so their first real run will be on a machine with
egress.

**The design file.** `docs/design/` holds the screens as a single portable HTML board
(`strategy-intelligence.html`): cover, dashboard at 1440, the same at 390, the 3D view, a
component sheet and a handoff page with tokens and redlines. It is generated from the app rather
than drawn beside it — the numbers come from the preview store, the scene is the real geometry
projected by the real camera (`make_scene_svg.js` imports `scene-math.js`), and the palette is
`tokens.json`, which `tests/test_design_board.py` checks against the CSS in `server.py` and
`livepage.py`. The build refuses to emit a file that would fetch anything at runtime, and
`tokens.tokens-studio.json` carries the same palette in the format Figma imports as variables.

**The one rule that did not change.** A live tape is display-only. `run_paper_session` and
`run_backtest` read loaded bars; nothing in `feeds.py` can reach the store, and
`test_the_live_view_never_writes_to_the_research_store` asserts the streams are untouched after
a page load. Watching a chart cannot change what the engine believes.

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
11. **The crypto feeds are unverified against the live venues here.** This sandbox's egress proxy
    closes the TLS handshake to both Binance and Coinbase, so the WebSocket code is proven
    against a real socket pair and stub transports, not against the exchanges. The endpoints,
    subscribe payloads and message shapes follow the published docs, and every feed reports its
    own failure loudly rather than guessing.
12. **No free Indian real-time feed exists.** Delayed polling is the keyless option and it is
    labelled delayed everywhere; a broker stream needs your credentials and is *your* feed, with
    your broker's terms attached. Tick data is not persisted — replaying it into a backtest
    would be a deliberate act of your own.
13. **Candles are one-minute and in-memory only.** The tape is a rolling window (240 candles per
    feed); restarting the process loses it. Nothing here builds history.

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

# 4b. Look at the market (display only; never an input to the steps above)
python -m cli.main research feeds                       # what is free, and what India requires
python -m cli.main research serve --feed binance:BTCUSDT --feed coinbase:BTC-USD
python -m cli.main research serve --feed replay:SYNTH --replay-csv SYNTH=bars.csv
#    then open http://localhost:32909/feeds for the 3D view

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
                        autopsy,versions,ml,hypothesis,pipeline,dashboard,paper,server,
                        feeds,livepage}.py
tradingagents/research/assets/{scene-math,scene,boot}.js
cli/research.py
tests/{research_fixtures,test_research_paper,test_research_server,test_research_feeds,
       test_research_livepage,test_research_data,test_research_features,test_research_engine,
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
