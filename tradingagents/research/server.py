"""The Strategy Intelligence web dashboard: look at the evidence, take a paper trade.

Why this exists
---------------
The CLI already prints the same snapshot and the HTML report embeds it, but both are
read-only artifacts. This serves that page from the research store and adds exactly
one action: **run a paper-trading session for a version**. That is the one step in
the research loop that a person should be able to trigger while looking at the
evidence — it risks no capital, changes no rules and grants no permission.

What it deliberately cannot do
------------------------------
* No approve, promote, rollback, live-enable or risk-policy endpoints. Those stay in
  the CLI, because they are the decisions the brief insists a human makes
  deliberately, with the champion's version typed out. A button in a browser is the
  wrong affordance for "grant live trading".
* No order path, no broker, no scheduler, no background thread: a request runs one
  forward simulation and writes one record.
* No new dependency: :mod:`http.server` and the standard library only.

The same process also serves ``/feeds`` — the 3D market view built from
:mod:`tradingagents.research.livepage` and the free feed hub in
:mod:`tradingagents.research.feeds`. That page is display-only: it reads a feed hub
and never writes to the store, so a live tape cannot leak into research evidence.

The page is server-rendered from the same snapshot the report uses, so there is one
source of truth and one theme. A POST must carry the per-process form token, which
keeps a random page in another tab from triggering a session.
"""

from __future__ import annotations

import json
import secrets
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from tradingagents.research.contracts import CostModel, Timeframe
from tradingagents.research.dashboard import build_snapshot, render_section_html
from tradingagents.research.data import (
    CsvDataSource,
    FrameDataSource,
    YahooDataSource,
    load_market_data,
)
from tradingagents.research.engine import EngineConfig
from tradingagents.research.features import build_feature_table
from tradingagents.research.feeds import FeedHub
from tradingagents.research.livepage import render_live_page
from tradingagents.research.paper import paper_history, paper_window_start, run_paper_session
from tradingagents.research.risk import load_policy
from tradingagents.research.store import ResearchStore
from tradingagents.research.versions import StrategyRegistry


class DashboardApp:
    """Request handling with no sockets in it, so it can be tested directly."""

    def __init__(self, store: ResearchStore, *, csv_dir: str | None = None,
                 ticker: str | None = None, timeframe: str = "1d",
                 context: str | None = None, confirm: str | None = None,
                 asset_type: str = "stock", initial_equity: float = 100_000.0,
                 costs: CostModel | None = None, start: str | None = None,
                 end: str | None = None, frames: dict | None = None,
                 feeds: FeedHub | None = None):
        self.store = store
        self.registry = StrategyRegistry(store)
        self.csv_dir = csv_dir
        self.ticker = ticker
        self.timeframe = timeframe
        self.context = context
        self.confirm = confirm
        self.asset_type = asset_type
        self.initial_equity = initial_equity
        self.costs = costs or CostModel()
        self.start = start
        self.end = end
        self.frames = frames
        # A hub with no feeds still renders a complete (and honest) live page.
        self.feeds = feeds if feeds is not None else FeedHub()
        self.token = secrets.token_urlsafe(18)
        self._flash: tuple[str, str] | None = None

    # ---------- data ----------

    def data_window(self, version: Any | None) -> tuple[str, str]:
        """The window to load: the version's own sample, widened by anything declared.

        A declared window can only *extend* the version's sample, never trim it: paper
        trading needs both the bars the version was chosen on and the bars that arrived
        afterwards, and a shorter window would silently drop one of them.
        """
        dataset = (getattr(version, "dataset", None) or {}) if version else {}
        recorded_start = str(dataset.get("first", ""))[:10]
        recorded_end = str(dataset.get("last", ""))[:10]
        begins = min(filter(None, (self.start, recorded_start)), default="")
        ends = max(filter(None, (self.end, recorded_end)), default="")
        if not begins or not ends:
            raise ValueError(
                "this dashboard has no data window: pass --start and --end, or use a "
                "version whose research session recorded the sample it was tested on")
        return begins, ends

    def table_for(self, version: Any):
        begins, ends = self.data_window(version)
        if self.frames is not None:
            source = FrameDataSource(self.frames)
        elif self.csv_dir:
            source = CsvDataSource(self.csv_dir)
        else:
            source = YahooDataSource()
        timeframes = [self.timeframe] + [tf for tf in (self.context, self.confirm) if tf]
        data = load_market_data(self.ticker or "SYNTH", timeframes, begins, ends,
                                asset_type=self.asset_type, source=source)
        if data.blocking_issues():
            raise ValueError("data quality gate failed: " + "; ".join(data.blocking_issues()))
        return build_feature_table(data, self.timeframe, self.context, self.confirm)

    def candidate_version(self) -> Any | None:
        return self.registry.champion_version() or (
            self.registry.versions()[-1] if self.registry.versions() else None)

    # ---------- actions ----------

    def suggested_start(self) -> str:
        """A start date to offer when a version recorded no split of its own.

        Only computed for local sources: asking a vendor on every page render would
        make the dashboard slow and would fail offline. A suggestion is never used
        implicitly — the form submits it, so a person still declares the start and
        the session record says whether the start came from the version or from them.
        """
        if self.start:
            return self.start
        if self.frames is None and not self.csv_dir:
            return ""
        try:
            begins, ends = self.data_window(None)
            source = FrameDataSource(self.frames) if self.frames is not None else CsvDataSource(
                self.csv_dir)
            frame = source.load(self.ticker or "SYNTH", _timeframe(self.timeframe), begins, ends)
        except Exception:
            return ""
        if len(frame) < 50:
            return ""
        return str(frame.index[int(len(frame) * 0.8)].date())

    def run_paper(self, version: str | None = None, *, min_trades: int = 5,
                  min_bars: int = 30, start: str | None = None) -> dict[str, Any]:
        """Trade the version forward and return the record that was written."""
        record = self.registry.get(version) if version else self.candidate_version()
        if record is None:
            raise ValueError("no version to paper trade")
        if start is None and paper_window_start(record) is None:
            raise ValueError(
                f"{record.version} has no recorded out-of-sample window, so paper trading "
                "needs an explicit start date. Enter one in the form (or pass --start): the "
                "session then records that the start was declared rather than derived.")
        table = self.table_for(record)
        policy = load_policy(self.store)
        config = EngineConfig(initial_equity=self.initial_equity, costs=self.costs,
                              risk_policy=policy)
        session = run_paper_session(self.registry, self.store, record.version, table, config,
                                    start=start, min_trades=min_trades, min_bars=min_bars,
                                    by="dashboard")
        return session.as_dict()

    # ---------- rendering ----------

    def snapshot(self) -> dict[str, Any]:
        return build_snapshot(self.store, ticker=self.ticker)

    def live_state(self) -> dict[str, Any]:
        """The display snapshot for ``/api/feeds``; read-only, never written anywhere."""
        return self.feeds.state()

    def live_page(self) -> str:
        return render_live_page(self.live_state(), back_href="/")

    def page(self, flash: str = "", tone: str = "info") -> str:
        snapshot = self.snapshot()
        versions = self.registry.versions()
        champion = self.registry.champion_version()
        sessions = paper_history(self.store, ticker=self.ticker, limit=10)
        try:
            window = " · ".join(self.data_window(champion)) if champion else "not set"
        except ValueError:
            window = "not set (pass --start and --end)"
        suggestion = self.suggested_start() if champion and not paper_window_start(champion) else ""
        return _PAGE.format(
            section=render_section_html(snapshot),
            flash=_flash_block(flash, tone),
            token=_attr(self.token),
            version_options=_version_options(versions, champion),
            start_value=_attr(suggestion),
            sessions=_session_table(sessions),
            ticker=_text(self.ticker or "—"),
            timeframe=_text(self.timeframe),
            window=_text(window),
            live="enabled" if snapshot.get("live_enabled") else "disabled",
            generated=_text(str(snapshot.get("generated_at", ""))),
        )


# --------------------------------------------------------------------------------------
# HTML helpers (everything that came from the store is escaped)
# --------------------------------------------------------------------------------------


def _timeframe(value: str) -> Timeframe:
    return Timeframe.parse(value)


def _text(value: Any) -> str:
    from html import escape

    return escape(str(value))


def _attr(value: Any) -> str:
    from html import escape

    return escape(str(value), quote=True)


def _flash_block(message: str, tone: str) -> str:
    if not message:
        return ""
    return f'<p class="flash {tone}">{_text(message)}</p>'


def _version_options(versions: list[Any], champion: Any) -> str:
    options = []
    for version in reversed(versions):
        label = f"{version.version} · {version.spec.name} · {version.status}"
        selected = " selected" if champion and version.version == champion.version else ""
        options.append(f'<option value="{_attr(version.version)}"{selected}>{_text(label)}'
                       "</option>")
    return "".join(options) or '<option value="">no versions yet</option>'


def _session_table(sessions: list[dict[str, Any]]) -> str:
    if not sessions:
        return ('<p class="note">No paper sessions recorded yet. A session trades the '
                "version forward over bars after its out-of-sample window; it places no "
                "orders.</p>")
    rows = []
    for session in reversed(sessions):
        expectancy = session.get("expectancy_r")
        rows.append(
            "<tr>"
            f"<td>{_text(session.get('paper_id', ''))}</td>"
            f"<td>{_text(session.get('version', ''))}</td>"
            f"<td>{_text(session.get('started', ''))} → {_text(session.get('ended', ''))}</td>"
            f"<td class=\"num\">{_text(session.get('bars', 0))}</td>"
            f"<td class=\"num\">{_text(session.get('trades', 0))}</td>"
            f"<td class=\"num\">{_num(expectancy, 'R')}</td>"
            f"<td class=\"num\">{_num(session.get('net_pnl'))}</td>"
            f"<td class=\"num\">{_pct(session.get('max_drawdown_pct'))}</td>"
            f"<td class=\"status {_attr(session.get('status', ''))}\">"
            f"{_text(session.get('status', ''))}</td>"
            "</tr>")
    return ("<table class=\"paper\"><thead><tr><th>session</th><th>version</th>"
            "<th>window</th><th>bars</th><th>trades</th><th>expectancy</th><th>net</th>"
            "<th>max DD</th><th>verdict</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def _num(value: Any, suffix: str = "") -> str:
    if isinstance(value, (int, float)):
        return f"{value:+,.3f}{suffix}" if suffix else f"{value:+,.2f}"
    return "—"


def _pct(value: Any) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "—"


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Intelligence · paper trading</title>
<style>
:root {{
  --ink: #12161c; --paper: #fbfaf7; --muted: #6b7280; --line: #e3e0d8;
  --accent: #1f5f8b; --ok: #1c6b46; --warn: #8a5a00; --bad: #8c2f2f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
}}
.page {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
header.masthead {{ border-bottom: 2px solid var(--ink); padding-bottom: 12px; margin-bottom: 22px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }}
h2 {{ font-size: 18px; margin: 30px 0 10px; }}
h3 {{ font-size: 15px; margin: 18px 0 6px; }}
p, dd, dt {{ margin: 0 0 6px; }}
.date, .note, .meta {{ color: var(--muted); font-size: 13px; }}
dl.run {{ display: grid; grid-template-columns: 200px 1fr; gap: 4px 18px; margin: 12px 0 4px; }}
dl.run div {{ display: contents; }}
dl.run dt {{ color: var(--muted); }}
dl.run dd {{ margin: 0; }}
section {{ margin-bottom: 8px; }}
.flash {{ padding: 10px 14px; border-left: 4px solid var(--accent); background: #eef4f9; margin: 0 0 18px; }}
.flash.ok {{ border-color: var(--ok); background: #eaf5ee; }}
.flash.warn {{ border-color: var(--warn); background: #fbf2df; }}
.flash.bad {{ border-color: var(--bad); background: #f9ecec; }}
form.paper {{ border: 1px solid var(--line); border-radius: 6px; padding: 16px 18px; background: #fff; }}
form.paper label {{ display: block; font-size: 13px; color: var(--muted); margin: 10px 0 4px; }}
form.paper select, form.paper input {{
  width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 4px;
  font: inherit; background: #fff; color: inherit;
}}
.row {{ display: flex; gap: 14px; flex-wrap: wrap; }}
.row > div {{ flex: 1 1 160px; }}
button {{
  margin-top: 16px; padding: 10px 18px; font: inherit; font-weight: 600; cursor: pointer;
  color: #fff; background: var(--accent); border: 0; border-radius: 4px;
}}
button:hover {{ background: #17496b; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }}
th {{ text-align: left; font-weight: 600; color: var(--muted); border-bottom: 1px solid var(--line); padding: 6px 8px; }}
td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.status.passed {{ color: var(--ok); font-weight: 600; }}
td.status.failed {{ color: var(--bad); font-weight: 600; }}
td.status.pending {{ color: var(--warn); }}
footer.credit {{ margin-top: 34px; border-top: 1px solid var(--line); padding-top: 12px; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="page">
<header class="masthead">
  <h1>Strategy Intelligence</h1>
  <p class="date">{ticker} · {timeframe} · sample {window} · live trading {live}</p>
  <p class="meta">Evidence as of {generated} · <a href="/feeds">3D live market view →</a></p>
</header>

{flash}

<section class="paper">
<h2 style="margin-top:0">Take a paper trade</h2>
<p class="note">Trades the selected version forward over bars <em>after</em> its
out-of-sample window and records the result. No broker, no orders, no capital. A passing
session only makes the version eligible for the approval conversation, which happens in
the CLI.</p>
<form class="paper" method="post" action="/paper">
  <input type="hidden" name="token" value="{token}">
  <label for="version">version</label>
  <select id="version" name="version">{version_options}</select>
  <div class="row">
    <div>
      <label for="min_trades">forward trades needed to pass</label>
      <input id="min_trades" name="min_trades" type="number" min="0" step="1" value="5">
    </div>
    <div>
      <label for="min_bars">forward bars needed to start</label>
      <input id="min_bars" name="min_bars" type="number" min="1" step="1" value="30">
    </div>
    <div>
      <label for="start">start (optional)</label>
      <input id="start" name="start" type="date" value="{start_value}">
      <p class="note" style="margin-top:4px">Defaults to the end of the version's
      out-of-sample window. A start you declare is recorded as declared: if it overlaps
      the sample the version was chosen on, the session is not a forward test.</p>
    </div>
  </div>
  <button type="submit">Run paper session</button>
</form>

<h2>Recorded sessions</h2>
{sessions}
</section>

{section}

<footer class="credit">Read-only apart from the paper-trade form. Approve, promote,
rollback, risk-policy changes and live trading are CLI-only by design: they need a
deliberate human decision, not a button.</footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------------------


def make_handler(app: DashboardApp) -> type[BaseHTTPRequestHandler]:
    """A request handler bound to one app instance."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "StrategyIntelligence"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            print(f"[dashboard] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8"
                  ) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/healthz":
                self._send(HTTPStatus.OK, json.dumps({"ok": True}), "application/json")
                return
            if parsed.path == "/api/snapshot":
                self._send(HTTPStatus.OK, json.dumps(app.snapshot(), default=str),
                           "application/json")
                return
            if parsed.path == "/api/feeds":
                self._send(HTTPStatus.OK, json.dumps(app.live_state(), default=str),
                           "application/json")
                return
            if parsed.path == "/feeds":
                try:
                    self._send(HTTPStatus.OK, app.live_page())
                except Exception as exc:  # a broken feed hub must not take the app down
                    self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _error_page(str(exc)))
                return
            if parsed.path in ("/", "/index.html"):
                flash = query.get("msg", [""])[0]
                tone = query.get("tone", ["info"])[0]
                if tone not in ("info", "ok", "warn", "bad"):
                    tone = "info"
                try:
                    self._send(HTTPStatus.OK, app.page(flash=flash, tone=tone))
                except Exception as exc:  # a half-configured dashboard still explains itself
                    self._send(HTTPStatus.INTERNAL_SERVER_ERROR, _error_page(str(exc)))
                return
            self._send(HTTPStatus.NOT_FOUND, "<p>not found</p>")

        def do_POST(self) -> None:  # noqa: N802 - stdlib naming
            parsed = urlparse(self.path)
            if parsed.path != "/paper":
                self._send(HTTPStatus.NOT_FOUND, "<p>not found</p>")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 8_192:
                self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "<p>form too large</p>")
                return
            fields = parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
            if fields.get("token", [""])[0] != app.token:
                self._send(HTTPStatus.FORBIDDEN,
                           "<p>This form is stale or came from another page. Reload the "
                           "dashboard and try again.</p>")
                return
            version = (fields.get("version", [""])[0] or "").strip() or None
            start = (fields.get("start", [""])[0] or "").strip() or None
            min_trades = _int(fields.get("min_trades", ["5"])[0], 5, minimum=0)
            min_bars = _int(fields.get("min_bars", ["30"])[0], 30, minimum=1)
            try:
                record = app.run_paper(version, min_trades=min_trades, min_bars=min_bars,
                                       start=start)
            except Exception as exc:  # a refused session is a message, not a crash
                self._redirect("/?tone=bad&msg=" + _quote(f"could not run a paper session: {exc}"))
                return
            tone = {"passed": "ok", "failed": "warn", "pending": "warn"}.get(
                record["status"], "info")
            message = (
                f"{record['version']} paper {record['status']}: {record['trades']} trades, "
                f"expectancy {record['expectancy_r']:+.3f}R, net {record['net_pnl']:+,.2f}, "
                f"{record['started']} to {record['ended']}")
            self._redirect(f"/?tone={tone}&msg=" + _quote(message))

    return Handler


def yahoo_price_fetcher(symbol: str) -> float | None:
    """The last close from the data layer, for a ``poll:`` feed.

    Delayed by whatever the vendor delays it by, which is why every ``poll`` feed is
    labelled *delayed* and never called live. Returns ``None`` rather than a guess when
    the vendor gives nothing useful.
    """
    import math
    from datetime import UTC, datetime, timedelta

    end = datetime.now(UTC).date()
    start = end - timedelta(days=7)
    frame = YahooDataSource().load(symbol, Timeframe.M5, start.isoformat(), end.isoformat())
    if frame is None or getattr(frame, "empty", True):
        return None
    if "close" not in list(getattr(frame, "columns", [])):
        return None
    closes = frame["close"].dropna()
    if closes.empty:
        return None
    close = float(closes.iloc[-1])
    return close if math.isfinite(close) and close > 0 else None


def _error_page(message: str) -> str:
    return ("<!doctype html><meta charset=utf-8><title>Strategy Intelligence</title>"
            "<body style=\"font:15px/1.55 system-ui;padding:32px;max-width:720px\">"
            "<h1 style=\"font-size:20px\">The dashboard could not render</h1>"
            f"<p><code>{_text(message)}</code></p>"
            "<p>Fix the cause and reload. Nothing was written by this request.</p></body>")


def _int(raw: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def serve(app: DashboardApp, host: str = "0.0.0.0", port: int = 32909
          ) -> ThreadingHTTPServer:
    """Bind and return the server without starting it (call ``serve_forever``)."""
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True
    return server
