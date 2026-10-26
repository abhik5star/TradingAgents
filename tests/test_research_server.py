"""The web dashboard: read the evidence, run a paper session, and nothing else."""

from __future__ import annotations

import json
import socket
import urllib.parse

import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.features import build_feature_table
from tradingagents.research.paper import paper_history
from tradingagents.research.server import DashboardApp
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import baseline_strategy
from tradingagents.research.versions import StrategyRegistry


@pytest.fixture(scope="module")
def bars():
    return synthetic_bars(n=1000, seed=17, regime_length=70)


@pytest.fixture()
def app(tmp_path, bars):
    store = ResearchStore(root=tmp_path / "store")
    registry = StrategyRegistry(store)
    table = build_feature_table(market_data(bars), "1d")
    cut = table.bars.index[700]
    registry.register(
        baseline_strategy("1d").with_version("V01"),
        dataset={**table.summary(),
                 "split": {"test": f"{table.bars.index[0].date()}..{cut.date()}"}},
        out_of_sample={"metrics": {"expectancy_r": 0.2, "trades": 40}})
    registry.register(baseline_strategy("1d").with_version("V02"), parent_version="V01",
                      dataset={**table.summary(),
                               "split": {"test": f"{table.bars.index[0].date()}..{cut.date()}"}})
    return DashboardApp(store, ticker="SYNTH", timeframe="1d",
                        frames={"1d": bars})


@pytest.mark.unit
def test_the_page_renders_the_snapshot_and_the_paper_form(app):
    page = app.page()
    assert "Strategy Intelligence" in page
    assert 'action="/paper"' in page
    assert 'name="token"' in page
    assert "V01" in page and "V02" in page
    assert "live trading disabled" in page
    # Only the one action exists: no approve/promote/live route on the page.
    for forbidden in ("/approve", "/promote", "/live", "/rollback"):
        assert f'action="{forbidden}"' not in page
    assert "CLI-only by design" in page


@pytest.mark.unit
def test_a_paper_session_can_be_run_from_the_dashboard(app):
    record = app.run_paper("V01", min_trades=5, min_bars=30)
    assert record["version"] == "V01"
    assert record["status"] in ("passed", "failed", "pending")
    assert record["paper_id"].startswith("PPR-")
    stored = paper_history(app.store, "V01")
    assert stored and stored[-1]["paper_id"] == record["paper_id"]
    assert stored[-1]["by"] == "dashboard"
    # The page shows it afterwards.
    assert record["paper_id"] in app.page()


@pytest.mark.unit
def test_the_dashboard_defaults_to_the_champion(app):
    from tradingagents.research.contracts import ExperimentStatus

    app.registry.set_status("V02", ExperimentStatus.PAPER_TESTING)
    app.registry.promote("V02")
    record = app.run_paper(None)
    assert record["version"] == "V02"


@pytest.mark.unit
def test_a_refused_session_is_a_message_not_a_crash(app):
    with pytest.raises(Exception, match="V99"):
        app.run_paper("V99")
    # A version whose record holds no out-of-sample window cannot start honestly:
    # the dashboard asks for a declared start instead of quietly choosing one.
    app.registry.register(baseline_strategy("1d").with_version("V03"))
    with pytest.raises(ValueError, match="explicit start date"):
        app.run_paper("V03")

    # Declared explicitly — both the start and the data window — it runs, and the
    # record says the start was a declaration rather than a recorded split.
    declared = DashboardApp(app.store, ticker="SYNTH", timeframe="1d",
                            frames={"1d": app.frames["1d"]},
                            start=str(app.frames["1d"].index[0].date()),
                            end=str(app.frames["1d"].index[-1].date()))
    record = declared.run_paper("V03", start=str(app.frames["1d"].index[800].date()),
                                min_bars=10)
    assert record["status"] in ("passed", "failed", "pending")
    assert record["start_source"] == "declared"


@pytest.mark.unit
def test_the_json_snapshot_is_the_same_one_the_report_uses(app):
    snapshot = app.snapshot()
    assert snapshot["champion"]["version"] == "V01"
    payload = json.dumps(snapshot, default=str)
    assert '"live_enabled": false' in payload


@pytest.mark.unit
def test_markup_from_the_store_cannot_escape_into_the_page(app):
    app.store.append("hypothesis", {
        "hypothesis_id": "H-x", "ticker": "SYNTH", "title": "<script>alert(1)</script>",
        "decision": "rejected", "status": "Rejected", "reason": "<img src=x>",
        "evidence": ["</dd><script>x()</script>"], "created_at": "2026-01-01T00:00:00Z"})
    page = app.page()
    assert "<script>" not in page.lower()
    assert "&lt;script&gt;" in page


def _exchange(app, method: str, path: str, body: bytes = b"") -> tuple[int, dict, str]:
    """Drive one request through the handler over an in-memory socket pair.

    The test suite forbids real network connections, and a dashboard handler is
    exactly the kind of code that deserves an end-to-end request rather than a
    mocked one, so the socket is a pair in this process.
    """
    import io

    from tradingagents.research.server import make_handler

    client, server = socket.socketpair()
    handler_class = make_handler(app)
    captured: dict[str, object] = {}

    class Recorder(handler_class):
        def send_response(self, code, message=None):
            captured["status"] = code
            super().send_response(code, message)

        def send_header(self, keyword, value):
            captured.setdefault("headers", {})
            captured["headers"][keyword.lower()] = value
            super().send_header(keyword, value)

    raw = io.BytesIO()
    handler = Recorder.__new__(Recorder)
    handler.rfile = io.BytesIO(body)
    handler.wfile = raw
    handler.client_address = ("127.0.0.1", 12345)
    handler.server = type("S", (), {"server_name": "test", "server_port": 0})()
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.path = path
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.headers = {"Content-Length": str(len(body)), "Host": "test"}  # noqa: F841
    handler.close_connection = True
    handler.log_message = lambda *a, **k: None
    # base handler needs these to be real attributes rather than a plain dict
    import email.message

    message = email.message.Message()
    message["Content-Length"] = str(len(body))
    handler.headers = message
    if method == "GET":
        handler.do_GET()
    else:
        handler.do_POST()
    client.close()
    server.close()
    payload = raw.getvalue()
    head, _, text = payload.partition(b"\r\n\r\n")
    return int(captured.get("status", 0)), dict(captured.get("headers", {})), text.decode(
        "utf-8", "replace")


@pytest.mark.unit
def test_over_http_the_form_token_is_enforced_and_the_redirect_reports_the_result(app):
    status, _headers, text = _exchange(app, "GET", "/healthz")
    assert status == 200 and json.loads(text) == {"ok": True}

    status, _headers, page = _exchange(app, "GET", "/")
    assert status == 200 and "Strategy Intelligence" in page

    # No token: refused, and no session written.
    body = urllib.parse.urlencode({"version": "V01"}).encode()
    status, _headers, text = _exchange(app, "POST", "/paper", body)
    assert status == 403
    assert "reload the dashboard" in text.lower()
    assert paper_history(app.store, "V01") == []

    # With the token it runs, redirects and says what happened.
    payload = urllib.parse.urlencode({"version": "V01", "token": app.token,
                                  "min_trades": "5", "min_bars": "30"}).encode()
    status, headers, _text = _exchange(app, "POST", "/paper", payload)
    assert status == 303
    assert headers["location"].startswith("/?tone=")
    assert "paper" in urllib.parse.unquote(headers["location"])
    assert paper_history(app.store, "V01")

    status, _headers, text = _exchange(app, "GET", "/api/snapshot")
    assert status == 200 and json.loads(text)["champion"]["version"] == "V01"

    status, _headers, _text = _exchange(app, "GET", "/nope")
    assert status == 404


# --------------------------------------------------------------------------------------
# The 3D live view, served by the same process and writing to nothing
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_the_live_view_is_served_even_with_no_feeds_configured(app):
    status, headers, payload = _exchange(app, "GET", "/api/feeds")
    assert status == 200
    assert headers["content-type"].startswith("application/json")
    state = json.loads(payload)
    assert state["feeds"] == []
    assert "no keyless public WebSocket" in state["indian_equity_note"]
    assert "display-only" in state["disclaimer"]

    status, headers, page = _exchange(app, "GET", "/feeds")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert "Markets in motion" in page
    assert 'id="stage-canvas"' in page
    assert "No feeds configured" in page


@pytest.mark.unit
def test_the_live_view_never_writes_to_the_research_store(app):
    streams = ("hypothesis", "experiment", "paper", "risk_event", "autopsy")
    before = {name: len(app.store.read(name)) for name in streams}
    _exchange(app, "GET", "/feeds")
    _exchange(app, "GET", "/api/feeds")
    after = {name: len(app.store.read(name)) for name in before}
    assert before == after


@pytest.mark.unit
def test_a_replay_hub_reaches_the_api_and_is_labelled_simulated(app):
    from tradingagents.research.feeds import FeedHub, parse_feed_specs

    replay = [{"open": 10 + index, "high": 11 + index, "low": 9 + index,
               "close": 10.5 + index, "volume": 1.0} for index in range(12)]
    hub = FeedHub.from_specs(parse_feed_specs(["replay:SYNTH"]),
                             replay_bars={"SYNTH": replay}, per_second=400)
    live_app = DashboardApp(app.store, ticker="SYNTH", timeframe="1d", feeds=hub)
    hub.start()
    try:
        status, _headers, payload = _exchange(live_app, "GET", "/api/feeds")
        assert status == 200
        state = json.loads(payload)
        feed = state["feeds"][0]
        assert feed["quality"] == "simulated"
        assert feed["provider"] == "replay"
        assert "simulated" in feed["note"]
        assert feed["series"]["candles"] >= 1
        assert feed["candles"][0]["close"] >= 10
    finally:
        hub.stop()

    _status, _headers, page = _exchange(live_app, "GET", "/feeds")
    assert "simulated" in page
    # Building the live page did not touch the paper record either.
    assert paper_history(app.store, "V01") == []
