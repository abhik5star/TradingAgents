"""The 3D page: readable without JavaScript, honest about its data, and no CDN.

There is no headless browser in this environment, so the page is checked the way a
person would check a page they cannot click: the HTML must carry the numbers, the
inline scripts must parse, and the geometry must produce sane vertices when the same
files are run under Node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tradingagents.research import livepage
from tradingagents.research.feeds import BarSeries, FeedHub, FeedSpec, ReplayFeed, Tick

ASSETS = Path(__file__).resolve().parents[1] / "tradingagents" / "research" / "assets"
NODE = shutil.which("node")


def _candles(count: int = 30) -> list[dict[str, object]]:
    bars = []
    for index in range(count):
        base = 100 + index
        bars.append({"time": f"2026-01-01T00:{index:02d}:00Z", "open": base,
                     "high": base + 2, "low": base - 1, "close": base + 1.5,
                     "volume": 10.0, "forming": index == count - 1})
    return bars


def _state(**overrides) -> dict[str, object]:
    feed = {
        "symbol": "SYNTH", "provider": "replay", "quality": "simulated",
        "status": "simulated", "connected": True, "error": "",
        "spec": {"symbol": "SYNTH", "provider": "replay", "url": "local bars",
                 "cost": "free, offline", "label": "SYNTH"},
        "series": {"symbol": "SYNTH", "price": 130.5, "change_pct": 12.34,
                   "trades": 240, "volume": 2400.0,
                   "last_at": "2026-01-01T00:29:00Z", "candles": 30},
        "candles": _candles(),
        "note": "simulated replay of historical bars on a compressed clock",
    }
    feed.update(overrides.pop("feed", {}))
    state = {"generated_at": "2026-01-01T00:30:00Z", "started_at": "2026-01-01T00:00:00Z",
             "feeds": [feed],
             "indian_equity_note": "There is no keyless public WebSocket for NSE or BSE.",
             "disclaimer": "Live feeds are display-only."}
    state.update(overrides)
    return state


@pytest.mark.unit
def test_the_page_carries_the_numbers_before_any_script_runs():
    page = livepage.render_live_page(_state())
    assert "Markets in motion" in page
    assert "SYNTH" in page
    assert "130.50" in page
    assert "+12.34%" in page
    assert "simulated" in page
    assert "240" in page  # tick count
    assert "There is no keyless public WebSocket" in page
    assert "<noscript>" in page
    assert 'id="tape-body"' in page and 'id="stage-canvas"' in page


@pytest.mark.unit
def test_the_page_loads_nothing_from_the_network():
    page = livepage.render_live_page(_state())
    for forbidden in ("http://", "https://", "//cdn", "fonts.googleapis",
                     "unpkg", "jsdelivr", "<link rel=\"stylesheet\""):
        assert forbidden not in page
    # Exactly three inline scripts plus the boot payload: no external src.
    assert page.count("<script") == 4
    assert "src=" not in page


@pytest.mark.unit
def test_the_boot_payload_round_trips_and_cannot_close_its_script_tag():
    page = livepage.render_live_page(_state(feed={"symbol": "</script><b>x</b>"}))
    marker = 'id="feeds-boot">'
    start = page.index(marker) + len(marker)
    boot = page[start: page.index("</script>", start)]
    assert "</script>" not in boot and "<" not in boot
    payload = json.loads(boot)
    assert payload["initial"]["feeds"][0]["symbol"] == "</script><b>x</b>"
    assert payload["poll_url"] == "/api/feeds"
    assert payload["poll_seconds"] >= 1


@pytest.mark.unit
def test_markup_in_a_symbol_or_note_cannot_escape_the_page():
    page = livepage.render_live_page(_state(feed={
        "symbol": "<img src=x onerror=alert(1)>",
        "note": "<script>alert(2)</script>"}))
    assert "<img src=x" not in page
    assert "<script>alert(2)</script>" not in page
    assert "&lt;img src=x" in page
    assert "&lt;script&gt;alert(2)" in page


@pytest.mark.unit
def test_a_non_finite_price_is_rendered_as_missing_not_as_infinity():
    page = livepage.render_live_page(_state(feed={
        "series": {"symbol": "SYNTH", "price": float("inf"), "change_pct": float("nan"),
                   "trades": 1, "candles": 0}}))
    # The inline scripts legitimately contain the word Infinity; the rendered document
    # must not, and the payload must carry null rather than a bare Infinity token.
    document = page.split('<script type="application/json"', 1)[0]
    assert "Infinity" not in document and "NaN" not in document
    assert "—" in document
    payload = json.loads(page.split('id="feeds-boot">', 1)[1].split("</script>", 1)[0])
    assert payload["initial"]["feeds"][0]["series"]["price"] is None
    assert payload["initial"]["feeds"][0]["series"]["change_pct"] is None


@pytest.mark.unit
def test_a_dashboard_with_no_feeds_explains_itself():
    page = livepage.render_live_page(FeedHub().state())
    assert "No feeds configured" in page
    assert "Nothing is being streamed, and nothing is being faked." in page
    assert "0 feed(s)" in page


@pytest.mark.unit
def test_the_page_lists_what_is_free_and_what_indian_brokers_need():
    page = livepage.render_live_page(_state())
    for provider in ("binance", "coinbase", "kraken", "generic", "poll", "replay"):
        assert provider in page
    for broker in ("Zerodha Kite", "Upstox", "Angel One SmartStream"):
        assert broker in page
    assert "--feed-url" in page  # the documented way to attach a broker stream


@pytest.mark.unit
def test_a_sparkline_needs_two_prices_to_draw_a_line():
    empty = livepage.sparkline([])
    assert "<polyline" not in empty and "spark-empty" in empty
    one = livepage.sparkline(_candles()[:1])
    assert "<polyline" not in one
    drawn = livepage.sparkline(_candles())
    assert drawn.count("<polyline") == 1
    assert "aria-label" in drawn
    # Coordinates stay inside the viewBox even when every price is identical.
    flat = livepage.sparkline([{"close": 100}, {"close": 100}, {"close": 100}])
    numbers = [float(value) for value in flat.split('points="', 1)[1].split('"', 1)[0]
               .replace(",", " ").split()]
    assert all(0 <= value <= 132 for value in numbers)
    # The end marker sits inside the box, not clipped by its edge.
    assert 'cx="132"' not in flat
    assert 'cx="130.0"' in flat  # width - 2, the last point of a 3-point line


@pytest.mark.unit
def test_the_inline_scripts_stay_small_enough_to_ship_in_a_page():
    assert livepage.asset_bytes() < 60_000
    for name in livepage.ASSET_FILES:
        assert livepage.read_asset(name).strip()


@pytest.mark.unit
def test_an_unknown_asset_is_refused():
    with pytest.raises(FileNotFoundError):
        livepage.read_asset("../server.py")


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_every_page_script_parses():
    for path in sorted(ASSETS.glob("*.js")):
        result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True,
                                timeout=60)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_scene_geometry_is_sane_when_run_under_node(tmp_path):
    script = tmp_path / "check.js"
    script.write_text(
        """
const path = require('path');
const m = require(process.argv[2]);
const feeds = [
  {symbol: 'A', color: '#6cc6f5', candles: [
    {open: 100, high: 110, low: 95, close: 105},
    {open: 105, high: 112, low: 100, close: 101},
    {open: 101, high: 108, low: 99, close: 107}]},
  {symbol: 'B', color: '#f2b134', candles: [
    {open: 10, high: 12, low: 9, close: 11}]},
];
const geometry = m.buildTracks(feeds, {maxCandles: 90});
if (geometry.vertexCount !== 4 * m.TRIANGLES_PER_CANDLE * 3) throw new Error('vertex count');
if (geometry.tracks.length !== 2) throw new Error('track count');
if (geometry.tracks[0].min !== 95 || geometry.tracks[0].max !== 112) throw new Error('range');
if (geometry.tracks[1].z <= geometry.tracks[0].z) throw new Error('tracks overlap');
if (Math.abs(geometry.tracks[0].changePct - 7) > 1e-9) throw new Error('change pct');
// Every normal is a unit axis vector and every colour channel is in range.
for (let i = 0; i < geometry.normals.length; i += 3) {
  const length = Math.hypot(geometry.normals[i], geometry.normals[i+1], geometry.normals[i+2]);
  if (Math.abs(length - 1) > 1e-6) throw new Error('normal not unit');
}
for (let i = 0; i < geometry.colors.length; i += 1) {
  if (!(geometry.colors[i] >= 0 && geometry.colors[i] <= 1)) throw new Error('colour range');
}
// The scene must project inside the viewport from the default camera.
const projection = m.perspective(42, 16 / 9, 0.1, 120);
const centre = [0, 0.62, 0];
const eye = m.orbitEye(centre, 0.55, 0.45, 9);
const vp = m.multiply(projection, m.lookAt(eye, centre, [0, 1, 0]));
for (let i = 0; i < geometry.positions.length; i += 3) {
  const point = m.project(vp, [geometry.positions[i], geometry.positions[i+1], geometry.positions[i+2]]);
  if (point[3] <= 0) throw new Error('point behind the camera');
  if (Math.abs(point[0]) > 1.02 || Math.abs(point[1]) > 1.02) throw new Error('scene outside the viewport');
}
// An empty track is reported, not crashed on.
const empty = m.buildTracks([{symbol: 'C', candles: []}], {});
if (empty.tracks[0].count !== 0 || empty.vertexCount !== 0) throw new Error('empty track');
const grid = m.buildGrid({lines: 4, step: 1});
if (grid.positions.length === 0 || grid.positions.length % 3 !== 0) throw new Error('grid');
console.log('scene-ok ' + geometry.vertexCount);
""",
        encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script), str(ASSETS / "scene-math.js")],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "scene-ok 432" in result.stdout


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_renderer_refuses_a_canvas_without_webgl(tmp_path):
    """No context means no scene: the page falls back to the table instead of a blank box."""
    script = tmp_path / "gl.js"
    script.write_text(
        """
global.window = {};
global.window.SceneMath = require(process.argv[2]);
global.window.devicePixelRatio = 1;
const deadCanvas = { getContext: () => null, addEventListener: () => {},
                     getBoundingClientRect: () => ({width: 1, height: 1}), width: 1, height: 1 };
require(process.argv[3]);
if (window.startMarketScene(deadCanvas) !== null) throw new Error('expected a refusal');
console.log('gl-fallback-ok');
""",
        encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script), str(ASSETS / "scene-math.js"), str(ASSETS / "scene.js")],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "gl-fallback-ok" in result.stdout


@pytest.mark.unit
@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_a_render_issues_one_draw_per_layer_with_the_whole_geometry(tmp_path):
    """A stub WebGL context drives a real frame: no browser, but the real code path.

    The interesting assertion is not "it did not throw" — it is that the triangle draw
    covers exactly the vertices the geometry built, that the grid draws lines, and that
    the matrix handed to the shader is a finite 16-float view-projection.
    """
    script = tmp_path / "render.js"
    script.write_text(
        """
global.window = {};
global.window.SceneMath = require(process.argv[2]);
global.window.devicePixelRatio = 2;
require(process.argv[3]);

const calls = { draw: [], uniforms: [] };
function fakeGl() {
  const constant = 1;
  return {
    VERTEX_SHADER: constant, FRAGMENT_SHADER: constant, COMPILE_STATUS: constant,
    LINK_STATUS: constant, ARRAY_BUFFER: constant, STATIC_DRAW: constant,
    FLOAT: constant, TRIANGLES: 4, LINES: 1, DEPTH_TEST: constant,
    BLEND: constant, SRC_ALPHA: constant, ONE_MINUS_SRC_ALPHA: constant,
    COLOR_BUFFER_BIT: constant, DEPTH_BUFFER_BIT: constant,
    createShader: () => ({}), shaderSource: () => {}, compileShader: () => {},
    getShaderParameter: () => true, getShaderInfoLog: () => '',
    createProgram: () => ({}), attachShader: () => {}, linkProgram: () => {},
    getProgramParameter: () => true, getProgramInfoLog: () => '',
    getAttribLocation: (_p, name) => name.length,
    getUniformLocation: (_p, name) => name,
    createBuffer: () => ({}), bindBuffer: () => {}, bufferData: () => {},
    enable: () => {}, disable: () => {}, enableVertexAttribArray: () => {},
    disableVertexAttribArray: () => {}, vertexAttribPointer: () => {},
    vertexAttrib3f: () => {}, useProgram: () => {},
    uniformMatrix4fv: (location, transpose, value) => calls.uniforms.push({ location, value }),
    uniform3fv: () => {}, viewport: () => {}, clear: () => {}, clearColor: () => {},
    blendFunc: () => {}, depthFunc: () => {},
    drawArrays: (mode, first, count) => calls.draw.push({ mode, first, count }),
  };
}

const canvas = {
  getContext: () => fakeGl(), addEventListener: () => {},
  getBoundingClientRect: () => ({ width: 800, height: 450 }), width: 0, height: 0,
  setPointerCapture: () => {},
};
const scene = window.startMarketScene(canvas);
if (!scene) throw new Error('no scene');
scene.setData([
  { symbol: 'A', color: '#6cc6f5', candles: [
    {open: 1, high: 2, low: 0.5, close: 1.5}, {open: 1.5, high: 3, low: 1, close: 2}]},
]);
scene.resize();  // boot.js calls this every frame, so the canvas follows its CSS box
if (canvas.width !== 1600 || canvas.height !== 900) throw new Error('dpr sizing');
scene.render(1000);
if (calls.draw.length !== 2) throw new Error('expected a grid draw and a candle draw');
const [grid, candles] = calls.draw;
if (grid.mode !== 1 || grid.count % 2 !== 0) throw new Error('grid must be line pairs');
if (candles.mode !== 4 || candles.count !== scene.geometry.vertexCount) {
  throw new Error('triangle draw must cover the whole geometry');
}
if (candles.count !== 2 * 36 * 3) throw new Error('two candles, thirty-six triangles each');
if (calls.uniforms.length < 2) throw new Error('view-projection was never uploaded');
for (const call of calls.uniforms) {
  const values = Array.from(call.value);
  if (values.length !== 16) throw new Error('matrix must be 4x4');
  if (!values.every(Number.isFinite)) throw new Error('non-finite matrix');
  if (values[15] === 0) throw new Error('perspective matrix looks wrong');
}
// A second frame keeps rendering; the camera drifts, the geometry does not.
scene.render(1016);
if (calls.draw.length !== 4) throw new Error('second frame did not draw');
if (scene.frames !== 2) throw new Error('frame counter');
// The newest candle projects near the middle of the canvas: that is where the price
// tag floats. A point behind the camera projects to nothing at all.
const track = scene.tracks[0];
const tag = scene.projectPoint([track.lastX, (track.last - track.min) * track.scaleY, track.z]);
if (!tag || !tag.visible) throw new Error('price tag would be off-screen');
if (tag.x < 200 || tag.x > 1400 || tag.y < 100 || tag.y > 800) throw new Error('tag position');
if (scene.projectPoint([0, 0, 500]) !== null) throw new Error('behind-camera point must project to null');
console.log('render-ok ' + candles.count);
""",
        encoding="utf-8")
    result = subprocess.run(
        [NODE, str(script), str(ASSETS / "scene-math.js"), str(ASSETS / "scene.js")],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "render-ok 216" in result.stdout


@pytest.mark.unit
def test_palette_colours_agree_between_server_and_scene():
    source = livepage.read_asset("scene-math.js")
    for colour in livepage.PALETTE:
        assert colour in source


@pytest.mark.unit
def test_a_replay_feed_renders_the_same_shape_the_api_returns():
    bars = [{"open": 10 + i, "high": 11 + i, "low": 9 + i, "close": 10.5 + i,
             "volume": 1.0} for i in range(8)]
    series = BarSeries("SYNTH")
    started = datetime(2026, 1, 1, tzinfo=UTC)
    ReplayFeed(FeedSpec(symbol="SYNTH", provider="replay"), series, bars, per_second=400,
               loop=False, start_at=started).run()
    empty = ReplayFeed(FeedSpec(symbol="SYNTH2", provider="replay"), BarSeries("SYNTH2"), [])
    empty.run()  # nothing to replay: a status, not a silence
    hub = FeedHub()
    hub.add(ReplayFeed(FeedSpec(symbol="SYNTH", provider="replay"), series, []))
    hub.add(empty)
    page = livepage.render_live_page(hub.state())
    assert "simulated" in page and "SYNTH" in page
    assert "no bars to replay" in page


@pytest.mark.unit
def test_a_tick_reaches_the_series_and_the_page():
    series = BarSeries("BTC-USD")
    series.add(Tick(symbol="BTC-USD", price=64000.0, quantity=0.5,
                    at=datetime(2026, 1, 1, tzinfo=UTC), source="test"))
    state = _state()
    state["feeds"][0]["series"] = series.summary()
    state["feeds"][0]["candles"] = series.candles()
    page = livepage.render_live_page(state)
    assert "64,000.00" in page


def _broken(symbol: str) -> float | None:
    raise OSError("vendor unreachable")


@pytest.mark.unit
def test_a_vendor_outage_reaches_the_feed_as_a_status(monkeypatch):
    """The poll path is delayed, and a vendor outage becomes a labelled status."""
    from tradingagents.research import server as dashboard_server

    class Broken:
        def __init__(self, cache_dir=None):
            pass

        def load(self, ticker, timeframe, start, end):
            raise OSError("vendor unreachable")

    monkeypatch.setattr(dashboard_server, "YahooDataSource", Broken)
    # The fetcher lets the vendor error out; PollFeed turns it into a status, which
    # test_a_poll_feed_is_labelled_delayed_and_reports_a_vendor_failure covers.
    with pytest.raises(OSError):
        dashboard_server.yahoo_price_fetcher("RELIANCE.NS")


@pytest.mark.unit
def test_the_poll_fetcher_reads_the_last_close(monkeypatch):
    import pandas as pd

    from tradingagents.research import server as dashboard_server

    seen: dict[str, str] = {}

    class Frame:
        empty = False
        columns = ["close"]

        def __getitem__(self, name):
            return pd.Series([1.0, 2.0, 2875.4], name=name)

    class Source:
        def __init__(self, cache_dir=None):
            pass

        def load(self, ticker, timeframe, start, end):
            seen["window"] = f"{start}..{end}"
            return Frame()

    monkeypatch.setattr(dashboard_server, "YahooDataSource", Source)
    assert dashboard_server.yahoo_price_fetcher("RELIANCE.NS") == 2875.4
    assert seen["window"].count("..") == 1
