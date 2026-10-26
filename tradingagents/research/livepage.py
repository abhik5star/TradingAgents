"""The 3D market view: free feeds drawn as a scene, never as evidence.

The page exists to answer one question honestly — *what is the market doing right
now* — without letting a live tape near the research loop. Three rules shape it:

* **Server first.** The same numbers are rendered into HTML by :func:`render_live_page`
  before any JavaScript runs, so a browser with no WebGL (or no JS at all) still sees
  every symbol, price and quality label.
* **No third-party code.** Three.js, d3 and friends would all be external fetches, and
  this page is served by a local process with no outbound network. The scene is
  therefore hand-written WebGL in :mod:`tradingagents.research.assets`, which also
  means the geometry can be unit-tested under Node.
* **Labels travel with the data.** Every feed carries ``quality`` — live, delayed or
  simulated — plus its endpoint, and the page repeats it in the legend, the table and
  the API payload. A simulated replay must never look like a market observation.

The page polls ``/api/feeds`` and rebuilds geometry only when the candle count or the
last candle's high/low/close changes, so a quiet market costs no allocations.
"""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

from tradingagents.research.contracts import utc_now
from tradingagents.research.feeds import INDIAN_BROKERS, PROVIDERS

ASSETS_DIR = Path(__file__).with_name("assets")
ASSET_FILES = ("scene-math.js", "scene.js", "boot.js")

#: Same order as the palette in ``assets/scene-math.js`` so the server-rendered legend
#: and the WebGL scene cannot disagree about which colour belongs to which symbol.
PALETTE = ("#6cc6f5", "#f2b134", "#67d18a", "#e86a92", "#b28df0", "#4fd1c5")

PAGE_TITLE = "Live market view · 3D"

#: How many candles a track draws; the boot payload is trimmed to the same count.
SCENE_CANDLES = 90


@lru_cache(maxsize=8)
def read_asset(name: str) -> str:
    """One of the page's JavaScript files, read once per process."""
    if name not in ASSET_FILES:
        raise FileNotFoundError(f"no such page asset: {name!r}")
    return (ASSETS_DIR / name).read_text(encoding="utf-8")


def asset_bytes() -> int:
    """Total bytes of inline script — a number the test suite keeps small on purpose."""
    return sum(len(read_asset(name).encode("utf-8")) for name in ASSET_FILES)


def palette_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    decimals = 2 if abs(number) >= 1000 else (4 if abs(number) >= 1 else 8)
    return f"{number:,.{decimals}f}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:+.2f}%"


def _tone(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "flat"
    if number > 0.0001:
        return "up"
    if number < -0.0001:
        return "down"
    return "flat"


_BACKTICK = re.compile(r"`([^`]+)`")


def _prose(text: Any) -> str:
    """Escape a sentence, then turn `backticked` fragments into inline code."""
    escaped = escape(str(text))
    return _BACKTICK.sub(lambda match: f"<code>{match.group(1)}</code>", escaped)


def _finite(value: Any) -> Any:
    """JSON-safe copy: ``json.dumps`` emits bare ``Infinity``/``NaN``, which is not JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _json_for_script(document: Any) -> str:
    """Serialise for a ``<script type="application/json">`` block, safely."""
    text = json.dumps(_finite(document), ensure_ascii=False, separators=(",", ":"),
                      default=str)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def sparkline(candles: list[dict[str, Any]], *, color: str = PALETTE[0], width: int = 132,
              height: int = 34, points: int = 40) -> str:
    """A server-rendered close-price line, so the table has shape without WebGL."""
    closes = [close for close in (_number(candle.get("close")) for candle in candles[-points:])
              if close is not None]
    if len(closes) < 2:
        return '<svg class="spark" viewBox="0 0 132 34" role="img" aria-label="no data">' \
               '<line x1="0" y1="17" x2="132" y2="17" class="spark-empty"/></svg>'
    low, high = min(closes), max(closes)
    span = (high - low) or 1.0
    # Leave room for the end marker, which would otherwise be half clipped.
    usable = width - 4
    step = usable / (len(closes) - 1)
    coordinates = []
    for index, close in enumerate(closes):
        x = 2 + index * step
        y = height - ((close - low) / span) * (height - 4) - 2
        coordinates.append(f"{x:.1f},{y:.1f}")
    last_x, last_y = coordinates[-1].split(",")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="close price over the last {len(closes)} candles">'
        f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{escape(color)}" '
        f'stroke-width="1.6" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="2.4" fill="{escape(color)}"/></svg>'
    )


# --------------------------------------------------------------------------------------
# Pieces of the page
# --------------------------------------------------------------------------------------


def _feeds(state: dict[str, Any]) -> list[dict[str, Any]]:
    feeds = state.get("feeds") or []
    return [dict(feed, color=feed.get("color") or palette_for(index))
            for index, feed in enumerate(feeds)]


def render_legend(state: dict[str, Any]) -> str:
    """Legend chips: the same markup ``boot.js`` rebuilds after each poll."""
    chips = []
    for feed in _feeds(state):
        series = feed.get("series") or {}
        chips.append(
            '<li class="legend-item {tone}">'
            '<span class="dot" style="background:{color}"></span>'
            '<span class="sym">{symbol}</span>'
            '<span class="px">{price}</span>'
            '<span class="chg">{change}</span>'
            '<span class="tag {quality}">{quality}</span>'
            "</li>".format(
                tone=_tone(series.get("change_pct")),
                color=escape(str(feed.get("color"))),
                symbol=escape(str(feed.get("symbol", "—"))),
                price=_price(series.get("price")),
                change=_percent(series.get("change_pct")),
                quality=escape(str(feed.get("quality", "unavailable"))),
            ))
    if not chips:
        return '<li class="legend-empty">No feeds configured. Start the server with ' \
               "<code>--feed binance:BTCUSDT</code>.</li>"
    return "".join(chips)


def render_cards(state: dict[str, Any]) -> str:
    cards = []
    for feed in _feeds(state):
        series = feed.get("series") or {}
        spec = feed.get("spec") or {}
        note = feed.get("note") or feed.get("error") or spec.get("cost") or ""
        cards.append(
            '<article class="card">'
            '<div class="card-head">'
            '<span class="card-sym" style="color:{color}">{symbol}</span>'
            '<span class="tag {quality}">{quality}</span>'
            "</div>"
            '<p class="card-price">{price}<span class="card-chg {tone}">{change}</span></p>'
            '<p class="card-meta">{provider} · {candles} candles · {trades} ticks</p>'
            '<p class="card-note">{note}</p>'
            "</article>".format(
                color=escape(str(feed.get("color"))),
                symbol=escape(str(feed.get("symbol", "—"))),
                quality=escape(str(feed.get("quality", "unavailable"))),
                price=_price(series.get("price")),
                tone=_tone(series.get("change_pct")),
                change=_percent(series.get("change_pct")),
                provider=escape(str(feed.get("provider", "—"))),
                candles=escape(str(series.get("candles", 0))),
                trades=escape(str(series.get("trades", 0))),
                note=_prose(note),
            ))
    if not cards:
        cards.append(
            '<article class="card card-empty"><div class="card-head">'
            '<span class="card-sym">No feeds</span></div>'
            "<p class=\"card-note\">This server was started without <code>--feed</code>. "
            "Nothing is being streamed, and nothing is being faked.</p></article>")
    return "".join(cards)


def render_tape(state: dict[str, Any]) -> str:
    rows = []
    for feed in _feeds(state):
        series = feed.get("series") or {}
        spec = feed.get("spec") or {}
        note = feed.get("note") or feed.get("error") or spec.get("url") or ""
        rows.append(
            "<tr>"
            "<td>{symbol}</td><td>{provider}</td>"
            '<td><span class="tag {quality}">{quality}</span></td>'
            "<td>{status}</td>"
            '<td class="num">{price}</td>'
            '<td class="num {tone}">{change}</td>'
            '<td class="num">{candles}</td><td class="num">{trades}</td>'
            "<td class=\"spark-cell\">{spark}</td>"
            '<td class="note">{note}</td>'
            "</tr>".format(
                symbol=escape(str(feed.get("symbol", "—"))),
                provider=escape(str(feed.get("provider", "—"))),
                quality=escape(str(feed.get("quality", "unavailable"))),
                status=escape(str(feed.get("status", "unavailable"))),
                price=_price(series.get("price")),
                tone=_tone(series.get("change_pct")),
                change=_percent(series.get("change_pct")),
                candles=escape(str(series.get("candles", 0))),
                trades=escape(str(series.get("trades", 0))),
                spark=sparkline(feed.get("candles") or [], color=str(feed.get("color"))),
                note=_prose(note),
            ))
    if not rows:
        rows.append('<tr><td colspan="10" class="empty">No feed configured for this '
                    "dashboard.</td></tr>")
    return "".join(rows)


def render_providers() -> str:
    rows = []
    for name, provider in PROVIDERS.items():
        rows.append(
            f"<tr><td><code>{escape(name)}</code></td><td>{escape(provider.kind)}</td>"
            f"<td>{escape(provider.quality)}</td><td>{escape(provider.cost)}</td>"
            f"<td>{escape(provider.notes)}</td></tr>")
    return "".join(rows)


def render_brokers() -> str:
    rows = []
    for entry in INDIAN_BROKERS.values():
        rows.append(
            "<tr><td>{broker}</td><td>{protocol}</td><td>{needs}</td><td>{free}</td></tr>"
            .format(
                broker=escape(entry.get("broker", "")),
                protocol=escape(entry.get("protocol", "")),
                needs=escape(entry.get("needs", "")),
                free=escape(entry.get("india_free", "")),
            ))
    return "".join(rows)


def _status_line(state: dict[str, Any]) -> str:
    feeds = _feeds(state)
    if not feeds:
        return "no feed configured"
    connected = sum(1 for feed in feeds if feed.get("connected"))
    kinds = sorted({str(feed.get("quality")) for feed in feeds})
    return f"{connected}/{len(feeds)} feeds up · {'/'.join(kinds)}"


# --------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------


def render_live_page(state: dict[str, Any], *, poll_url: str = "/api/feeds",
                     poll_seconds: float = 2.5, back_href: str | None = "/",
                     title: str = PAGE_TITLE) -> str:
    """Render the whole 3D page from one ``FeedHub.state()`` snapshot."""
    feeds = _feeds(state)
    # The boot payload only has to draw the first frame; the scene shows at most 90
    # candles per track, so sending 240 to the browser would be waste, not detail.
    boot_feeds = [dict(feed, candles=(feed.get("candles") or [])[-SCENE_CANDLES:])
                  for feed in feeds]
    boot = {"initial": dict(state, feeds=boot_feeds), "poll_url": poll_url,
            "poll_seconds": max(1.0, float(poll_seconds))}
    generated = str(state.get("generated_at") or utc_now())
    started = state.get("started_at")
    back = (f'<a class="back" href="{escape(back_href, quote=True)}">← dashboard</a>'
            if back_href else "")
    note = str(state.get("indian_equity_note") or "")
    disclaimer = str(state.get("disclaimer") or "")
    return _PAGE.format(
        title=escape(title),
        back=back,
        status_line=escape(_status_line(state)),
        generated=escape(generated),
        started=escape(f" · collecting since {started}" if started else ""),
        legend=render_legend(state),
        cards=render_cards(state),
        rows=render_tape(state),
        providers=render_providers(),
        brokers=render_brokers(),
        indian_note=_prose(note),
        disclaimer=_prose(disclaimer),
        boot=_json_for_script(boot),
        scene_math=read_asset("scene-math.js"),
        scene=read_asset("scene.js"),
        boot_js=read_asset("boot.js"),
        feed_count=len(feeds),
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --ink: #12161c; --paper: #fbfaf7; --muted: #6b7280; --line: #e3e0d8;
  --accent: #1f5f8b; --ok: #1c6b46; --warn: #8a5a00; --bad: #8c2f2f;
  --stage: #080c12; --stage-ink: #e8eef6; --stage-muted: #8fa3b8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
}}
.page {{ max-width: 1080px; margin: 0 auto; padding: 32px 20px 64px; }}
header.masthead {{ border-bottom: 2px solid var(--ink); padding-bottom: 12px; margin-bottom: 20px; }}
.eyebrow {{
  margin: 0 0 4px; font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); font-weight: 700;
}}
h1 {{ font-size: 27px; margin: 0 0 4px; letter-spacing: -0.015em; }}
h2 {{ font-size: 17px; margin: 30px 0 10px; }}
h3 {{ font-size: 14px; margin: 20px 0 6px; color: var(--muted); text-transform: uppercase;
      letter-spacing: .08em; }}
p, dd, dt {{ margin: 0 0 6px; }}
.date, .note, .meta {{ color: var(--muted); font-size: 13px; }}
.back {{ float: right; font-size: 13px; color: var(--accent); text-decoration: none; }}
.back:hover {{ text-decoration: underline; }}
.notice {{
  border-left: 4px solid var(--accent); background: #eef4f9; padding: 10px 14px;
  margin: 0 0 18px; font-size: 13px;
}}
.notice strong {{ font-weight: 700; }}
.stage {{
  position: relative; border-radius: 12px; overflow: hidden; background:
  radial-gradient(120% 90% at 20% 0%, #16273a 0%, #0a1018 45%, #05080c 100%);
  aspect-ratio: 16 / 9; min-height: 340px;
  box-shadow: 0 18px 40px -22px rgba(8, 12, 18, .8), 0 0 0 1px rgba(18, 26, 38, .9) inset;
  color: var(--stage-ink);
  touch-action: none;
}}
#stage-canvas {{ display: block; width: 100%; height: 100%; cursor: grab; }}
#stage-canvas:active {{ cursor: grabbing; }}
.hud {{ position: absolute; display: flex; gap: 8px; align-items: center; }}
.hud-top {{ top: 12px; left: 14px; right: 14px; justify-content: space-between;
            pointer-events: none; }}
.hud-meta {{ font-size: 11px; color: var(--stage-muted); font-variant-numeric: tabular-nums;
             letter-spacing: .04em; }}
.pill {{
  font-size: 11px; font-weight: 700; letter-spacing: .09em; padding: 5px 10px;
  border-radius: 999px; background: rgba(143, 163, 184, .16); color: var(--stage-muted);
  border: 1px solid rgba(143, 163, 184, .3);
}}
.pill.live {{ color: #7cf0b4; border-color: rgba(124, 240, 180, .45);
              background: rgba(28, 107, 70, .28); }}
.pill.sim {{ color: #f6cd6b; border-color: rgba(246, 205, 107, .45);
             background: rgba(138, 90, 0, .28); }}
.pill.warn {{ color: #ffcf8a; border-color: rgba(255, 207, 138, .4);
              background: rgba(138, 90, 0, .22); }}
.pill.down {{ color: #ff9d9d; border-color: rgba(255, 157, 157, .4);
              background: rgba(140, 47, 47, .25); }}
.hud-controls {{ top: 12px; right: 14px; flex-direction: row; }}
.hud-controls button {{
  pointer-events: auto; font: inherit; font-size: 11px; letter-spacing: .04em;
  padding: 5px 10px; border-radius: 999px; cursor: pointer;
  color: var(--stage-ink); background: rgba(232, 238, 246, .08);
  border: 1px solid rgba(232, 238, 246, .18); backdrop-filter: blur(6px);
}}
.hud-controls button:hover {{ background: rgba(232, 238, 246, .16); }}
.hud-controls button[aria-pressed="true"] {{ border-color: rgba(124, 240, 180, .5);
                                            color: #9ff5c8; }}
.legend {{
  list-style: none; margin: 0; padding: 0; position: absolute; left: 14px; bottom: 12px;
  display: flex; flex-wrap: wrap; gap: 6px; max-width: calc(100% - 28px);
}}
.legend-item {{
  display: flex; align-items: center; gap: 7px; font-size: 11px;
  padding: 4px 9px; border-radius: 999px; background: rgba(8, 12, 18, .62);
  border: 1px solid rgba(232, 238, 246, .14); color: var(--stage-ink);
  font-variant-numeric: tabular-nums; backdrop-filter: blur(6px);
}}
.legend-item .dot {{ width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 8px currentColor; }}
.legend-item .sym {{ font-weight: 700; letter-spacing: .03em; }}
.legend-item .px {{ color: var(--stage-muted); }}
.legend-item .chg.up {{ color: #7cf0b4; }}
.legend-item .chg.down {{ color: #ff9d9d; }}
.legend-item .chg.flat {{ color: var(--stage-muted); }}
.legend-empty {{ font-size: 12px; color: var(--stage-muted); }}
.legend-empty code {{ color: var(--stage-ink); }}
.stage-hint {{
  position: absolute; right: 14px; bottom: 12px; font-size: 10.5px; letter-spacing: .05em;
  color: rgba(143, 163, 184, .8); text-transform: uppercase;
}}
.stage-fallback {{
  position: absolute; inset: 0; display: grid; place-content: center; padding: 24px;
  text-align: center; font-size: 13px; color: var(--stage-muted);
  background: rgba(5, 8, 12, .86);
}}
.stage.flat #stage-canvas {{ visibility: hidden; }}
.stage-labels {{ position: absolute; inset: 0; pointer-events: none; overflow: hidden; }}
.price-tag {{
  position: absolute; top: 0; left: 0; display: none; white-space: nowrap;
  font-size: 10.5px; font-variant-numeric: tabular-nums; letter-spacing: .02em;
  padding: 2px 7px; border-radius: 999px; color: var(--stage-ink);
  background: rgba(8, 12, 18, .72); border: 1px solid rgba(232, 238, 246, .18);
  box-shadow: 0 6px 16px -10px rgba(0, 0, 0, .9);
}}
.price-tag b {{ font-weight: 700; letter-spacing: .04em; margin-right: 4px; }}
.price-tag.up b {{ color: #7cf0b4; }}
.price-tag.down b {{ color: #ff9d9d; }}
.price-tag.sim {{ border-color: rgba(246, 205, 107, .4); }}
.stage.flat .stage-labels {{ display: none; }}
.stage.flat::after {{
  content: "Flat view — the tape table below carries every number."; position: absolute;
  inset: 0; display: grid; place-content: center; font-size: 13px; color: var(--stage-muted);
}}
.kpis {{ display: grid; gap: 10px; margin: 16px 0 0;
         grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); }}
.card {{ border: 1px solid var(--line); border-radius: 8px; background: #fff;
         padding: 12px 14px; }}
.card-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }}
.card-sym {{ font-weight: 700; letter-spacing: .02em; font-size: 13px; }}
.card-price {{ font-size: 20px; font-variant-numeric: tabular-nums; margin: 4px 0 2px;
               letter-spacing: -0.01em; }}
.card-chg {{ font-size: 12px; margin-left: 8px; font-weight: 600; }}
.card-meta, .card-note {{ font-size: 11.5px; color: var(--muted); margin: 0; }}
.card-note {{ margin-top: 4px; }}
.card-empty {{ grid-column: 1 / -1; }}
.up {{ color: var(--ok); }} .down {{ color: var(--bad); }} .flat {{ color: var(--muted); }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }}
th {{ text-align: left; font-weight: 600; color: var(--muted); border-bottom: 1px solid var(--line);
      padding: 6px 8px; white-space: nowrap; }}
td {{ border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: middle; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.note {{ color: var(--muted); font-size: 11.5px; max-width: 300px; }}
td.empty {{ color: var(--muted); font-style: italic; }}
.tag {{
  display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; padding: 2px 7px; border-radius: 999px;
  border: 1px solid currentColor; color: var(--muted);
}}
.tag.live {{ color: var(--ok); }}
.tag.delayed {{ color: var(--warn); }}
.tag.simulated {{ color: #7a4a9c; }}
.tag.unavailable {{ color: var(--bad); }}
.spark {{ width: 132px; height: 34px; display: block; }}
.spark-empty {{ stroke: var(--line); stroke-width: 1.4; }}
.spark-cell {{ padding: 2px 8px; }}
code {{ font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
        background: #f1efe9; padding: 1px 5px; border-radius: 4px; }}
.reality {{ border-top: 1px solid var(--line); margin-top: 34px; padding-top: 6px; }}
footer.credit {{ margin-top: 30px; border-top: 1px solid var(--line); padding-top: 12px;
                 color: var(--muted); font-size: 12px; }}
@media (max-width: 640px) {{
  .stage {{ aspect-ratio: 4 / 3; min-height: 260px; }}
  .stage-hint {{ display: none; }}
  .hud-controls button {{ padding: 4px 8px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .stage {{ scroll-behavior: auto; }}
}}
</style>
</head>
<body>
<div class="page">
<header class="masthead">
  {back}
  <p class="eyebrow">Live market view · 3D</p>
  <h1>Markets in motion</h1>
  <p class="date">{status_line}</p>
  <p class="meta">{feed_count} feed(s) · built {generated}{started}</p>
</header>

<div class="notice">
  <strong>Display only.</strong> Nothing on this page reaches a backtest, a paper
  verdict or a research result: those run on loaded bars. Each feed states what it
  is — real-time, delayed or simulated — and an unreachable endpoint is reported as
  unavailable rather than papered over.
</div>

<section class="stage-wrap">
  <div class="stage" id="stage">
    <canvas id="stage-canvas" aria-label="3D candlestick scene of the configured feeds"></canvas>
    <div class="hud hud-top">
      <span class="pill" id="stage-status">connecting…</span>
      <span class="hud-meta"><span id="stage-fps">—</span> · <span id="stage-clock">—</span></span>
    </div>
    <div class="hud hud-controls">
      <button type="button" id="btn-rotate" aria-pressed="true">Auto-rotate</button>
      <button type="button" id="btn-reset">Reset view</button>
      <button type="button" id="btn-flat" aria-pressed="false">Flat table</button>
    </div>
    <ul class="legend" id="legend">{legend}</ul>
    <p class="stage-hint">drag to orbit · scroll to zoom · double-click to pause</p>
    <div class="stage-fallback" id="stage-fallback" hidden></div>
  </div>
  <noscript>
    <p class="note">JavaScript is off, so the 3D stage cannot draw. Every feed, price and
    label is in the table below — the page is complete without the scene.</p>
  </noscript>
</section>

<section class="kpis">{cards}</section>

<section class="tape">
  <h2>The tape</h2>
  <table>
    <thead>
      <tr>
        <th>symbol</th><th>provider</th><th>quality</th><th>status</th>
        <th>last</th><th>change</th><th>candles</th><th>ticks</th>
        <th>last 40 candles</th><th>source / note</th>
      </tr>
    </thead>
    <tbody id="tape-body">{rows}</tbody>
  </table>
</section>

<section class="reality">
  <h2>What is actually free</h2>
  <p class="note">Crypto has free, keyless, real-time public WebSockets. Indian equities
  do not — the exchange feeds are licensed through brokers.</p>
  <p class="note">{indian_note}</p>
  <table>
    <thead><tr><th>provider</th><th>kind</th><th>what you get</th><th>cost</th>
    <th>notes</th></tr></thead>
    <tbody>{providers}</tbody>
  </table>
  <h3>Indian equities: what each broker needs</h3>
  <table>
    <thead><tr><th>broker</th><th>protocol</th><th>credentials</th><th>free?</th></tr></thead>
    <tbody>{brokers}</tbody>
  </table>
  <p class="note">A broker stream can be added without touching this code:
  <code>--feed generic:RELIANCE --feed-url wss://… --feed-subscribe
  '{{"type":"subscribe"}}'</code>.
  The key comes from the environment, is never written to the store, and is never shown
  on this page.</p>
</section>

<footer class="credit">{disclaimer} Live trading stays disabled; this view cannot place an
order.</footer>
</div>

<script type="application/json" id="feeds-boot">{boot}</script>
<script>{scene_math}</script>
<script>{scene}</script>
<script>{boot_js}</script>
</body>
</html>
"""
