# Design file: Strategy Intelligence

A Figma-grade file for the research dashboard and the 3D live market view, built from the
running application rather than from taste. Open **`strategy-intelligence.html`** in a browser:
it needs no server, no fonts and no network.

> Arena reports the Figma connector as *unsupported*, so nothing here is pushed into a Figma
> account. The file reproduces what a Figma document would carry — frames, a layer tree, an
> inspector, tokens, components, redlines — and `tokens.tokens-studio.json` is the palette in
> the format the Tokens Studio plugin imports into Figma as real variables, if you want it there.

## What is in it

| Frame | Size | What it shows |
| --- | --- | --- |
| 00 Cover | 1440 | The argument of the product, contents, palette |
| 01 Dashboard | 1440 | Champion V01, the promotion pipeline, the `NO TRADE` verdict with its six failed gates, KPIs, the paper-session record, the one permitted action |
| 02 Dashboard · narrow | 390 | The same hierarchy in one column |
| 03 3D live market view | 1440 | The stage, price tags, feed legend, tape table with server-rendered sparklines, and the honest provider/broker tables |
| 04 Components | 1440 | Buttons, badges, form, table row, KPI tile, verdict states, feed states, empty state, and the market skin |
| 05 Handoff | 1440 | Type scale, spacing, radius, contrast, copy rules, redlines |

Board interactions: click a layer on the left to inspect it, click a frame title to jump,
`fit` to resize, **Present** to hide the tool and show the frames alone.

## Where the numbers come from

Nothing on the board is invented:

* the metrics are the preview store's real values (`/api/snapshot`: champion V01, 3 hypotheses
  all rejected, one paper session at +0.678R over 14 forward trades);
* the scene is `scene.svg`, generated from the application's own geometry and camera
  (`tradingagents/research/assets/scene-math.js`), not an illustration;
* the palette is `tokens.json`, which `tests/test_design_board.py` checks against the CSS the
  server and the live page actually ship.

## Rebuilding

```bash
node docs/design/make_scene_svg.js > docs/design/scene.svg   # optional: refresh the scene
python docs/design/build_board.py                            # inline CSS + scene into one file
```

`build_board.py` also regenerates `tokens.tokens-studio.json`. The build refuses to write a board
that would fetch anything at runtime.
