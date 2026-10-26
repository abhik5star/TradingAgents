#!/usr/bin/env python3
"""Build the design board into one portable file.

The board is authored as `board.html` plus two stylesheets, because a single file with
everything inlined is unreadable to edit. This script stitches them together and inlines
the scene drawing, so the result opens from disk with no server, no fonts and no network —
the same constraint the dashboard itself is built under.

    python docs/design/build_board.py

Writes: docs/design/strategy-intelligence.html
        docs/design/tokens.tokens-studio.json   (the same palette, in the format the
                                                 Tokens Studio plugin imports into Figma)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "strategy-intelligence.html"


def read(name: str) -> str:
    path = HERE / name
    if not path.exists():
        raise SystemExit(f"missing design source: {path}")
    return path.read_text(encoding="utf-8")


def inline_scene(svg: str) -> str:
    """Strip the fixed size so the SVG fills the stage box, and drop the XML prologue."""
    svg = svg.strip()
    svg = re.sub(r"\swidth=\"\d+\"", "", svg, count=1)
    svg = re.sub(r"\sheight=\"\d+\"", "", svg, count=1)
    # Give it a stable id so the board can refer to it and so duplicate ids cannot clash.
    return svg.replace("<svg ", '<svg id="scene-art" ', 1)


def write_tokens_studio() -> None:
    """Emit the palette in Tokens Studio's format, which Figma imports as variables.

    There is no Figma connector in this environment, so the handoff is a file: paste this
    into the Tokens Studio plugin and the two themes arrive as real colour variables
    rather than as screenshots somebody has to eyedrop.
    """
    import json

    tokens = json.loads(read("tokens.json"))
    document: dict[str, object] = {
        "$themes": [
            {"name": "evidence", "selectedTokenSets": {"global": "source"}},
            {"name": "market", "selectedTokenSets": {"global": "source"}},
        ],
        "$metadata": {"tokenSetOrder": ["global"]},
        "global": {},
    }

    colours: dict[str, object] = {}
    for theme, palette in tokens["themes"].items():
        colours[theme] = {
            name: {"$value": value, "$type": "color"} for name, value in palette.items()
        }
    colours["series"] = {
        f"track-{index + 1}": {"$value": value, "$type": "color"}
        for index, value in enumerate(tokens["series"])
    }
    document["global"]["color"] = colours

    document["global"]["space"] = {
        str(value): {"$value": f"{value}px", "$type": "spacing"} for value in tokens["space"]
    }
    document["global"]["radius"] = {
        name: {"$value": f"{value}px", "$type": "borderRadius"}
        for name, value in tokens["radius"].items()
        if name != "pill"
    }
    document["global"]["radius"]["pill"] = {"$value": "999px", "$type": "borderRadius"}
    document["global"]["typography"] = {
        "family": {"$value": tokens["type"]["family"], "$type": "fontFamilies"},
        "mono": {"$value": tokens["type"]["mono"], "$type": "fontFamilies"},
    }
    (HERE / "tokens.tokens-studio.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    template = read("board.html")
    board_css = read("board.css")
    frame_css = read("frame.css")
    scene = inline_scene(read("scene.svg"))

    for placeholder in ("/*{{BOARD_CSS}}*/", "/*{{FRAME_CSS}}*/", "<!--{{SCENE}}-->"):
        if placeholder not in template:
            raise SystemExit(f"template is missing {placeholder}")

    html = (template
            .replace("/*{{BOARD_CSS}}*/", board_css)
            .replace("/*{{FRAME_CSS}}*/", frame_css)
            .replace("<!--{{SCENE}}-->", scene))

    # A design file must not reach for anything it cannot carry. The SVG XML namespace is
    # an identifier, not a request, so it is the one URL-shaped string allowed through.
    probe = html.replace('xmlns="http://www.w3.org/2000/svg"', "xmlns:svg-namespace")
    for forbidden in ("http://", "https://", "<script src", '<link rel="stylesheet"'):
        if forbidden in probe:
            raise SystemExit(f"the built board must be self-contained: found {forbidden}")

    OUTPUT.write_text(html, encoding="utf-8")
    write_tokens_studio()
    frames = html.count('class="frame ')
    print(f"wrote {OUTPUT.relative_to(HERE.parent.parent)} "
          f"({len(html.encode('utf-8')):,} bytes, {frames} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
