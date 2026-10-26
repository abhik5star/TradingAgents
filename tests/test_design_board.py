"""The design file is a deliverable too, so it is checked like one.

The board is hand-authored HTML that a build step stitches into a single portable file.
These tests hold the two properties that make it usable — it must open from disk with no
network, and it must not drift from the palette the application actually renders.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DESIGN = Path(__file__).resolve().parents[1] / "docs" / "design"
BOARD = DESIGN / "strategy-intelligence.html"
NODE = shutil.which("node")


@pytest.fixture(scope="module")
def board() -> str:
    assert BOARD.exists(), "build it with: python docs/design/build_board.py"
    return BOARD.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_board_opens_from_disk_with_no_network(board):
    probe = board.replace('xmlns="http://www.w3.org/2000/svg"', "xmlns:svg")
    for forbidden in ("http://", "https://", "<script src", "<link", "@import", "url(http"):
        assert forbidden not in probe, f"the design file must be self-contained: {forbidden}"
    # Fonts are the system stack, not a webfont.
    assert "font-family" in board and "fonts.googleapis" not in board


@pytest.mark.unit
def test_every_frame_is_present_and_the_scene_is_embedded(board):
    assert board.count('class="frame ') == 6
    assert board.count('id="scene-art"') == 1
    assert board.count("<polygon") > 300, "the 3D scene should be embedded, not referenced"


@pytest.mark.unit
def test_the_layers_panel_points_at_things_that_exist(board):
    ids = set(re.findall(r'id="([^"]+)"', board))
    targets = set(re.findall(r'data-target="([^"]+)"', board))
    meta = set(re.findall(r"^\s{4}(\w+): \{ kind:", board, re.M))
    assert targets, "no selectable layers"
    assert targets <= ids, f"layers without a target: {sorted(targets - ids)}"
    assert targets <= meta, f"layers without inspector metadata: {sorted(targets - meta)}"
    assert not [identifier for identifier in ids if board.count(f'id="{identifier}"') > 1]


@pytest.mark.unit
def test_the_board_scripts_parse(board):
    scripts = re.findall(r"<script>(.*?)</script>", board, re.S)
    assert len(scripts) == 1
    if NODE is None:
        pytest.skip("node is not installed")
    path = Path("/tmp/design_board_script.js")
    path.write_text(scripts[0], encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(path)], capture_output=True, text=True,
                            timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_the_tokens_are_the_ones_the_app_renders():
    """A design file that invents a palette is a screenshot with extra steps."""
    from tradingagents.research.livepage import PALETTE as live_palette

    tokens = json.loads((DESIGN / "tokens.json").read_text(encoding="utf-8"))
    assert tuple(tokens["series"]) == tuple(live_palette)

    market = tokens["themes"]["market"]
    source = (DESIGN.parents[1] / "tradingagents" / "research" / "livepage.py").read_text(
        encoding="utf-8")
    for name in ("stage", "live", "sim", "muted"):
        assert market[name].lower() in source.lower(), f"{name} is not in the shipped CSS"

    evidence = tokens["themes"]["evidence"]
    server = (DESIGN.parents[1] / "tradingagents" / "research" / "server.py").read_text(
        encoding="utf-8")
    for name in ("ink", "paper", "accent", "ok", "warn", "bad"):
        assert evidence[name].lower() in server.lower(), f"{name} is not in the dashboard CSS"


@pytest.mark.unit
def test_the_figma_token_export_matches_the_json_palette():
    tokens = json.loads((DESIGN / "tokens.json").read_text(encoding="utf-8"))
    exported = json.loads((DESIGN / "tokens.tokens-studio.json").read_text(encoding="utf-8"))
    colours = exported["global"]["color"]
    for theme, palette in tokens["themes"].items():
        for name, value in palette.items():
            assert colours[theme][name]["$value"] == value
            assert colours[theme][name]["$type"] == "color"
    assert [entry["name"] for entry in exported["$themes"]] == ["evidence", "market"]


@pytest.mark.unit
def test_the_board_does_not_claim_anything_the_engine_would_not():
    """Copy rules, checked where they can be checked: no guarantees, no false 'live'."""
    board = BOARD.read_text(encoding="utf-8")
    lowered = board.lower()
    for promise in ("guaranteed", "risk-free", "risk free", "will profit", "sure win"):
        assert promise not in lowered
    # The simulated feeds in the sample must be labelled as such wherever they appear.
    assert "simulated" in lowered
    assert "live trading disabled" in lowered
    # And the paper action is the only one offered as a button.
    buttons = re.findall(r'class="btn[^"]*">([^<]+)<', board)
    assert "Run paper session" in buttons
    assert not [label for label in buttons if "approve" in label.lower()
                and "cli only" not in label.lower()]
