"""A ten-second demo: generated bars, a real research run, a real dashboard.

Everything in this module is **synthetic and labelled as such**. It exists so a new user
can see the whole loop — hypothesis → gates → version → paper session → dashboard —
without a vendor account, a network connection or a decision about which stock to study.

Two rules keep the demo honest:

* the bars are generated here and written to a directory the user passes, never into the
  research store that holds real evidence;
* every number the demo produces is therefore a statement about *this generator*, not
  about a market. The CLI says so on start and the dashboard keeps saying it, because the
  version it registers records ``source: synthetic`` in its dataset.

Everything else — the gates, the risk policy, the paper verdict — is the real engine. The
demo does not take a shortcut through it; it runs the same code path a real research
session would, on data that is openly fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Repeated in the CLI banner and in the dashboard so the provenance is impossible to miss.
DEMO_DISCLAIMER = (
    "demo data: synthetic bars generated locally. Any number you see describes this "
    "generator, not a market. Point --csv (or drop --csv) at real bars for real work.")

#: The symbols the demo writes, and what each one is for. The three replays are the tracks
#: in the 3D view; SYNTH is the instrument the research loop actually studies.
DEMO_SYMBOLS = {
    "SYNTH": {"days": 1700, "start": "2015-01-02", "seed": 51, "regime": 120,
              "role": "the instrument the hypothesis loop studies"},
    "ALPHA": {"days": 520, "start": "2022-01-03", "seed": 5, "regime": 60,
              "role": "3D track 1 (replayed, simulated)"},
    "BETA": {"days": 520, "start": "2022-01-03", "seed": 23, "regime": 40,
             "role": "3D track 2 (replayed, simulated)"},
    "GAMMA": {"days": 520, "start": "2022-01-03", "seed": 41, "regime": 90,
              "role": "3D track 3 (replayed, simulated)"},
}

#: Where the demo studies: the loop runs on the first part, paper trading the rest.
DEMO_WINDOW = {"start": "2015-01-02", "research_end": "2020-06-30",
               "paper_end": "2021-07-08"}


def demo_bars(*, days: int = 1700, seed: int = 51, start: str = "2015-01-02",
              price: float = 100.0, regime_length: int = 120,
              volatility: float = 0.012) -> pd.DataFrame:
    """A deterministic OHLCV frame with alternating trend, range and volatile regimes.

    Deliberately the same generator as ``tests/research_fixtures.synthetic_bars`` — same
    drift cycle, same draw order, same volume law — so the demo, the documentation and the
    test fixtures all describe one series. The same seed therefore produces the same bars
    on every machine, and a screenshot can be reproduced exactly.
    """
    rng = np.random.default_rng(seed)
    drift_cycle = (0.0016, 0.0, -0.0014, 0.0, 0.0020, -0.0004)
    volatility_cycle = (1.0, 0.7, 1.2, 1.8, 0.9, 1.4)

    index = pd.bdate_range(start=start, periods=days, tz="UTC")
    opens = np.empty(days)
    highs = np.empty(days)
    lows = np.empty(days)
    closes = np.empty(days)
    volumes = np.empty(days)

    current = price
    for i in range(days):
        regime = (i // regime_length) % len(drift_cycle)
        drift = drift_cycle[regime]
        scale = volatility * volatility_cycle[regime]
        shock = rng.normal(0.0, scale)
        gap = rng.normal(0.0, scale / 3)
        open_ = max(0.5, current * (1 + gap))
        close = max(0.5, open_ * (1 + drift + shock))
        wick = abs(rng.normal(0.0, scale / 2)) * close
        opens[i] = open_
        closes[i] = close
        highs[i] = max(open_, close) + wick
        lows[i] = max(0.25, min(open_, close) - wick)
        volumes[i] = float(rng.lognormal(mean=11.0, sigma=0.30) * (1 + abs(shock) * 12))
        current = close

    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": volumes}, index=index)


def write_demo_csvs(directory: str | Path, *, force: bool = False) -> dict[str, Path]:
    """Write one CSV per demo symbol into ``directory`` (``<SYMBOL>-1d.csv``).

    Returns the paths written. Existing files are left alone unless ``force`` is set, so
    re-running the demo is cheap and never rewrites data a user has edited.
    """
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for symbol, spec in DEMO_SYMBOLS.items():
        path = target / f"{symbol}-1d.csv"
        if path.exists() and not force:
            written[symbol] = path
            continue
        frame = demo_bars(days=spec["days"], seed=spec["seed"], start=spec["start"],
                          regime_length=spec["regime"])
        frame.to_csv(path, index_label="date")
        written[symbol] = path
    return written


def demo_summary(paths: dict[str, Path]) -> list[dict[str, Any]]:
    """One row per demo symbol, for the banner: symbol, bars, closes, and why it exists."""
    rows: list[dict[str, Any]] = []
    for symbol, path in paths.items():
        try:
            frame = pd.read_csv(path, index_col="date")
        except (OSError, ValueError):  # a CSV we cannot read is reported, not fatal
            rows.append({"symbol": symbol, "bars": 0, "first": None, "last": None,
                         "role": DEMO_SYMBOLS.get(symbol, {}).get("role", ""),
                         "path": str(path)})
            continue
        rows.append({
            "symbol": symbol,
            "bars": int(len(frame)),
            "first": float(frame["close"].iloc[0]) if len(frame) else None,
            "last": float(frame["close"].iloc[-1]) if len(frame) else None,
            "role": DEMO_SYMBOLS.get(symbol, {}).get("role", ""),
            "path": str(path),
        })
    return rows
