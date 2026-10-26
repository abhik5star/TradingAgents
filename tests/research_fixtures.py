"""Deterministic synthetic market data for the research-engine tests.

This module exists **only** for tests. Nothing here is loaded by the package at
runtime, and no backtest in the dashboard or CLI can be produced from it: the
research engine refuses to report on data it did not load through a declared
source. Its purpose is to make the engine's behaviour (look-ahead, costs,
stops, splits, rollback) assertable without a network connection.

The generator alternates trending and ranging regimes so the regime classifier
and the regime-filter hypotheses have something real to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.contracts import Timeframe
from tradingagents.research.data import ExchangeSpec, MarketData, exchange_for, quality_report


def synthetic_bars(
    n: int = 900,
    *,
    seed: int = 11,
    start: str = "2022-01-03",
    timeframe: Timeframe | str = Timeframe.D1,
    price: float = 100.0,
    base_volatility: float = 0.012,
    regime_length: int = 120,
    tz: str = "UTC",
) -> pd.DataFrame:
    """A deterministic OHLCV frame with alternating trend/range/volatile regimes.

    ``seed`` makes the series reproducible, so a failing assertion can be re-run
    exactly. Prices follow a random walk whose drift and volatility change every
    ``regime_length`` bars.
    """
    timeframe = Timeframe.parse(timeframe)
    rng = np.random.default_rng(seed)
    drift_cycle = (0.0016, 0.0, -0.0014, 0.0, 0.0020, -0.0004)
    volatility_cycle = (1.0, 0.7, 1.2, 1.8, 0.9, 1.4)

    if timeframe is Timeframe.D1:
        index = pd.bdate_range(start=start, periods=n, tz=tz)
    else:
        index = _intraday_index(start, n, timeframe, tz)

    closes = np.empty(n)
    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    volumes = np.empty(n)
    current = price
    for i in range(n):
        regime = (i // regime_length) % len(drift_cycle)
        drift = drift_cycle[regime]
        volatility = base_volatility * volatility_cycle[regime]
        shock = rng.normal(0.0, volatility)
        gap = rng.normal(0.0, volatility / 3)
        open_price = max(0.5, current * (1 + gap))
        close_price = max(0.5, open_price * (1 + drift + shock))
        wick = abs(rng.normal(0.0, volatility / 2)) * close_price
        highs[i] = max(open_price, close_price) + wick
        lows[i] = max(0.25, min(open_price, close_price) - wick)
        opens[i], closes[i] = open_price, close_price
        volumes[i] = float(rng.lognormal(mean=11.0, sigma=0.30) * (1 + abs(shock) * 12))
        current = close_price

    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": volumes}, index=index)


def _intraday_index(start: str, n: int, timeframe: Timeframe, tz: str) -> pd.DatetimeIndex:
    """Bars inside a 09:30-16:00 New York session, converted to UTC."""
    minutes = timeframe.minutes
    per_session = max(1, 390 // minutes)
    day = pd.Timestamp(start, tz="America/New_York")
    stamps: list[pd.Timestamp] = []
    cursor = day.normalize() + pd.Timedelta(hours=9, minutes=30)
    while len(stamps) < n:
        if cursor.weekday() < 5:
            for slot in range(per_session):
                if len(stamps) >= n:
                    break
                stamps.append(cursor + pd.Timedelta(minutes=slot * minutes))
        cursor = (cursor + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=30)
    return pd.DatetimeIndex(stamps).tz_convert("UTC")


def market_data(bars: pd.DataFrame, ticker: str = "SYNTH", *,
                asset_type: str = "stock", timeframes: list[Timeframe | str] | None = None,
                min_bars: int = 200, as_of: str | None = None) -> MarketData:
    """Wrap synthetic bars as :class:`MarketData`, with the usual quality reports.

    ``timeframes`` defaults to the bar size itself; a daily frame can also be
    resampled into a coarser one when a test needs multi-timeframe context.
    """
    from tradingagents.research.data import resample_frame

    spec = exchange_for(ticker, asset_type)
    wanted = [Timeframe.parse(tf) for tf in (timeframes or [Timeframe.D1])]
    frames = {}
    for timeframe in wanted:
        frames[timeframe] = resample_frame(bars, timeframe) if timeframe is not Timeframe.D1 else bars
    # The fixture is "as of" its last bar: a synthetic series is not stale, it is
    # simply not live, and tests are asserting on the engine, not on freshness.
    reference = as_of or str(bars.index[-1].date())
    reports = {tf.value: quality_report(frame, ticker, tf, spec, start=None, end=reference,
                                        min_bars=min_bars)
               for tf, frame in frames.items()}
    return MarketData(ticker=ticker, frames=frames, asset_type=asset_type, exchange=spec,
                      quality=reports, provenance={"source": "synthetic_test_fixture"})


def trending_bars(n: int = 400, *, seed: int = 5, up: bool = True) -> pd.DataFrame:
    """A clean directional series, for tests that need the trend to be unambiguous."""
    bars = synthetic_bars(n=n, seed=seed, base_volatility=0.006, regime_length=n)
    direction = 1 if up else -1
    drift = np.linspace(0, 0.25 * direction, n)
    closes = 100 * (1 + drift)
    opens = np.concatenate([[100.0], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.002
    lows = np.minimum(opens, closes) * 0.998
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": bars["volume"].to_numpy()}, index=bars.index)


def equity_spec() -> ExchangeSpec:
    return ExchangeSpec("America/New_York", (("09:30", "16:00"),), "us_equities")
