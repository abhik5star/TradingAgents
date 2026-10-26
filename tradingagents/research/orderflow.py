"""Order flow and microstructure — only where the data actually exists.

The engine cannot invent a bid/ask tape out of candles. So this module is built
around an explicit capability check:

* With trade-print data (timestamp, price, size, aggressor side) it computes
  volume delta, cumulative volume delta, aggressive buying/selling, absorption
  and large-print imbalance.
* With quote or depth snapshots it computes the quoted spread, book imbalance and
  depth withdrawal.
* With an open-interest (and optional funding) series it computes open-interest
  change and the classic price/OI quadrants.
* With **only** OHLCV it computes *candle proxies*, and every column is prefixed
  ``proxy_`` and reported as a proxy wherever it is shown. A candle proxy is not
  order flow, and treating it as though it were is how backtests get fictional
  alpha.

:func:`capabilities` is the single place that answers "what do we have?"; the
dashboard and every experiment record print its answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FlowCapabilities:
    """What the loaded data can and cannot support."""

    trades: bool = False
    quotes: bool = False
    order_book: bool = False
    open_interest: bool = False
    funding: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def true_order_flow(self) -> bool:
        """True only with real trade prints (or a book), never from candles."""
        return self.trades or self.order_book

    def as_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "quotes": self.quotes,
            "order_book": self.order_book,
            "open_interest": self.open_interest,
            "funding": self.funding,
            "true_order_flow": self.true_order_flow,
            "candle_proxies_only": not self.true_order_flow,
            "notes": list(self.notes),
        }


def capabilities(*, trades: pd.DataFrame | None = None, quotes: pd.DataFrame | None = None,
                 order_book: pd.DataFrame | None = None, open_interest: pd.Series | None = None,
                 funding: pd.Series | None = None) -> FlowCapabilities:
    """Detect what was supplied. Empty data counts as absent, not as zero flow."""
    found = FlowCapabilities(
        trades=trades is not None and len(trades) > 0,
        quotes=quotes is not None and len(quotes) > 0,
        order_book=order_book is not None and len(order_book) > 0,
        open_interest=open_interest is not None and len(open_interest.dropna()) > 0,
        funding=funding is not None and len(funding.dropna()) > 0,
    )
    if not found.trades:
        found.notes.append("no trade prints: volume delta and CVD are unavailable")
    if not found.order_book:
        found.notes.append("no order-book snapshots: quote imbalance and depth are unavailable")
    if not found.open_interest:
        found.notes.append("no open interest: derivative positioning is unavailable")
    return found


def _side_column(trades: pd.DataFrame) -> pd.Series:
    if "side" not in trades.columns:
        raise ValueError("trade prints need a 'side' column of buy/sell (aggressor side)")
    side = trades["side"].astype(str).str.lower().str.strip()
    unknown = ~side.isin(("buy", "sell", "b", "s"))
    if unknown.any():
        raise ValueError("trade side must be buy or sell; refusing to guess the aggressor")
    return side.map({"buy": 1.0, "b": 1.0, "sell": -1.0, "s": -1.0})


def trade_flow_features(trades: pd.DataFrame) -> pd.DataFrame:
    """Volume delta, CVD and absorption from genuine trade prints.

    ``trades`` needs ``timestamp``, ``price``, ``size`` and ``side`` (the
    aggressor). Absorption is flagged where a bar of heavy one-sided flow barely
    moves price: the classic "someone is taking the other side" reading, offered
    as a hypothesis to test rather than a fact.
    """
    required = {"timestamp", "price", "size"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trade prints are missing {sorted(missing)}")
    frame = trades.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame["signed_size"] = frame["size"].astype(float) * _side_column(frame)
    frame = frame.set_index("timestamp")

    grouped = frame.resample("1min").agg(
        buy_volume=("signed_size", lambda s: float(s[s > 0].sum())),
        sell_volume=("signed_size", lambda s: float(-s[s < 0].sum())),
        price_open=("price", "first"), price_close=("price", "last"),
        prints=("size", "size"),
    )
    grouped["delta"] = grouped["buy_volume"] - grouped["sell_volume"]
    grouped["cvd"] = grouped["delta"].cumsum()
    grouped["aggressive_ratio"] = grouped["buy_volume"] / (
        (grouped["buy_volume"] + grouped["sell_volume"]).replace(0.0, np.nan))
    movement = (grouped["price_close"] - grouped["price_open"]).abs()
    turnover = grouped["buy_volume"] + grouped["sell_volume"]
    grouped["absorption"] = ((turnover > turnover.rolling(20, min_periods=5).mean() * 1.5)
                             & (movement <= movement.rolling(20, min_periods=5).mean())).astype(float)
    return grouped


def order_book_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Quoted spread, depth and book imbalance from order-book snapshots.

    ``snapshots`` needs ``timestamp``, ``bid``, ``ask``, ``bid_size`` and
    ``ask_size``. Imbalance is ``(bid_size - ask_size) / (bid_size + ask_size)``
    in ``[-1, 1]``; a series of it falling while price rises is liquidity being
    pulled from the bid, not "support".
    """
    required = {"timestamp", "bid", "ask", "bid_size", "ask_size"}
    missing = required - set(snapshots.columns)
    if missing:
        raise ValueError(f"order-book snapshots are missing {sorted(missing)}")
    frame = snapshots.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").set_index("timestamp")
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["spread_bps"] = frame["spread"] / ((frame["ask"] + frame["bid"]) / 2) * 1e4
    frame["mid"] = (frame["ask"] + frame["bid"]) / 2
    frame["imbalance"] = ((frame["bid_size"] - frame["ask_size"])
                          / (frame["bid_size"] + frame["ask_size"]).replace(0.0, np.nan))
    frame["depth"] = frame["bid_size"] + frame["ask_size"]
    frame["depth_change"] = frame["depth"].pct_change()
    return frame


def open_interest_features(open_interest: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Open-interest change and the price/OI quadrant reading.

    Rising price with rising OI is new long exposure; rising price with falling OI
    is short covering. The quadrant is a *description* of the data, not a signal.
    """
    series = pd.Series(open_interest, dtype=float).sort_index()
    price = pd.Series(close, dtype=float).reindex(series.index)
    change = series.diff()
    price_change = price.pct_change()
    quadrant = np.select(
        [(price_change > 0) & (change > 0), (price_change > 0) & (change < 0),
         (price_change < 0) & (change > 0), (price_change < 0) & (change < 0)],
        ["long_buildup", "short_covering", "short_buildup", "long_liquidation"],
        default="flat",
    )
    return pd.DataFrame({"open_interest": series, "oi_change": change,
                         "oi_change_pct": series.pct_change(), "oi_quadrant": quadrant},
                        index=series.index)


def candle_flow_proxies(frame: pd.DataFrame) -> pd.DataFrame:
    """Volume-based *proxies* from OHLCV, named so nobody mistakes them for flow.

    The delta proxy splits a bar's volume by where the close sits in its range.
    It is a candlestick heuristic; the columns say ``proxy`` and reports label
    them as such.
    """
    span = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    location = ((frame["close"] - frame["low"]) / span).clip(0.0, 1.0)
    volume = frame["volume"].fillna(0.0)
    delta = (2 * location - 1) * volume
    features = pd.DataFrame(index=frame.index)
    features["proxy_delta"] = delta
    features["proxy_cvd"] = delta.cumsum()
    features["proxy_cvd_slope_20"] = (features["proxy_cvd"].diff(20)
                                      / volume.rolling(20, min_periods=20).sum().replace(0.0, np.nan))
    features["proxy_volume_z"] = ((volume - volume.rolling(20, min_periods=20).mean())
                                  / volume.rolling(20, min_periods=20).std(ddof=0).replace(0.0, np.nan))
    features["proxy_absorption"] = ((features["proxy_volume_z"] > 1.5)
                                    & (span <= span.rolling(20, min_periods=5).mean())).astype(float)
    return features


@dataclass
class ExecutionQuality:
    """Measured execution against the assumptions a backtest was given."""

    fills: int
    avg_spread_bps: float | None = None
    avg_slippage_bps: float | None = None
    assumed_slippage_bps: float | None = None
    worse_than_assumed_pct: float | None = None
    rejected: int = 0

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def execution_quality(fills: pd.DataFrame | None, assumed_slippage_bps: float | None = None,
                      rejected: int = 0) -> ExecutionQuality:
    """Compare realised fills with the backtest's cost assumptions.

    ``fills`` needs ``expected_price`` and ``fill_price`` (and optionally
    ``spread_bps``). Without fills the result says so rather than returning a
    flattering zero.
    """
    if fills is None or len(fills) == 0:
        return ExecutionQuality(fills=0, assumed_slippage_bps=assumed_slippage_bps, rejected=rejected)
    frame = fills.copy()
    if not {"expected_price", "fill_price"} <= set(frame.columns):
        raise ValueError("fills need expected_price and fill_price to measure slippage")
    side = frame.get("side", pd.Series(["buy"] * len(frame), index=frame.index))
    sign = side.astype(str).str.lower().map({"buy": 1.0, "long": 1.0, "sell": -1.0,
                                            "short": -1.0}).fillna(1.0)
    slippage_bps = ((frame["fill_price"] - frame["expected_price"]) / frame["expected_price"]
                    * 1e4 * sign)
    average = float(slippage_bps.mean())
    worse = (float((slippage_bps > slip).mean() * 100)
             if (slip := assumed_slippage_bps) is not None else None)
    return ExecutionQuality(
        fills=int(len(frame)),
        avg_spread_bps=(float(frame["spread_bps"].mean())
                        if "spread_bps" in frame.columns else None),
        avg_slippage_bps=average,
        assumed_slippage_bps=assumed_slippage_bps,
        worse_than_assumed_pct=worse,
        rejected=rejected,
    )


#: Names of the proxy columns, so the dashboard can say plainly that only proxies exist.
PROXY_COLUMNS = ("proxy_delta", "proxy_cvd", "proxy_cvd_slope_20", "proxy_volume_z",
                 "proxy_absorption", "proxy_poc")
