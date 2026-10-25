"""Build the single feature table every strategy reads.

One table per instrument, indexed by the trading timeframe's bar timestamps, with
columns grouped by prefix:

* ``sma_``, ``ema_``, ``rsi_``, ``atr_``, ``adx_``, ``supertrend_``, ``bb_`` … —
  technical indicators,
* ``bos_``, ``choch_``, ``sweep_``, ``fvg_``, ``swing_``, ``deal_range_`` … — SMC/ICT
  structure,
* ``pattern_``, ``support``, ``resistance``, ``retest_``, ``or_`` … — price action,
* ``regime_`` — the classified market state,
* ``htf_`` — the same measurements taken on the higher timeframe, re-stamped so
  only closed higher-timeframe bars are visible,
* ``ltf_`` — lower-timeframe confirmation,
* ``proxy_`` — candle-volume proxies, explicitly not order flow.

The table is *causal by construction*: every source series is a rolling or
shifted function of past bars, and cross-timeframe alignment goes through
:func:`~tradingagents.research.regimes.align_known`, which stamps a coarse bar's
values only after that bar has closed.

``FeatureTable.version`` fingerprints the builder and the schema; experiments
store it, so a result can never be read against a different feature contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.research.contracts import stable_hash
from tradingagents.research.data import MarketData
from tradingagents.research.indicators import INDICATOR_VERSION, indicator_features
from tradingagents.research.orderflow import candle_flow_proxies
from tradingagents.research.patterns import pattern_features
from tradingagents.research.regimes import align_known, classify_regime, timeframe_agreement
from tradingagents.research.smc import smc_features

FEATURE_SCHEMA_VERSION = "1.2.0"

# Columns carried across timeframes as context. Deliberately narrow: shipping the
# whole table across timeframes multiplies columns nobody asked for, and every
# extra column is another chance to overfit.
_CONTEXT_COLUMNS = (
    "structure_dir", "supertrend_dir", "ema_stack_up", "ema_stack_down",
    "rsi_14", "atr_pct", "adx", "bb_width", "deal_range_position",
    "regime_id", "in_kill_zone",
)
_CONFIRM_COLUMNS = (
    "structure_dir", "bos_up", "bos_down", "choch_up", "choch_down",
    "ema_stack_up", "ema_stack_down", "rsi_14", "retest_up", "retest_down",
    "pattern_any_bullish", "pattern_any_bearish", "supertrend_dir",
)


@dataclass
class FeatureTable:
    """Bar-aligned features, the bars themselves, and how the table was built."""

    ticker: str
    timeframe: str
    bars: pd.DataFrame
    features: pd.DataFrame
    version: str
    data: MarketData | None = None
    context_timeframe: str | None = None
    confirm_timeframe: str | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.bars) != len(self.features):
            raise ValueError("features and bars must cover the same bars")
        if not self.bars.index.equals(self.features.index):
            raise ValueError("features must be indexed by the bars they describe")

    @property
    def columns(self) -> list[str]:
        return list(self.features.columns)

    @property
    def regimes(self) -> pd.Series:
        return self.features["regime"]

    def slice(self, start: str | None = None, end: str | None = None) -> FeatureTable:
        """A date window of the same table, keeping the builder's metadata."""
        bars = self.bars.loc[start:end] if start or end else self.bars
        features = self.features.loc[bars.index]
        return FeatureTable(ticker=self.ticker, timeframe=self.timeframe, bars=bars,
                            features=features, version=self.version, data=self.data,
                            context_timeframe=self.context_timeframe,
                            confirm_timeframe=self.confirm_timeframe, notes=list(self.notes))

    def summary(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "bars": int(len(self.bars)),
            "first": self.bars.index[0].isoformat(),
            "last": self.bars.index[-1].isoformat(),
            "features": len(self.features.columns),
            "version": self.version,
            "context_timeframe": self.context_timeframe,
            "confirm_timeframe": self.confirm_timeframe,
            "regime_share": self.regimes.value_counts(normalize=True).round(4).to_dict(),
            "notes": list(self.notes),
        }


def _join(left: pd.DataFrame, right: pd.DataFrame, source: str) -> pd.DataFrame:
    """Join a feature block, failing loudly on a duplicate column name.

    Two blocks that both define ``range_high`` would silently overwrite each other
    under pandas' default suffixes; here it is an error with the offending names,
    because a strategy must never read a column whose meaning depends on join order.
    """
    overlap = sorted(set(left.columns) & set(right.columns))
    if overlap:
        raise ValueError(f"feature block {source} duplicates existing columns: {overlap}")
    return left.join(right)


def _timeframe_features(data: MarketData, timeframe, columns: tuple[str, ...]) -> pd.DataFrame:
    """Indicators, SMC, price action and regime for one timeframe, narrow columns."""
    frame = data.frame(timeframe)
    session_keys = pd.Series(frame.index.tz_convert(data.exchange.tz).date, index=frame.index)
    features = indicator_features(frame, session_keys)
    features = _join(features, smc_features(frame, data.exchange), "smc_features")
    features = _join(features, pattern_features(frame, data.exchange), "pattern_features")
    features = _join(features, classify_regime(frame), "classify_regime")
    return features[[c for c in columns if c in features.columns]]


def build_feature_table(
    data: MarketData,
    primary_timeframe: str | None = None,
    context_timeframe: str | None = None,
    confirm_timeframe: str | None = None,
) -> FeatureTable:
    """Assemble the feature table for ``data``.

    ``primary_timeframe`` defaults to the finest loaded frame. ``context_timeframe``
    and ``confirm_timeframe`` are optional; when a requested frame was not loaded,
    the table records that in ``notes`` and continues without it, so a run
    degrades to single-timeframe research rather than inventing context.
    """
    primary = primary_timeframe or data.primary_timeframe
    from tradingagents.research.contracts import Timeframe

    primary_tf = Timeframe.parse(primary)
    notes: list[str] = []

    frame = data.frame(primary_tf)
    session_keys = pd.Series(frame.index.tz_convert(data.exchange.tz).date, index=frame.index)
    features = indicator_features(frame, session_keys)
    features = _join(features, smc_features(frame, data.exchange), "smc_features")
    features = _join(features, pattern_features(frame, data.exchange), "pattern_features")
    features = _join(features, candle_flow_proxies(frame), "candle_flow_proxies")
    features = _join(features, classify_regime(frame), "classify_regime")

    context_tf = Timeframe.parse(context_timeframe) if context_timeframe else None
    if context_tf is not None:
        if context_tf.minutes <= primary_tf.minutes:
            raise ValueError(
                f"the context timeframe ({context_tf.value}) must be coarser than the "
                f"trading timeframe ({primary_tf.value})")
        if context_tf not in data.frames:
            notes.append(f"context timeframe {context_tf.value} was not loaded; context omitted")
            context_tf = None
        else:
            context = _timeframe_features(data, context_tf, _CONTEXT_COLUMNS)
            features = _join(features, align_known(context, context_tf, frame.index, "htf_"),
                             "higher timeframe context")
            agreement = timeframe_agreement(features["structure_dir"],
                                            features["htf_structure_dir"])
            features = _join(features, agreement, "timeframe agreement")

    confirm_tf = Timeframe.parse(confirm_timeframe) if confirm_timeframe else None
    if confirm_tf is not None:
        if confirm_tf.minutes >= primary_tf.minutes:
            raise ValueError(
                f"the confirmation timeframe ({confirm_tf.value}) must be finer than the "
                f"trading timeframe ({primary_tf.value})")
        if confirm_tf not in data.frames:
            notes.append(f"confirmation timeframe {confirm_tf.value} was not loaded; "
                         "confirmation omitted")
            confirm_tf = None
        else:
            confirmation = _timeframe_features(data, confirm_tf, _CONFIRM_COLUMNS)
            features = _join(features, align_known(confirmation, confirm_tf, frame.index, "ltf_"),
                             "lower timeframe confirmation")

    features["price"] = frame["close"]
    features["hour_utc"] = frame.index.hour.astype(float)
    features["bar_index"] = np.arange(len(frame), dtype=float)
    features = features.replace([np.inf, -np.inf], np.nan)

    # A column that is empty for the whole sample is reported rather than quietly
    # kept: at daily resolution, for instance, an opening range is not defined.
    unavailable = sorted(column for column in features.columns
                         if features[column].notna().sum() == 0)
    if unavailable:
        notes.append("not available at this resolution (all empty): "
                     + ", ".join(unavailable[:12])
                     + ("…" if len(unavailable) > 12 else ""))

    version = stable_hash({
        "schema": FEATURE_SCHEMA_VERSION,
        "indicators": INDICATOR_VERSION,
        "columns": sorted(features.columns),
        "context": context_tf.value if context_tf else None,
        "confirm": confirm_tf.value if confirm_tf else None,
    })[:12]

    return FeatureTable(
        ticker=data.ticker,
        timeframe=primary_tf.value,
        bars=frame,
        features=features,
        version=version,
        data=data,
        context_timeframe=context_tf.value if context_tf else None,
        confirm_timeframe=confirm_tf.value if confirm_tf else None,
        notes=notes,
    )


#: Feature columns a strategy may reference, by group. Used by the strategy lab and
#: by the dashboard, so a candidate's rules can be explained in words.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "trend": ("sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "ema_50", "ema_stack_up",
              "ema_stack_down", "supertrend_dir", "supertrend_up", "trend_slope_20",
              "trend_r2_20", "macd_hist", "macd_bull"),
    "momentum": ("rsi_14", "stoch_rsi_k", "momentum_10", "roc_20", "macd"),
    "volatility": ("atr_14", "atr_pct", "bb_width", "bb_pctb", "compression"),
    "volume": ("volume_z", "volume_sma_20", "obv_slope_20", "vwap", "price_above_vwap"),
    "structure": ("bos_up", "bos_down", "choch_up", "choch_down", "structure_dir",
                  "sweep_buy_side", "sweep_sell_side", "displacement"),
    "liquidity": ("fvg_bull", "fvg_bear", "in_bull_ob", "in_bear_ob", "premium", "discount",
                  "equilibrium", "ote_long", "ote_short", "deal_range_position"),
    "levels": ("resistance", "support", "at_resistance", "at_support", "pdh", "pdl", "pwh",
               "pwl", "range_high", "range_low", "range_compression"),
    "patterns": ("pattern_double_top", "pattern_double_bottom", "pattern_hs_top",
                 "pattern_hs_bottom", "pattern_bull_flag", "pattern_bear_flag",
                 "pattern_triangle", "breakout_confirmed_up", "breakout_confirmed_down",
                 "false_breakout_up", "false_breakout_down", "retest_up", "retest_down"),
    "session": ("session_id", "in_kill_zone", "kill_zone_id", "minutes_of_day",
                "in_cash_session", "minutes_since_open", "or_break_up", "or_break_down"),
    "regime": ("regime_id", "regime_adx", "regime_volatility_pct"),
    "context": ("htf_structure_dir", "htf_supertrend_dir", "htf_regime_id", "mtf_aligned_up",
                "mtf_aligned_down", "mtf_conflict"),
    "confirmation": ("ltf_structure_dir", "ltf_bos_up", "ltf_bos_down", "ltf_choch_up",
                     "ltf_choch_down", "ltf_retest_up", "ltf_retest_down"),
    "flow_proxy": ("proxy_delta", "proxy_cvd_slope_20", "proxy_volume_z", "proxy_absorption"),
}


def feature_group(name: str) -> tuple[str, ...]:
    try:
        return FEATURE_GROUPS[name]
    except KeyError:
        raise KeyError(f"unknown feature group {name!r}; expected one of "
                       f"{sorted(FEATURE_GROUPS)}") from None
