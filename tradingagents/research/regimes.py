"""Regime classification and multi-timeframe agreement.

Higher timeframes set the context, the trading timeframe identifies the setup and
a lower timeframe confirms execution. The rules below are deterministic, use only
closed bars, and return :class:`~tradingagents.research.contracts.Regime`.

Why this matters for research: most "improvements" found on a whole sample are
regime bets in disguise. Reporting performance per regime, and refusing to trade
when the timeframe picture conflicts, is what stops a trend filter from being
credited for a bull market.

Thresholds are declared constants, not tuned per experiment. An experiment that
changes them must say so in its hypothesis record, and the parameter-sensitivity
gate will then show how much the result depended on that choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.contracts import Regime, Timeframe
from tradingagents.research.indicators import adx, atr, bollinger, ema

# Declared, not fitted: ADX 25 is the conventional "trending" boundary and the
# volatility bands are percentiles of the instrument's own recent history.
TREND_ADX = 25.0
QUIET_ADX = 20.0
VOLATILE_PERCENTILE = 0.90
COMPRESSED_PERCENTILE = 0.30
LOOKBACK = 100


def classify_regime(frame: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """One regime label per bar, with the measurements that produced it.

    The order of the tests is the definition:

    1. ``volatile`` — ATR percentile above 0.90 and ADX below 25. High energy,
       no direction: the state that punishes trend entries hardest.
    2. ``trend_up`` / ``trend_down`` — ADX at least 25 and the matching
       directional indicator on top.
    3. ``breakout`` — price outside its 20-bar Donchian range while the 20-bar
       range width was compressed in the prior 20 bars.
    4. ``sideways`` — ADX below 20 inside a compressed range.
    5. ``unstable`` — everything else, including a warm-up bar where the
       measurement itself is not yet available. Unstable trades are not taken.
    """
    direction = adx(frame, 14)
    average_range = atr(frame, 14)
    volatility = (average_range / frame["close"]).rolling(lookback, min_periods=20).rank(pct=True)
    bands = bollinger(frame["close"], 20, 2.0)
    width_percentile = bands["bb_width"].rolling(lookback, min_periods=20).rank(pct=True)
    prior_high = frame["high"].rolling(20, min_periods=20).max().shift(1)
    prior_low = frame["low"].rolling(20, min_periods=20).min().shift(1)
    compressed_before = width_percentile.shift(20)
    trend_up = (ema(frame["close"], 20) > ema(frame["close"], 50)).astype(float)

    labels: list[str] = []
    for i in range(len(frame)):
        adx_value = direction["adx"].iloc[i]
        volatile = volatility.iloc[i]
        width = width_percentile.iloc[i]
        breakout = (np.isfinite(prior_high.iloc[i]) and np.isfinite(prior_low.iloc[i])
                    and (frame["close"].iloc[i] > prior_high.iloc[i]
                         or frame["close"].iloc[i] < prior_low.iloc[i]))
        compressed = (np.isfinite(compressed_before.iloc[i])
                      and compressed_before.iloc[i] <= COMPRESSED_PERCENTILE)
        if not np.isfinite(adx_value):
            labels.append(Regime.UNSTABLE.value)
        elif np.isfinite(volatile) and volatile >= VOLATILE_PERCENTILE and adx_value < TREND_ADX:
            labels.append(Regime.VOLATILE.value)
        elif adx_value >= TREND_ADX and direction["di_plus"].iloc[i] > direction["di_minus"].iloc[i]:
            labels.append(Regime.TREND_UP.value)
        elif adx_value >= TREND_ADX and direction["di_minus"].iloc[i] > direction["di_plus"].iloc[i]:
            labels.append(Regime.TREND_DOWN.value)
        elif breakout and compressed:
            labels.append(Regime.BREAKOUT.value)
        elif adx_value < QUIET_ADX and np.isfinite(width) and width <= COMPRESSED_PERCENTILE * 2:
            labels.append(Regime.SIDEWAYS.value)
        else:
            labels.append(Regime.UNSTABLE.value)

    regime_series = pd.Series(labels, index=frame.index, name="regime")
    codes = {member.value: float(i) for i, member in enumerate(Regime)}
    return pd.DataFrame({
        "regime": regime_series,
        "regime_id": regime_series.map(codes).astype(float),
        "regime_adx": direction["adx"],
        "regime_volatility_pct": volatility,
        "regime_width_pct": width_percentile,
        "regime_trend_up": trend_up,
    }, index=frame.index)


def align_known(features: pd.DataFrame, timeframe: Timeframe, target_index: pd.DatetimeIndex,
                prefix: str) -> pd.DataFrame:
    """Re-stamp a feature table onto another timeframe's clock without look-ahead.

    A source feature computed at the close of a bar is known at
    ``bar_start + timeframe``. It is attached to the last target timestamp at or
    after that instant, so a 4H feature cannot inform a decision taken while the
    4H bar was still forming.
    """
    if features is None or len(features) == 0 or len(target_index) == 0:
        return pd.DataFrame(index=target_index)
    folded = features.copy()
    folded["__known_at"] = folded.index + pd.Timedelta(minutes=timeframe.minutes)
    folded = folded.sort_values("__known_at").drop_duplicates("__known_at", keep="last")
    target = pd.DataFrame({"__known_at": pd.DatetimeIndex(target_index)})
    merged = pd.merge_asof(target, folded, on="__known_at", direction="backward",
                           allow_exact_matches=True)
    merged = merged.set_index("__known_at")
    merged.columns = [f"{prefix}{column}" for column in merged.columns]
    return merged.reindex(pd.DatetimeIndex(target_index))


def timeframe_agreement(primary_structure: pd.Series, context_structure: pd.Series) -> pd.DataFrame:
    """Whether the trading and context timeframes point the same way.

    ``mtf_aligned_up`` requires both structures bullish; ``mtf_aligned_down`` both
    bearish. ``mtf_conflict`` marks the case a discretionary trader would also
    refuse: the two timeframes disagree while both have an opinion.
    """
    up = (primary_structure > 0) & (context_structure > 0)
    down = (primary_structure < 0) & (context_structure < 0)
    conflict = (primary_structure * context_structure) < 0
    return pd.DataFrame({
        "mtf_aligned_up": up.astype(float),
        "mtf_aligned_down": down.astype(float),
        "mtf_conflict": conflict.astype(float),
    })


def regime_mask(regimes: pd.Series, allowed: list[str] | None) -> pd.Series:
    """A boolean mask of bars whose regime is in ``allowed`` (all bars when None)."""
    if not allowed:
        return pd.Series(True, index=regimes.index)
    wanted = {Regime(value).value if not isinstance(value, Regime) else value.value
              for value in allowed}
    return regimes.isin(wanted)


def summarise_regimes(regimes: pd.Series) -> dict[str, float]:
    """Share of bars spent in each regime, over bars that have a label."""
    counts = regimes.value_counts(normalize=True)
    return {str(label): round(float(share), 4) for label, share in counts.items()}
