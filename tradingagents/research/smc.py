"""Smart Money Concepts (SMC) and ICT structure, as measurable rules.

Discretionary SMC talk ("institutional order flow", "the market wants liquidity")
cannot be tested. Every concept below is therefore reduced to a rule that another
person can re-implement from this file and get the same series:

* **Swing high/low** — a bar whose high (low) exceeds the ``k`` bars on each side.
  It is only *confirmed* ``k`` bars later, and the confirmed series says so: the
  feature appears at the confirmation bar, never at the bar itself.
* **BOS / CHoCH** — a close beyond the last confirmed swing, in the direction of
  the prevailing structure (BOS) or against it (CHoCH).
* **Liquidity sweep** — a bar that trades beyond the last confirmed swing and
  closes back inside it. Buy-side liquidity sits above swing highs, sell-side
  below swing lows.
* **Fair value gap** — a three-bar imbalance where bar *i*'s low is above bar
  *i-2*'s high (bullish), or its high is below bar *i-2*'s low (bearish).
* **Order block** — the last opposite-colour candle before a displacement candle
  that closes beyond it, with displacement defined as a range of at least
  ``displacement_atr`` average true ranges.
* **Premium / discount** — the position of price inside the dealing range formed
  by the last confirmed swing high and low.
* **Previous day / week extremes** — computed from *completed* calendar periods
  in the exchange's own time zone.

Everything is causal. Where a concept needs future bars to exist (a swing), the
feature is stamped with the bar that confirmed it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.contracts import Session
from tradingagents.research.data import ExchangeSpec, in_cash_session, kill_zone, session_of
from tradingagents.research.indicators import atr


def confirmed_swings(frame: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Swing levels as known at each bar, stamped when they became confirmable.

    ``swing_high``/``swing_low`` hold the most recent *confirmed* level, and
    ``new_swing_high``/``new_swing_low`` mark the confirming bar itself.
    """
    high, low = frame["high"], frame["low"]
    window = 2 * k + 1
    is_high = high == high.rolling(window, center=True, min_periods=window).max()
    is_low = low == low.rolling(window, center=True, min_periods=window).min()
    # A swing at bar i is knowable only at bar i + k.
    confirmed_high = is_high.shift(k).fillna(False).astype(bool)
    confirmed_low = is_low.shift(k).fillna(False).astype(bool)

    # The level is the swing bar's own extreme, stamped at the bar that confirmed
    # it: ``high`` at the confirming bar is a different price entirely.
    level_high = high.shift(k).where(confirmed_high).ffill()
    level_low = low.shift(k).where(confirmed_low).ffill()
    return pd.DataFrame({
        "swing_high": level_high,
        "swing_low": level_low,
        "new_swing_high": confirmed_high.astype(float),
        "new_swing_low": confirmed_low.astype(float),
    }, index=frame.index)


def market_structure(frame: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """BOS, CHoCH, structure direction and liquidity sweeps, bar by bar.

    State machine, all of it dependent on levels that were already confirmed:

    * ``structure_dir`` is ``+1`` after a bullish break, ``-1`` after a bearish
      one, and ``0`` until the first break.
    * A close above the last confirmed swing high while structure is already up
      is a **BOS**; while structure is down it is a **CHoCH** (and the same in
      reverse). That is the standard distinction, made explicit.
    * A bar that trades through the swing level but closes back behind it is a
      **sweep** of that side's liquidity, not a break.
    """
    swings = confirmed_swings(frame, k)
    close = frame["close"]
    high, low = frame["high"], frame["low"]
    bos_up = np.zeros(len(frame))
    bos_down = np.zeros(len(frame))
    choch_up = np.zeros(len(frame))
    choch_down = np.zeros(len(frame))
    sweep_high = np.zeros(len(frame))
    sweep_low = np.zeros(len(frame))
    direction = np.zeros(len(frame), dtype=int)

    last_high = np.nan
    last_low = np.nan
    structure = 0
    swing_high_values = swings["swing_high"].to_numpy()
    swing_low_values = swings["swing_low"].to_numpy()
    for i in range(len(frame)):
        if np.isfinite(swing_high_values[i]):
            last_high = swing_high_values[i]
        if np.isfinite(swing_low_values[i]):
            last_low = swing_low_values[i]

        if np.isfinite(last_high) and close.iloc[i] > last_high:
            if structure >= 0:
                bos_up[i] = 1.0
            else:
                choch_up[i] = 1.0
            structure = 1
            last_high = np.nan       # the level has been consumed by the break
        elif np.isfinite(last_low) and close.iloc[i] < last_low:
            if structure <= 0:
                bos_down[i] = 1.0
            else:
                choch_down[i] = 1.0
            structure = -1
            last_low = np.nan
        else:
            if np.isfinite(last_high) and high.iloc[i] > last_high and close.iloc[i] < last_high:
                sweep_high[i] = 1.0
            if np.isfinite(last_low) and low.iloc[i] < last_low and close.iloc[i] > last_low:
                sweep_low[i] = 1.0
        direction[i] = structure

    return pd.DataFrame({
        "bos_up": bos_up, "bos_down": bos_down,
        "choch_up": choch_up, "choch_down": choch_down,
        "sweep_buy_side": sweep_high, "sweep_sell_side": sweep_low,
        "structure_dir": direction.astype(float),
    }, index=frame.index)


def fair_value_gaps(frame: pd.DataFrame, max_age: int = 50) -> pd.DataFrame:
    """Unmitigated three-bar imbalances and the distance to the nearest one.

    A bullish gap exists at bar *i* when ``low[i] > high[i-2]``; its zone is
    ``[high[i-2], low[i]]``. The zone stays "unmitigated" until a later bar
    trades back into it, which is exactly how a retest entry is described.
    """
    high, low, close = frame["high"], frame["low"], frame["close"]
    bull = (low > high.shift(2)).fillna(False).to_numpy()
    bear = (high < low.shift(2)).fillna(False).to_numpy()
    bull_bottom, bull_top = high.shift(2).to_numpy(), low.to_numpy()
    bear_bottom, bear_top = high.to_numpy(), low.shift(2).to_numpy()

    in_bull_gap = np.zeros(len(frame))
    in_bear_gap = np.zeros(len(frame))
    distance_bull = np.full(len(frame), np.nan)
    distance_bear = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        # Zones created up to and including this bar; a zone dies when price
        # closes through it or when it is older than ``max_age`` bars.
        for j in range(max(0, i - max_age), i + 1):
            if bull[j] and low.iloc[i] <= bull_top[j] and close.iloc[i] >= bull_bottom[j]:
                in_bull_gap[i] = 1.0
                distance_bull[i] = close.iloc[i] - bull_top[j]
            if bear[j] and high.iloc[i] >= bear_bottom[j] and close.iloc[i] <= bear_top[j]:
                in_bear_gap[i] = 1.0
                distance_bear[i] = bear_bottom[j] - close.iloc[i]
    return pd.DataFrame({
        "fvg_bull": in_bull_gap, "fvg_bear": in_bear_gap,
        "fvg_bull_distance": distance_bull, "fvg_bear_distance": distance_bear,
    }, index=frame.index)


def order_blocks(frame: pd.DataFrame, displacement_atr: float = 1.2,
                 max_age: int = 60) -> pd.DataFrame:
    """Last-opposite-candle order blocks, and whether price is retesting one.

    Bullish block: the most recent down candle before a bar whose range exceeds
    ``displacement_atr`` ATRs and which closes above that candle's high. The zone
    is that candle's low..high. Bearish block is the mirror image. A block is
    "live" for ``max_age`` bars, and ``in_bull_ob``/``in_bear_ob`` mark a retest.
    """
    average_range = atr(frame, 14).to_numpy()
    open_, high, low, close = (frame[c].to_numpy() for c in ("open", "high", "low", "close"))
    in_bull = np.zeros(len(frame))
    in_bear = np.zeros(len(frame))
    bull_zones: list[tuple[int, float, float]] = []
    bear_zones: list[tuple[int, float, float]] = []

    for i in range(2, len(frame)):
        if not np.isfinite(average_range[i]) or average_range[i] <= 0:
            continue
        span = high[i] - low[i]
        displaced = span >= displacement_atr * average_range[i]
        if displaced and close[i] > open_[i]:
            for j in range(i - 1, max(-1, i - 12), -1):
                if close[j] < open_[j]:
                    bull_zones.append((i, low[j], high[j]))
                    break
        if displaced and close[i] < open_[i]:
            for j in range(i - 1, max(-1, i - 12), -1):
                if close[j] > open_[j]:
                    bear_zones.append((i, low[j], high[j]))
                    break

        bull_zones = [zone for zone in bull_zones if i - zone[0] <= max_age]
        bear_zones = [zone for zone in bear_zones if i - zone[0] <= max_age]
        if any(bottom <= close[i] <= top for _, bottom, top in bull_zones):
            in_bull[i] = 1.0
        if any(bottom <= close[i] <= top for _, bottom, top in bear_zones):
            in_bear[i] = 1.0

    return pd.DataFrame({"in_bull_ob": in_bull, "in_bear_ob": in_bear}, index=frame.index)


def dealing_range(frame: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Premium/discount location inside the range the last confirmed swings define."""
    swings = confirmed_swings(frame, k)
    top, bottom = swings["swing_high"], swings["swing_low"]
    span = (top - bottom).replace(0.0, np.nan)
    position = (frame["close"] - bottom) / span
    return pd.DataFrame({
        "deal_range_high": top,
        "deal_range_low": bottom,
        "deal_range_position": position,
        "premium": (position > 0.5).astype(float).where(position.notna()),
        "discount": (position < 0.5).astype(float).where(position.notna()),
        "equilibrium": ((position >= 0.45) & (position <= 0.55)).astype(float).where(position.notna()),
        "ote_long": ((position >= 0.618) & (position <= 0.79)).astype(float).where(position.notna()),
        "ote_short": ((position >= 0.21) & (position <= 0.382)).astype(float).where(position.notna()),
    })


def period_extremes(frame: pd.DataFrame, spec: ExchangeSpec) -> pd.DataFrame:
    """Previous day and previous week high/low, in the exchange's local calendar.

    Both are computed from periods that had already closed, so the value stamped
    on a bar never uses that bar's own session.
    """
    local = frame.index.tz_convert(spec.tz)
    local_date = pd.Series(local.date, index=frame.index)
    iso = local.isocalendar()
    week_key = pd.Series(list(zip(iso["year"], iso["week"], strict=True)), index=frame.index)

    daily = frame.groupby(local_date).agg(day_high=("high", "max"), day_low=("low", "min"),
                                          day_close=("close", "last"))
    previous_day = daily.shift(1)
    weekly = frame.groupby(week_key).agg(week_high=("high", "max"), week_low=("low", "min"))
    previous_week = weekly.shift(1)

    return pd.DataFrame({
        "pdh": local_date.map(previous_day["day_high"]).astype(float),
        "pdl": local_date.map(previous_day["day_low"]).astype(float),
        "pdc": local_date.map(previous_day["day_close"]).astype(float),
        "pwh": week_key.map(previous_week["week_high"]).astype(float),
        "pwl": week_key.map(previous_week["week_low"]).astype(float),
    }, index=frame.index)


def displacement(frame: pd.DataFrame, multiple: float = 1.5) -> pd.Series:
    """A candle whose range is at least ``multiple`` ATRs, in the direction of its body."""
    span = frame["high"] - frame["low"]
    average_range = atr(frame, 14)
    strong = span > multiple * average_range
    return pd.Series(np.where(strong, np.sign(frame["close"] - frame["open"]), 0.0),
                     index=frame.index)


def session_features(frame: pd.DataFrame, spec: ExchangeSpec) -> pd.DataFrame:
    """Session bucket, kill-zone flag and time-of-day, all in exchange-local time.

    ``session`` is stored as a number so the feature table stays numeric; the
    mapping is fixed by the order of :class:`~tradingagents.research.contracts.Session`.
    """
    index = frame.index
    sessions = [session_of(ts, spec) for ts in index]
    zones = [kill_zone(ts, spec) for ts in index]
    local = index.tz_convert(spec.tz)
    minutes = pd.Series(local.hour * 60 + local.minute, index=index)
    session_code = {member: float(code) for code, member in enumerate(Session)}
    in_session = pd.Series([float(in_cash_session(ts, spec)) for ts in index], index=index)
    return pd.DataFrame({
        "session_id": pd.Series([session_code[s] for s in sessions], index=index),
        "in_kill_zone": pd.Series([float(zone is not None) for zone in zones], index=index),
        "kill_zone_id": pd.Series([{"asia": 1.0, "london": 2.0, "new_york_am": 3.0,
                                    "new_york_pm": 4.0}.get(zone, 0.0) for zone in zones], index=index),
        "minutes_of_day": minutes.astype(float),
        "in_cash_session": in_session,
        "day_of_week": pd.Series(local.dayofweek, index=index).astype(float),
    })


def smc_features(frame: pd.DataFrame, spec: ExchangeSpec, k: int = 2) -> pd.DataFrame:
    """The full SMC/ICT feature block for one timeframe."""
    index = frame.index
    swings = confirmed_swings(frame, k)
    structure = market_structure(frame, k)
    features = pd.DataFrame(index=index)
    features["swing_high"] = swings["swing_high"]
    features["swing_low"] = swings["swing_low"]
    features["new_swing_high"] = swings["new_swing_high"]
    features["new_swing_low"] = swings["new_swing_low"]
    features["distance_to_swing_high"] = (swings["swing_high"] - frame["close"]) / frame["close"]
    features["distance_to_swing_low"] = (frame["close"] - swings["swing_low"]) / frame["close"]
    features = features.join(structure)
    features = features.join(fair_value_gaps(frame))
    features = features.join(order_blocks(frame))
    features = features.join(dealing_range(frame, k))
    features = features.join(period_extremes(frame, spec))
    features["displacement"] = displacement(frame)
    features["above_pdh"] = (frame["close"] > features["pdh"]).astype(float)
    features["below_pdl"] = (frame["close"] < features["pdl"]).astype(float)
    features["swept_pdh"] = ((frame["high"] > features["pdh"])
                             & (frame["close"] < features["pdh"])).astype(float)
    features["swept_pdl"] = ((frame["low"] < features["pdl"])
                             & (frame["close"] > features["pdl"])).astype(float)
    features = features.join(session_features(frame, spec))
    return features
