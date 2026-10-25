"""Price action, levels, supply/demand and objectively defined chart patterns.

Every pattern here has a written definition, a confirmation bar, and a decay
window: a pattern is either live at bar *i* or it is not. Nothing is detected
from bars that had not printed yet, and no pattern is described as bullish or
bearish without the break that confirms it — a double top that has not broken
its neckline is just two highs.

Patterns are features, not conclusions. Whether any of them is worth trading is
decided by :mod:`tradingagents.research.validation` on out-of-sample data after
costs, not by the label attached here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.research.contracts import Session
from tradingagents.research.data import ExchangeSpec
from tradingagents.research.indicators import atr
from tradingagents.research.smc import confirmed_swings


def _active(signal: np.ndarray, hold_bars: int) -> np.ndarray:
    """Spread a confirmed signal forward ``hold_bars`` bars (never backwards)."""
    if hold_bars <= 1:
        return signal
    active = np.zeros_like(signal, dtype=float)
    for i in np.flatnonzero(signal):
        active[i:i + hold_bars] = 1.0
    return active


def support_resistance(frame: pd.DataFrame, k: int = 2, tolerance_atr: float = 0.5,
                       hold_bars: int = 10) -> pd.DataFrame:
    """Nearest support and resistance from confirmed swings, plus a zone test.

    Levels are the most recent confirmed swing highs and lows. A bar is *at*
    support when it trades within ``tolerance_atr`` ATRs of the nearest confirmed
    low below it (and symmetrically for resistance).
    """
    swings = confirmed_swings(frame, k)
    average_range = atr(frame, 14)
    tolerance = average_range * tolerance_atr
    resistance = swings["swing_high"].to_numpy()
    support = swings["swing_low"].to_numpy()
    close = frame["close"].to_numpy()
    tol = tolerance.to_numpy()

    at_resistance = np.zeros(len(frame))
    at_support = np.zeros(len(frame))
    for i in range(len(frame)):
        if np.isfinite(resistance[i]) and np.isfinite(tol[i]):
            at_resistance[i] = float(abs(close[i] - resistance[i]) <= tol[i])
        if np.isfinite(support[i]) and np.isfinite(tol[i]):
            at_support[i] = float(abs(close[i] - support[i]) <= tol[i])

    features = pd.DataFrame(index=frame.index)
    features["resistance"] = resistance
    features["support"] = support
    features["at_resistance"] = _active(at_resistance, hold_bars)
    features["at_support"] = _active(at_support, hold_bars)
    features["room_to_resistance"] = (resistance - close) / close
    features["room_to_support"] = (close - support) / close
    return features


def supply_demand(frame: pd.DataFrame, base_bars: int = 3, impulse_atr: float = 1.5,
                  max_age: int = 60) -> pd.DataFrame:
    """Base-plus-impulse demand and supply zones, and whether price is inside one.

    A demand zone is the range of the last ``base_bars`` bars of small range
    before a candle that closes up by more than ``impulse_atr`` ATRs. Supply is
    the mirror. Zones expire after ``max_age`` bars.
    """
    open_, high, low, close = (frame[c].to_numpy() for c in ("open", "high", "low", "close"))
    average_range = atr(frame, 14).to_numpy()
    demand: list[tuple[int, float, float]] = []
    supply: list[tuple[int, float, float]] = []
    in_demand = np.zeros(len(frame))
    in_supply = np.zeros(len(frame))

    for i in range(base_bars, len(frame)):
        if not np.isfinite(average_range[i]):
            continue
        base_slice = slice(i - base_bars, i)
        base_high = high[base_slice].max()
        base_low = low[base_slice].min()
        base_span = base_high - base_low
        displacement_up = (close[i] - open_[i]) > impulse_atr * average_range[i]
        displacement_down = (open_[i] - close[i]) > impulse_atr * average_range[i]
        if displacement_up and base_span <= average_range[i]:
            demand.append((i, base_low, base_high))
        if displacement_down and base_span <= average_range[i]:
            supply.append((i, base_low, base_high))
        demand = [zone for zone in demand if i - zone[0] <= max_age]
        supply = [zone for zone in supply if i - zone[0] <= max_age]
        if any(bottom <= close[i] <= top for _, bottom, top in demand):
            in_demand[i] = 1.0
        if any(bottom <= close[i] <= top for _, bottom, top in supply):
            in_supply[i] = 1.0

    # Order-block retests come from :mod:`tradingagents.research.smc`; keeping them
    # out of this function means one rule has exactly one column.
    return pd.DataFrame({"in_demand_zone": in_demand, "in_supply_zone": in_supply},
                        index=frame.index)


def _swing_points(frame: pd.DataFrame, k: int, kind: str) -> list[tuple[int, float]]:
    """Confirmed swing points as ``(confirmation_index, price)`` pairs."""
    swings = confirmed_swings(frame, k)
    column = "swing_high" if kind == "high" else "swing_low"
    flag = "new_swing_high" if kind == "high" else "new_swing_low"
    points: list[tuple[int, float]] = []
    values = swings[column].to_numpy()
    fresh = swings[flag].to_numpy()
    for i in range(len(frame)):
        if fresh[i] > 0 and np.isfinite(values[i]):
            points.append((i, float(values[i])))
    return points


def chart_patterns(frame: pd.DataFrame, k: int = 2, tolerance_atr: float = 0.4,
                   min_separation: int = 5, hold_bars: int = 10) -> pd.DataFrame:
    """Double top/bottom, head-and-shoulders, flags and triangles.

    Each pattern is only set on the bar whose close completes it:

    * **double top** — the last two confirmed swing highs agree within
      ``tolerance_atr`` ATRs and the close breaks below the lowest low between
      them.
    * **double bottom** — the mirror, breaking above the highest high between.
    * **head and shoulders** — three consecutive swing highs with the middle one
      highest, the two shoulders agreeing within tolerance, and a close below the
      lower of the two intervening lows (inverse for the bottom).
    * **bull/bear flag** — a displacement candle followed by a counter-trend
      drift of at least ``min_separation`` bars that stays inside the
      displacement's range, then a close back through the flag extreme.
    * **triangle** — the last three swing highs fall while the last three swing
      lows rise (converging), measured on normalised slopes.
    """
    open_, high, low, close = (frame[c].to_numpy() for c in ("open", "high", "low", "close"))
    average_range = atr(frame, 14).to_numpy()
    swing_highs = _swing_points(frame, k, "high")
    swing_lows = _swing_points(frame, k, "low")
    high_positions = dict(swing_highs)
    low_positions = dict(swing_lows)

    double_top = np.zeros(len(frame))
    double_bottom = np.zeros(len(frame))
    hs_top = np.zeros(len(frame))
    hs_bottom = np.zeros(len(frame))
    bull_flag = np.zeros(len(frame))
    bear_flag = np.zeros(len(frame))
    triangle = np.zeros(len(frame))

    highs_seen: list[tuple[int, float]] = []
    lows_seen: list[tuple[int, float]] = []
    for i in range(len(frame)):
        if i in high_positions:
            highs_seen.append((i, high_positions[i]))
        if i in low_positions:
            lows_seen.append((i, low_positions[i]))
        tolerance = tolerance_atr * average_range[i] if np.isfinite(average_range[i]) else np.nan

        if len(highs_seen) >= 2 and np.isfinite(tolerance):
            first, second = highs_seen[-2], highs_seen[-1]
            between = low[first[0]: second[0] + 1]
            if (second[0] - first[0] >= min_separation
                    and abs(first[1] - second[1]) <= tolerance
                    and len(between) and close[i] < between.min()):
                double_top[i] = 1.0
        if len(lows_seen) >= 2 and np.isfinite(tolerance):
            first, second = lows_seen[-2], lows_seen[-1]
            between = high[first[0]: second[0] + 1]
            if (second[0] - first[0] >= min_separation
                    and abs(first[1] - second[1]) <= tolerance
                    and len(between) and close[i] > between.max()):
                double_bottom[i] = 1.0

        if len(highs_seen) >= 3 and np.isfinite(tolerance):
            left, head, right = highs_seen[-3], highs_seen[-2], highs_seen[-1]
            if (head[1] > left[1] and head[1] > right[1]
                    and abs(left[1] - right[1]) <= tolerance
                    and right[0] > head[0] > left[0]):
                neckline = min(low[left[0]: head[0] + 1].min(), low[head[0]: right[0] + 1].min())
                if close[i] < neckline:
                    hs_top[i] = 1.0
        if len(lows_seen) >= 3 and np.isfinite(tolerance):
            left, head, right = lows_seen[-3], lows_seen[-2], lows_seen[-1]
            if (head[1] < left[1] and head[1] < right[1]
                    and abs(left[1] - right[1]) <= tolerance
                    and right[0] > head[0] > left[0]):
                neckline = max(high[left[0]: head[0] + 1].max(), high[head[0]: right[0] + 1].max())
                if close[i] > neckline:
                    hs_bottom[i] = 1.0

        if i >= min_separation + 1 and np.isfinite(average_range[i]):
            # A flag: displacement bar, then a quiet pullback that holds inside
            # that bar's range, then a close back out of the pullback's extreme.
            for j in range(i - min_separation, i):
                if not np.isfinite(average_range[j]) or j + 1 >= i:
                    continue
                span = high[j] - low[j]
                if span < 1.5 * average_range[j]:
                    continue
                pullback_high = high[j + 1: i + 1].max()
                pullback_low = low[j + 1: i + 1].min()
                if close[j] > open_[j]:
                    if pullback_low >= low[j] and close[i] > pullback_high:
                        bull_flag[i] = 1.0
                        break
                else:
                    if pullback_high <= high[j] and close[i] < pullback_low:
                        bear_flag[i] = 1.0
                        break

        if len(highs_seen) >= 3 and len(lows_seen) >= 3:
            high_slope = _normalised_slope(highs_seen[-3:], close[i])
            low_slope = _normalised_slope(lows_seen[-3:], close[i])
            if (np.isfinite(high_slope) and np.isfinite(low_slope)
                    and high_slope < 0 < low_slope
                    and abs(high_slope + low_slope)
                    < 0.5 * max(abs(high_slope), abs(low_slope), 1e-9)):
                triangle[i] = 1.0

    features = pd.DataFrame(index=frame.index)
    for name, series in (("pattern_double_top", double_top),
                         ("pattern_double_bottom", double_bottom),
                         ("pattern_hs_top", hs_top),
                         ("pattern_hs_bottom", hs_bottom),
                         ("pattern_bull_flag", bull_flag),
                         ("pattern_bear_flag", bear_flag),
                         ("pattern_triangle", triangle)):
        features[name] = _active(series, hold_bars)
    features["pattern_any_bearish"] = features[[
        "pattern_double_top", "pattern_hs_top", "pattern_bear_flag"]].max(axis=1)
    features["pattern_any_bullish"] = features[[
        "pattern_double_bottom", "pattern_hs_bottom", "pattern_bull_flag"]].max(axis=1)
    return features


def _normalised_slope(points: list[tuple[int, float]], reference: float) -> float:
    """Least-squares slope per bar, divided by price so instruments are comparable."""
    if len(points) < 2 or not reference:
        return np.nan
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    if np.ptp(x) == 0:
        return np.nan
    slope = np.polyfit(x - x.mean(), y, 1)[0]
    return float(slope / reference)


def range_and_compression(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Range boundaries, range state and price compression.

    ``range_width`` is the trailing high-low span as a fraction of price, and
    ``range_compression`` is its percentile over the last 100 windows, so 0.1
    means "the tightest 10% of ranges this instrument has had recently".
    """
    high = frame["high"].rolling(window, min_periods=window).max()
    low = frame["low"].rolling(window, min_periods=window).min()
    width = (high - low) / frame["close"]
    features = pd.DataFrame(index=frame.index)
    features["range_high"] = high
    features["range_low"] = low
    features["range_width"] = width
    features["range_compression"] = width.rolling(100, min_periods=20).rank(pct=True).rsub(1.0)
    features["in_range"] = (features["range_compression"] > 0.7).astype(float)
    return features


def gap_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Session gaps and whether they were filled.

    A gap up is an open above the previous bar's high; it is "filled" once a later
    bar trades back to that high. Only bars available at the time are consulted.
    """
    previous_high = frame["high"].shift(1)
    previous_low = frame["low"].shift(1)
    gap_up = (frame["open"] > previous_high).fillna(False)
    gap_down = (frame["open"] < previous_low).fillna(False)
    reference = np.where(gap_up, previous_high, np.where(gap_down, previous_low, np.nan))
    filled = np.zeros(len(frame))
    open_gap = np.nan
    gap_reference = np.nan
    for i in range(len(frame)):
        if gap_up.iloc[i] or gap_down.iloc[i]:
            open_gap = 1.0 if gap_up.iloc[i] else -1.0
            gap_reference = reference[i]
        if (np.isfinite(open_gap) and np.isfinite(gap_reference)
                and ((open_gap > 0 and frame["low"].iloc[i] <= gap_reference)
                     or (open_gap < 0 and frame["high"].iloc[i] >= gap_reference))):
            filled[i] = 1.0
            open_gap = np.nan
    features = pd.DataFrame({
        "gap_up": gap_up.astype(float).to_numpy(),
        "gap_down": gap_down.astype(float).to_numpy(),
        "gap_filled": filled,
        "gap_pct": (frame["open"] - frame["close"].shift(1)) / frame["close"].shift(1),
    }, index=frame.index)
    return features


def opening_range(frame: pd.DataFrame, spec: ExchangeSpec, minutes: int = 30) -> pd.DataFrame:
    """Opening range, its breakout, and the time since the session opened.

    The range is built from the first ``minutes`` of each cash session only, and
    it is not available for breakout use until that window has closed.
    """
    local = frame.index.tz_convert(spec.tz)
    if spec.continuous:
        session_key = pd.Series(local.date, index=frame.index)
        open_minute = 0
    else:
        session_key = pd.Series(local.date, index=frame.index)
        open_minute = min(int(s.partition(":")[0]) * 60 + int(s.partition(":")[2]) for s, _ in spec.sessions)
    minutes_of_day = pd.Series(local.hour * 60 + local.minute, index=frame.index)
    in_window = (minutes_of_day >= open_minute) & (minutes_of_day < open_minute + minutes)
    within = frame[in_window]
    or_high = within.groupby(session_key[in_window])["high"].max()
    or_low = within.groupby(session_key[in_window])["low"].min()

    high_map = session_key.map(or_high).astype(float)
    low_map = session_key.map(or_low).astype(float)
    minutes_since_open = (minutes_of_day - open_minute).astype(float)
    window_closed = minutes_since_open >= minutes
    features = pd.DataFrame({
        "or_high": high_map.where(window_closed),
        "or_low": low_map.where(window_closed),
        "minutes_since_open": minutes_since_open,
        "or_break_up": ((frame["close"] > high_map) & window_closed).astype(float),
        "or_break_down": ((frame["close"] < low_map) & window_closed).astype(float),
    }, index=frame.index)
    features["or_break_up"] = _active(features["or_break_up"].to_numpy(), 5)
    features["or_break_down"] = _active(features["or_break_down"].to_numpy(), 5)
    return features


def breakout_quality(frame: pd.DataFrame, window: int = 20, hold_bars: int = 5) -> pd.DataFrame:
    """Breakouts, false breakouts and retests of the level that broke.

    A false breakout is a bar that trades beyond the prior Donchian extreme and
    closes back inside it. A retest is a later touch of the broken level that
    holds, tracked for ``hold_bars`` bars.
    """
    prior_high = frame["high"].rolling(window, min_periods=window).max().shift(1)
    prior_low = frame["low"].rolling(window, min_periods=window).min().shift(1)
    close, high, low = frame["close"], frame["high"], frame["low"]
    breakout_up = close > prior_high
    breakout_down = close < prior_low
    false_up = (high > prior_high) & (close < prior_high)
    false_down = (low < prior_low) & (close > prior_low)

    retest_up = np.zeros(len(frame))
    retest_down = np.zeros(len(frame))
    level, direction, age = np.nan, 0, 0
    for i in range(len(frame)):
        if breakout_up.iloc[i]:
            level, direction, age = float(prior_high.iloc[i]), 1, 0
        elif breakout_down.iloc[i]:
            level, direction, age = float(prior_low.iloc[i]), -1, 0
        if direction and np.isfinite(level):
            age += 1
            if age > window:
                direction = 0
                continue
            if direction > 0 and low.iloc[i] <= level <= close.iloc[i]:
                retest_up[i] = 1.0
            elif direction < 0 and close.iloc[i] <= level <= high.iloc[i]:
                retest_down[i] = 1.0

    return pd.DataFrame({
        "breakout_confirmed_up": breakout_up.astype(float).to_numpy(),
        "breakout_confirmed_down": breakout_down.astype(float).to_numpy(),
        "false_breakout_up": false_up.astype(float).to_numpy(),
        "false_breakout_down": false_down.astype(float).to_numpy(),
        "retest_up": _active(retest_up, hold_bars),
        "retest_down": _active(retest_down, hold_bars),
        "prior_high_20": prior_high,
        "prior_low_20": prior_low,
    }, index=frame.index)


def session_extremes(frame: pd.DataFrame, spec: ExchangeSpec) -> pd.DataFrame:
    """Running session high/low and the close's position inside it."""
    local = frame.index.tz_convert(spec.tz)
    key = pd.Series(local.date, index=frame.index)
    running_high = frame.groupby(key)["high"].cummax()
    running_low = frame.groupby(key)["low"].cummin()
    span = (running_high - running_low).replace(0.0, np.nan)
    return pd.DataFrame({
        "session_high": running_high,
        "session_low": running_low,
        "session_position": (frame["close"] - running_low) / span,
    }, index=frame.index)


def pattern_features(frame: pd.DataFrame, spec: ExchangeSpec) -> pd.DataFrame:
    """The full price-action feature block for one timeframe."""
    features = pd.DataFrame(index=frame.index)
    features = features.join(support_resistance(frame))
    features = features.join(supply_demand(frame))
    features = features.join(chart_patterns(frame))
    features = features.join(range_and_compression(frame))
    features = features.join(gap_features(frame))
    minutes = 30 if spec.continuous else 30
    features = features.join(opening_range(frame, spec, minutes=minutes))
    features = features.join(breakout_quality(frame))
    features = features.join(session_extremes(frame, spec))
    return features


#: Session labels by numeric id, for reports that print a session name.
SESSION_NAMES = {float(i): member.value for i, member in enumerate(Session)}
