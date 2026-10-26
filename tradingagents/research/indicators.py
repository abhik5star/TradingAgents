"""Technical indicators as explicit, causal series.

Every function here takes a bar frame and returns values that use only bars up to
and including the current one. There is no ``shift(-1)`` anywhere in this module:
a feature computed at bar *i* may be used to decide at the close of bar *i* and
filled at the open of bar *i+1*, and nothing more.

The implementations are deliberately local rather than delegated to a library,
because the research engine has to state in one place what "ADX" or "Supertrend"
means for the strategies it reports on, and because a library default that
changes between versions would silently invalidate every stored result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bump when a formula changes: stored experiments keep the version they ran under.
INDICATOR_VERSION = "1.0.0"


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    ranges = pd.concat([
        frame["high"] - frame["low"],
        (frame["high"] - previous_close).abs(),
        (frame["low"] - previous_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's average true range."""
    return true_range(frame).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's relative strength index, bounded to 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    values = 100 - (100 / (1 + rs))
    # A flat-loss window is maximum strength, not missing data.
    return values.where(avg_loss.ne(0) | avg_gain.isna(), 100.0)


def stoch_rsi(series: pd.Series, window: int = 14, smooth: int = 3) -> pd.Series:
    """Stochastic RSI, 0-100."""
    values = rsi(series, window)
    lowest = values.rolling(window, min_periods=window).min()
    highest = values.rolling(window, min_periods=window).max()
    spread = (highest - lowest).replace(0.0, np.nan)
    raw = (values - lowest) / spread * 100
    raw = raw.where(spread.notna(), 50.0)
    return raw.rolling(smooth, min_periods=1).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line})


def adx(frame: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's directional movement: +DI, -DI and ADX."""
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    smoothing = lambda s: pd.Series(s, index=frame.index).ewm(  # noqa: E731 - one expression
        alpha=1 / window, adjust=False, min_periods=window).mean()
    tr = smoothing(true_range(frame))
    di_plus = 100 * smoothing(plus_dm) / tr.replace(0.0, np.nan)
    di_minus = 100 * smoothing(minus_dm) / tr.replace(0.0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    return pd.DataFrame({"di_plus": di_plus, "di_minus": di_minus,
                         "adx": dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()})


def supertrend(frame: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend line and direction (+1 up-trend, -1 down-trend).

    Points for which there is not yet a full ATR window carry NaN rather than a
    guessed direction, so a strategy cannot act on a line that never formed.
    """
    average_range = atr(frame, window)
    midpoint = (frame["high"] + frame["low"]) / 2
    upper = midpoint + multiplier * average_range
    lower = midpoint - multiplier * average_range
    close = frame["close"].to_numpy()
    upper_values, lower_values, direction = upper.to_numpy(), lower.to_numpy(), np.zeros(len(frame))
    final_upper = np.full(len(frame), np.nan)
    final_lower = np.full(len(frame), np.nan)
    trend = np.zeros(len(frame))
    started = False
    for i in range(len(frame)):
        if np.isnan(upper_values[i]) or np.isnan(lower_values[i]):
            continue
        if not started:
            trend[i] = 1.0
            final_upper[i], final_lower[i] = upper_values[i], lower_values[i]
            started = True
            continue
        # Bands ratchet: they only tighten while the trend holds.
        final_upper[i] = (upper_values[i] if (upper_values[i] < final_upper[i - 1]
                                              or close[i - 1] > final_upper[i - 1])
                          else final_upper[i - 1])
        final_lower[i] = (lower_values[i] if (lower_values[i] > final_lower[i - 1]
                                              or close[i - 1] < final_lower[i - 1])
                          else final_lower[i - 1])
        if trend[i - 1] == 1.0:
            trend[i] = -1.0 if close[i] < final_lower[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > final_upper[i] else -1.0
    direction[:] = trend
    line = np.where(direction > 0, final_lower, final_upper)
    result = pd.DataFrame({"supertrend": line, "supertrend_dir": direction},
                          index=frame.index)
    # A warm-up bar has no ATR, hence no band: both columns stay empty.
    return result.mask(result["supertrend"].isna())


def bollinger(series: pd.Series, window: int = 20, deviations: float = 2.0) -> pd.DataFrame:
    middle = sma(series, window)
    spread = series.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = middle + deviations * spread, middle - deviations * spread
    width = (upper - lower) / middle.replace(0.0, np.nan)
    position = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": middle, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": width, "bb_pctb": position})


def anchored_vwap(frame: pd.DataFrame, session_keys: pd.Series | None = None) -> pd.Series:
    """Volume-weighted average price, re-anchored each session.

    The anchor matters: a VWAP carried across sessions is not a VWAP anyone
    trades against. ``session_keys`` defaults to the UTC date of each bar.
    """
    keys = session_keys if session_keys is not None else pd.Series(
        frame.index.tz_convert("UTC").date, index=frame.index)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    volume = frame["volume"].fillna(0.0)
    traded = (typical * volume).groupby(keys).cumsum()
    total = volume.groupby(keys).cumsum()
    return traded / total.replace(0.0, np.nan)


def obv(frame: pd.DataFrame) -> pd.Series:
    direction = np.sign(frame["close"].diff().fillna(0.0))
    return (direction * frame["volume"].fillna(0.0)).cumsum()


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Least-squares slope over ``window`` bars, normalised by price level.

    Vectorised through the closed-form covariance form, which is exactly the
    ordinary least-squares slope for evenly spaced points.
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = (x_centered ** 2).sum()

    def slope(values: np.ndarray) -> float:
        return float((x_centered * (values - values.mean())).sum() / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True) / series


def rolling_r2(series: pd.Series, window: int) -> pd.Series:
    """Coefficient of determination of a linear fit over ``window`` bars (0-1)."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denominator = (x_centered ** 2).sum()

    def r2(values: np.ndarray) -> float:
        fitted = values.mean() + x_centered * ((x_centered * (values - values.mean())).sum()
                                               / denominator)
        residual = ((values - fitted) ** 2).sum()
        total = ((values - values.mean()) ** 2).sum()
        return 0.0 if total == 0 else float(1 - residual / total)

    return series.rolling(window, min_periods=window).apply(r2, raw=True)


def volume_profile_proxy(frame: pd.DataFrame, window: int = 50, bins: int = 20) -> pd.DataFrame:
    """Rolling volume-by-price *proxy*, labelled as such wherever it is used.

    Only candle volume is available from OHLCV, so each bar's volume is assigned
    to the price bin containing its typical price. That is a proxy for a volume
    profile, not a traded volume-at-price tape: a reader must not mistake one for
    the other, which is why the returned columns are named ``proxy_*``.
    """
    typical = ((frame["high"] + frame["low"] + frame["close"]) / 3).to_numpy()
    volume = frame["volume"].fillna(0.0).to_numpy()
    poc = np.full(len(frame), np.nan)
    for i in range(window - 1, len(frame)):
        prices = typical[i - window + 1: i + 1]
        volumes = volume[i - window + 1: i + 1]
        if not volumes.sum() or not np.isfinite(prices).all():
            continue
        edges = np.linspace(prices.min(), prices.max() + 1e-12, bins + 1)
        index = np.clip(np.digitize(prices, edges) - 1, 0, bins - 1)
        tally = np.zeros(bins)
        np.add.at(tally, index, volumes)
        centroids = (edges[:-1] + edges[1:]) / 2
        poc[i] = centroids[int(np.argmax(tally))]
    return pd.DataFrame({"proxy_poc": poc}, index=frame.index)


def fib_levels(frame: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Fibonacci retracement levels of the trailing swing, and where price sits.

    The swing is measured over a fixed lookback of bars that have already closed,
    so the levels move only when new bars arrive.
    """
    high = frame["high"].rolling(window, min_periods=window).max()
    low = frame["low"].rolling(window, min_periods=window).min()
    span = (high - low).replace(0.0, np.nan)
    close = frame["close"]
    return pd.DataFrame({
        "fib_lookback_high": high,
        "fib_lookback_low": low,
        "range_position": (close - low) / span,
        "fib_382": high - span * 0.382,
        "fib_500": high - span * 0.500,
        "fib_618": high - span * 0.618,
    })


def indicator_features(frame: pd.DataFrame, session_keys: pd.Series | None = None) -> pd.DataFrame:
    """The standard indicator feature block for one timeframe.

    Column names are the contract the strategy DSL refers to, so they are stable
    and prefixed where a value is a proxy rather than a measurement.
    """
    close, volume = frame["close"], frame["volume"].fillna(0.0)
    features = pd.DataFrame(index=frame.index)

    features["sma_20"] = sma(close, 20)
    features["sma_50"] = sma(close, 50)
    features["sma_200"] = sma(close, 200)
    features["ema_9"] = ema(close, 9)
    features["ema_21"] = ema(close, 21)
    features["ema_50"] = ema(close, 50)
    features["vwap"] = anchored_vwap(frame, session_keys)
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi_k"] = stoch_rsi(close, 14, 3)
    features = features.join(macd(close))
    features["atr_14"] = atr(frame, 14)
    features["atr_pct"] = features["atr_14"] / close
    features = features.join(adx(frame, 14))
    features = features.join(supertrend(frame, 10, 3.0))
    features = features.join(bollinger(close, 20, 2.0))
    features["volume_sma_20"] = sma(volume, 20)
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std(ddof=0)
    features["volume_z"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)
    features["obv_slope_20"] = rolling_slope(obv(frame), 20)
    features["momentum_10"] = close.pct_change(10)
    features["roc_20"] = close.pct_change(20)
    features["trend_slope_20"] = rolling_slope(close, 20)
    features["trend_r2_20"] = rolling_r2(close, 20)

    # Breakout confirmation compares today's close with the *prior* Donchian
    # extreme, so the extreme being broken is one that existed before the bar.
    prior_high = frame["high"].rolling(20, min_periods=20).max().shift(1)
    prior_low = frame["low"].rolling(20, min_periods=20).min().shift(1)
    features["donchian_high_20"] = prior_high
    features["donchian_low_20"] = prior_low
    features["breakout_up_20"] = (close > prior_high).astype(float)
    features["breakout_down_20"] = (close < prior_low).astype(float)

    features["body_ratio"] = (close - frame["open"]).abs() / (frame["high"] - frame["low"]).replace(0.0, np.nan)
    features["close_location"] = ((close - frame["low"])
                                  / (frame["high"] - frame["low"]).replace(0.0, np.nan))
    features["compression"] = (
        features["bb_width"].rolling(20, min_periods=20).rank(pct=True).rsub(1.0))
    features = features.join(fib_levels(frame, 50))
    features["price_above_vwap"] = (close > features["vwap"]).astype(float)
    features["ema_stack_up"] = ((features["ema_9"] > features["ema_21"])
                                & (features["ema_21"] > features["ema_50"])).astype(float)
    features["ema_stack_down"] = ((features["ema_9"] < features["ema_21"])
                                  & (features["ema_21"] < features["ema_50"])).astype(float)
    features["supertrend_up"] = (features["supertrend_dir"] > 0).astype(float)
    features["macd_bull"] = (features["macd_hist"] > 0).astype(float)
    return features


#: The columns a strategy condition may reference without a suffix.
INDICATOR_COLUMNS: tuple[str, ...] = tuple(sorted(indicator_features(
    pd.DataFrame({"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5],
                  "close": [1.5, 2.5], "volume": [10.0, 20.0]},
                 index=pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC"))
).columns))


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Where the latest value sits in its own trailing window (0 = lowest)."""
    return series.rolling(window, min_periods=window).rank(pct=True)
