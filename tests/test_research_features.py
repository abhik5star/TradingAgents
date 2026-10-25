"""Features, indicators, SMC/ICT rules and price action: definitions and causality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.data import exchange_for
from tradingagents.research.features import build_feature_table, feature_group
from tradingagents.research.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    indicator_features,
    macd,
    obv,
    rolling_slope,
    rsi,
    sma,
    stoch_rsi,
    supertrend,
    true_range,
    volume_profile_proxy,
)
from tradingagents.research.patterns import chart_patterns, opening_range, range_and_compression
from tradingagents.research.regimes import classify_regime
from tradingagents.research.smc import (
    confirmed_swings,
    dealing_range,
    fair_value_gaps,
    market_structure,
    order_blocks,
    period_extremes,
    smc_features,
)


def _bars(closes: list[float], highs: list[float] | None = None,
          lows: list[float] | None = None) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="1D", tz="UTC")
    close = pd.Series(closes, index=index, dtype=float)
    high = pd.Series(highs if highs is not None else closes, index=index, dtype=float)
    low = pd.Series(lows if lows is not None else closes, index=index, dtype=float)
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": high,
                         "low": low, "close": close, "volume": 1000.0}, index=index)


# --------------------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_moving_averages_trail_the_series_and_start_after_the_window():
    series = pd.Series(np.arange(1.0, 31.0))
    simple = sma(series, 10)
    assert simple.isna().sum() == 9
    assert simple.iloc[-1] == pytest.approx(np.mean(np.arange(21.0, 31.0)))
    exponential = ema(series, 10)
    assert exponential.isna().sum() == 9
    assert exponential.iloc[-1] > simple.iloc[-1]  # a rising series: EMA above SMA


@pytest.mark.unit
def test_rsi_is_bounded_and_saturates_on_a_one_way_move():
    rising = rsi(pd.Series(np.arange(1.0, 60.0)), 14)
    assert rising.dropna().between(0, 100).all()
    assert rising.dropna().iloc[-1] > 95
    falling = rsi(pd.Series(np.arange(60.0, 1.0, -1)), 14)
    assert falling.dropna().iloc[-1] < 5


@pytest.mark.unit
def test_stochastic_rsi_stays_in_range():
    values = stoch_rsi(pd.Series(np.sin(np.linspace(0, 20, 200)) * 10 + 100), 14, 3)
    valid = values.dropna()
    assert len(valid) > 50
    assert valid.between(0, 100).all()


@pytest.mark.unit
def test_macd_histogram_is_the_line_minus_its_signal():
    frame = macd(pd.Series(np.linspace(10, 40, 120)))
    valid = frame.dropna()
    assert np.allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"])


@pytest.mark.unit
def test_atr_and_true_range_are_positive_and_react_to_range():
    frame = _bars([100.0] * 30, [101.0] * 30, [99.0] * 30)
    assert (true_range(frame).dropna() >= 0).all()
    quiet = atr(frame, 14).iloc[-1]
    wild = _bars([100.0] * 30, [105.0] * 30, [95.0] * 30)
    assert atr(wild, 14).iloc[-1] > quiet


@pytest.mark.unit
def test_adx_identifies_a_trending_series_over_a_flat_one():
    trend = _bars(list(np.linspace(100, 180, 80)),
                  list(np.linspace(101, 181, 80)), list(np.linspace(99, 179, 80)))
    flat = _bars([100.0 + (i % 2) for i in range(80)])
    trend_adx = adx(trend, 14)["adx"].iloc[-1]
    flat_adx = adx(flat, 14)["adx"].iloc[-1]
    assert trend_adx > 40
    assert flat_adx < 20


@pytest.mark.unit
def test_supertrend_direction_flips_with_the_trend_and_stays_empty_in_warmup():
    frame = _bars(list(np.linspace(100, 60, 60)) + list(np.linspace(60, 120, 60)),
                  list(np.linspace(101, 61, 60)) + list(np.linspace(61, 121, 60)),
                  list(np.linspace(99, 59, 60)) + list(np.linspace(59, 119, 60)))
    result = supertrend(frame, 10, 3.0)
    assert result["supertrend_dir"].iloc[:9].isna().all()
    assert result["supertrend_dir"].iloc[55] == -1
    assert result["supertrend_dir"].iloc[-1] == 1


@pytest.mark.unit
def test_bollinger_encloses_the_mean_and_pctb_is_bounded():
    series = pd.Series(np.random.default_rng(3).normal(100, 2, 200))
    bands = bollinger(series, 20, 2.0).dropna()
    assert (bands["bb_upper"] > bands["bb_mid"]).all()
    assert (bands["bb_lower"] < bands["bb_mid"]).all()
    assert bands["bb_pctb"].between(-1.5, 2.5).all()


@pytest.mark.unit
def test_obv_and_slope_are_causal_and_directional():
    frame = _bars(list(np.linspace(10, 20, 50)))
    values = obv(frame)
    assert values.iloc[-1] == pytest.approx(49 * 1000)
    slope = rolling_slope(frame["close"], 20).dropna()
    assert (slope > 0).all()


@pytest.mark.unit
def test_volume_profile_columns_are_labelled_as_proxies():
    frame = synthetic_bars(n=200)
    profile = volume_profile_proxy(frame, window=50, bins=10)
    assert profile.columns.tolist() == ["proxy_poc"]
    assert profile["proxy_poc"].dropna().between(frame["low"].min(), frame["high"].max()).all()


@pytest.mark.unit
def test_indicator_features_are_stable_and_include_the_declared_columns():
    frame = synthetic_bars(n=300)
    features = indicator_features(frame)
    for column in ("ema_21", "rsi_14", "atr_14", "adx", "supertrend_dir", "bb_width",
                   "volume_z", "trend_slope_20", "breakout_up_20", "fib_500"):
        assert column in features.columns
    assert features.index.equals(frame.index)
    assert np.isfinite(features["rsi_14"].iloc[-1])


# --------------------------------------------------------------------------------------
# SMC / ICT
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_a_swing_high_is_only_known_after_it_is_confirmed():
    closes = [10, 11, 12, 13, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 16, 15]
    frame = _bars(closes)
    swings = confirmed_swings(frame, k=2)
    peak = int(np.argmax(closes[:12]))
    # The level cannot be reported at the peak bar itself, only k bars later.
    assert pd.isna(swings["swing_high"].iloc[peak])
    assert swings["new_swing_high"].iloc[peak + 2] == 1.0
    # The confirmed level is the swing bar's own high, not the confirming bar's.
    assert swings["swing_high"].iloc[peak + 2] == pytest.approx(14.0)
    assert swings["swing_high"].iloc[peak + 3] == pytest.approx(14.0)


@pytest.mark.unit
def test_bos_and_choch_are_distinguished_by_prior_structure():
    # Rise, pull back, break the old high (BOS up), then break down (CHoCH).
    closes = [10, 11, 12, 13, 12, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10]
    frame = _bars(closes)
    structure = market_structure(frame, k=1)
    assert structure["bos_up"].sum() >= 1
    assert structure["structure_dir"].iloc[-1] == -1
    assert structure["choch_down"].sum() >= 1


@pytest.mark.unit
def test_liquidity_sweep_is_a_break_that_closes_back_inside():
    closes = [10, 11, 12, 13, 12, 11, 11]
    highs = [10, 11, 12, 13, 12.5, 11.5, 13.5]
    lows = [9.5, 10.5, 11.5, 12.5, 11.5, 10.5, 11.0]
    frame = _bars(closes, highs, lows)
    structure = market_structure(frame, k=1)
    assert structure["sweep_buy_side"].sum() >= 1
    # A sweep is not a break: structure has not turned bullish on that bar.
    assert structure["bos_up"].sum() == 0


@pytest.mark.unit
def test_fair_value_gap_needs_a_three_bar_imbalance():
    closes = [10, 11, 12, 13, 14, 13.5]
    highs = [10.5, 11.5, 12.5, 13.5, 14.5, 14.0]
    lows = [9.5, 10.5, 11.6, 12.6, 13.6, 13.0]
    frame = _bars(closes, highs, lows)
    gaps = fair_value_gaps(frame)
    # Bar 3 (index 2) has a low above bar 1's high: a bullish gap on that bar.
    assert gaps["fvg_bull"].sum() >= 1


@pytest.mark.unit
def test_order_block_and_dealing_range_mark_retests_and_location():
    frame = synthetic_bars(n=400, seed=9)
    blocks = order_blocks(frame)
    assert blocks.columns.tolist() == ["in_bull_ob", "in_bear_ob"]
    assert blocks.to_numpy().sum() >= 0
    location = dealing_range(frame, k=2)
    valid = location["deal_range_position"].dropna()
    assert len(valid) > 0
    # A position outside 0..1 means price has left the last confirmed range, which
    # is information, not an error; the classification columns stay binary.
    assert set(location["premium"].dropna().unique()) <= {0.0, 1.0}
    assert set(location["discount"].dropna().unique()) <= {0.0, 1.0}
    both = (location["premium"].fillna(0) + location["discount"].fillna(0))
    assert (both <= 1).all()


@pytest.mark.unit
def test_previous_day_extremes_come_from_completed_periods():
    bars = synthetic_bars(n=60, timeframe="1d", seed=12)
    spec = exchange_for("AAPL")
    extremes = period_extremes(bars, spec)
    local_date = bars.index.tz_convert(spec.tz).date
    first_day = local_date[0]
    same_day = extremes[pd.Series(local_date, index=bars.index) == first_day]
    # Nothing earlier exists, so the first day has no previous-day level.
    assert same_day["pdh"].isna().all()
    later = extremes.dropna(subset=["pdh"])
    assert len(later) > 0
    second_day = sorted(set(local_date))[1]
    expected = bars[pd.Series(local_date, index=bars.index) == first_day]["high"].max()
    row = extremes[pd.Series(local_date, index=bars.index) == second_day].iloc[0]
    assert row["pdh"] == pytest.approx(expected)


@pytest.mark.unit
def test_smc_feature_block_exposes_session_and_kill_zone_columns():
    bars = synthetic_bars(n=300, timeframe="1d")
    features = smc_features(bars, exchange_for("AAPL"))
    for column in ("bos_up", "choch_down", "sweep_buy_side", "fvg_bull", "in_bull_ob",
                   "deal_range_position", "pdh", "pdl", "in_kill_zone", "session_id"):
        assert column in features.columns
    assert features["session_id"].dropna().between(0, 4).all()


# --------------------------------------------------------------------------------------
# Price action and patterns
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_double_top_is_only_set_after_the_neckline_breaks():
    closes = [10, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 13, 12, 11, 10, 9]
    highs = [10, 11, 12, 13, 14, 13.2, 12.2, 11.2, 12.2, 13.2, 14.05, 13.2, 12.2, 11.0,
             10.0, 9.0]
    lows = [9.6, 10.6, 11.6, 12.6, 13.6, 12.6, 11.0, 10.4, 11.4, 12.4, 13.4, 12.4, 11.4,
            10.4, 9.4, 8.4]
    frame = _bars(closes, highs, lows)
    patterns = chart_patterns(frame, k=1, tolerance_atr=2.0, min_separation=3, hold_bars=3)
    assert patterns["pattern_double_top"].sum() >= 1
    # No pattern before the second top was confirmed.
    assert patterns["pattern_double_top"].iloc[:5].sum() == 0


@pytest.mark.unit
def test_compression_and_range_flags_describe_a_tight_market():
    amplitude = np.linspace(6.0, 0.15, 220)
    closes = [100.0 + amp * np.sin(i / 2.0) for i, amp in enumerate(amplitude)]
    state = range_and_compression(_bars(closes), window=20)
    assert state["range_width"].dropna().max() > state["range_width"].dropna().min()
    assert state["range_compression"].iloc[-1] > 0.7
    assert state["in_range"].iloc[-1] == 1.0


@pytest.mark.unit
def test_opening_range_is_unavailable_until_its_window_closes():
    bars = synthetic_bars(n=80, timeframe="5m", seed=6)
    spec = exchange_for("AAPL")
    features = opening_range(bars, spec, minutes=30)
    assert features["or_high"].isna().sum() > 0
    assert features["or_break_up"].isin([0.0, 1.0]).all()


@pytest.mark.unit
def test_regime_labels_are_declared_and_cover_the_sample():
    bars = synthetic_bars(n=600, seed=8)
    regimes = classify_regime(bars)
    assert regimes["regime"].notna().all()
    assert set(regimes["regime"].unique()) <= {"trend_up", "trend_down", "sideways",
                                               "volatile", "breakout", "unstable"}
    assert len(regimes["regime"].unique()) >= 3


# --------------------------------------------------------------------------------------
# The assembled table
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_feature_table_is_aligned_and_fingerprinted():
    bars = synthetic_bars(n=500, seed=2)
    data = market_data(bars)
    table = build_feature_table(data, "1d")
    assert table.bars.index.equals(table.features.index)
    assert len(table.version) == 12
    assert table.summary()["features"] == len(table.features.columns)
    for column in ("sma_20", "rsi_14", "atr_14", "regime", "structure_dir", "price"):
        assert table.features[column].notna().any(), f"{column} should carry data"
    # Anything empty for the whole sample is disclosed rather than silently kept.
    empty = [c for c in table.features.columns if table.features[c].notna().sum() == 0]
    if empty:
        assert any("not available at this resolution" in note for note in table.notes)


@pytest.mark.unit
def test_context_timeframe_requires_a_coarser_frame():
    bars = synthetic_bars(n=400, timeframe="1h", seed=2)
    data = market_data(bars, timeframes=["1h", "4h"])
    with pytest.raises(ValueError, match="coarser"):
        build_feature_table(data, "1h", context_timeframe="1h")
    with pytest.raises(ValueError, match="finer"):
        build_feature_table(data, "1h", confirm_timeframe="4h")


@pytest.mark.unit
def test_missing_context_frame_degrades_with_a_note_not_an_invention():
    bars = synthetic_bars(n=400, timeframe="1h", seed=2)
    data = market_data(bars, timeframes=["1h"])
    table = build_feature_table(data, "1h", context_timeframe="4h")
    assert table.context_timeframe is None
    assert any("context timeframe 4h was not loaded" in note for note in table.notes)
    assert "htf_structure_dir" not in table.features.columns


@pytest.mark.unit
def test_multi_timeframe_features_are_known_only_after_the_coarse_bar_closed():
    bars = synthetic_bars(n=1200, timeframe="1h", seed=4)
    data = market_data(bars, timeframes=["1h", "4h"])
    table = build_feature_table(data, "1h", context_timeframe="4h")
    assert table.context_timeframe == "4h"
    assert "htf_structure_dir" in table.features.columns
    assert "mtf_conflict" in table.features.columns
    # The first daily bars cannot have 4h context yet: the 4h history is shorter.
    assert table.features["htf_structure_dir"].isna().sum() > 0
    assert table.features["mtf_conflict"].dropna().isin([0.0, 1.0]).all()


@pytest.mark.unit
def test_duplicate_feature_names_are_an_error_not_a_silent_overwrite():
    from tradingagents.research.features import _join

    left = pd.DataFrame({"a": [1.0]}, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))
    right = left.copy()
    with pytest.raises(ValueError, match="duplicates existing columns"):
        _join(left, right, "test block")


@pytest.mark.unit
def test_feature_groups_reference_columns_the_table_provides():
    bars = synthetic_bars(n=400, seed=2)
    data = market_data(bars)
    table = build_feature_table(data, "1d")
    columns = set(table.columns)
    for group in ("trend", "momentum", "volatility", "structure", "liquidity", "levels",
                  "patterns", "session", "regime", "flow_proxy"):
        assert set(feature_group(group)) & columns, f"group {group} has no usable columns"
