"""Market data: normalisation, calendars, quality gates and look-ahead-safe alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.contracts import Session, Timeframe
from tradingagents.research.data import (
    CsvDataSource,
    DataTrustError,
    FrameDataSource,
    MarketData,
    exchange_for,
    in_cash_session,
    kill_zone,
    load_market_data,
    normalize_frame,
    quality_report,
    require_trustworthy,
    resample_frame,
    session_of,
)


@pytest.mark.unit
def test_normalize_frame_drops_impossible_bars_and_sorts():
    index = pd.to_datetime([
        "2026-01-02 10:00", "2026-01-02 09:00", "2026-01-02 09:00", "2026-01-02 12:00",
    ]).tz_localize("UTC")
    frame = pd.DataFrame({
        "Open": [10.0, 11.0, 11.5, 0.0],
        "High": [12.0, 11.4, 12.0, 1.0],
        "Low": [9.0, 10.5, 11.0, 0.5],
        "Close": [11.0, 11.2, 11.8, 0.0],
        "Volume": [100, 200, 250, 300],
    }, index=index)
    normalized = normalize_frame(frame)
    assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
    assert normalized.index.is_monotonic_increasing
    assert len(normalized) == 2  # duplicate kept once, non-positive price dropped
    assert normalized.index.tz is not None


@pytest.mark.unit
def test_normalize_frame_requires_ohlc_columns():
    with pytest.raises(ValueError, match="missing"):
        normalize_frame(pd.DataFrame({"close": [1.0]}, index=pd.date_range(
            "2026-01-01", periods=1, tz="UTC")))


@pytest.mark.unit
def test_resample_to_coarser_timeframe_aggregates_ohlcv():
    bars = synthetic_bars(n=60, timeframe=Timeframe.M5, seed=4)
    hourly = resample_frame(bars, Timeframe.H1)
    assert len(hourly) < len(bars)
    # Nothing is created or lost: volume sums to the same total and each hour's
    # high is the maximum of the five-minute bars it contains.
    assert hourly["volume"].sum() == pytest.approx(bars["volume"].sum())
    bin_key = bars.index.floor("1h")
    for stamp in hourly.index:
        inside = bars[bin_key == stamp]
        if len(inside):
            assert hourly.loc[stamp, "high"] == pytest.approx(inside["high"].max())
            assert hourly.loc[stamp, "low"] == pytest.approx(inside["low"].min())


@pytest.mark.unit
def test_exchange_calendars_follow_suffix_and_asset_type():
    assert exchange_for("RELIANCE.NS").timezone == "Asia/Kolkata"
    assert exchange_for("AAPL").timezone == "America/New_York"
    assert exchange_for("BTC-USD").continuous is True
    assert exchange_for("BTCUSD", asset_type="crypto").continuous is True
    # India does not observe daylight saving, so the offset is stable all year.
    winter = pd.Timestamp("2026-01-15 04:00", tz="UTC")
    summer = pd.Timestamp("2026-07-15 04:00", tz="UTC")
    assert winter.tz_convert("Asia/Kolkata").hour == summer.tz_convert("Asia/Kolkata").hour


@pytest.mark.unit
def test_session_windows_and_kill_zones_are_declared_not_guessed():
    spec = exchange_for("AAPL")
    assert in_cash_session(pd.Timestamp("2026-03-10 15:00", tz="UTC"), spec)
    assert not in_cash_session(pd.Timestamp("2026-03-10 02:00", tz="UTC"), spec)
    # 13:00 UTC during New York summer is 09:00 in New York: the NY AM kill zone.
    assert kill_zone(pd.Timestamp("2026-07-14 13:00", tz="UTC"), spec) == "new_york_am"
    assert session_of(pd.Timestamp("2026-07-14 13:00", tz="UTC"), spec) is Session.OVERLAP
    assert session_of(pd.Timestamp("2026-07-14 02:00", tz="UTC")) is Session.ASIA
    assert session_of(pd.Timestamp("2026-07-14 23:00", tz="UTC")) is Session.OFF_HOURS


@pytest.mark.unit
def test_quality_gate_blocks_thin_stale_and_impossible_series():
    spec = exchange_for("AAPL")
    thin = synthetic_bars(n=50)
    report = quality_report(thin, "SYNTH", Timeframe.D1, spec, min_bars=200)
    assert not report.ok and any("bars" in issue for issue in report.blocking)

    good = synthetic_bars(n=400)
    stale = quality_report(good, "SYNTH", Timeframe.D1, spec, end="2027-01-01")
    assert not stale.ok and any("stale" in issue for issue in stale.blocking)

    broken = good.copy()
    broken.iloc[10, 0] = -5.0
    bad = quality_report(broken, "SYNTH", Timeframe.D1, spec, end=str(good.index[-1].date()))
    assert not bad.ok and any("non-positive" in issue for issue in bad.blocking)

    healthy = quality_report(good, "SYNTH", Timeframe.D1, spec, end=str(good.index[-1].date()))
    assert healthy.ok and healthy.coverage is not None


@pytest.mark.unit
def test_load_market_data_reports_each_failure_instead_of_crashing():
    with pytest.raises(ValueError, match="no usable market data"):
        load_market_data("NOPE", ["1d"], "2026-01-01", "2026-02-01",
                         source=FrameDataSource({}))

    bars = synthetic_bars(n=400)
    data = load_market_data("SYNTH", ["1d"], str(bars.index[0].date()),
                            str(bars.index[-1].date()),
                            source=FrameDataSource({"1d": bars}))
    assert data.ok
    require_trustworthy(data)  # no raise


@pytest.mark.unit
def test_require_trustworthy_raises_with_the_blocking_issues():
    bars = synthetic_bars(n=30)
    data = market_data(bars, min_bars=200)
    with pytest.raises(DataTrustError) as error:
        require_trustworthy(data)
    assert "bars" in str(error.value)


@pytest.mark.unit
def test_context_is_merged_only_after_the_source_bar_closed():
    bars = synthetic_bars(n=400)
    data = market_data(bars)
    primary = data.frame(Timeframe.D1)
    higher = resample_frame(bars, Timeframe.H4)
    data.frames[Timeframe.H4] = higher
    target = primary.index[50:60]
    merged = data.context(Timeframe.H4, target, columns=["close"])
    for stamp in target:
        # The last H4 bar whose close is at or before the daily stamp.
        eligible = higher[higher.index + pd.Timedelta(hours=4) <= stamp]
        if eligible.empty:
            assert pd.isna(merged.loc[stamp, "close"])
        else:
            assert merged.loc[stamp, "close"] == pytest.approx(eligible["close"].iloc[-1])


@pytest.mark.unit
def test_csv_source_reads_one_file_per_timeframe(tmp_path):
    bars = synthetic_bars(n=300)
    bars.to_csv(tmp_path / "SYNTH-1d.csv")
    source = CsvDataSource(tmp_path)
    loaded = source.load("SYNTH", Timeframe.D1, str(bars.index[0].date()),
                         str(bars.index[-1].date()))
    assert len(loaded) == len(bars)
    with pytest.raises(FileNotFoundError):
        source.load("OTHER", Timeframe.D1, "2026-01-01", "2026-02-01")


@pytest.mark.unit
def test_market_data_summary_is_jsonable_and_states_quality():
    bars = synthetic_bars(n=400)
    data = market_data(bars)
    summary = data.summary()
    assert summary["ticker"] == "SYNTH"
    assert summary["timeframes"]["1d"]["bars"] == len(bars)
    assert summary["ok"] is True
    assert isinstance(summary["quality"]["1d"]["blocking"], list)


@pytest.mark.unit
def test_market_data_rejects_a_frame_mismatched_with_features():
    bars = synthetic_bars(n=400)
    data = market_data(bars)
    assert len(data.frame()) == len(bars)
    with pytest.raises(KeyError, match="no 1m frame"):
        data.frame(Timeframe.M1)


@pytest.mark.unit
def test_market_data_constructor_normalizes_index_to_utc():
    bars = synthetic_bars(n=250, tz="America/New_York")
    data = MarketData(ticker="SYNTH", frames={"1d": bars})
    assert str(data.frame().index.tz) == "UTC"


@pytest.mark.unit
def test_the_yahoo_source_delegates_to_the_data_layer(monkeypatch):
    """The vendor call belongs to dataflows; the research adapter only translates."""
    from tradingagents.dataflows.vendors.yahoo import ohlcv as yahoo_ohlcv
    from tradingagents.research.data import Timeframe, YahooDataSource

    calls: list[dict] = []

    def fake(symbol, start, end, *, interval, cache_dir, fill_gaps=True):
        calls.append({"symbol": symbol, "start": start, "end": end, "interval": interval,
                      "cache_dir": cache_dir})
        index = pd.date_range("2026-01-05 09:30", periods=8, freq="1h", tz="UTC")
        return pd.DataFrame({"Date": index.tz_convert(None),
                             "Open": [100.0] * 8, "High": [101.0] * 8,
                             "Low": [99.0] * 8, "Close": [100.5] * 8,
                             "Volume": [1_000.0] * 8})

    monkeypatch.setattr(yahoo_ohlcv, "load_ohlcv_window", fake)
    source = YahooDataSource("/tmp/research-cache")
    frame = source.load("TEST", Timeframe.H4, "2026-01-05", "2026-01-06")

    assert calls == [{"symbol": "TEST", "start": "2026-01-05", "end": "2026-01-06",
                      "interval": "1h", "cache_dir": "/tmp/research-cache"}]
    # 4h bars are resampled from the 1h frame Yahoo actually serves: the eight
    # hourly bars fall into 08:00, 12:00 and 16:00 UTC buckets.
    assert len(frame) == 3
    assert (frame.index[1] - frame.index[0]) == pd.Timedelta(hours=4)
