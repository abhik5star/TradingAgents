"""Market data: loading, resampling, exchange calendars and quality gating.

Three jobs, in this order:

1. **Load** bars per timeframe from a declared source (Yahoo, a CSV file, or any
   object implementing :class:`MarketDataSource`). Loaded frames are normalised:
   UTC index, lowercase ``open/high/low/close/volume``, sorted, de-duplicated.
2. **Align** timeframes so only information that had actually printed at the
   decision timestamp is ever visible. A higher-timeframe bar is usable only
   after its own close; a lower-timeframe bar only if it closed by then.
3. **Judge** the data. A series with impossible prices, missing bars or a stale
   tail is reported as blocking, and the run refuses to trade it. Vendor
   breakage must never be scored as a strategy loss.

Intraday timeframes need an intraday source. :class:`YahooDataSource` requests
the right interval and refuses to build an intraday series out of daily bars,
because a "5m" frame resampled from daily closes is a fabrication, not data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.research.contracts import DataQualityReport, Session, Timeframe

logger = logging.getLogger(__name__)

OHLCV = ("open", "high", "low", "close", "volume")
UTC = ZoneInfo("UTC")


# --------------------------------------------------------------------------------------
# Exchange calendar: time zone, cash hours, kill zones
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangeSpec:
    """When an instrument actually trades, in its own time zone.

    ``sessions`` are local-time ``(start, end)`` HH:MM pairs. An empty tuple
    means continuous trading (crypto). Daylight saving is handled by the zone
    database through ``zoneinfo``, so London and New York hours stay correct
    across the changeovers and India never shifts.
    """

    timezone: str
    sessions: tuple[tuple[str, str], ...] = (("09:30", "16:00"),)
    name: str = "us_equities"

    @property
    def continuous(self) -> bool:
        return not self.sessions

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


# Suffix → venue. Ordered longest-suffix-first by the lookup below.
_EXCHANGES: tuple[tuple[str, ExchangeSpec], ...] = (
    (".NS", ExchangeSpec("Asia/Kolkata", (("09:15", "15:30"),), "nse_india")),
    (".BO", ExchangeSpec("Asia/Kolkata", (("09:15", "15:30"),), "bse_india")),
    (".T", ExchangeSpec("Asia/Tokyo", (("09:00", "15:00"),), "tokyo")),
    (".HK", ExchangeSpec("Asia/Hong_Kong", (("09:30", "16:00"),), "hong_kong")),
    (".L", ExchangeSpec("Europe/London", (("08:00", "16:30"),), "london")),
    (".DE", ExchangeSpec("Europe/Berlin", (("09:00", "17:30"),), "xetra")),
    (".PA", ExchangeSpec("Europe/Paris", (("09:00", "17:30"),), "paris")),
    (".AS", ExchangeSpec("Europe/Amsterdam", (("09:00", "17:30"),), "amsterdam")),
    (".SW", ExchangeSpec("Europe/Zurich", (("09:00", "17:30"),), "zurich")),
    (".SI", ExchangeSpec("Asia/Singapore", (("09:00", "17:00"),), "sgx")),
    (".AX", ExchangeSpec("Australia/Sydney", (("10:00", "16:00"),), "asx")),
    (".TO", ExchangeSpec("America/Toronto", (("09:30", "16:00"),), "tsx")),
    (".SA", ExchangeSpec("America/Sao_Paulo", (("10:00", "17:00"),), "b3")),
    (".KS", ExchangeSpec("Asia/Seoul", (("09:00", "15:30"),), "krx")),
    (".TW", ExchangeSpec("Asia/Taipei", (("09:00", "13:30"),), "twse")),
)

_CRYPTO_SUFFIXES = ("-USD", "-USDT", "-EUR", "-BTC", "-ETH", "=F")
_CRYPTO_EXCHANGE = ExchangeSpec("UTC", (), "crypto_24_7")


def exchange_for(ticker: str, asset_type: str = "stock") -> ExchangeSpec:
    """The calendar for ``ticker``: crypto is continuous, futures nearly so, else by suffix."""
    symbol = normalize_symbol(ticker).upper()
    if asset_type == "crypto" or symbol.endswith(_CRYPTO_SUFFIXES):
        return _CRYPTO_EXCHANGE
    for suffix, spec in _EXCHANGES:
        if symbol.endswith(suffix):
            return spec
    return ExchangeSpec("America/New_York", (("09:30", "16:00"),), "us_equities")


def _minutes(hhmm: str) -> int:
    hour, _, minute = hhmm.partition(":")
    return int(hour) * 60 + int(minute)


def session_of(ts: pd.Timestamp, spec: ExchangeSpec | None = None) -> Session:
    """The liquidity session of a UTC timestamp.

    The buckets are UTC liquidity windows — Asia 00:00-07:00, London 07:00-12:00,
    the London/New York overlap 12:00-16:00 (the deepest liquidity of the day),
    New York alone 16:00-21:00, then off-hours. Defining them in UTC rather than in
    each venue's local clock keeps the label comparable across instruments, and
    daylight saving is handled once, by the conversion from the bar's UTC stamp.

    ``spec`` is accepted for symmetry with :func:`kill_zone` and to let a caller
    pass the instrument's calendar without branching; the mapping itself does not
    depend on it.
    """
    del spec  # the window definition is deliberately instrument-independent
    stamp = ts if ts.tzinfo else pd.Timestamp(ts).tz_localize("UTC")
    minute = stamp.tz_convert("UTC").hour * 60 + stamp.tz_convert("UTC").minute
    if minute < 7 * 60:
        return Session.ASIA
    if minute < 12 * 60:
        return Session.LONDON
    if minute < 16 * 60:
        return Session.OVERLAP
    if minute < 21 * 60:
        return Session.NEW_YORK
    return Session.OFF_HOURS


# ICT kill zones, expressed in each venue's own local time. They are declared
# windows, not opinions: a setup outside them is simply not a kill-zone setup.
_KILL_ZONES: tuple[tuple[str, str, int, int], ...] = (
    ("asia", "Asia/Tokyo", 0, 180),
    ("london", "Europe/London", 420, 600),
    ("new_york_am", "America/New_York", 510, 660),
    ("new_york_pm", "America/New_York", 810, 960),
)


def kill_zone(ts: pd.Timestamp, spec: ExchangeSpec) -> str | None:
    """Name of the ICT kill zone ``ts`` falls in, or ``None``.

    Kill zones are defined in fixed venue-local windows. Crypto bars are mapped
    through the New York windows, the convention for the 24/7 market.
    """
    if spec.continuous:
        local = ts.tz_convert("America/New_York") if ts.tzinfo else ts.tz_localize("UTC").tz_convert(
            "America/New_York")
    else:
        local = ts.tz_convert(spec.tz) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(spec.tz)
    minute = local.hour * 60 + local.minute
    for name, tz_name, start, end in _KILL_ZONES:
        if ZoneInfo(tz_name) is None:  # pragma: no cover - zoneinfo always resolves
            continue
        if local.tzinfo != ZoneInfo(tz_name):
            candidate = ts.tz_convert(tz_name) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(tz_name)
            zone_minute = candidate.hour * 60 + candidate.minute
        else:
            zone_minute = minute
        if start <= zone_minute < end:
            return name
    return None


def in_cash_session(ts: pd.Timestamp, spec: ExchangeSpec) -> bool:
    """Whether a bar's timestamp falls inside a declared cash session."""
    if spec.continuous:
        return True
    local = ts.tz_convert(spec.tz) if ts.tzinfo else ts.tz_localize("UTC").tz_convert(spec.tz)
    minute = local.hour * 60 + local.minute
    return any(_minutes(start) <= minute < _minutes(end) for start, end in spec.sessions)


# --------------------------------------------------------------------------------------
# Normalisation and resampling
# --------------------------------------------------------------------------------------


def normalize_frame(frame: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """Return a clean OHLCV frame: UTC index, lowercase columns, no impossible bars.

    Rows that cannot be traded on are dropped rather than repaired: a bar with a
    non-positive price, a high below its low, or no timestamp is not a price.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(OHLCV), index=pd.DatetimeIndex([], name="timestamp"))

    data = frame.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        index = pd.to_datetime(data.index)
    else:
        date_column = next((c for c in ("Date", "Datetime", "date", "timestamp", "index")
                            if c in data.columns), None)
        if date_column is None:
            # A CSV written from a frame with an unnamed index comes back as
            # ``Unnamed: 0``; accept it only when it actually parses as timestamps.
            for candidate in data.columns:
                if str(candidate).startswith("Unnamed"):
                    parsed = pd.to_datetime(data[candidate], errors="coerce")
                    if parsed.notna().mean() > 0.9:
                        date_column = candidate
                        break
        if date_column is None:
            raise ValueError("market data has no date column or DatetimeIndex")
        index = pd.to_datetime(data[date_column])
        data = data.drop(columns=[date_column])

    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    data = data.rename(columns={"adj_close": "adj_close"})
    missing = [c for c in ("open", "high", "low", "close") if c not in data.columns]
    if missing:
        raise ValueError(f"market data is missing {missing}")
    if "volume" not in data.columns:
        data["volume"] = 0.0

    index = pd.DatetimeIndex(index)
    if index.tz is None:
        index = index.tz_localize(timezone)
    index = index.tz_convert("UTC")
    data.index = index
    data.index.name = "timestamp"

    for column in OHLCV:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[~data.index.isna()]
    data = data.dropna(subset=["open", "high", "low", "close"])
    data = data[(data[["open", "high", "low", "close"]] > 0).all(axis=1)]
    data = data[data["high"] >= data["low"]]
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    return data[list(OHLCV)]


def resample_frame(frame: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate a frame to a coarser timeframe, using each bar's own UTC hour.

    Only coarser or equal timeframes are allowed: manufacturing a finer series
    from a coarser one invents highs and lows that never traded.
    """
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=list(OHLCV), index=pd.DatetimeIndex([], name="timestamp"))
    rule = timeframe.pandas_rule
    aggregated = frame.resample(rule, label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


def quality_report(frame: pd.DataFrame, ticker: str, timeframe: Timeframe,
                   spec: ExchangeSpec, start: str | None = None, end: str | None = None,
                   min_bars: int = 200, max_stale_days: float = 10.0) -> DataQualityReport:
    """Judge a series. Blocking findings stop a run; warnings only narrow confidence."""
    report = DataQualityReport(
        ticker=ticker,
        timeframe=timeframe.value,
        start=start or (str(frame.index[0].date()) if len(frame) else ""),
        end=end or (str(frame.index[-1].date()) if len(frame) else ""),
        bars=int(len(frame)),
    )
    if frame is None or len(frame) == 0:
        report.blocking.append("no bars in the requested range")
        return report

    span_days = max(1.0, (frame.index[-1] - frame.index[0]).total_seconds() / 86400)
    if spec.continuous:
        per_day = 1440 / timeframe.minutes
        report.expected_bars = int(span_days * per_day)
    else:
        session_minutes = sum(_minutes(e) - _minutes(s) for s, e in spec.sessions)
        per_day = max(1.0, session_minutes / timeframe.minutes)
        # ~5 of every 7 calendar days trade; holidays make this an upper bound.
        report.expected_bars = int(span_days * 5 / 7 * per_day) or None

    diffs = frame.index.to_series().diff().dropna()
    if len(diffs):
        step = diffs.median()
        report.gaps = int((diffs > max(step * 3, pd.Timedelta(minutes=timeframe.minutes * 3))).sum())

    report.duplicate_timestamps = int(frame.index.duplicated().sum())
    report.non_positive_prices = int((frame[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    report.zero_volume_bars = int((frame["volume"] <= 0).sum())

    reference = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    report.stale_days = round((reference - frame.index[-1]).total_seconds() / 86400, 2)

    if report.bars < min_bars:
        report.blocking.append(
            f"only {report.bars} bars; at least {min_bars} are required to evaluate a strategy")
    if report.non_positive_prices:
        report.blocking.append(f"{report.non_positive_prices} bars with a non-positive price")
    if report.duplicate_timestamps:
        report.blocking.append(f"{report.duplicate_timestamps} duplicate timestamps")
    if report.coverage is not None and report.coverage < 0.6:
        report.blocking.append(f"data covers only {report.coverage:.0%} of expected bars")
    if report.stale_days is not None and report.stale_days > max_stale_days:
        report.blocking.append(
            f"series ends {report.stale_days:.1f} days before {reference.date()}; "
            "a stale tail cannot be backtested")
    if report.gaps > max(3, report.bars * 0.02):
        report.warnings.append(f"{report.gaps} gaps larger than three bar intervals")
    if report.zero_volume_bars > report.bars * 0.5:
        report.warnings.append(
            f"{report.zero_volume_bars} bars report no volume; volume-derived features are unreliable")
    return report


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


class MarketDataSource(Protocol):
    """Anything that can return normalized bars for one ticker and timeframe."""

    def load(self, ticker: str, timeframe: Timeframe, start: str, end: str) -> pd.DataFrame:
        ...


_YAHOO_INTERVALS = {
    Timeframe.M1: "1m", Timeframe.M5: "5m", Timeframe.M15: "15m",
    Timeframe.M30: "30m", Timeframe.H1: "1h", Timeframe.H4: "1h", Timeframe.D1: "1d",
}


class YahooDataSource:
    """A declared window of Yahoo OHLCV, fetched by the data layer.

    The vendor call itself stays in :mod:`tradingagents.dataflows`: that layer owns
    caching, retries and the error taxonomy, so a rate limit surfaces as a vendor
    error instead of quietly becoming an empty feature table. This adapter only
    translates timeframes into Yahoo intervals and resamples what Yahoo cannot
    serve directly (4h bars come from 1h bars).
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, ticker: str, timeframe: Timeframe, start: str, end: str) -> pd.DataFrame:
        from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv_window

        interval = _YAHOO_INTERVALS[timeframe]
        raw = load_ohlcv_window(ticker, start, end, interval=interval,
                                cache_dir=str(self.cache_dir) if self.cache_dir else None)
        frame = normalize_frame(raw)
        return self._resample(frame, timeframe)

    @staticmethod
    def _resample(frame: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        if timeframe is Timeframe.H4:
            return resample_frame(frame, Timeframe.H4)
        return frame


class CsvDataSource:
    """Bars from one CSV file per (ticker, timeframe), for offline and reproducible work.

    Path template: ``<directory>/{ticker}-{timeframe}.csv``. A single ``{ticker}.csv``
    also works when only one timeframe is held.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self, ticker: str, timeframe: Timeframe, start: str, end: str) -> pd.DataFrame:
        safe = safe_ticker_component(normalize_symbol(ticker))
        candidates = [self.directory / f"{safe}-{timeframe.value}.csv", self.directory / f"{safe}.csv"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"no CSV for {ticker} {timeframe.value} under {self.directory}")
        frame = normalize_frame(pd.read_csv(path))
        window = frame.loc[str(start):str(end)]
        return window


class FrameDataSource:
    """In-memory frames, keyed by timeframe. Used by the CLI cache and by tests."""

    def __init__(self, frames: dict[str | Timeframe, pd.DataFrame]):
        self.frames = {Timeframe.parse(k): v for k, v in frames.items()}

    def load(self, ticker: str, timeframe: Timeframe, start: str, end: str) -> pd.DataFrame:
        frame = self.frames.get(timeframe)
        if frame is None:
            raise KeyError(f"no frame supplied for {timeframe.value}")
        frame = normalize_frame(frame)
        return frame.loc[str(start):str(end)]


# --------------------------------------------------------------------------------------
# MarketData: multi-timeframe bundle with look-ahead-safe alignment
# --------------------------------------------------------------------------------------


@dataclass
class MarketData:
    """Every timeframe of one instrument, with its calendar and data-quality verdicts."""

    ticker: str
    frames: dict[Timeframe, pd.DataFrame]
    asset_type: str = "stock"
    exchange: ExchangeSpec = field(default_factory=lambda: exchange_for("SPY"))
    quality: dict[str, DataQualityReport] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames = {Timeframe.parse(tf): frame for tf, frame in self.frames.items()}
        self.frames = {tf: normalize_frame(frame) for tf, frame in self.frames.items() if len(frame)}

    @property
    def timeframes(self) -> list[Timeframe]:
        return sorted(self.frames, key=lambda tf: tf.minutes)

    @property
    def primary_timeframe(self) -> Timeframe:
        if not self.frames:
            raise ValueError("this MarketData holds no frames")
        # The finest frame is the execution timeframe; coarser ones give context.
        return self.timeframes[0]

    def frame(self, timeframe: Timeframe | str | None = None) -> pd.DataFrame:
        tf = Timeframe.parse(timeframe) if timeframe else self.primary_timeframe
        if tf not in self.frames:
            raise KeyError(f"{self.ticker} has no {tf.value} frame; loaded {[t.value for t in self.frames]}")
        return self.frames[tf]

    def blocking_issues(self) -> list[str]:
        return [f"{tf}: {issue}" for tf, report in sorted(self.quality.items())
                for issue in report.blocking]

    def warnings(self) -> list[str]:
        return [f"{tf}: {issue}" for tf, report in sorted(self.quality.items())
                for issue in report.warnings]

    @property
    def ok(self) -> bool:
        """True only when every loaded timeframe passed its quality gate."""
        return bool(self.quality) and not self.blocking_issues()

    def context(self, timeframe: Timeframe, target_index: pd.DatetimeIndex,
                columns: list[str] | None = None, prefix: str = "") -> pd.DataFrame:
        """Context bars as known at each target timestamp, with no look-ahead.

        For every timestamp in ``target_index`` this picks the last source bar
        whose **close** was at or before it. A source bar that closes after the
        decision cannot contribute, which is the whole point of the exercise.
        """
        source = self.frame(timeframe)
        if source.empty or len(target_index) == 0:
            return pd.DataFrame(index=target_index)
        chosen = source[columns] if columns else source
        folded = chosen.copy()
        folded["__known_at"] = folded.index + pd.Timedelta(minutes=timeframe.minutes)
        folded = folded.sort_values("__known_at")
        target = pd.DataFrame({"__known_at": pd.DatetimeIndex(target_index).sort_values()})
        merged = pd.merge_asof(target, folded, on="__known_at", direction="backward",
                               allow_exact_matches=True)
        merged = merged.set_index("__known_at").rename(columns={
            c: f"{prefix}{c}" for c in merged.columns if c not in ("__known_at",)})
        return merged.reindex(pd.DatetimeIndex(target_index))

    def summary(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "exchange": self.exchange.name,
            "timezone": self.exchange.timezone,
            "timeframes": {
                tf.value: {"bars": int(len(frame)),
                           "first": frame.index[0].isoformat(),
                           "last": frame.index[-1].isoformat()}
                for tf, frame in sorted(self.frames.items(), key=lambda kv: kv[0].minutes)
            },
            "quality": {tf: report.as_dict() for tf, report in self.quality.items()},
            "provenance": self.provenance,
            "ok": self.ok,
        }


def load_market_data(
    ticker: str,
    timeframes: list[str | Timeframe],
    start: str,
    end: str,
    *,
    asset_type: str = "stock",
    source: MarketDataSource | None = None,
    min_bars: int = 200,
) -> MarketData:
    """Load, normalise and grade every requested timeframe for one instrument.

    Raises ``ValueError`` when no timeframe could be loaded; otherwise returns a
    bundle whose ``ok`` flag says whether the data is trustworthy enough to
    trade. The caller decides what to do with a failed gate; the default is to
    refuse new trades.
    """
    wanted = [Timeframe.parse(tf) for tf in timeframes]
    spec = exchange_for(ticker, asset_type)
    source = source or YahooDataSource()
    frames: dict[Timeframe, pd.DataFrame] = {}
    reports: dict[str, DataQualityReport] = {}
    for tf in wanted:
        try:
            raw = source.load(ticker, tf, start, end)
        except Exception as exc:  # an unreachable vendor is a data problem, not a crash
            logger.warning("Could not load %s %s: %s", ticker, tf.value, exc)
            reports[tf.value] = DataQualityReport(
                ticker=ticker, timeframe=tf.value, start=start, end=end, bars=0,
                blocking=[f"load failed: {exc}"])
            continue
        frame = normalize_frame(raw)
        report = quality_report(frame, ticker, tf, spec, start=start, end=end, min_bars=min_bars)
        reports[tf.value] = report
        if len(frame):
            frames[tf] = frame

    if not frames:
        reasons = "; ".join(f"{tf}: {'; '.join(r.blocking)}" for tf, r in reports.items())
        raise ValueError(f"no usable market data for {ticker} between {start} and {end} ({reasons})")

    return MarketData(ticker=ticker, frames=frames, asset_type=asset_type, exchange=spec,
                      quality=reports,
                      provenance={"source": type(source).__name__, "requested": [t.value for t in wanted],
                                  "start": start, "end": end})


def require_trustworthy(data: MarketData) -> None:
    """Stop a run whose data failed its gate, naming every blocking issue.

    Called before any backtest that could influence a decision, so poor data
    blocks new trades rather than being averaged into a result.
    """
    issues = data.blocking_issues()
    if issues:
        raise DataTrustError(data.ticker, issues)


class DataTrustError(RuntimeError):
    """Raised when a series is not fit to trade or evaluate."""

    def __init__(self, ticker: str, issues: list[str]):
        self.ticker = ticker
        self.issues = issues
        super().__init__(f"{ticker}: data quality gate failed: " + "; ".join(issues))


def parse_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate a research window, rejecting a reversed or future-dated range."""
    try:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dates must be YYYY-MM-DD, got {start!r}..{end!r}") from exc
    if end_ts <= start_ts:
        raise ValueError(f"the window ends before it starts: {start}..{end}")
    if start_ts > pd.Timestamp(datetime.now(UTC)):
        raise ValueError(f"the window starts in the future: {start}")
    return start_ts, end_ts


def default_intraday_window(timeframe: Timeframe) -> tuple[str, str]:
    """A window a vendor will actually serve for ``timeframe`` (Yahoo's caps)."""
    days = {Timeframe.M1: 5, Timeframe.M5: 30, Timeframe.M15: 55, Timeframe.M30: 55,
            Timeframe.H1: 300, Timeframe.H4: 300, Timeframe.D1: 1500}[timeframe]
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
