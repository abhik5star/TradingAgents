"""Paper trading: a version trades forward, on bars the research never saw.

A backtest answers "what would this have done on the sample I already looked at".
Paper trading answers the harder question: "what does it do on bars that arrived
after the decision was made". The distinction matters because every gate in
:mod:`tradingagents.research.validation` is evaluated on data the researcher has
already seen; the only way to find out whether an edge survives contact with new
bars is to run the version forward and record what happens.

What this module is
-------------------
* one **session** per evaluation: a version, a window that starts after that
  version's out-of-sample test window, the trades it took, and their numbers;
* an **append-only** record per session, so re-running with more bars adds
  evidence instead of overwriting it;
* the ``paper`` field on the version, which is what
  :class:`tradingagents.research.pipeline.PromotionPipeline` reads before anyone
  may request approval.

What this module is not
-----------------------
There is no broker, no order path and no clock. A session is a forward simulation
over bars that already exist in the loaded frame; it is called paper trading
because no capital is at risk and because those bars are *outside* the sample the
candidate was selected on — not because it streams a live feed. Every record names
the window it covered, so a two-week forward window cannot be mistaken for a live
track record.

Live trading is never touched here. A passing session makes a version eligible for
the approval conversation; it cannot enable anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradingagents.research.contracts import ExperimentStatus, jsonable, utc_now
from tradingagents.research.engine import EngineConfig, run_backtest
from tradingagents.research.features import FeatureTable
from tradingagents.research.store import ResearchStore, new_id

#: Statuses a paper session can report.
PENDING = "pending"
PASSED = "passed"
FAILED = "failed"

#: How many of the session's trades are kept on the record for the dashboard.
TRADE_SAMPLE = 10


@dataclass
class PaperSession:
    """One forward evaluation of one version, with the numbers it actually produced."""

    paper_id: str
    version: str
    ticker: str
    timeframe: str
    started: str
    ended: str
    bars: int
    trades: int
    expectancy_r: float
    net_pnl: float
    win_rate: float | None
    max_drawdown_pct: float
    cost_share_of_gross: float | None
    profit_factor: float | None
    status: str
    min_trades: int
    note: str
    start_source: str = "version_split"
    sample_trades: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    @property
    def passed(self) -> bool:
        return self.status == PASSED

    def as_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__.copy())

    def summary_line(self) -> str:
        if self.status == PENDING:
            return f"{self.version} paper {self.status}: {self.note}"
        return (f"{self.version} paper {self.status}: {self.trades} trades, "
                f"expectancy {self.expectancy_r:+.3f}R, net {self.net_pnl:+,.2f}, "
                f"max DD {self.max_drawdown_pct:.1%}, {self.started}..{self.ended}")


# --------------------------------------------------------------------------------------
# Where the forward window starts
# --------------------------------------------------------------------------------------


def _index_timestamp(value: str | pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp:
    """A timestamp comparable with ``index``, whether the index is tz-aware or naive."""
    stamp = pd.Timestamp(value)
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def paper_window_start(version: Any) -> pd.Timestamp | None:
    """The last day of the version's out-of-sample window, if the record kept it.

    A version registered by the hypothesis loop stores its split; one seeded by
    hand may not, in which case the caller has to declare the start explicitly
    rather than have this module guess a comfortable one.
    """
    split = (getattr(version, "dataset", None) or {}).get("split") or {}
    bounds = split.get("test") or ""
    if not bounds:
        return None
    end = str(bounds).split("..")[-1].strip()
    return pd.Timestamp(end) if end else None


# --------------------------------------------------------------------------------------
# Running a session
# --------------------------------------------------------------------------------------


def _trade_sample(trades: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "entry_time": str(trade.entry_time),
            "exit_time": str(trade.exit_time),
            "side": getattr(trade.side, "value", str(trade.side)),
            "entry_price": round(float(trade.entry_price), 4),
            "exit_price": round(float(trade.exit_price), 4),
            "r_multiple": round(float(trade.r_multiple), 4),
            "net_pnl": round(float(trade.net_pnl), 4),
            "exit_reason": trade.exit_reason,
            "regime": trade.regime,
        }
        for trade in trades[:TRADE_SAMPLE]
    ]


def run_paper_session(
    registry: Any,
    store: ResearchStore,
    version: str,
    table: FeatureTable,
    config: EngineConfig | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    min_trades: int = 5,
    min_bars: int = 30,
    by: str = "user",
) -> PaperSession:
    """Trade ``version`` forward over ``table`` and record what it did.

    ``start`` defaults to the end of the version's out-of-sample window, which is
    the earliest bar an honest paper trade can use: everything before it was
    already used to choose the strategy. Only bars *after* it are traded.

    A window with fewer than ``min_bars`` forward bars is recorded as ``pending``,
    so the dashboard shows that paper trading has not started rather than a
    zero-trade session that reads like a failure.
    """
    record = registry.get(version)  # raises VersionError for an unknown version
    config = config or EngineConfig()

    declared_start = start is not None
    begins = pd.Timestamp(start) if declared_start else paper_window_start(record)
    if begins is None:
        raise ValueError(
            f"{version} has no recorded out-of-sample window, so there is no honest "
            "point to start paper trading from; pass an explicit start date")

    begins = _index_timestamp(begins, table.bars.index)
    forward = table.bars.loc[table.bars.index > begins]
    through = str(table.bars.index[-1].date())

    if len(forward) < min_bars:
        pending = PaperSession(
            paper_id=new_id("PPR"), version=version, ticker=table.ticker,
            timeframe=table.timeframe, started=str(begins.date()), ended=through,
            bars=int(len(forward)), trades=0, expectancy_r=0.0, net_pnl=0.0, win_rate=None,
            max_drawdown_pct=0.0, cost_share_of_gross=None, profit_factor=None,
            status=PENDING, min_trades=min_trades,
            start_source="declared" if declared_start else "version_split",
            note=(f"only {len(forward)} bar(s) after the out-of-sample window; paper trading "
                  f"needs at least {min_bars}. Re-run once more data has printed."))
        return _persist(registry, store, pending, by=by)

    window = table.slice(str(forward.index[0].date()), through)
    result = run_backtest(record.spec, window, config)
    metrics = result.metrics
    passed = metrics.trades >= min_trades and metrics.expectancy_r > 0
    span = f"{forward.index[0].date()}..{forward.index[-1].date()}"
    if passed:
        note = (f"paper trading supported {version}: {metrics.trades} forward trades at "
                f"{metrics.expectancy_r:+.3f}R over {span}. That is evidence, not a guarantee.")
    else:
        missing = []
        if metrics.trades < min_trades:
            missing.append(f"{metrics.trades} of {min_trades} required trades")
        if metrics.expectancy_r <= 0:
            missing.append(f"expectancy {metrics.expectancy_r:+.3f}R is not positive")
        note = (f"paper trading did not support {version}: " + "; ".join(missing)
                + ". The candidate stays unapproved.")
    if declared_start:
        # Saying it plainly: a window a person declared is only a forward test if it
        # really starts after the sample the version was chosen on.
        note += (" The start date was declared rather than taken from a recorded "
                 "out-of-sample split; if that window overlaps the data the version was "
                 "selected on, this is not an out-of-sample test.")
    session = PaperSession(
        paper_id=new_id("PPR"), version=version, ticker=window.ticker,
        timeframe=window.timeframe, started=str(forward.index[0].date()),
        ended=str(forward.index[-1].date()), bars=int(len(forward)),
        trades=int(metrics.trades), expectancy_r=round(float(metrics.expectancy_r), 6),
        net_pnl=round(float(metrics.net_pnl), 4),
        win_rate=None if metrics.win_rate is None else round(float(metrics.win_rate), 4),
        max_drawdown_pct=round(float(metrics.max_drawdown_pct), 6),
        cost_share_of_gross=(None if metrics.cost_share_of_gross is None
                             else round(float(metrics.cost_share_of_gross), 6)),
        profit_factor=None if metrics.profit_factor is None
        else round(float(metrics.profit_factor), 6),
        status=PASSED if passed else FAILED, min_trades=min_trades, note=note,
        start_source="declared" if declared_start else "version_split",
        sample_trades=_trade_sample(list(result.trades)))
    return _persist(registry, store, session, by=by, blocked=len(result.blocked))


def _persist(registry: Any, store: ResearchStore, session: PaperSession, *, by: str,
             blocked: int = 0) -> PaperSession:
    """Append the session, then update the version's paper field and status."""
    payload = session.as_dict()
    if blocked:
        payload["blocked_entries"] = int(blocked)
    store.append("paper", {**payload, "kind": "paper_session", "by": by})
    registry.update(session.version, paper=payload, _by=by)
    current = registry.get(session.version).status
    if session.status != PENDING and current in (ExperimentStatus.RESEARCH.value,
                                                 ExperimentStatus.BACKTESTING.value,
                                                 ExperimentStatus.VALIDATION.value):
        registry.set_status(
            session.version, ExperimentStatus.PAPER_TESTING,
            reason=(f"{session.status} paper session {session.paper_id} over "
                    f"{session.started}..{session.ended}"), by=by)
    return session


# --------------------------------------------------------------------------------------
# Reading the record back
# --------------------------------------------------------------------------------------


def paper_history(store: ResearchStore, version: str | None = None,
                  ticker: str | None = None,
                  limit: int | None = None) -> list[dict[str, Any]]:
    """Every recorded paper session, oldest first, optionally for one version or ticker."""
    records = store.read("paper")
    if version:
        records = [record for record in records if record.get("version") == version]
    if ticker:
        records = [record for record in records if record.get("ticker") == ticker]
    if limit:
        records = records[-limit:]
    return records


def latest_paper(store: ResearchStore, version: str | None = None,
                 ticker: str | None = None) -> dict[str, Any] | None:
    """The most recent session, or ``None`` when nothing has been paper traded."""
    records = paper_history(store, version=version, ticker=ticker)
    return records[-1] if records else None
