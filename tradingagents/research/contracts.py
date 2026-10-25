"""Shared, serialisable vocabulary for the research engine.

Every record the engine writes is a dataclass from this module (or one that
mirrors its shape), so the on-disk format is one schema rather than many. Enums
are ``StrEnum`` so a record reads the same in JSON, in the CLI and in the report.

The single invariant this module protects: a stored record is *data*, never a
promise. Numbers are what a backtest measured, and every metric records the
sample it was measured on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Timeframe(StrEnum):
    """A bar size the engine understands, and how many bars a session holds."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}[self.value]

    @property
    def pandas_rule(self) -> str:
        return {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                "1h": "1h", "4h": "4h", "1d": "1D"}[self.value]

    @classmethod
    def parse(cls, value: str | Timeframe) -> Timeframe:
        if isinstance(value, Timeframe):
            return value
        for member in cls:
            if member.value == str(value).strip().lower():
                return member
        raise ValueError(f"unknown timeframe {value!r}; expected one of {[m.value for m in cls]}")


class Regime(StrEnum):
    """Market state a setup is evaluated in. Classification is rule-based."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    UNSTABLE = "unstable"


class Session(StrEnum):
    """Cash-session buckets in exchange-local time (crypto trades in all of them)."""

    ASIA = "asia"
    LONDON = "london"
    OVERLAP = "london_ny_overlap"
    NEW_YORK = "new_york"
    OFF_HOURS = "off_hours"


class ExperimentStatus(StrEnum):
    """The seven states the dashboard shows. Nothing skips a state."""

    RESEARCH = "Research"
    BACKTESTING = "Backtesting"
    VALIDATION = "Validation"
    PAPER_TESTING = "Paper Testing"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    ROLLED_BACK = "Rolled Back"


class DeploymentStage(StrEnum):
    """Paper-first promotion pipeline, in order."""

    PROPOSED = "proposed"
    BACKTESTED = "backtested"
    OOS_VALIDATED = "oos_validated"
    WALK_FORWARD = "walk_forward"
    ROBUSTNESS = "robustness"
    PAPER_TESTED = "paper_tested"
    RISK_REVIEWED = "risk_reviewed"
    AWAITING_APPROVAL = "awaiting_user_approval"
    LIVE_ELIGIBLE = "live_eligible"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class MistakeCategory(StrEnum):
    """Autopsy verdicts. "Normal statistical loss" is a distinct outcome."""

    STRATEGY_LOGIC = "strategy_logic"
    DATA_QUALITY = "data_quality"
    REGIME_CLASSIFICATION = "regime_classification"
    EXECUTION = "execution"
    RISK_MANAGEMENT = "risk_management"
    MODEL_PREDICTION = "model_prediction"
    NORMAL_LOSS = "normal_statistical_loss"
    UNKNOWN = "unknown_or_insufficient_evidence"


class TradeSide(StrEnum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class CostModel:
    """Explicit execution assumptions. Nothing is free unless declared so.

    ``spread_bps`` is the *full* quoted spread; a fill pays half of it on each
    side. ``tax_bps_sell`` applies to the sell leg only (stamp duty, STT, ...).
    ``funding_bps_per_day`` is the derivative carry, charged while a position is
    open on a perpetual. Defaults are deliberately non-zero: a research engine
    that assumes frictionless fills overstates every strategy it tests.
    """

    commission_bps: float = 2.0
    min_commission: float = 0.0
    slippage_bps: float = 2.0
    spread_bps: float = 2.0
    tax_bps_sell: float = 0.0
    funding_bps_per_day: float = 0.0

    def entry_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.spread_bps / 2

    def exit_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.spread_bps / 2 + self.tax_bps_sell

    def stressed(self, multiple: float = 2.0) -> CostModel:
        """The same model with every friction multiplied, for robustness gates."""
        return CostModel(
            commission_bps=self.commission_bps * multiple,
            min_commission=self.min_commission,
            slippage_bps=self.slippage_bps * multiple,
            spread_bps=self.spread_bps * multiple,
            tax_bps_sell=self.tax_bps_sell * multiple,
            funding_bps_per_day=self.funding_bps_per_day * multiple,
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class TradeRecord:
    """One completed round trip, with the evidence an autopsy needs."""

    trade_id: str
    strategy_id: str
    version: str
    ticker: str
    timeframe: str
    side: TradeSide
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_price: float
    target_price: float | None
    gross_pnl: float
    cost: float
    net_pnl: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    regime: str
    session: str
    planned_risk: float
    mae_r: float = 0.0
    mfe_r: float = 0.0
    signals: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass
class DataQualityReport:
    """What the loader could and could not trust about a series.

    ``blocking`` is the list that makes a run refuse to trade: without it a
    vendor outage looks like a strategy failure. ``warnings`` narrows confidence
    but does not stop evaluation.
    """

    ticker: str
    timeframe: str
    start: str
    end: str
    bars: int
    expected_bars: int | None = None
    gaps: int = 0
    duplicate_timestamps: int = 0
    zero_volume_bars: int = 0
    non_positive_prices: int = 0
    stale_days: float | None = None
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def coverage(self) -> float | None:
        if not self.expected_bars:
            return None
        return min(1.0, self.bars / self.expected_bars)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        data["coverage"] = self.coverage
        return data


@dataclass
class Metrics:
    """Performance of one strategy on one sample. Never a forecast."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float | None = None
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    total_cost: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float | None = None
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    total_return_pct: float = 0.0
    exposure_pct: float = 0.0
    avg_bars_held: float = 0.0
    cost_share_of_gross: float | None = None
    by_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    by_session: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return jsonable(asdict(self))


@dataclass
class GateResult:
    """One acceptance check: what was required, what was measured, and whether it passed."""

    name: str
    passed: bool
    required: str
    measured: str
    detail: str = ""
    blocking: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_hash(payload: Any) -> str:
    """A short, stable digest of a JSON-able payload (order-independent for dicts)."""
    text = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, timestamps and numpy scalars into JSON values.

    Hand-rolled rather than ``json.dumps(default=...)`` so the *store* owns one
    conversion, and a round trip through disk returns plain Python types that
    the dataclasses can consume again.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value  # NaN is not JSON
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        return jsonable(value.item())
    if hasattr(value, "tolist"):  # numpy array
        return jsonable(value.tolist())
    return str(value)


# Keys whose values must never reach the learning memory.
_SENSITIVE = ("api_key", "apikey", "token", "secret", "password", "passwd", "credential",
              "authorization", "cookie", "private_key", "session_id")


def scrub_secrets(value: Any) -> Any:
    """Drop credential-shaped keys from anything about to be persisted.

    Applied on the way into the store rather than on the way out, so a secret
    never lands on disk in the first place.
    """
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SENSITIVE):
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = scrub_secrets(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [scrub_secrets(item) for item in value]
    return value
