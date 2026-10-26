"""The risk engine: independent, authoritative, and impossible to loosen from AI output.

Everything else in this package is advisory. This module is not. It is consulted
before every entry, it can only *reduce* a position or refuse it, and its policy
cannot be weakened by a strategy version, an ML probability, an LLM suggestion or
a hypothesis that claims better returns would follow.

The rules it enforces, in order:

1. **No stop, no trade.** An order without a stop price is refused; stops are not
   removable by configuration.
2. **Data first.** A failing data-quality gate blocks new entries outright.
3. **Account protection.** Daily loss, peak-to-trough drawdown and a minimum
   equity floor halt new entries; they do not liquidate, because closing a
   position at a bad moment is its own risk.
4. **Budget.** Per-trade risk is capped at ``max_risk_per_trade`` of equity,
   concurrency and trades-per-day are capped, and leverage is capped.
5. **Distrust.** Model drift or an execution anomaly cuts size (never raises it)
   and is recorded.

Every refusal returns a reason string that is stored. The dashboard reports the
latest one; the acceptance criteria in the brief depend on being able to say
*why* the system is not trading.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any

from tradingagents.research.contracts import TradeSide, utc_now


@dataclass(frozen=True)
class RiskPolicy:
    """The safety envelope. Defaults are conservative and meant to stay that way."""

    max_risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    min_equity_fraction: float = 0.5
    max_leverage: float = 1.0
    max_open_positions: int = 1
    max_trades_per_day: int = 6
    require_data_quality: bool = True
    require_no_model_drift: bool = False
    drift_size_multiplier: float = 0.5
    emergency_stop: bool = False
    allow_short: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RiskPolicy:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})

    def weakened_by(self, other: RiskPolicy) -> list[str]:
        """Which of ``other``'s limits are looser than this policy's.

        The comparison is explicit per field rather than a generic "bigger is
        worse", because for two of them smaller is the safe direction
        (``drift_size_multiplier``, ``min_equity_fraction`` is a floor).
        """
        loosened: list[str] = []
        caps = ("max_risk_per_trade", "max_daily_loss_pct", "max_drawdown_pct", "max_leverage")
        for name in caps:
            if getattr(other, name) > getattr(self, name):
                loosened.append(f"{name}: {getattr(self, name)} -> {getattr(other, name)}")
        if other.max_open_positions > self.max_open_positions:
            loosened.append(f"max_open_positions: {self.max_open_positions} -> "
                            f"{other.max_open_positions}")
        if other.max_trades_per_day > self.max_trades_per_day:
            loosened.append(f"max_trades_per_day: {self.max_trades_per_day} -> "
                            f"{other.max_trades_per_day}")
        if self.require_data_quality and not other.require_data_quality:
            loosened.append("require_data_quality: on -> off")
        if self.require_no_model_drift and not other.require_no_model_drift:
            loosened.append("require_no_model_drift: on -> off")
        if other.drift_size_multiplier > self.drift_size_multiplier:
            loosened.append(f"drift_size_multiplier: {self.drift_size_multiplier} -> "
                            f"{other.drift_size_multiplier}")
        if other.min_equity_fraction < self.min_equity_fraction:
            loosened.append(f"min_equity_fraction: {self.min_equity_fraction} -> "
                            f"{other.min_equity_fraction}")
        if self.emergency_stop and not other.emergency_stop:
            loosened.append("emergency_stop: engaged -> released")
        if self.allow_short and not other.allow_short:
            pass  # restricting shorts is a tightening, not a loosening
        return loosened


@dataclass
class RiskState:
    """The account as the risk engine sees it during a run or a live session."""

    equity: float
    peak_equity: float
    day: date | None = None
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    consecutive_losses: int = 0
    blocked: int = 0
    halted_reason: str | None = None

    def roll_day(self, day: date, equity: float | None = None) -> None:
        """Start a new session: reset the daily counters against that day's opening equity."""
        if self.day != day:
            self.day = day
            self.day_start_equity = float(self.equity if equity is None else equity)
            self.day_pnl = 0.0
            self.trades_today = 0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1 - self.equity / self.peak_equity)

    @property
    def daily_loss_pct(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, -self.day_pnl / self.day_start_equity)

    def as_dict(self) -> dict[str, Any]:
        return {"equity": round(self.equity, 4), "peak_equity": round(self.peak_equity, 4),
                "day": str(self.day) if self.day else None,
                "day_pnl": round(self.day_pnl, 4),
                "trades_today": self.trades_today, "open_positions": self.open_positions,
                "consecutive_losses": self.consecutive_losses, "blocked": self.blocked,
                "drawdown_pct": round(self.drawdown_pct, 6),
                "daily_loss_pct": round(self.daily_loss_pct, 6),
                "halted_reason": self.halted_reason}


@dataclass(frozen=True)
class OrderIntent:
    """What a strategy wants to do, before risk has had its say."""

    ticker: str
    timeframe: str
    side: TradeSide
    price: float
    quantity: float
    stop_price: float | None
    risk_amount: float
    equity: float
    regime: str = ""

    @property
    def notional(self) -> float:
        return abs(self.price * self.quantity)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["side"] = self.side.value if isinstance(self.side, TradeSide) else str(self.side)
        payload["notional"] = self.notional
        return payload


@dataclass
class RiskDecision:
    """Allow or refuse, with every check that produced the answer."""

    allowed: bool
    checks: dict[str, bool]
    reasons: list[str] = field(default_factory=list)
    size_multiplier: float = 1.0

    @property
    def blocked_reason(self) -> str:
        return "; ".join(self.reasons)

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "checks": dict(self.checks),
                "reasons": list(self.reasons),
                "size_multiplier": round(self.size_multiplier, 6)}


@dataclass
class PolicyReview:
    """The outcome of trying to change the risk policy."""

    allowed: bool
    weakened: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"allowed": self.allowed, "weakened": list(self.weakened),
                "reasons": list(self.reasons)}


class RiskEngine:
    """Evaluate orders against a policy, and record every refusal.

    ``store`` is optional; when given, each block and each policy attempt is
    appended to the research memory so the reason survives a restart.
    """

    def __init__(self, policy: RiskPolicy | None = None, store: Any = None):
        self.policy = policy or RiskPolicy()
        self.store = store
        self.blocks: list[dict[str, Any]] = []

    # ---------- policy ----------

    def review_policy_change(self, proposed: RiskPolicy) -> PolicyReview:
        """Refuse any change that loosens a limit, and name what it would loosen."""
        weakened = self.policy.weakened_by(proposed)
        if weakened:
            review = PolicyReview(
                allowed=False, weakened=weakened,
                reasons=["the risk policy can only be tightened, never loosened: "
                         + "; ".join(weakened)])
            self._record("risk_event", {"id": f"RSK-{utc_now()}", "kind": "policy_change_rejected",
                                        "weakened": weakened, "reasons": review.reasons})
            return review
        return PolicyReview(allowed=True, reasons=["change tightens the policy"])

    def apply_policy_change(self, proposed: RiskPolicy) -> PolicyReview:
        """Apply ``proposed`` only if it does not weaken the current policy."""
        review = self.review_policy_change(proposed)
        if review.allowed:
            self.policy = proposed
            self._record("risk_event", {"id": f"RSK-{utc_now()}", "kind": "policy_change_applied",
                                        "policy": proposed.as_dict()})
        return review

    # ---------- evaluation ----------

    def evaluate(self, intent: OrderIntent, state: RiskState, *, quality_ok: bool = True,
                 model_drift: bool = False,
                 execution_anomaly: bool = False) -> RiskDecision:
        """Allow, shrink or refuse an order. Never raises the requested size."""
        policy = self.policy
        checks: dict[str, bool] = {}
        reasons: list[str] = []

        if intent.side is TradeSide.SHORT:
            checks["shorts_allowed"] = policy.allow_short
            if not policy.allow_short:
                reasons.append("short positions are disabled by policy")
        checks["stop_present"] = intent.stop_price is not None and intent.stop_price > 0
        if not checks["stop_present"]:
            reasons.append("no stop price: a stop cannot be removed or omitted")
        checks["emergency_stop_released"] = not policy.emergency_stop
        if policy.emergency_stop:
            reasons.append("the emergency stop is engaged")
        checks["data_quality"] = quality_ok or not policy.require_data_quality
        if not checks["data_quality"]:
            reasons.append("market data failed its quality gate")
        checks["daily_loss"] = state.daily_loss_pct < policy.max_daily_loss_pct
        if not checks["daily_loss"]:
            reasons.append(f"daily loss {state.daily_loss_pct:.2%} reached the "
                           f"{policy.max_daily_loss_pct:.2%} limit")
        checks["drawdown"] = state.drawdown_pct < policy.max_drawdown_pct
        if not checks["drawdown"]:
            reasons.append(f"drawdown {state.drawdown_pct:.2%} reached the "
                           f"{policy.max_drawdown_pct:.2%} limit")
        floor = policy.min_equity_fraction
        checks["equity_floor"] = (state.equity >= state.peak_equity * floor
                                  if state.peak_equity else True)
        if not checks["equity_floor"]:
            reasons.append(f"equity is below {floor:.0%} of its peak")
        checks["open_positions"] = state.open_positions < policy.max_open_positions
        if not checks["open_positions"]:
            reasons.append(f"{state.open_positions} position(s) already open; "
                           f"policy allows {policy.max_open_positions}")
        checks["trades_today"] = state.trades_today < policy.max_trades_per_day
        if not checks["trades_today"]:
            reasons.append(f"{state.trades_today} trades today; policy allows "
                           f"{policy.max_trades_per_day}")
        checks["leverage"] = intent.notional <= max(0.0, intent.equity) * policy.max_leverage * 1.0001
        if not checks["leverage"]:
            reasons.append(f"notional {intent.notional:,.0f} exceeds "
                           f"{policy.max_leverage:g}x equity {intent.equity:,.0f}")

        if policy.require_no_model_drift and model_drift:
            checks["model_drift"] = False
            reasons.append("model drift detected and policy requires a clean model")
        else:
            checks["model_drift"] = True

        # Size can only come down. The per-trade cap is expressed in equity terms.
        multiplier = 1.0
        allowed_risk = policy.max_risk_per_trade * max(intent.equity, 1e-9)
        if intent.risk_amount > allowed_risk > 0:
            multiplier = min(multiplier, allowed_risk / intent.risk_amount)
            reasons.append(f"size reduced to the {policy.max_risk_per_trade:.2%} per-trade cap")
        if model_drift and not policy.require_no_model_drift:
            multiplier = min(multiplier, policy.drift_size_multiplier)
            reasons.append(f"model drift: size cut to {policy.drift_size_multiplier:.0%}")
        if execution_anomaly:
            multiplier = min(multiplier, policy.drift_size_multiplier)
            reasons.append(f"execution anomaly: size cut to {policy.drift_size_multiplier:.0%}")

        allowed = all(checks.values())
        if not allowed:
            state.blocked += 1
            state.halted_reason = reasons[0] if reasons else "risk block"
            self._record("risk_event", {
                "id": f"RSK-{utc_now()}", "kind": "entry_blocked", "intent": intent.as_dict(),
                "checks": checks, "reasons": reasons,
                "state": state.as_dict(), "policy": policy.as_dict(),
            })
            self.blocks.append({"timestamp": utc_now(), "reasons": reasons,
                                "intent": intent.as_dict()})
            multiplier = 0.0
        return RiskDecision(allowed=allowed, checks=checks, reasons=reasons,
                            size_multiplier=round(max(0.0, multiplier), 6))

    # ---------- bookkeeping ----------

    def record_exit(self, pnl: float, state: RiskState) -> None:
        state.day_pnl += 0.0  # the engine sets day_pnl from realised equity; keep both in step
        if pnl < 0:
            state.consecutive_losses += 1
        else:
            state.consecutive_losses = 0

    def halt(self, reason: str, state: RiskState) -> None:
        """Stop new entries for a reason outside the per-order checks (data outage, manual)."""
        state.halted_reason = reason
        self._record("risk_event", {"id": f"RSK-{utc_now()}", "kind": "halt", "reason": reason,
                                    "state": state.as_dict()})

    def resume(self, state: RiskState) -> None:
        state.halted_reason = None

    def _record(self, kind: str, payload: dict[str, Any]) -> None:
        if self.store is not None:
            # Memory must never break trading evaluation.
            with contextlib.suppress(Exception):
                self.store.append(kind, payload)


def tightened(policy: RiskPolicy, **changes: Any) -> tuple[RiskPolicy, PolicyReview]:
    """Build a tightened policy and say whether the engine would accept it.

    A convenience for the paper-to-live review: a candidate may ask for *less*
    risk, and this returns the reviewed policy without applying it.
    """
    proposed = replace(policy, **changes)
    return proposed, RiskEngine(policy).review_policy_change(proposed)


def load_policy(store: Any, default: RiskPolicy | None = None) -> RiskPolicy:
    """The persisted risk policy, or the conservative default.

    The policy lives in the registry document rather than in a strategy version, so
    changing a strategy cannot change the limits it is allowed to trade under.
    """
    try:
        document = store.load_registry()
    except Exception:  # a missing/corrupt registry falls back to the safe default
        return default or RiskPolicy()
    payload = document.get("risk_policy")
    if not payload:
        return default or RiskPolicy()
    try:
        return RiskPolicy.from_dict(payload)
    except TypeError:
        return default or RiskPolicy()


def save_policy(store: Any, policy: RiskPolicy) -> None:
    """Persist a policy that has already been reviewed by :class:`RiskEngine`."""
    document = store.load_registry()
    document["risk_policy"] = policy.as_dict()
    document["risk_policy_updated_at"] = utc_now()
    store.save_registry(document)
