"""Strategy rules as declarative, testable data.

A strategy is a :class:`StrategySpec`: a set of entry conditions, optional
filters, exit rules, sizing and the timeframes it expects. Conditions are
expressions over the feature table from
:mod:`tradingagents.research.features`, so a rule can be printed, hashed,
compared against an earlier version, and re-run years later with the same
meaning.

This is the *strategy engine*: deterministic rules over historical bars. It is
not the LLM and it is not an ML model. An LLM may propose a rule change; it can
only do so by producing a :class:`RuleChange` object that goes through exactly
the same backtest and validation as a hand-written one, and the ML layer may
supply a probability that a rule set uses as one more condition. All three meet
at :func:`StrategySpec.evaluate` and nowhere else.

The lab is bounded on purpose. :func:`generate_candidates` takes a budget and a
seed, deduplicates by rule signature, and refuses to enumerate the combinatorial
space: 2-5 conditions from declared feature groups, one hypothesis at a time.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.research.contracts import stable_hash

# Operators the DSL understands. Comparisons are element-wise over features;
# crossing operators compare with the previous bar, so they stay causal.
COMPARISONS = (">", ">=", "<", "<=", "==", "!=")
CROSSINGS = ("cross_up", "cross_down")
MEMBERSHIP = ("in", "not_in")
OPERATORS = COMPARISONS + CROSSINGS + MEMBERSHIP


@dataclass(frozen=True)
class Condition:
    """One rule: ``left <op> right`` over feature columns or literals."""

    left: str
    op: str
    right: float | str | tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        if self.op not in OPERATORS:
            raise ValueError(f"unknown operator {self.op!r}; expected one of {OPERATORS}")
        if not self.left:
            raise ValueError("a condition needs a left-hand feature")
        if self.op in MEMBERSHIP:
            values = self.right if isinstance(self.right, tuple) else (self.right,)
            object.__setattr__(self, "right", tuple(float(v) for v in values))
        elif not isinstance(self.right, str):
            object.__setattr__(self, "right", float(self.right))

    def feature_names(self) -> list[str]:
        names = [self.left]
        if isinstance(self.right, str):
            names.append(self.right)
        return names

    def evaluate(self, features: pd.DataFrame) -> pd.Series:
        """The condition's truth value per bar. Missing data is False, never True."""
        left = features[self.left]
        if self.op in MEMBERSHIP:
            result = left.isin(self.right)
            if self.op == "not_in":
                result = ~result
            return result.fillna(False)
        right = features[self.right] if isinstance(self.right, str) else self.right
        if self.op in CROSSINGS:
            previous_left = left.shift(1)
            previous_right = right.shift(1) if isinstance(right, pd.Series) else right
            if self.op == "cross_up":
                above_now = left > right
                below_before = previous_left <= previous_right
            else:
                above_now = left < right
                below_before = previous_left >= previous_right
            return (above_now & below_before).fillna(False)
        operation = {
            ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        }[self.op]
        return operation(left, right).fillna(False)

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.op in MEMBERSHIP:
            joined = ", ".join(f"{value:g}" for value in self.right)
            return f"{self.left} {self.op} [{joined}]"
        right = self.right if isinstance(self.right, str) else f"{self.right:g}"
        return f"{self.left} {self.op} {right}"

    def as_dict(self) -> dict[str, Any]:
        return {"left": self.left, "op": self.op, "right": self.right, "label": self.label}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Condition:
        right = payload["right"]
        if isinstance(right, (list, tuple)):
            right = tuple(right)
        return cls(left=payload["left"], op=payload["op"], right=right,
                   label=payload.get("label", ""))


@dataclass(frozen=True)
class ExitRules:
    """Stops, targets, trailing and partial exits, all in ATR / R units.

    Distances are ATR multiples measured at entry, so a stop means the same thing
    in a quiet market and a violent one. ``target_r`` is in risk multiples after
    costs are ignored for placement (costs are charged on the fill, so the net
    result already reflects them).
    """

    stop_atr: float = 1.5
    target_r: float | None = 2.0
    trailing_atr: float | None = None
    trail_after_r: float = 1.0
    partial_at_r: float | None = None
    partial_fraction: float = 0.5
    move_stop_to_breakeven_at_r: float | None = None
    time_stop_bars: int | None = None

    def __post_init__(self) -> None:
        if self.stop_atr <= 0:
            raise ValueError("a stop distance must be positive; a zero stop is not a risk model")
        if self.target_r is not None and self.target_r <= 0:
            raise ValueError("a target must be positive")
        if self.partial_at_r is not None and not 0 < self.partial_fraction < 1:
            raise ValueError("a partial exit must close a fraction between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PositionSizing:
    """Sizing from a fixed risk fraction, bounded by leverage and instrument limits."""

    risk_per_trade: float = 0.005
    max_leverage: float = 1.0
    min_quantity: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be within (0, 0.05]: an AI cannot raise it")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class StrategySpec:
    """A complete, versioned rule set."""

    strategy_id: str
    version: str
    name: str
    long_entry: tuple[Condition, ...] = ()
    short_entry: tuple[Condition, ...] = ()
    filters: tuple[Condition, ...] = ()
    exits: ExitRules = field(default_factory=ExitRules)
    sizing: PositionSizing = field(default_factory=PositionSizing)
    allowed_regimes: tuple[str, ...] = ()
    timeframe: str = "1d"
    context_timeframe: str | None = None
    confirm_timeframe: str | None = None
    description: str = ""
    parent_version: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.strategy_id or not self.version:
            raise ValueError("a strategy needs an id and a version")
        if not self.long_entry and not self.short_entry:
            raise ValueError("a strategy needs at least one entry rule")

    # ----- identity -----

    def signature(self) -> str:
        """A digest of the rules only: two versions that trade identically match.

        Version, name and provenance are excluded, which is what makes this useful
        for duplicate detection across the lab's candidate stream.
        """
        return stable_hash({
            "long": [c.as_dict() for c in self.long_entry],
            "short": [c.as_dict() for c in self.short_entry],
            "filters": [c.as_dict() for c in self.filters],
            "exits": self.exits.as_dict(),
            "sizing": self.sizing.as_dict(),
            "regimes": sorted(self.allowed_regimes),
            "timeframe": self.timeframe,
            "context": self.context_timeframe,
            "confirm": self.confirm_timeframe,
        })[:16]

    def required_columns(self) -> list[str]:
        names: set[str] = set()
        for condition in (*self.long_entry, *self.short_entry, *self.filters):
            names.update(condition.feature_names())
        return sorted(names)

    def validate_columns(self, columns: Iterable[str]) -> None:
        """Fail before a backtest if a rule names a column the table does not have."""
        available = set(columns)
        missing = [name for name in self.required_columns() if name not in available]
        if missing:
            raise ValueError(
                f"{self.name} references features that are not in this feature table: {missing}")

    # ----- evaluation -----

    def evaluate(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-bar long/short signals and the filter mask, all causal.

        A bar can never be both long and short: with conflicting rules the signal
        is dropped and counted as "conflicted", because a strategy that does not
        know its own direction has no business sizing one.
        """
        def all_true(conditions: tuple[Condition, ...]) -> pd.Series:
            mask = pd.Series(True, index=features.index)
            for condition in conditions:
                mask &= condition.evaluate(features)
            return mask

        filters = all_true(self.filters) if self.filters else pd.Series(True, index=features.index)
        if self.allowed_regimes and "regime" in features.columns:
            filters &= features["regime"].isin(list(self.allowed_regimes))
        long_signal = all_true(self.long_entry) if self.long_entry else pd.Series(
            False, index=features.index)
        short_signal = all_true(self.short_entry) if self.short_entry else pd.Series(
            False, index=features.index)
        conflicted = long_signal & short_signal
        return pd.DataFrame({
            "long": (long_signal & filters & ~conflicted).fillna(False),
            "short": (short_signal & filters & ~conflicted).fillna(False),
            "filtered": filters.fillna(False),
            "conflicted": conflicted.fillna(False),
        }, index=features.index)

    def with_version(self, version: str, *, name: str | None = None,
                     description: str | None = None) -> StrategySpec:
        return replace(self, version=version, name=name or self.name,
                       description=self.description if description is None else description,
                       parent_version=self.parent_version or self.version)

    # ----- serialisation -----

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "name": self.name,
            "long_entry": [c.as_dict() for c in self.long_entry],
            "short_entry": [c.as_dict() for c in self.short_entry],
            "filters": [c.as_dict() for c in self.filters],
            "exits": self.exits.as_dict(),
            "sizing": self.sizing.as_dict(),
            "allowed_regimes": list(self.allowed_regimes),
            "timeframe": self.timeframe,
            "context_timeframe": self.context_timeframe,
            "confirm_timeframe": self.confirm_timeframe,
            "description": self.description,
            "parent_version": self.parent_version,
            "provenance": self.provenance,
            "signature": self.signature(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategySpec:
        return cls(
            strategy_id=payload["strategy_id"],
            version=payload["version"],
            name=payload["name"],
            long_entry=tuple(Condition.from_dict(c) for c in payload.get("long_entry", ())),
            short_entry=tuple(Condition.from_dict(c) for c in payload.get("short_entry", ())),
            filters=tuple(Condition.from_dict(c) for c in payload.get("filters", ())),
            exits=ExitRules(**payload.get("exits", {})),
            sizing=PositionSizing(**payload.get("sizing", {})),
            allowed_regimes=tuple(payload.get("allowed_regimes", ())),
            timeframe=payload.get("timeframe", "1d"),
            context_timeframe=payload.get("context_timeframe"),
            confirm_timeframe=payload.get("confirm_timeframe"),
            description=payload.get("description", ""),
            parent_version=payload.get("parent_version"),
            provenance=payload.get("provenance", {}),
        )

    def describe(self) -> str:
        """A plain-language rendering of the rules, for reports and autopsies."""
        lines = [f"{self.name} ({self.version}) on {self.timeframe}"]
        if self.context_timeframe:
            lines.append(f"  context: {self.context_timeframe}"
                         + (f", confirmation: {self.confirm_timeframe}"
                            if self.confirm_timeframe else ""))
        for label, conditions in (("filters", self.filters), ("long if", self.long_entry),
                                  ("short if", self.short_entry)):
            if conditions:
                lines.append(f"  {label}: " + " AND ".join(c.describe() for c in conditions))
        if self.allowed_regimes:
            lines.append("  regimes: " + ", ".join(self.allowed_regimes))
        exits = [f"stop {self.exits.stop_atr:g} ATR"]
        if self.exits.target_r is not None:
            exits.append(f"target {self.exits.target_r:g}R")
        if self.exits.trailing_atr is not None:
            exits.append(f"trail {self.exits.trailing_atr:g} ATR after "
                         f"{self.exits.trail_after_r:g}R")
        if self.exits.partial_at_r is not None:
            exits.append(f"partial {self.exits.partial_fraction:.0%} at "
                         f"{self.exits.partial_at_r:g}R")
        if self.exits.time_stop_bars:
            exits.append(f"time stop {self.exits.time_stop_bars} bars")
        lines.append("  exits: " + ", ".join(exits))
        lines.append(f"  risk: {self.sizing.risk_per_trade:.2%} of equity per trade, "
                     f"max leverage {self.sizing.max_leverage:g}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------------------


def baseline_strategy(timeframe: str = "1d", context_timeframe: str | None = None) -> StrategySpec:
    """The simple, honest baseline every candidate must beat.

    Long only: a moving-average stack that agrees with the long-term trend and
    with Supertrend, entered on a momentum cross, stopped 1.5 ATR away, target 2R.
    No regime filter, no session filter, no ML. If a clever candidate cannot beat
    this after costs on out-of-sample data, it has not earned a deployment.
    """
    long_entry = (
        Condition("ema_stack_up", "==", 1.0, "EMA 9 > 21 > 50"),
        Condition("price", ">", "sma_200", "close above the 200-period average"),
        Condition("supertrend_up", "==", 1.0, "Supertrend is up"),
        Condition("macd_hist", ">", 0.0, "MACD histogram positive"),
    )
    return StrategySpec(
        strategy_id="baseline_trend_follow",
        version="V01",
        name="Baseline trend follow",
        long_entry=long_entry,
        exits=ExitRules(stop_atr=1.5, target_r=2.0),
        sizing=PositionSizing(risk_per_trade=0.005, max_leverage=1.0),
        timeframe=timeframe,
        context_timeframe=context_timeframe,
        description="Long-only trend continuation; the reference every experiment is measured against.",
    )


def baseline_with_trend_measure(feature_columns: Iterable[str]) -> StrategySpec:
    """The baseline, adapted to whichever trend-measure column the table carries.

    ``close_above_sma200`` is not a stored column (it is derived), so this helper
    swaps in the equivalent condition and keeps the baseline applicable to any
    table.
    """
    columns = set(feature_columns)
    if "sma_200" in columns:
        condition = Condition("price", ">", "sma_200", "close above the 200-period average")
    elif "sma_50" in columns:
        condition = Condition("price", ">", "sma_50", "close above the 50-period average")
    else:
        condition = Condition("trend_slope_20", ">", 0.0, "positive 20-bar trend slope")
    spec = baseline_strategy()
    return replace(spec, long_entry=(condition, *spec.long_entry[1:]))


# --------------------------------------------------------------------------------------
# Rule changes (the only way a hypothesis may alter a strategy)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleChange:
    """A single, explainable modification a hypothesis proposes.

    ``kind`` is one of the declared mutations below; ``params`` carries the
    specifics. Anything not in the list cannot be built, which keeps an LLM's idea
    inside the space the lab can actually test and report on.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    KINDS = (
        "add_filter", "add_regime_filter", "add_session_filter", "add_confirmation",
        "tighten_stop", "loosen_stop", "raise_target", "lower_target", "add_trailing_stop",
        "add_partial_exit", "add_time_stop", "reduce_risk", "require_htf_alignment",
    )

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"unknown rule change {self.kind!r}; expected one of {list(self.KINDS)}")

    def describe(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.kind}({detail})" if detail else self.kind

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params), "rationale": self.rationale}


def apply_rule_change(spec: StrategySpec, change: RuleChange) -> StrategySpec:
    """Return the candidate produced by applying ``change`` to ``spec``.

    Every mutation is additive or tightening except the ones that explicitly
    loosen an exit, and none of them can touch the risk engine's policy: sizing
    here is bounded by :class:`PositionSizing` and the independent risk layer
    re-checks it on every order.
    """
    params = change.params
    if change.kind == "add_filter":
        condition = Condition.from_dict(params["condition"])
        return replace(spec, filters=(*spec.filters, condition))
    if change.kind == "add_regime_filter":
        regimes = tuple(params.get("regimes", ()))
        return replace(spec, allowed_regimes=tuple(sorted(set(spec.allowed_regimes) | set(regimes))))
    if change.kind == "add_session_filter":
        sessions = tuple(float(s) for s in params.get("sessions", ()))
        return replace(spec, filters=(*spec.filters,
                                      Condition("session_id", "in", sessions,
                                                "session filter")))
    if change.kind == "add_confirmation":
        condition = Condition.from_dict(params["condition"])
        return replace(spec, filters=(*spec.filters, condition))
    if change.kind == "require_htf_alignment":
        direction = params.get("direction", "up")
        column = "mtf_aligned_down" if direction == "down" else "mtf_aligned_up"
        return replace(spec, filters=(*spec.filters,
                                      Condition(column, "==", 1.0, f"{direction} timeframe alignment")))
    if change.kind in ("tighten_stop", "loosen_stop"):
        factor = float(params.get("factor", 0.8 if change.kind == "tighten_stop" else 1.25))
        return replace(spec, exits=replace(spec.exits,
                                           stop_atr=round(spec.exits.stop_atr * factor, 3)))
    if change.kind in ("raise_target", "lower_target"):
        factor = float(params.get("factor", 1.25 if change.kind == "raise_target" else 0.8))
        current = spec.exits.target_r if spec.exits.target_r is not None else 2.0
        return replace(spec, exits=replace(spec.exits, target_r=round(current * factor, 3)))
    if change.kind == "add_trailing_stop":
        trailing = float(params.get("trailing_atr", 1.0))
        after = float(params.get("trail_after_r", 1.0))
        return replace(spec, exits=replace(spec.exits, trailing_atr=trailing,
                                           trail_after_r=after))
    if change.kind == "add_partial_exit":
        return replace(spec, exits=replace(
            spec.exits,
            partial_at_r=float(params.get("partial_at_r", 1.0)),
            partial_fraction=float(params.get("partial_fraction", 0.5))))
    if change.kind == "add_time_stop":
        return replace(spec, exits=replace(spec.exits,
                                           time_stop_bars=int(params.get("bars", 20))))
    if change.kind == "reduce_risk":
        return replace(spec, sizing=replace(
            spec.sizing, risk_per_trade=round(spec.sizing.risk_per_trade
                                              * float(params.get("factor", 0.5)), 6)))
    raise ValueError(f"unhandled rule change {change.kind!r}")  # pragma: no cover - guarded above


def apply_rule_changes(spec: StrategySpec, changes: list[RuleChange]) -> StrategySpec:
    """Apply changes in order; a rejected change raises rather than silently skipping."""
    result = spec
    for change in changes:
        result = apply_rule_change(result, change)
    return result


# --------------------------------------------------------------------------------------
# The discovery lab
# --------------------------------------------------------------------------------------

# Direction-appropriate condition templates per feature group. Each entry is
# (feature, operator, right, human label) for a long-side trigger; the short side
# is the mirror where one exists.
_LONG_TEMPLATES: dict[str, list[tuple[str, str, float | str, str]]] = {
    "trend": [("ema_stack_up", "==", 1.0, "EMA stack up"),
              ("supertrend_up", "==", 1.0, "Supertrend up"),
              ("macd_hist", ">", 0.0, "MACD histogram positive"),
              ("price_above_vwap", "==", 1.0, "price above VWAP")],
    "momentum": [("rsi_14", ">", 50.0, "RSI above 50"),
                 ("stoch_rsi_k", ">", 50.0, "Stochastic RSI above 50"),
                 ("momentum_10", ">", 0.0, "positive 10-bar momentum")],
    "structure": [("bos_up", "==", 1.0, "break of structure up"),
                  ("choch_up", "==", 1.0, "change of character up"),
                  ("sweep_sell_side", "==", 1.0, "sell-side liquidity sweep")],
    "liquidity": [("fvg_bull", "==", 1.0, "bullish fair value gap"),
                  ("in_bull_ob", "==", 1.0, "bullish order block retest"),
                  ("discount", "==", 1.0, "discount half of the dealing range")],
    "levels": [("at_support", "==", 1.0, "at support"),
               ("retest_up", "==", 1.0, "retest of a broken level"),
               ("breakout_confirmed_up", "==", 1.0, "confirmed breakout")],
    "patterns": [("pattern_any_bullish", "==", 1.0, "bullish chart pattern")],
    "session": [("in_kill_zone", "==", 1.0, "inside a kill zone")],
    "confirmation": [("ltf_bos_up", "==", 1.0, "lower-timeframe BOS up"),
                     ("ltf_structure_dir", ">", 0.0, "lower-timeframe structure up")],
    "regime": [("regime_adx", ">", 25.0, "ADX above 25")],
    "flow_proxy": [("proxy_cvd_slope_20", ">", 0.0, "positive CVD proxy slope")],
}

_SHORT_TEMPLATES: dict[str, list[tuple[str, str, float | str, str]]] = {
    "trend": [("ema_stack_down", "==", 1.0, "EMA stack down"),
              ("supertrend_up", "==", 0.0, "Supertrend down"),
              ("macd_hist", "<", 0.0, "MACD histogram negative")],
    "momentum": [("rsi_14", "<", 50.0, "RSI below 50"),
                 ("stoch_rsi_k", "<", 50.0, "Stochastic RSI below 50")],
    "structure": [("bos_down", "==", 1.0, "break of structure down"),
                  ("choch_down", "==", 1.0, "change of character down"),
                  ("sweep_buy_side", "==", 1.0, "buy-side liquidity sweep")],
    "liquidity": [("fvg_bear", "==", 1.0, "bearish fair value gap"),
                  ("in_bear_ob", "==", 1.0, "bearish order block retest"),
                  ("premium", "==", 1.0, "premium half of the dealing range")],
    "levels": [("at_resistance", "==", 1.0, "at resistance"),
               ("retest_down", "==", 1.0, "retest of a broken level")],
    "patterns": [("pattern_any_bearish", "==", 1.0, "bearish chart pattern")],
    "confirmation": [("ltf_bos_down", "==", 1.0, "lower-timeframe BOS down")],
}

_REGIME_FILTERS = {
    "trend_only": (("trend_up", "trend_down"),),
    "no_sideways": (("trend_up", "trend_down", "breakout", "volatile"),),
}


def _conditions_for(templates: list[tuple[str, str, float | str, str]]) -> Condition:
    feature, op, right, label = templates[0]
    return Condition(feature, op, right, label)


def _template_library(direction: str, groups: Iterable[str],
                      columns: Iterable[str]) -> list[Condition]:
    available = set(columns)
    library: list[Condition] = []
    templates = _LONG_TEMPLATES if direction == "long" else _SHORT_TEMPLATES
    for group in groups:
        for feature, op, right, label in templates.get(group, []):
            if feature in available:
                library.append(Condition(feature, op, right, label))
    # Deduplicate while preserving order.
    seen, unique = set(), []
    for condition in library:
        if condition.label not in seen:
            seen.add(condition.label)
            unique.append(condition)
    return unique


def generate_candidates(
    base: StrategySpec,
    columns: Iterable[str],
    *,
    budget: int = 12,
    seed: int = 7,
    min_conditions: int = 2,
    max_conditions: int = 5,
    groups: Iterable[str] = ("trend", "momentum", "structure", "liquidity", "levels",
                             "patterns", "session"),
    allow_short: bool = True,
    existing_signatures: Iterable[str] = (),
) -> list[StrategySpec]:
    """Bounded, reproducible candidate generation.

    ``budget`` caps how many candidates come back. Combination size is 2-5
    conditions drawn from declared feature groups, and any candidate whose rule
    signature already exists (or repeats within this batch) is dropped, so the
    lab cannot burn its experiment budget re-running the same strategy under a new
    name. The base strategy's own filters are always inherited, so a candidate is
    a *superset* of the baseline's tradeable conditions plus new evidence.
    """
    rng = random.Random(seed)
    known = set(existing_signatures) | {base.signature()}
    long_library = _template_library("long", groups, columns)
    short_library = _template_library("short", groups, columns) if allow_short else []
    if len(long_library) < min_conditions:
        return []

    candidates: list[StrategySpec] = []
    attempts = 0
    max_attempts = budget * 25
    while len(candidates) < budget and attempts < max_attempts:
        attempts += 1
        size = rng.randint(min(min_conditions, len(long_library)),
                          min(max_conditions, len(long_library)))
        chosen = tuple(sorted(rng.sample(long_library, size), key=lambda c: c.label))
        short_conditions: tuple[Condition, ...] = ()
        if short_library:
            short_size = rng.randint(1, min(3, len(short_library))) if rng.random() < 0.5 else 0
            if short_size:
                short_conditions = tuple(sorted(rng.sample(short_library, short_size),
                                                key=lambda c: c.label))
        spec = replace(
            base,
            strategy_id=f"{base.strategy_id}_cand",
            name=" · ".join(c.label for c in chosen[:3]),
            long_entry=tuple(dict.fromkeys((*base.long_entry, *chosen))),
            short_entry=short_conditions,
            description=f"Lab candidate: base rules plus {', '.join(c.label for c in chosen)}.",
        )
        if spec.signature() in known:
            continue
        known.add(spec.signature())
        candidates.append(spec)
    return candidates


def duplicate_candidates(candidates: Iterable[StrategySpec],
                         existing_signatures: Iterable[str] = (),
                         overlap_threshold: float = 0.8) -> list[StrategySpec]:
    """Candidates that repeat already-known rules, by signature or near-identical rules.

    Near-duplicates are compared on the set of condition labels: two candidates
    that differ only in one condition out of five are the same idea, and testing
    both inflates the multiple-testing count for no evidence gain.
    """
    known = set(existing_signatures)
    kept_labels: list[set[str]] = []
    duplicates: list[StrategySpec] = []
    for spec in candidates:
        labels = {c.label or c.describe() for c in (*spec.long_entry, *spec.short_entry,
                                                    *spec.filters)}
        if spec.signature() in known:
            duplicates.append(spec)
            continue
        near = next((other for other in kept_labels
                     if len(labels & other) / max(1, len(labels | other)) >= overlap_threshold),
                    None)
        if near is not None:
            duplicates.append(spec)
            continue
        kept_labels.append(labels)
        known.add(spec.signature())
    return duplicates


def candidate_brief(spec: StrategySpec) -> str:
    """One line describing a candidate, for CLI output."""
    parts = [c.label or c.describe() for c in spec.long_entry[-4:]]
    return " + ".join(parts) + (f" | short: {len(spec.short_entry)} rules"
                                if spec.short_entry else " | long only")


#: A fixed default parameter grid used by the sensitivity gate. Deliberately small:
#: a wide grid on one dataset is a multiple-testing machine.
DEFAULT_PARAMETER_GRID = {
    "stop_atr": (1.0, 1.5, 2.0, 2.5),
    "target_r": (1.5, 2.0, 2.5, 3.0),
}


def parameter_variants(spec: StrategySpec,
                       grid: dict[str, tuple[float, ...]] | None = None) -> list[StrategySpec]:
    """Every combination in a declared grid, as separate specs for the stability test."""
    grid = grid or DEFAULT_PARAMETER_GRID
    keys = list(grid)
    variants: list[StrategySpec] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        exits = replace(spec.exits, **dict(zip(keys, values, strict=True)))
        variants.append(replace(spec, exits=exits,
                                version=f"{spec.version}-p{len(variants):02d}"))
    return variants


def np_where(condition: pd.Series, when_true: float, when_false: float) -> np.ndarray:
    """Small helper used by tests and reports; kept public so behaviour is shared."""
    return np.where(condition.to_numpy(), when_true, when_false)
