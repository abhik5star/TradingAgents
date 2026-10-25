"""Evidence-driven strategy research for TradingAgents.

This package is the research and continuous-improvement layer. It sits beside the
existing LLM decision pipeline rather than replacing it:

* the :mod:`~tradingagents.graph` agents read evidence and write a decision,
* the strategy engine here executes *declared rules* on historical bars,
* the ML models are trained on historical data and produce probabilities,
* an LLM may propose research *ideas*, but it cannot approve, size or promote a
  strategy, and never writes to the risk engine's policy.

Every claim this package makes is reproducible from stored records: hypotheses,
experiments, backtest artifacts, model versions, risk blocks and approvals all
persist under ``results_dir/research`` (append-only JSONL plus atomic JSON
registries), so a restart does not lose the learning and an old result is never
silently overwritten.

Nothing here promises profit. A run that rejects every candidate and reports
``NO TRADE`` is a successful run.
"""

from __future__ import annotations

__all__ = [
    "AcceptanceCriteria",
    "CostModel",
    "ExperimentStatus",
    "GateResult",
    "Hypothesis",
    "MarketData",
    "MistakeCategory",
    "Regime",
    "ResearchStore",
    "RiskEngine",
    "RiskPolicy",
    "StrategyRegistry",
    "StrategySpec",
    "Timeframe",
    "TradeRecord",
    "run_hypothesis_loop",
]


def __getattr__(name: str):
    """Import heavy submodules lazily, so ``import tradingagents.research`` stays cheap."""
    _map = {
        "AcceptanceCriteria": ("tradingagents.research.validation", "AcceptanceCriteria"),
        "GateResult": ("tradingagents.research.validation", "GateResult"),
        "CostModel": ("tradingagents.research.contracts", "CostModel"),
        "ExperimentStatus": ("tradingagents.research.contracts", "ExperimentStatus"),
        "Hypothesis": ("tradingagents.research.hypothesis", "Hypothesis"),
        "MarketData": ("tradingagents.research.data", "MarketData"),
        "MistakeCategory": ("tradingagents.research.contracts", "MistakeCategory"),
        "Regime": ("tradingagents.research.contracts", "Regime"),
        "ResearchStore": ("tradingagents.research.store", "ResearchStore"),
        "RiskEngine": ("tradingagents.research.risk", "RiskEngine"),
        "RiskPolicy": ("tradingagents.research.risk", "RiskPolicy"),
        "StrategyRegistry": ("tradingagents.research.versions", "StrategyRegistry"),
        "StrategySpec": ("tradingagents.research.strategies", "StrategySpec"),
        "Timeframe": ("tradingagents.research.contracts", "Timeframe"),
        "TradeRecord": ("tradingagents.research.contracts", "TradeRecord"),
        "run_hypothesis_loop": ("tradingagents.research.hypothesis", "run_hypothesis_loop"),
    }
    try:
        module_name, attribute = _map[name]
    except KeyError:  # pragma: no cover - normal attribute error path
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    from importlib import import_module

    return getattr(import_module(module_name), attribute)
