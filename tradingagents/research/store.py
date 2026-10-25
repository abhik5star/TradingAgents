"""Durable, append-only memory for the research engine.

Layout under ``<results_dir>/research``::

    registry.json             immutable strategy versions, champion pointer, approvals
    hypotheses.jsonl          every hypothesis, including rejected ones
    experiments.jsonl         every experiment, including the ones that failed
    autopsies.jsonl           per-trade post-mortems
    risk_events.jsonl         every blocked or halted decision, and why
    deployments.jsonl         promotion, approval and rollback events
    models.json               model registry (candidate / champion / rolled back)
    holdouts.json             which holdout windows have already been looked at
    runs/<run_id>/            trades.csv, equity.csv, metrics.json per experiment
    snapshots/dashboard.json  the last dashboard snapshot

JSONL files are append-only: a record is never rewritten, so an old result cannot
be edited into looking better. Registries that must change (the champion pointer,
an approval) are written whole through a temp file, under a cross-process lock
from :mod:`tradingagents.dataflows.files`.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from tradingagents.dataflows.files import locked
from tradingagents.research.contracts import jsonable, scrub_secrets, stable_hash, utc_now

logger = logging.getLogger(__name__)

# Record kinds that are append-only, and the file each one lands in.
STREAMS = {
    "hypothesis": "hypotheses.jsonl",
    "experiment": "experiments.jsonl",
    "autopsy": "autopsies.jsonl",
    "risk_event": "risk_events.jsonl",
    "deployment": "deployments.jsonl",
    "paper": "paper_sessions.jsonl",
    "ml_event": "ml_events.jsonl",
    "error": "errors.jsonl",
}


def new_id(prefix: str) -> str:
    """A readable, collision-resistant identifier such as ``H-20261025-3f2a91``."""
    stamp = utc_now()[:10].replace("-", "")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


class ResearchStore:
    """The learning memory. One instance per process; safe across processes.

    Nothing here talks to an LLM or a broker. It only reads and writes evidence.
    """

    def __init__(self, config: dict | None = None, root: str | Path | None = None):
        cfg = config or {}
        base = root or cfg.get("research_dir") or Path(cfg.get("results_dir", "results")) / "research"
        self.root = Path(base).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "snapshots").mkdir(exist_ok=True)

    # ---------- paths ----------

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.json"

    @property
    def models_path(self) -> Path:
        return self.root / "models.json"

    @property
    def holdouts_path(self) -> Path:
        return self.root / "holdouts.json"

    def run_dir(self, run_id: str) -> Path:
        safe = "".join(ch for ch in run_id if ch.isalnum() or ch in "-_")
        path = self.root / "runs" / (safe or "run")
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------- append-only streams ----------

    def append(self, kind: str, record: dict[str, Any]) -> dict[str, Any]:
        """Append ``record`` to its stream and return what was written.

        Credential-shaped keys are scrubbed first, and the stored copy is the
        JSON-normalised one, so a caller and a later reader see identical data.
        """
        if kind not in STREAMS:
            raise ValueError(f"unknown record kind {kind!r}; expected one of {sorted(STREAMS)}")
        stored = jsonable(scrub_secrets(record))
        if "recorded_at" not in stored:
            stored["recorded_at"] = utc_now()
        path = self.root / STREAMS[kind]
        line = json.dumps(stored, sort_keys=True, separators=(",", ":"), default=str)
        with locked(path), open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return stored

    def read(self, kind: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Read a stream oldest-first. A corrupt line is skipped, not fatal."""
        if kind not in STREAMS:
            raise ValueError(f"unknown record kind {kind!r}; expected one of {sorted(STREAMS)}")
        path = self.root / STREAMS[kind]
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:  # a torn write must not poison the memory
                logger.warning("Skipped a corrupt line in %s", path.name)
        return records[-limit:] if limit else records

    def find(self, kind: str, record_id: str) -> dict[str, Any] | None:
        """The first record whose ``id`` (or ``experiment_id``/``hypothesis_id``) matches."""
        for record in self.read(kind):
            if record_id in (record.get("id"), record.get("experiment_id"),
                             record.get("hypothesis_id"), record.get("trade_id")):
                return record
        return None

    # ---------- mutable documents, written whole ----------

    def _load_doc(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Rebuilt unreadable %s from its default", path.name)
            return default

    def _write_doc(self, path: Path, document: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with locked(path):
            temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(json.dumps(jsonable(document), indent=2, sort_keys=True), encoding="utf-8")
            temp.replace(path)

    def load_registry(self) -> dict[str, Any]:
        return self._load_doc(self.registry_path, {"versions": [], "champion": None, "approvals": []})

    def save_registry(self, document: dict[str, Any]) -> None:
        self._write_doc(self.registry_path, document)

    def load_models(self) -> dict[str, Any]:
        return self._load_doc(self.models_path, {"models": [], "champion": None})

    def save_models(self, document: dict[str, Any]) -> None:
        self._write_doc(self.models_path, document)

    def load_holdouts(self) -> dict[str, Any]:
        return self._load_doc(self.holdouts_path, {"windows": {}})

    def save_holdouts(self, document: dict[str, Any]) -> None:
        self._write_doc(self.holdouts_path, document)

    # ---------- experiment artifacts ----------

    def write_artifact(self, run_id: str, name: str, payload: Any) -> Path:
        """Persist a backtest artifact (trades, equity, metrics) beside its run."""
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "._-") or "artifact.json"
        path = self.run_dir(run_id) / safe
        text = payload if isinstance(payload, str) else json.dumps(
            jsonable(payload), indent=2, sort_keys=True)
        with locked(path):
            temp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            temp.replace(path)
        return path

    def read_artifact(self, run_id: str, name: str) -> Any | None:
        path = self.run_dir(run_id) / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # ---------- dashboard ----------

    def write_snapshot(self, snapshot: dict[str, Any]) -> Path:
        path = self.root / "snapshots" / "dashboard.json"
        self._write_doc(path, snapshot)
        return path

    def read_snapshot(self) -> dict[str, Any] | None:
        path = self.root / "snapshots" / "dashboard.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # ---------- helpers used by the gates ----------

    def error(self, where: str, message: str, **context: Any) -> None:
        """Record a failure so a later run can explain what went wrong."""
        self.append("error", {"id": new_id("ERR"), "where": where, "message": message,
                              "context": context})

    def dataset_fingerprint(self, frames: dict[str, Any]) -> str:
        """Fingerprint a multi-timeframe dataset from its bar counts and edges.

        Cheap enough to run on every experiment, and stable across processes: it
        identifies the *input* a result belongs to without storing the bars.
        """
        summary = {
            tf: {"bars": int(len(frame)), "first": str(frame.index[0]),
                 "last": str(frame.index[-1]), "close_last": float(frame["close"].iloc[-1])}
            for tf, frame in sorted(frames.items()) if len(frame)
        }
        return stable_hash(summary)[:16]
