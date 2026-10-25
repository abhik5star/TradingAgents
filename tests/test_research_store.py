"""The learning memory: append-only streams, scrubbing and durable artifacts."""

from __future__ import annotations

import json

import pytest

from tradingagents.research.store import STREAMS, ResearchStore, new_id


@pytest.mark.unit
def test_streams_append_and_read_back_oldest_first(tmp_path):
    store = ResearchStore(root=tmp_path)
    first = store.append("hypothesis", {"hypothesis_id": "H-1", "title": "first"})
    second = store.append("hypothesis", {"hypothesis_id": "H-2", "title": "second"})
    assert first["recorded_at"] and second["recorded_at"]
    records = store.read("hypothesis")
    assert [r["hypothesis_id"] for r in records] == ["H-1", "H-2"]
    assert (tmp_path / STREAMS["hypothesis"]).exists()
    assert store.read("hypothesis", limit=1)[0]["hypothesis_id"] == "H-2"


@pytest.mark.unit
def test_imported_identifiers_are_generated_and_unique():
    ids = {new_id("H") for _ in range(50)}
    assert len(ids) == 50
    assert all(identifier.startswith("H-") for identifier in ids)


@pytest.mark.unit
def test_credentials_never_reach_the_memory(tmp_path):
    store = ResearchStore(root=tmp_path)
    stored = store.append("experiment", {
        "id": "E-1", "api_key": "sk-secret", "context": {"access_token": "abc",
                                                          "nested": {"password": "p"}},
        "note": "safe"})
    assert stored["api_key"] == "[redacted]"
    assert stored["context"]["access_token"] == "[redacted]"
    assert stored["context"]["nested"]["password"] == "[redacted]"
    assert stored["note"] == "safe"
    raw = (tmp_path / STREAMS["experiment"]).read_text()
    assert "sk-secret" not in raw and "abc" not in raw


@pytest.mark.unit
def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    store = ResearchStore(root=tmp_path)
    store.append("autopsy", {"trade_id": "T-1", "verdict": "normal_statistical_loss"})
    with open(tmp_path / STREAMS["autopsy"], "a", encoding="utf-8") as handle:
        handle.write("{not valid json\n")
    records = store.read("autopsy")
    assert len(records) == 1 and records[0]["trade_id"] == "T-1"


@pytest.mark.unit
def test_unknown_stream_kinds_are_rejected():
    store = ResearchStore()
    with pytest.raises(ValueError, match="unknown record kind"):
        store.append("invented", {})
    with pytest.raises(ValueError, match="unknown record kind"):
        store.read("invented")


@pytest.mark.unit
def test_find_locates_records_by_any_identifier(tmp_path):
    store = ResearchStore(root=tmp_path)
    store.append("hypothesis", {"hypothesis_id": "H-9", "title": "x"})
    store.append("experiment", {"id": "EXP-9", "hypothesis_id": "H-9", "accepted": False})
    assert store.find("hypothesis", "H-9")["title"] == "x"
    assert store.find("experiment", "EXP-9")["accepted"] is False
    assert store.find("hypothesis", "missing") is None


@pytest.mark.unit
def test_artifacts_are_written_and_read_inside_the_run_directory(tmp_path):
    store = ResearchStore(root=tmp_path)
    path = store.write_artifact("H-1", "metrics.json", {"trades": 3, "expectancy_r": 0.1})
    assert path.exists()
    assert store.read_artifact("H-1", "metrics.json")["trades"] == 3
    assert store.read_artifact("H-1", "absent.json") is None
    # A path-traversal attempt in an id cannot escape the runs directory.
    outside = store.write_artifact("../escape", "x.json", {"a": 1})
    assert tmp_path.resolve() in outside.resolve().parents


@pytest.mark.unit
def test_registry_holdouts_and_snapshot_documents_round_trip(tmp_path):
    store = ResearchStore(root=tmp_path)
    assert store.load_registry()["versions"] == []
    store.save_registry({"versions": [{"version": "V01"}], "champion": "V01"})
    assert store.load_registry()["champion"] == "V01"

    store.save_holdouts({"windows": {"abc:2026": 2}})
    assert store.load_holdouts()["windows"]["abc:2026"] == 2

    store.write_snapshot({"champion": {"version": "V01"}})
    assert store.read_snapshot()["champion"]["version"] == "V01"
    assert json.loads((tmp_path / "snapshots" / "dashboard.json").read_text())["champion"]


@pytest.mark.unit
def test_an_unreadable_document_falls_back_to_its_default(tmp_path):
    store = ResearchStore(root=tmp_path)
    store.registry_path.write_text("{broken", encoding="utf-8")
    assert store.load_registry()["versions"] == []


@pytest.mark.unit
def test_errors_are_recorded_with_context(tmp_path):
    store = ResearchStore(root=tmp_path)
    store.error("hypothesis_loop", "boom", hypothesis_id="H-1")
    record = store.read("error")[-1]
    assert record["where"] == "hypothesis_loop"
    assert record["context"]["hypothesis_id"] == "H-1"


@pytest.mark.unit
def test_dataset_fingerprint_is_stable_and_sensitive_to_the_data():
    import pandas as pd

    store = ResearchStore()
    index = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    frame = pd.DataFrame({"open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
                          "close": [1.0, 2, 3], "volume": [1.0, 1, 1]}, index=index)
    first = store.dataset_fingerprint({"1d": frame})
    assert first == store.dataset_fingerprint({"1d": frame})
    frame.loc[index[-1], "close"] = 4.0
    assert store.dataset_fingerprint({"1d": frame}) != first
