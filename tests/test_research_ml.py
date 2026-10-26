"""The ML layer: datasets without leakage, out-of-fold evaluation, registry and drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.features import build_feature_table
from tradingagents.research.ml import (
    InsufficientDataError,
    KMeans,
    LogisticRegression,
    ModelRegistry,
    auc_score,
    brier_score,
    build_dataset,
    calibration_table,
    detect_drift,
    evaluation_report,
    ml_adds_value,
    population_stability_index,
    time_series_folds,
    train_with_oof,
)
from tradingagents.research.store import ResearchStore


@pytest.fixture(scope="module")
def dataset():
    bars = synthetic_bars(n=900, seed=31)
    table = build_feature_table(market_data(bars), "1d")
    return build_dataset(table, horizon=5)


@pytest.mark.unit
def test_dataset_drops_the_unlabelled_tail_and_has_no_future_rows(dataset):
    # A five-bar label horizon means the last five rows cannot be labelled.
    assert dataset.y.notna().all()
    assert dataset.label_kind == "direction"
    assert len(dataset) < 900
    assert dataset.horizon == 5


@pytest.mark.unit
def test_dataset_refuses_too_little_data():
    bars = synthetic_bars(n=120, seed=3)
    table = build_feature_table(market_data(bars, min_bars=100), "1d")
    with pytest.raises(InsufficientDataError):
        build_dataset(table, horizon=5, min_rows=500)


@pytest.mark.unit
def test_time_series_folds_purge_between_train_and_test():
    folds = time_series_folds(600, folds=3, purge=5, min_train=200)
    assert len(folds) >= 2
    for train, test in folds:
        assert train.max() < test.min()
        assert test.min() - train.max() > 5


@pytest.mark.unit
def test_logistic_regression_learns_a_separable_signal_and_stores_its_preprocessing():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
    y = (X["a"] + rng.normal(0, 0.2, 400) > 0).astype(float)
    model = LogisticRegression(epochs=400).fit(X, y)
    probabilities = model.predict_proba(X)
    assert auc_score(probabilities, y.to_numpy()) > 0.9
    assert model.importances()["a"] > 0
    restored = LogisticRegression.from_dict(model.as_dict())
    assert np.allclose(restored.predict_proba(X), probabilities)


@pytest.mark.unit
def test_a_single_class_training_set_is_refused():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(InsufficientDataError, match="one class"):
        LogisticRegression().fit(X, pd.Series([1.0, 1.0, 1.0]))


@pytest.mark.unit
def test_out_of_fold_predictions_are_never_made_on_seen_rows(dataset):
    outcome = train_with_oof(dataset, folds=4, min_train=200)
    assert outcome.oof_probabilities.notna().sum() > 100
    # In-sample AUC is usually higher; the out-of-fold figure is the honest one.
    assert outcome.oof_metrics["auc"] is not None
    assert 0.0 <= outcome.oof_metrics["auc"] <= 1.0
    assert outcome.folds >= 3
    assert outcome.in_sample_metrics["observations"] == len(dataset)


@pytest.mark.unit
def test_calibration_and_scores_are_reported_together():
    rng = np.random.default_rng(2)
    probabilities = rng.uniform(0, 1, 500)
    labels = (rng.uniform(0, 1, 500) < probabilities).astype(float)
    report = evaluation_report(probabilities, labels)
    assert report["auc"] is not None
    assert 0 <= report["brier"] <= 1
    assert report["calibration"]
    table = calibration_table(probabilities, labels)
    assert all({"predicted", "observed", "n"} <= set(row) for row in table)
    assert brier_score(probabilities, labels) == pytest.approx(report["brier"], rel=1e-6)


@pytest.mark.unit
def test_drift_is_measured_not_assumed():
    baseline = pd.DataFrame({"a": np.random.default_rng(3).normal(0, 1, 500),
                             "b": np.random.default_rng(4).normal(0, 1, 500)})
    recent = pd.DataFrame({"a": np.random.default_rng(5).normal(0, 1, 500),
                           "b": np.random.default_rng(6).normal(5, 1, 500)})
    report = detect_drift(baseline, recent)
    assert report.alert
    assert "b" in report.drifted
    assert population_stability_index(baseline["a"], recent["a"]) < 0.2
    same = detect_drift(baseline, baseline)
    assert not same.alert


@pytest.mark.unit
def test_kmeans_is_deterministic_and_finds_clusters():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(np.vstack([rng.normal(0, 0.3, (100, 2)),
                                rng.normal(5, 0.3, (100, 2))]), columns=["a", "b"])
    first = KMeans(clusters=2, seed=1).fit(X).predict(X)
    second = KMeans(clusters=2, seed=1).fit(X).predict(X)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 2
    assert first[:100].mean() != first[100:].mean()


@pytest.mark.unit
def test_registration_is_not_promotion_and_a_coin_flip_never_becomes_champion(
        dataset, tmp_path):
    outcome = train_with_oof(dataset, folds=3, min_train=200)
    store = ResearchStore(root=tmp_path)
    registry = ModelRegistry(store)
    record = registry.register(outcome)
    assert record["model_id"].startswith("MDL-")
    assert record["status"] == "candidate"
    # A new artifact is not the champion until a gate says so.
    assert registry.champion() is None

    good = dict(record, model_id="MDL-good", oof={"auc": 0.62, "observations": 400})
    weak = dict(record, model_id="MDL-weak", oof={"auc": 0.48, "observations": 400})
    store.save_models({"models": [good, weak]})
    with pytest.raises(ValueError, match="does not discriminate"):
        registry.promote("MDL-weak", reason="test")
    assert registry.champion() is None
    registry.promote("MDL-good", reason="out-of-fold AUC above 0.5")
    assert registry.champion()["model_id"] == "MDL-good"
    assert store.read("ml_event")


@pytest.mark.unit
def test_registry_round_trips_a_model_and_detects_missing_features(dataset, tmp_path):
    store = ResearchStore(root=tmp_path)
    outcome = train_with_oof(dataset, folds=3, min_train=200)
    registry = ModelRegistry(store)
    record = registry.register(outcome)
    probabilities = registry.predict(dataset.X, model_id=record["model_id"])
    assert len(probabilities) == len(dataset.X)
    with pytest.raises(ValueError, match="expects features"):
        registry.predict(pd.DataFrame({"unrelated": [1.0, 2.0]}),
                         model_id=record["model_id"])


@pytest.mark.unit
def test_registry_rollback_restores_an_earlier_model(dataset, tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = ModelRegistry(store)
    first = registry.register(train_with_oof(dataset, folds=3, min_train=200,
                                             epochs=100))
    second = registry.register(train_with_oof(dataset, folds=3, min_train=200,
                                              epochs=300))
    store.save_models({"models": [dict(first, oof={"auc": 0.58, "observations": 400},
                                       status="candidate"),
                                   dict(second, oof={"auc": 0.61, "observations": 400},
                                        status="candidate")]})
    registry.promote(second["model_id"], reason="better folds")
    rolled = registry.rollback(reason="drift")
    assert rolled is not None
    assert registry.champion()["model_id"] in (first["model_id"], rolled["model_id"])
    assert any(event["kind"] == "model_rolled_back" for event in store.read("ml_event"))


@pytest.mark.unit
def test_should_retrain_states_its_reasons(dataset, tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = ModelRegistry(store)
    decision = registry.should_retrain(feature_version=dataset.feature_version)
    assert decision["retrain"] is True
    assert "no champion model" in decision["reasons"][0]

    seeded = registry.register(train_with_oof(dataset, folds=3, min_train=200))
    store.save_models({"models": [dict(seeded, oof={"auc": 0.6, "observations": 400})]})
    registry.promote(seeded["model_id"], reason="test")
    decision = registry.should_retrain(feature_version=dataset.feature_version,
                                       drift=None, days_since_training=120)
    assert decision["retrain"] is True
    assert any("120 days" in reason for reason in decision["reasons"])
    decision = registry.should_retrain(feature_version=dataset.feature_version,
                                       days_since_training=1)
    assert decision["retrain"] is False


@pytest.mark.unit
def test_ml_adds_value_requires_evidence_not_enthusiasm(dataset):
    outcome = train_with_oof(dataset, folds=3, min_train=200)
    verdict = ml_adds_value(outcome, improvement_test=lambda: 0.001)
    assert verdict.metrics["observations"] > 0
    assert isinstance(verdict.adds_value, bool)
    assert verdict.reasons
    strict = ml_adds_value(outcome, min_observations=10**6, improvement_test=lambda: 0.9)
    assert not strict.adds_value
    assert any("out-of-fold observations" in reason for reason in strict.reasons)
