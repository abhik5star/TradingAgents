"""Machine learning — only where training code, data and evaluation actually exist.

Three different things are kept separate on purpose, and this module implements
exactly one of them:

* an **LLM** may propose research ideas (see
  :class:`tradingagents.research.hypothesis.IdeaProvider`), and never trains or
  approves anything;
* the **strategy engine** executes declared rules
  (:mod:`tradingagents.research.strategies`);
* an **ML model** is trained here, on historical features and labels, evaluated
  out of fold, registered with its coefficients and metrics, and only used through
  :meth:`ModelRegistry.predict`.

Everything is NumPy. There is no scikit-learn dependency and no hidden
auto-tuning: logistic regression with L2, k-means for regime discovery, rank
importance, binned calibration, population-stability drift. That is enough to
answer the only question that matters — *does a model add measurable value over
the rule-based baseline?* — and small enough that the result is reproducible from
the stored artifact.

Labels use the future by definition (a label is an outcome). They are never used
as a feature: :func:`build_dataset` drops the unlabelled tail and
:func:`time_series_folds` purges ``horizon`` rows between train and test, so the
last training label cannot overlap the first test bar.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.research.contracts import jsonable, stable_hash, utc_now
from tradingagents.research.features import FeatureTable
from tradingagents.research.store import ResearchStore, new_id


class InsufficientDataError(RuntimeError):
    """Raised when a model would be trained on too little evidence to evaluate."""


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------


@dataclass
class Dataset:
    """Features, labels and the bookkeeping that makes a split honest."""

    X: pd.DataFrame
    y: pd.Series
    horizon: int
    feature_version: str
    description: str = ""
    label_kind: str = "direction"

    def __len__(self) -> int:
        return len(self.X)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"rows": len(self.X), "features": list(self.X.columns),
                            "horizon": self.horizon, "version": self.feature_version})[:16]

    def split(self, train_fraction: float = 0.7, purge: int | None = None
              ) -> tuple[Dataset, Dataset]:
        """Chronological train/test split with a purge gap of one label horizon."""
        purge = self.horizon if purge is None else purge
        cut = int(len(self) * train_fraction)
        train_end = cut - purge
        if train_end < 50 or len(self) - cut < 20:
            raise InsufficientDataError(
                f"{len(self)} labelled rows cannot support a {train_fraction:.0%} split "
                f"with a {purge}-row purge")
        return (Dataset(self.X.iloc[:train_end], self.y.iloc[:train_end], self.horizon,
                        self.feature_version, self.description, self.label_kind),
                Dataset(self.X.iloc[cut:], self.y.iloc[cut:], self.horizon,
                        self.feature_version, self.description, self.label_kind))

    def summary(self) -> dict[str, Any]:
        return {"rows": len(self), "features": len(self.X.columns), "horizon": self.horizon,
                "label_kind": self.label_kind, "positive_share": round(float(self.y.mean()), 4),
                "feature_version": self.feature_version, "fingerprint": self.fingerprint}


#: Columns that are outcomes or identifiers rather than predictors.
_NON_PREDICTIVE = ("regime", "bar_index", "price", "hour_utc")


def build_dataset(table: FeatureTable, horizon: int = 5, *,
                  feature_columns: list[str] | None = None,
                  label_kind: str = "direction",
                  min_rows: int = 200) -> Dataset:
    """Build a supervised dataset from a feature table.

    ``direction`` labels are 1 when the close ``horizon`` bars later is higher than
    the current close, 0 otherwise. Rows whose label is not yet available (the last
    ``horizon`` bars) are dropped, and non-predictive columns such as the regime
    label and the bar index are excluded by default so the model cannot memorise a
    timestamp.
    """
    if horizon < 1:
        raise ValueError("a label horizon must be at least one bar")
    close = table.bars["close"]
    forward = close.shift(-horizon) / close - 1
    # The tail has no outcome yet. It must be dropped *before* thresholding, or
    # every unlabelled bar silently becomes a "down" example.
    usable = forward.notna() & np.isfinite(forward)
    if label_kind == "direction":
        labels = (forward > 0).astype(float).where(usable)
    elif label_kind == "return":
        labels = forward
    else:
        raise ValueError(f"unknown label kind {label_kind!r}; expected direction or return")

    columns = feature_columns or [c for c in table.features.columns
                                  if c not in _NON_PREDICTIVE]
    frame = table.features[columns].copy()
    frame, labels = frame[usable], labels[usable]
    if len(frame) < min_rows:
        raise InsufficientDataError(
            f"{len(frame)} labelled rows is below the {min_rows} needed to train and "
            "evaluate a model")
    # A feature that never varies carries no information and breaks standardisation.
    varying = [c for c in frame.columns if frame[c].notna().sum() >= len(frame) * 0.5
               and frame[c].nunique(dropna=True) > 1]
    if not varying:
        raise InsufficientDataError("no feature varies enough to train on")
    return Dataset(frame[varying], labels.astype(float), horizon, table.version,
                   description=f"{table.ticker} {table.timeframe}", label_kind=label_kind)


def time_series_folds(n: int, folds: int = 4, purge: int = 5,
                      min_train: int = 100) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with a purge gap between train and test indices."""
    if folds < 2:
        raise ValueError("at least two folds are needed to evaluate a model")
    usable = n - min_train - purge
    if usable < folds * 10:
        raise InsufficientDataError(
            f"{n} rows cannot fill {folds} folds with a {min_train}-row minimum train window")
    step = usable // folds
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(folds):
        train_end = min_train + fold * step
        test_start = train_end + purge
        test_end = min(n, test_start + step)
        if test_start >= test_end:
            continue
        splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


class LogisticRegression:
    """L2-regularised logistic regression in NumPy, with stored preprocessing."""

    def __init__(self, learning_rate: float = 0.2, epochs: int = 600, l2: float = 1e-3,
                 seed: int = 0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.coefficients: np.ndarray | None = None
        self.intercept: float = 0.0
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.medians: np.ndarray | None = None
        self.features: list[str] = []

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        values = X.to_numpy(dtype=float)
        if fit:
            self.features = list(X.columns)
            with warnings.catch_warnings():
                # An all-NaN column is a data-shape fact, not a numerical accident:
                # it is median-filled with zero and carries no information.
                warnings.simplefilter("ignore", RuntimeWarning)
                medians = np.nanmedian(values, axis=0)
            self.medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isnan(values), self.medians, values)
        if fit:
            self.mean = values.mean(axis=0)
            self.scale = values.std(axis=0)
            self.scale = np.where(self.scale > 0, self.scale, 1.0)
        return (values - self.mean) / self.scale

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
        values = self._prepare(X, fit=True)
        target = y.to_numpy(dtype=float)
        if len(np.unique(target)) < 2:
            raise InsufficientDataError("the training labels contain only one class")
        weights = np.zeros(values.shape[1])
        intercept = 0.0
        n = len(values)
        rng = np.random.default_rng(self.seed)
        weights = rng.normal(0, 0.001, size=values.shape[1])
        for _ in range(self.epochs):
            linear = values @ weights + intercept
            probability = 1 / (1 + np.exp(-np.clip(linear, -30, 30)))
            error = probability - target
            weights -= self.learning_rate * (values.T @ error / n + self.l2 * weights)
            intercept -= self.learning_rate * float(error.mean())
        self.coefficients, self.intercept = weights, float(intercept)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.coefficients is None:
            raise RuntimeError("the model is not fitted")
        values = self._prepare(X)
        return 1 / (1 + np.exp(-np.clip(values @ self.coefficients + self.intercept, -30, 30)))

    def importances(self) -> dict[str, float]:
        if self.coefficients is None:
            return {}
        return dict(sorted(zip(self.features, np.round(self.coefficients, 6), strict=True),
                           key=lambda item: abs(item[1]), reverse=True))

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "logistic_regression", "features": self.features,
                "coefficients": self.coefficients.tolist() if self.coefficients is not None else None,
                "intercept": self.intercept,
                "mean": self.mean.tolist() if self.mean is not None else None,
                "scale": self.scale.tolist() if self.scale is not None else None,
                "medians": self.medians.tolist() if self.medians is not None else None,
                "hyperparameters": {"learning_rate": self.learning_rate, "epochs": self.epochs,
                                    "l2": self.l2, "seed": self.seed}}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LogisticRegression:
        params = payload.get("hyperparameters", {})
        model = cls(**{k: params[k] for k in ("learning_rate", "epochs", "l2", "seed")
                       if k in params})
        model.features = list(payload["features"])
        model.coefficients = np.asarray(payload["coefficients"], dtype=float)
        model.intercept = float(payload["intercept"])
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.scale = np.asarray(payload["scale"], dtype=float)
        model.medians = np.asarray(payload["medians"], dtype=float)
        return model


class KMeans:
    """Deterministic k-means for regime discovery, with a seeded k-means++ start."""

    def __init__(self, clusters: int = 3, iterations: int = 100, seed: int = 0):
        self.clusters = clusters
        self.iterations = iterations
        self.seed = seed
        self.centres: np.ndarray | None = None
        self.features: list[str] = []
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        values = X.to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0)
        if fit:
            self.features = list(X.columns)
            self.mean = values.mean(axis=0)
            self.scale = np.where(values.std(axis=0) > 0, values.std(axis=0), 1.0)
        return (values - self.mean) / self.scale

    def fit(self, X: pd.DataFrame) -> KMeans:
        values = self._prepare(X, fit=True)
        rng = np.random.default_rng(self.seed)
        centres = [values[rng.integers(len(values))]]
        for _ in range(1, self.clusters):
            distances = np.min([((values - centre) ** 2).sum(axis=1) for centre in centres], axis=0)
            total = distances.sum()
            if total <= 0:
                centres.append(values[rng.integers(len(values))])
                continue
            centres.append(values[rng.choice(len(values), p=distances / total)])
        centres = np.asarray(centres)
        for _ in range(self.iterations):
            labels = ((values[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
            updated = np.array([values[labels == k].mean(axis=0) if (labels == k).any()
                                else centres[k] for k in range(self.clusters)])
            if np.allclose(updated, centres):
                break
            centres = updated
        self.centres = centres
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, prepared: bool = False) -> np.ndarray:
        values = X if prepared else self._prepare(X)
        distances = ((values[:, None, :] - self.centres[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "kmeans", "features": self.features,
                "centres": self.centres.tolist() if self.centres is not None else None,
                "mean": self.mean.tolist() if self.mean is not None else None,
                "scale": self.scale.tolist() if self.scale is not None else None,
                "clusters": self.clusters, "seed": self.seed}


# --------------------------------------------------------------------------------------
# Evaluation: discrimination, calibration, importance
# --------------------------------------------------------------------------------------


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUC (Mann-Whitney U), 0.5 for no discrimination."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positives, negatives = scores[labels > 0.5], scores[labels <= 0.5]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    order = pd.Series(np.concatenate([positives, negatives])).rank().to_numpy()
    rank_sum = order[: len(positives)].sum()
    return float((rank_sum - len(positives) * (len(positives) + 1) / 2)
                 / (len(positives) * len(negatives)))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels)) ** 2))


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc_significance(probabilities: np.ndarray, labels: np.ndarray, *,
                     permutations: int = 200, seed: int = 0) -> dict[str, Any]:
    """Permutation test for "is this AUC better than chance?".

    The labels are shuffled ``permutations`` times and the observed AUC is compared
    with the shuffled distribution. That answers a narrow, honest question — does
    the ranking carry information — and says nothing about profitability: a
    discriminating model can still lose money after costs.
    """
    scores = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=float)
    observed = auc_score(scores, target)
    if observed is None:
        return {"auc": None, "p_value": 1.0, "permutations": 0,
                "note": "one class in the labels; no discrimination can be measured"}
    rng = np.random.default_rng(seed)
    at_least = 0
    for _ in range(permutations):
        shuffled = rng.permutation(target)
        chance = auc_score(scores, shuffled)
        if chance is not None and chance >= observed:
            at_least += 1
    return {"auc": observed, "p_value": (at_least + 1) / (permutations + 1),
            "permutations": permutations,
            "note": "a permutation p-value on ranking, not a claim of profit"}


def calibration_table(probabilities: np.ndarray, labels: np.ndarray,
                      bins: int = 10) -> list[dict[str, float]]:
    """Reliability: predicted probability versus observed frequency per bin.

    A model whose 70% bucket wins 52% of the time is not calibrated, and its
    probability must not be used for sizing.
    """
    frame = pd.DataFrame({"p": np.asarray(probabilities, dtype=float),
                          "y": np.asarray(labels, dtype=float)})
    frame["bin"] = pd.cut(frame["p"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    rows: list[dict[str, float]] = []
    for interval, group in frame.groupby("bin", observed=True):
        rows.append({"bin_low": round(float(interval.left), 3),
                     "bin_high": round(float(interval.right), 3),
                     "n": int(len(group)),
                     "predicted": round(float(group["p"].mean()), 4),
                     "observed": round(float(group["y"].mean()), 4)})
    return rows


def evaluation_report(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Discrimination and calibration together: one without the other is misleading."""
    calibration = calibration_table(probabilities, labels)
    weighted_error = sum(abs(row["predicted"] - row["observed"]) * row["n"]
                         for row in calibration) / max(1, sum(row["n"] for row in calibration))
    return {"auc": auc_score(probabilities, labels), "brier": round(brier_score(
        probabilities, labels), 6), "log_loss": round(log_loss(probabilities, labels), 6),
        "calibration_error": round(float(weighted_error), 4),
        "calibration": calibration, "observations": int(len(labels)),
        "base_rate": round(float(np.mean(labels)), 4)}


def permutation_importance(model: Any, X: pd.DataFrame, y: pd.Series,
                           n_repeats: int = 5, seed: int = 3) -> dict[str, float]:
    """AUC lost when each feature is shuffled. A negative value means noise."""
    base = auc_score(model.predict_proba(X), y.to_numpy()) or 0.5
    rng = np.random.default_rng(seed)
    importance: dict[str, float] = {}
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            score = auc_score(model.predict_proba(shuffled), y.to_numpy()) or 0.5
            drops.append(base - score)
        importance[column] = round(float(np.mean(drops)), 6)
    return dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI between a training distribution and a live one (>0.2 is usually drift)."""
    left = pd.Series(expected).dropna().to_numpy(dtype=float)
    right = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(left) < 10 or len(right) < 10:
        return 0.0
    edges = np.quantile(left, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    left_counts = np.histogram(left, bins=edges)[0] / len(left)
    right_counts = np.histogram(right, bins=edges)[0] / len(right)
    left_counts = np.clip(left_counts, 1e-6, None)
    right_counts = np.clip(right_counts, 1e-6, None)
    return float(np.sum((right_counts - left_counts) * np.log(right_counts / left_counts)))


@dataclass
class DriftReport:
    """Per-feature drift and the verdict a run needs to decide whether to trade."""

    psi: dict[str, float]
    max_psi: float
    drifted: list[str]
    threshold: float = 0.2

    @property
    def alert(self) -> bool:
        return bool(self.drifted)

    def as_dict(self) -> dict[str, Any]:
        return {"psi": {k: round(v, 4) for k, v in self.psi.items()},
                "max_psi": round(self.max_psi, 4), "drifted": self.drifted,
                "threshold": self.threshold, "alert": self.alert}


def detect_drift(training: pd.DataFrame, recent: pd.DataFrame, threshold: float = 0.2,
                 max_features: int = 30) -> DriftReport:
    """Compare the training feature distribution with the most recent window."""
    columns = [c for c in training.columns if c in recent.columns][:max_features]
    psi = {column: population_stability_index(training[column], recent[column])
           for column in columns}
    drifted = sorted([column for column, value in psi.items() if value >= threshold],
                     key=lambda column: -psi[column])
    return DriftReport(psi=psi, max_psi=max(psi.values(), default=0.0), drifted=drifted,
                       threshold=threshold)


# --------------------------------------------------------------------------------------
# Training, out-of-fold predictions and the model registry
# --------------------------------------------------------------------------------------


@dataclass
class TrainingOutcome:
    """A trained model with the out-of-fold evidence that justifies (or blocks) its use."""

    model: LogisticRegression
    dataset: Dataset
    oof_probabilities: pd.Series
    oof_metrics: dict[str, Any]
    in_sample_metrics: dict[str, Any]
    importances: dict[str, float]
    folds: int

    def as_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset.summary(), "oof": self.oof_metrics,
                "in_sample": self.in_sample_metrics, "folds": self.folds,
                "importance": dict(list(self.importances.items())[:10])}


def train_with_oof(dataset: Dataset, folds: int = 4, min_train: int = 120,
                   seed: int = 0, **model_kwargs: Any) -> TrainingOutcome:
    """Walk-forward training: each fold predicts only on rows its model never saw.

    The concatenated out-of-fold probabilities are the *only* predictions allowed
    to be used as a feature or a filter. In-sample predictions are reported for
    contrast and never used.
    """
    splits = time_series_folds(len(dataset), folds=folds, purge=dataset.horizon,
                               min_train=min_train)
    if not splits:
        raise InsufficientDataError("no walk-forward fold could be built")
    probabilities = pd.Series(np.nan, index=dataset.X.index, dtype=float)
    for train_index, test_index in splits:
        model = LogisticRegression(seed=seed, **model_kwargs)
        model.fit(dataset.X.iloc[train_index], dataset.y.iloc[train_index])
        probabilities.iloc[test_index] = model.predict_proba(dataset.X.iloc[test_index])
    evaluated = probabilities.notna()
    y_evaluated = dataset.y[evaluated]
    oof_metrics = evaluation_report(probabilities[evaluated].to_numpy(), y_evaluated.to_numpy())
    final = LogisticRegression(seed=seed, **model_kwargs)
    final.fit(dataset.X, dataset.y)
    in_sample = evaluation_report(final.predict_proba(dataset.X), dataset.y.to_numpy())
    return TrainingOutcome(model=final, dataset=dataset, oof_probabilities=probabilities,
                           oof_metrics=oof_metrics, in_sample_metrics=in_sample,
                           importances=permutation_importance(final, dataset.X, dataset.y),
                           folds=len(splits))


class ModelRegistry:
    """Stored models with a champion pointer, drift checks and rollback."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def _document(self) -> dict[str, Any]:
        document = self.store.load_models()
        document.setdefault("models", [])
        return document

    def models(self) -> list[dict[str, Any]]:
        return list(self._document()["models"])

    def champion(self) -> dict[str, Any] | None:
        name = self._document().get("champion")
        return next((m for m in self.models() if m["model_id"] == name), None) if name else None

    def register(self, outcome: TrainingOutcome, *, status: str = "candidate") -> dict[str, Any]:
        """Store a model artifact and its evaluation. Never overwrites an earlier model."""
        document = self._document()
        model_id = new_id("MDL")
        record = {
            "model_id": model_id, "kind": "logistic_regression", "status": status,
            "created_at": utc_now(), "artifact": outcome.model.as_dict(),
            "dataset": outcome.dataset.summary(), "oof": outcome.oof_metrics,
            "in_sample": outcome.in_sample_metrics,
            "importance": dict(list(outcome.importances.items())[:15]),
            "folds": outcome.folds,
            "feature_version": outcome.dataset.feature_version,
        }
        document["models"].append(record)
        # Registration is not promotion: a fresh artifact sits as a candidate until
        # `promote` has checked its out-of-fold discrimination on the stored evidence.
        self.store.save_models(document)
        self.store.append("ml_event", {"id": new_id("MLE"), "kind": "model_registered",
                                       "model_id": model_id,
                                       "oof_auc": outcome.oof_metrics.get("auc")})
        return record

    def promote(self, model_id: str, *, reason: str) -> dict[str, Any]:
        """Point the champion at ``model_id`` only if its out-of-fold AUC beats 0.5."""
        document = self._document()
        record = next((m for m in document["models"] if m["model_id"] == model_id), None)
        if record is None:
            raise ValueError(f"no model {model_id!r}")
        auc = (record.get("oof") or {}).get("auc")
        if auc is None or auc <= 0.5:
            raise ValueError(
                f"model {model_id} has out-of-fold AUC {auc}; a model that does not "
                "discriminate cannot become champion")
        previous = document.get("champion")
        document["champion"] = model_id
        record["status"] = "champion"
        for other in document["models"]:
            if other["model_id"] == previous and previous != model_id:
                other["status"] = "retired"
        self.store.save_models(document)
        self.store.append("ml_event", {"id": new_id("MLE"), "kind": "model_promoted",
                                       "model_id": model_id, "from": previous,
                                       "reason": reason})
        return record

    def rollback(self, *, reason: str) -> dict[str, Any] | None:
        """Return the champion pointer to the previous non-drifted model."""
        document = self._document()
        current = document.get("champion")
        candidates = [m for m in document["models"]
                      if m["model_id"] != current and m.get("status") in ("retired", "candidate")]
        if not candidates:
            return None
        target = candidates[-1]
        document["champion"] = target["model_id"]
        target["status"] = "champion"
        for model in document["models"]:
            if model["model_id"] == current:
                model["status"] = "rolled_back"
        self.store.save_models(document)
        self.store.append("ml_event", {"id": new_id("MLE"), "kind": "model_rolled_back",
                                       "to": target["model_id"], "from": current,
                                       "reason": reason})
        return target

    def load(self, model_id: str | None = None) -> LogisticRegression:
        record = (next((m for m in self.models() if m["model_id"] == model_id), None)
                  if model_id else self.champion())
        if record is None:
            raise ValueError("no such model in the registry")
        return LogisticRegression.from_dict(record["artifact"])

    def predict(self, features: pd.DataFrame, model_id: str | None = None) -> pd.Series:
        model = self.load(model_id)
        missing = [c for c in model.features if c not in features.columns]
        if missing:
            raise ValueError(f"the model expects features that are not in the table: {missing}")
        return pd.Series(model.predict_proba(features[model.features]), index=features.index,
                         name="ml_probability")

    def drift(self, features: pd.DataFrame, window: int = 200,
              threshold: float = 0.2) -> DriftReport | None:
        """Drift of the champion's inputs against its training distribution.

        The stored medians are not a distribution, so drift compares the recent
        window with the earlier part of the same table: a change *within* the
        live series is the signal that the model's world has moved.
        """
        record = self.champion()
        if record is None or len(features) < window * 2:
            return None
        columns = [c for c in record["artifact"]["features"] if c in features.columns]
        earlier = features[columns].iloc[-2 * window:-window]
        recent = features[columns].iloc[-window:]
        return detect_drift(earlier, recent, threshold=threshold)

    def should_retrain(self, feature_version: str, drift: DriftReport | None = None,
                       new_rows: int = 0, days_since_training: float | None = None,
                       max_days: float = 90.0, min_new_rows: int = 250) -> dict[str, Any]:
        """Whether a retrain is due, and the reason. Schedule plus evidence, not a guess."""
        reasons: list[str] = []
        champion = self.champion()
        if champion is None:
            reasons.append("no champion model is registered")
        if champion is not None and champion.get("feature_version") != feature_version:
            reasons.append("the feature contract changed since training")
        if drift is not None and drift.alert:
            reasons.append(f"drift detected in {len(drift.drifted)} feature(s): "
                           f"max PSI {drift.max_psi:.2f}")
        if days_since_training is not None and days_since_training >= max_days:
            reasons.append(f"{days_since_training:.0f} days since the last training run")
        if new_rows >= min_new_rows:
            reasons.append(f"{new_rows} new labelled rows are available")
        return {"retrain": bool(reasons), "reasons": reasons,
                "champion": champion["model_id"] if champion else None}


@dataclass
class MLVerdict:
    """Whether the model earned the right to filter trades."""

    adds_value: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return jsonable({"adds_value": self.adds_value, "reasons": self.reasons,
                         "metrics": self.metrics})


def ml_adds_value(outcome: TrainingOutcome, *, min_auc: float = 0.53,
                  min_observations: int = 300,
                  improvement_test: Callable[[], float] | None = None,
                  max_p_value: float = 0.05) -> MLVerdict:
    """Decide whether a trained model is worth using, on evidence from out-of-fold data.

    Requirements: enough out-of-fold observations, AUC above a floor that is barely
    better than chance *and* calibrated probabilities, plus a significance test
    supplied by the caller (usually a permutation test of filtered versus unfiltered
    trade outcomes). A model that fails any of these is reported as not adding value
    and is not used, even if its in-sample numbers were beautiful.
    """
    reasons: list[str] = []
    metrics = outcome.oof_metrics
    observations = int(metrics.get("observations", 0))
    auc = metrics.get("auc")
    if observations < min_observations:
        reasons.append(f"only {observations} out-of-fold observations")
    if auc is None or auc < min_auc:
        reasons.append(f"out-of-fold AUC {auc} is below the {min_auc} floor")
    if metrics.get("calibration_error", 1.0) > 0.10:
        reasons.append(f"calibration error {metrics['calibration_error']:.2f} is too large "
                       "to size positions from these probabilities")
    p_value = improvement_test() if improvement_test is not None else 1.0
    metrics["improvement_p_value"] = p_value
    if p_value > max_p_value:
        reasons.append(f"the improvement over the rule baseline is not significant "
                       f"(p={p_value:.3f})")
    return MLVerdict(adds_value=not reasons, reasons=reasons or ["the model adds measurable value"],
                     metrics={"auc": auc, "observations": observations,
                              "p_value": p_value})
