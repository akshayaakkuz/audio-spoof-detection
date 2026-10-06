"""Training and evaluation for the spoof-detection baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import read_manifest
from .features import FEATURE_NAMES, features_from_file


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute EER using the closest point between FPR and FNR."""
    false_positive, true_positive, _ = roc_curve(labels, scores)
    false_negative = 1 - true_positive
    index = np.nanargmin(np.abs(false_positive - false_negative))
    return float((false_positive[index] + false_negative[index]) / 2)


def _matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack([features_from_file(row["path"]) for row in rows])
    labels = np.asarray([int(row["label"].lower() in {"spoof", "fake", "1"}) for row in rows])
    return features, labels


def train(manifest: str | Path, output: str | Path, seed: int = 42) -> dict[str, object]:
    rows = read_manifest(manifest)
    train_rows = [row for row in rows if row.get("split", "train") == "train"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        raise ValueError("Manifest requires a test split for honest evaluation")
    x_train, y_train = _matrix(train_rows)
    x_test, y_test = _matrix(test_rows)

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, random_state=seed))]
    )
    pipeline.fit(x_train, y_train)
    scores = pipeline.predict_proba(x_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "eer": equal_error_rate(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
    }
    payload = {"pipeline": pipeline, "feature_names": FEATURE_NAMES, "metrics": metrics}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination)
    return metrics


def predict(model_path: str | Path, audio_path: str | Path) -> dict[str, object]:
    payload = joblib.load(model_path)
    features = features_from_file(audio_path).reshape(1, -1)
    probability = float(payload["pipeline"].predict_proba(features)[0, 1])
    return {"label": "spoof" if probability >= 0.5 else "bonafide", "spoof_probability": probability}

