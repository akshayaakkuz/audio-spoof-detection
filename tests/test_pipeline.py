from pathlib import Path

import numpy as np

from audio_spoof.data import make_demo_dataset, read_manifest
from audio_spoof.features import FEATURE_NAMES, extract_features
from audio_spoof.model import predict, train


def test_features_are_finite() -> None:
    rate = 16_000
    time = np.arange(rate) / rate
    features = extract_features(np.sin(2 * np.pi * 220 * time), rate)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_end_to_end(tmp_path: Path) -> None:
    manifest = make_demo_dataset(tmp_path / "demo", samples_per_class=10)
    model_path = tmp_path / "model.joblib"
    metrics = train(manifest, model_path)
    first_audio = read_manifest(manifest)[0]["path"]
    result = predict(model_path, first_audio)
    assert model_path.exists()
    assert metrics["roc_auc"] >= 0.9
    assert 0 <= result["spoof_probability"] <= 1

