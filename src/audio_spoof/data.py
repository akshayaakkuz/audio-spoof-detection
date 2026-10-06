"""Dataset manifest and deterministic demo-data utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read CSV columns: path,label,split (split is optional)."""
    manifest = Path(path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest must contain path and label columns")
    for row in rows:
        audio_path = Path(row["path"])
        if not audio_path.is_absolute():
            row["path"] = str((manifest.parent / audio_path).resolve())
    return rows


def make_demo_dataset(output: str | Path, samples_per_class: int = 60, seed: int = 42) -> Path:
    """Create a small deterministic dataset for pipeline checks, not benchmarking."""
    root = Path(output)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sample_rate = 16_000
    rows: list[tuple[str, str, str]] = []

    for label in ("bonafide", "spoof"):
        for index in range(samples_per_class):
            duration = rng.uniform(0.8, 1.4)
            time = np.arange(int(duration * sample_rate)) / sample_rate
            fundamental = rng.uniform(95, 240)
            envelope = np.minimum(time * 8, 1) * np.minimum((duration - time) * 8, 1)
            signal = sum(
                (1 / harmonic) * np.sin(2 * np.pi * fundamental * harmonic * time)
                for harmonic in range(1, 6)
            )
            signal *= envelope
            if label == "bonafide":
                signal += rng.normal(0, 0.025, signal.size)
            else:
                # Simulate common vocoder clues: periodic high-frequency energy and quantization.
                signal += 0.08 * np.sin(2 * np.pi * 5_200 * time)
                signal = np.round(signal * 32) / 32
            signal = 0.8 * signal / (np.max(np.abs(signal)) + 1e-12)
            filename = f"{label}_{index:03d}.wav"
            wavfile.write(audio_dir / filename, sample_rate, (signal * 32767).astype(np.int16))
            split = "test" if index % 5 == 0 else "train"
            rows.append((f"audio/{filename}", label, split))

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("path", "label", "split"))
        writer.writerows(rows)
    return manifest

