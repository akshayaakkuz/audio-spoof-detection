# Audio Spoof Detection

A reproducible machine-learning baseline for distinguishing bona fide speech from
synthetic or replayed audio. The project turns short WAV recordings into interpretable
spectral features and trains a regularized logistic-regression classifier. It reports
accuracy, ROC-AUC, equal error rate (EER), and a confusion matrix on a held-out test split.

This repository focuses on the part that is often missing from ML demos: a clean data
contract, leakage-aware evaluation, deterministic experiments, and an inference command.

## What this demonstrates

- Audio preprocessing and sample-rate normalization
- Frame-level spectral feature engineering with NumPy and SciPy
- Reproducible training with a scikit-learn pipeline
- Security-relevant evaluation using EER in addition to accuracy
- A command-line interface and an automated end-to-end test

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"

spoof-detect make-demo --output data/demo
spoof-detect train --manifest data/demo/manifest.csv --output artifacts/model.joblib
spoof-detect predict --model artifacts/model.joblib --audio data/demo/audio/bonafide_000.wav
pytest
```

The generated data is deliberately easy and exists only to prove that the pipeline works.
Its score is **not a research result**. For a meaningful experiment, use ASVspoof or
another public corpus and retain its official train/development/evaluation partition.

## Dataset format

Training expects a CSV manifest. Paths may be absolute or relative to the manifest.

```csv
path,label,split
audio/train_001.wav,bonafide,train
audio/train_002.wav,spoof,train
audio/test_001.wav,bonafide,test
audio/test_002.wav,spoof,test
```

Accepted spoof labels are `spoof`, `fake`, and `1`; other labels are treated as bona fide.
Only uncompressed WAV input is used so decoding behavior remains explicit and reproducible.

## Method

Each recording is converted to mono, resampled to 16 kHz, and divided into 25 ms frames
with a 10 ms hop. The extractor summarizes six signals that expose common synthesis and
replay artifacts: energy, zero-crossing rate, spectral centroid, spectral roll-off,
spectral flatness, and the ratio of energy above and below 4 kHz. Mean and standard
deviation statistics plus duration produce 13 features per recording.

The classifier is logistic regression after standardization. This transparent baseline
is intentionally small: its coefficients can be inspected, it trains on a CPU, and it
provides a credible reference point before adding a neural audio encoder.

## Evaluation protocol

The `test` partition is never used during fitting. The training command reports:

| Metric | Why it is included |
|---|---|
| ROC-AUC | Threshold-independent ranking quality |
| EER | Operating point where false accepts and false rejects are equal |
| Accuracy | Easy-to-read result at a fixed 0.5 threshold |
| Confusion matrix | Shows the direction of errors |

For publishable results, preserve speaker-disjoint official splits, report uncertainty
across seeds, and evaluate unseen attacks separately. Accuracy alone can hide class
imbalance and poor security behavior.

## Limitations and next experiments

Hand-crafted features may detect recording conditions rather than synthesis itself and
will not generalize reliably to unseen codecs, speakers, or generators. The demo generator
contains obvious artifacts and must never be cited as evidence of real-world performance.
The next useful experiment is to compare this baseline against a pretrained speech model
such as wav2vec 2.0 using the same split and to publish per-attack results.

## Repository layout

```text
src/audio_spoof/   feature extraction, data utilities, model, and CLI
tests/             unit and end-to-end pipeline tests
configs/           documented experiment defaults
```

## License

MIT. Dataset licenses remain with their respective owners.
