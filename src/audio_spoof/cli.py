"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .data import make_demo_dataset
from .model import predict, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio spoof-detection baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="generate a tiny pipeline-check dataset")
    demo.add_argument("--output", default="data/demo")
    demo.add_argument("--samples-per-class", type=int, default=60)
    fit = commands.add_parser("train", help="train and evaluate from a CSV manifest")
    fit.add_argument("--manifest", required=True)
    fit.add_argument("--output", default="artifacts/model.joblib")
    infer = commands.add_parser("predict", help="score one WAV file")
    infer.add_argument("--model", required=True)
    infer.add_argument("--audio", required=True)
    args = parser.parse_args()

    if args.command == "make-demo":
        result = {"manifest": str(make_demo_dataset(args.output, args.samples_per_class))}
    elif args.command == "train":
        result = train(args.manifest, args.output)
    else:
        result = predict(args.model, args.audio)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

