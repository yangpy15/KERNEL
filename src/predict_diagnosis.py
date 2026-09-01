"""Generate Stage 1 ICD predictions for KERNEL inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from diagnosis_model import predict_diagnoses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--n-visits", type=int)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override the checkpoint token length (defaults to the checkpoint value).",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    predict_diagnoses(parse_args())


if __name__ == "__main__":
    main()
