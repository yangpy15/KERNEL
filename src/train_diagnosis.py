"""Train the Stage 1 patient-context-to-ICD cross-encoder used by KERNEL."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import DEFAULT_ICD_LABEL_PATH
from diagnosis_model import train_diagnosis_cross_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--icd-label-path", type=Path, default=DEFAULT_ICD_LABEL_PATH)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--n-train-visits", type=int)
    parser.add_argument("--n-validation-visits", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--device", default="auto", help="PyTorch device such as auto, cuda, cuda:0, or cpu.")
    args = parser.parse_args()
    return args


def main() -> None:
    train_diagnosis_cross_encoder(parse_args())


if __name__ == "__main__":
    main()
