"""Shared utilities for KERNEL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model-ready input field names; users prepare these fields for their own data.
CHIEF_COMPLAINT_COLUMN = "CC"
VISIT_ID_COLUMN = "visit_row_id"
PATIENT_CONTEXT_COLUMN = "patient_full_context"
GOLD_ICD_CODE_COLUMN = "gold_icd_codes"
GOLD_ICD_TITLE_COLUMN = "gold_icd_titles"
GOLD_TEST_TAG_COLUMN = "gold_test_tags"
RECOMMENDED_TAG_COLUMN = "recommended_test_tags"
FINAL_EXPLANATION_COLUMN = "final_explanation"

# Reference vocabulary field names.
TAG_LIST_COLUMN = "Tags_name"

# Stage 1 prediction field names.
PREDICTION_RANK_COLUMN = "prediction_rank"
PREDICTED_ICD_CODE_COLUMN = "predicted_icd_code"
PREDICTED_ICD_TITLE_COLUMN = "predicted_icd_title"


DEFAULT_TEST_TAG_LIST_PATH = PROJECT_ROOT / "data" / "test_tag_list.xlsx"
DEFAULT_ICD_LABEL_PATH = PROJECT_ROOT / "data" / "icd10_titles.csv"

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).strip().split())


def split_terms(value: Any, separators: str = r"[;|]+") -> list[str]:
    text = clean_text(value)
    return [part.strip() for part in re.split(separators, text) if part.strip()] if text else []


def join_unique(values: Iterable[Any], sep: str = ";") -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return sep.join(output)


def load_allowed_tags(path: str | Path) -> list[str]:
    path = Path(path)
    frame = pd.read_excel(path, dtype="string", keep_default_na=False)
    if TAG_LIST_COLUMN not in frame.columns:
        raise ValueError(f"{path} is missing required column: {TAG_LIST_COLUMN}")
    tags = frame[TAG_LIST_COLUMN].map(clean_text).tolist()
    allowed_tags = list(dict.fromkeys(tag for tag in tags if tag))
    if not allowed_tags:
        raise ValueError(f"{path} does not contain any non-empty {TAG_LIST_COLUMN} values.")
    return allowed_tags
