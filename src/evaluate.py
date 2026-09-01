"""Evaluate patient-level laboratory-category recommendations.

This supports mixed cohorts that include visits with no gold laboratory category.
Gold visits define the evaluation cohort. Every gold visit must have exactly one prediction row; missing rows, duplicate IDs, and prediction IDs absent from the gold cohort are rejected.
Unknown prediction or gold categories always stop evaluation.
It reports:
- Laboratory-category set metrics on lab-positive gold visits.
- Overall set metrics across all visits, with no-lab represented by an all-zero vector over the 21 laboratory-category labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from common import (
    DEFAULT_TEST_TAG_LIST_PATH,
    GOLD_TEST_TAG_COLUMN,
    RECOMMENDED_TAG_COLUMN,
    VISIT_ID_COLUMN,
    clean_text,
    load_allowed_tags,
    split_terms,
)

VALIDATION_REPORT_COLUMNS = [
    "source",
    "issue_type",
    "visit_row_id",
    "unknown_test_tag",
]


def validation_record(
    source: str,
    issue_type: str,
    unknown_test_tag: str,
    *,
    visit_row_id: str = "",
) -> dict[str, str]:
    return {
        "source": source,
        "issue_type": issue_type,
        "visit_row_id": visit_row_id,
        "unknown_test_tag": unknown_test_tag,
    }


def write_validation_report(path: Path | None, records: list[dict[str, str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=VALIDATION_REPORT_COLUMNS).to_csv(path, index=False)


def gold_rows(
    gold_path: Path,
    allowed_tags: list[str],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    gold = pd.read_csv(gold_path, dtype="string", keep_default_na=False)
    required = {VISIT_ID_COLUMN, GOLD_TEST_TAG_COLUMN}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"{gold_path} is missing required gold column(s): {sorted(missing)}")
    gold[VISIT_ID_COLUMN] = gold[VISIT_ID_COLUMN].map(clean_text)
    rows = []
    validation_records: list[dict[str, str]] = []
    allowed = set(allowed_tags)
    for _, row in gold.iterrows():
        visit_row_id = clean_text(row.get(VISIT_ID_COLUMN))
        raw_tags = split_terms(row.get(GOLD_TEST_TAG_COLUMN))
        tags = [tag for tag in raw_tags if tag in allowed]
        for tag in raw_tags:
            if tag not in allowed:
                validation_records.append(
                    validation_record(
                        "gold",
                        "unknown_gold_tag",
                        tag,
                        visit_row_id=visit_row_id,
                    )
                )
        rows.append(
            {
                VISIT_ID_COLUMN: visit_row_id,
                "gold_tags": tags,
            }
        )
    return pd.DataFrame(rows), validation_records


def validate_visit_ids(frame: pd.DataFrame, path: Path, source_name: str) -> None:
    empty_count = int(frame[VISIT_ID_COLUMN].eq("").sum())
    if empty_count:
        raise ValueError(f"{path} contains {empty_count} empty {source_name} visit_row_id value(s).")
    duplicate_mask = frame[VISIT_ID_COLUMN].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = frame.loc[duplicate_mask, VISIT_ID_COLUMN].drop_duplicates().tolist()
        preview = ", ".join(duplicate_ids[:10])
        suffix = " ..." if len(duplicate_ids) > 10 else ""
        raise ValueError(
            f"{path} contains {len(duplicate_ids)} duplicate {source_name} visit_row_id value(s): "
            f"{preview}{suffix}"
        )


def set_metrics(predicted: set[str], gold: set[str], universe: set[str]) -> dict[str, Any]:
    tp = predicted & gold
    fp = predicted - gold
    fn = gold - predicted
    if not predicted and not gold:
        precision = recall = f1 = 1.0
    else:
        precision = len(tp) / len(predicted) if predicted else 0.0
        recall = len(tp) / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hamming_loss = (len(fp) + len(fn)) / len(universe) if universe else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hamming_loss": hamming_loss,
        "exact_match": predicted == gold,
    }


def summarize_patient_metrics(
    per_visit: pd.DataFrame,
    prefix: str,
) -> dict[str, Any]:
    """Average each visit's set-based metrics so every patient has equal weight."""
    if per_visit.empty:
        return {
            "patient_macro_precision": 0.0,
            "patient_macro_recall": 0.0,
            "patient_macro_f1": 0.0,
            "exact_match": 0.0,
            "hamming_loss": 0.0,
        }
    return {
        "patient_macro_precision": float(per_visit[f"{prefix}_precision"].mean()),
        "patient_macro_recall": float(per_visit[f"{prefix}_recall"].mean()),
        "patient_macro_f1": float(per_visit[f"{prefix}_f1"].mean()),
        "exact_match": float(per_visit[f"{prefix}_exact_match"].mean()),
        "hamming_loss": float(per_visit[f"{prefix}_hamming_loss"].mean()),
    }


def format_validation_error(
    records: list[dict[str, str]],
    report_path: Path | None,
) -> str:
    counts = Counter(record["issue_type"] for record in records)
    count_text = ", ".join(
        f"{issue_type}={count}" for issue_type, count in sorted(counts.items())
    )
    examples = []
    seen = set()
    for record in records:
        key = (record["issue_type"], record["visit_row_id"], record["unknown_test_tag"])
        if key in seen:
            continue
        seen.add(key)
        location = (
            f"visit_row_id={record['visit_row_id']}"
            if record["visit_row_id"]
            else record["source"]
        )
        examples.append(f"{record['unknown_test_tag']!r} ({location})")
        if len(examples) == 10:
            break
    message = f"Evaluation validation failed ({count_text}). Examples: {', '.join(examples)}."
    if report_path is not None:
        message += f" See {report_path}."
    return message


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_report_path = getattr(args, "validation_report_path", None)

    allowed_tags = load_allowed_tags(args.test_tag_list_path)
    allowed_tag_set = set(allowed_tags)
    pred = pd.read_csv(args.prediction_path, dtype="string", keep_default_na=False)
    required_prediction_columns = {VISIT_ID_COLUMN, RECOMMENDED_TAG_COLUMN}
    missing_prediction_columns = required_prediction_columns - set(pred.columns)
    if missing_prediction_columns:
        raise ValueError(
            f"{args.prediction_path} is missing required prediction column(s): "
            f"{sorted(missing_prediction_columns)}"
        )
    pred[VISIT_ID_COLUMN] = pred[VISIT_ID_COLUMN].map(clean_text)
    gold, gold_validation = gold_rows(args.gold_path, allowed_tags)
    validate_visit_ids(pred, args.prediction_path, "prediction")
    validate_visit_ids(gold, args.gold_path, "gold")

    prediction_validation: list[dict[str, str]] = []
    for _, row in pred.iterrows():
        visit_row_id = clean_text(row.get(VISIT_ID_COLUMN))
        for tag in split_terms(row.get(RECOMMENDED_TAG_COLUMN)):
            if tag not in allowed_tag_set:
                prediction_validation.append(
                    validation_record(
                        "prediction",
                        "unknown_predicted_tag",
                        tag,
                        visit_row_id=visit_row_id,
                    )
                )

    pred_ids = set(pred[VISIT_ID_COLUMN])
    gold_ids = set(gold[VISIT_ID_COLUMN])
    extra_prediction_ids = sorted(pred_ids - gold_ids)
    if extra_prediction_ids:
        preview = ", ".join(extra_prediction_ids[:10])
        suffix = " ..." if len(extra_prediction_ids) > 10 else ""
        raise ValueError(
            f"{args.prediction_path} contains {len(extra_prediction_ids)} visit_row_id value(s) absent from "
            f"{args.gold_path}: {preview}{suffix}"
        )

    validation_records = gold_validation + prediction_validation
    write_validation_report(validation_report_path, validation_records)
    if validation_records:
        raise ValueError(format_validation_error(validation_records, validation_report_path))
    if gold.empty:
        raise ValueError("The prepared gold file contains no visits.")

    missing_prediction_ids = sorted(gold_ids - pred_ids)
    if missing_prediction_ids:
        preview = ", ".join(missing_prediction_ids[:10])
        suffix = " ..." if len(missing_prediction_ids) > 10 else ""
        raise ValueError(
            f"{args.prediction_path} is missing prediction rows for "
            f"{len(missing_prediction_ids)} gold visit(s): {preview}{suffix}"
        )

    prediction_columns = [VISIT_ID_COLUMN, RECOMMENDED_TAG_COLUMN]
    pred_for_merge = pred.reindex(columns=prediction_columns).copy()
    merged = gold.merge(pred_for_merge, on=VISIT_ID_COLUMN, how="left", validate="one_to_one", sort=False)

    universe_tags = allowed_tag_set
    per_visit = []
    for _, row in merged.iterrows():
        pred_tags = set(split_terms(row.get(RECOMMENDED_TAG_COLUMN)))
        gold_tags = set(row.get("gold_tags") or [])
        tag_metric = set_metrics(pred_tags, gold_tags, universe_tags)
        overall_metric = set_metrics(pred_tags, gold_tags, universe_tags)
        per_visit.append(
            {
                VISIT_ID_COLUMN: row[VISIT_ID_COLUMN],
                "gold_test_tags": ";".join(sorted(gold_tags)),
                "predicted_test_tags": ";".join(sorted(pred_tags)),
                "tag_precision": tag_metric["precision"],
                "tag_recall": tag_metric["recall"],
                "tag_f1": tag_metric["f1"],
                "tag_hamming_loss": tag_metric["hamming_loss"],
                "tag_exact_match": tag_metric["exact_match"],
                "overall_precision": overall_metric["precision"],
                "overall_recall": overall_metric["recall"],
                "overall_f1": overall_metric["f1"],
                "overall_hamming_loss": overall_metric["hamming_loss"],
                "overall_exact_match": overall_metric["exact_match"],
            }
        )
    per_visit_df = pd.DataFrame(per_visit)

    lab_positive = per_visit_df.loc[per_visit_df["gold_test_tags"].ne("")]
    tag_summary = summarize_patient_metrics(lab_positive, "tag")
    overall_summary = summarize_patient_metrics(per_visit_df, "overall")

    summary = pd.DataFrame(
        [
            {"metric_group": "lab_positive_recommendation", "n": len(lab_positive), **tag_summary},
            {"metric_group": "overall_recommendation", "n": len(per_visit_df), **overall_summary},
        ]
    )
    return summary, per_visit_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--gold-path", type=Path, required=True)
    parser.add_argument("--output-summary-path", type=Path, required=True)
    parser.add_argument("--output-per-visit-path", type=Path, required=True)
    parser.add_argument(
        "--validation-report-path",
        type=Path,
        help="Optional CSV audit written when unknown gold or predicted categories are found.",
    )
    parser.add_argument("--test-tag-list-path", type=Path, default=DEFAULT_TEST_TAG_LIST_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, per_visit = evaluate(args)
    for path in (args.output_summary_path, args.output_per_visit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_path, index=False)
    per_visit.to_csv(args.output_per_visit_path, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        "saved:",
        args.output_summary_path,
        args.output_per_visit_path,
        *([args.validation_report_path] if args.validation_report_path is not None else []),
    )


if __name__ == "__main__":
    main()
