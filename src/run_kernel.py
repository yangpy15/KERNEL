"""KERNEL laboratory-test recommendation mainline.

The workflow includes embedding retrieval and MedlinePlus evidence for the final clinician-facing rationale, with optional external diagnosis-KG expansion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from inference_utils import (
    DEFAULT_ICD2TEST_TAG_PATH,
    RETRIEVAL_MODEL_NAME,
    build_context_rag_evidence_for_visit,
    build_context_rag_index,
    build_context_rag_train_rows,
    build_transformers_caller,
    build_prior_evidence_for_visit,
    build_train_test_tag_priors,
    build_triplets,
    format_candidate_tag_knowledge,
    format_context_rag_evidence,
    format_rag_tag_counts,
    format_ranked_icd,
    format_training_prior_evidence,
    format_working_diagnoses,
    load_icd2test_tag,
    normalize_prediction_columns,
    parse_llm_json,
    parse_or_recover_llm_response,
    prepare_training_evidence_rows,
    retry_prompt_for_invalid_json,
    retry_prompt_for_invalid_tags,
    summarize_candidate_tags,
    summarize_icds,
    validate_recommendations,
)
from common import (
    CHIEF_COMPLAINT_COLUMN,
    DEFAULT_TEST_TAG_LIST_PATH,
    FINAL_EXPLANATION_COLUMN,
    GOLD_ICD_CODE_COLUMN,
    GOLD_ICD_TITLE_COLUMN,
    PATIENT_CONTEXT_COLUMN,
    PROJECT_ROOT,
    RECOMMENDED_TAG_COLUMN,
    VISIT_ID_COLUMN,
    clean_text,
    join_unique,
    load_allowed_tags,
    split_terms,
)
from prompts import (
    FINAL_RATIONALE_PROMPT,
    KERNEL_FIRST_PASS_PROMPT,
    REASSESSMENT_PROMPT,
    render_prompt,
)
from retrieval import TextEncoder


DEFAULT_RATIONALE_KG_PATH = PROJECT_ROOT / "data" / "MedlinePlus_kg.json"

# Published KERNEL configuration. Edit these values in code only when running a deliberate method variant; clinical/resource/output paths remain CLI inputs.
KERNEL_SETTINGS: dict[str, Any] = {
    "visit_id_path": None,
    "visit_id_column": VISIT_ID_COLUMN,
    "n_visits": None,
    "top_k_icd": 15,
    "kg_max_icds": 10,
    "kg_max_cuis_per_icd": 5,
    "prior_top_n": 8,
    "prior_min_visits": 5,
    "prior_min_rate": 0.0,
    "prior_max_icds": 10,
    "context_rag_top_k": 20,
    "context_rag_prior_top_n": 8,
    "context_rag_examples": 2,
    "context_rag_embedding_batch_size": 128,
    "kg_aware_rag_top_k": 20,
    "kg_aware_rag_prior_top_n": 8,
    "kg_aware_rag_examples": 2,
    "kg_aware_rag_embedding_batch_size": 128,
    "medlineplus_max_tags": 8,  # Maximum finalized lab categories supplied as rationale evidence.
    "medlineplus_max_tests_per_tag": 2,  # Maximum MedlinePlus entries supplied per category.
    "medlineplus_summary_max_chars": 220,
    "temperature": 0.0,
    "hf_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "hf_device_map": "auto",
    "hf_max_new_tokens": 768,
    "hf_generation_batch_size": 16,
    "hf_reassessment_generation_batch_size": 16,
    "hf_rationale_generation_batch_size": 16,
    "generate_final_rationale": True,
    "final_rationale_max_evidence_chars": 900,
    "evidence_completion_min_sources": 2,
    "evidence_completion_prior_min_rate": 0.75,
    "evidence_completion_rag_min_fraction": 0.75,
    "evidence_completion_max_added_tags": 3,
    "candidate_screening_min_sources": 2,
    "candidate_screening_prior_min_rate": 0.50,
    "candidate_screening_rag_min_fraction": 0.50,
    "candidate_screening_max_candidates": 8,
    "candidate_screening_remove_max_sources": 1,
    "candidate_screening_max_remove_candidates": 4,
    "reassessment_max_added_tags": 3,
    "reassessment_max_removed_tags": 2,
    "reassessment_min_evidence_strength": "moderate",
    "reassessment_min_patient_applicability": "moderate",
    "reassessment_remove_max_evidence_strength": "weak",
    "reassessment_remove_max_patient_applicability": "weak",
}


def load_eval_visit_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {VISIT_ID_COLUMN, PATIENT_CONTEXT_COLUMN, CHIEF_COMPLAINT_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required evaluation column(s): {sorted(missing)}")
    frame[VISIT_ID_COLUMN] = frame[VISIT_ID_COLUMN].map(clean_text)
    frame[PATIENT_CONTEXT_COLUMN] = frame[PATIENT_CONTEXT_COLUMN].map(clean_text)
    frame[CHIEF_COMPLAINT_COLUMN] = frame[CHIEF_COMPLAINT_COLUMN].map(clean_text)
    empty_visit_id = frame[VISIT_ID_COLUMN].eq("")
    if empty_visit_id.any():
        print(
            f"error: skipping {int(empty_visit_id.sum())} evaluation row(s) because "
            f"{VISIT_ID_COLUMN} is empty.",
            file=sys.stderr,
        )
        frame = frame.loc[~empty_visit_id].reset_index(drop=True)
    empty_context = frame[PATIENT_CONTEXT_COLUMN].eq("")
    if empty_context.any():
        for visit_row_id in frame.loc[empty_context, VISIT_ID_COLUMN].tolist():
            print(
                f"error: skipping visit {visit_row_id!r} because {PATIENT_CONTEXT_COLUMN} is empty.",
                file=sys.stderr,
            )
        frame = frame.loc[~empty_context].reset_index(drop=True)
    return frame


def select_eval_visits_by_ids(visits: pd.DataFrame, visit_id_path: Path, visit_id_column: str) -> pd.DataFrame:
    ids = pd.read_csv(visit_id_path, dtype="string")
    if visit_id_column not in ids.columns:
        visit_id_column = VISIT_ID_COLUMN
    requested_ids: list[str] = []
    seen: set[str] = set()
    for value in ids[visit_id_column].tolist():
        text = clean_text(value)
        if text and text not in seen:
            requested_ids.append(text)
            seen.add(text)
    selected = visits.loc[visits[VISIT_ID_COLUMN].map(clean_text).isin(requested_ids)].copy()
    order = {visit_id: i for i, visit_id in enumerate(requested_ids)}
    selected["_visit_order"] = selected[VISIT_ID_COLUMN].map(clean_text).map(order)
    selected = selected.sort_values("_visit_order").drop(columns=["_visit_order"]).reset_index(drop=True)
    missing = len(requested_ids) - len(selected)
    if missing:
        print(f"warning: {missing} requested visit ids were not found in the eval table")
    return selected


def filter_review_for_visit_ids(review_path: Path, visit_ids: list[str]) -> pd.DataFrame:
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    if VISIT_ID_COLUMN not in review.columns:
        raise ValueError(f"{review_path} is missing required column: {VISIT_ID_COLUMN}")
    review[VISIT_ID_COLUMN] = review[VISIT_ID_COLUMN].map(clean_text)
    requested = {clean_text(value) for value in visit_ids}
    return review.loc[review[VISIT_ID_COLUMN].isin(requested)].copy()


def require_stage1_prediction_coverage(
    eval_visit_ids: list[str],
    review: pd.DataFrame,
) -> None:
    prediction_ids = set(review[VISIT_ID_COLUMN].map(clean_text).unique())
    missing_ids = sorted(set(eval_visit_ids) - prediction_ids)
    if not missing_ids:
        return
    preview = ", ".join(missing_ids[:10])
    suffix = " ..." if len(missing_ids) > 10 else ""
    raise ValueError(
        f"Stage 1 ICD predictions are missing for {len(missing_ids)} evaluation visit(s): "
        f"{preview}{suffix}"
    )


def predicted_icds_by_visit(review: pd.DataFrame, top_k_icd: int) -> dict[str, list[dict[str, str]]]:
    if review.empty:
        return {}
    normalized = review.copy()
    normalized["combined_rank_num"] = pd.to_numeric(normalized["llm_icd_rank"], errors="coerce").fillna(999999).astype(int)
    normalized = normalized.sort_values([VISIT_ID_COLUMN, "combined_rank_num"], ascending=[True, True])
    out: dict[str, list[dict[str, str]]] = {}
    for visit_id, group in normalized.groupby(VISIT_ID_COLUMN, sort=False):
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for _, row in group.head(top_k_icd).iterrows():
            code = clean_text(row.get("llm_icd_code"))
            title = clean_text(row.get("llm_icd_title"))
            if not code or code in seen:
                continue
            rows.append({"rank": clean_text(row.get("llm_icd_rank")), "code": code, "title": title})
            seen.add(code)
        out[clean_text(visit_id)] = rows
    return out


def load_kg_resources(
    diagnosis_kg_path: Path,
    concept_metadata_path: Path,
) -> dict[str, Any]:
    diagnosis = pd.read_csv(diagnosis_kg_path, dtype="string")
    required = {"diagnosis_cui", "diagnosis_name", "icd10_title", "icd10_title_matched"}
    missing = required - set(diagnosis.columns)
    if missing:
        raise ValueError(f"{diagnosis_kg_path} is missing columns: {sorted(missing)}")
    diagnosis = diagnosis.loc[
        diagnosis["icd10_title_matched"].fillna("").astype(str).str.lower().eq("true")
    ].copy()
    diagnosis["icd10_title_norm"] = diagnosis["icd10_title"].map(lambda value: clean_text(value).lower())
    diagnosis["diagnosis_cui"] = diagnosis["diagnosis_cui"].map(clean_text)
    diagnosis["diagnosis_name"] = diagnosis["diagnosis_name"].map(clean_text)
    diagnosis = diagnosis.loc[diagnosis["diagnosis_cui"].ne("") & diagnosis["diagnosis_name"].ne("")]

    metadata = pd.read_csv(concept_metadata_path, dtype="string")
    metadata["cui"] = metadata["cui"].map(clean_text)
    metadata["name"] = metadata["name"].map(clean_text)
    metadata_by_cui = metadata.drop_duplicates("cui").set_index("cui").to_dict("index")

    return {
        "diagnosis_by_icd_title": {
            title: group.drop_duplicates("diagnosis_cui").to_dict("records")
            for title, group in diagnosis.groupby("icd10_title_norm", sort=False)
        },
        "metadata_by_cui": metadata_by_cui,
    }


def concept_category(meta: dict[str, Any]) -> str:
    if clean_text(meta.get("is_lab_test")).lower() == "true":
        return "lab_test"
    if clean_text(meta.get("is_diagnostic_aid")).lower() == "true":
        return "diagnostic_aid"
    if clean_text(meta.get("is_symptom_or_finding")).lower() == "true":
        return "symptom_or_finding"
    if clean_text(meta.get("is_diagnosis_like")).lower() == "true":
        return "diagnosis_like"
    return clean_text(meta.get("context_category")) or "other"


def expand_one_icd(
    icd: dict[str, str],
    kg: dict[str, Any],
    max_cuis: int,
) -> dict[str, Any]:
    title = clean_text(icd.get("title"))
    title_norm = title.lower()
    rows = kg["diagnosis_by_icd_title"].get(title_norm, [])[:max_cuis]
    concepts = []
    metadata_by_cui = kg["metadata_by_cui"]
    for row in rows:
        cui = clean_text(row.get("diagnosis_cui"))
        name = clean_text(row.get("diagnosis_name"))
        if not cui or not name:
            continue
        meta = metadata_by_cui.get(cui, {})
        concepts.append(
            {
                "cui": cui,
                "name": name,
                "category": concept_category(meta),
                "semantic_types": clean_text(meta.get("semantic_types")),
            }
        )
    return {
        "rank": clean_text(icd.get("rank")),
        "icd_code": clean_text(icd.get("code")),
        "icd_title": title,
        "diagnosis_concepts": concepts,
    }


def build_kg_expansion(
    predicted_icds: list[dict[str, str]],
    kg: dict[str, Any],
    max_icds: int,
    max_cuis_per_icd: int,
) -> dict[str, Any]:
    expansions = []
    for icd in predicted_icds[:max_icds]:
        expanded = expand_one_icd(icd, kg, max_cuis_per_icd)
        if expanded["diagnosis_concepts"]:
            expansions.append(expanded)
    return {"icd_expansions": expansions}


def kg_terms_from_expansion(expansion: dict[str, Any]) -> list[str]:
    terms = []
    for icd in expansion.get("icd_expansions", []) or []:
        terms.append(clean_text(icd.get("icd_title")))
        for concept in icd.get("diagnosis_concepts", []) or []:
            terms.append(clean_text(concept.get("name")))
    return [term for term in terms if term]


def _format_kg_concepts(
    expansion: dict[str, Any] | None,
    heading: str,
    relation: str,
    max_names_per_icd: int,
    note: str = "",
) -> str:
    if not expansion:
        return ""
    lines = [heading]
    if note:
        lines.append(note)
    for icd in expansion.get("icd_expansions", []) or []:
        label = f"{clean_text(icd.get('icd_code'))} {clean_text(icd.get('icd_title'))}".strip()
        names = [
            name
            for row in icd.get("diagnosis_concepts", []) or []
            if (name := clean_text(row.get("name")))
        ]
        if label and names:
            lines.append(f"- {label}: {relation}: {', '.join(names[:max_names_per_icd])}.")
    minimum_lines = 2 if note else 1
    return "\n".join(lines) if len(lines) > minimum_lines else ""


def format_kg_expansion(expansion: dict[str, Any] | None, max_names_per_icd: int = 5) -> str:
    return _format_kg_concepts(
        expansion,
        "Knowledge-graph diagnosis concept expansion:",
        "mapped diagnosis concepts",
        max_names_per_icd,
    )


def format_kg_diagnostic_evidence(expansion: dict[str, Any] | None, max_names_per_icd: int = 6) -> str:
    return _format_kg_concepts(
        expansion,
        "Knowledge-graph-derived diagnostic evidence:",
        "expanded diagnosis concepts",
        max_names_per_icd,
        "The following diagnosis concepts are directly expanded from the predicted diagnostic categories and are provided as diagnostic context, not as ordering rules.",
    )


def format_compact_candidate_tag_knowledge(candidate_tags: list[dict[str, Any]]) -> str:
    rows = []
    for row in candidate_tags:
        tag = clean_text(row.get("test_tag"))
        if not tag:
            continue
        linked = []
        for icd in row.get("linked_icds", []) or []:
            if label := format_ranked_icd(icd):
                linked.append(label)
        if linked:
            rows.append(f"- {tag}: " + "; ".join(linked) + ".")
        else:
            rows.append(f"- {tag}.")
    if not rows:
        return "- No clinician-curated diagnostic-to-test links were available."
    return "\n".join(rows)


def load_medlineplus_kg(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"MedlinePlus KG file does not exist: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by test_tag.")
    return {clean_text(tag): rows for tag, rows in data.items() if clean_text(tag) and isinstance(rows, list)}


def build_medlineplus_evidence(
    candidate_tag_order: list[str],
    kg_by_tag: dict[str, list[dict[str, Any]]] | None,
    max_tags: int,
    max_tests_per_tag: int,
) -> dict[str, Any]:
    if not kg_by_tag:
        return {}
    rows = []
    for tag in candidate_tag_order:
        if len(rows) >= max_tags:
            break
        tests = kg_by_tag.get(tag, [])
        if not tests:
            continue
        selected = []
        for item in tests:
            if len(selected) >= max_tests_per_tag:
                break
            selected.append(
                {
                    "lab_test_title": clean_text(item.get("lab_test_title")),
                    "purpose": clean_text(item.get("purpose")),
                    "summary": clean_text(item.get("summary")),
                }
            )
        if selected:
            rows.append({"test_tag": tag, "lab_tests": selected})
    return {"test_tag_lab_kg": rows}


def medlineplus_explanation_lines(evidence: dict[str, Any] | None, max_summary_chars: int = 160) -> list[str]:
    if not evidence:
        return []
    lines = []
    for tag_row in evidence.get("test_tag_lab_kg", []) or []:
        tag = clean_text(tag_row.get("test_tag"))
        snippets = []
        for item in tag_row.get("lab_tests", []) or []:
            title = clean_text(item.get("lab_test_title"))
            purpose = clean_text(item.get("purpose"))
            summary = clean_text(item.get("summary"))
            detail = purpose or summary[:max_summary_chars]
            if title and detail:
                snippets.append(f"{title}: {detail}")
        if tag and snippets:
            lines.append(f"{tag} - " + " | ".join(snippets))
    return lines


def build_final_medlineplus_evidence(
    final_recommended_tags: list[str],
    kg_by_tag: dict[str, list[dict[str, Any]]],
    max_tags: int,
    max_tests_per_tag: int,
    max_summary_chars: int,
) -> tuple[dict[str, Any], str]:
    """Build MedlinePlus evidence only for the already-finalized tag set."""
    evidence = build_medlineplus_evidence(
        final_recommended_tags,
        kg_by_tag,
        max_tags,
        max_tests_per_tag,
    )
    rationale = join_unique(medlineplus_explanation_lines(evidence, max_summary_chars))
    return evidence, rationale


def evidence_tags_from_prior(
    evidence: dict[str, Any] | None,
    min_rate: float,
) -> dict[str, list[str]]:
    support: dict[str, list[str]] = {}
    if not evidence:
        return support
    for source_key, source_label in [
        ("chief_complaint_priors", "disease_prior_cc"),
        ("predicted_icd_priors", "disease_prior_icd"),
    ]:
        for source in evidence.get(source_key, []) or []:
            label = clean_text(source.get("chief_complaint")) or f"{clean_text(source.get('code'))} {clean_text(source.get('title'))}".strip()
            for item in source.get("common_test_tags", []) or []:
                tag = clean_text(item.get("test_tag"))
                try:
                    rate = float(item.get("visit_rate", 0.0) or 0.0)
                except (TypeError, ValueError):
                    rate = 0.0
                if tag and rate >= min_rate:
                    support.setdefault(tag, []).append(f"{source_label}:{label}:{rate:.2f}")
    return support


def evidence_tags_from_rag(
    evidence: dict[str, Any] | None,
    source_label: str,
    min_fraction: float,
) -> dict[str, list[str]]:
    support: dict[str, list[str]] = {}
    if not evidence:
        return support
    retrieved_count = int(evidence.get("retrieved_count", 0) or 0)
    if retrieved_count <= 0:
        return support
    for item in evidence.get("common_test_tags", []) or []:
        tag = clean_text(item.get("test_tag"))
        count = int(item.get("visit_count", 0) or 0)
        fraction = count / retrieved_count
        if tag and fraction >= min_fraction:
            support.setdefault(tag, []).append(f"{source_label}:{count}/{retrieved_count}")
    return support


def merge_support_maps(*maps: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for support_map in maps:
        for tag, reasons in support_map.items():
            bucket = merged.setdefault(tag, [])
            for reason in reasons:
                if reason not in bucket:
                    bucket.append(reason)
    return merged


def complete_recommendations_from_evidence(
    recommended_tags: list[str],
    allowed_tags: list[str],
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
    min_sources: int,
    prior_min_rate: float,
    rag_min_fraction: float,
    max_added_tags: int,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    if min_sources <= 0:
        return recommended_tags, [], {}
    allowed = set(allowed_tags)
    existing = {tag for tag in recommended_tags if tag in allowed}
    support = build_candidate_support_map(
        training_prior_evidence,
        context_rag_evidence,
        kg_aware_rag_evidence,
        prior_min_rate,
        rag_min_fraction,
    )
    candidates = []
    for tag, reasons in support.items():
        source_types = {reason.split(":", 1)[0] for reason in reasons}
        if tag in allowed and tag not in existing and len(source_types) >= min_sources:
            candidates.append((tag, len(source_types), len(reasons), reasons))
    candidates.sort(key=lambda row: (-row[1], -row[2], row[0]))
    added = [tag for tag, _, _, _ in candidates]
    if max_added_tags > 0:
        added = added[:max_added_tags]
    completed = split_terms(join_unique(recommended_tags + added))
    added_support = {tag: support.get(tag, []) for tag in added}
    return completed, added, added_support


def evidence_source_types(reasons: list[str]) -> list[str]:
    return sorted({clean_text(reason.split(":", 1)[0]) for reason in reasons if clean_text(reason)})


def build_candidate_support_map(
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
    prior_min_rate: float,
    rag_min_fraction: float,
) -> dict[str, list[str]]:
    return merge_support_maps(
        evidence_tags_from_prior(training_prior_evidence, prior_min_rate),
        evidence_tags_from_rag(context_rag_evidence, "context_rag", rag_min_fraction),
        evidence_tags_from_rag(kg_aware_rag_evidence, "kg_aware_rag", rag_min_fraction),
    )


def build_candidate_screening(
    recommended_tags: list[str],
    allowed_tags: list[str],
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
    min_sources: int,
    prior_min_rate: float,
    rag_min_fraction: float,
    max_candidates: int,
    remove_max_sources: int,
    max_remove_candidates: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    allowed = set(allowed_tags)
    recommended = {tag for tag in recommended_tags if tag in allowed}
    support = build_candidate_support_map(
        training_prior_evidence,
        context_rag_evidence,
        kg_aware_rag_evidence,
        prior_min_rate,
        rag_min_fraction,
    )
    add_candidates = []
    for tag, reasons in support.items():
        source_types = evidence_source_types(reasons)
        if tag not in allowed or tag in recommended or len(source_types) < min_sources:
            continue
        add_candidates.append(
            {
                "test_tag": tag,
                "screening_reason": "supported_by_structured_evidence_but_omitted_by_first_pass_llm",
                "current_status": "not_recommended",
                "source_types": source_types,
                "support_reasons": reasons,
                "source_type_count": len(source_types),
                "support_reason_count": len(reasons),
            }
        )

    remove_candidates = []
    for tag in sorted(recommended):
        reasons = support.get(tag, [])
        source_types = evidence_source_types(reasons)
        if len(source_types) > remove_max_sources:
            continue
        remove_candidates.append(
            {
                "test_tag": tag,
                "screening_reason": "selected_by_first_pass_llm_but_limited_structured_evidence",
                "current_status": "recommended",
                "source_types": source_types,
                "support_reasons": reasons or ["limited_structured_support_under_current_thresholds"],
                "source_type_count": len(source_types),
                "support_reason_count": len(reasons),
            }
        )

    add_candidates.sort(
        key=lambda row: (
            -int(row.get("source_type_count", 0)),
            -int(row.get("support_reason_count", 0)),
            clean_text(row.get("test_tag")),
        )
    )
    remove_candidates.sort(
        key=lambda row: (
            int(row.get("source_type_count", 0)),
            int(row.get("support_reason_count", 0)),
            clean_text(row.get("test_tag")),
        )
    )
    candidates = cap_screened_candidates(add_candidates, remove_candidates, max_candidates, max_remove_candidates)
    return candidates, support


def format_screened_candidates(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for idx, row in enumerate(candidates, start=1):
        tag = clean_text(row.get("test_tag"))
        status = clean_text(row.get("current_status"))
        screening_reason = clean_text(row.get("screening_reason"))
        source_types = ", ".join(row.get("source_types", []) or [])
        reasons = "; ".join(row.get("support_reasons", []) or [])
        lines.append(
            f"{idx}. {tag}\n"
            f"   - current first-pass status: {status}\n"
            f"   - screening reason: {screening_reason}\n"
            f"   - supporting source types: {source_types or 'none above threshold'}\n"
            f"   - structured evidence: {reasons}"
        )
    return "\n".join(lines)


SCORE_ORDER = {"weak": 0, "moderate": 1, "strong": 2}


def normalize_score(value: Any) -> str:
    score = clean_text(value).lower()
    return score if score in SCORE_ORDER else "weak"


def score_meets_threshold(score: str, threshold: str) -> bool:
    return SCORE_ORDER.get(normalize_score(score), 0) >= SCORE_ORDER.get(normalize_score(threshold), 1)


def score_at_most_threshold(score: str, threshold: str) -> bool:
    return SCORE_ORDER.get(normalize_score(score), 0) <= SCORE_ORDER.get(normalize_score(threshold), 0)


def cap_screened_candidates(
    add_candidates: list[dict[str, Any]],
    remove_candidates: list[dict[str, Any]],
    max_candidates: int,
    max_remove_candidates: int,
) -> list[dict[str, Any]]:
    if max_remove_candidates > 0:
        remove_candidates = remove_candidates[:max_remove_candidates]
    if max_candidates <= 0:
        return add_candidates + remove_candidates
    remove_candidates = remove_candidates[:max_candidates]
    add_budget = max(max_candidates - len(remove_candidates), 0)
    return add_candidates[:add_budget] + remove_candidates


def format_reassessment_prompt(
    patient_context: str,
    predicted_icds: list[dict[str, str]],
    current_recommended_tags: list[str],
    candidate_tags: list[dict[str, Any]],
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_expansion: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
    screened_candidates: list[dict[str, Any]],
    allowed_tags: list[str],
    min_evidence_strength: str,
    min_patient_applicability: str,
    remove_max_evidence_strength: str,
    remove_max_patient_applicability: str,
) -> str:
    kg_enabled = kg_expansion is not None
    kg_text = format_kg_expansion(kg_expansion) if kg_enabled else ""
    prior_text = format_training_prior_evidence(training_prior_evidence)
    context_rag_text = format_compact_rag_evidence(
        context_rag_evidence,
        "Patient-context similar visits from the local training data:",
    )
    kg_rag_text = (
        format_compact_rag_evidence(
            kg_aware_rag_evidence,
            "Diagnosis-aware similar visits from the local training data:",
        )
        if kg_enabled
        else ""
    )
    kg_section = (
        (
            format_kg_diagnostic_evidence(kg_expansion)
            or kg_text
            or "No knowledge-graph diagnostic evidence available."
        )
        if kg_enabled
        else ""
    )
    applicability_anchor = (
        "working diagnoses, or KG-expanded diagnostic concepts"
        if kg_enabled
        else "working diagnoses"
    )
    remove_instruction = (
        "- For candidates already recommended, use remove only when the patient context clearly does not support the tag despite first-pass selection, "
        f"and rate evidence_strength at most {normalize_score(remove_max_evidence_strength)} and patient_applicability at most {normalize_score(remove_max_patient_applicability)}.\n"
    )
    return render_prompt(
        REASSESSMENT_PROMPT,
        applicability_anchor=applicability_anchor,
        min_evidence_strength=normalize_score(min_evidence_strength),
        min_patient_applicability=normalize_score(min_patient_applicability),
        remove_instruction=remove_instruction,
        patient_context=patient_context,
        working_diagnoses=format_working_diagnoses(predicted_icds),
        current_recommended_tags=join_unique(current_recommended_tags) or "none",
        kg_section=kg_section,
        candidate_knowledge=format_compact_candidate_tag_knowledge(candidate_tags),
        prior_evidence=prior_text or "No disease-level prior evidence available.",
        context_rag_evidence=context_rag_text or "No patient-context RAG evidence available.",
        kg_rag_evidence=(
            kg_rag_text or "No KG-aware RAG evidence available."
            if kg_enabled
            else ""
        ),
        screened_candidates=format_screened_candidates(screened_candidates),
        allowed_tags="; ".join(allowed_tags),
        decision_options="add, do_not_add, keep, or remove",
    )


def parse_reassessment_response(
    text: str,
    allowed_tags: list[str],
    screened_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = set(allowed_tags)
    candidate_set = {clean_text(row.get("test_tag")) for row in screened_candidates}
    candidate_set = {tag for tag in candidate_set if tag in allowed}
    try:
        parsed = parse_llm_json(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Reassessment response is not a JSON object.", text, 0)

    decisions = []
    add_tags = []
    remove_tags = []
    valid_decisions = {"add", "do_not_add", "keep", "remove"}
    for item in parsed.get("tag_decisions", []) or []:
        if not isinstance(item, dict):
            continue
        tag = clean_text(item.get("test_tag"))
        decision = clean_text(item.get("decision")).lower()
        if tag not in candidate_set or decision not in valid_decisions:
            continue
        decisions.append(
            {
                "test_tag": tag,
                "decision": decision,
                "evidence_strength": normalize_score(item.get("evidence_strength")),
                "patient_applicability": normalize_score(item.get("patient_applicability")),
                "rationale": clean_text(item.get("rationale")),
            }
        )
        if decision == "add" and tag not in add_tags:
            add_tags.append(tag)
        if decision == "remove" and tag not in remove_tags:
            remove_tags.append(tag)

    for tag in parsed.get("add_tags", []) or []:
        tag = clean_text(tag)
        if tag in candidate_set and tag not in add_tags:
            add_tags.append(tag)
    for tag in parsed.get("remove_tags", []) or []:
        tag = clean_text(tag)
        if tag in candidate_set and tag not in remove_tags:
            remove_tags.append(tag)

    return {
        "tag_decisions": decisions,
        "add_tags": add_tags,
        "remove_tags": remove_tags,
        "final_explanation": clean_text(parsed.get("final_explanation")),
    }


def apply_reassessment(
    recommended_tags: list[str],
    reassessment_response: dict[str, Any],
    allowed_tags: list[str],
    max_added_tags: int,
    max_removed_tags: int,
    min_evidence_strength: str,
    min_patient_applicability: str,
    remove_max_evidence_strength: str,
    remove_max_patient_applicability: str,
) -> tuple[list[str], list[str], list[str]]:
    allowed = set(allowed_tags)
    add_candidates = [tag for tag in split_terms(join_unique(reassessment_response.get("add_tags", []))) if tag in allowed]
    decision_by_tag = {
        clean_text(item.get("test_tag")): item
        for item in reassessment_response.get("tag_decisions", []) or []
        if clean_text(item.get("test_tag"))
    }
    add_tags = []
    for tag in add_candidates:
        decision = decision_by_tag.get(tag, {})
        if clean_text(decision.get("decision")).lower() != "add":
            continue
        if not score_meets_threshold(clean_text(decision.get("evidence_strength")), min_evidence_strength):
            continue
        if not score_meets_threshold(clean_text(decision.get("patient_applicability")), min_patient_applicability):
            continue
        add_tags.append(tag)
    remove_candidates = [tag for tag in split_terms(join_unique(reassessment_response.get("remove_tags", []))) if tag in allowed]
    remove_tags = []
    for tag in remove_candidates:
        decision = decision_by_tag.get(tag, {})
        if clean_text(decision.get("decision")).lower() != "remove":
            continue
        if not score_at_most_threshold(clean_text(decision.get("evidence_strength")), remove_max_evidence_strength):
            continue
        if not score_at_most_threshold(clean_text(decision.get("patient_applicability")), remove_max_patient_applicability):
            continue
        remove_tags.append(tag)
    if max_added_tags > 0:
        add_tags = add_tags[:max_added_tags]
    if max_removed_tags > 0:
        remove_tags = remove_tags[:max_removed_tags]
    remove_set = set(remove_tags)
    updated = [tag for tag in recommended_tags if tag in allowed and tag not in remove_set]
    for tag in add_tags:
        if tag not in updated:
            updated.append(tag)
    return updated, add_tags, remove_tags


def format_compact_rag_evidence(context_rag_evidence: dict[str, Any] | None, title: str) -> str:
    tag_counts = format_rag_tag_counts(context_rag_evidence)
    if not tag_counts:
        return ""
    return f"{title}\n- Common ordered test_tags: {tag_counts}."


def train_rows_with_kg_text(
    train_rows: pd.DataFrame,
    kg: dict[str, Any],
    max_cuis_per_icd: int,
) -> pd.DataFrame:
    rows = []
    for _, row in train_rows.iterrows():
        context = clean_text(row.get(PATIENT_CONTEXT_COLUMN))
        tags = row.get("test_tags", [])
        if not context or not tags:
            continue
        icd_records = [
            {"rank": "", "code": code, "title": title}
            for code, title in zip(
                row.get(GOLD_ICD_CODE_COLUMN, []),
                row.get(GOLD_ICD_TITLE_COLUMN, []),
                strict=False,
            )
        ]
        expansion = build_kg_expansion(icd_records, kg, len(icd_records), max_cuis_per_icd)
        kg_text = " ".join(kg_terms_from_expansion(expansion))
        retrieval_diagnosis_text = " ".join(
            [join_unique(row.get(GOLD_ICD_TITLE_COLUMN, [])), kg_text]
        ).strip()
        rows.append(
            {
                "train_visit_row_id": clean_text(row.get("train_visit_row_id")),
                PATIENT_CONTEXT_COLUMN: context,
                "kg_retrieval_text": " ".join([context, retrieval_diagnosis_text]).strip(),
                CHIEF_COMPLAINT_COLUMN: clean_text(row.get(CHIEF_COMPLAINT_COLUMN)),
                GOLD_ICD_TITLE_COLUMN: row.get(GOLD_ICD_TITLE_COLUMN, []),
                "test_tags": tags,
            }
        )
    return pd.DataFrame(rows)


def build_kg_aware_rag_index(
    train_rows: pd.DataFrame,
    embedding_batch_size: int,
    encoder: TextEncoder,
) -> dict[str, Any]:
    indexed = train_rows.copy()
    indexed[PATIENT_CONTEXT_COLUMN] = indexed["kg_retrieval_text"].fillna(
        indexed[PATIENT_CONTEXT_COLUMN]
    ).map(clean_text)
    return build_context_rag_index(indexed, embedding_batch_size, encoder)


def format_kg_augmented_prompt(
    patient_context: str,
    predicted_icds: list[dict[str, str]],
    candidate_tags: list[dict[str, Any]],
    allowed_tags: list[str],
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_expansion: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
) -> str:
    kg_enabled = kg_expansion is not None
    kg_text = format_kg_diagnostic_evidence(kg_expansion) if kg_enabled else ""
    kg_section = f"\n{kg_text}\n" if kg_text else ""
    disease_prior_text = format_training_prior_evidence(training_prior_evidence)
    disease_prior_section = f"\n{disease_prior_text}\n" if disease_prior_text else ""
    context_rag_text = format_context_rag_evidence(context_rag_evidence)
    context_rag_section = f"\n{context_rag_text}\n" if context_rag_text else ""
    kg_rag_text = format_context_rag_evidence(kg_aware_rag_evidence) if kg_enabled else ""
    if kg_rag_text:
        kg_rag_text = kg_rag_text.replace(
            "In similar patient-context visits from the local training data",
            "In diagnosis-aware similar visits from the local training data",
            1,
        )
    kg_rag_section = f"\n{kg_rag_text}\n" if kg_rag_text else ""
    if kg_enabled:
        support_sources = (
            "clinician ICD-to-test_tag mapping, local training-set disease-level prior, "
            "patient-context RAG, KG-aware RAG, or knowledge-graph diagnostic evidence"
        )
    else:
        support_sources = (
            "clinician ICD-to-test_tag mapping, local training-set disease-level prior, "
            "or patient-context RAG"
        )
    recall_relaxed_instruction = (
        "- ED laboratory ordering often includes complementary rule-in/rule-out and safety-assessment categories, not only tests for the single most likely diagnosis.\n"
        "- For this first-pass recommendation, avoid stopping after only the most obvious core labs.\n"
        f"- Include a test_tag when the patient context supports a relevant clinical question and at least one of the following supports it: {support_sources}.\n"
        "- Prefer tags supported by both patient context and local training-set evidence, but do not ignore a clinically important test_tag solely because local prior evidence is sparse.\n"
        "- Pay special attention to commonly missed but context-dependent categories: Microbiology/Infection, Blood Gas&Acid-Base, Coagulation, Cardiovascular, Genetic/Reproductive/Pregnancy, Blood Bank/Transfusion, Endocrine, and Metabolic.\n"
        "- Still be selective: do not add a tag solely because it is common, appears in one ICD mapping, or is weakly related to a nonspecific abnormal vital sign.\n"
    )
    return render_prompt(
        KERNEL_FIRST_PASS_PROMPT,
        patient_context=patient_context,
        working_diagnoses=format_working_diagnoses(predicted_icds),
        kg_section=kg_section,
        candidate_knowledge=format_candidate_tag_knowledge(candidate_tags),
        disease_prior_section=disease_prior_section,
        context_rag_section=context_rag_section,
        kg_rag_section=kg_rag_section,
        kg_instruction=(
            "- The knowledge-graph-derived diagnostic evidence is background diagnostic context, not an ordering guideline."
            if kg_enabled
            else ""
        ),
        kg_concepts_phrase="KG-expanded concepts, " if kg_enabled else "",
        recall_relaxed_instruction=recall_relaxed_instruction.rstrip(),
        allowed_tags="; ".join(allowed_tags),
        rationale_evidence_phrase="KG concepts, " if kg_enabled else "",
    )


def truncate_for_prompt(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_final_rationale_prompt(
    patient_context: str,
    predicted_icds: list[dict[str, str]],
    final_recommended_tags: list[str],
    llm_recommended_tags: list[str],
    reassessment_added_tags: list[str],
    reassessment_removed_tags: list[str],
    evidence_completion_added_tags: list[str],
    training_prior_evidence: dict[str, Any] | None,
    context_rag_evidence: dict[str, Any] | None,
    kg_expansion: dict[str, Any] | None,
    kg_aware_rag_evidence: dict[str, Any] | None,
    medlineplus_selected_rationale: str,
    max_evidence_chars: int,
) -> str:
    diagnosis_lines = []
    for icd in predicted_icds[:10]:
        title = clean_text(icd.get("title"))
        if title:
            diagnosis_lines.append(f"- {title}")
    diagnosis_text = "\n".join(diagnosis_lines) or "No structured working diagnostic categories available."
    evidence_blocks = {
        "Local ordering patterns": format_training_prior_evidence(training_prior_evidence),
        "Patient-context similar cases": format_context_rag_evidence(context_rag_evidence),
        "Similar prior emergency visits with related diagnostic context": format_context_rag_evidence(kg_aware_rag_evidence),
        "Expanded diagnostic context": format_kg_diagnostic_evidence(kg_expansion),
        "External laboratory reference information": medlineplus_selected_rationale,
    }
    evidence_text = "\n".join(
        f"{label}:\n{truncate_for_prompt(value, max_evidence_chars)}"
        for label, value in evidence_blocks.items()
        if clean_text(value)
    )
    return render_prompt(
        FINAL_RATIONALE_PROMPT,
        patient_context=patient_context,
        working_diagnoses=diagnosis_text,
        first_pass_tags=join_unique(llm_recommended_tags) or "none",
        final_tags=join_unique(final_recommended_tags) or "none",
        reassessment_added_tags=join_unique(reassessment_added_tags) or "none",
        reassessment_removed_tags=join_unique(reassessment_removed_tags) or "none",
        evidence_completion_added_tags=join_unique(evidence_completion_added_tags) or "none",
        evidence_summary=evidence_text or "No additional structured evidence available.",
    )


def parse_final_rationale_response(text: str) -> str:
    parsed = parse_llm_json(text)
    return clean_text(parsed.get("final_explanation") or parsed.get("final_summary_rationale") or parsed.get("rationale"))


def resolve_final_rationale(
    initial_response: str,
    prompt: str,
    visit_id: str,
    final_recommended_tags: list[str],
    first_pass_tags: list[str],
    first_pass_explanation: str,
    llm_generate,
) -> str:
    response_text = initial_response
    last_error: Exception | None = None
    for attempt in range(2):
        if attempt == 1:
            retry_prompt = (
                prompt
                + "\n\nThe previous response could not be used. Return exactly one JSON object "
                'with a non-empty "final_explanation" string and no other text.'
            )
            try:
                response_text = llm_generate_many(
                    llm_generate,
                    [retry_prompt],
                    1,
                    "final rationale retry",
                )[0]
            except Exception as exc:
                last_error = exc
                break
        try:
            explanation = parse_final_rationale_response(response_text)
            if not explanation:
                raise ValueError("Final rationale response contains no explanation.")
            return explanation
        except Exception as exc:
            last_error = exc

    final_tag_set = {clean_text(tag) for tag in final_recommended_tags if clean_text(tag)}
    first_pass_tag_set = {clean_text(tag) for tag in first_pass_tags if clean_text(tag)}
    message = (
        f"final rationale failed for visit {clean_text(visit_id)!r} after one retry: "
        f"{type(last_error).__name__}: {last_error}"
    )
    if final_tag_set != first_pass_tag_set:
        raise RuntimeError(
            message + "; refusing to use the stale first-pass explanation because final tags changed."
        ) from last_error
    print(
        "warning: " + message + "; using the first-pass explanation because final tags did not change.",
        file=sys.stderr,
    )
    return clean_text(first_pass_explanation)


def generation_batch_size(args: argparse.Namespace, stage: str = "first_pass") -> int:
    base = max(int(getattr(args, "hf_generation_batch_size", 16) or 16), 1)
    if stage == "reassessment":
        return max(int(getattr(args, "hf_reassessment_generation_batch_size", 0) or base), 1)
    if stage == "rationale":
        return max(int(getattr(args, "hf_rationale_generation_batch_size", 0) or base), 1)
    return base


def llm_generate_many(llm_generate, prompts: list[str], batch_size: int, desc: str) -> list[str]:
    if not prompts:
        return []
    outputs = llm_generate(prompts, batch_size=batch_size, desc=desc)
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"{desc} returned {len(outputs)} outputs for {len(prompts)} prompts; "
            "batch alignment cannot be guaranteed."
        )
    return outputs


def parse_first_pass_response(
    prompt: str,
    response_text: str,
    llm_generate,
    allowed_tags: list[str],
) -> tuple[dict[str, Any], list[str], str, str]:
    response_json: dict[str, Any] = {}
    invalid_tags: list[str] = []
    error = ""
    try:
        try:
            response_json = parse_or_recover_llm_response(response_text, allowed_tags)
        except json.JSONDecodeError:
            retry_prompt = retry_prompt_for_invalid_json(prompt, response_text, allowed_tags)
            response_text = llm_generate_many(
                llm_generate,
                [retry_prompt],
                1,
                "JSON repair retry",
            )[0]
            response_json = parse_or_recover_llm_response(response_text, allowed_tags)
        response_json, invalid_tags = validate_recommendations(response_json, allowed_tags)
        if invalid_tags:
            retry_prompt = retry_prompt_for_invalid_tags(prompt, invalid_tags)
            response_text = llm_generate_many(
                llm_generate,
                [retry_prompt],
                1,
                "invalid-tag repair retry",
            )[0]
            response_json = parse_or_recover_llm_response(response_text, allowed_tags)
            response_json, invalid_tags = validate_recommendations(response_json, allowed_tags)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        response_json = {}
        invalid_tags = []
    return response_json, invalid_tags, error, response_text


def build_visit_state(
    visit_id: str,
    group: pd.DataFrame,
    eval_row: pd.Series | None,
    predicted_icds_map: dict[str, list[dict[str, str]]],
    kg: dict[str, Any] | None,
    train_priors: dict[str, Any] | None,
    context_rag_index: dict[str, Any] | None,
    kg_aware_rag_index: dict[str, Any] | None,
    allowed_tags: list[str],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    visit_id_text = clean_text(visit_id)
    if eval_row is not None:
        first = eval_row
    elif not group.empty:
        first = group.iloc[0]
    else:
        return None
    patient_context = clean_text(first.get(PATIENT_CONTEXT_COLUMN))
    chief_complaint = clean_text(first.get(CHIEF_COMPLAINT_COLUMN))
    if not patient_context and not group.empty:
        patient_context = clean_text(group.iloc[0].get(PATIENT_CONTEXT_COLUMN))
    predicted_icds = predicted_icds_map.get(visit_id_text, [])
    if not predicted_icds and not group.empty:
        predicted_icds = summarize_icds(group)
    candidate_tags = summarize_candidate_tags(group) if not group.empty and "test_tag" in group.columns else []
    kg_expansion = (
        build_kg_expansion(
            predicted_icds,
            kg,
            args.kg_max_icds,
            args.kg_max_cuis_per_icd,
        )
        if kg is not None
        else None
    )
    training_prior_evidence = build_prior_evidence_for_visit(
        chief_complaint,
        predicted_icds,
        train_priors,
        args.prior_top_n,
        args.prior_max_icds,
    )
    context_rag_evidence = build_context_rag_evidence_for_visit(
        patient_context,
        context_rag_index,
        args.context_rag_top_k,
        args.context_rag_prior_top_n,
        args.context_rag_examples,
    )
    if kg_aware_rag_index is not None and kg_expansion is not None:
        diagnosis_query_text = " ".join(kg_terms_from_expansion(kg_expansion))
        kg_query_context = " ".join([patient_context, diagnosis_query_text]).strip()
        kg_aware_rag_evidence = build_context_rag_evidence_for_visit(
            kg_query_context,
            kg_aware_rag_index,
            args.kg_aware_rag_top_k,
            args.kg_aware_rag_prior_top_n,
            args.kg_aware_rag_examples,
        )
    else:
        kg_aware_rag_evidence = None
    prompt = format_kg_augmented_prompt(
        patient_context,
        predicted_icds,
        candidate_tags,
        allowed_tags,
        training_prior_evidence,
        context_rag_evidence,
        kg_expansion,
        kg_aware_rag_evidence,
    )
    return {
        "visit_id": visit_id,
        "visit_id_text": visit_id_text,
        "first": first,
        "patient_context": patient_context,
        "chief_complaint": chief_complaint,
        "predicted_icds": predicted_icds,
        "candidate_tags": candidate_tags,
        "kg_expansion": kg_expansion,
        "training_prior_evidence": training_prior_evidence,
        "context_rag_evidence": context_rag_evidence,
        "kg_aware_rag_evidence": kg_aware_rag_evidence,
        "prompt": prompt,
    }


OUTPUT_COLUMNS = [VISIT_ID_COLUMN, RECOMMENDED_TAG_COLUMN, FINAL_EXPLANATION_COLUMN]


def build_output_row(
    visit_row_id: str,
    final_recommended_tags: list[str],
    final_explanation: str,
) -> dict[str, str]:
    return {
        VISIT_ID_COLUMN: clean_text(visit_row_id),
        RECOMMENDED_TAG_COLUMN: join_unique(final_recommended_tags),
        FINAL_EXPLANATION_COLUMN: clean_text(final_explanation),
    }


def generate_kernel_recommendations(args: argparse.Namespace) -> pd.DataFrame:
    allowed_tags = load_allowed_tags(args.test_tag_list_path)
    icd2tag = load_icd2test_tag(args.icd2test_tag_path, allowed_tags)

    # Validate evaluation inputs before building retrieval indexes or loading the LLM.
    eval_visits = load_eval_visit_table(args.eval_path)
    if args.visit_id_path:
        eval_visits = select_eval_visits_by_ids(eval_visits, args.visit_id_path, args.visit_id_column)
        if args.n_visits is not None:
            eval_visits = eval_visits.head(args.n_visits).copy()
    elif args.n_visits is not None:
        eval_visits = eval_visits.head(args.n_visits).copy()
    visit_ids = eval_visits[VISIT_ID_COLUMN].map(clean_text).tolist()
    review = filter_review_for_visit_ids(args.review_path, visit_ids)
    require_stage1_prediction_coverage(visit_ids, review)
    review = normalize_prediction_columns(review)
    triplets = build_triplets(review, icd2tag, allowed_tags, args.top_k_icd)
    predicted_icds_map = predicted_icds_by_visit(review, args.top_k_icd)

    train = pd.read_csv(args.train_gold_path, dtype="string", keep_default_na=False)
    training_evidence_rows = prepare_training_evidence_rows(
        train,
        allowed_tags,
    )

    kg: dict[str, Any] | None = None
    if args.diagnosis_kg_path is not None:
        print("loading optional diagnosis knowledge-graph resources:", args.diagnosis_kg_path)
        kg = load_kg_resources(
            args.diagnosis_kg_path,
            args.concept_metadata_path,
        )
    else:
        print("diagnosis knowledge graph: disabled")
    print("loading MedlinePlus final-rationale evidence:", args.rationale_kg_path)
    medlineplus_kg = load_medlineplus_kg(args.rationale_kg_path)

    print("building training-derived test_tag priors:", args.train_gold_path)
    train_priors = build_train_test_tag_priors(
        training_evidence_rows,
        args.prior_min_visits,
        args.prior_top_n,
        args.prior_min_rate,
    )

    print("building embedding-based context RAG prior index:", args.train_gold_path)
    retrieval_encoder = TextEncoder(RETRIEVAL_MODEL_NAME)
    context_rows = build_context_rag_train_rows(training_evidence_rows)
    context_rag_index = build_context_rag_index(
        context_rows,
        args.context_rag_embedding_batch_size,
        retrieval_encoder,
    )
    print("context RAG prior index visits:", len(context_rows))

    kg_aware_rag_index = None
    if kg is not None:
        print("building embedding-based KG-aware similar-case retrieval index:", args.train_gold_path)
        kg_train_rows = train_rows_with_kg_text(
            training_evidence_rows,
            kg,
            args.kg_max_cuis_per_icd,
        )
        kg_aware_rag_index = build_kg_aware_rag_index(
            kg_train_rows,
            args.kg_aware_rag_embedding_batch_size,
            retrieval_encoder,
        )
        print("KG-aware RAG index visits:", len(kg_train_rows))

    output_rows: list[dict[str, str]] = []
    llm_generate = build_transformers_caller(args)

    triplet_groups_by_visit = (
        {clean_text(visit_id): group.copy() for visit_id, group in triplets.groupby(VISIT_ID_COLUMN, sort=False)}
        if not triplets.empty
        else {}
    )
    visit_groups = [
        (clean_text(row.get(VISIT_ID_COLUMN)), triplet_groups_by_visit.get(clean_text(row.get(VISIT_ID_COLUMN)), pd.DataFrame()), row)
        for _, row in eval_visits.iterrows()
    ]
    if visit_groups:
        first_pass_batch_size = generation_batch_size(args, "first_pass")
        reassessment_batch_size = generation_batch_size(args, "reassessment")
        rationale_batch_size = generation_batch_size(args, "rationale")
        print(
            "Transformers generation batches:",
            f"first_pass={first_pass_batch_size}",
            f"reassessment={reassessment_batch_size}",
            f"rationale={rationale_batch_size}",
        )
        batch_starts = list(range(0, len(visit_groups), first_pass_batch_size))
        for start in tqdm(
            batch_starts,
            total=len(batch_starts),
            desc="generate KERNEL LLM recommendations",
            unit="batch",
        ):
            chunk = visit_groups[start : start + first_pass_batch_size]
            states = [
                build_visit_state(
                    visit_id,
                    group,
                    eval_row,
                    predicted_icds_map,
                    kg,
                    train_priors,
                    context_rag_index,
                    kg_aware_rag_index,
                    allowed_tags,
                    args,
                )
                for visit_id, group, eval_row in chunk
            ]
            states = [state for state in states if state is not None]
            if not states:
                continue
            first_pass_responses = llm_generate_many(
                llm_generate,
                [state["prompt"] for state in states],
                first_pass_batch_size,
                "first-pass LLM",
            )
            runtime_rows: list[dict[str, Any]] = []
            for state, response_text in zip(states, first_pass_responses):
                response_json, _, error, _ = parse_first_pass_response(
                    state["prompt"],
                    response_text,
                    llm_generate,
                    allowed_tags,
                )
                if error:
                    print(f"warning: first-pass generation failed for {state['visit_id_text']}: {error}")
                recommended = response_json.get("recommended_tags", []) if response_json else []
                llm_recommended_tags = [
                    clean_text(item.get("test_tag"))
                    for item in recommended
                    if isinstance(item, dict) and clean_text(item.get("test_tag"))
                ]
                llm_returned_empty_recommendation = bool(response_json) and isinstance(recommended, list) and len(llm_recommended_tags) == 0
                final_recommended_tags = list(llm_recommended_tags)
                candidate_screening_candidates: list[dict[str, Any]] = []
                reassessment_prompt = ""
                if not llm_returned_empty_recommendation:
                    candidate_screening_candidates, _ = build_candidate_screening(
                        final_recommended_tags,
                        allowed_tags,
                        state["training_prior_evidence"],
                        state["context_rag_evidence"],
                        state["kg_aware_rag_evidence"],
                        args.candidate_screening_min_sources,
                        args.candidate_screening_prior_min_rate,
                        args.candidate_screening_rag_min_fraction,
                        args.candidate_screening_max_candidates,
                        args.candidate_screening_remove_max_sources,
                        args.candidate_screening_max_remove_candidates,
                    )
                    if candidate_screening_candidates:
                        reassessment_prompt = format_reassessment_prompt(
                            state["patient_context"],
                            state["predicted_icds"],
                            final_recommended_tags,
                            state["candidate_tags"],
                            state["training_prior_evidence"],
                            state["context_rag_evidence"],
                            state["kg_expansion"],
                            state["kg_aware_rag_evidence"],
                            candidate_screening_candidates,
                            allowed_tags,
                            args.reassessment_min_evidence_strength,
                            args.reassessment_min_patient_applicability,
                            args.reassessment_remove_max_evidence_strength,
                            args.reassessment_remove_max_patient_applicability,
                        )
                runtime_rows.append(
                    {
                        "state": state,
                        "response_json": response_json,
                        "error": error,
                        "llm_recommended_tags": llm_recommended_tags,
                        "llm_returned_empty_recommendation": llm_returned_empty_recommendation,
                        "final_recommended_tags": final_recommended_tags,
                        "candidate_screening_candidates": candidate_screening_candidates,
                        "reassessment_prompt": reassessment_prompt,
                        "reassessment_response": {},
                        "reassessment_added_tags": [],
                        "reassessment_removed_tags": [],
                    }
                )

            reassessment_rows = [
                row
                for row in runtime_rows
                if row["reassessment_prompt"] and not row["error"]
            ]
            reassessment_responses = llm_generate_many(
                llm_generate,
                [row["reassessment_prompt"] for row in reassessment_rows],
                reassessment_batch_size,
                "reassessment LLM",
            )
            for row, raw_response in zip(reassessment_rows, reassessment_responses):
                try:
                    row["reassessment_response"] = parse_reassessment_response(
                        raw_response,
                        allowed_tags,
                        row["candidate_screening_candidates"],
                    )
                    (
                        row["final_recommended_tags"],
                        row["reassessment_added_tags"],
                        row["reassessment_removed_tags"],
                    ) = apply_reassessment(
                        row["final_recommended_tags"],
                        row["reassessment_response"],
                        allowed_tags,
                        args.reassessment_max_added_tags,
                        args.reassessment_max_removed_tags,
                        args.reassessment_min_evidence_strength,
                        args.reassessment_min_patient_applicability,
                        args.reassessment_remove_max_evidence_strength,
                        args.reassessment_remove_max_patient_applicability,
                    )
                except Exception as exc:
                    print(f"warning: reassessment failed for {row['state']['visit_id_text']}: {type(exc).__name__}: {exc}")

            rationale_rows: list[dict[str, Any]] = []
            for row in runtime_rows:
                state = row["state"]
                row["evidence_completion_added_tags"] = []
                if not row["llm_returned_empty_recommendation"]:
                    (
                        row["final_recommended_tags"],
                        row["evidence_completion_added_tags"],
                        _,
                    ) = complete_recommendations_from_evidence(
                        row["final_recommended_tags"],
                        allowed_tags,
                        state["training_prior_evidence"],
                        state["context_rag_evidence"],
                        state["kg_aware_rag_evidence"],
                        args.evidence_completion_min_sources,
                        args.evidence_completion_prior_min_rate,
                        args.evidence_completion_rag_min_fraction,
                        args.evidence_completion_max_added_tags,
                    )
                _, row["medlineplus_explanation"] = build_final_medlineplus_evidence(
                    row["final_recommended_tags"],
                    medlineplus_kg,
                    args.medlineplus_max_tags,
                    args.medlineplus_max_tests_per_tag,
                    args.medlineplus_summary_max_chars,
                )
                row["first_pass_final_explanation"] = clean_text(row["response_json"].get("final_explanation")) if row["response_json"] else ""
                row["final_rationale_prompt"] = ""
                row["final_explanation"] = row["first_pass_final_explanation"]
                if args.generate_final_rationale:
                    row["final_rationale_prompt"] = format_final_rationale_prompt(
                        state["patient_context"],
                        state["predicted_icds"],
                        row["final_recommended_tags"],
                        row["llm_recommended_tags"],
                        row["reassessment_added_tags"],
                        row["reassessment_removed_tags"],
                        row["evidence_completion_added_tags"],
                        state["training_prior_evidence"],
                        state["context_rag_evidence"],
                        state["kg_expansion"],
                        state["kg_aware_rag_evidence"],
                        row["medlineplus_explanation"],
                        args.final_rationale_max_evidence_chars,
                    )
                    rationale_rows.append(row)

            rationale_responses = llm_generate_many(
                llm_generate,
                [row["final_rationale_prompt"] for row in rationale_rows],
                rationale_batch_size,
                "final rationale LLM",
            )
            for row, raw_response in zip(rationale_rows, rationale_responses):
                row["final_explanation"] = resolve_final_rationale(
                    raw_response,
                    row["final_rationale_prompt"],
                    row["state"]["visit_id_text"],
                    row["final_recommended_tags"],
                    row["llm_recommended_tags"],
                    row["first_pass_final_explanation"],
                    llm_generate,
                )

            for row in runtime_rows:
                output_rows.append(
                    build_output_row(
                        row["state"]["visit_id_text"],
                        row["final_recommended_tags"],
                        row["final_explanation"],
                    )
                )
    return pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # User-provided clinical data and generated outputs.
    parser.add_argument("--review-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--train-gold-path", type=Path, required=True)
    parser.add_argument(
        "--diagnosis-kg-path",
        type=Path,
        help="Optional diagnosis knowledge graph; requires --concept-metadata-path.",
    )
    parser.add_argument(
        "--concept-metadata-path",
        type=Path,
        help="Optional concept metadata; requires --diagnosis-kg-path.",
    )
    parser.add_argument("--output-path", type=Path, required=True)

    # Bundled resources may be replaced with compatible user-provided files.
    parser.add_argument("--icd2test-tag-path", type=Path, default=DEFAULT_ICD2TEST_TAG_PATH)
    parser.add_argument("--test-tag-list-path", type=Path, default=DEFAULT_TEST_TAG_LIST_PATH)
    parser.add_argument("--rationale-kg-path", type=Path, default=DEFAULT_RATIONALE_KG_PATH)

    args = parser.parse_args()
    if (args.diagnosis_kg_path is None) != (args.concept_metadata_path is None):
        parser.error(
            "--diagnosis-kg-path and --concept-metadata-path must be provided together"
        )
    for name, value in KERNEL_SETTINGS.items():
        setattr(args, name, value)
    return args


def main() -> None:
    args = parse_args()
    output = generate_kernel_recommendations(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print("saved KERNEL recommendations:", args.output_path, len(output))


if __name__ == "__main__":
    main()
