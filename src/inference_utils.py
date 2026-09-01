"""Shared utilities for KERNEL laboratory recommendation inference.

This script keeps lab-test recommendation separate from diagnosis ranking:
1. Read top ICD predictions from a review CSV.
2. Link ICD predictions to allowed test_tag values.
3. Ask an LLM to choose only from the allowed test_tag list and explain why.

The module contains input preparation, EOP, retrieval, prompt construction, Hugging Face Transformers inference, and output validation used by the main program.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from common import (
    CHIEF_COMPLAINT_COLUMN,
    GOLD_ICD_CODE_COLUMN,
    GOLD_ICD_TITLE_COLUMN,
    GOLD_TEST_TAG_COLUMN,
    PATIENT_CONTEXT_COLUMN,
    PREDICTED_ICD_CODE_COLUMN,
    PREDICTED_ICD_TITLE_COLUMN,
    PREDICTION_RANK_COLUMN,
    PROJECT_ROOT,
    VISIT_ID_COLUMN,
    clean_text,
    join_unique,
    split_terms,
)
from prompts import (
    INVALID_TAG_REPAIR_PROMPT,
    JSON_REPAIR_PROMPT,
    SYSTEM_PROMPT,
    render_prompt,
)
from retrieval import TextEncoder


DEFAULT_ICD2TEST_TAG_PATH = PROJECT_ROOT / "data" / "icd2test_tag.csv"
RETRIEVAL_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def load_icd2test_tag(path: Path, allowed_tags: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    required = {"icd_code", "icd_title", "test_tag"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    allowed = set(allowed_tags)
    out = frame.copy()
    out["icd_code"] = out["icd_code"].map(clean_text)
    out["icd_title"] = out["icd_title"].map(clean_text)
    out["test_tag"] = out["test_tag"].map(clean_text)
    out = out.loc[out["icd_code"].ne("") & out["test_tag"].isin(allowed)].copy()
    return out.drop_duplicates(["icd_code", "test_tag"])


def prior_key_text(value: Any) -> str:
    return clean_text(value).lower()


def prior_cc_keys(value: Any) -> list[str]:
    text = prior_key_text(value)
    if not text:
        return []
    parts = [prior_key_text(part) for part in re.split(r"[,;|]+", text) if prior_key_text(part)]
    return list(dict.fromkeys([text] + parts))

def add_prior_counts(
    counts: dict[str, Counter],
    totals: Counter,
    key: str,
    tags: list[str],
) -> None:
    if not key or not tags:
        return
    totals[key] += 1
    bucket = counts.setdefault(key, Counter())
    for tag in tags:
        bucket[tag] += 1


def format_prior_bucket(counter: Counter, total: int, top_n: int, min_rate: float) -> list[dict[str, Any]]:
    rows = []
    for tag, count in counter.most_common():
        rate = count / total if total else 0.0
        if rate < min_rate:
            continue
        rows.append({"test_tag": tag, "visit_count": int(count), "visit_rate": round(rate, 3)})
        if len(rows) >= top_n:
            break
    return rows


def prepare_training_evidence_rows(
    train: pd.DataFrame,
    allowed_tags: list[str],
) -> pd.DataFrame:
    required = {
        VISIT_ID_COLUMN,
        PATIENT_CONTEXT_COLUMN,
        CHIEF_COMPLAINT_COLUMN,
        GOLD_TEST_TAG_COLUMN,
        GOLD_ICD_CODE_COLUMN,
        GOLD_ICD_TITLE_COLUMN,
    }
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"Prepared training data is missing columns: {sorted(missing)}")
    allowed = set(allowed_tags)
    rows = []
    unknown_tags: set[str] = set()
    for _, row in train.iterrows():
        tags = list(dict.fromkeys(split_terms(row.get(GOLD_TEST_TAG_COLUMN))))
        unknown_tags.update(tag for tag in tags if tag not in allowed)
        codes = split_terms(row.get(GOLD_ICD_CODE_COLUMN))
        titles = split_terms(row.get(GOLD_ICD_TITLE_COLUMN))
        if len(codes) != len(titles):
            visit_id = clean_text(row.get(VISIT_ID_COLUMN))
            raise ValueError(
                f"Prepared training visit {visit_id!r} has {len(codes)} gold ICD block codes "
                f"but {len(titles)} titles."
            )
        deduplicated_codes: list[str] = []
        deduplicated_titles: list[str] = []
        title_by_code: dict[str, str] = {}
        for code, title in zip(codes, titles, strict=True):
            existing_title = title_by_code.get(code)
            if existing_title is not None:
                if existing_title != title:
                    visit_id = clean_text(row.get(VISIT_ID_COLUMN))
                    raise ValueError(
                        f"Prepared training visit {visit_id!r} assigns conflicting titles "
                        f"to ICD block {code!r}: {existing_title!r} and {title!r}."
                    )
                continue
            title_by_code[code] = title
            deduplicated_codes.append(code)
            deduplicated_titles.append(title)
        rows.append(
            {
                "train_visit_row_id": clean_text(row.get(VISIT_ID_COLUMN)),
                PATIENT_CONTEXT_COLUMN: clean_text(row.get(PATIENT_CONTEXT_COLUMN)),
                CHIEF_COMPLAINT_COLUMN: clean_text(row.get(CHIEF_COMPLAINT_COLUMN)),
                "test_tags": tags,
                GOLD_ICD_CODE_COLUMN: deduplicated_codes,
                GOLD_ICD_TITLE_COLUMN: deduplicated_titles,
            }
        )
    if unknown_tags:
        raise ValueError(
            f"Prepared training data contains lab categories absent from the allowed vocabulary: "
            f"{sorted(unknown_tags)}"
        )
    return pd.DataFrame(rows)


def build_train_test_tag_priors(
    train_rows: pd.DataFrame,
    min_visits: int,
    top_n: int,
    min_rate: float,
) -> dict[str, Any]:
    cc_counts: dict[str, Counter] = {}
    cc_totals: Counter = Counter()
    icd_counts: dict[str, Counter] = {}
    icd_totals: Counter = Counter()
    icd_titles: dict[str, str] = {}
    for _, row in train_rows.iterrows():
        tags = row.get("test_tags", [])
        if not tags:
            continue
        for key in prior_cc_keys(row.get(CHIEF_COMPLAINT_COLUMN)):
            add_prior_counts(cc_counts, cc_totals, key, tags)
        codes = row.get(GOLD_ICD_CODE_COLUMN, [])
        titles = row.get(GOLD_ICD_TITLE_COLUMN, [])
        title_lookup = dict(zip(codes, titles, strict=False))
        for code in codes:
            add_prior_counts(icd_counts, icd_totals, code, tags)
            if title_lookup.get(code):
                icd_titles[code] = title_lookup[code]

    return {
        "cc": {
            key: {
                "visit_count": int(cc_totals[key]),
                "common_test_tags": format_prior_bucket(counter, int(cc_totals[key]), top_n, min_rate),
            }
            for key, counter in cc_counts.items()
            if cc_totals[key] >= min_visits and format_prior_bucket(counter, int(cc_totals[key]), top_n, min_rate)
        },
        "icd": {
            key: {
                "visit_count": int(icd_totals[key]),
                "icd_title": icd_titles.get(key, ""),
                "common_test_tags": format_prior_bucket(counter, int(icd_totals[key]), top_n, min_rate),
            }
            for key, counter in icd_counts.items()
            if icd_totals[key] >= min_visits and format_prior_bucket(counter, int(icd_totals[key]), top_n, min_rate)
        },
    }


def build_prior_evidence_for_visit(
    chief_complaint: str,
    predicted_icds: list[dict[str, str]],
    priors: dict[str, Any] | None,
    top_n: int,
    max_icds: int,
) -> dict[str, Any]:
    if not priors:
        return {}
    cc_rows = []
    seen_cc = set()
    for key in prior_cc_keys(chief_complaint):
        if key in seen_cc:
            continue
        seen_cc.add(key)
        prior = priors.get("cc", {}).get(key)
        if prior:
            cc_rows.append({"chief_complaint": key, **prior, "common_test_tags": prior["common_test_tags"][:top_n]})

    icd_rows = []
    seen_icd = set()
    for icd in predicted_icds[:max_icds]:
        code = clean_text(icd.get("code"))
        if not code or code in seen_icd:
            continue
        seen_icd.add(code)
        prior = priors.get("icd", {}).get(code)
        if prior:
            icd_rows.append(
                {
                    "rank": icd.get("rank", ""),
                    "code": code,
                    "title": clean_text(icd.get("title")) or prior.get("icd_title", ""),
                    "visit_count": prior["visit_count"],
                    "common_test_tags": prior["common_test_tags"][:top_n],
                }
            )
    evidence = {}
    if cc_rows:
        evidence["chief_complaint_priors"] = cc_rows
    if icd_rows:
        evidence["predicted_icd_priors"] = icd_rows
    return evidence


def build_context_rag_train_rows(
    train_rows: pd.DataFrame,
) -> pd.DataFrame:
    mask = train_rows[PATIENT_CONTEXT_COLUMN].ne("") & train_rows["test_tags"].map(bool)
    columns = [
        "train_visit_row_id",
        PATIENT_CONTEXT_COLUMN,
        CHIEF_COMPLAINT_COLUMN,
        GOLD_ICD_TITLE_COLUMN,
        "test_tags",
    ]
    return train_rows.loc[mask, columns].reset_index(drop=True)


def build_context_rag_index(
    train_rows: pd.DataFrame,
    embedding_batch_size: int,
    encoder: TextEncoder,
) -> dict[str, Any]:
    if train_rows.empty:
        raise ValueError("No training visits with patient_full_context and mapped test_tags were available for context RAG.")
    contexts = train_rows[PATIENT_CONTEXT_COLUMN].fillna("").tolist()
    embeddings = encoder._embed_many(contexts, batch_size=embedding_batch_size)
    matrix = np.stack([embedding.numpy() for embedding in embeddings]).astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-9, None)
    neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
    neighbors.fit(matrix)
    return {
        "train_rows": train_rows,
        "encoder": encoder,
        "matrix": matrix,
        "neighbors": neighbors,
        "embedding_batch_size": embedding_batch_size,
    }


def retrieve_context_rag_rows(patient_context: str, index: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    train_rows = index["train_rows"]
    top_k = min(max(int(top_k), 1), len(train_rows))

    embeddings = index["encoder"]._embed_many([patient_context])
    if embeddings is None:
        raise RuntimeError("Context RAG embedding retriever requires a loaded transformer encoder.")
    query = embeddings[0].numpy().astype("float32")
    query = query / max(float(np.linalg.norm(query)), 1e-9)
    distances, indices = index["neighbors"].kneighbors(query.reshape(1, -1), n_neighbors=top_k)

    rows = []
    for distance, row_index in zip(distances[0], indices[0], strict=False):
        row = train_rows.iloc[int(row_index)]
        rows.append(
            {
                "train_visit_row_id": clean_text(row.get("train_visit_row_id")),
                "similarity": round(1.0 - float(distance), 4),
                "chief_complaint": clean_text(row.get(CHIEF_COMPLAINT_COLUMN)),
                "diagnoses": join_unique(row.get(GOLD_ICD_TITLE_COLUMN, [])),
                "test_tags": row.get("test_tags", []),
            }
        )
    return rows


def build_context_rag_evidence_for_visit(
    patient_context: str,
    index: dict[str, Any] | None,
    top_k: int,
    prior_top_n: int,
    examples: int,
) -> dict[str, Any]:
    if not index:
        return {}
    retrieved = retrieve_context_rag_rows(patient_context, index, top_k)
    counter: Counter = Counter()
    for row in retrieved:
        counter.update(row.get("test_tags", []))
    common_tags = [
        {"test_tag": tag, "visit_count": int(count), "retrieved_count": len(retrieved)}
        for tag, count in counter.most_common(prior_top_n)
    ]
    representative = []
    for row in retrieved[: max(int(examples), 0)]:
        representative.append(
            {
                "similarity": row.get("similarity"),
                "chief_complaint": row.get("chief_complaint", ""),
                "diagnoses": row.get("diagnoses", ""),
                "test_tags": row.get("test_tags", []),
            }
        )
    evidence = {
        "retriever": "embedding",
        "retrieved_count": len(retrieved),
        "common_test_tags": common_tags,
    }
    if representative:
        evidence["representative_similar_visits"] = representative
    return evidence


def normalize_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the published Stage 1 prediction columns for mainline use."""
    required = {
        VISIT_ID_COLUMN,
        PREDICTION_RANK_COLUMN,
        PREDICTED_ICD_CODE_COLUMN,
        PREDICTED_ICD_TITLE_COLUMN,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Stage 1 predictions are missing columns: {sorted(missing)}")
    out = frame.copy()
    out["llm_icd_rank"] = (
        pd.to_numeric(out[PREDICTION_RANK_COLUMN], errors="coerce")
        .fillna(999999)
        .astype(int)
    )
    out["llm_icd_code"] = out[PREDICTED_ICD_CODE_COLUMN].map(clean_text)
    out["llm_icd_title"] = out[PREDICTED_ICD_TITLE_COLUMN].map(clean_text)
    return out


def build_triplets(
    review: pd.DataFrame,
    icd2tag: pd.DataFrame,
    allowed_tags: list[str],
    top_k_icd: int,
) -> pd.DataFrame:
    rows = []
    tag_by_code = {
        code: group[["icd_title", "test_tag"]].to_dict("records")
        for code, group in icd2tag.groupby("icd_code", sort=False)
    }
    allowed = set(allowed_tags)
    review["combined_rank_num"] = pd.to_numeric(review["llm_icd_rank"], errors="coerce").fillna(999999).astype(int)
    review = review.sort_values([VISIT_ID_COLUMN, "combined_rank_num"], ascending=[True, True])
    for visit_id, group in review.groupby(VISIT_ID_COLUMN, sort=False):
        for _, row in group.head(top_k_icd).iterrows():
            code = clean_text(row.get("llm_icd_code"))
            title = clean_text(row.get("llm_icd_title"))
            if not code:
                continue
            mapped = tag_by_code.get(code, [])
            for tag_row in mapped:
                tag = clean_text(tag_row.get("test_tag"))
                if tag not in allowed:
                    continue
                rows.append(
                    {
                        VISIT_ID_COLUMN: visit_id,
                        "combined_rank": row.get("llm_icd_rank", row.get("combined_rank", "")),
                        "icd_code": code,
                        "icd_title": title or clean_text(tag_row.get("icd_title")),
                        "test_tag": tag,
                        PATIENT_CONTEXT_COLUMN: row.get(PATIENT_CONTEXT_COLUMN, ""),
                    }
                )
    return pd.DataFrame(rows)


def summarize_icds(group: pd.DataFrame) -> list[dict[str, str]]:
    icds = []
    seen = set()
    for _, row in group.sort_values("combined_rank").iterrows():
        code = clean_text(row.get("icd_code"))
        if not code or code in seen:
            continue
        icds.append(
            {
                "rank": clean_text(row.get("combined_rank")),
                "code": code,
                "title": clean_text(row.get("icd_title")),
            }
        )
        seen.add(code)
    return icds


def summarize_candidate_tags(group: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for tag, tag_group in group.groupby("test_tag", sort=False):
        linked = []
        for _, row in tag_group.drop_duplicates(["icd_code"]).iterrows():
            linked.append(
                {
                    "code": clean_text(row.get("icd_code")),
                    "title": clean_text(row.get("icd_title")),
                    "rank": clean_text(row.get("combined_rank")),
                }
            )
        rows.append(
            {
                "test_tag": clean_text(tag),
                "linked_icds": linked,
            }
        )
    return rows


def format_ranked_icd(icd: dict[str, Any]) -> str:
    rank = clean_text(icd.get("rank"))
    code = clean_text(icd.get("code"))
    title = clean_text(icd.get("title"))
    prefix = f"rank {rank} " if rank else ""
    if code and title:
        return f"{prefix}{code} {title}"
    if code:
        return f"{prefix}{code}"
    return f"{prefix}{title}".strip()


def format_working_diagnoses(predicted_icds: list[dict[str, str]]) -> str:
    lines = []
    for icd in predicted_icds:
        text = format_ranked_icd(icd)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines) if lines else "- No ICD diagnosis candidates were available."


def format_candidate_tag_knowledge(candidate_tags: list[dict[str, Any]]) -> str:
    lines = []
    for row in candidate_tags:
        tag = clean_text(row.get("test_tag"))
        if not tag:
            continue
        linked = []
        for icd in row.get("linked_icds", []) or []:
            text = format_ranked_icd(icd)
            if text:
                linked.append(text)
        if linked:
            lines.append(f"- {tag}: clinically linked to {', '.join(linked)}.")
        else:
            lines.append(f"- {tag}: clinically linked to the predicted diagnosis set.")
    return "\n".join(lines) if lines else "- No clinician-curated diagnostic-to-test links were available."


def format_prior_tag_list(rows: list[dict[str, Any]]) -> str:
    tags = []
    for item in rows:
        tag = clean_text(item.get("test_tag"))
        if not tag:
            continue
        rate = item.get("visit_rate")
        try:
            percent = int(round(float(rate) * 100))
            tags.append(f"{tag} ({percent}%)")
        except (TypeError, ValueError):
            tags.append(tag)
    return ", ".join(tags)


def format_training_prior_evidence(training_prior_evidence: dict[str, Any] | None) -> str:
    if not training_prior_evidence:
        return ""
    lines = ["In similar disease visits from the local training data, clinicians commonly ordered:"]
    for row in training_prior_evidence.get("chief_complaint_priors", []) or []:
        tags = format_prior_tag_list(row.get("common_test_tags", []) or [])
        if not tags:
            continue
        complaint = clean_text(row.get("chief_complaint"))
        count = clean_text(row.get("visit_count"))
        count_text = f", n={count}" if count else ""
        lines.append(f"- Chief complaint {complaint}{count_text}: {tags}.")
    for row in training_prior_evidence.get("predicted_icd_priors", []) or []:
        tags = format_prior_tag_list(row.get("common_test_tags", []) or [])
        if not tags:
            continue
        code = clean_text(row.get("code"))
        title = clean_text(row.get("title"))
        count = clean_text(row.get("visit_count"))
        count_text = f", n={count}" if count else ""
        label = f"{code} {title}".strip()
        lines.append(f"- {label}{count_text}: {tags}.")
    return "\n".join(lines) if len(lines) > 1 else ""


def format_rag_tag_counts(context_rag_evidence: dict[str, Any] | None) -> str:
    if not context_rag_evidence:
        return ""
    retrieved_count = int(context_rag_evidence.get("retrieved_count", 0) or 0)
    if retrieved_count <= 0:
        return ""
    tag_parts = [
        f"{tag} ({int(row.get('visit_count', 0) or 0)}/{retrieved_count})"
        for row in context_rag_evidence.get("common_test_tags", []) or []
        if (tag := clean_text(row.get("test_tag")))
    ]
    return ", ".join(tag_parts)


def format_context_rag_evidence(context_rag_evidence: dict[str, Any] | None) -> str:
    tag_counts = format_rag_tag_counts(context_rag_evidence)
    if not tag_counts:
        return ""
    lines = [
        "In similar patient-context visits from the local training data, clinicians commonly ordered:",
        f"- Common ordered test_tags: {tag_counts}.",
    ]
    for idx, row in enumerate(context_rag_evidence.get("representative_similar_visits", []) or [], start=1):
        complaint = clean_text(row.get("chief_complaint")) or "unknown chief complaint"
        diagnosis = clean_text(row.get("diagnoses")) or "diagnosis not shown"
        tags = join_unique(row.get("test_tags", []))
        lines.append(f"- Representative similar visit {idx}: chief complaint {complaint}; diagnosis {diagnosis}; ordered {tags}.")
    return "\n".join(lines)


def parse_llm_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def recover_recommendations_from_text(text: str, allowed_tags: list[str]) -> dict[str, Any]:
    """Best-effort recovery when a local LLM returns nearly-JSON text."""
    recovered: list[dict[str, Any]] = []
    seen = set()
    for match in re.finditer(r'"test_tag"\s*:\s*"([^"]+)"', text):
        tag = clean_text(match.group(1))
        if tag not in allowed_tags or tag in seen:
            continue
        recovered.append(
            {
                "test_tag": tag,
                "rationale": "Recovered from non-JSON local model response.",
                "linked_icd_codes": [],
            }
        )
        seen.add(tag)

    if not recovered:
        for tag in allowed_tags:
            if re.search(rf"(?<![A-Za-z0-9/&-]){re.escape(tag)}(?![A-Za-z0-9/&-])", text):
                recovered.append(
                    {
                        "test_tag": tag,
                        "rationale": "Recovered from non-JSON local model response.",
                        "linked_icd_codes": [],
                    }
                )
                seen.add(tag)
    return {
        "recommended_tags": recovered,
        "final_explanation": "Recovered from non-JSON local model response." if recovered else "",
    }


def parse_or_recover_llm_response(text: str, allowed_tags: list[str]) -> dict[str, Any]:
    try:
        parsed = parse_llm_json(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError(
                "The response must be a JSON object.",
                text,
                0,
            )
        return parsed
    except json.JSONDecodeError:
        recovered = recover_recommendations_from_text(text, allowed_tags)
        if recovered.get("recommended_tags"):
            return recovered
        raise


def validate_recommendations(response: dict[str, Any], allowed_tags: list[str]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(response, dict):
        raise TypeError("The recommendation response must be a JSON object.")
    allowed = set(allowed_tags)
    valid = []
    invalid = []
    raw_recommendations = response.get("recommended_tags", []) or []
    if not isinstance(raw_recommendations, list):
        raise TypeError("recommended_tags must be a JSON list.")
    for item in raw_recommendations:
        if isinstance(item, str):
            cleaned = {
                "test_tag": clean_text(item),
                "rationale": "",
                "linked_icd_codes": [],
            }
        elif isinstance(item, dict):
            cleaned = dict(item)
        else:
            continue
        tag = clean_text(cleaned.get("test_tag"))
        if tag in allowed:
            cleaned["test_tag"] = tag
            valid.append(cleaned)
        elif tag:
            invalid.append(tag)
    cleaned_response = dict(response)
    cleaned_response["recommended_tags"] = valid
    return cleaned_response, invalid


def retry_prompt_for_invalid_json(prompt: str, response_text: str, allowed_tags: list[str]) -> str:
    return render_prompt(
        JSON_REPAIR_PROMPT,
        allowed_tags=json.dumps(allowed_tags, ensure_ascii=False),
        original_prompt=prompt,
        invalid_response=response_text[:4000],
    )


def retry_prompt_for_invalid_tags(prompt: str, invalid_tags: list[str]) -> str:
    return render_prompt(
        INVALID_TAG_REPAIR_PROMPT,
        original_prompt=prompt,
        invalid_tags=json.dumps(invalid_tags, ensure_ascii=False),
    )


def chat_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": prompt},
    ]


def build_transformers_caller(args: argparse.Namespace):
    model_name = args.hf_model_name
    print(
        "loading Transformers model:",
        model_name,
        f"(device_map={args.hf_device_map}, dtype=bfloat16)",
    )
    shared_model_kwargs = {"trust_remote_code": False}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **shared_model_kwargs)
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first %}{{ bos_token }}{% endif %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
            "{{ message['content'] | trim }}<|eot_id|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )
        print("warning: tokenizer.chat_template is not set; using Llama-3 fallback chat template.")
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.hf_device_map,
        **shared_model_kwargs,
    }
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    def render_template(messages: list[dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate_kwargs() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": args.hf_max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        return kwargs

    def decode_generated(outputs, input_length: int) -> list[str]:
        texts = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        return [text.strip() for text in texts]

    def generate_many(prompts: list[str], batch_size: int | None = None, desc: str | None = None) -> list[str]:
        if not prompts:
            return []
        batch_size = max(int(batch_size or getattr(args, "hf_generation_batch_size", 16) or 16), 1)
        outputs: list[str] = []
        iterator = range(0, len(prompts), batch_size)
        if desc and len(prompts) > batch_size:
            iterator = tqdm(
                list(iterator),
                total=(len(prompts) + batch_size - 1) // batch_size,
                desc=desc,
                unit="batch",
            )
        for start in iterator:
            chunk = prompts[start : start + batch_size]
            rendered = [render_template(chat_messages(prompt)) for prompt in chunk]
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            device = getattr(model, "device", None)
            if device is not None:
                inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, **generate_kwargs())
            outputs.extend(decode_generated(generated, inputs["input_ids"].shape[-1]))
        return outputs

    return generate_many
