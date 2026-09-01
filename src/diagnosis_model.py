"""Stage 1 ICD cross-encoder training and prediction utilities for KERNEL."""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

from common import (
    GOLD_ICD_CODE_COLUMN,
    PATIENT_CONTEXT_COLUMN,
    PREDICTED_ICD_CODE_COLUMN,
    PREDICTED_ICD_TITLE_COLUMN,
    PREDICTION_RANK_COLUMN,
    VISIT_ID_COLUMN,
    clean_text,
    split_terms,
)


# Runtime configuration and visit-table loading

def resolve_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_reproducibility(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_random_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_visit_table(
    path: Path,
    context_column: str,
    limit: int | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="string")
    frame = frame.head(limit).copy() if limit is not None else frame.copy()
    required = {VISIT_ID_COLUMN, context_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required visit column(s): {sorted(missing)}")
    frame[VISIT_ID_COLUMN] = frame[VISIT_ID_COLUMN].map(clean_text)
    frame[context_column] = frame[context_column].map(clean_text)
    frame = frame.loc[frame[context_column].ne("")].reset_index(drop=True) # Remove empty patient context
    return frame


# ICD label vocabulary and training-target preparation

def load_icd_label_rows(path: Path) -> list[dict[str, Any]]:
    """Load the already-mapped ICD block vocabulary used by Stage 1."""
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {"parent_title", "parent_icd", "icd_title", "icd_code"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required ICD-label columns: {sorted(missing)}")
    return frame.to_dict("records")

def build_label_vocabulary(icd_ranges: list[dict[str, Any]]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    codes: list[str] = []
    titles: dict[str, str] = {}
    groups: dict[str, str] = {}
    for row in icd_ranges:
        code = clean_text(row.get("icd_code"))
        if not code or code in titles:
            continue
        codes.append(code)
        # `icd_title` is the block-level diagnostic label described in the paper.
        titles[code] = clean_text(row.get("icd_title"))
        groups[code] = (
            clean_text(row.get("parent_title"))
            or clean_text(row.get("parent_icd"))
            or code[:1]
        )
    if not codes:
        raise ValueError("The ICD title table did not produce any valid ICD labels.")
    return codes, titles, groups


def build_target_matrix(visits: pd.DataFrame, label_codes: list[str]):
    label_index = {code: index for index, code in enumerate(label_codes)}
    targets = torch.zeros((len(visits), len(label_codes)), dtype=torch.float32)
    for visit_index, value in enumerate(visits[GOLD_ICD_CODE_COLUMN].tolist()):
        for code in split_terms(value):
            label_index_value = label_index.get(code)
            if label_index_value is not None:
                targets[visit_index, label_index_value] = 1.0
    return targets


def prepare_labeled_visits(
    path: Path,
    context_column: str,
    label_codes: list[str],
    limit: int | None,
    split_name: str,
) -> tuple[pd.DataFrame, torch.Tensor]:
    visits = load_visit_table(path, context_column, limit)
    if GOLD_ICD_CODE_COLUMN not in visits.columns:
        raise ValueError(f"{path} is missing required column: {GOLD_ICD_CODE_COLUMN}")
    visits[GOLD_ICD_CODE_COLUMN] = visits[GOLD_ICD_CODE_COLUMN].map(clean_text)
    visits = visits.loc[visits[GOLD_ICD_CODE_COLUMN].ne("")].reset_index(drop=True)
    if visits.empty:
        raise RuntimeError(f"No {split_name} visits contain prepared ICD block labels.")
    allowed_codes = set(label_codes)
    unknown_codes = sorted(
        {
            code
            for value in visits[GOLD_ICD_CODE_COLUMN]
            for code in split_terms(value)
            if code not in allowed_codes
        }
    )
    if unknown_codes:
        raise ValueError(
            f"{path} contains ICD block codes absent from the label vocabulary: {unknown_codes[:10]}"
        )
    return visits, build_target_matrix(visits, label_codes)


def context_texts(visits: pd.DataFrame, context_column: str) -> list[str]:
    return visits[context_column].map(clean_text).tolist()


# Cross-encoder construction and shared encoding utilities


def build_label_descriptions(
    label_codes: list[str],
    label_groups: dict[str, str],
) -> dict[str, str]:
    return {
        code: f"ICD group: {label_groups[code]}" if clean_text(label_groups.get(code)) else ""
        for code in label_codes
    }

# Combine the ICD block code, title, and description as the label-side input.
def label_pair_text(code: str, title: str, description: str = "") -> str:
    base = f"ICD block {clean_text(code)}: {clean_text(title)}"
    description = clean_text(description)
    return f"{base}. {description}" if description else base


def build_label_texts(
    label_codes: list[str],
    label_titles: dict[str, str],
    label_descriptions: dict[str, str],
) -> list[str]:
    return [
        label_pair_text(code, label_titles.get(code, ""), label_descriptions.get(code, ""))
        for code in label_codes
    ]


def make_cross_encoder(model_name: str, dropout: float, pooling: str):
    class DiagnosisCrossEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.pooling = pooling
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(int(self.encoder.config.hidden_size), 1)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            output = self.encoder(**inputs)
            if self.pooling == "cls":
                pooled = output.last_hidden_state[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1).expand(output.last_hidden_state.size()).float()
                pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.head(self.dropout(pooled)).squeeze(-1)

    return DiagnosisCrossEncoder()

# left = patient context; right = ICD description
def tokenize_pairs(tokenizer, left: list[str], right: list[str], max_length: int, device):
    encoded = tokenizer(
        left,
        right,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    return {key: value.to(device) for key, value in encoded.items() if key in allowed}

# mixed precision
def autocast_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device, enabled: bool):
    use_scaler = enabled and device.type == "cuda"
    return torch.amp.GradScaler("cuda", enabled=use_scaler)


# Training-pair sampling
# For each visit, retain positive labels, and then extract some hard negatives and random negatives
def sample_training_pairs(
    targets,
    label_codes: list[str],
    label_groups: dict[str, str],
    hard_negatives_per_visit: int,
    random_negatives_per_visit: int,
    max_positives_per_visit: int | None,
    rng: random.Random,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[int, int, float]] = []
    all_indices = list(range(len(label_codes)))
    for visit_index in range(targets.size(0)):
        positive_indices = targets[visit_index].nonzero(as_tuple=False).flatten().tolist()
        if not positive_indices:
            continue
        if max_positives_per_visit is not None and len(positive_indices) > max_positives_per_visit:
            positive_indices = rng.sample(positive_indices, max_positives_per_visit)
        positive_set = set(positive_indices)
        pairs.extend((visit_index, label_index, 1.0) for label_index in positive_indices)

        positive_groups = {label_groups[label_codes[index]] for index in positive_indices}

        # hard negative and positive ICDs belong to the same group
        hard_pool = [
            index
            for index in all_indices
            if index not in positive_set and label_groups[label_codes[index]] in positive_groups
        ]
        hard_count = min(hard_negatives_per_visit, len(hard_pool))
        hard_negatives = rng.sample(hard_pool, hard_count) if hard_count else []

        remaining = [index for index in all_indices if index not in positive_set and index not in hard_negatives]
        random_count = min(random_negatives_per_visit, len(remaining))
        random_negatives = rng.sample(remaining, random_count) if random_count else []
        pairs.extend((visit_index, label_index, 0.0) for label_index in hard_negatives + random_negatives)
    rng.shuffle(pairs)
    return pairs


# Shared Stage 1 scoring

def score_contexts(
    model,
    tokenizer,
    contexts: list[str],
    label_texts: list[str],
    batch_size: int,
    max_length: int,
    device,
    description: str,
):
    rows = []
    model.eval()
    with torch.no_grad():
        for context in tqdm(
            contexts,
            total=len(contexts),
            desc=description,
            unit="visit",
        ):
            scores = []
            for start in range(0, len(label_texts), batch_size):
                right = label_texts[start : start + batch_size]
                encoded = tokenize_pairs(tokenizer, [context] * len(right), right, max_length, device)
                scores.append(model(**encoded).detach().cpu())
            rows.append(torch.cat(scores) if scores else torch.empty(0))
    return torch.stack(rows) if rows else torch.empty((0, len(label_texts)))


def sigmoid_relevance_scores(logits):
    """Convert BCE logits to independent, uncalibrated relevance scores."""
    return torch.sigmoid(logits)


# Validation and checkpoint saving

# hit@K measures whether at least one gold diagnosis is included among the Top-K predictions.
def validation_hit_at_k(
    model,
    tokenizer,
    visits: pd.DataFrame,
    targets,
    label_texts: list[str],
    context_column: str,
    batch_size: int,
    max_length: int,
    top_k: int,
    device,
) -> dict[str, float | int]:
    logits = score_contexts(
        model,
        tokenizer,
        context_texts(visits, context_column),
        label_texts,
        batch_size,
        max_length,
        device,
        "validate Stage 1",
    )
    if logits.numel() == 0:
        return {"n": 0, "hit": 0, "rate": 0.0}
    k = min(max(top_k, 1), len(label_texts))
    top_indices = logits.topk(k=k, dim=1).indices
    hits = 0
    for row_index in range(len(visits)):
        gold = set(targets[row_index].nonzero(as_tuple=False).flatten().tolist())
        predicted = set(top_indices[row_index].tolist())
        hits += int(bool(gold & predicted))
    return {"n": len(visits), "hit": hits, "rate": hits / max(len(visits), 1)}


def save_checkpoint(
    path: Path,
    model,
    label_codes: list[str],
    label_titles: dict[str, str],
    label_descriptions: dict[str, str],
    max_length: int,
    epoch: int,
    metrics: dict[str, float | int],
    *,
    model_name: str,
    context_column: str,
    seed: int,
    validation_top_k: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "dropout": float(model.dropout.p),
            "cross_encoder_pooling": model.pooling,
            "label_codes": label_codes,
            "label_titles": label_titles,
            "label_descriptions": label_descriptions,
            "context_column": context_column,
            "objective": "patient_context_icd_cross_encoder",
            "cross_encoder": True,
            "max_length": max_length,
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "best_epoch": epoch,
            "validation_top_k": validation_top_k,
            "validation_hit_at_k": metrics.get("hit", 0),
            "validation_n": metrics.get("n", 0),
            "validation_hit_at_k_rate": metrics.get("rate", 0.0),
            "score_transform": "sigmoid",
            "score_interpretation": "uncalibrated_pairwise_relevance",
        },
        path,
    )


# Stage 1 training

def train_diagnosis_cross_encoder(args) -> None:
    context_column = PATIENT_CONTEXT_COLUMN
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    pooling = "mean"
    dropout = 0.1
    batch_size = 32
    eval_batch_size = 64
    encoder_learning_rate = 2e-5
    classifier_learning_rate = 1e-4
    weight_decay = 1e-4
    positive_weight = 4.0
    hard_negatives_per_visit = 8
    random_negatives_per_visit = 12
    max_positives_per_visit = None
    validation_top_k = 15
    mixed_precision = True
    max_grad_norm = 1.0
    seed = 13

    configure_reproducibility(seed)
    device = resolve_device(args.device)

    icd_label_rows = load_icd_label_rows(args.icd_label_path)
    label_codes, label_titles, label_groups = build_label_vocabulary(icd_label_rows)
    label_descriptions = build_label_descriptions(label_codes, label_groups)

    train_visits, train_targets = prepare_labeled_visits(
        args.train_path,
        context_column,
        label_codes,
        args.n_train_visits,
        "training",
    )
    validation_visits, validation_targets = prepare_labeled_visits(
        args.validation_path,
        context_column,
        label_codes,
        args.n_validation_visits,
        "validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = make_cross_encoder(model_name, dropout, pooling).to(device)
    encoder_parameters = []
    classifier_parameters = []
    for name, parameter in model.named_parameters():
        (encoder_parameters if name.startswith("encoder.") else classifier_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": classifier_parameters, "lr": classifier_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    scaler = make_grad_scaler(device, mixed_precision)
    contexts = context_texts(train_visits, context_column)
    label_texts = build_label_texts(label_codes, label_titles, label_descriptions)
    best_hit = -1
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        pairs = sample_training_pairs(
            train_targets,
            label_codes,
            label_groups,
            hard_negatives_per_visit,
            random_negatives_per_visit,
            max_positives_per_visit,
            random.Random(seed + epoch - 1),
        )
        starts = list(range(0, len(pairs), batch_size))
        model.train()
        total_loss = 0.0
        for start in tqdm(
            starts,
            total=len(starts),
            desc=f"Stage 1 epoch {epoch}",
            unit="batch",
        ):
            batch = pairs[start : start + batch_size]
            left = [contexts[visit_index] for visit_index, _, _ in batch]
            right = [label_texts[label_index] for _, label_index, _ in batch]
            labels = torch.tensor([target for _, _, target in batch], dtype=torch.float32, device=device)
            encoded = tokenize_pairs(tokenizer, left, right, args.max_length, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, mixed_precision):
                logits = model(**encoded)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item())

        average_loss = total_loss / max(len(starts), 1)
        metrics = validation_hit_at_k(
            model,
            tokenizer,
            validation_visits,
            validation_targets,
            label_texts,
            context_column,
            eval_batch_size,
            args.max_length,
            validation_top_k,
            device,
        )
        print(
            f"epoch {epoch} loss={average_loss:.4f} "
            f"val_hit@{validation_top_k}={metrics['hit']}/{metrics['n']} ({metrics['rate']:.1%})"
        )
        if int(metrics["hit"]) > best_hit or (int(metrics["hit"]) == best_hit and average_loss < best_loss):
            best_hit = int(metrics["hit"])
            best_loss = average_loss
            save_checkpoint(
                args.checkpoint_path,
                model,
                label_codes,
                label_titles,
                label_descriptions,
                args.max_length,
                epoch,
                metrics,
                model_name=model_name,
                context_column=context_column,
                seed=seed,
                validation_top_k=validation_top_k,
            )
            print("saved best Stage 1 checkpoint:", args.checkpoint_path)


# Checkpoint loading and Stage 1 inference

def load_checkpoint(path: Path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    required_model_settings = (
        "model_name",
        "dropout",
        "cross_encoder_pooling",
        "context_column",
    )
    missing_settings = [key for key in required_model_settings if checkpoint.get(key) in (None, "")]
    if missing_settings:
        raise ValueError(f"{path} is missing model settings: {', '.join(missing_settings)}.")
    model_name = clean_text(checkpoint["model_name"])
    pooling = clean_text(checkpoint["cross_encoder_pooling"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = make_cross_encoder(
        model_name,
        float(checkpoint["dropout"]),
        pooling,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, tokenizer, model


def predict_diagnoses(args) -> pd.DataFrame:
    batch_size = 64
    device = resolve_device("auto")
    checkpoint, tokenizer, model = load_checkpoint(args.checkpoint_path, device)
    context_column = clean_text(checkpoint["context_column"])
    checkpoint_max_length = checkpoint.get("max_length")
    max_length = (
        args.max_length
        if args.max_length is not None
        else int(checkpoint_max_length or 0)
    )
    if max_length <= 0:
        raise ValueError(
            f"{args.checkpoint_path} does not contain a valid max_length; "
            "provide --max-length explicitly."
        )
    visits = load_visit_table(args.input_path, context_column, limit=args.n_visits)
    if visits.empty:
        raise RuntimeError(
            f"{args.input_path} does not contain any visits with non-empty {context_column}."
        )
    label_codes = [
        clean_text(value)
        for value in checkpoint.get("label_codes", [])
        if clean_text(value)
    ]
    label_titles = {
        clean_text(key): clean_text(value)
        for key, value in checkpoint.get("label_titles", {}).items()
    }
    label_descriptions = {
        clean_text(key): clean_text(value)
        for key, value in checkpoint.get("label_descriptions", {}).items()
    }
    if not label_codes:
        raise ValueError(f"{args.checkpoint_path} does not contain ICD label codes.")
    label_texts = build_label_texts(label_codes, label_titles, label_descriptions)
    logits = score_contexts(
        model,
        tokenizer,
        context_texts(visits, context_column),
        label_texts,
        batch_size,
        max_length,
        device,
        "predict Stage 1",
    )
    relevance_scores = sigmoid_relevance_scores(logits)
    top_k = min(max(args.top_k, 1), len(label_codes))
    top_scores, top_indices = relevance_scores.topk(k=top_k, dim=1)
    rows = []
    for visit_index, visit in visits.iterrows():
        for rank, (score, label_index) in enumerate(
            zip(top_scores[visit_index].tolist(), top_indices[visit_index].tolist()),
            start=1,
        ):
            code = label_codes[label_index]
            rows.append(
                {
                    VISIT_ID_COLUMN: clean_text(visit.get(VISIT_ID_COLUMN)),
                    PREDICTION_RANK_COLUMN: rank,
                    PREDICTED_ICD_CODE_COLUMN: code,
                    PREDICTED_ICD_TITLE_COLUMN: label_titles.get(code, ""),
                    "prediction_score": float(score),
                    context_column: clean_text(visit.get(context_column)),
                }
            )
    output = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print("saved Stage 1 predictions:", args.output_path, len(output))
    return output


__all__ = [
    "predict_diagnoses",
    "train_diagnosis_cross_encoder",
]
