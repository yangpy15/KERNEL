"""Frozen Hugging Face text encoder for KERNEL retrieval indexes."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from transformers import AutoModel, AutoTokenizer

from common import clean_text


class TextEncoder:
    """Mean-pooled encoder."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        if not self.model_name:
            raise ValueError("Text encoder model_name must not be empty.")
        self.backend = ""
        self.tokenizer = None
        self.model = None
        self.device = None
        self.embedding_cache: dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            self.backend = f"hf_encoder:{self.model_name}"
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load text encoder {self.model_name!r}. Install transformers "
                "and make the model available locally."
            ) from exc

    def _embed_many(
        self, texts: Iterable[Any], batch_size: int = 32
    ) -> list[Any]:
        """Encode texts in batches and cache CPU tensors by exact input text."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text encoder is not initialized.")
        normalized = [clean_text(text) for text in texts]
        missing: list[str] = []
        seen: set[str] = set()
        for text in normalized:
            if text not in self.embedding_cache and text not in seen:
                missing.append(text)
                seen.add(text)

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (
                (token_embeddings * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )
            for text, embedding in zip(batch, embeddings.detach().cpu(), strict=False):
                self.embedding_cache[text] = embedding
        return [self.embedding_cache[text] for text in normalized]
