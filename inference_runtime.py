"""Shared PapersFlow contract and GPU runtime for ZeroEntropy open-weight models."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

EMBED_MODEL_ALIAS = "zembed-1"
EMBED_MODEL_ID = "zeroentropy/zembed-1-embedding"
EMBED_MODEL_REVISION = "cf13c81f3274394053d166740294f7eea4586f7a"
EMBED_DIMENSIONS = 2_560
EMBED_MAX_BATCH = 64
EMBED_MAX_INPUT_CHARS = 8_000

RERANK_MODEL_ALIAS = "zerank-2"
RERANK_MODEL_ID = "zeroentropy/zerank-2-reranker"
RERANK_MODEL_REVISION = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
RERANK_MAX_DOCUMENTS = 100
RERANK_MAX_DOCUMENT_CHARS = 1_000
RERANK_MAX_QUERY_CHARS = 8_000

MODEL_ROOT = Path("/models")
EMBED_MODEL_PATH = MODEL_ROOT / "zembed-1-embedding"
RERANK_MODEL_PATH = MODEL_ROOT / "zerank-2-reranker"


class ContractError(ValueError):
    """A caller payload does not match PapersFlow's bounded model contract."""


def _require_object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ContractError("request body must be a JSON object")
    return payload


def parse_embed_payload(payload: object) -> tuple[list[str], str]:
    body = _require_object(payload)
    if body.get("model") != EMBED_MODEL_ALIAS:
        raise ContractError(f"model must be {EMBED_MODEL_ALIAS}")
    if body.get("dimensions") != EMBED_DIMENSIONS:
        raise ContractError(f"dimensions must be {EMBED_DIMENSIONS}")
    input_type = body.get("input_type")
    if input_type not in {"document", "query"}:
        raise ContractError("input_type must be document or query")
    texts = body.get("input")
    if not isinstance(texts, list) or not 1 <= len(texts) <= EMBED_MAX_BATCH:
        raise ContractError(f"input must contain between 1 and {EMBED_MAX_BATCH} strings")
    if any(not isinstance(text, str) or len(text) > EMBED_MAX_INPUT_CHARS for text in texts):
        raise ContractError(
            f"each input must be a string no longer than {EMBED_MAX_INPUT_CHARS} characters"
        )
    return texts, input_type


def parse_rerank_payload(payload: object) -> tuple[str, list[str], int]:
    body = _require_object(payload)
    if body.get("model") != RERANK_MODEL_ALIAS:
        raise ContractError(f"model must be {RERANK_MODEL_ALIAS}")
    query = body.get("query")
    if not isinstance(query, str) or len(query) > RERANK_MAX_QUERY_CHARS:
        raise ContractError(
            f"query must be a string no longer than {RERANK_MAX_QUERY_CHARS} characters"
        )
    documents = body.get("documents")
    if not isinstance(documents, list) or not 1 <= len(documents) <= RERANK_MAX_DOCUMENTS:
        raise ContractError(f"documents must contain between 1 and {RERANK_MAX_DOCUMENTS} strings")
    if any(
        not isinstance(document, str) or len(document) > RERANK_MAX_DOCUMENT_CHARS
        for document in documents
    ):
        raise ContractError(
            f"each document must be a string no longer than {RERANK_MAX_DOCUMENT_CHARS} characters"
        )
    top_n = body.get("top_n", len(documents))
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= len(documents):
        raise ContractError("top_n must be an integer between 1 and the document count")
    return query, documents, top_n


def calibrated_relevance_score(logit: float) -> float:
    """Map zerank-2's raw Yes logit to its historical 0-1 API score."""
    if not math.isfinite(logit):
        raise RuntimeError("reranker returned a non-finite score")
    scaled = logit / 5
    if scaled >= 0:
        return 1 / (1 + math.exp(-scaled))
    exponent = math.exp(scaled)
    return exponent / (1 + exponent)


def ranked_scores(scores: list[float], top_n: int) -> list[dict[str, int | float]]:
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("reranker returned a non-finite score")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_n]
    return [{"index": index, "relevance_score": scores[index]} for index in order]


def embed_request_bytes(texts: list[str]) -> int:
    return sum(len(text.encode("utf-8")) for text in texts)


def rerank_request_bytes(query: str, documents: list[str]) -> int:
    query_bytes = len(query.encode("utf-8"))
    return sum(150 + query_bytes + len(document.encode("utf-8")) for document in documents)


def embed_response(output: dict[str, Any], texts: list[str]) -> dict[str, object]:
    return {
        "results": [{"embedding": embedding} for embedding in output["embeddings"]],
        "usage": {
            "total_bytes": embed_request_bytes(texts),
            "total_tokens": output["input_tokens"],
        },
    }


def rerank_response(
    output: dict[str, Any],
    query: str,
    documents: list[str],
    *,
    e2e_latency: float,
    inference_latency: float,
) -> dict[str, object]:
    return {
        "results": output["results"],
        "total_bytes": rerank_request_bytes(query, documents),
        "total_tokens": output["input_tokens"],
        "actual_latency_mode": "fast",
        "e2e_latency": e2e_latency,
        "inference_latency": inference_latency,
    }


class ZeroEntropyRuntime:
    """Load both pinned BF16 models once and serve bounded inference calls."""

    def __init__(
        self,
        *,
        embed_model_path: Path = EMBED_MODEL_PATH,
        rerank_model_path: Path = RERANK_MODEL_PATH,
        device: str = "cuda",
    ) -> None:
        self.embed_model_path = embed_model_path
        self.rerank_model_path = rerank_model_path
        self.device = device

    def load(self) -> None:
        if not self.embed_model_path.is_dir() or not self.rerank_model_path.is_dir():
            raise RuntimeError("model weights are missing from the container")

        import numpy as np
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer

        torch.set_float32_matmul_precision("high")
        with ThreadPoolExecutor(max_workers=2) as executor:
            embedder = executor.submit(
                SentenceTransformer,
                str(self.embed_model_path),
                device=self.device,
                local_files_only=True,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            reranker = executor.submit(
                CrossEncoder,
                str(self.rerank_model_path),
                device=self.device,
                local_files_only=True,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            self.embedder = embedder.result()
            self.reranker = reranker.result()
        self.np = np
        self.torch = torch

    def warmup(self) -> None:
        self.embed(["PapersFlow warmup"], "query")
        self.rerank("PapersFlow warmup", ["PapersFlow warmup"], 1)

    def embed(self, texts: list[str], input_type: str) -> dict[str, Any]:
        encode = (
            self.embedder.encode_query if input_type == "query" else self.embedder.encode_document
        )
        prompt_names = (
            ("query", "question") if input_type == "query" else ("document", "passage", "corpus")
        )
        prompt = next(
            (self.embedder.prompts[name] for name in prompt_names if name in self.embedder.prompts),
            "",
        )
        encoded_inputs = self.embedder.tokenizer(
            [f"{prompt}{text}" for text in texts],
            add_special_tokens=True,
            max_length=self.embedder.max_seq_length,
            return_length=True,
            truncation=True,
        )
        input_tokens = sum(int(length) for length in encoded_inputs["length"])
        with self.torch.inference_mode():
            embeddings = encode(
                texts,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                truncate_dim=EMBED_DIMENSIONS,
                show_progress_bar=False,
            )
        matrix = self.np.asarray(embeddings, dtype=self.np.float32)
        if matrix.shape != (len(texts), EMBED_DIMENSIONS) or not self.np.isfinite(matrix).all():
            raise RuntimeError("embedder returned an invalid matrix")
        return {"embeddings": matrix.tolist(), "input_tokens": input_tokens}

    def rerank(self, query: str, documents: list[str], top_n: int) -> dict[str, Any]:
        pairs = [(query, document) for document in documents]
        encoded_pairs = self.reranker.tokenizer(
            [query] * len(documents),
            documents,
            add_special_tokens=True,
            max_length=self.reranker.max_seq_length,
            return_length=True,
            truncation=True,
        )
        input_tokens = sum(int(length) for length in encoded_pairs["length"])
        with self.torch.inference_mode():
            predictions = self.reranker.predict(
                pairs,
                batch_size=8,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        raw_scores = self.np.asarray(predictions, dtype=self.np.float32).reshape(-1).tolist()
        if len(raw_scores) != len(documents):
            raise RuntimeError("reranker returned a mismatched score batch")
        scores = [calibrated_relevance_score(score) for score in raw_scores]
        return {"results": ranked_scores(scores, top_n), "input_tokens": input_tokens}
