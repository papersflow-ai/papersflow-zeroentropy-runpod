"""RunPod load-balancing HTTP server for PapersFlow's ZeroEntropy contract."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from inference_runtime import (
    EMBED_MODEL_ALIAS,
    EMBED_MODEL_REVISION,
    RERANK_MODEL_ALIAS,
    RERANK_MODEL_REVISION,
    ContractError,
    ZeroEntropyRuntime,
    embed_response,
    parse_embed_payload,
    parse_rerank_payload,
    rerank_response,
)


def create_app(
    models: ZeroEntropyRuntime | None = None, *, load_on_startup: bool = True
) -> FastAPI:
    runtime = models or ZeroEntropyRuntime()
    inference_lock = threading.Lock()
    startup: dict[str, float] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if load_on_startup:
            started = time.perf_counter()
            runtime.load()
            runtime.warmup()
            startup["seconds"] = time.perf_counter() - started
        yield

    app = FastAPI(
        title="PapersFlow ZeroEntropy Models",
        version="1",
        lifespan=lifespan,
    )

    @app.get("/ping")
    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "startup_seconds": startup.get("seconds"),
            "models": {
                EMBED_MODEL_ALIAS: EMBED_MODEL_REVISION,
                RERANK_MODEL_ALIAS: RERANK_MODEL_REVISION,
            },
        }

    @app.post("/v1/models/embed")
    def embed(payload: dict[str, object]) -> dict[str, object]:
        try:
            texts, input_type = parse_embed_payload(payload)
        except ContractError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        with inference_lock:
            output = runtime.embed(texts, input_type)
        return embed_response(output, texts)

    @app.post("/v1/models/rerank")
    def rerank(payload: dict[str, object]) -> dict[str, object]:
        request_started = time.perf_counter()
        try:
            query, documents, top_n = parse_rerank_payload(payload)
        except ContractError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        with inference_lock:
            inference_started = time.perf_counter()
            output = runtime.rerank(query, documents, top_n)
            inference_seconds = time.perf_counter() - inference_started
        return rerank_response(
            output,
            query,
            documents,
            e2e_latency=time.perf_counter() - request_started,
            inference_latency=inference_seconds,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")), workers=1)
