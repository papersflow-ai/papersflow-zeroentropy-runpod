FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

LABEL org.opencontainers.image.source="https://github.com/papersflow-ai/papersflow" \
    org.opencontainers.image.description="Pinned ZeroEntropy embedding and reranking for PapersFlow"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/tmp/huggingface-cache \
    HF_HUB_CACHE=/tmp/huggingface-cache/hub \
    HF_XET_HIGH_PERFORMANCE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install \
        "fastapi==0.141.1" \
        "huggingface-hub[hf_xet]==1.29.0" \
        "pydantic==2.13.5" \
        "sentence-transformers==6.0.0" \
        "torch==2.13.0" \
        "uvicorn[standard]==0.52.4"

WORKDIR /service
COPY inference_runtime.py download_models.py ./

RUN python download_models.py embed
RUN python download_models.py rerank \
    && rm -rf /tmp/huggingface-cache /models/*/.cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY runpod_app.py ./

EXPOSE 80
CMD ["python", "runpod_app.py"]
