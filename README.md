# PapersFlow ZeroEntropy self-hosting

This app serves the Apache-2.0 `zeroentropy/zembed-1-embedding` and
`zeroentropy/zerank-2-reranker` weights behind one GPU process. The Modal and
RunPod entrypoints preserve PapersFlow's existing `/v1/models/embed` and
`/v1/models/rerank` JSON contracts.

Both deployments are development-only and scale to zero. Modal uses a pinned
A10G, a two-second scale-down window, and a Volume for the weights. RunPod uses
a load-balancing endpoint with zero active workers, one maximum worker, and its
five-second minimum idle timeout. The RunPod image bakes both immutable model
snapshots into the image, so it needs no continuously billed network volume.

## Modal development deploy

From the repository root, after `modal setup`:

```bash
pnpm --filter @papersflow/zeroentropy-modal models:download
pnpm --filter @papersflow/zeroentropy-modal deploy:dev
```

The first command downloads the two pinned Hugging Face revisions into the
`papersflow-zeroentropy-model-cache-dev-v1` volume. The second deploys the
authenticated ASGI endpoint in the current Modal workspace's `main`
environment.

## RunPod image

The reproducible local build is:

```bash
pnpm --filter @papersflow/zeroentropy-modal image:build:runpod
```

The manual `ZeroEntropy RunPod Image` workflow publishes an immutable image
tagged with the source commit. The endpoint must expose port 80 as HTTP and use
the image's default command. No Hugging Face credential or model volume is
required at runtime because the revisions are downloaded during the image
build. Keep active/minimum workers at zero to avoid idle GPU spend.

Set `ZEROENTROPY_API_BASE_URL` to the selected provider's HTTPS origin, without
a path, query, or fragment. Store that provider's bearer value as
`ZEROENTROPY_INFERENCE_TOKEN`. PapersFlow requires a valid origin/token pair and
otherwise keeps its existing BM25 fallback; it never sends a hosted-provider
credential to a self-hosted origin.

## Verify and benchmark

With the base URL loaded and the applicable provider token mapped to the
benchmark-only `ZEROENTROPY_BENCHMARK_TOKEN` shell variable without echoing it:

```bash
uv run --project apps/zeroentropy-modal --locked \
  python apps/zeroentropy-modal/benchmark.py \
  --compute-profile runpod-flex-24gb --latency-runs 5

uv run --project apps/zeroentropy-modal --locked \
  python apps/zeroentropy-modal/benchmark.py \
  --compute-profile runpod-flex-24gb --latency-runs 5 --quality
```

Latency output reports HTTP time to first byte and complete end-to-end time.
Because these endpoints return one non-streaming JSON response, time to first
byte occurs after inference; it is the closest meaningful TTFT analogue. The
quality run evaluates embedding-only and top-50 reranked retrieval on the
revision-pinned 50-query NanoSciFact split and reports Recall@10, MRR@10, and
nDCG@10. Select `modal-a10g` or `runpod-flex-24gb` to report the matching
compute estimate and idle tail alongside ZeroEntropy's published per-token
prices. Provider billing remains authoritative.
