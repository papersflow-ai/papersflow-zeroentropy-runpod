"""Download the two immutable Hugging Face revisions into a container image."""

from __future__ import annotations

import argparse

from inference_runtime import (
    EMBED_MODEL_ID,
    EMBED_MODEL_PATH,
    EMBED_MODEL_REVISION,
    RERANK_MODEL_ID,
    RERANK_MODEL_PATH,
    RERANK_MODEL_REVISION,
)


def main() -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("embed", "rerank"))
    args = parser.parse_args()
    if args.model == "embed":
        repo_id, revision, path = EMBED_MODEL_ID, EMBED_MODEL_REVISION, EMBED_MODEL_PATH
    else:
        repo_id, revision, path = RERANK_MODEL_ID, RERANK_MODEL_REVISION, RERANK_MODEL_PATH
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=path)


if __name__ == "__main__":
    main()
