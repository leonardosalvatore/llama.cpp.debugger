"""HTTP client for a llama-server running with ``--embeddings``.

We piggy-back on the ``openai`` package (already a dependency of the chat
client) and just point its ``base_url`` at the embedding server. That gives
us batching, retries, and the standard ``client.embeddings.create(...)``
shape for free.

Three quirks specific to llama.cpp + nomic-embed-text that this module hides:

1. **Pooling**: the embedding server must be started with
   ``--pooling mean|cls|last``. With ``--pooling none`` the OpenAI-compatible
   ``/v1/embeddings`` endpoint returns an error. We don't validate this
   client-side; if the server is misconfigured the first request fails with
   a clear 400.

2. **Nomic prefix convention**: nomic-embed-text-v1.5 is trained with task
   prefixes - documents go in as ``"search_document: <text>"`` and queries
   as ``"search_query: <text>"``. Without these prefixes retrieval quality
   drops noticeably. ``EmbeddingClient`` adds them automatically when
   ``model_family == "nomic"`` (the default). For bge / e5 / generic models
   set ``model_family="generic"`` and the text is sent verbatim.

3. **Per-input ubatch limit**: llama-server in embedding mode rejects any
   single input that exceeds ``--ubatch-size`` (default 512, we bump to
   2048 in start-llama-embedding-server.sh) with HTTP 500 ``"input (N
   tokens) is too large to process"``. The ingest chunker tries hard to
   stay under this with a char-cap, but token density is content-dependent
   (Lottie JSON-as-C-string blobs and dense draw code tokenize at
   ~1.85 chars/token) and one outlier can break a 2k-chunk build. The
   client therefore catches the 500 and retries with progressively smaller
   batches / hard-truncated text instead of bubbling the failure up.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np
from openai import APIStatusError, OpenAI


@dataclass
class EmbeddingClient:
    """Talks to a single llama-server ``/v1/embeddings`` endpoint."""

    host: str = "127.0.0.1"
    port: int = 53426
    # llama.cpp ignores the `model` field on /v1/embeddings (it serves
    # whatever was loaded with --model), so any non-empty string works.
    # We still accept one so downstream tooling that does care can pass it.
    model: str = "nomic-embed-text-v1.5"
    # "nomic" -> prepend search_document: / search_query:
    # "generic" -> send text as-is (bge, e5, generic chat models)
    model_family: str = "nomic"
    # Per-request batch size. llama-server batches internally too; this is
    # mostly to keep individual HTTP bodies sane.
    batch_size: int = 32
    timeout: float = 120.0
    api_key: str = "sk-no-key-required"  # llama-server doesn't check it.

    # Internal: cached embedding dim, populated lazily by the recovery
    # path when it needs to emit a zero-vector placeholder. Not a
    # constructor arg.
    _dim_cache: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _wrap(self, text: str, *, role: str) -> str:
        if self.model_family != "nomic":
            return text
        prefix = "search_document: " if role == "document" else "search_query: "
        return prefix + text

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of corpus chunks. Returns float32 ``(N, dim)``."""
        return self._embed(texts, role="document")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns float32 ``(dim,)``."""
        vecs = self._embed([text], role="query")
        return vecs[0]

    def _embed(self, texts: Sequence[str], *, role: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        rows: List[np.ndarray] = []
        for batch in _chunks(list(texts), self.batch_size):
            payload = [self._wrap(t, role=role) for t in batch]
            for vec in self._embed_with_recovery(payload):
                rows.append(vec)

        out = np.vstack(rows)
        # Normalize to unit length so downstream cosine == dot product. Both
        # sqlite-vec's vec_distance_cosine and our numpy fallback handle
        # un-normalized inputs, but normalizing now keeps later math cheap
        # and makes scores directly comparable across runs.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)

    # ------------------------------------------------------------------
    # Failure recovery: per-batch -> per-pair -> per-item -> truncate
    # ------------------------------------------------------------------

    def _embed_with_recovery(self, payload: List[str]) -> List[np.ndarray]:
        """Wrap ``embeddings.create`` with progressive fallback for the
        "input too large" 500 we get from llama-server when one chunk
        exceeds --ubatch-size.

        Strategy on a "too large" 500:
          1. If batch has >1 item: bisect, retry each half. Avoids
             re-embedding 31 fine chunks just because chunk #32 is huge.
          2. If batch is a single item: hard-truncate to half its current
             char length and retry. We log a warning so the operator knows
             retrieval for that chunk will be partial.
          3. If even the truncated single item fails (extremely unlikely),
             fall back to returning a zero vector so the build can finish
             - the chunk is effectively unsearchable but the corpus stays
             consistent.
        """
        try:
            resp = self._client.embeddings.create(
                model=self.model,
                input=payload,
            )
            return [np.asarray(item.embedding, dtype=np.float32)
                    for item in resp.data]
        except APIStatusError as exc:
            msg = str(exc).lower()
            if "too large" not in msg and exc.status_code != 500:
                raise

            if len(payload) > 1:
                mid = len(payload) // 2
                left = self._embed_with_recovery(payload[:mid])
                right = self._embed_with_recovery(payload[mid:])
                return left + right

            text = payload[0]
            if len(text) > 200:
                cut = max(200, len(text) // 2)
                print(
                    f"[embed] WARN: chunk too large for ubatch "
                    f"({len(text)} chars); truncating to {cut} chars and "
                    f"retrying. Consider lowering C_CHUNK_MAX_CHARS or "
                    f"raising llama-server --ubatch-size.",
                    file=sys.stderr,
                )
                return self._embed_with_recovery([text[:cut]])

            # Already short and still failing - emit a zero placeholder
            # rather than crashing the whole build.
            print(
                f"[embed] ERROR: chunk of {len(text)} chars still too large; "
                f"emitting zero vector so the build can continue.",
                file=sys.stderr,
            )
            dim = self._cached_dim()
            return [np.zeros(dim, dtype=np.float32)]

    def _cached_dim(self) -> int:
        """Lazy probe so the recovery path can size a placeholder vector
        without forcing an extra round-trip on the happy path."""
        if self._dim_cache:
            return self._dim_cache
        self._dim_cache = self.probe_dim()
        return self._dim_cache

    def probe_dim(self) -> int:
        """One-shot embed of a dummy string to discover the model's dim."""
        v = self.embed_query("ping")
        return int(v.shape[0])


def _chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]
