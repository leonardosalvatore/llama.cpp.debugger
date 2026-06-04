"""Write journald output into the vector store (UI-private).

This is the "add logs to the DB" write path. It lives in the UI
subpackage rather than ``systemd_mcp.vectordb`` on purpose: the chat
MCP server stays read-only on the store (it only ever calls
``rag_search``), and the proper log-namespacing design (a ``kind``
column, ``rag_ingest_*`` tools) is still future work documented in
``systemd_mcp/vectordb/README.md``. Until then logs and LVGL docs
share one ``chunks`` table and are distinguished purely by a synthetic
``log://`` path prefix:

    path    = "log://debian@127.0.0.1:2222/sshd"   (or .../all)
    heading = "Jun 03 13:53:01 .. Jun 03 13:59:42" (timestamp range)
    title   = "journalctl"

So a ``rag_search`` hit whose ``path`` starts with ``log://`` is a log
chunk; the RAG panel hides the "open file" affordance for those, and a
future system-prompt line can tell the chat agent to ignore them when
it only wants LVGL grounding.
"""

from __future__ import annotations

import re
from typing import List, Optional

from systemd_mcp.vectordb.embed import EmbeddingClient
from systemd_mcp.vectordb.store import ChunkRow, VectorStore

# journald default ("short") timestamp: "Jun 03 13:53:01".
_SYSLOG_TS_RE = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b")
# --output=short-iso / ISO-ish: "2026-06-03T13:53:01" or "2026-06-03 13:53:01".
_ISO_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b")

# Default per-chunk char cap. Matches _RAG_HIT_TEXT_MAX_CHARS rationale
# in server.py: small enough that a burst of log hits doesn't blow the
# chat server's context, big enough to hold a few related journal lines.
DEFAULT_LOG_CHUNK_CHARS = 600


def _is_entry_start(line: str) -> bool:
    """True if ``line`` begins a new journal entry (has a leading timestamp).

    Continuation lines (stack traces, multi-line messages) have no
    timestamp prefix and get folded into the current entry.
    """
    return bool(_SYSLOG_TS_RE.match(line) or _ISO_TS_RE.match(line))


def _leading_timestamp(line: str) -> Optional[str]:
    """Return the human timestamp at the start of ``line``, or None."""
    m = _SYSLOG_TS_RE.match(line)
    if m:
        return m.group(0).strip()
    m = _ISO_TS_RE.match(line)
    if m:
        return m.group(0).replace("T", " ").strip()
    return None


def chunk_log_text(text: str, max_chars: int = DEFAULT_LOG_CHUNK_CHARS) -> List[str]:
    """Split journal output into embeddable chunks.

    Strategy: group physical lines into journal *entries* on timestamp
    boundaries (continuation lines fold into the preceding entry), then
    greedily pack whole entries into chunks up to ``max_chars``. A single
    entry larger than ``max_chars`` (rare - a giant stack trace) is
    hard-sliced so nothing exceeds the cap. Input that has no journald
    timestamps at all (e.g. a plain app log) degrades gracefully to a
    line-window pack since every line then "starts an entry".
    """
    if not text.strip():
        return []

    # 1. Fold physical lines into entries.
    entries: List[str] = []
    current: List[str] = []
    for raw_line in text.splitlines():
        if _is_entry_start(raw_line) and current:
            entries.append("\n".join(current))
            current = [raw_line]
        elif _is_entry_start(raw_line):
            current = [raw_line]
        else:
            if current:
                current.append(raw_line)
            elif raw_line.strip():
                # Leading non-timestamped content (no entry open yet).
                current = [raw_line]
    if current:
        entries.append("\n".join(current))

    # 2. Greedily pack entries into <= max_chars chunks, hard-slicing
    #    any single oversized entry.
    chunks: List[str] = []
    buf = ""
    for entry in entries:
        if len(entry) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(entry), max_chars):
                chunks.append(entry[i : i + max_chars])
            continue
        if not buf:
            buf = entry
        elif len(buf) + 1 + len(entry) <= max_chars:
            buf += "\n" + entry
        else:
            chunks.append(buf)
            buf = entry
    if buf:
        chunks.append(buf)
    return chunks


def _ts_range(chunk: str) -> str:
    """Build a ``"first .. last"`` timestamp heading for a chunk."""
    stamps = [
        ts for ts in (_leading_timestamp(ln) for ln in chunk.splitlines()) if ts
    ]
    if not stamps:
        return ""
    if len(stamps) == 1:
        return stamps[0]
    return f"{stamps[0]} .. {stamps[-1]}"


def make_log_path(host: str, port: int, username: str, service: Optional[str]) -> str:
    """Synthetic provenance path for a log chunk (the ``log://`` scheme)."""
    return f"log://{username}@{host}:{port}/{service or 'all'}"


def embed_log_lines(
    client: EmbeddingClient,
    store: VectorStore,
    text: str,
    *,
    host: str,
    port: int,
    username: str,
    service: Optional[str] = None,
    max_chars: int = DEFAULT_LOG_CHUNK_CHARS,
) -> int:
    """Chunk, embed, and store ``text`` as log chunks. Returns chunk count.

    Blocking (network embed + sqlite write): call it through
    ``WorkerPool.run`` from the UI, never directly on the loop thread.
    """
    chunks = chunk_log_text(text, max_chars=max_chars)
    if not chunks:
        return 0

    vectors = client.embed_documents(chunks)
    path = make_log_path(host, port, username, service)
    rows = [
        ChunkRow(
            path=path,
            heading=_ts_range(chunk),
            title="journalctl",
            text=chunk,
        )
        for chunk in chunks
    ]
    store.add_batch(rows, vectors)
    # Record provenance so a later reopen can recover the embedding dim
    # (search() short-circuits to zero hits when the store dim is unknown)
    # and so `info` reports the model. Mirrors what the CLI build does.
    store.set_meta(
        dim=int(vectors.shape[1]),
        model_name=getattr(client, "model", "") or "",
        model_family=getattr(client, "model_family", "") or "",
    )
    return len(chunks)


__all__ = [
    "chunk_log_text",
    "embed_log_lines",
    "make_log_path",
    "DEFAULT_LOG_CHUNK_CHARS",
]
