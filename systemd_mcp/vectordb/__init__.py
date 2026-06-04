"""Vector database subpackage for llama.cpp.debugger.

Self-contained RAG store: ingest the LVGL docs (MD/MDX) **and** source
(C/H under ``src/`` and ``examples/``) corpus, embed each chunk through
a llama-server running with ``--embeddings``, and persist the vectors
in a single sqlite-vec file. Driven from the CLI via
``llama_debugger_vectordb`` (see :mod:`.cli`) and consumed by the chat
agent's ``rag_search`` tool in :mod:`systemd_mcp.server`.

Layout:

* :mod:`.embed`   - OpenAI-protocol embedding client with nomic prefix
                    handling and a bisect-and-truncate fallback for
                    llama-server's "input too large" 500 (talks to a
                    second llama-server on port 53426 by default).
* :mod:`.store`   - sqlite-vec wrapper: ``add_batch`` / ``search`` /
                    ``info`` / ``clear`` / ``set_meta`` over the
                    ``chunks`` / ``chunks_vec`` / ``meta`` tables, with
                    a numpy brute-force fallback when sqlite-vec is
                    unavailable.
* :mod:`.ingest`  - LVGL-aware ingestion: sparse + shallow clone, walk
                    ``docs`` / ``src`` / ``examples``, dispatch to the
                    MDX cleaner + heading splitter for prose or the
                    line-window chunker for C/H sources, embed, store.
* :mod:`.cli`     - argparse front-end (``build`` / ``query`` / ``info``
                    / ``delete``).

See ``README.md`` in this directory for the schema, mermaid build /
query diagrams, sizing caps, env vars, and the soft-fail contract.
"""

from .embed import EmbeddingClient
from .store import VectorStore

__all__ = ["EmbeddingClient", "VectorStore"]
