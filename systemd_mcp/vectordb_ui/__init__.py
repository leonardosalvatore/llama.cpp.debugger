"""Toga desktop UI for the llama.cpp.debugger vector store.

A native (GTK on Linux) three-tab control panel over the same
``systemd_mcp.vectordb`` building blocks the chat agent uses:

* **Journald** - stream ``journalctl`` from any SSH host (defaults to the
  QEMU SUT) and push selected output into the vector DB.
* **Search** - run RAG queries against the DB with the embedding
  llama-server, exactly like the ``rag_search`` MCP tool.
* **Manage** - inspect / switch / rename / delete / clear / rebuild the
  ``.db`` file.

This subpackage is intentionally isolated from ``systemd_mcp.vectordb``:
nothing in the server, chat CLI, or vectordb CLI imports it, so a
headless box can skip the ``toga`` / ``toga-gtk`` extras without
breaking the rest of the project. The reverse dependency (this package
on ``vectordb``) is the only direction.

Run it with ``poetry run llama_debugger_vectordb_ui`` (see
:mod:`.cli`) or ``python -m systemd_mcp.vectordb_ui``.
"""

from __future__ import annotations

from .cli import main

__all__ = ["main"]
