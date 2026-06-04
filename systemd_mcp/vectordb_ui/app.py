"""Toga application shell: window, tabs, shared state, blocking services.

``VectorDbUiApp`` owns the asyncio loop (via Toga), the :class:`WorkerPool`,
the currently-selected DB path, and a tiny pub/sub so the Manage tab can
tell the others "the DB changed, refresh". The three panels are built in
:meth:`startup`.

The module-level ``*_blocking`` helpers are the functions panels hand to
``WorkerPool.run``. They open a fresh :class:`VectorStore` per call (sqlite
connections are not safe to share across the pool's threads) and mirror
the soft-fail shape of ``rag_search`` in ``systemd_mcp/server.py`` by
raising typed exceptions the panels translate into dialogs.

Panel imports are deferred into :meth:`startup` so this module finishes
loading before the panels import back from it - avoids a circular import
while keeping every file in the planned layout.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

import toga

from systemd_mcp.vectordb.embed import EmbeddingClient
from systemd_mcp.vectordb.store import SearchHit, VectorStore

from .cli import UiConfig
from .workers import WorkerPool


# ---------------------------------------------------------------------------
# Blocking service helpers (run via WorkerPool.run, never on the loop thread)
# ---------------------------------------------------------------------------


class RagError(Exception):
    """Human-readable, already-actionable failure from a blocking helper."""


def make_embed_client(config: UiConfig) -> EmbeddingClient:
    return EmbeddingClient(
        host=config.embed_host,
        port=config.embed_port,
        model=config.embed_model,
        model_family=config.embed_family,
    )


def store_info_blocking(db_path: str) -> Dict[str, Any]:
    """Return ``store.info()`` or a sentinel dict if the file is absent."""
    if not os.path.exists(db_path):
        return {
            "db_path": db_path,
            "db_size_bytes": 0,
            "backend": "n/a",
            "dim": None,
            "chunk_count": 0,
            "file_count": 0,
            "meta": {},
            "exists": False,
        }
    with VectorStore(db_path) as store:
        info = store.info()
    info["exists"] = True
    return info


def search_blocking(
    config: UiConfig, db_path: str, query: str, k: int
) -> List[SearchHit]:
    """Embed ``query`` and KNN-search ``db_path``. Raises :class:`RagError`.

    Same failure taxonomy as the ``rag_search`` MCP tool: missing DB,
    empty DB, unreachable embedding server each produce a distinct,
    actionable message instead of a raw traceback.
    """
    if not os.path.exists(db_path):
        raise RagError(
            f"vector DB not found at {db_path}.\n"
            f"Build it from the Manage tab, or run "
            f"`poetry run llama_debugger_vectordb build`."
        )
    with VectorStore(db_path) as store:
        info = store.info()
        if info["chunk_count"] == 0:
            raise RagError(
                f"vector DB at {db_path} is empty.\n"
                f"Ingest some logs (Journald tab) or build the LVGL "
                f"corpus (Manage tab)."
            )
        embed = make_embed_client(config)
        try:
            qvec = embed.embed_query(query)
        except Exception as exc:  # noqa: BLE001 - re-wrap as actionable
            raise RagError(
                f"embedding server at {embed.base_url} unreachable "
                f"({type(exc).__name__}: {exc}).\n"
                f"Start it with ./start-llama-embedding-server.sh."
            ) from exc
        return store.search(qvec, k=k)


def clear_store_blocking(db_path: str) -> None:
    if not os.path.exists(db_path):
        return
    with VectorStore(db_path) as store:
        store.clear()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class VectorDbUiApp(toga.App):
    """Three-tab control panel: Journald / Search / Manage."""

    def __init__(self, config: UiConfig) -> None:
        self.config = config
        self._db_path = config.db_path
        self._refresh_callbacks: List[Callable[[], None]] = []
        self.pool: WorkerPool | None = None
        super().__init__(
            formal_name="Vector DB UI",
            app_id="io.llamadbg.vectordb_ui",
        )

    # -- shared state -------------------------------------------------------

    @property
    def db_path(self) -> str:
        return self._db_path

    def set_db_path(self, new_path: str) -> None:
        """Point the whole app at a different .db file and refresh panels."""
        self._db_path = new_path
        self.notify_db_changed()

    def register_refresh(self, callback: Callable[[], None]) -> None:
        """Register a 'DB changed' listener (panels call this in __init__)."""
        self._refresh_callbacks.append(callback)

    def notify_db_changed(self) -> None:
        for cb in self._refresh_callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001 - one bad panel shouldn't break others
                pass

    # -- lifecycle ----------------------------------------------------------

    def startup(self) -> None:
        self.pool = WorkerPool(self.loop)

        # Deferred imports (see module docstring): panels import helpers
        # from this module, so this module must be fully loaded first.
        from .journald_panel import JournaldPanel
        from .manage_panel import ManagePanel
        from .rag_panel import RagPanel

        self.journald_panel = JournaldPanel(self)
        self.rag_panel = RagPanel(self)
        self.manage_panel = ManagePanel(self)

        container = toga.OptionContainer(
            content=[
                toga.OptionItem("Journald", self.journald_panel.content),
                toga.OptionItem("Search", self.rag_panel.content),
                toga.OptionItem("Manage", self.manage_panel.content),
            ]
        )

        self.main_window = toga.MainWindow(
            title="llama.cpp.debugger - Vector DB UI",
            size=(1100, 820),
        )
        self.main_window.content = container
        self.main_window.on_close = self._on_close
        self.main_window.show()

        # Paint initial DB stats once everything is wired.
        self.notify_db_changed()

    def _on_close(self, window: toga.Window, **kwargs: Any) -> bool:
        # Stop any live journald stream and drop the worker pool cleanly.
        try:
            self.journald_panel.stop_stream()
        except Exception:  # noqa: BLE001
            pass
        if self.pool is not None:
            self.pool.shutdown()
        return True


__all__ = [
    "VectorDbUiApp",
    "RagError",
    "make_embed_client",
    "store_info_blocking",
    "search_blocking",
    "clear_store_blocking",
]
