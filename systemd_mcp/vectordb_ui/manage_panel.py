"""Manage tab: inspect / switch / rename / delete / clear / rebuild the DB.

All mutating actions go behind a confirm dialog. File picks (Switch,
Rename) use Toga's native save-file dialog so the user can navigate to
or name a ``.db`` that need not exist yet. The destructive ops and the
LVGL rebuild run through ``WorkerPool.run`` so the UI stays responsive;
rebuild shows an indeterminate running progress bar because
``ingest.build`` reports no incremental callback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import toga
from toga.constants import COLUMN, ROW
from toga.style.pack import Pack

from systemd_mcp.vectordb.ingest import build as ingest_build
from systemd_mcp.vectordb.store import format_info

from .app import clear_store_blocking, make_embed_client, store_info_blocking

if TYPE_CHECKING:
    from .app import VectorDbUiApp

_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _rename_with_sidecars(old: str, new: str) -> None:
    os.rename(old, new)
    for suffix in _SIDECAR_SUFFIXES:
        if os.path.exists(old + suffix):
            try:
                os.rename(old + suffix, new + suffix)
            except OSError:
                pass


def _delete_with_sidecars(path: str) -> None:
    for target in (path, *(path + s for s in _SIDECAR_SUFFIXES)):
        if os.path.exists(target):
            try:
                os.unlink(target)
            except OSError:
                pass


def _build_blocking(config: Any, db_path: str) -> Dict[str, Any]:
    embed = make_embed_client(config)
    stats = ingest_build(db_path=db_path, source=config.build_source, embed_client=embed)
    return {"chunks": stats.chunks_inserted, "files": stats.files_seen}


class ManagePanel:
    def __init__(self, app: "VectorDbUiApp") -> None:
        self._app = app

        self._path_label = toga.Label("", style=Pack(font_weight="bold"))
        self._summary_label = toga.Label("", style=Pack(margin_bottom=4))
        self._detail = toga.MultilineTextInput(
            readonly=True, style=Pack(flex=1, height=240, font_family="monospace")
        )
        self._progress = toga.ProgressBar(max=None, style=Pack(flex=1, margin_top=6))
        self._progress.style.visibility = "hidden"
        self._status = toga.Label("", style=Pack(margin_top=4, font_size=10))

        buttons = toga.Box(style=Pack(direction=ROW, gap=8, margin_top=6))
        buttons.add(toga.Button("Switch DB...", on_press=self._on_switch,
                                style=Pack(flex=1)))
        buttons.add(toga.Button("Rename file", on_press=self._on_rename,
                                style=Pack(flex=1)))
        buttons.add(toga.Button("Delete file", on_press=self._on_delete,
                                style=Pack(flex=1)))
        buttons.add(toga.Button("Clear chunks", on_press=self._on_clear,
                                style=Pack(flex=1)))
        buttons.add(toga.Button("Build LVGL", on_press=self._on_build,
                                style=Pack(flex=1)))

        self.content = toga.Box(style=Pack(direction=COLUMN, margin=10, gap=4, flex=1))
        self.content.add(self._path_label)
        self.content.add(self._summary_label)
        self.content.add(toga.Label("Details:", style=Pack(font_size=10)))
        self.content.add(self._detail)
        self.content.add(buttons)
        self.content.add(self._progress)
        self.content.add(self._status)

        app.register_refresh(self._schedule_refresh)

    # -- refresh ------------------------------------------------------------

    def _schedule_refresh(self) -> None:
        # register_refresh callbacks are sync; bounce into the loop.
        self._app.loop.create_task(self._refresh_async())

    async def _refresh_async(self) -> None:
        db_path = self._app.db_path
        self._path_label.text = f"DB: {db_path}"
        try:
            info = await self._app.pool.run(store_info_blocking, db_path)
        except Exception as exc:  # noqa: BLE001
            self._summary_label.text = f"(could not read DB: {exc})"
            self._detail.value = ""
            return
        if not info.get("exists", False):
            self._summary_label.text = "file does not exist yet (will be created on first write)"
            self._detail.value = ""
            return
        self._summary_label.text = (
            f"{info['chunk_count']} chunks - {info['file_count']} sources - "
            f"dim={info['dim']} - {info['backend']}"
        )
        self._detail.value = format_info(info)

    # -- actions ------------------------------------------------------------

    async def _on_switch(self, widget: toga.Widget) -> None:
        current = Path(self._app.db_path)
        chosen = await self._app.main_window.dialog(
            toga.SaveFileDialog("Select or name a .db file",
                                suggested_filename=current.name,
                                file_types=["db"])
        )
        if chosen is None:
            return
        self._app.set_db_path(str(chosen))
        self._status.text = f"Switched to {chosen}"

    async def _on_rename(self, widget: toga.Widget) -> None:
        old = self._app.db_path
        if not os.path.exists(old):
            await self._app.main_window.dialog(
                toga.InfoDialog("Nothing to rename", f"{old} does not exist.")
            )
            return
        chosen = await self._app.main_window.dialog(
            toga.SaveFileDialog("New name for the .db file",
                                suggested_filename=Path(old).name,
                                file_types=["db"])
        )
        if chosen is None:
            return
        try:
            await self._app.pool.run(_rename_with_sidecars, old, str(chosen))
        except Exception as exc:  # noqa: BLE001
            await self._app.main_window.dialog(
                toga.ErrorDialog("Rename failed", f"{type(exc).__name__}: {exc}")
            )
            return
        self._app.set_db_path(str(chosen))
        self._status.text = f"Renamed to {chosen}"

    async def _on_delete(self, widget: toga.Widget) -> None:
        path = self._app.db_path
        if not os.path.exists(path):
            await self._app.main_window.dialog(
                toga.InfoDialog("Nothing to delete", f"{path} does not exist.")
            )
            return
        ok = await self._app.main_window.dialog(
            toga.ConfirmDialog("Delete DB file?",
                               f"Permanently delete {path} (and -wal/-shm sidecars)?")
        )
        if not ok:
            return
        try:
            await self._app.pool.run(_delete_with_sidecars, path)
        except Exception as exc:  # noqa: BLE001
            await self._app.main_window.dialog(
                toga.ErrorDialog("Delete failed", f"{type(exc).__name__}: {exc}")
            )
            return
        self._status.text = f"Deleted {path}"
        self._app.notify_db_changed()

    async def _on_clear(self, widget: toga.Widget) -> None:
        path = self._app.db_path
        ok = await self._app.main_window.dialog(
            toga.ConfirmDialog("Clear all chunks?",
                               f"Empty every chunk from {path} but keep the file?")
        )
        if not ok:
            return
        try:
            await self._app.pool.run(clear_store_blocking, path)
        except Exception as exc:  # noqa: BLE001
            await self._app.main_window.dialog(
                toga.ErrorDialog("Clear failed", f"{type(exc).__name__}: {exc}")
            )
            return
        self._status.text = "Cleared all chunks."
        self._app.notify_db_changed()

    async def _on_build(self, widget: toga.Widget) -> None:
        path = self._app.db_path
        ok = await self._app.main_window.dialog(
            toga.ConfirmDialog(
                "Rebuild from LVGL source?",
                f"Destructively rebuild {path} from {self._app.config.build_source}.\n"
                f"This clones LVGL and embeds the whole corpus - can take "
                f"several minutes and needs the embedding server running.",
            )
        )
        if not ok:
            return
        self._progress.style.visibility = "visible"
        self._progress.start()
        self._status.text = "Building... (this can take a few minutes)"
        try:
            result = await self._app.pool.run(_build_blocking, self._app.config, path)
        except Exception as exc:  # noqa: BLE001
            await self._app.main_window.dialog(
                toga.ErrorDialog("Build failed", f"{type(exc).__name__}: {exc}")
            )
            self._status.text = "Build failed (see dialog)."
            return
        finally:
            self._progress.stop()
            self._progress.style.visibility = "hidden"
        self._status.text = (
            f"Built {result['chunks']} chunks from {result['files']} files."
        )
        self._app.notify_db_changed()
