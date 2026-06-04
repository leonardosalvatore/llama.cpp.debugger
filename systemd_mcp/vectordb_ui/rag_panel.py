"""Search tab: embed a query, KNN the DB, show expandable hits.

Mirrors the ``rag_search`` MCP tool. The heavy work (embed + sqlite KNN)
runs through ``WorkerPool.run`` in :func:`app.search_blocking`; this file
is just widgets + result rendering. Each hit renders as a header button
(score / path) that toggles between a truncated preview and the full
chunk text. Document-corpus hits get an "Open file" button; ``log://``
hits (ingested journal output) do not, since there's no on-disk file.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, List

import toga
from toga.constants import COLUMN, ROW
from toga.style.pack import Pack

from systemd_mcp.vectordb.store import SearchHit

from .app import RagError, search_blocking

if TYPE_CHECKING:
    from .app import VectorDbUiApp

_PREVIEW_CHARS = 300


def _parse_int(value: object, default: int, lo: int | None = None,
               hi: int | None = None) -> int:
    """Best-effort int parse from a TextInput value, clamped to [lo, hi].

    We use TextInput (not NumberInput/GtkSpinButton) for numeric fields:
    GtkSpinButton trips a noisy ``gtk_box_gadget_distribute: size >= 0``
    assertion when laid out in tight rows, and steppers add nothing here.
    """
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


class _HitRow:
    """One result row: collapsible header + detail text + optional open."""

    def __init__(self, app: "VectorDbUiApp", index: int, hit: SearchHit) -> None:
        self._app = app
        self._hit = hit
        self._expanded = False

        is_log = hit.path.startswith("log://")
        label = f"#{index}  {hit.score:.3f}  {hit.path}"
        if hit.title:
            label += f"  ({hit.title})"

        self._header = toga.Button(label, on_press=self._toggle,
                                   style=Pack(text_align="left", flex=1))

        controls = toga.Box(style=Pack(direction=ROW, gap=6))
        controls.add(self._header)
        if not is_log:
            controls.add(toga.Button("Open file", on_press=self._open_file,
                                     style=Pack(width=110)))

        self._detail = toga.MultilineTextInput(
            readonly=True,
            value=self._preview_text(),
            style=Pack(height=70, flex=1),
        )

        self.content = toga.Box(
            style=Pack(direction=COLUMN, gap=2, margin_bottom=8)
        )
        self.content.add(controls)
        if hit.heading:
            self.content.add(toga.Label(hit.heading, style=Pack(font_size=9)))
        self.content.add(self._detail)

    def _preview_text(self) -> str:
        text = self._hit.text.strip()
        if len(text) <= _PREVIEW_CHARS:
            return text
        return text[:_PREVIEW_CHARS].rstrip() + "  [...click header to expand]"

    async def _toggle(self, widget: toga.Widget) -> None:
        self._expanded = not self._expanded
        if self._expanded:
            self._detail.value = self._hit.text.strip()
            self._detail.style.height = 220
        else:
            self._detail.value = self._preview_text()
            self._detail.style.height = 70

    async def _open_file(self, widget: toga.Widget) -> None:
        # Doc-corpus paths are repo-relative (docs/..., src/...). Resolve
        # against CWD; if it doesn't exist locally, tell the user where it
        # lives rather than spawning xdg-open on a missing path.
        path = self._hit.path
        if os.path.exists(path):
            try:
                subprocess.Popen(["xdg-open", path])
            except Exception as exc:  # noqa: BLE001
                await self._app.main_window.dialog(
                    toga.ErrorDialog("Open failed", str(exc))
                )
        else:
            await self._app.main_window.dialog(
                toga.InfoDialog(
                    "Not a local file",
                    f"'{path}' is a path inside the LVGL corpus, not a file "
                    f"on this machine. Open it in the cloned repo under "
                    f".cache/lvgl/ if you need the full source.",
                )
            )


class RagPanel:
    def __init__(self, app: "VectorDbUiApp") -> None:
        self._app = app
        self._rows: List[_HitRow] = []

        self._query = toga.MultilineTextInput(
            placeholder="Ask the vector DB - e.g. 'how do I create an "
            "lv_checkbox', or 'sshd authentication failures'",
            style=Pack(height=70, flex=1),
        )
        self._k = toga.TextInput(value="5", style=Pack(width=70))
        self._search_btn = toga.Button("Search", on_press=self._on_search,
                                        style=Pack(width=110))
        self._status = toga.Label("", style=Pack(margin_top=2, font_size=10))

        top = toga.Box(style=Pack(direction=ROW, gap=8, margin_bottom=6))
        top.add(self._query)
        kbox = toga.Box(style=Pack(direction=COLUMN, gap=4))
        kbox.add(toga.Label("top-k (1-20)", style=Pack(font_size=10)))
        kbox.add(self._k)
        kbox.add(self._search_btn)
        top.add(kbox)

        self._results_box = toga.Box(style=Pack(direction=COLUMN, gap=2))
        scroll = toga.ScrollContainer(
            horizontal=False, content=self._results_box, style=Pack(flex=1)
        )

        self.content = toga.Box(style=Pack(direction=COLUMN, margin=10, gap=4, flex=1))
        self.content.add(top)
        self.content.add(self._status)
        self.content.add(scroll)

    def _clear_results(self) -> None:
        for child in list(self._results_box.children):
            self._results_box.remove(child)
        self._rows.clear()

    async def _on_search(self, widget: toga.Widget) -> None:
        query = (self._query.value or "").strip()
        if not query:
            self._status.text = "Enter a query first."
            return
        k = _parse_int(self._k.value, 5, 1, 20)
        self._clear_results()
        self._status.text = f"Searching (k={k}) ..."
        self._search_btn.enabled = False
        try:
            hits = await self._app.pool.run(
                search_blocking, self._app.config, self._app.db_path, query, k
            )
        except RagError as exc:
            self._status.text = "No results (see dialog)."
            await self._app.main_window.dialog(
                toga.ErrorDialog("Search unavailable", str(exc))
            )
            return
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            self._status.text = "Search failed (see dialog)."
            await self._app.main_window.dialog(
                toga.ErrorDialog("Search failed", f"{type(exc).__name__}: {exc}")
            )
            return
        finally:
            self._search_btn.enabled = True

        if not hits:
            self._status.text = "No matching chunks."
            return
        self._status.text = f"{len(hits)} hit(s) for '{query}'."
        for i, hit in enumerate(hits, 1):
            row = _HitRow(self._app, i, hit)
            self._rows.append(row)
            self._results_box.add(row.content)
