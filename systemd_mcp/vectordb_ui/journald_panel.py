"""Journald tab: stream journalctl over SSH, optionally ingest into the DB.

The SSH read runs in a daemon thread via ``WorkerPool.spawn_stream``;
each line comes back on the loop thread through ``on_line``. A
``threading.Event`` (owned by the returned :class:`StreamHandle`) lets
the panel cancel a live ``-f`` tail when the user clicks Stop, switches
host, or closes the window - the same idiom as ``_tail_bg_log`` in
``systemd_mcp/cli.py``.

The "Ingest visible -> DB" button embeds whatever is currently in the log
view and stores it as ``log://`` chunks (see :mod:`.ingest_log`).
"""

from __future__ import annotations

import asyncio
import shlex
import time
from typing import TYPE_CHECKING, Iterator, List, Optional

import paramiko
import toga
from toga.constants import COLUMN, ROW
from toga.style.pack import Pack

from systemd_mcp.vectordb.store import VectorStore

from .app import make_embed_client
from .ingest_log import embed_log_lines
from .workers import StreamHandle

if TYPE_CHECKING:
    from .app import VectorDbUiApp

# Ring-buffer cap for the on-screen log so a runaway journal can't grow
# the textarea without bound.
_LOG_BUFFER_MAX_CHARS = 200_000

# How often (seconds) to flush newly-arrived lines into the text widget.
# Lines arrive one call_soon_threadsafe at a time; rewriting the whole
# MultilineTextInput per line is O(n^2) and floods the loop on a big
# fetch (e.g. -n 5000). We coalesce them and repaint at most ~12x/sec.
_FLUSH_INTERVAL_S = 0.08


def _parse_int(value: object, default: int, lo: int | None = None,
               hi: int | None = None) -> int:
    """Best-effort int parse from a TextInput value, clamped to [lo, hi].

    Numeric fields use TextInput rather than NumberInput because the
    GtkSpinButton backing NumberInput trips a noisy
    ``gtk_box_gadget_distribute: size >= 0`` assertion in tight rows.
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


def _journal_lines(
    stop_event,
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    service: Optional[str],
    lines: int,
    follow: bool,
) -> Iterator[str]:
    """Blocking generator yielding journalctl output line by line.

    Works for both the one-shot (``follow=False``) and live (``-f``)
    cases; in follow mode it runs until ``stop_event`` is set.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        host,
        port=int(port),
        username=username,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=10.0,
    )
    try:
        cmd = "journalctl"
        if service:
            cmd += f" -u {shlex.quote(service)}"
        cmd += f" -n {int(lines)} --no-pager"
        if follow:
            cmd += " -f"

        transport = ssh.get_transport()
        if transport is None:
            raise RuntimeError("ssh transport unavailable")
        chan = transport.open_session()
        chan.settimeout(0.5)
        chan.exec_command(cmd)

        buf = bytearray()
        while not stop_event.is_set():
            got = False
            if chan.recv_ready():
                data = chan.recv(65536)
                if data:
                    buf.extend(data)
                    got = True
            if chan.recv_stderr_ready():
                err = chan.recv_stderr(65536)
                if err:
                    buf.extend(err)
                    got = True
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buf[:nl])
                del buf[: nl + 1]
                yield line.decode(errors="replace")
            if (
                chan.exit_status_ready()
                and not chan.recv_ready()
                and not chan.recv_stderr_ready()
            ):
                if buf:
                    yield bytes(buf).decode(errors="replace")
                    buf.clear()
                break
            if not got:
                time.sleep(0.1)
    finally:
        try:
            ssh.close()
        except Exception:  # noqa: BLE001
            pass


def _ingest_blocking(
    config,
    db_path: str,
    text: str,
    *,
    host: str,
    port: int,
    username: str,
    service: Optional[str],
) -> int:
    client = make_embed_client(config)
    with VectorStore(db_path) as store:
        return embed_log_lines(
            client, store, text,
            host=host, port=port, username=username, service=service,
        )


class JournaldPanel:
    def __init__(self, app: "VectorDbUiApp") -> None:
        self._app = app
        self._stream: Optional[StreamHandle] = None
        self._buffer = ""
        # Coalesced line buffer: on_line appends here, a throttled timer
        # flushes the batch into the widget in one repaint.
        self._pending: List[str] = []
        self._flush_handle: Optional[asyncio.TimerHandle] = None

        cfg = app.config
        self._host = toga.TextInput(value=cfg.sut_host, style=Pack(flex=1))
        self._port = toga.TextInput(value=str(cfg.sut_port), style=Pack(width=90))
        self._user = toga.TextInput(value=cfg.sut_user, style=Pack(width=130))
        self._password = toga.PasswordInput(value=cfg.sut_password,
                                             style=Pack(width=130))
        self._service = toga.TextInput(placeholder="service (optional)",
                                       style=Pack(flex=1))
        self._lines = toga.TextInput(value="200", style=Pack(width=90))
        self._live = toga.Switch("Live (-f)")
        self._connect_btn = toga.Button("Connect", on_press=self._on_connect,
                                        style=Pack(width=110))

        row1 = toga.Box(style=Pack(direction=ROW, gap=6, margin_bottom=4))
        row1.add(toga.Label("host", style=Pack(width=34, font_size=10)))
        row1.add(self._host)
        row1.add(toga.Label("port", style=Pack(width=30, font_size=10)))
        row1.add(self._port)
        row1.add(toga.Label("user", style=Pack(width=32, font_size=10)))
        row1.add(self._user)
        row1.add(toga.Label("pass", style=Pack(width=32, font_size=10)))
        row1.add(self._password)

        row2 = toga.Box(style=Pack(direction=ROW, gap=6, margin_bottom=4))
        row2.add(toga.Label("unit", style=Pack(width=34, font_size=10)))
        row2.add(self._service)
        row2.add(toga.Label("lines", style=Pack(width=34, font_size=10)))
        row2.add(self._lines)
        row2.add(self._live)
        row2.add(self._connect_btn)

        self._log_view = toga.MultilineTextInput(
            readonly=True,
            style=Pack(flex=1, font_family="monospace"),
        )

        self._ingest_btn = toga.Button("Ingest visible -> DB",
                                       on_press=self._on_ingest,
                                       style=Pack(width=180))
        self._clear_btn = toga.Button("Clear view", on_press=self._on_clear_view,
                                      style=Pack(width=110))
        self._status = toga.Label("", style=Pack(flex=1, font_size=10))
        footer = toga.Box(style=Pack(direction=ROW, gap=8, margin_top=4))
        footer.add(self._ingest_btn)
        footer.add(self._clear_btn)
        footer.add(self._status)

        self.content = toga.Box(style=Pack(direction=COLUMN, margin=10, gap=2, flex=1))
        self.content.add(row1)
        self.content.add(row2)
        self.content.add(self._log_view)
        self.content.add(footer)

    # -- streaming ----------------------------------------------------------

    async def _on_connect(self, widget: toga.Widget) -> None:
        if self._stream is not None and self._stream.is_running:
            self.stop_stream()
            return

        host = (self._host.value or "").strip()
        if not host:
            self._status.text = "Enter a host."
            return
        follow = bool(self._live.value)
        self._reset_pending()
        self._set_buffer("")  # fresh view per connect
        self._status.text = f"Connecting to {host} ..."
        self._connect_btn.text = "Stop"

        self._stream = self._app.pool.spawn_stream(
            lambda stop: _journal_lines(
                stop,
                host=host,
                port=_parse_int(self._port.value, 22, 1, 65535),
                username=(self._user.value or "").strip(),
                password=self._password.value or "",
                service=(self._service.value or "").strip() or None,
                lines=_parse_int(self._lines.value, 200, 1, 100000),
                follow=follow,
            ),
            on_line=self._on_line,
            on_done=self._on_stream_done,
            on_error=self._on_stream_error,
        )

    def _on_line(self, line: str) -> None:
        # Cheap: just queue. The throttled flush does the expensive
        # widget repaint once per batch instead of once per line.
        self._pending.append(line)
        if not self._status.text.startswith("Streaming"):
            self._status.text = "Streaming..."
        if self._flush_handle is None:
            loop = asyncio.get_running_loop()
            self._flush_handle = loop.call_later(
                _FLUSH_INTERVAL_S, self._flush_pending
            )

    def _flush_pending(self) -> None:
        """Append all coalesced lines to the view in a single repaint."""
        self._flush_handle = None
        if not self._pending:
            return
        chunk = "\n".join(self._pending) + "\n"
        self._pending.clear()
        self._set_buffer(self._buffer + chunk)

    def _on_stream_done(self) -> None:
        # Drain anything still queued so the tail of the log isn't lost.
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        self._flush_pending()
        self._connect_btn.text = "Connect"
        if self._stream is not None and not self._stream.stopping:
            self._status.text = "Stream ended."
        self._stream = None

    def _on_stream_error(self, exc: BaseException) -> None:
        self._reset_pending()
        self._connect_btn.text = "Connect"
        self._stream = None
        self._status.text = f"SSH error: {type(exc).__name__}: {exc}"

    def _reset_pending(self) -> None:
        """Drop any queued lines and cancel a scheduled flush."""
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        self._pending.clear()

    def stop_stream(self) -> None:
        """Cancel a running stream (also called on app close)."""
        if self._stream is not None:
            self._stream.stop()
            self._flush_pending()  # show whatever already arrived
            self._status.text = "Stopped."
            self._connect_btn.text = "Connect"

    # -- buffer / view ------------------------------------------------------

    def _set_buffer(self, text: str) -> None:
        if len(text) > _LOG_BUFFER_MAX_CHARS:
            text = text[-_LOG_BUFFER_MAX_CHARS:]
        self._buffer = text
        self._log_view.value = text
        try:
            self._log_view.scroll_to_bottom()
        except Exception:  # noqa: BLE001 - older backends may lack it
            pass

    async def _on_clear_view(self, widget: toga.Widget) -> None:
        self._reset_pending()
        self._set_buffer("")
        self._status.text = "View cleared."

    # -- ingest -------------------------------------------------------------

    async def _on_ingest(self, widget: toga.Widget) -> None:
        text = self._buffer.strip()
        if not text:
            self._status.text = "Nothing in the view to ingest."
            return
        self._ingest_btn.enabled = False
        self._status.text = "Embedding + storing log chunks ..."
        try:
            count = await self._app.pool.run(
                _ingest_blocking,
                self._app.config,
                self._app.db_path,
                text,
                host=(self._host.value or "").strip(),
                port=_parse_int(self._port.value, 22, 1, 65535),
                username=(self._user.value or "").strip(),
                service=(self._service.value or "").strip() or None,
            )
        except Exception as exc:  # noqa: BLE001
            await self._app.main_window.dialog(
                toga.ErrorDialog("Ingest failed", f"{type(exc).__name__}: {exc}")
            )
            self._status.text = "Ingest failed (see dialog)."
            return
        finally:
            self._ingest_btn.enabled = True
        self._status.text = f"Ingested {count} chunk(s) into the DB."
        self._app.notify_db_changed()
        await self._app.main_window.dialog(
            toga.InfoDialog(
                "Ingested",
                f"Stored {count} log chunk(s) as 'log://' entries in\n"
                f"{self._app.db_path}.\nFind them from the Search tab.",
            )
        )
