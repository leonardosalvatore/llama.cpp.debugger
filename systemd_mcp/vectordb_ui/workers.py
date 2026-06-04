"""Off-the-event-loop work for the Toga UI.

Toga runs on a single asyncio event loop that also owns every GTK
widget. Touching a widget from another thread is undefined behavior,
the same hazard LVGL has - so this module is the one place threads are
allowed, and it guarantees results come back *on the loop thread*.

Two primitives:

* :meth:`WorkerPool.run` - ``await pool.run(fn, *args)`` runs a blocking
  callable (``paramiko.exec_command``, ``embed.embed_*``,
  ``store.search`` / ``add_batch`` / ``build``) in a thread and returns
  its result to the awaiting coroutine. Because the caller is an async
  Toga handler, the continuation after the ``await`` is already back on
  the loop thread - no manual marshaling needed.

* :meth:`WorkerPool.spawn_stream` - for the open-ended ``journalctl -f``
  case where there is no single result, just a sequence of lines. Runs a
  blocking generator in a daemon thread and pushes each line back via
  ``loop.call_soon_threadsafe`` so the panel's ``on_line`` callback fires
  on the loop thread. A :class:`StreamHandle` exposes ``stop()`` (sets a
  ``threading.Event`` the generator checks between lines) so the panel
  can cancel cleanly when the user clicks Stop or switches host.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Optional


class StreamHandle:
    """Handle to a running stream worker. Cancel via :meth:`stop`."""

    def __init__(self, stop_event: threading.Event, thread: threading.Thread) -> None:
        self._stop = stop_event
        self._thread = thread

    def stop(self) -> None:
        """Signal the producer to stop. Idempotent; returns immediately.

        The worker checks the event between lines, so the SSH channel is
        torn down at the next line boundary rather than instantly - good
        enough for a log tail and avoids killing a half-read line.
        """
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive()


class WorkerPool:
    """Thread pool + stream spawner bound to one asyncio event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, *, max_workers: int = 4) -> None:
        self._loop = loop
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="vdbui"
        )

    async def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` in the pool; await the result.

        Exceptions raised by ``fn`` propagate to the awaiting coroutine,
        so callers wrap the ``await`` in try/except and surface failures
        through a Toga dialog.
        """
        return await self._loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )

    def spawn_stream(
        self,
        produce: Callable[[threading.Event], Iterator[str]],
        *,
        on_line: Callable[[str], None],
        on_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> StreamHandle:
        """Run a blocking line generator in a daemon thread.

        ``produce(stop_event)`` must yield strings and should check
        ``stop_event.is_set()`` periodically (at minimum between lines).
        ``on_line`` / ``on_done`` / ``on_error`` are all invoked on the
        loop thread, so they may freely mutate widgets.
        """
        stop_event = threading.Event()

        def _worker() -> None:
            try:
                for line in produce(stop_event):
                    if stop_event.is_set():
                        break
                    self._loop.call_soon_threadsafe(on_line, line)
            except BaseException as exc:  # noqa: BLE001 - report, never crash the thread
                if on_error is not None:
                    self._loop.call_soon_threadsafe(on_error, exc)
            finally:
                if on_done is not None:
                    self._loop.call_soon_threadsafe(on_done)

        thread = threading.Thread(target=_worker, daemon=True, name="vdbui-stream")
        thread.start()
        return StreamHandle(stop_event, thread)

    def shutdown(self) -> None:
        """Stop accepting new work and drop the pool (called on app exit)."""
        self._executor.shutdown(wait=False, cancel_futures=True)
