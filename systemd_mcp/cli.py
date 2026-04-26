"""Command-line client for llama.cpp.debugger.

Talks to a local ``llama-server`` (started via ``./start-llama-server.sh``)
through its OpenAI-compatible ``/v1/chat/completions`` endpoint and exposes
the same tool surface as the FastMCP server in ``systemd_mcp.server``.

The thinking trace ships in the ``reasoning_content`` delta field thanks to
``--reasoning-format deepseek`` in ``start-llama-server.sh``.

Two output modes, both driven through the ``Sink`` abstraction below:

* default (stdout): the classic streaming print-to-terminal behavior.
* ``--split-screen``: a full-screen ``prompt_toolkit`` TUI with
  - top half:    chat with the model (reasoning, answer, tool calls).
  - bottom half: every command sent over SSH to the SUT and its raw output.
  - one-line input field at the very bottom.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

from openai import OpenAI

from systemd_mcp import server as srv


DEFAULT_SYSTEM_PROMPT = (
    "You are llama.cpp.debugger, an autonomous SRE/debug agent for an "
    "embedded Debian Linux SUT (System Under Test). You drive the SUT only "
    "through the provided tools. Tools are grouped by namespace:\n"
    "  systemd_*       systemd / journald management\n"
    "  linux_*         filesystem and process inspection\n"
    "  compiler_*      gcc / g++ / make / cmake builds\n"
    "  gdb_*           debugger control via a persistent tmux session\n"
    "  configuration_* point the agent at a different SSH target\n"
    "Default target is debian@127.0.0.1:2222 (the QEMU box from "
    "run_linux_in_qemu.sh); call configuration_setTargetHost to switch.\n"
    "\n"
    "GDB workflow for debugging a program from source:\n"
    "  A. Source-level debug (start under gdb): "
    "compile with `compiler_gcc(... flags='-g -O0')` -> "
    "`gdb_start_session_run(binary)` (loads, NOT executing) -> "
    "`gdb_break(location)` -> `gdb_run()` (actually starts) -> "
    "`gdb_continue/gdb_step/gdb_next/gdb_print/gdb_backtrace`.\n"
    "  B. Attach to a running program: "
    "`linux_run_in_background(command='./binary')` returns PID -> "
    "`gdb_start_session_attach(pid)` -> `gdb_break(location)` -> "
    "`gdb_continue` (the inferior is already running, no `gdb_run` needed).\n"
    "\n"
    "Rules to avoid wasted tool calls:\n"
    "* To execute a shell command on the SUT, call `linux_run_command` "
    "directly. Do NOT write a wrapper script and chmod it just to run a one-liner.\n"
    "* To run a program in the background, call `linux_run_in_background`. "
    "Do NOT invent nohup/&/disown wrappers yourself.\n"
    "* Do NOT list `/usr/bin`, `/bin`, or other large system directories - "
    "their contents will not help and will blow the context window.\n"
    "* If a tool reports 'No space left on device', call linux_disk_usage and "
    "report - do NOT delete /var/log, /var/cache, /usr, or anything outside "
    "/tmp and the user's working directory.\n"
    "Always answer concisely and prefer running tools over guessing."
)


# ---------------------------------------------------------------------------
# Tool spec / registry. Mirrors @mcp.tool() decorations in server.py so the
# model sees them through the OpenAI tool-calling protocol.
# ---------------------------------------------------------------------------


def _tool(name: str, desc: str, props: Dict[str, Any] | None = None,
          required: List[str] | None = None) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}


TOOL_SPEC: List[Dict[str, Any]] = [
    _tool(
        "configuration_setTargetHost",
        "Point all subsequent tools at a different SSH target. Defaults match the QEMU SUT.",
        {
            "host": {**_STR, "description": "Hostname or IP of the SUT."},
            "port": {**_INT, "description": "SSH port (default 2222)."},
            "username": _STR,
            "password": _STR,
        },
    ),
    _tool(
        "configuration_getTargetHost",
        "Return the currently configured SSH target (password redacted).",
    ),

    _tool("systemd_get_service_status",
          "Run `systemctl status <service>` on the SUT.",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_read_journal",
          "Read recent journal lines, optionally filtered by service.",
          {"service_name": _STR, "lines": {**_INT, "minimum": 1, "maximum": 1000}}),
    _tool("systemd_restart_service", "Restart a systemd service (sudo).",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_start_service", "Start a systemd service (sudo).",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_stop_service", "Stop a systemd service (sudo).",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_enable_service", "Enable a systemd service to start at boot.",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_disable_service", "Disable a systemd service from auto-start.",
          {"service_name": _STR}, ["service_name"]),
    _tool("systemd_list_services", "List every systemd service unit and its state."),
    _tool("systemd_get_uptime", "Return the SUT's uptime."),
    _tool("systemd_daemon_reload", "Reload systemd unit files (sudo)."),

    _tool("linux_list_directory", "List a directory (`ls -lh`).",
          {"path": _STR, "show_hidden": _BOOL}),
    _tool("linux_read_file", "Read up to `max_bytes` bytes from a file.",
          {"path": _STR, "max_bytes": _INT}, ["path"]),
    _tool("linux_write_file", "Overwrite a file with the given content (sudo).",
          {"path": _STR, "content": _STR}, ["path", "content"]),
    _tool("linux_append_file", "Append content to a file (sudo).",
          {"path": _STR, "content": _STR}, ["path", "content"]),
    _tool("linux_remove", "Remove a file or (recursively) a directory (sudo).",
          {"path": _STR, "recursive": _BOOL}, ["path"]),
    _tool("linux_make_directory", "Create a directory (sudo, `-p` by default).",
          {"path": _STR, "parents": _BOOL}, ["path"]),
    _tool("linux_copy", "Copy a file or directory (sudo).",
          {"src": _STR, "dst": _STR, "recursive": _BOOL}, ["src", "dst"]),
    _tool("linux_move", "Move/rename a file or directory (sudo).",
          {"src": _STR, "dst": _STR}, ["src", "dst"]),
    _tool("linux_find", "Find files by name pattern under a path.",
          {"path": _STR, "name_pattern": _STR}, ["path"]),
    _tool("linux_grep", "Grep for a pattern under a path (recursive by default).",
          {"pattern": _STR, "path": _STR, "recursive": _BOOL}, ["pattern"]),
    _tool("linux_get_processes", "List all processes (`ps -ef`)."),
    _tool("linux_disk_usage", "Show disk usage for a path (`df -h`).",
          {"path": _STR}),
    _tool("linux_which", "Locate an executable on the SUT's PATH.",
          {"binary": _STR}, ["binary"]),
    _tool("linux_run_command",
          "Run an arbitrary shell command on the SUT (foreground, returns stdout/stderr + exit code). "
          "Use this for one-shot commands; do NOT write a wrapper script first.",
          {"command": _STR, "cwd": _STR}, ["command"]),
    _tool("linux_run_in_background",
          "Launch a long-running process detached from the SSH session, redirecting output to log_path "
          "(default /tmp/llamadbg_bg.log) and returning its PID. Use this whenever the user asks to run "
          "something 'in the background' so it survives the tool call - then attach gdb via gdb_start_session_attach(pid).",
          {"command": _STR, "cwd": _STR, "log_path": _STR}, ["command"]),

    _tool("compiler_gcc",
          "Compile a single source file with gcc (C) or g++ (C++).",
          {
              "source": _STR,
              "output": _STR,
              "flags": _STR,
              "language": {**_STR, "enum": ["c", "c++"]},
          }, ["source"]),
    _tool("compiler_make", "Run make in a directory.",
          {"directory": _STR, "target": _STR, "jobs": _INT}),
    _tool("compiler_cmake_configure",
          "Run `cmake -S source_dir -B build_dir -DCMAKE_BUILD_TYPE=...`.",
          {
              "source_dir": _STR,
              "build_dir": _STR,
              "build_type": {**_STR, "enum": ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"]},
              "extra_flags": _STR,
          }, ["source_dir", "build_dir"]),
    _tool("compiler_cmake_build", "Run `cmake --build build_dir`.",
          {"build_dir": _STR, "target": _STR, "jobs": _INT}, ["build_dir"]),

    _tool("gdb_start_session_attach",
          "Attach gdb to a running process by PID inside a fresh tmux session.",
          {"pid": _INT}, ["pid"]),
    _tool("gdb_start_session_run",
          "Open a binary under gdb (LOADED but NOT EXECUTING). "
          "Required workflow: gdb_start_session_run -> gdb_break (optional) -> gdb_run -> gdb_continue/step/print.",
          {"binary": _STR, "args": _STR}, ["binary"]),
    _tool("gdb_start_session_core",
          "Open a core dump for a binary in a fresh tmux gdb session.",
          {"binary": _STR, "core": _STR}, ["binary", "core"]),
    _tool("gdb_send_command",
          "Send an arbitrary command to the running gdb session and return the pane.",
          {"command": _STR}, ["command"]),
    _tool("gdb_read_output", "Capture the last N lines of the gdb pane.",
          {"lines": _INT}),
    _tool("gdb_break", "Set a breakpoint (e.g. `main`, `foo.c:42`).",
          {"location": _STR}, ["location"]),
    _tool("gdb_run",
          "Start (or restart) the program under gdb (`run`). "
          "Call this AFTER gdb_start_session_run before any continue/step/print."),
    _tool("gdb_continue",
          "Continue execution. Requires that gdb_run was called first; "
          "otherwise gdb returns 'The program is not being run'."),
    _tool("gdb_step", "Step into (`step`)."),
    _tool("gdb_next", "Step over (`next`)."),
    _tool("gdb_finish", "Run until the current function returns (`finish`)."),
    _tool("gdb_print", "Evaluate `print expr`.", {"expr": _STR}, ["expr"]),
    _tool("gdb_backtrace", "Show a backtrace.", {"depth": _INT}),
    _tool("gdb_info_registers", "Show CPU register state (`info registers`)."),
    _tool("gdb_info_threads", "Show all threads (`info threads`)."),
    _tool("gdb_list_breakpoints", "Show all current breakpoints (`info breakpoints`)."),
    _tool("gdb_quit", "Quit the active gdb session and tear down the tmux pane."),
]


def _unwrap(name: str) -> Callable[..., Any]:
    obj = getattr(srv, name)
    return obj.fn if hasattr(obj, "fn") else obj


REGISTRY: Dict[str, Callable[..., Any]] = {
    spec["function"]["name"]: _unwrap(spec["function"]["name"])
    for spec in TOOL_SPEC
}


# ---------------------------------------------------------------------------
# Sink abstraction. The streaming/tool-dispatch loop is shared between modes
# and only differs in WHERE chat tokens, SSH events, and the input prompt land.
# ---------------------------------------------------------------------------


class Sink(ABC):
    """A surface that receives chat tokens + SSH events and supplies user input."""

    @abstractmethod
    def banner(self, text: str) -> None: ...

    @abstractmethod
    def write_chat(self, text: str, kind: str = "answer") -> None: ...

    @abstractmethod
    def write_ssh(self, cmd: str, out: str) -> None: ...

    @abstractmethod
    def read_user(self, prompt: str) -> Optional[str]: ...

    def close(self) -> None:
        return None


class StdoutSink(Sink):
    """Classic streaming-print behavior. Matches the pre-TUI CLI byte-for-byte.

    SSH wire events do NOT print here (the dispatch loop already shows
    ``[tool result] ...``); they would be redundant noise.
    """

    def banner(self, text: str) -> None:
        print(text, flush=True)

    def write_chat(self, text: str, kind: str = "answer") -> None:
        end = "\n" if kind in {"newline", "user_echo"} else ""
        print(text, end=end, flush=True)

    def write_ssh(self, cmd: str, out: str) -> None:
        return None

    def read_user(self, prompt: str) -> Optional[str]:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None


# --- TUI sink ----------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class TuiSink(Sink):
    """Full-screen split-pane TUI driven by ``prompt_toolkit``.

    Top frame: chat with the model (reasoning, answer, tool calls/results).
    Bottom frame: SSH wire to the SUT - every command and its raw output.
    Footer: one-line input field. Ctrl-C / Ctrl-Q exits.
    """

    _MAX_CHARS = 200_000  # ring-buffer cap per panel

    def __init__(self) -> None:
        from prompt_toolkit import Application
        from prompt_toolkit.document import Document
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Frame, TextArea

        self._Document = Document

        self._chat_lock = threading.Lock()
        self._ssh_lock = threading.Lock()
        self._chat_text = ""
        self._ssh_text = ""
        self._input_q: queue.Queue[Optional[str]] = queue.Queue()

        self.chat_area = TextArea(
            scrollbar=True,
            read_only=True,
            focusable=False,
            wrap_lines=True,
            style="class:chat",
        )
        self.ssh_area = TextArea(
            scrollbar=True,
            read_only=True,
            focusable=False,
            wrap_lines=True,
            style="class:ssh",
        )
        self.input_area = TextArea(
            height=1,
            prompt="you> ",
            multiline=False,
            wrap_lines=False,
            style="class:input",
            accept_handler=self._on_accept,
        )

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-q")
        def _quit(event):  # noqa: ANN001
            self._input_q.put(None)
            event.app.exit()

        style = Style.from_dict({
            "frame.border": "#777777",
            "frame.label": "bold #88c0d0",
            "chat": "",
            "ssh": "#a3be8c",
            "input": "bold",
        })

        self.layout = Layout(
            HSplit([
                Frame(self.chat_area, title="chat (model)"),
                Frame(self.ssh_area, title="ssh wire (SUT)"),
                self.input_area,
            ]),
            focused_element=self.input_area,
        )
        self.app = Application(
            layout=self.layout,
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
        )

    # -- prompt_toolkit callbacks -------------------------------------------

    def _on_accept(self, buffer) -> bool:  # noqa: ANN001
        text = buffer.text
        # Echo the user's line into the chat panel so context is preserved.
        self._append_chat(f"\nyou> {text}\n")
        self._input_q.put(text)
        return False

    # -- internal append helpers --------------------------------------------

    def _append_chat(self, text: str) -> None:
        if not text:
            return
        with self._chat_lock:
            self._chat_text += text
            if len(self._chat_text) > self._MAX_CHARS:
                self._chat_text = self._chat_text[-self._MAX_CHARS:]
            payload = self._chat_text
        self._set_area(self.chat_area, payload)

    def _append_ssh(self, text: str) -> None:
        if not text:
            return
        with self._ssh_lock:
            self._ssh_text += text
            if len(self._ssh_text) > self._MAX_CHARS:
                self._ssh_text = self._ssh_text[-self._MAX_CHARS:]
            payload = self._ssh_text
        self._set_area(self.ssh_area, payload)

    def _set_area(self, area: Any, text: str) -> None:
        # Buffer mutation must happen on the UI thread; schedule via call_soon.
        def _do() -> None:
            area.buffer.set_document(
                self._Document(text, cursor_position=len(text)),
                bypass_readonly=True,
            )

        try:
            loop = self.app.loop  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            loop = None
        if loop is not None and self.app.is_running:
            try:
                loop.call_soon_threadsafe(_do)
            except RuntimeError:
                _do()
        else:
            _do()
        if self.app.is_running:
            self.app.invalidate()

    # -- Sink interface -----------------------------------------------------

    def banner(self, text: str) -> None:
        self._append_chat(text + "\n")

    def write_chat(self, text: str, kind: str = "answer") -> None:
        if kind in {"newline", "user_echo"} and not text.endswith("\n"):
            text = text + "\n"
        self._append_chat(text)

    def write_ssh(self, cmd: str, out: str) -> None:
        line = f"\n[{_ts()}] $ {cmd}\n{out}"
        if not line.endswith("\n"):
            line += "\n"
        self._append_ssh(line)

    def read_user(self, prompt: str) -> Optional[str]:
        # In TUI mode the prompt label is fixed ("you> "); callers may pass any
        # value but we ignore it for layout consistency.
        del prompt
        try:
            return self._input_q.get()
        except KeyboardInterrupt:
            return None

    def close(self) -> None:
        if self.app.is_running:
            try:
                self.app.exit()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Streaming helpers (shared by both sinks)
# ---------------------------------------------------------------------------


def _stream_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    sink: Sink,
) -> Dict[str, Any]:
    """One streaming round-trip with llama-server. Returns the assistant message."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    stream = client.chat.completions.create(**kwargs)

    thinking = ""
    content = ""
    tool_calls: Dict[int, Dict[str, Any]] = {}
    in_thinking_block = False
    in_content_block = False

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not in_thinking_block:
                in_thinking_block = True
                sink.write_chat("\n[thinking]\n", kind="header")
            thinking += reasoning
            sink.write_chat(reasoning, kind="thinking")

        if delta.content:
            if in_thinking_block and not in_content_block:
                sink.write_chat("\n[answer]\n", kind="header")
            in_content_block = True
            content += delta.content
            sink.write_chat(delta.content, kind="answer")

        for tc in (delta.tool_calls or []):
            idx = tc.index if tc.index is not None else 0
            slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments

    if in_thinking_block or in_content_block:
        sink.write_chat("\n", kind="newline")

    assembled_tool_calls = [
        {
            "id": slot["id"] or f"call_{i}",
            "type": "function",
            "function": {
                "name": slot["name"],
                "arguments": slot["arguments"] or "{}",
            },
        }
        for i, slot in sorted(tool_calls.items())
        if slot["name"]
    ]

    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    if assembled_tool_calls:
        msg["tool_calls"] = assembled_tool_calls
    if thinking:
        msg["reasoning_content"] = thinking
    messages.append(msg)
    return msg


def _decode_args(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except json.JSONDecodeError:
        return {}


def _invoke_tool(name: str, arguments: Dict[str, Any]) -> str:
    fn = REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        result = fn(**arguments)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Tool execution failed ({type(exc).__name__}): {exc}"


def _run_turn(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
    sink: Sink,
) -> None:
    while True:
        msg = _stream_once(client, model, messages, tools, sink)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return
        for call in tool_calls:
            name = call["function"]["name"]
            args = _decode_args(call["function"]["arguments"])
            sink.write_chat(f"\n[tool] {name}({json.dumps(args)})\n", kind="tool_call")
            result = _invoke_tool(name, args)
            preview = result[:400] + ("..." if len(result) > 400 else "")
            sink.write_chat(f"[tool result] {preview}\n", kind="tool_result")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": result,
            })


# ---------------------------------------------------------------------------
# main / mode dispatch
# ---------------------------------------------------------------------------


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the local llama.cpp server.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional prompt to send immediately before entering interactive mode.",
    )
    parser.add_argument(
        "-m", "--model",
        default=os.environ.get("LLAMA_MODEL", "local"),
        help="Model label to send to llama-server (server ignores this and uses the loaded GGUF).",
    )
    parser.add_argument(
        "--llama-host",
        default=os.environ.get("LLAMA_HOST", "127.0.0.1"),
        help="llama-server host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--llama-port",
        type=int,
        default=int(os.environ.get("LLAMA_PORT", "53425")),
        help="llama-server port (default 53425, matches start-llama-server.sh).",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT,
                        help="Custom system prompt.")
    parser.add_argument("--single", action="store_true",
                        help="Run a single exchange (non-interactive) and exit.")
    parser.add_argument("--no-tools", action="store_true",
                        help="Disable tool calling entirely (chat only).")
    parser.add_argument("--split-screen", "--tui", action="store_true",
                        dest="split_screen",
                        help="Render a full-screen TUI: top=chat, bottom=SSH wire to the SUT.")
    return parser.parse_args(list(argv))


def _build_client(host: str, port: int) -> OpenAI:
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


def _banner_line(args: argparse.Namespace) -> str:
    return (
        f"llama.cpp.debugger -> http://{args.llama_host}:{args.llama_port}/v1 "
        f"(target SUT: {srv._TARGET['username']}@{srv._TARGET['host']}:{srv._TARGET['port']})"
    )


def _model_loop(
    args: argparse.Namespace,
    client: OpenAI,
    tools: List[Dict[str, Any]] | None,
    messages: List[Dict[str, Any]],
    sink: Sink,
) -> None:
    """Drive the chat. Identical for both stdout and TUI sinks."""
    interactive = not args.single

    if args.prompt:
        sink.write_chat(f"you> {args.prompt}\n", kind="user_echo")
        messages.append({"role": "user", "content": args.prompt})
        _run_turn(client, args.model, messages, tools, sink)
        if not interactive:
            sink.close()
            return

    while True:
        user_input = sink.read_user("\nyou> ")
        if user_input is None:
            break
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit", "q"}:
            break
        messages.append({"role": "user", "content": stripped})
        _run_turn(client, args.model, messages, tools, sink)
    sink.close()


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    client = _build_client(args.llama_host, args.llama_port)
    tools = None if args.no_tools else TOOL_SPEC
    messages: List[Dict[str, Any]] = [{"role": "system", "content": args.system}]

    if args.split_screen:
        sink = TuiSink()
        srv.set_ssh_tap(sink.write_ssh)
        sink.banner(_banner_line(args))
        worker = threading.Thread(
            target=_model_loop,
            args=(args, client, tools, messages, sink),
            daemon=True,
        )
        worker.start()
        try:
            sink.app.run()
        finally:
            srv.set_ssh_tap(None)
            # The worker may still be inside an OpenAI streaming call or an
            # SSH command; we leave it as a daemon so the process exits.
        return

    sink = StdoutSink()
    sink.banner(_banner_line(args))
    _model_loop(args, client, tools, messages, sink)


if __name__ == "__main__":
    main()
