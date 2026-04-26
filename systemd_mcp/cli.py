"""Command-line client for llama.cpp.debugger.

Talks to a local ``llama-server`` (started via ``./start-llama-server.sh``)
through its OpenAI-compatible ``/v1/chat/completions`` endpoint and exposes
the same tool surface as the FastMCP server in ``systemd_mcp.server``.

The thinking trace ships in the ``reasoning_content`` delta field thanks to
``--reasoning-format deepseek`` in ``start-llama-server.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, Iterable, List

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
    # configuration_*
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

    # systemd_*
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

    # linux_*
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

    # compiler_*
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

    # gdb_*
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
    _tool("gdb_step", "Step into."),
    _tool("gdb_next", "Step over."),
    _tool("gdb_finish", "Run until the current function returns."),
    _tool("gdb_print", "Evaluate `print expr` in gdb.",
          {"expr": _STR}, ["expr"]),
    _tool("gdb_backtrace", "Print a backtrace of `depth` frames.",
          {"depth": _INT}),
    _tool("gdb_info_registers", "Dump CPU registers."),
    _tool("gdb_info_threads", "List threads in the inferior."),
    _tool("gdb_list_breakpoints", "List currently set breakpoints."),
    _tool("gdb_quit", "Quit gdb and tear down the tmux session."),
]


def _unwrap(name: str) -> Callable[..., Any]:
    """Return the plain Python callable for a server.py tool.

    ``@mcp.tool()`` wraps each function in a FastMCP ``FunctionTool`` object
    that is not directly callable; the original function lives on ``.fn``.
    """
    obj = getattr(srv, name)
    return obj.fn if hasattr(obj, "fn") else obj


REGISTRY: Dict[str, Callable[..., Any]] = {
    spec["function"]["name"]: _unwrap(spec["function"]["name"])
    for spec in TOOL_SPEC
}


# ---------------------------------------------------------------------------
# CLI plumbing
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
    return parser.parse_args(list(argv))


def _build_client(host: str, port: int) -> OpenAI:
    return OpenAI(base_url=f"http://{host}:{port}/v1", api_key="not-needed")


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


def _stream_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]] | None,
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
                print("\n[thinking]", flush=True)
            thinking += reasoning
            print(reasoning, end="", flush=True)

        if delta.content:
            if in_thinking_block and not in_content_block:
                print("\n[answer]", flush=True)
            in_content_block = True
            content += delta.content
            print(delta.content, end="", flush=True)

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
        print()

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
) -> None:
    while True:
        msg = _stream_once(client, model, messages, tools)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return
        for call in tool_calls:
            name = call["function"]["name"]
            args = _decode_args(call["function"]["arguments"])
            print(f"\n[tool] {name}({json.dumps(args)})", flush=True)
            result = _invoke_tool(name, args)
            print(f"[tool result] {result[:400]}{'...' if len(result) > 400 else ''}",
                  flush=True)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": result,
            })


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    client = _build_client(args.llama_host, args.llama_port)
    tools = None if args.no_tools else TOOL_SPEC
    messages: List[Dict[str, Any]] = [{"role": "system", "content": args.system}]
    interactive = not args.single

    print(f"llama.cpp.debugger -> http://{args.llama_host}:{args.llama_port}/v1 "
          f"(target SUT: {srv._TARGET['username']}@{srv._TARGET['host']}:{srv._TARGET['port']})",
          flush=True)

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        _run_turn(client, args.model, messages, tools)
        if not interactive:
            return

    while True:
        try:
            user_input = input("\nyou> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit", "q"}:
            break
        messages.append({"role": "user", "content": stripped})
        _run_turn(client, args.model, messages, tools)


if __name__ == "__main__":
    main()
