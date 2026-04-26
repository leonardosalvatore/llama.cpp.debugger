"""FastMCP server for llama.cpp.debugger.

Exposes four tool namespaces, all driven over SSH by ``_run_ssh_cmd`` against
a configurable target host (defaults to the QEMU SUT brought up by
``run_linux_in_qemu.sh``):

* ``systemd_*``    - systemd / journald management
* ``linux_*``      - generic filesystem and process inspection
* ``compiler_*``   - gcc / g++ / make / cmake invocations
* ``gdb_*``        - debugger control via a persistent ``tmux`` session
* ``configuration_*`` - point the agent at a different target host
"""

from __future__ import annotations

import shlex
import socket
import sys
import time
from typing import Any

import paramiko
from fastmcp import FastMCP

mcp = FastMCP("llama.cpp.debugger")


# --- Target host configuration ------------------------------------------------

_TARGET: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 2222,
    "username": "debian",
    "password": "debian",
}

# Persistent tmux session name used by all gdb_* tools on the target host.
_GDB_TMUX = "llamadbg"

# Hard cap for any tool output returned to the model. Without this a single
# `ls /usr/bin` (~3000 lines) blows the LLM context window.
_MAX_TOOL_OUTPUT_CHARS = 6000
_MAX_TOOL_OUTPUT_LINES = 200


def _truncate(text: str) -> str:
    """Cap ``text`` to a budget that won't blow the model's context window."""
    if not text:
        return text
    lines = text.splitlines()
    truncated = False
    if len(lines) > _MAX_TOOL_OUTPUT_LINES:
        head = _MAX_TOOL_OUTPUT_LINES // 2
        tail = _MAX_TOOL_OUTPUT_LINES - head
        omitted = len(lines) - _MAX_TOOL_OUTPUT_LINES
        lines = (
            lines[:head]
            + [f"... [{omitted} lines omitted] ..."]
            + lines[-tail:]
        )
        truncated = True
    out = "\n".join(lines)
    if len(out) > _MAX_TOOL_OUTPUT_CHARS:
        keep = _MAX_TOOL_OUTPUT_CHARS - 64
        out = out[: keep // 2] + "\n... [output truncated] ...\n" + out[-keep // 2 :]
        truncated = True
    if truncated and not out.endswith("..."):
        out += "\n[output truncated to fit context]"
    return out


# Hard ceiling so a misbehaving remote command can never hang the whole agent.
# Backgrounded jobs that forget to release the SSH channel will hit this;
# foreground commands that legitimately take long should override per-call.
_SSH_DEFAULT_TIMEOUT = 60.0

# Optional observer that gets invoked with every (command, raw_output) pair
# sent over SSH. Used by the split-screen TUI to mirror the SSH wire to a
# dedicated panel. Stays None by default so non-TUI usage has zero overhead.
_SSH_TAP: Any = None  # type: Callable[[str, str], None] | None


def set_ssh_tap(tap: Any) -> None:
    """Register a ``tap(cmd, output)`` callback fired on every SSH command.

    Pass ``None`` to detach. The tap receives the *raw* (untruncated) output
    so the TUI can paginate it independently of the model-facing truncation.
    """
    global _SSH_TAP
    _SSH_TAP = tap


def _run_ssh_cmd(cmd: str, timeout: float = _SSH_DEFAULT_TIMEOUT) -> str:
    """Run ``cmd`` on the configured target host over SSH and return stdout+stderr.

    Caps total wait time at ``timeout`` seconds. If the remote side never
    sends EOF (typical bug: a backgrounded process inherits the SSH channel
    fd's), the channel is force-closed and any partial output returned with
    a trailing ``[ssh channel timed out ...]`` marker.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        _TARGET["host"],
        port=int(_TARGET["port"]),
        username=_TARGET["username"],
        password=_TARGET["password"],
        allow_agent=False,
        look_for_keys=False,
        timeout=10.0,
    )
    try:
        transport = ssh.get_transport()
        if transport is None:
            raise RuntimeError("ssh transport unavailable")
        chan = transport.open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)
        out_buf = bytearray()
        err_buf = bytearray()
        deadline = time.monotonic() + timeout
        timed_out = False
        while True:
            got_any = False
            if chan.recv_ready():
                out_buf.extend(chan.recv(65536))
                got_any = True
            if chan.recv_stderr_ready():
                err_buf.extend(chan.recv_stderr(65536))
                got_any = True
            if (
                chan.exit_status_ready()
                and not chan.recv_ready()
                and not chan.recv_stderr_ready()
            ):
                break
            if time.monotonic() > deadline:
                timed_out = True
                break
            if not got_any:
                try:
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
        if timed_out:
            try:
                chan.close()
            except Exception:  # noqa: BLE001
                pass
            err_buf.extend(
                f"\n[ssh channel timed out after {timeout:.1f}s]".encode()
            )
        out = out_buf.decode(errors="replace")
        err = err_buf.decode(errors="replace")
        if err and not out:
            raw = err
        elif err:
            raw = f"{out}\n[stderr]\n{err}"
        else:
            raw = out
        if _SSH_TAP is not None:
            try:
                _SSH_TAP(cmd, raw)
            except Exception:  # noqa: BLE001
                pass
        return _truncate(raw)
    except (socket.timeout, paramiko.SSHException) as exc:
        msg = f"[ssh error: {type(exc).__name__}: {exc}]"
        if _SSH_TAP is not None:
            try:
                _SSH_TAP(cmd, msg)
            except Exception:  # noqa: BLE001
                pass
        return msg
    finally:
        ssh.close()


def _log(name: str, **kwargs: Any) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"[llama.cpp.debugger] {name}({args})", file=sys.stderr, flush=True)


# --- configuration_* ----------------------------------------------------------


@mcp.tool()
def configuration_setTargetHost(
    host: str = "127.0.0.1",
    port: int = 2222,
    username: str = "debian",
    password: str = "debian",
) -> str:
    """Point all subsequent tools at a different SSH target.

    Defaults to the QEMU SUT exposed by ``run_linux_in_qemu.sh``
    (127.0.0.1:2222, debian/debian).
    """
    _log("configuration_setTargetHost", host=host, port=port, username=username)
    _TARGET.update(host=host, port=int(port), username=username, password=password)
    return f"Target set to {username}@{host}:{port}"


@mcp.tool()
def configuration_getTargetHost() -> dict[str, Any]:
    """Return the currently configured target host (password redacted)."""
    _log("configuration_getTargetHost")
    return {
        "host": _TARGET["host"],
        "port": _TARGET["port"],
        "username": _TARGET["username"],
        "password": "***" if _TARGET["password"] else "",
    }


# --- systemd_* ----------------------------------------------------------------


@mcp.tool()
def systemd_get_service_status(service_name: str) -> str:
    """Return ``systemctl status`` for the given service."""
    _log("systemd_get_service_status", service_name=service_name)
    return _run_ssh_cmd(f"systemctl status --no-pager {shlex.quote(service_name)}")


@mcp.tool()
def systemd_read_journal(service_name: str | None = None, lines: int = 50) -> str:
    """Read recent journal entries, optionally filtered by service."""
    _log("systemd_read_journal", service_name=service_name, lines=lines)
    if service_name:
        return _run_ssh_cmd(
            f"journalctl -u {shlex.quote(service_name)} -n {int(lines)} --no-pager"
        )
    return _run_ssh_cmd(f"journalctl -n {int(lines)} --no-pager")


@mcp.tool()
def systemd_restart_service(service_name: str) -> str:
    """Restart the given systemd service (uses sudo)."""
    _log("systemd_restart_service", service_name=service_name)
    return _run_ssh_cmd(f"sudo systemctl restart {shlex.quote(service_name)} 2>&1 && echo OK")


@mcp.tool()
def systemd_start_service(service_name: str) -> str:
    """Start the given systemd service (uses sudo)."""
    _log("systemd_start_service", service_name=service_name)
    return _run_ssh_cmd(f"sudo systemctl start {shlex.quote(service_name)} 2>&1 && echo OK")


@mcp.tool()
def systemd_stop_service(service_name: str) -> str:
    """Stop the given systemd service (uses sudo)."""
    _log("systemd_stop_service", service_name=service_name)
    return _run_ssh_cmd(f"sudo systemctl stop {shlex.quote(service_name)} 2>&1 && echo OK")


@mcp.tool()
def systemd_enable_service(service_name: str) -> str:
    """Enable the given systemd service so it starts on boot (uses sudo)."""
    _log("systemd_enable_service", service_name=service_name)
    return _run_ssh_cmd(f"sudo systemctl enable {shlex.quote(service_name)} 2>&1")


@mcp.tool()
def systemd_disable_service(service_name: str) -> str:
    """Disable the given systemd service from auto-start (uses sudo)."""
    _log("systemd_disable_service", service_name=service_name)
    return _run_ssh_cmd(f"sudo systemctl disable {shlex.quote(service_name)} 2>&1")


@mcp.tool()
def systemd_list_services() -> str:
    """List every systemd service unit with its current state."""
    _log("systemd_list_services")
    return _run_ssh_cmd("systemctl list-units --type=service --all --no-pager")


@mcp.tool()
def systemd_get_uptime() -> str:
    """Return the system uptime string."""
    _log("systemd_get_uptime")
    return _run_ssh_cmd("uptime")


@mcp.tool()
def systemd_daemon_reload() -> str:
    """Run ``systemctl daemon-reload`` to pick up unit file changes."""
    _log("systemd_daemon_reload")
    return _run_ssh_cmd("sudo systemctl daemon-reload 2>&1 && echo OK")


# --- linux_* ------------------------------------------------------------------


@mcp.tool()
def linux_list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """List directory contents (``ls -lh``)."""
    _log("linux_list_directory", path=path, show_hidden=show_hidden)
    flags = "-lhA" if show_hidden else "-lh"
    return _run_ssh_cmd(f"ls {flags} {shlex.quote(path)}")


@mcp.tool()
def linux_read_file(path: str, max_bytes: int = 65536) -> str:
    """Read up to ``max_bytes`` bytes from ``path``."""
    _log("linux_read_file", path=path, max_bytes=max_bytes)
    return _run_ssh_cmd(f"head -c {int(max_bytes)} {shlex.quote(path)}")


@mcp.tool()
def linux_write_file(path: str, content: str) -> str:
    """Overwrite ``path`` with ``content`` (uses sudo so system files work too)."""
    _log("linux_write_file", path=path, bytes=len(content))
    cmd = (
        f"sudo tee {shlex.quote(path)} > /dev/null << 'LLAMA_DBG_EOF'\n"
        f"{content}\n"
        f"LLAMA_DBG_EOF\n"
        f"echo wrote $(wc -c < {shlex.quote(path)}) bytes to {shlex.quote(path)}"
    )
    return _run_ssh_cmd(cmd)


@mcp.tool()
def linux_append_file(path: str, content: str) -> str:
    """Append ``content`` to ``path`` (uses sudo)."""
    _log("linux_append_file", path=path, bytes=len(content))
    cmd = (
        f"sudo tee -a {shlex.quote(path)} > /dev/null << 'LLAMA_DBG_EOF'\n"
        f"{content}\n"
        f"LLAMA_DBG_EOF\n"
        f"echo appended to {shlex.quote(path)}"
    )
    return _run_ssh_cmd(cmd)


@mcp.tool()
def linux_remove(path: str, recursive: bool = False) -> str:
    """Remove a file or (recursively) a directory (uses sudo)."""
    _log("linux_remove", path=path, recursive=recursive)
    flags = "-rf" if recursive else "-f"
    return _run_ssh_cmd(f"sudo rm {flags} {shlex.quote(path)} 2>&1 && echo OK")


@mcp.tool()
def linux_make_directory(path: str, parents: bool = True) -> str:
    """Create a directory (uses sudo, ``-p`` by default)."""
    _log("linux_make_directory", path=path, parents=parents)
    flags = "-p" if parents else ""
    return _run_ssh_cmd(f"sudo mkdir {flags} {shlex.quote(path)} 2>&1 && echo OK")


@mcp.tool()
def linux_copy(src: str, dst: str, recursive: bool = False) -> str:
    """Copy ``src`` to ``dst`` (uses sudo)."""
    _log("linux_copy", src=src, dst=dst, recursive=recursive)
    flags = "-r" if recursive else ""
    return _run_ssh_cmd(
        f"sudo cp {flags} {shlex.quote(src)} {shlex.quote(dst)} 2>&1 && echo OK"
    )


@mcp.tool()
def linux_move(src: str, dst: str) -> str:
    """Move/rename ``src`` to ``dst`` (uses sudo)."""
    _log("linux_move", src=src, dst=dst)
    return _run_ssh_cmd(f"sudo mv {shlex.quote(src)} {shlex.quote(dst)} 2>&1 && echo OK")


@mcp.tool()
def linux_find(path: str, name_pattern: str = "*") -> str:
    """Run ``find PATH -name PATTERN``."""
    _log("linux_find", path=path, name_pattern=name_pattern)
    return _run_ssh_cmd(
        f"find {shlex.quote(path)} -name {shlex.quote(name_pattern)} 2>&1 | head -200"
    )


@mcp.tool()
def linux_grep(pattern: str, path: str = ".", recursive: bool = True) -> str:
    """Run grep on the target host."""
    _log("linux_grep", pattern=pattern, path=path, recursive=recursive)
    flags = "-rni" if recursive else "-ni"
    return _run_ssh_cmd(
        f"grep {flags} {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null | head -200"
    )


@mcp.tool()
def linux_get_processes() -> str:
    """List processes (``ps -ef``)."""
    _log("linux_get_processes")
    return _run_ssh_cmd("ps -ef")


@mcp.tool()
def linux_disk_usage(path: str = "/") -> str:
    """Return ``df -h PATH``."""
    _log("linux_disk_usage", path=path)
    return _run_ssh_cmd(f"df -h {shlex.quote(path)}")


@mcp.tool()
def linux_which(binary: str) -> str:
    """Return the full path of an executable (or empty if not on PATH)."""
    _log("linux_which", binary=binary)
    return _run_ssh_cmd(f"which {shlex.quote(binary)} || echo 'not found: {binary}'")


@mcp.tool()
def linux_run_command(command: str, cwd: str = "") -> str:
    """Run an arbitrary shell ``command`` on the target host and return stdout/stderr.

    Use this for one-shot foreground commands (run a binary, exec a script,
    chain a pipeline). For long-running / never-returning programs that
    should keep running after the tool returns, use ``linux_run_in_background``
    instead. Output is capped to fit the model context.
    """
    _log("linux_run_command", command=command, cwd=cwd or None)
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    return _run_ssh_cmd(f"{prefix}{command} 2>&1; echo --rc=$?--")


@mcp.tool()
def linux_run_in_background(
    command: str, cwd: str = "", log_path: str = "/tmp/llamadbg_bg.log"
) -> str:
    """Launch ``command`` detached from the SSH session and return its PID.

    Stdout+stderr are redirected to ``log_path`` (default ``/tmp/llamadbg_bg.log``).
    Use this when the prompt asks to "run the program in the background", to
    spawn a long-running process for later inspection / gdb attach. Returns
    a single line ``PID=<pid>`` plus the log path.
    """
    _log("linux_run_in_background", command=command, cwd=cwd or None, log_path=log_path)
    inner = command if not cwd else f"cd {shlex.quote(cwd)} && exec {command}"
    wrapped = (
        f"setsid bash -c {shlex.quote(inner)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null &\n"
        f"BG_PID=$!\n"
        f"disown 2>/dev/null || true\n"
        f"echo PID=$BG_PID\n"
        f"echo log={log_path}\n"
        f"exit 0\n"
    )
    return _run_ssh_cmd(wrapped)


# --- compiler_* ---------------------------------------------------------------


@mcp.tool()
def compiler_gcc(
    source: str,
    output: str | None = None,
    flags: str = "-g -O0",
    language: str = "c",
) -> str:
    """Compile ``source`` with ``gcc`` (C) or ``g++`` (C++)."""
    _log("compiler_gcc", source=source, output=output, flags=flags, language=language)
    cc = "g++" if language.lower() in {"c++", "cpp", "cxx"} else "gcc"
    out = output or (source.rsplit(".", 1)[0] + ".out")
    cmd = f"{cc} {flags} -o {shlex.quote(out)} {shlex.quote(source)} 2>&1; echo ---rc=$?---"
    return _run_ssh_cmd(cmd)


@mcp.tool()
def compiler_make(directory: str = ".", target: str | None = None, jobs: int | None = None) -> str:
    """Run ``make`` in ``directory`` (optionally with ``-jN`` and a target)."""
    _log("compiler_make", directory=directory, target=target, jobs=jobs)
    parts = ["make", "-C", shlex.quote(directory)]
    if jobs:
        parts.append(f"-j{int(jobs)}")
    if target:
        parts.append(shlex.quote(target))
    parts.append("2>&1; echo ---rc=$?---")
    return _run_ssh_cmd(" ".join(parts))


@mcp.tool()
def compiler_cmake_configure(
    source_dir: str,
    build_dir: str,
    build_type: str = "Debug",
    extra_flags: str = "",
) -> str:
    """Run ``cmake -S source_dir -B build_dir`` with the given build type."""
    _log(
        "compiler_cmake_configure",
        source_dir=source_dir,
        build_dir=build_dir,
        build_type=build_type,
    )
    cmd = (
        f"cmake -S {shlex.quote(source_dir)} -B {shlex.quote(build_dir)} "
        f"-DCMAKE_BUILD_TYPE={shlex.quote(build_type)} {extra_flags} 2>&1; "
        f"echo ---rc=$?---"
    )
    return _run_ssh_cmd(cmd)


@mcp.tool()
def compiler_cmake_build(build_dir: str, target: str | None = None, jobs: int | None = None) -> str:
    """Run ``cmake --build build_dir`` (optionally targeting one rule)."""
    _log("compiler_cmake_build", build_dir=build_dir, target=target, jobs=jobs)
    parts = ["cmake", "--build", shlex.quote(build_dir)]
    if target:
        parts += ["--target", shlex.quote(target)]
    if jobs:
        parts += ["-j", str(int(jobs))]
    parts.append("2>&1; echo ---rc=$?---")
    return _run_ssh_cmd(" ".join(parts))


# --- gdb_* (persistent tmux session on the target) ----------------------------


def _gdb_kill_existing() -> None:
    _run_ssh_cmd(f"tmux kill-session -t {_GDB_TMUX} 2>/dev/null; true")


def _gdb_start(inner_cmd: str) -> str:
    """Spawn a fresh tmux session running ``inner_cmd`` (a gdb invocation)."""
    _gdb_kill_existing()
    spawn = (
        f"tmux new-session -d -s {_GDB_TMUX} "
        f"-x 200 -y 50 "
        f"{shlex.quote(inner_cmd)} 2>&1"
    )
    _run_ssh_cmd(spawn)
    time.sleep(1.0)
    return _gdb_capture()


def _gdb_capture(lines: int = 80) -> str:
    raw = _run_ssh_cmd(
        f"tmux capture-pane -t {_GDB_TMUX} -p -S -{int(lines)} 2>&1 || "
        f"echo 'gdb session not running'"
    )
    return "\n".join(_strip_blank_padding(raw.splitlines()))


def _strip_blank_padding(lines: list[str]) -> list[str]:
    """Drop leading/trailing all-whitespace lines (tmux scrollback padding)."""
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _gdb_send(line: str, settle: float = 0.4, capture_lines: int = 80) -> str:
    quoted = shlex.quote(line)
    _run_ssh_cmd(
        f"tmux send-keys -t {_GDB_TMUX} {quoted} Enter 2>&1 || "
        f"echo 'gdb session not running'"
    )
    time.sleep(settle)
    return _gdb_capture(capture_lines)


@mcp.tool()
def gdb_start_session_attach(pid: int) -> str:
    """Attach gdb to a running process by PID inside a fresh tmux session."""
    _log("gdb_start_session_attach", pid=pid)
    return _gdb_start(f"sudo gdb -q -p {int(pid)}")


@mcp.tool()
def gdb_start_session_run(binary: str, args: str = "") -> str:
    """Open ``binary`` (with optional ``args``) under gdb in a fresh tmux session.

    The binary is LOADED but NOT EXECUTING. Typical workflow:
        1. gdb_start_session_run(binary)
        2. gdb_break(location)              # optional
        3. gdb_run()                        # actually starts the program
        4. gdb_continue / gdb_step / gdb_next / gdb_print ...
    Calling gdb_continue before gdb_run yields "The program is not being run".
    """
    _log("gdb_start_session_run", binary=binary, args=args)
    inner = f"sudo gdb -q --args {shlex.quote(binary)} {args}".rstrip()
    return _gdb_start(inner)


@mcp.tool()
def gdb_run() -> str:
    """Start (or restart) the inferior under gdb (``run``).

    Required after ``gdb_start_session_run`` before any execution-related
    command (``continue``, ``step``, ``print`` of a runtime variable, ...)
    will work. After this returns the program is running under gdb's
    control, paused at the first breakpoint if one was set.
    """
    _log("gdb_run")
    return _gdb_send("run", settle=0.8)


@mcp.tool()
def gdb_start_session_core(binary: str, core: str) -> str:
    """Open a core dump for ``binary`` in a fresh tmux session."""
    _log("gdb_start_session_core", binary=binary, core=core)
    return _gdb_start(f"sudo gdb -q {shlex.quote(binary)} {shlex.quote(core)}")


@mcp.tool()
def gdb_send_command(command: str) -> str:
    """Send an arbitrary command to the running gdb tmux session and return the pane."""
    _log("gdb_send_command", command=command)
    return _gdb_send(command)


@mcp.tool()
def gdb_read_output(lines: int = 200) -> str:
    """Capture the last ``lines`` of the gdb tmux pane without sending a command."""
    _log("gdb_read_output", lines=lines)
    return _gdb_capture(lines)


@mcp.tool()
def gdb_break(location: str) -> str:
    """Set a breakpoint at ``location`` (e.g. ``main``, ``foo.c:42``)."""
    _log("gdb_break", location=location)
    return _gdb_send(f"break {location}")


@mcp.tool()
def gdb_continue() -> str:
    """Continue execution (``continue``)."""
    _log("gdb_continue")
    return _gdb_send("continue", settle=0.6)


@mcp.tool()
def gdb_step() -> str:
    """Step into (``step``)."""
    _log("gdb_step")
    return _gdb_send("step")


@mcp.tool()
def gdb_next() -> str:
    """Step over (``next``)."""
    _log("gdb_next")
    return _gdb_send("next")


@mcp.tool()
def gdb_finish() -> str:
    """Run until the current function returns (``finish``)."""
    _log("gdb_finish")
    return _gdb_send("finish", settle=0.6)


@mcp.tool()
def gdb_print(expr: str) -> str:
    """Evaluate ``print expr`` in gdb."""
    _log("gdb_print", expr=expr)
    return _gdb_send(f"print {expr}")


@mcp.tool()
def gdb_backtrace(depth: int = 20) -> str:
    """Print a backtrace (``bt N``)."""
    _log("gdb_backtrace", depth=depth)
    return _gdb_send(f"bt {int(depth)}")


@mcp.tool()
def gdb_info_registers() -> str:
    """Dump CPU registers (``info registers``)."""
    _log("gdb_info_registers")
    return _gdb_send("info registers")


@mcp.tool()
def gdb_info_threads() -> str:
    """List threads in the inferior (``info threads``)."""
    _log("gdb_info_threads")
    return _gdb_send("info threads")


@mcp.tool()
def gdb_list_breakpoints() -> str:
    """List currently set breakpoints (``info breakpoints``)."""
    _log("gdb_list_breakpoints")
    return _gdb_send("info breakpoints")


@mcp.tool()
def gdb_quit() -> str:
    """Quit gdb and tear down the tmux session."""
    _log("gdb_quit")
    _run_ssh_cmd(
        f"tmux send-keys -t {_GDB_TMUX} 'quit' Enter 2>/dev/null; "
        f"sleep 0.2; tmux send-keys -t {_GDB_TMUX} 'y' Enter 2>/dev/null; true"
    )
    time.sleep(0.4)
    _gdb_kill_existing()
    return "gdb session closed"


def main() -> None:
    """Start the FastMCP server.

    Attempts a best-effort SSH probe against the configured target so the
    operator can see at boot whether the SUT is reachable, but does NOT abort
    if it fails: the agent can still call ``configuration_setTargetHost`` at
    runtime to point the server at a different (or now-running) host.
    """
    try:
        probe = _run_ssh_cmd("echo 'llama.cpp.debugger MCP server up'").strip()
        print(
            f"[llama.cpp.debugger] target {_TARGET['username']}@{_TARGET['host']}:"
            f"{_TARGET['port']} reachable: {probe}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[llama.cpp.debugger] WARNING: target "
            f"{_TARGET['username']}@{_TARGET['host']}:{_TARGET['port']} not reachable "
            f"({type(exc).__name__}: {exc}). Use configuration_setTargetHost to repoint.",
            file=sys.stderr,
            flush=True,
        )
    mcp.run()


if __name__ == "__main__":
    main()
