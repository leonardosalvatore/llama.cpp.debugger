"""FastMCP server for llama.cpp.debugger.

Exposes five tool namespaces, all driven over SSH by ``_run_ssh_cmd`` against
a configurable target host (defaults to the QEMU SUT brought up by
``run_linux_in_qemu.sh``), plus one local-only retrieval namespace:

* ``systemd_*``    - systemd / journald management
* ``linux_*``      - generic filesystem and process inspection
* ``compiler_*``   - gcc / g++ / make / cmake invocations
* ``gdb_*``        - debugger control via a persistent ``tmux`` session
* ``perf_*``       - Linux ``perf`` profiling (stat / record / report /
                     top_functions), plus ``perf_heatmap`` which renders a
                     function-overhead treemap PNG on the host and
                     ``perf_open_hotspot`` which opens the recording in the
                     interactive Hotspot GUI on the host.
* ``configuration_*`` - point the agent at a different target host
* ``rag_*``        - dense retrieval (``rag_search``) over the LVGL docs
                     vector DB built by ``llama_debugger_vectordb`` (uses the
                     embedding llama-server on port 53426; soft-fails when
                     either the server or the DB is missing).
"""

from __future__ import annotations

import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
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

# Env vars exported on every SSH command run by ``_run_ssh_cmd``. paramiko's
# ``exec_command`` uses a non-interactive, non-login shell so ``~/.bashrc`` /
# ``~/.profile`` are NOT sourced and most env vars set by sshd are stripped
# (and ``SendEnv`` / ``AcceptEnv`` are off by default). We therefore prepend
# explicit ``export K=V; ...`` to the command itself.
#
# DISPLAY=:0 lets GUI programs (lvglsim, glxgears, ...) reach whatever X
# server / XWayland socket the desktop session opened on the SUT. If you
# need an XAUTHORITY cookie or a different display, mutate this dict via
# ``configuration_setRemoteEnv``.
_REMOTE_ENV: dict[str, str] = {
    "DISPLAY": ":0",
}


def _seed_remote_env_from_environ() -> None:
    """Merge ``LLAMA_DEBUGGER_REMOTE_ENV`` (``K=V,K=V``) into ``_REMOTE_ENV``.

    Lets a launcher (e.g. ``run_mcp_demo.sh``) pre-set SUT env vars such as
    ``XAUTHORITY`` deterministically at startup, so the model never has to
    call ``configuration_setRemoteEnv`` for them - small models tend to
    confuse that with ``configuration_setTargetHost`` and derail.
    """
    raw = os.environ.get("LLAMA_DEBUGGER_REMOTE_ENV", "").strip()
    if not raw:
        return
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        if key:
            _REMOTE_ENV[key] = value.strip()


_seed_remote_env_from_environ()

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

# Optional observer fired right after ``linux_run_in_background`` has spawned a
# process on the SUT. Receives ``(pid, log_path, target_config)`` so a CLI can
# open its own SSH session and tail the log until the process exits. Stays
# None by default so non-streaming usage has zero overhead.
_BG_TAP: Any = None  # type: Callable[[int, str, dict[str, Any]], None] | None


def set_ssh_tap(tap: Any) -> None:
    """Register a ``tap(cmd, output)`` callback fired on every SSH command.

    Pass ``None`` to detach. The tap receives the *raw* (untruncated) output
    so the TUI can paginate it independently of the model-facing truncation.
    """
    global _SSH_TAP
    _SSH_TAP = tap


def set_bg_tap(tap: Any) -> None:
    """Register a ``tap(pid, log_path, target)`` callback fired on bg launches.

    Invoked once per successful ``linux_run_in_background`` call, after the
    PID is known. ``target`` is a shallow copy of the current SSH target
    (host/port/username/password) so the consumer can open its own session.
    Pass ``None`` to detach.
    """
    global _BG_TAP
    _BG_TAP = tap


def _env_export_prefix() -> str:
    """Return ``export K=V; ...`` for every entry in ``_REMOTE_ENV``.

    Empty when the dict is empty, so we never inject a stray no-op statement.
    Values are ``shlex.quote``-d so anything goes (spaces, ``$``, quotes, ...).
    """
    if not _REMOTE_ENV:
        return ""
    parts = [f"export {k}={shlex.quote(v)};" for k, v in _REMOTE_ENV.items()]
    return " ".join(parts) + " "


def _run_ssh_cmd(cmd: str, timeout: float = _SSH_DEFAULT_TIMEOUT) -> str:
    """Run ``cmd`` on the configured target host over SSH and return stdout+stderr.

    Caps total wait time at ``timeout`` seconds. If the remote side never
    sends EOF (typical bug: a backgrounded process inherits the SSH channel
    fd's), the channel is force-closed and any partial output returned with
    a trailing ``[ssh channel timed out ...]`` marker.

    Every command is prefixed with the ``_REMOTE_ENV`` exports so GUI
    binaries inherit ``DISPLAY`` / ``XAUTHORITY`` / etc.
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
        chan.exec_command(_env_export_prefix() + cmd)
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


def _probe_ssh(
    host: str, port: int, username: str, password: str, timeout: float = 8.0
) -> tuple[str, str]:
    """Test-connect to an SSH target WITHOUT touching the live ``_TARGET``.

    Returns ``("ok", "")`` on success, ``("auth", detail)`` when the
    credentials are rejected, or ``("unreachable", detail)`` for any
    network / timeout / protocol failure. ``configuration_setTargetHost``
    uses this to validate a change before committing it, so a wrong password
    guess can never lock the agent out of a working SUT.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            host,
            port=int(port),
            username=username,
            password=password,
            allow_agent=False,
            look_for_keys=False,
            timeout=timeout,
        )
        return ("ok", "")
    except paramiko.AuthenticationException as exc:
        return ("auth", f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ("unreachable", f"{type(exc).__name__}: {exc}")
    finally:
        try:
            ssh.close()
        except Exception:  # noqa: BLE001
            pass


# --- configuration_* ----------------------------------------------------------


# Redaction / placeholder strings a model tends to invent for a password
# field it doesn't actually know - it frequently echoes back the "***" it
# saw from configuration_getTargetHost, or a literal "redacted". Treat these
# as "leave unchanged" so a stray configuration_setTargetHost call can't
# clobber the working SUT credentials and lock the agent out (which then
# surfaces as AuthenticationException on every later tool call).
_PLACEHOLDER_SECRETS = {
    "", "*", "**", "***", "****", "redacted", "<redacted>", "[redacted]",
    "changeme", "password", "your_password", "none", "null", "xxx",
}


@mcp.tool()
def configuration_setTargetHost(
    host: str | None = None,
    port: int | None = None,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Point all subsequent tools at a different SSH target.

    Only the fields you pass are changed; omitted (or null) fields keep
    their current value, so you can retarget just the host/port without
    resupplying the password. The change is VALIDATED with a quick test
    connection before it takes effect: if authentication fails the change is
    rejected and the previous working target is kept, so a wrong/guessed
    password can never lock the agent out. NEVER invent a password (redaction
    placeholders like "***" / "redacted" are ignored too). You rarely need
    this at all - the initial target is already the QEMU SUT from
    run_linux_in_qemu.sh (127.0.0.1:2222, debian/debian). To set an env var
    on the SUT (e.g. XAUTHORITY) use configuration_setRemoteEnv instead.
    """
    candidate = dict(_TARGET)
    if host:
        candidate["host"] = host
    if port:
        candidate["port"] = int(port)
    if username:
        candidate["username"] = username
    if password is not None and password.strip().lower() not in _PLACEHOLDER_SECRETS:
        candidate["password"] = password

    if candidate == _TARGET:
        _log("configuration_setTargetHost", unchanged=True,
             host=_TARGET["host"], port=_TARGET["port"],
             username=_TARGET["username"])
        return (
            f"Target unchanged: {_TARGET['username']}@{_TARGET['host']}:"
            f"{_TARGET['port']} (nothing to update)."
        )

    kind, detail = _probe_ssh(
        candidate["host"], candidate["port"],
        candidate["username"], candidate["password"],
    )
    if kind == "auth":
        _log("configuration_setTargetHost", rejected="auth",
             host=candidate["host"], port=candidate["port"],
             username=candidate["username"])
        return (
            f"Target NOT changed: authentication failed for "
            f"{candidate['username']}@{candidate['host']}:{candidate['port']} "
            f"({detail}). Kept the previous working target "
            f"{_TARGET['username']}@{_TARGET['host']}:{_TARGET['port']}. Never "
            f"guess credentials - omit fields you don't want to change."
        )

    _TARGET.update(candidate)
    _log("configuration_setTargetHost", host=_TARGET["host"],
         port=_TARGET["port"], username=_TARGET["username"])
    if kind == "unreachable":
        return (
            f"Target set to {_TARGET['username']}@{_TARGET['host']}:"
            f"{_TARGET['port']}, but it is not reachable yet ({detail}); "
            f"tools will start working once it is up."
        )
    return f"Target set to {_TARGET['username']}@{_TARGET['host']}:{_TARGET['port']}"


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


@mcp.tool()
def configuration_setRemoteEnv(name: str, value: str) -> str:
    """Set an env var that gets exported on every SSH command.

    Use this to make GUI programs reach the SUT desktop, e.g.
    ``DISPLAY=:0`` (default) or ``XAUTHORITY=/home/debian/.Xauthority``.
    Pre-set: DISPLAY=:0.
    """
    _log("configuration_setRemoteEnv", var=name, value=value)
    _REMOTE_ENV[name] = value
    return f"Remote env updated: {name}={value}"


@mcp.tool()
def configuration_unsetRemoteEnv(name: str) -> str:
    """Drop ``name`` from the always-exported remote env."""
    _log("configuration_unsetRemoteEnv", var=name)
    existed = _REMOTE_ENV.pop(name, None) is not None
    return f"Remote env removed: {name}" if existed else f"Remote env had no {name}"


@mcp.tool()
def configuration_getRemoteEnv() -> dict[str, str]:
    """Return the dict of env vars exported on every SSH command."""
    _log("configuration_getRemoteEnv")
    return dict(_REMOTE_ENV)


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


# --- File-mutation helpers ---------------------------------------------------
#
# All linux_* tools that *write* on the SUT take an explicit ``sudo`` flag,
# defaulting to ``False`` so the common case (mkdir/rm/cp/mv/write under
# /home/debian, /tmp, build dirs) doesn't escalate. Set ``sudo=True`` only
# when the path is system-owned (/etc/, /usr/, /var/, /opt/, ...). Same
# rationale as ``compiler_*`` running unprivileged.


def _maybe_sudo(sudo: bool) -> str:
    """Return ``"sudo "`` when escalation is requested, else empty string."""
    return "sudo " if sudo else ""


@mcp.tool()
def linux_write_file(path: str, content: str, sudo: bool = False) -> str:
    """Overwrite ``path`` with ``content``.

    Defaults to running unprivileged. Set ``sudo=True`` for paths the
    invoking user (``debian``) cannot write directly - typically
    ``/etc/``, ``/usr/``, ``/var/``, ``/opt/`` and any other root-owned
    tree.
    """
    _log("linux_write_file", path=path, bytes=len(content), sudo=sudo)
    cmd = (
        f"{_maybe_sudo(sudo)}tee {shlex.quote(path)} > /dev/null << 'LLAMA_DBG_EOF'\n"
        f"{content}\n"
        f"LLAMA_DBG_EOF\n"
        f"echo wrote $(wc -c < {shlex.quote(path)}) bytes to {shlex.quote(path)}"
    )
    return _run_ssh_cmd(cmd)


@mcp.tool()
def linux_append_file(path: str, content: str, sudo: bool = False) -> str:
    """Append ``content`` to ``path``.

    Defaults to running unprivileged; pass ``sudo=True`` for system paths.
    """
    _log("linux_append_file", path=path, bytes=len(content), sudo=sudo)
    cmd = (
        f"{_maybe_sudo(sudo)}tee -a {shlex.quote(path)} > /dev/null << 'LLAMA_DBG_EOF'\n"
        f"{content}\n"
        f"LLAMA_DBG_EOF\n"
        f"echo appended to {shlex.quote(path)}"
    )
    return _run_ssh_cmd(cmd)


@mcp.tool()
def linux_remove(path: str, recursive: bool = False, sudo: bool = False) -> str:
    """Remove a file or (recursively) a directory.

    Defaults to running unprivileged; pass ``sudo=True`` for root-owned
    targets.
    """
    _log("linux_remove", path=path, recursive=recursive, sudo=sudo)
    flags = "-rf" if recursive else "-f"
    return _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}rm {flags} {shlex.quote(path)} 2>&1 && echo OK"
    )


@mcp.tool()
def linux_make_directory(
    path: str, parents: bool = True, sudo: bool = False
) -> str:
    """Create a directory (``-p`` by default).

    Defaults to running unprivileged - the common case is ``mkdir`` under
    ``/home/debian``, ``/tmp``, or a build tree the SUT user owns. Pass
    ``sudo=True`` only for root-owned trees like ``/etc/`` or ``/var/``.
    """
    _log("linux_make_directory", path=path, parents=parents, sudo=sudo)
    flags = "-p" if parents else ""
    return _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}mkdir {flags} {shlex.quote(path)} 2>&1 && echo OK"
    )


@mcp.tool()
def linux_copy(
    src: str, dst: str, recursive: bool = False, sudo: bool = False
) -> str:
    """Copy ``src`` to ``dst``.

    Defaults to running unprivileged; pass ``sudo=True`` if either
    endpoint is in a root-owned tree.
    """
    _log("linux_copy", src=src, dst=dst, recursive=recursive, sudo=sudo)
    flags = "-r" if recursive else ""
    return _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}cp {flags} {shlex.quote(src)} {shlex.quote(dst)} "
        f"2>&1 && echo OK"
    )


@mcp.tool()
def linux_move(src: str, dst: str, sudo: bool = False) -> str:
    """Move/rename ``src`` to ``dst``.

    Defaults to running unprivileged; pass ``sudo=True`` if either
    endpoint is in a root-owned tree.
    """
    _log("linux_move", src=src, dst=dst, sudo=sudo)
    return _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}mv {shlex.quote(src)} {shlex.quote(dst)} "
        f"2>&1 && echo OK"
    )


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
    result = _run_ssh_cmd(wrapped)
    if _BG_TAP is not None:
        pid: int | None = None
        for line in result.splitlines():
            stripped = line.strip()
            if stripped.startswith("PID="):
                try:
                    pid = int(stripped[4:].strip())
                except ValueError:
                    pid = None
                break
        if pid is not None:
            try:
                _BG_TAP(pid, log_path, dict(_TARGET))
            except Exception:  # noqa: BLE001
                pass
    return result


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
def gdb_start_session_attach(pid: int, sudo: bool = True) -> str:
    """Attach gdb to a running process by PID inside a fresh tmux session.

    Defaults to ``sudo=True`` because attaching across UIDs (debugging a
    systemd service running as root, a setuid binary, ...) requires
    ptrace privileges. Pass ``sudo=False`` when attaching to a process
    owned by the SUT login user (e.g. a foreground program you launched
    yourself via ``linux_run_in_background``).
    """
    _log("gdb_start_session_attach", pid=pid, sudo=sudo)
    return _gdb_start(f"{_maybe_sudo(sudo)}gdb -q -p {int(pid)}")


@mcp.tool()
def gdb_start_session_run(
    binary: str, args: str = "", sudo: bool = False
) -> str:
    """Open ``binary`` (with optional ``args``) under gdb in a fresh tmux session.

    The binary is LOADED but NOT EXECUTING. Typical workflow:
        1. gdb_start_session_run(binary)
        2. gdb_break(location)              # optional
        3. gdb_run()                        # actually starts the program
        4. gdb_continue / gdb_step / gdb_next / gdb_print ...
    Calling gdb_continue before gdb_run yields "The program is not being run".

    Defaults to running unprivileged; pass ``sudo=True`` only when the
    binary needs root to inspect/run (rare - ptrace of self does not
    require it).
    """
    _log("gdb_start_session_run", binary=binary, args=args, sudo=sudo)
    inner = f"{_maybe_sudo(sudo)}gdb -q --args {shlex.quote(binary)} {args}".rstrip()
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
def gdb_start_session_core(binary: str, core: str, sudo: bool = False) -> str:
    """Open a core dump for ``binary`` in a fresh tmux session.

    Defaults to running unprivileged; pass ``sudo=True`` only when the
    binary or core file lives in a root-owned tree the SUT user can't
    read.
    """
    _log("gdb_start_session_core", binary=binary, core=core, sudo=sudo)
    return _gdb_start(
        f"{_maybe_sudo(sudo)}gdb -q {shlex.quote(binary)} {shlex.quote(core)}"
    )


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


# --- perf_* (Linux perf profiling on the target) ------------------------------
#
# All perf tools run over SSH on the SUT. Profiling a long-running or GUI
# program (e.g. ``lvglsim -b GLFW``) means ``perf record`` has to be
# time-boxed, otherwise it never returns and the SSH channel hangs until the
# hard timeout. ``perf_record`` therefore wraps the target command in
# ``timeout`` so recording stops on its own and the channel is released.
#
# The recorded ``perf.data`` stays on the SUT. ``perf_report`` /
# ``perf_top_functions`` parse it into text / JSON, and ``perf_heatmap``
# pulls that JSON back and renders a treemap heatmap PNG *on the host*
# running this server (see ``systemd_mcp.perf_heatmap``), so the operator or
# agent can display it directly.
#
# perf needs a relaxed ``kernel.perf_event_paranoid`` (set to -1 by the
# cloud-init provisioning in ``run_linux_in_qemu.sh``) to record as the
# unprivileged ``debian`` user; every tool also accepts ``sudo=True`` as a
# fallback on hosts that were not provisioned that way.

_PERF_DEFAULT_DATA = "/tmp/llamadbg_perf.data"

# Extra head-room added to a recording's ``duration`` when sizing the SSH
# channel timeout, to cover perf's own startup + post-processing (symbol
# resolution, writing perf.data) after the profiled command exits.
_PERF_RECORD_TIMEOUT_PAD = 45.0


@mcp.tool()
def perf_stat(
    command: str, cwd: str = "", events: str = "", duration: int = 0, sudo: bool = False
) -> str:
    """Run ``perf stat`` on ``command`` and return the counter summary.

    Quick top-line view (cycles, instructions, IPC, cache/branch misses,
    context switches, task-clock) without recording a full profile. Pass a
    comma-separated ``events`` list to override the default counter set. For
    a long-running / GUI program set ``duration`` (seconds) so it is
    time-boxed with ``timeout``; for a program that exits on its own leave
    it at 0. Use ``perf_record`` + ``perf_report`` / ``perf_heatmap`` for a
    per-function breakdown.
    """
    _log("perf_stat", command=command, cwd=cwd or None, events=events or None,
         duration=duration, sudo=sudo)
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    ev = f"-e {shlex.quote(events)} " if events else ""
    box = f"timeout --signal=INT {int(duration)} " if int(duration) > 0 else ""
    # Wrap in `bash -c` (see perf_record) so `command` is shell-parsed: inline
    # env prefixes (`XAUTHORITY=... ./app`), pipes, and quoting all work.
    target = f"{box}bash -c {shlex.quote(command)}"
    cap = (int(duration) + _PERF_RECORD_TIMEOUT_PAD) if int(duration) > 0 else 120.0
    return _run_ssh_cmd(
        f"{prefix}{_maybe_sudo(sudo)}perf stat {ev}-- {target} 2>&1; echo --rc=$?--",
        timeout=cap,
    )


@mcp.tool()
def perf_record(
    command: str,
    cwd: str = "",
    duration: int = 10,
    frequency: int = 999,
    call_graph: bool = True,
    output: str = _PERF_DEFAULT_DATA,
    sudo: bool = False,
) -> str:
    """Sample ``command`` with ``perf record`` for ``duration`` seconds, then stop.

    ``command`` is time-boxed with ``timeout`` so long-running / GUI
    programs (lvglsim, glxgears, a daemon, ...) stop recording on their own
    and this call returns. Records at ``frequency`` Hz and, when
    ``call_graph`` is set (default), captures the call graph so
    ``perf_report`` can build a caller/callee tree. Writes the profile to
    ``output`` on the SUT (default ``/tmp/llamadbg_perf.data``). Returns
    perf's own record summary; follow with ``perf_report`` /
    ``perf_top_functions`` / ``perf_heatmap`` to read it.
    """
    _log("perf_record", command=command, cwd=cwd or None, duration=duration,
         frequency=frequency, call_graph=call_graph, output=output, sudo=sudo)
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    cg = "-g " if call_graph else ""
    dur = max(1, int(duration))
    # Run the profiled command through `bash -c` so shell syntax and inline
    # env-var prefixes (e.g. `XAUTHORITY=... ./app -flag`) work. `timeout` and
    # `perf` exec their argv directly (no shell), so without this wrapper a
    # command like `FOO=bar ./app` makes timeout try to exec a program named
    # literally "FOO=bar" and fail with rc=127 - profiling the wrapper instead
    # of the target. `bash -c 'single cmd'` exec-optimizes into the target, so
    # `timeout`'s SIGINT still reaches it directly.
    target = f"timeout --signal=INT {dur} bash -c {shlex.quote(command)}"
    # Delete any stale profile first (output is an absolute path, so cwd is
    # irrelevant here). Without this, a recording that captures no useful
    # samples - or a run where the target failed to launch - would leave the
    # PREVIOUS run's perf.data in place, and perf_report / perf_heatmap would
    # silently visualize old data.
    cmd = (
        f"{_maybe_sudo(sudo)}rm -f {shlex.quote(output)} 2>/dev/null; "
        f"{prefix}{_maybe_sudo(sudo)}perf record -F {int(frequency)} {cg}"
        f"-o {shlex.quote(output)} -- {target} "
        f"2>&1; echo --rc=$?--"
    )
    return _run_ssh_cmd(cmd, timeout=dur + _PERF_RECORD_TIMEOUT_PAD)


@mcp.tool()
def perf_report(
    data_file: str = _PERF_DEFAULT_DATA, limit: int = 40, sudo: bool = False
) -> str:
    """Return a text ``perf report`` (symbol overhead table) for ``data_file``."""
    _log("perf_report", data_file=data_file, limit=limit, sudo=sudo)
    return _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}perf report -i {shlex.quote(data_file)} --stdio "
        f"--percent-limit 0.1 2>&1 | head -n {int(limit) + 20}",
        timeout=120.0,
    )


def _parse_perf_report(text: str, limit: int) -> list[dict[str, Any]]:
    """Extract ``(percent, command, dso, symbol)`` rows from ``perf report --stdio``.

    Matches the flat (self-overhead) layout::

        23.45%  lvglsim  liblvgl.so         [.] lv_draw_sw_blend
        10.11%  lvglsim  [kernel.kallsyms]  [k] copy_user_generic

    The single-letter map marker (``[.]`` user, ``[k]`` kernel, ...) anchors
    the split between the shared-object and symbol columns, which is more
    robust than counting whitespace when a DSO path contains spaces.
    """
    import re

    rx = re.compile(
        r"^\s*(\d+\.\d+)%\s+(\S+)\s+(.+?)\s+(\[[.\w]\])\s+(.+?)\s*$"
    )
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = rx.match(line)
        if not m:
            continue
        out.append(
            {
                "percent": float(m.group(1)),
                "command": m.group(2),
                "dso": m.group(3).strip(),
                "kind": m.group(4),
                "symbol": m.group(5).strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


def _perf_top_functions_impl(
    data_file: str, limit: int, sudo: bool
) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    # --no-children collapses the report to a single self-overhead column
    # even when perf.data carries call-graph info (otherwise perf emits both
    # a Children and a Self column, which would misalign the parser and, more
    # importantly, plot inclusive time rather than where the CPU actually is).
    raw = _run_ssh_cmd(
        f"{_maybe_sudo(sudo)}perf report -i {shlex.quote(data_file)} --stdio "
        f"-g none --no-children --percent-limit 0.01 2>&1",
        timeout=120.0,
    )
    funcs = _parse_perf_report(raw, limit)
    if not funcs:
        return {"functions": [], "count": 0, "error": _truncate(raw)}
    return {"functions": funcs, "count": len(funcs)}


@mcp.tool()
def perf_top_functions(
    data_file: str = _PERF_DEFAULT_DATA, limit: int = 25, sudo: bool = False
) -> dict[str, Any]:
    """Return the hottest functions in ``data_file`` as structured JSON.

    Parses a flat (self-overhead) ``perf report`` into a list of
    ``{percent, command, dso, kind, symbol}`` sorted hottest first. This is
    the machine-readable form ``perf_heatmap`` renders into a PNG. Soft-fails
    with ``functions=[]`` + an ``error`` string (raw perf output) when the
    data file is missing or has no resolvable samples.
    """
    _log("perf_top_functions", data_file=data_file, limit=limit, sudo=sudo)
    return _perf_top_functions_impl(data_file, limit, sudo)


@mcp.tool()
def perf_heatmap(
    data_file: str = _PERF_DEFAULT_DATA,
    output_png: str = "",
    limit: int = 30,
    title: str = "",
    sudo: bool = False,
) -> dict[str, Any]:
    """Render a function-level CPU heatmap PNG from a ``perf.data`` on the SUT.

    Pulls the hottest functions (``perf_top_functions``) over SSH, then
    renders a treemap *on the host running this server* (not the SUT): each
    tile is a function, sized by its CPU self-overhead and colored on a
    blue(cold)->red(hot) heat scale. Returns ``{"png": <local path>,
    "functions": [...], "count": N}`` so the operator/agent can open/display
    the image directly. Soft-fails with an ``error`` string when there are no
    samples or matplotlib is not installed (``poetry sync`` to add it).
    """
    _log("perf_heatmap", data_file=data_file, output_png=output_png or None,
         limit=limit, title=title or None, sudo=sudo)
    result = _perf_top_functions_impl(data_file, limit, sudo)
    funcs = result.get("functions", [])
    if not funcs:
        return {
            "png": "",
            "functions": [],
            "error": result.get("error", "no samples parsed from perf.data"),
        }
    try:
        from systemd_mcp.perf_heatmap import render_heatmap
    except ImportError as exc:
        return {
            "png": "",
            "functions": funcs,
            "error": (
                f"heatmap rendering needs matplotlib ({exc}); run `poetry sync`."
            ),
        }
    out = output_png or os.path.abspath("perf_heatmap.png")
    try:
        render_heatmap(
            funcs, out, title=title or f"perf function heatmap ({data_file})"
        )
    except Exception as exc:  # noqa: BLE001 - keep the chat alive
        return {
            "png": "",
            "functions": funcs,
            "error": f"heatmap render failed: {type(exc).__name__}: {exc}",
        }
    return {"png": out, "functions": funcs, "count": len(funcs)}


# Host path the SUT perf.data is copied to before opening it in Hotspot.
_HOST_PERF_DEFAULT = os.path.join(tempfile.gettempdir(), "llamadbg_perf.data")


def _sftp_get(remote_path: str, local_path: str) -> None:
    """Copy ``remote_path`` from the SUT to ``local_path`` on the host via SFTP."""
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
        sftp = ssh.open_sftp()
        try:
            sftp.get(remote_path, local_path)
        finally:
            sftp.close()
    finally:
        ssh.close()


@mcp.tool()
def perf_open_hotspot(
    data_file: str = _PERF_DEFAULT_DATA,
    local_path: str = "",
    with_symbols: bool = True,
    use_debuginfod: bool = False,
    sudo: bool = False,
) -> dict[str, Any]:
    """Copy a perf.data off the SUT and open it in Hotspot on the host.

    Hotspot (the KDAB perf GUI) runs on the HOST, but the recording lives on
    the SUT, so this pulls it over SFTP and launches ``hotspot`` on it,
    detached. When ``with_symbols`` (default), it first runs ``perf archive``
    on the SUT to bundle the build-id'd binaries/libraries and extracts them
    into the host build-id cache (``~/.debug``) so Hotspot can resolve SUT
    symbols across machines.

    ``use_debuginfod`` is False by default: Hotspot's perfparser otherwise
    queries any configured ``DEBUGINFOD_URLS`` for every unresolved build-id,
    which stalls indefinitely ("Loading Results...") when the SUT's distro
    libraries aren't on that server. Since the build-id archive already
    supplies the relevant symbols, we disable debuginfod for the launched
    Hotspot unless this is set True.

    Returns ``{"opened": bool, "local_path": ..., "symbols": ...}``;
    soft-fails with an ``error`` if ``hotspot`` is not on the host PATH or the
    perf.data is missing. Requires a prior perf_record.
    """
    _log("perf_open_hotspot", data_file=data_file, local_path=local_path or None,
         with_symbols=with_symbols, use_debuginfod=use_debuginfod, sudo=sudo)
    hotspot = shutil.which("hotspot")
    if hotspot is None:
        return {
            "opened": False,
            "error": "hotspot not found on host PATH; install it (e.g. `apt install hotspot`).",
        }
    local_path = local_path or _HOST_PERF_DEFAULT

    check = _run_ssh_cmd(
        f"test -s {shlex.quote(data_file)} && echo OK || echo MISSING"
    )
    if "OK" not in check:
        return {
            "opened": False,
            "error": (
                f"no usable perf.data at {data_file} on the SUT - run "
                f"perf_record first."
            ),
        }

    symbols_note = "not requested"
    if with_symbols:
        symbols_note = _perf_pull_symbols(data_file, local_path, sudo)

    try:
        _sftp_get(data_file, local_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "opened": False,
            "error": f"failed to copy perf.data to host: {type(exc).__name__}: {exc}",
        }

    env = os.environ.copy()
    if not use_debuginfod:
        # Empty (not unset) so elfutils' debuginfod client stays disabled even
        # if a system-wide /etc/debuginfod/*.urls would otherwise apply.
        env["DEBUGINFOD_URLS"] = ""
    try:
        subprocess.Popen(
            [hotspot, local_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "opened": False,
            "local_path": local_path,
            "error": f"failed to launch hotspot: {type(exc).__name__}: {exc}",
        }
    return {"opened": True, "local_path": local_path, "symbols": symbols_note}


def _perf_pull_symbols(data_file: str, local_path: str, sudo: bool) -> str:
    """Best-effort: bundle SUT build-id objects via ``perf archive`` and
    extract them into the host ``~/.debug`` cache for cross-machine symbols.

    Returns a short human-readable status; never raises (symbol resolution
    is a nice-to-have, not a hard requirement for opening the profile).
    """
    try:
        remote_dir = os.path.dirname(data_file) or "."
        remote_base = os.path.basename(data_file)
        out = _run_ssh_cmd(
            f"cd {shlex.quote(remote_dir)} && "
            f"{_maybe_sudo(sudo)}perf archive {shlex.quote(remote_base)} 2>&1; "
            f"ls -1 {shlex.quote(remote_base)}.tar.bz2 2>/dev/null",
            timeout=120.0,
        )
        remote_tar = None
        for line in out.splitlines():
            line = line.strip()
            if line.endswith(".tar.bz2"):
                remote_tar = line if line.startswith("/") else os.path.join(remote_dir, line)
        if not remote_tar:
            return "perf archive produced no bundle; SUT symbols may show as addresses"

        local_tar = local_path + ".tar.bz2"
        _sftp_get(remote_tar, local_tar)
        debug_dir = os.path.expanduser("~/.debug")
        os.makedirs(debug_dir, exist_ok=True)
        with tarfile.open(local_tar, "r:*") as tf:
            try:
                tf.extractall(debug_dir, filter="data")  # py>=3.12
            except TypeError:
                tf.extractall(debug_dir)
        return f"build-id symbols extracted to {debug_dir}"
    except Exception as exc:  # noqa: BLE001
        return f"symbol archive skipped ({type(exc).__name__}: {exc})"


# --- rag_* (local LVGL docs retrieval) ----------------------------------------
#
# Unlike every other namespace in this server, rag_* does NOT touch the SUT.
# It opens a local sqlite-vec store written by ``llama_debugger_vectordb``
# and queries an embedding llama-server on the host (default
# 127.0.0.1:53426). Both can be missing, so this namespace soft-fails:
# instead of raising, every tool returns ``{"hits": [], "error": "..."}``
# so the chat model gets an actionable hint without crashing the turn.
#
# Configurable via env vars (chosen at process start, set by run_mcp_cli.sh
# or the operator's shell):
#
#   LLAMA_DEBUGGER_VECTORDB    Path to the .db file.
#                              Default: systemd_mcp/vectordb/vector-database.db
#   LLAMA_DEBUGGER_EMBED_HOST  Embedding server host. Default: 127.0.0.1
#   LLAMA_DEBUGGER_EMBED_PORT  Embedding server port. Default: 53426
#
# Imports are deferred inside the tool body so the MCP server still starts
# cleanly when numpy / sqlite-vec aren't installed (e.g. someone running
# the agent without the vectordb extras).


_RAG_DB_PATH_DEFAULT = "systemd_mcp/vectordb/vector-database.db"

# Per-hit ``text`` length cap returned to the model. The ingester writes
# chunks of up to ~2500 chars; multiplied by k=5 hits that's 12.5 KB per
# rag_search call, and 4-5 calls per turn was overflowing the chat
# server's 32k context window. The agent only needs a snippet to decide
# whether a chunk is useful and to cite it; the full source is on disk
# and can be pulled with linux_read_file if needed. 600 chars (~150-200
# tokens) keeps a 5-call burst at ~3 KB / ~750 tokens of corpus content
# in the conversation history.
_RAG_HIT_TEXT_MAX_CHARS = int(
    os.environ.get("LLAMA_DEBUGGER_RAG_TEXT_CHARS", "600")
)


def _rag_config() -> dict[str, Any]:
    return {
        "db_path": os.environ.get("LLAMA_DEBUGGER_VECTORDB", _RAG_DB_PATH_DEFAULT),
        "embed_host": os.environ.get("LLAMA_DEBUGGER_EMBED_HOST", "127.0.0.1"),
        "embed_port": int(os.environ.get("LLAMA_DEBUGGER_EMBED_PORT", "53426")),
    }


def _trim_hit_text(text: str, limit: int = _RAG_HIT_TEXT_MAX_CHARS) -> str:
    """Cap a single hit's ``text`` to ``limit`` chars.

    Keeps the head of the chunk: for code chunks the head is the function
    signature / opening lines (the most diagnostic part); for docs the
    head is the prose right after the heading. The ``heading`` and
    ``path`` fields on the hit already give the agent everything it
    needs to fetch the rest with ``linux_read_file`` if a particular
    chunk turns out to matter.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    extra = len(text) - limit
    return (
        text[:limit].rstrip()
        + f"\n... [{extra} more chars in this chunk; "
        f"read file for full context]"
    )


@mcp.tool()
def rag_search(query: str, k: int = 5) -> dict[str, Any]:
    """Search the LVGL docs vector DB and return the top-k matching chunks.

    The DB must have been built once with ``llama_debugger_vectordb build``;
    embedding requests go to the second llama-server started by
    ``./start-llama-embedding-server.sh``. Both are checked at call time -
    if either is missing, the tool returns ``{"hits": [], "error": "..."}``
    rather than raising, so the chat keeps moving.

    Each hit has ``score`` (cosine similarity in [0,1], higher is better),
    ``path`` (relative to the LVGL repo, e.g. ``docs/src/widgets/label.mdx``),
    ``heading`` (breadcrumb like "Animations > Common Components"),
    ``title``, and ``text`` (the chunk content, truncated to keep the
    chat server's context budget under control - override the cap via
    the ``LLAMA_DEBUGGER_RAG_TEXT_CHARS`` env var). When a particular
    hit looks promising and you need the full chunk, fetch the file
    with ``linux_read_file(path)`` rather than re-querying with bigger k.
    """
    _log("rag_search", query=query, k=k)
    cfg = _rag_config()
    k = max(1, min(int(k), 20))

    try:
        from systemd_mcp.vectordb.embed import EmbeddingClient
        from systemd_mcp.vectordb.store import VectorStore
    except ImportError as exc:
        return {
            "hits": [],
            "error": (
                f"vectordb extras not installed ({exc}); run `poetry sync` "
                f"to pick up numpy + sqlite-vec."
            ),
        }

    if not os.path.exists(cfg["db_path"]):
        return {
            "hits": [],
            "error": (
                f"vector DB not found at {cfg['db_path']}; build it first "
                f"with `poetry run llama_debugger_vectordb build` "
                f"(or set LLAMA_DEBUGGER_VECTORDB to point elsewhere)."
            ),
        }

    try:
        with VectorStore(cfg["db_path"]) as store:
            info = store.info()
            if info["chunk_count"] == 0:
                return {
                    "hits": [],
                    "error": (
                        f"vector DB at {cfg['db_path']} is empty; rebuild "
                        f"with `poetry run llama_debugger_vectordb build`."
                    ),
                }

            embed = EmbeddingClient(host=cfg["embed_host"], port=cfg["embed_port"])
            try:
                qvec = embed.embed_query(query)
            except Exception as exc:  # noqa: BLE001 - explicit soft-fail
                return {
                    "hits": [],
                    "error": (
                        f"embedding server at {embed.base_url} unreachable "
                        f"({type(exc).__name__}: {exc}); start it with "
                        f"`./start-llama-embedding-server.sh`."
                    ),
                }

            hits = store.search(qvec, k=k)
    except Exception as exc:  # noqa: BLE001 - bubble nothing into the chat
        return {"hits": [], "error": f"rag_search failed: {type(exc).__name__}: {exc}"}

    return {
        "hits": [
            {
                "score": round(h.score, 4),
                "path": h.path,
                "heading": h.heading,
                "title": h.title,
                "text": _trim_hit_text(h.text),
            }
            for h in hits
        ],
    }


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
