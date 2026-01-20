import sys

from fastmcp import FastMCP
import paramiko

mcp = FastMCP("Linux-SRE-Agent")


def _run_ssh_cmd(cmd: str) -> str:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("127.0.0.1", port=2222, username="debian", password="debian")
    stdin, stdout, _ = ssh.exec_command(cmd)
    return stdout.read().decode()


@mcp.tool()
def get_service_status(service_name: str):
    """Return systemd status for the given service."""
    print(f"[systemd_mcp] get_service_status({service_name})", file=sys.stderr, flush=True)
    return _run_ssh_cmd(f"systemctl status {service_name}")


@mcp.tool()
def read_journal_logs(service_name: str | None = None, lines: int = 20):
    """Fetch journal lines for a service, or for all if service_name is omitted."""
    print(
        f"[systemd_mcp] read_journal_logs({service_name}, lines={lines})",
        file=sys.stderr,
        flush=True,
    )
    if service_name:
        return _run_ssh_cmd(f"journalctl -u {service_name} -n {lines} --no-pager")
    return _run_ssh_cmd(f"journalctl -n {lines} --no-pager")


@mcp.tool()
def restart_service(service_name: str):
    """Restart the given systemd service."""
    print(f"[systemd_mcp] restart_service({service_name})", file=sys.stderr, flush=True)
    _run_ssh_cmd(f"sudo systemctl restart {service_name}")
    return f"Restart command sent for {service_name}"


@mcp.tool()
def get_uptime():
    """Return the system uptime string."""
    print("[systemd_mcp] get_uptime()", file=sys.stderr, flush=True)
    return _run_ssh_cmd("uptime")


@mcp.tool()
def get_all_service():
    """List all systemd services."""
    print("[systemd_mcp] get_all_service()", file=sys.stderr, flush=True)
    return _run_ssh_cmd("systemctl list-units --type=service --all --no-pager")


@mcp.tool()
def run_ssh_command(command: str):
    """Execute an arbitrary command over SSH."""
    print(f"[systemd_mcp] run_ssh_command(command={command!r})", file=sys.stderr, flush=True)
    return _run_ssh_cmd(command)


def main():
    """Start the FastMCP server."""
    print(_run_ssh_cmd("echo 'Linux-SRE-Agent started'"), file=sys.stderr, flush=True)

    mcp.run()
