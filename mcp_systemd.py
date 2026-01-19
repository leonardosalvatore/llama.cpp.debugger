from fastmcp import FastMCP
import paramiko # For SSH into the VM

mcp = FastMCP("Linux-SRE-Agent")

def run_ssh_cmd(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('127.0.0.1', port=2222, username='root', password='password')
    stdin, stdout, stderr = ssh.exec_command(cmd)
    return stdout.read().decode()

@mcp.tool()
def get_service_status(service_name: str):
    """Checks the systemd status of a service."""
    return run_ssh_cmd(f"systemctl status {service_name}")

@mcp.tool()
def read_journal_logs(service_name: str, lines: int = 20):
    """Fetches the latest logs for a specific service."""
    return run_ssh_cmd(f"journalctl -u {service_name} -n {lines} --no-pager")

@mcp.tool()
def restart_service(service_name: str):
    """Attempts to restart a systemd service."""
    run_ssh_cmd(f"systemctl restart {service_name}")
    return f"Restart command sent for {service_name}"
