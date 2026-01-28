"""Command line interface for chatting with an Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, Iterable, List

import paramiko
from ollama import chat


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's concisely. "
    "The user will ask you to work on a SUT (system under test, is a Debian linux OS), you are interfacing to it with tools that will let access to systemd, journald, "
    "and standard Linux utilities. Any prompt from the user should be solved using the available tools."
)


TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "Retrieve the detailed status output for a systemd service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the systemd unit (for example, ssh or nginx).",
                    }
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_journal_logs",
            "description": "Fetch journal lines for a service, or all logs if service_name is omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the systemd unit (optional).",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of log lines to return (default 100).",
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart a systemd service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the systemd unit to restart.",
                    }
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_uptime",
            "description": "Retrieve the current system uptime string.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_service",
            "description": "List all systemd services on the target host. A service can be active, inactive, or failed.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ssh_command",
            "description": "Execute an arbitrary command over SSH and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute on the remote host.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def _run_ssh_cmd(cmd: str) -> str:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("127.0.0.1", port=2222, username="debian", password="debian")
    try:
        _stdin, stdout, _stderr = ssh.exec_command(cmd)
        return stdout.read().decode()
    finally:
        ssh.close()


def get_service_status(service_name: str) -> str:
    return _run_ssh_cmd(f"systemctl status --no-pager {service_name}")


def read_journal_logs(service_name: str | None = None, lines: int = 100) -> str:
    if service_name:
        return _run_ssh_cmd(f"journalctl -u {service_name} -n {lines} --no-pager")
    return _run_ssh_cmd(f"journalctl -n {lines} --no-pager")


def restart_service(service_name: str) -> str:
    return _run_ssh_cmd(f"systemctl restart {service_name}")


def get_uptime() -> str:
    return _run_ssh_cmd("uptime -p")


def get_all_service() -> str:
    return _run_ssh_cmd("systemctl list-units --type=service --all --no-pager")


def run_ssh_command(command: str) -> str:
    return _run_ssh_cmd(command)


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with an Ollama model.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional prompt to send immediately before entering interactive mode.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="qwen3",
        help="Ollama model name to use (default: qwen3).",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt to steer the assistant.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single exchange (non-interactive) and exit immediately.",
    )
    return parser.parse_args(list(argv))


def _chat_stream(
    messages: List[Dict[str, Any]],
    model: str,
    tools: List[Any] | None,
) -> Dict[str, Any]:
    stream = chat(
        model=model,
        messages=messages,
        tools=tools,
        stream=True,
        think=True,
    )

    thinking = ""
    content = ""
    tool_calls: List[Any] = []
    done_thinking = False

    for chunk in stream:
        message = chunk.message
        if message.thinking:
            thinking += message.thinking
            print(message.thinking, end="", flush=True)
        if message.content:
            if not done_thinking:
                done_thinking = True
                print("\n")
            content += message.content
            print(message.content, end="", flush=True)
        if message.tool_calls:
            tool_calls.extend(message.tool_calls)
            print(message.tool_calls)

    if thinking or content or tool_calls:
        messages.append(
            {
                "role": "assistant",
                "thinking": thinking,
                "content": content,
                "tool_calls": tool_calls,
            }
        )

    return {
        "role": "assistant",
        "thinking": thinking,
        "content": content,
        "tool_calls": tool_calls,
    }


def _decode_arguments(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        return json.loads(raw_args)
    return {}


def _extract_call_name_args(call: Any) -> tuple[str | None, Dict[str, Any]]:
    if hasattr(call, "function"):
        func = call.function
        name = getattr(func, "name", None)
        args = getattr(func, "arguments", {})
        return name, _decode_arguments(args)
    if isinstance(call, dict):
        func = call.get("function", {})
        name = func.get("name")
        args = func.get("arguments", {})
        return name, _decode_arguments(args)
    return None, {}


def _invoke_tool(call: Any, registry: Dict[str, Callable[..., str]]) -> str:
    name, arguments = _extract_call_name_args(call)
    if not name or name not in registry:
        return f"Unknown tool: {name}"
    try:
        return registry[name](**arguments)
    except Exception as exc:  # noqa: BLE001
        return f"Tool execution failed: {exc}"


def _run_turn(
    messages: List[Dict[str, Any]],
    model: str,
    tools: List[Any] | None,
    registry: Dict[str, Callable[..., str]],
) -> None:
    while True:
        reply = _chat_stream(messages, model, tools)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:
            if not reply.get("content"):
                print()
            break
        for call in tool_calls:
            name, _args = _extract_call_name_args(call)
            result = _invoke_tool(call, registry)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name or "unknown",
                    "content": result,
                }
            )


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": args.system}]
    interactive = not args.single
    tools: List[Any] = TOOL_SPEC
    registry: Dict[str, Callable[..., str]] = {
        "get_service_status": get_service_status,
        "read_journal_logs": read_journal_logs,
        "restart_service": restart_service,
        "get_uptime": get_uptime,
        "get_all_service": get_all_service,
        "run_ssh_command": run_ssh_command,
    }

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        _run_turn(messages, args.model, tools, registry)
        if not interactive:
            return

    while True:
        try:
            user_input = input("\nyou> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit", "q"}:
            break
        messages.append({"role": "user", "content": stripped})
        _run_turn(messages, args.model, tools, registry)


if __name__ == "__main__":
    main()
