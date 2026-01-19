"""Command line interface for interacting with the systemd MCP server via Ollama."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List

try:  # Ollama 0.1.0+ exports ResponseError, earlier versions keep it internal
    from ollama import Client, ResponseError
except ImportError:  # pragma: no cover - fallback for older ollama packages
    from ollama import Client  # type: ignore
    from ollama._types import ResponseError  # type: ignore

from systemd_mcp import server


DEFAULT_SYSTEM_PROMPT = (
    "You are a Linux SRE assistant. Use the available tools to manage systemd services "
    "whenever the user asks for status checks, log reviews, or restarts."
)

# The CLI imports the server tools so we obey the same SSH execution path.
TOOL_REGISTRY = {
    "get_service_status": server.get_service_status,
    "read_journal_logs": server.read_journal_logs,
    "restart_service": server.restart_service,
}

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
            "description": "Read the latest journal lines for a systemd service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Name of the systemd unit.",
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of log lines to return (default 20).",
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                "required": ["service_name"],
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
]


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the systemd MCP server through an Ollama-hosted llama3 model."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional prompt to send immediately before entering interactive mode.",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="llama3",
        help="Ollama model name to use (default: llama3).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override Ollama host URL (defaults to environment configuration).",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Custom system prompt to steer the assistant.",
    )
    parser.add_argument(
        "--max-tool-steps",
        type=int,
        default=8,
        help="Maximum chained tool invocations allowed per user turn (default: 8).",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run a single exchange (non-interactive) and exit immediately.",
    )
    return parser.parse_args(list(argv))


def _decode_arguments(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, str) and raw_args.strip():
        return json.loads(raw_args)
    if isinstance(raw_args, dict):
        return raw_args
    return {}


def _invoke_tool(call: Dict[str, Any]) -> str:
    name = call.get("function", {}).get("name")
    if name not in TOOL_REGISTRY:
        return f"Unsupported tool requested: {name}"
    arguments = _decode_arguments(call.get("function", {}).get("arguments"))
    try:
        result = TOOL_REGISTRY[name](**arguments)
    except Exception as exc:  # noqa: BLE001 - we surface tool failures to the model
        return f"Tool execution failed: {exc}"
    return result


def _chat_once(
    client: Client,
    messages: List[Dict[str, Any]],
    model: str,
    max_tool_steps: int,
    tool_spec: List[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    steps = 0
    while True:
        response = client.chat(model=model, messages=messages, tools=tool_spec)
        message = response.get("message", {})
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_spec or not tool_calls:
            return message
        for call in tool_calls:
            steps += 1
            if steps > max_tool_steps:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "Tool depth limit reached; unable to continue automatically.",
                    }
                )
                return messages[-1]
            tool_output = _invoke_tool(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": call.get("function", {}).get("name"),
                    "content": tool_output,
                }
            )


def _print_assistant_reply(message: Dict[str, Any]) -> None:
    content = message.get("content", "").strip()
    if content:
        print(f"assistant> {content}")


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    client = Client(host=args.host) if args.host else Client()

    messages: List[Dict[str, Any]] = [{"role": "system", "content": args.system}]
    interactive = not args.single or not args.prompt
    tool_support = True

    def chat_with_fallback() -> Dict[str, Any]:
        nonlocal tool_support
        try:
            tool_spec = TOOL_SPEC if tool_support else None
            return _chat_once(client, messages, args.model, args.max_tool_steps, tool_spec)
        except ResponseError as exc:  # model might not support tool calls
            message = str(exc).lower()
            if tool_support and "does not support tools" in message:
                print(
                    "warning: model lacks tool support; falling back to plain chat.",
                    file=sys.stderr,
                )
                tool_support = False
                return _chat_once(client, messages, args.model, args.max_tool_steps, None)
            raise

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        reply = chat_with_fallback()
        _print_assistant_reply(reply)
        if not interactive:
            return

    while True:
        try:
            user_input = input("you> ")
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
        reply = chat_with_fallback()
        _print_assistant_reply(reply)


if __name__ == "__main__":
    main()
