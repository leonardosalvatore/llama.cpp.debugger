#!/usr/bin/env bash
# Thin launcher for the chat agent. Forwards every flag and the optional
# positional prompt to ``llama_debugger_mcp_cli``.
#
# Common usages:
#   ./run_mcp_cli.sh                                # interactive chat
#   ./run_mcp_cli.sh "list the running services"    # interactive, but seed
#                                                   # the first turn with this
#                                                   # prompt (still drops back
#                                                   # into the prompt afterwards)
#   ./run_mcp_cli.sh --single "uptime?"             # one-shot, exit when done
#   ./run_mcp_cli.sh --split-screen                 # full-screen TUI mode
#   ./run_mcp_cli.sh --no-tools "hello"             # chat only, no tool calls
#
# Anything not consumed here is passed through verbatim, so flags and
# positional args mix freely:
#   ./run_mcp_cli.sh --tail-bg-stdout --single "compile and run /tmp/x.c"

set -euo pipefail

poetry sync

# `--tail-bg-stdout` is on by default because the most common interactive
# workflow is "launch a long-running program on the SUT and watch its
# log". Drop it (or pass `--split-screen`) on the command line to opt out.
exec poetry run llama_debugger_mcp_cli --tail-bg-stdout "$@"
