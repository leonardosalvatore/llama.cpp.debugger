#!/usr/bin/env bash
set -euo pipefail

poetry sync
#poetry run llama_debugger_mcp_cli --split-screen
poetry run llama_debugger_mcp_cli --split-screen
