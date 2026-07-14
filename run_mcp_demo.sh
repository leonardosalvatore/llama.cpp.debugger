#!/usr/bin/env bash
# Demo launcher: profiles the LVGL GLFW simulator on the SUT and renders a
# CPU function heatmap PNG, driving the perf_* MCP tools end-to-end through
# the chat agent (perf_record -> perf_heatmap).
#
# It does the fiddly, model-unfriendly setup ITSELF (deterministically) and
# then seeds run_mcp_cli.sh with a minimal prompt, so even a small model only
# has to make two tool calls and can't derail on SSH creds or X cookies.
#
# Prerequisites:
#   * the QEMU SUT is up (GUI=1 GUI_ACCEL=1 ./run_linux_in_qemu.sh) with
#     lvglsim built at $LVGLSIM_BIN, and
#   * the chat llama-server is up (./start-llama-server.sh).
#
# Tunables (override via env):
#   LVGLSIM_BIN    path to lvglsim on the SUT
#   LVGLSIM_ARGS   arguments passed to lvglsim (backend selection etc.)
#   PERF_DURATION  seconds to record
#   HEATMAP_PNG    where the heatmap PNG is written on the HOST
#   SUT_XAUTH      path on the SUT for the copied X cookie
#
# Any extra *flags* are forwarded verbatim, so you can watch the SSH wire
# while it works:
#   ./run_mcp_demo.sh --split-screen

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LVGLSIM_BIN="${LVGLSIM_BIN:-/home/debian/Dev/lv_port_linux/build/bin/lvglsim}"
LVGLSIM_ARGS="${LVGLSIM_ARGS:--b GLFW}"
PERF_DURATION="${PERF_DURATION:-12}"
HEATMAP_PNG="${HEATMAP_PNG:-/tmp/lvglsim_heatmap.png}"
SUT_XAUTH="${SUT_XAUTH:-/tmp/xauth.debian}"
SUT_PERF_DATA="${SUT_PERF_DATA:-/tmp/llamadbg_perf.data}"
# OPEN_HOTSPOT=0 to skip launching the Hotspot GUI at the end.
OPEN_HOTSPOT="${OPEN_HOTSPOT:-1}"

# Ensure the venv matches the lock file (run_mcp_cli.sh also does this; it is
# idempotent, and we need the deps below for the pre-flight X setup).
poetry sync

# --- Deterministic X setup ---------------------------------------------------
# lvglsim -b GLFW opens an X11 window, which needs a valid MIT-MAGIC-COOKIE.
# We do NOT leave this to the model: small models reliably derail here,
# confusing configuration_setRemoteEnv with configuration_setTargetHost and
# even guessing SSH passwords. Instead we copy the live display-manager
# cookie to a debian-readable path over the existing SSH channel, then hand
# XAUTHORITY to the agent via LLAMA_DEBUGGER_REMOTE_ENV (which the server
# exports on every SUT command). Non-fatal: a desktop that is already logged
# in has a valid ~/.Xauthority regardless.
poetry run python - "$SUT_XAUTH" <<'PY' || true
import sys
from systemd_mcp import server as srv
dst = sys.argv[1]
cmd = (
    f"sudo cp /var/run/lightdm/root/:0 {dst} 2>/dev/null "
    f"&& sudo chown debian:debian {dst} && echo XAUTH_SETUP_OK "
    f"|| echo XAUTH_SETUP_SKIPPED (using existing ~/.Xauthority)"
)
print(srv._run_ssh_cmd(cmd).strip())
PY

export LLAMA_DEBUGGER_REMOTE_ENV="XAUTHORITY=${SUT_XAUTH}"

DEMO_PROMPT="Profile the LVGL binary ${LVGLSIM_BIN} (arguments: ${LVGLSIM_ARGS}) and show me a CPU function heatmap. The display and X authority are ALREADY configured for you (DISPLAY and XAUTHORITY are set on every command), so do NOT copy cookies, do NOT set environment variables, and do NOT call any configuration_* tool. Do EXACTLY two tool calls, in order: (1) perf_record with command=${LVGLSIM_BIN} ${LVGLSIM_ARGS} and duration=${PERF_DURATION}; (2) perf_heatmap with data_file=/tmp/llamadbg_perf.data and output_png=${HEATMAP_PNG}. Then report the PNG path and the top 5 hottest functions from the perf_heatmap result. Do not run the binary any other way and do not write helper scripts."

# This script supplies its own prompt (the single positional arg the CLI
# accepts), so only forward *flags* from "$@". A stray positional (e.g. a
# trailing ".") would otherwise collide with the prompt and make argparse
# bail with a cryptic "unrecognized arguments" dump - warn and drop it.
FLAGS=()
for arg in "$@"; do
  if [[ "$arg" == -* ]]; then
    FLAGS+=("$arg")
  else
    echo "run_mcp_demo.sh: ignoring unexpected argument '$arg' (this demo" \
         "supplies its own prompt; pass only flags like --split-screen)." >&2
  fi
done

"$SCRIPT_DIR/run_mcp_cli.sh" --single "${FLAGS[@]+"${FLAGS[@]}"}" "$DEMO_PROMPT"

# Open the rendered heatmap in the host's default image viewer once the chat
# finishes. Guarded so a missing file or no xdg-open doesn't error out.
if command -v xdg-open >/dev/null 2>&1 && [ -f "$HEATMAP_PNG" ]; then
  xdg-open "$HEATMAP_PNG" >/dev/null 2>&1 || true
fi

# Also pull the raw perf.data off the SUT and open it in Hotspot (interactive
# flame graph / caller-callee), with build-id symbols. Set OPEN_HOTSPOT=0 to
# skip. Non-fatal if hotspot isn't installed - the tool soft-fails.
if [ "$OPEN_HOTSPOT" = "1" ]; then
  poetry run python - "$SUT_PERF_DATA" <<'PY' || true
import sys, json
from systemd_mcp import server as srv
print(json.dumps(srv.perf_open_hotspot.fn(data_file=sys.argv[1]), indent=2))
PY
fi
