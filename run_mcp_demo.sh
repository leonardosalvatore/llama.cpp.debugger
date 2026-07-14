#!/usr/bin/env bash
# Demo launcher: profiles the LVGL GLFW simulator on the SUT and renders a
# CPU function heatmap PNG, driving the perf_* MCP tools end-to-end through
# the chat agent (perf_record -> perf_heatmap).
#
# It just seeds run_mcp_cli.sh with a canned prompt. Prerequisites:
#   * the QEMU SUT is up (GUI=1 GUI_ACCEL=1 ./run_linux_in_qemu.sh) with
#     lvglsim built at $LVGLSIM_BIN, and
#   * the chat llama-server is up (./start-llama-server.sh).
#
# Tunables (override via env):
#   LVGLSIM_BIN    path to lvglsim on the SUT
#   LVGLSIM_ARGS   arguments passed to lvglsim (backend selection etc.)
#   PERF_DURATION  seconds to record
#   HEATMAP_PNG    where the heatmap PNG is written on the HOST
#
# Any extra CLI flags are forwarded verbatim, so you can watch the SSH wire
# while it works:
#   ./run_mcp_demo.sh --split-screen

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LVGLSIM_BIN="${LVGLSIM_BIN:-/home/debian/Dev/lv_port_linux/build/bin/lvglsim}"
LVGLSIM_ARGS="${LVGLSIM_ARGS:--b GLFW}"
PERF_DURATION="${PERF_DURATION:-12}"
HEATMAP_PNG="${HEATMAP_PNG:-/tmp/lvglsim_heatmap.png}"

DEMO_PROMPT="Profile the LVGL binary ${LVGLSIM_BIN} (arguments: ${LVGLSIM_ARGS}) and show me a CPU function heatmap. Do EXACTLY these steps in order, one tool call each:
1. Call linux_run_command with command: sudo cp /var/run/lightdm/root/:0 /tmp/xauth.debian && sudo chown debian:debian /tmp/xauth.debian   (this is the X cookie the GLFW window needs).
2. Call configuration_setRemoteEnv with name=XAUTHORITY and value=/tmp/xauth.debian   (do NOT put XAUTHORITY= inside any command string; DISPLAY=:0 is already set for you).
3. Call perf_record with command=${LVGLSIM_BIN} ${LVGLSIM_ARGS} (the plain binary and its args, no env prefix), duration=${PERF_DURATION}.
4. Call perf_heatmap with data_file=/tmp/llamadbg_perf.data and output_png=${HEATMAP_PNG}.
5. Report the PNG path and the top 5 hottest functions from the perf_heatmap result. Do not run the binary any other way and do not write helper scripts."

exec "$SCRIPT_DIR/run_mcp_cli.sh" --single "$@" "$DEMO_PROMPT"
