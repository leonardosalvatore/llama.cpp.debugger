#!/usr/bin/env bash
# Start llama-server with the bundled ROCm binary and GGUF model.
#
# Usage:
#   ./start-llama-server.sh [-p PORT] [-c CTX] [model.gguf]
#
# Flags / env:
#   -p, --port      <int>   server port (env: LLAMA_PORT, default 53425)
#   -c, --ctx-size  <int>   context window in tokens (env: LLAMA_CTX, default 32768)
#
# Examples:
#   ./start-llama-server.sh
#   ./start-llama-server.sh -p 53425 -c 65536
#   LLAMA_CTX=65536 ./start-llama-server.sh
#   ./start-llama-server.sh /path/to/model.gguf -p 53425

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../llama.cpp/llama-b8532-bin-ubuntu-rocm-7.2-x64/llama-b8532"
#DEFAULT_MODEL="${LLAMA_DIR}/Ministral-3-8B-Reasoning-2512-Q5_K_M.gguf"
#DEFAULT_MODEL="${LLAMA_DIR}/Ministral-3-8B-Reasoning-2512-Q8_0.gguf"
#DEFAULT_MODEL="${LLAMA_DIR}/Qwen3.5-9b-Sushi-Coder-RL.Q4_K_M.gguf"
DEFAULT_MODEL="${LLAMA_DIR}/qwen2.5-coder-7b-instruct-q8_0.gguf"
#DEFAULT_MODEL="${LLAMA_DIR}/ruvltra-claude-code-0.5b-q4_k_m.gguf"

PORT=""
MODEL=""
CTX=""

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -c|--ctx-size)
            CTX="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,14p' "$0"
            exit 0
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

# Allow env vars as fallback; finally fall back to baked-in defaults.
PORT="${PORT:-${LLAMA_PORT:-53425}}"
CTX="${CTX:-${LLAMA_CTX:-32768}}"
MODEL="${MODEL:-$DEFAULT_MODEL}"

if [ -z "$MODEL" ]; then
    echo "ERROR: no model specified." >&2
    echo "  Pass one as: $0 /path/to/model.gguf" >&2
    echo "  Or uncomment a DEFAULT_MODEL line in this script." >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi

cd "$LLAMA_DIR"

ln -s /opt/rocm/lib/libamdhip64.so.6 ./libamdhip64.so.7 2>/dev/null
ln -s /opt/rocm/lib/libhipblas.so.2 ./libhipblas.so.3 2>/dev/null
ln -s /opt/rocm/lib/librocblas.so.4 ./librocblas.so.5 2>/dev/null

export LD_LIBRARY_PATH=.:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

echo "Starting llama-server..."
echo "  Binary : ${LLAMA_DIR}/llama-server"
echo "  Model  : ${MODEL}"
echo "  Port   : ${PORT}"
echo "  Ctx    : ${CTX} tokens"
echo ""

# --jinja            : use the GGUF's embedded Jinja chat template. REQUIRED for
#                      reasoning / tool-calling models (Ministral-3-Reasoning,
#                      Qwen 2.5, DeepSeek, etc.) so the "thinking" channel and
#                      tool_calls round-trip through the correct template.
# -c $CTX            : context window in tokens. llama.cpp.debugger ships ~45
#                      tool schemas (~5-8k tokens just for tools) plus reasoning
#                      traces and tool-call history, so anything below 32k
#                      truncates mid-conversation. Override with the LLAMA_CTX
#                      env var if you want a different size:
#                          LLAMA_CTX=65536 ./start-llama-server.sh
#                      KV-cache cost is linear in context; for an 8B model
#                      32k ~= 4 GB, 64k ~= 8 GB on top of the weights.
# --reasoning-format : "deepseek" splits the thinking trace out of the visible
#                      content so the client sees only the final reply. Works
#                      for Ministral-3-Reasoning too (it uses a similar
#                      <think>...</think> convention). Drop this flag if you
#                      want the raw reasoning in the response.
# exec so the script's pid becomes the llama-server pid itself. Without this,
# the bots launcher's g_child_pid points at /bin/bash, and on SIGTERM bash
# dies while llama-server gets reparented to init and keeps running. Sharing
# a single pid makes the launcher's kill() reach the actual server.
exec env HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    ./llama-server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -ngl 99 \
    -c "$CTX" \
    --jinja \
    --reasoning-format deepseek
