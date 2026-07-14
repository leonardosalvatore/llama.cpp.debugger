#!/usr/bin/env bash
# Start a SECOND llama-server instance dedicated to embeddings.
#
# This is the companion to start-llama-server.sh: that one runs the chat /
# reasoning model on port 53425 and is tuned for tool-calling. This one runs
# a small embedding-only model on a separate port (default 53426) so the
# llama_debugger_vectordb CLI can build / query the LVGL docs vector store
# without disturbing the chat server.
#
# Why a second instance?
#   * llama-server's --embeddings flag changes how the model is invoked
#     (no token generation, pooled hidden states only).
#   * Embedding-class GGUFs (nomic-embed-text, bge-*, e5-mistral) are tiny
#     compared to a chat model and are tuned specifically for retrieval.
#   * A chat model's mean-pooled hidden states make poor embeddings.
#
# Usage:
#   ./start-llama-embedding-server.sh [-p PORT] [-c CTX] [-g GPU] [model.gguf]
#
# Flags / env:
#   -p, --port      <int>   server port (env: LLAMA_EMB_PORT, default 53426)
#   -c, --ctx-size  <int>   context window in tokens (env: LLAMA_EMB_CTX,
#                           default 8192 - matches nomic-embed-text-v1.5)
#   -g, --gpu       <id>    HIP device index (env: LLAMA_EMB_GPU, default 1).
#                           Default of 1 targets the integrated GPU on a
#                           dual-GPU Legion-style rig so the embedding model
#                           (~140 MiB on disk + ~30 MiB compute buf for
#                           nomic-embed-text-v1.5.Q8_0) leaves the discrete
#                           card free for the chat model. Set to "0" to share
#                           the discrete GPU, or to any HIP_VISIBLE_DEVICES
#                           value (e.g. "0,1").
#
# Examples:
#   ./start-llama-embedding-server.sh
#   ./start-llama-embedding-server.sh -p 53426 -c 8192
#   ./start-llama-embedding-server.sh -g 0           # share the big GPU
#   LLAMA_EMB_GPU=0 ./start-llama-embedding-server.sh
#   LLAMA_EMB_CTX=2048 ./start-llama-embedding-server.sh
#   ./start-llama-embedding-server.sh /path/to/bge-small-en-v1.5.Q8_0.gguf

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${SCRIPT_DIR}/../llama.cpp/llama-b9940-bin-ubuntu-rocm-7.2-x64/llama-b9940"

# Recommended embedding model for English MD/MDX docs (LVGL): nomic-embed-text
# v1.5 in Q8_0. 768-dim, 8k context, ~150 MB, MTEB-strong.
# Other tested options (uncomment to switch):
#   - bge-small-en-v1.5.Q8_0.gguf  : 384-dim, ~35 MB, faster/smaller, English only
#   - bge-m3.Q8_0.gguf             : 1024-dim, multilingual + dense+sparse
#   - e5-mistral-7b-instruct.Q4_K_M: heavy but state-of-the-art
DEFAULT_MODEL="${LLAMA_DIR}/nomic-embed-text-v1.5.Q8_0.gguf"
#DEFAULT_MODEL="${LLAMA_DIR}/bge-small-en-v1.5.Q8_0.gguf"
#DEFAULT_MODEL="${LLAMA_DIR}/bge-m3.Q8_0.gguf"

PORT=""
MODEL=""
CTX=""
GPU=""

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
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,38p' "$0"
            exit 0
            ;;
        *)
            MODEL="$1"
            shift
            ;;
    esac
done

PORT="${PORT:-${LLAMA_EMB_PORT:-53426}}"
CTX="${CTX:-${LLAMA_EMB_CTX:-8192}}"
GPU="${GPU:-${LLAMA_EMB_GPU:-1}}"
MODEL="${MODEL:-$DEFAULT_MODEL}"

if [ -z "$MODEL" ]; then
    echo "ERROR: no embedding model specified." >&2
    echo "  Pass one as: $0 /path/to/model.gguf" >&2
    echo "  Or uncomment a DEFAULT_MODEL line in this script." >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: embedding model file not found: $MODEL" >&2
    echo "  Hint: download a GGUF, e.g." >&2
    echo "    huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF \\" >&2
    echo "      nomic-embed-text-v1.5.Q8_0.gguf --local-dir \"$LLAMA_DIR\"" >&2
    exit 1
fi

cd "$LLAMA_DIR"

# Same ROCm shim as start-llama-server.sh - the prebuilt llama-server links
# against ROCm 7 .so names but Debian/Ubuntu typically ship ROCm 6.
ln -s /opt/rocm/lib/libamdhip64.so.6 ./libamdhip64.so.7 2>/dev/null
ln -s /opt/rocm/lib/libhipblas.so.2 ./libhipblas.so.3 2>/dev/null
ln -s /opt/rocm/lib/librocblas.so.4 ./librocblas.so.5 2>/dev/null

# rocBLAS Tensile kernels: rocBLAS resolves ./rocblas/library/TensileLibrary*.dat
# relative to the *current working directory*, not its own .so location. Since
# we cd into LLAMA_DIR (which has no rocblas/ subdir), this fails the moment
# the model dispatches to a Tensile-backed GEMM with:
#     rocBLAS error: Cannot read ./rocblas/library/TensileLibrary.dat
#     filesystem error: ... No such file or directory [./rocblas/library]
# The chat model only triggers this for specific shapes, but BERT-style
# embedding models (nomic-embed-text, bge-*) hit it on the very first
# request. Two fixes belt-and-suspenders:
#   1. Tell rocBLAS exactly where its Tensile library lives.
#   2. Symlink ./rocblas -> /opt/rocm/lib/rocblas for older rocBLAS versions
#      that ignore the env var.
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library
ln -s /opt/rocm/lib/rocblas ./rocblas 2>/dev/null

export LD_LIBRARY_PATH=.:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH

echo "Starting llama-server (EMBEDDINGS)..."
echo "  Binary  : ${LLAMA_DIR}/llama-server"
echo "  Model   : ${MODEL}"
echo "  Port    : ${PORT}"
echo "  Ctx     : ${CTX} tokens"
echo "  Pooling : mean"
echo "  GPU     : HIP_VISIBLE_DEVICES=${GPU}"
echo ""

# --embeddings        : switch the server into embedding mode. /v1/chat is
#                       still served but generation is meaningless; what we
#                       want is /v1/embeddings and /embedding (native).
# --pooling mean      : REQUIRED for /v1/embeddings (the OpenAI-compatible
#                       endpoint refuses --pooling none). 'mean' matches what
#                       nomic-embed-text and bge-* expect at training time.
#                       Use 'cls' for some BERT-style models, 'last' for
#                       decoder-style (e5-mistral).
# -c $CTX             : embedding ctx. nomic-embed-text-v1.5 is trained with
#                       2k; longer inputs get truncated.
# -b 2048 -ub 2048    : physical batch / micro-batch size. In embedding mode
#                       llama-server enforces n_batch == n_ubatch (it logs
#                       'setting n_batch = n_ubatch = 512 to avoid assertion
#                       failure' and silently caps n_batch to 512 if you only
#                       pass --batch-size). The /v1/embeddings handler then
#                       returns HTTP 500 on any input larger than the ubatch:
#                           "input (543 tokens) is too large to process.
#                            increase the physical batch size"
#                       Bumping both to 2048 lets a single chunk be up to the
#                       model's training context. Cost is ~30 MiB extra
#                       compute buffer on the iGPU - negligible compared to
#                       its 15 GiB GTT pool.
exec env HIP_VISIBLE_DEVICES="$GPU" HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    ./llama-server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    -ngl 99 \
    -c "$CTX" \
    -b 2048 \
    -ub 2048 \
    --embeddings \
    --pooling mean
