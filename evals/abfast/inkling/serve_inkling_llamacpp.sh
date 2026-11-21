#!/usr/bin/env bash
# One-card llama.cpp SYCL serve of Inkling-2-Mini IQ4_NL for abfast. Holds the card lease for the life of the server.
# Usage (DO NOT run without GPU approval):  evals/abfast/inkling/serve_inkling_llamacpp.sh      # card 1, port 18160
#   CARD=0 PORT=18160 CTX=262144 NP=2 evals/abfast/inkling/serve_inkling_llamacpp.sh
# Stop: Ctrl-C, or: docker stop inkling-llamacpp
# 2026-11-21 run notes: -kvu is required (split per-slot KV made C2 decode collapse to ~1.5 tok/s/stream at 262144);
#   UB 64 so the eager-KQ compute buffer fits next to the 262144 unified pool (512 wanted 20 GB, 128 wanted 9.5 GB);
#   -cram 0 because the host prompt cache stalled each new request by ~2-4 min. Needs the SYCL strided-SCALE patch
#   (/mnt/vm_8tb/b70/build/llamacpp-inkling/sycl-scale-strided.patch), else 70 SCALE ops per token run on the CPU.
# Env: CARD (default 1), PORT (18160), CTX (262144 = 2 slots sharing one unified KV pool), NP (2), UB (64),
#      GGML_SYCL_ENABLE_OPT (unset; set 0 only if output is garbage, see issue 29107 / PR 30604 already merged),
#      GGML_SYCL_ENABLE_GRAPH (unset = backend default 0; 1 = SYCL graph replay, +45% tg in llama-bench 2026-11-21),
#      EXTRA (extra llama-server args), KEYFILE (api key file; default the abfast dd key)
set -euo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
CARD=${CARD:-1}; PORT=${PORT:-18160}; CTX=${CTX:-262144}; NP=${NP:-2}; UB=${UB:-64}
KEYFILE=${KEYFILE:-/mnt/vm_8tb/b70/secrets/dd_api_key}
case "$PORT" in 18080|18124) echo "refusing reserved port $PORT" >&2; exit 2;; esac
if [ -z "${B70_INKLING_IN_LEASE:-}" ]; then
  export B70_INKLING_IN_LEASE=1 CARD PORT CTX NP UB KEYFILE
  [ -n "${GGML_SYCL_ENABLE_OPT:-}" ] && export GGML_SYCL_ENABLE_OPT
  [ -n "${GGML_SYCL_ENABLE_GRAPH:-}" ] && export GGML_SYCL_ENABLE_GRAPH
  exec "$R/bin/gpu-run" --card "$CARD" bash "$0" "$@"
fi
M=/mnt/cache/b70/models/inkling-2-mini
SRC=/mnt/vm_8tb/b70/build/llamacpp-inkling/llama.cpp
GGUF=Inkling-2-Mini-IQ4_NL.gguf
[ -s "$M/$GGUF" ] || { echo "missing $M/$GGUF" >&2; exit 1; }
[ -x "$SRC/build-sycl/bin/llama-server" ] || { echo "missing build: $SRC/build-sycl/bin/llama-server" >&2; exit 1; }
KEYARG=""; KEYMNT=""
if [ -r "$KEYFILE" ]; then KEYMNT="-v $KEYFILE:/key:ro"; KEYARG="--api-key-file /key"; fi
OPT=""; [ -n "${GGML_SYCL_ENABLE_OPT:-}" ] && OPT="-e GGML_SYCL_ENABLE_OPT=$GGML_SYCL_ENABLE_OPT"
[ -n "${GGML_SYCL_ENABLE_GRAPH:-}" ] && OPT="$OPT -e GGML_SYCL_ENABLE_GRAPH=$GGML_SYCL_ENABLE_GRAPH"
docker rm -f inkling-llamacpp >/dev/null 2>&1 || true
# -fa off: Inkling's relative-position bias is passed to attention as a KQ bias; llama.cpp disables flash attention
#   for kq_b (build_attn_mha: use_flash_attn = flash_attn && kq_b == nullptr), so request the eager path explicitly.
# Text only: no --mmproj (the IQ4_NL GGUF has no vision tower; mmproj-Inkling-2-Mini-*.gguf not downloaded).
exec docker run --rm --name inkling-llamacpp --memory 90g --network host --ipc=host \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -v "$SRC":/src:ro -v "$M":/m:ro $KEYMNT \
  -e ZE_AFFINITY_MASK="$CARD" -e ONEAPI_DEVICE_SELECTOR=level_zero:0 $OPT \
  --entrypoint bash b70/flashnext-llamacpp-runtime:neo26.22 -c \
  "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH=/src/build-sycl/bin:\$LD_LIBRARY_PATH; \
   exec /src/build-sycl/bin/llama-server -m /m/$GGUF -ngl 99 -fa off -sm none -c $CTX -np $NP -kvu -ub $UB -cram 0 \
   --host 0.0.0.0 --port $PORT --jinja --alias inkling-2-mini-iq4nl-llamacpp $KEYARG ${EXTRA:-}"
