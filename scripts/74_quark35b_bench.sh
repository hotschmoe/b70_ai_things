#!/usr/bin/env bash
# Serve nameistoken Qwen3.6-35B-A3B Quark W8A8 INT8 (int8 MoE) on intel/llm-scaler-vllm,
# following steveseguin's recipe, adapted to OUR 2x B70 (TP=2). Two non-trivial deltas vs
# his localmaxxing snippet, both REQUIRED on our box (see docs/kernel/20 + JOURNAL 06-22):
#   1) Device exposure: his env double-pins ONEAPI_DEVICE_SELECTOR + ZE_AFFINITY_MASK (his
#      4-card values). On our box that double-pin makes the model-inspect subprocess abort
#      with SYCL "No device of requested type available". Our proven TP>1 path exposes both
#      cards with NO pin -> DEVSEL/AFFMASK default empty here.
#   2) int8 MoE kernel: the 0.14.1 image's quark_moe.py only wires fp8 MoE -> raises
#      "Unsupported FusedMoe scheme" on this int8 ckpt. We mount a patched quark_moe.py
#      (contrib/llm_scaler_quark_int8_moe) that adds QuarkW8A8Int8MoEMethod.
# Route every GPU touch via gpu-run.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70
NAME=vllm_quark35b
IMG="${IMG:-intel/llm-scaler-vllm:0.14.0-b8.3.1}"
CKPT="${CKPT:-$ROOT/models/Qwen3.6-35B-A3B-Quark-W8A8-INT8}"
TP="${TP:-2}"
PORT="${PORT:-8000}"
# Served name encodes the REAL quant (steve served int8 as "...-fp8" -- a mislabel; CLAUDE.md [!] rule).
SERVED="${SERVED:-qwen36-35b-a3b-quark-w8a8-int8}"
# Device pins: EMPTY = expose both cards (our proven multi-card path). Set to e.g. "level_zero:0,1"
# and "0,1" to reproduce steve's pin (known to abort inspection on this host).
DEVSEL="${DEVSEL:-}"
AFFMASK="${AFFMASK:-}"
IFACE="${IFACE:-lo}"           # his eth1 was his host default-route iface; in-container -> loopback
PATCH="${PATCH:-$ROOT/patches/quark_moe.py}"          # int8 MoE method (experts)
PATCH_LIN="${PATCH_LIN:-$ROOT/patches/quark.py}"      # int8 dequant linear scheme (attn/shared_expert)
QDIR=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/quark
QUARK_MOE_DST=$QDIR/quark_moe.py
QUARK_DST=$QDIR/quark.py
LOGF="$ROOT/results/quark35b_tp${TP}_serve.log"
mkdir -p "$ROOT/results"

if [ ! -f "$PATCH" ];     then echo "[!] missing patched quark_moe.py at $PATCH"; exit 2; fi
if [ ! -f "$PATCH_LIN" ]; then echo "[!] missing patched quark.py at $PATCH_LIN"; exit 2; fi

docker rm -f "$NAME" 2>/dev/null || true
echo "=== serve Quark-W8A8 35B INT8 MoE  TP=$TP  on $IMG  (steve recipe; +int8-MoE patch) ==="

# Env = steve's Quark MoE marker + OUR proven Battlemage multi-GPU stability env (vLLM #41663).
# Steve's box-specific CCL env (CCL_TOPO_P2P_ACCESS=1 + his OFI iface) fails on our 2x B70 with
# oneCCL `zeMemOpenIpcHandle ... ZE_RESULT_ERROR_INVALID_ARGUMENT` -- our cards have no working
# GPU P2P, so the collective must use pidfd IPC exchange + host-staged (P2P_ACCESS=0). The
# VLLM_XPU_* graph flags are no-ops on this 0.14.1 image (only VLLM_XPU_FP8_DTYPE in envs.py).
ENVS=(
  -e VLLM_USE_V1=1
  -e PYTHONDONTWRITEBYTECODE=1
  -e VLLM_XPU_QUARK_W8A8_MOE=1
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e CCL_ENABLE_SYCL_KERNELS="${SYCLKERNELS:-0}"
  -e CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
  -e SYCL_UR_USE_LEVEL_ZERO_V2=0
  -e CCL_ATL_TRANSPORT=ofi
  -e CCL_ZE_IPC_EXCHANGE=pidfd
  -e CCL_TOPO_P2P_ACCESS=0
  -e FI_TCP_IFACE="$IFACE"
  -e CCL_KVS_IFACE="$IFACE"
)
[ -n "$DEVSEL" ]  && ENVS+=( -e ONEAPI_DEVICE_SELECTOR="$DEVSEL" )
[ -n "$AFFMASK" ] && ENVS+=( -e ZE_AFFINITY_MASK="$AFFMASK" )

docker run -d --name "$NAME" --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -p "$PORT:$PORT" --ipc=host --shm-size 32g -v "$ROOT:$ROOT" \
  -v "$PATCH:$QUARK_MOE_DST:ro" \
  -v "$PATCH_LIN:$QUARK_DST:ro" \
  "${ENVS[@]}" \
  --entrypoint vllm \
  "$IMG" serve "$CKPT" --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --served-model-name "$SERVED" --dtype auto --quantization quark \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --max-model-len "${MAXLEN:-32768}" --max-num-batched-tokens 8192 --max-num-seqs "${MAXSEQS:-48}" \
  --gpu-memory-utilization "${UTIL:-0.95}" --kv-cache-dtype auto --no-enable-prefix-caching \
  --language-model-only --enforce-eager --generation-config vllm \
  >/dev/null 2>&1

ok=0
for i in $(seq 1 200); do
  curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { echo "container EXITED early (see log)"; break; }
  sleep 5
done

docker logs "$NAME" > "$LOGF" 2>&1 || true

if [ "$ok" = 1 ]; then
  SID=$(curl -s --max-time 8 "http://localhost:$PORT/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  echo "=== HEALTHY -- real 35B int8 MoE at TP=$TP! served id=$SID ==="
  echo "--- [!] verify the REAL ckpt loaded (weight-load lines; not the 0.6B default) ---"
  grep -iE "loading.*weights|model loading took|GiB|Quark|w8a8|int8|MoE|QuarkW8A8Int8" "$LOGF" | tail -14
  echo "--- gen probe (greedy) ---"
  curl -s --max-time 40 "http://localhost:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SID\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" | head -c 600; echo
  echo "=== concurrency sweep (in 2048 / out 128, c=1 2 4 8) ==="
  env NAME="$NAME" MODEL="$SID" LABEL="qwen36-35b-quark-w8a8-int8-tp${TP}" TOKPATH="$CKPT" PORT="$PORT" \
    IN="${IN:-2048}" OUT="${OUT:-128}" CONC="${CONC:-1 2 4 8}" bash "$ROOT/35_sweep_bench.sh" || true
else
  echo "=== NOT HEALTHY at TP=$TP -- real failure (worker traceback / OOM / CCL / scheme) ==="
  echo "--- error signatures ---"
  grep -iE "error|traceback|fail|oom|out of memory|killed|wait_for_ready|WorkerProc|Unsupported FusedMoe|No device of requested|ccl|level.zero|assert|exception" "$LOGF" | tail -50
  echo "--- last 30 log lines ---"
  tail -30 "$LOGF"
  echo "(full log: $LOGF)"
fi
docker stop "$NAME" 2>/dev/null || true
echo "=== quark35b TP=$TP run done -- full log at $LOGF ==="
