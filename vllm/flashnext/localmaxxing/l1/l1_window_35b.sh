#!/usr/bin/env bash
# L1 localmaxxing window: Qwen3.6-35B-A3B GGUF, no speculation, ONE card.
# Serves X5's selected recipe from the frozen L1 snapshot:
#   fork  = worktree vllm-lmx-20261021 (branch b70/lmx-20261021)
#   repo  = lmx-l1/frozen-repo (harness/plugins/bin/registry copy)
# health -> integrity checks -> serve -> promo gate -> lmx measure-local
# -> stop -> health. Never submits.
#
#   QUANT=UD-IQ3_XXS MAX_LEN=131072 TAG=iq3 bash l1_window_35b.sh
set -uo pipefail
L=/mnt/vm_8tb/b70/build/vllm-flashnext/lmx-l1
FR=$L/frozen-repo
H=$FR/vllm/flashnext/harness
LMX=$FR/vllm/flashnext/localmaxxing
W=/mnt/vm_8tb/b70/build/vllm-flashnext/vllm-lmx-20261021
B=/mnt/vm_8tb/b70/build/vllm-flashnext
ENGINE_COMMIT=cfd2b6ffdb8fc2a96c09712e7d04c946b149ca95
QUANT=${QUANT:?QUANT}
MAX_LEN=${MAX_LEN:?MAX_LEN}
TAG=${TAG:?TAG}
CARD=1
PORT=${PORT:-18160}
OUT=$L/runs/$TAG
NAME=vllm-fn-l1-$TAG
ALIAS=qwen3.6-35b-a3b-${QUANT,,}-x5-mtp0
HARD_KB=8388608
mkdir -p "$OUT"
test ! -e "$OUT/started" || { echo "refusing to overwrite $OUT"; exit 2; }
touch "$OUT/started"
exec > >(tee -a "$OUT/window.log") 2>&1
echo "=== L1 35B window $TAG QUANT=$QUANT MAX_LEN=$MAX_LEN $(date -u +%FT%TZ) ==="
export B70_AGENT=L1 B70_GPU_LOCK_TIMEOUT=7200 B70_CPU_LOCK_TIMEOUT=1800
export PYTHONIOENCODING=ascii:backslashreplace
if systemctl --user is-active -q hotschmoe-dd; then echo DD_ACTIVE_ABORT; exit 3; fi
AVAIL=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo); echo "MEM_AVAILABLE_KB:$AVAIL"
[ "$AVAIL" -lt 25165824 ] && { echo ABORT_LOW_RAM_START; exit 4; }

# Integrity: frozen engine and frozen repo copy.
( cd "$W" && test "$(git rev-parse HEAD)" = "$ENGINE_COMMIT" && test -z "$(git status --porcelain)" ) \
  || { echo ENGINE_NOT_FROZEN; exit 5; }
( cd "$FR" && sha256sum -c --quiet "$L/frozen-repo.final.sha256" ) || { echo FROZEN_REPO_CHANGED; exit 5; }
EXT=("$B/x5/q8-fast/libb70_x5_xpu.so" "$B/x5/gdn-bf16/libb70_gdn_xpu.so"
     "$B/x5/eventless-lazy/libx5_eventless_stub.so" "$B/x5/eventless-lazy/libx5_eventless_helper.so"
     "$FR/vllm/flashnext/plugins/b70_gguf_xpu/b70_gguf_xpu/_C.o3g.so")
sha256sum "${EXT[@]}" | tee "$OUT/so-before.sha256"
WEIGHTS=/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-${QUANT}.gguf
stat -c '%n %s %Y' "$WEIGHTS" | tee "$OUT/weights.stat"
echo "engine_commit=$ENGINE_COMMIT" > "$OUT/engine.txt"

"$FR/bin/gpu-run" --card $CARD "$FR/bin/xpu-health" --card $CARD > "$OUT/health-before.txt" 2>&1
echo "HEALTH_BEFORE_RC:$?"; tail -n 1 "$OUT/health-before.txt"
grep -q 'xpu-health: HEALTHY' "$OUT/health-before.txt" || { echo HEALTH_BEFORE_FAIL; exit 6; }

# X5 selected recipe flags (harness/x5_recipe_35b.sh), MTP off, X5_LMX=1 route.
export QUANT MTP_TOKENS=0 CARD PORT NAME MAX_LEN X5_LMX=1
export B70_X5_GDN_BF16=1 B70_D3_GDN_SPEC=0 B70_X5_GDN_MIXED=0 B70_X5_GDN_RAGGED=0
export B70_X5_EVENTLESS=1 B70_X5_EVENTLESS_LAZY=1
export B70_X5_PACK_LINEAR=1 B70_X5_Q8_LINEAR=1 B70_X5_CANONICAL=1
export B70_X5_BATCH_LINEAR=0 B70_X5_Q8_FAST=1
env | grep -E '^(B70_|QUANT|MTP_TOKENS|MAX_LEN|X5_LMX)' | grep -vE '^B70_(AGENT|GPU_LOCK|CPU_LOCK)' | sort > "$OUT/recipe-env.txt"
"$H/x5_serve_proxy.sh" > "$OUT/serve.log" 2>&1 &
SPID=$!
( while kill -0 $SPID 2>/dev/null; do
    a=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$a" -lt $HARD_KB ]; then echo "ABORT_LOW_RAM $a"; docker stop -t 10 "$NAME" >/dev/null 2>&1; exit 0; fi
    sleep 2; done ) &
WD=$!
ready=0
for i in $(seq 1 600); do
  kill -0 $SPID 2>/dev/null || { echo "serve exited before ready"; break; }
  if curl -sf --max-time 3 "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" \
     && grep -q "$ALIAS" "$OUT/models.json"; then ready=1; echo "READY $(date -u +%T)"; break; fi
  sleep 3
done
GATE_RC=9; LMX_RC=9
if [ $ready = 1 ]; then
  grep -E "KV cache size|Maximum concurrency|Model loading took|b70_x5_d3 gdn fused call" "$OUT/serve.log" | head -n 6
  curl -sf "http://127.0.0.1:$PORT/version" > "$OUT/version.json"; cat "$OUT/version.json"; echo
  "$FR/bin/cpu-run" --quiet python3 "$H/promo_gate.py" --base-url "http://127.0.0.1:$PORT" \
    --profile proxy35b --expect-alias "$ALIAS" --out "$OUT/gate.json" > "$OUT/gate.log" 2>&1
  GATE_RC=$?; echo "GATE_RC:$GATE_RC"; cat "$OUT/gate.log"
  VERSION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$OUT/version.json")
  cat > "$OUT/lmx-command.txt" <<EOF
# vLLM (fork branch b70/lmx-20261021 @ ${ENGINE_COMMIT:0:12}) + out-of-tree b70_gguf_xpu / b70_x5_xpu plugins, one Arc Pro B70, no speculative decoding
$(tr '\n' ' ' < "$OUT/recipe-env.txt")B70_GGUF=1 B70_FUSE_SHARED=1 B70_PREFILL=0 \\
vllm serve Qwen/Qwen3.6-35B-A3B --load-format b70_gguf --quantization b70_gguf (GGUF Qwen3.6-35B-A3B-${QUANT}.gguf) \\
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len $MAX_LEN --max-num-seqs 2 --max-num-batched-tokens 1024 \\
  --gpu-memory-utilization 0.90 --language-model-only --attention-backend FLASH_ATTN --no-enable-prefix-caching \\
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8]}'  (no --speculative-config)
EOF
  "$FR/bin/cpu-run" --quiet python3 "$LMX/lmx_vllm.py" --measure-local \
    --base-url "http://127.0.0.1:$PORT" --served-model hotschmoe-dd --alias "$ALIAS" \
    --hf-id unsloth/Qwen3.6-35B-A3B-MTP-GGUF --quantization "Unsloth-Dynamic-${QUANT#UD-}" \
    --hardware "$LMX/hardware_card1.json" --engine-version "$VERSION" \
    --engine-commit "$ENGINE_COMMIT" --engine-build "b70-lmx-20261021 (vllm fork b70/lmx-20261021, X5 35B recipe)" \
    --command-file "$OUT/lmx-command.txt" --mode no-spec --out "$OUT/localmaxxing"
  LMX_RC=$?; echo "LMX_RC:$LMX_RC"
  for f in "$OUT"/localmaxxing/*/validation.json; do echo "$f"; cat "$f"; done
  curl -sf "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-end.txt"
  curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models-end.json"
fi
kill $WD 2>/dev/null
docker stop -t 15 "$NAME" >/dev/null 2>&1 || true
wait $SPID; SERVE_RC=$?; echo "SERVE_RC:$SERVE_RC"
sha256sum "${EXT[@]}" > "$OUT/so-after.sha256"
cmp -s "$OUT/so-before.sha256" "$OUT/so-after.sha256" && echo SO_UNCHANGED || echo SO_CHANGED
( cd "$W" && test "$(git rev-parse HEAD)" = "$ENGINE_COMMIT" && test -z "$(git status --porcelain)" ) && echo ENGINE_STILL_FROZEN || echo ENGINE_CHANGED
( cd "$FR" && sha256sum -c --quiet "$L/frozen-repo.final.sha256" ) && echo FROZEN_REPO_UNCHANGED || echo FROZEN_REPO_CHANGED
"$FR/bin/gpu-run" --card $CARD "$FR/bin/xpu-health" --card $CARD > "$OUT/health-after.txt" 2>&1
echo "HEALTH_AFTER_RC:$?"; tail -n 1 "$OUT/health-after.txt"
echo "L1_WINDOW_DONE ready=$ready gate=$GATE_RC lmx=$LMX_RC serve=$SERVE_RC"
