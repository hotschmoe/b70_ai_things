#!/usr/bin/env bash
# Q8 validation: serve Qwable W4A16 int4-AutoRound (inc auto-detected) + verify served id + coherence + decode bench.
# Run UNDER the lease: gpu-run bash q8_validate.sh   (30_serve starts a DETACHED container; we wait for health, probe, stop).
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL=/models/Qwable-5-27B-Coder-int4-AutoRound; SERVED=qwable-27b-int4; PORT=18080; NAME=vllm_q8val
HOSTMODEL="$ROOT/models/$(basename "$MODEL")"   # MODEL is the CONTAINER path; check the HOST path for existence
[ -d "$HOSTMODEL" ] || { echo "[!] host $HOSTMODEL missing"; exit 2; }
LOGF="$ROOT/results/q8_validate.log"; mkdir -p "$ROOT/results"; : > "$LOGF"
docker rm -f "$NAME" 2>/dev/null || true
echo "=== serve $SERVED (v0230 GRAPH=1 NOMM=1, inc auto) ===" | tee -a "$LOGF"
env IMG=vllm-xpu-env:v0230 MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 DTYPE=auto UTIL=0.92 NOMM=1 \
    MAXLEN=8192 MAXSEQS=8 CAPSIZES=1,2,4,8,16,32,64 NAME="$NAME" PORT="$PORT" \
    bash ./30_serve_w4a8_graph.sh >>"$LOGF" 2>&1
echo "--- waiting for health (up to 360s; GRAPH=1 capture is slow) ---" | tee -a "$LOGF"
ok=0
for i in $(seq 1 36); do
  if curl -sf --max-time 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then echo "[!] container died during load" | tee -a "$LOGF"; docker logs "$NAME" 2>&1 | grep -iE "error|traceback|quant|inc|auto.?round|XPUwNa16|dim|unsupported" | tail -25 | tee -a "$LOGF"; exit 3; fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "[!] health timeout"; docker logs "$NAME" 2>&1 | tail -25 | tee -a "$LOGF"; docker stop "$NAME" 2>/dev/null; exit 3; }
echo "=== HEALTHY. served id (CLAUDE.md check): ===" | tee -a "$LOGF"
curl -s "http://localhost:$PORT/v1/models" | python3 -m json.tool 2>/dev/null | grep -E '"id"|"root"' | tee -a "$LOGF"
echo "--- coherence (greedy code gen) ---" | tee -a "$LOGF"
curl -s --max-time 120 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"Write a Python function merge_sorted(a, b) that merges two sorted lists into one sorted list, with a docstring and an example.\",\"max_tokens\":180,\"temperature\":0}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['choices'][0]['text'])" 2>/dev/null | tee -a "$LOGF"
echo "--- decode bench (TTFT-cancelled, ignore_eos) ---" | tee -a "$LOGF"
P="Write a detailed, well-commented thread-safe LRU cache in Python with get/put and a usage example."
gt(){ curl -s --max-time 240 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"prompt\":\"$P\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'; }
gt 8 >/dev/null; s0=$(date +%s.%N); ns=$(gt 64); s1=$(date +%s.%N); l0=$(date +%s.%N); nl=$(gt 352); l1=$(date +%s.%N)
awk -v ts=$s0 -v te=$s1 -v tl0=$l0 -v tl1=$l1 -v ns="${ns:-0}" -v nl="${nl:-0}" 'BEGIN{dt=(tl1-tl0)-(te-ts);dn=nl-ns;printf "qwable-27b-int4 GRAPH=1: decode_tps=%.2f  (ns=%d nl=%d; Lorbus 27B int4 ~30.8 ref)\n",(dt>0)?dn/dt:0,ns,nl}' | tee -a "$LOGF"
docker stop "$NAME" 2>/dev/null || true
echo "=== q8 validate DONE -> $LOGF ===" | tee -a "$LOGF"
