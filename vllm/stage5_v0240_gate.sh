#!/usr/bin/env bash
# Stage 5 of the vLLM v0.24.0 rebase: the headline GPU gate. Brings the sglang daily driver DOWN
# (frees both cards), serves the 27B W8A8 on vLLM v0.24.0 (:int8g-v0240 + torch-2.12 int8+GDN .so),
# runs the concurrent mixed prefill+decode coherence gate + a perf probe, then ALWAYS restores the
# daily driver (trap EXIT). Minimal custom surface: PUSH_AR=0 (plain oneCCL), CGMODE=NONE (stable).
#
# Usage:  /mnt/vm_8tb/b70/gpu-run bash vllm/stage5_v0240_gate.sh
#   (must hold BOTH cards -> run under gpu-run. The daily driver is stopped for the duration.)
set -uo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
ROOT=/mnt/vm_8tb/b70
SHELF="$REPO/rdy_to_serve/vllm/qwen36-27b-w8a8"
KDIR="$ROOT/w8a8_kernel_v0240"
PORT="${PORT:-30011}"
NAME=vllm_v0240_gate
IMG=vllm-xpu-env:int8g-v0240
GATE_LOG="$ROOT/build24/stage5_gate.log"

say(){ echo "[$(date +%H:%M:%S)] $*"; }

# Pause the dd-watchdog so it can't probe/incident during the test. It is OBSERVE-ONLY for a down
# container (heals only health-200+garbage), so this is belt-and-suspenders; owned by us (uid 1000).
WDOG_PID="$(pgrep -f 'bin/dd-watchdog' | head -1)"
resume_wdog() { [ -n "$WDOG_PID" ] && kill -CONT "$WDOG_PID" 2>/dev/null && say "resumed dd-watchdog ($WDOG_PID)"; }

restore_daily() {
  say "=== RESTORE: stop test serve + bring daily driver back (RADIX=1) ==="
  NAME=$NAME PORT=$PORT bash "$SHELF/serve.sh" stop >/dev/null 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  DD_ENV="RADIX=1" bash "$REPO/vllm/daily_driver_serve.sh" start 2>&1 | tail -6 || \
     say "!! daily driver restore FAILED -- run: DD_ENV=RADIX=1 bash vllm/daily_driver_serve.sh start"
  resume_wdog
}
trap restore_daily EXIT
[ -n "$WDOG_PID" ] && kill -STOP "$WDOG_PID" 2>/dev/null && say "paused dd-watchdog ($WDOG_PID) for the test"

# 0) sanity: kernel .so present
for f in _xpu_C.abi3.so libgdn_attn_kernels_xe_2.so; do
  [ -f "$KDIR/$f" ] || { say "MISSING $KDIR/$f (run Stage 2 build first)"; exit 2; }
done
docker image inspect "$IMG" >/dev/null 2>&1 || { say "MISSING image $IMG (run Stage 3/4 bake first)"; exit 2; }

# 1) bring the daily driver down (frees both cards)
say "=== stopping sglang daily driver ==="
bash "$REPO/vllm/daily_driver_serve.sh" stop 2>&1 | tail -4 || true
sleep 3

# 2) serve v0.24.0 (stable coherence config). GDN .so + lib from the v0240 kernel dir; PUSH_AR off.
say "=== serve 27B W8A8 on vLLM v0.24.0 (:int8g-v0240) TP=2, CGMODE=NONE, MTP, PUSH_AR=0 ==="
export IMG NAME PORT
export GDN_SO="$KDIR/_xpu_C.abi3.so"
export GDN_LIB="$KDIR/libgdn_attn_kernels_xe_2.so"
export PUSH_AR=0                # plain oneCCL: minimal custom .so surface for the coherence gate
export CGMODE=NONE              # stable decode path (shelf-proven ~25 t/s; PIECEWISE probed separately)
export GRAPH=1                  # torch.compile on, replay off (CGMODE=NONE)
export SERVED=qwen36-27b-w8a8-sqgptq-mtp
export SToffMAXLEN=""           # (unused placeholder)
export MAXLEN="${MAXLEN:-8192}" MAXSEQS="${MAXSEQS:-8}"
bash "$SHELF/serve.sh" start 2>&1 | tee "$GATE_LOG" | tail -30 || say "(serve.sh start returned non-zero -- checking /health directly)"

BASE="http://localhost:$PORT/v1"
# Robust: continue to the concurrent gate as long as the server is actually healthy, even if the
# basic single-prompt gen_probe flagged something (we want to characterize concurrent behavior).
if ! curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
  say "!! server NOT healthy on :$PORT -- serve failed to come up. Tail:"; docker logs "$NAME" 2>&1 | tail -25
  exit 1
fi
say "=== served model check ==="
curl -s "$BASE/models" 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -E '"id"|max_model_len' | head

# 3) HEADLINE GATE: concurrent mixed prefill+decode coherence
say "=== GATE: concurrent mixed prefill+decode coherence (the '!!!!' test) ==="
python3 "$REPO/vllm/gate_concurrent_coherence.py" "$BASE" "$SERVED" 3 6 200 2>&1 | tee -a "$GATE_LOG"
GATE_RC=${PIPESTATUS[0]}

# 4) perf probe (aggregate + single-stream decode t/s) if coherent
if [ "$GATE_RC" = 0 ]; then
  say "=== PERF: single-stream + concurrent decode t/s ==="
  python3 "$REPO/evals/orchestrator/concurrent_probe.py" "$BASE" "$SERVED" 1 128 2>&1 | tee -a "$GATE_LOG"
  python3 "$REPO/evals/orchestrator/concurrent_probe.py" "$BASE" "$SERVED" 4 128 2>&1 | tee -a "$GATE_LOG"
  say "=== GATE PASS (coherent). Perf logged to $GATE_LOG ==="
else
  say "=== GATE FAIL (garbage under concurrent load) -- v0.24.0 does NOT fix the mixed-batch corruption on this stack ==="
fi

# 5) teardown handled by trap (restore daily driver)
say "=== stopping test serve (trap will restore daily driver) ==="
exit $GATE_RC
