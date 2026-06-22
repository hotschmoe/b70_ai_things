#!/usr/bin/env bash
# M1 -- single-card MTP on/off WITH graph capture (MTP_TODO M1). Three configs under ONE gpu-run lease:
#   A = MTP-OFF  PIECEWISE              (the denominator)
#   B = MTP-ON   PIECEWISE  spec=5      (confirms our -19% with capture; GDN verify still eager)
#   C = MTP-ON   FULL via TRITON_ATTN   (the frontier: capture attention+GDN in the verify pass; PR #34482)
# Per config: serve -> batch-1 decode bench (TTFT-cancelled: 256/(t_long-t_short)) -> grep accept -> stop.
set -uo pipefail
ROOT=/mnt/vm_8tb/b70; cd "$ROOT"
MODEL="${MODEL:-/models/Lorbus_Qwen3.6-27B-int4-AutoRound}"
SERVED="${SERVED:-qwen36-27b-int4}"
IMG="${IMG:-vllm-xpu-env:v0230}"
SPECTOK="${SPECTOK:-5}"
PORT=18080
SUMM="$ROOT/results/m1_summary_$(echo $SERVED).txt"
PROMPT="Write a detailed Python implementation of a thread-safe LRU cache class with get and put methods, including docstrings and an example usage at the bottom. Then explain the time complexity."
: > "$SUMM"
echo "=== M1 bench: $SERVED on $IMG, spec=$SPECTOK ===" | tee -a "$SUMM"

gen_tok() { # $1=max_tokens -> prints completion_tokens
  curl -s --max-time 200 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"$PROMPT\",\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    | grep -oE '"completion_tokens":[0-9]+' | grep -oE '[0-9]+'
}
bench() { # $1=label
  local label="$1"
  curl -s --max-time 90 "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":8,\"temperature\":0,\"ignore_eos\":true}" >/dev/null 2>&1
  local s0 s1 l0 l1 ns nl
  s0=$(date +%s.%N); ns=$(gen_tok 64);  s1=$(date +%s.%N)
  l0=$(date +%s.%N); nl=$(gen_tok 320); l1=$(date +%s.%N)
  local acc; acc=$(docker logs vllm_m1 2>&1 | grep -oiE "acceptance length[^0-9]*[0-9.]+|num_accepted[^0-9]*[0-9.]+|draft acceptance[^0-9]*[0-9.]+" | tail -1)
  awk -v ts="$s0" -v te="$s1" -v tl0="$l0" -v tl1="$l1" -v ns="${ns:-0}" -v nl="${nl:-0}" -v lab="$label" -v acc="$acc" \
    'BEGIN{tshort=te-ts; tlong=tl1-tl0; dt=tlong-tshort; dn=nl-ns; tps=(dt>0)?dn/dt:0; printf "%-26s decode_tps=%6.2f  (long %d tok %.2fs | short %d tok %.2fs)  accept=[%s]\n", lab, tps, nl, tlong, ns, tshort, acc}'
}
serve() { # $@ = extra env KEY=VAL
  docker rm -f vllm_m1 2>/dev/null || true
  env IMG="$IMG" MODEL="$MODEL" SERVED="$SERVED" GRAPH=1 DTYPE=auto UTIL=0.92 MAXLEN=8192 MAXSEQS=8 \
      CAPSIZES=1,2,4,8 NOMM=1 NAME=vllm_m1 "$@" bash ./30_serve_w4a8_graph.sh
}

echo ">>> Config A: MTP-OFF PIECEWISE" | tee -a "$SUMM"
serve CGMODE=PIECEWISE >/dev/null 2>&1
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "A_MTPoff_PIECEWISE" | tee -a "$SUMM"; else echo "A_MTPoff_PIECEWISE SERVE-FAIL" | tee -a "$SUMM"; fi
docker stop vllm_m1 >/dev/null 2>&1 || true; sleep 5

echo ">>> Config B: MTP-ON PIECEWISE spec=$SPECTOK" | tee -a "$SUMM"
serve CGMODE=PIECEWISE COMPILESZ= SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECTOK}" >/dev/null 2>&1
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "B_MTPon_PIECEWISE_s$SPECTOK" | tee -a "$SUMM"; else echo "B_MTPon_PIECEWISE SERVE-FAIL" | tee -a "$SUMM"; fi
docker stop vllm_m1 >/dev/null 2>&1 || true; sleep 5

echo ">>> Config C: MTP-ON FULL via TRITON_ATTN spec=$SPECTOK (the frontier)" | tee -a "$SUMM"
serve CGMODE=FULL ATTN=TRITON_ATTN TRITONSHIM=1 COMPILESZ= SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPECTOK}" >/tmp/m1c.log 2>&1
if curl -sf http://localhost:$PORT/health >/dev/null 2>&1; then bench "C_MTPon_FULL_TRITON_s$SPECTOK" | tee -a "$SUMM"; else
  echo "C_MTPon_FULL_TRITON SERVE-FAIL -- crash signature:" | tee -a "$SUMM"
  docker logs vllm_m1 2>&1 | grep -iE "error|traceback|work_group_scratch|TRITON|FULL|notimplement|assert|no attribute|capture" | tail -20 | tee -a "$SUMM"
fi
docker stop vllm_m1 >/dev/null 2>&1 || true

echo "=== M1 SUMMARY ===" | tee -a "$SUMM"; cat "$SUMM"
echo "=== M1 done; summary $SUMM ==="
