#!/usr/bin/env bash
# CPU-only: rotate_w4.py RTN-int4 g128 variants. Run under bin/cpu-run (+ memory scope).
set -uo pipefail
W=/mnt/vm_8tb/b70/build/tensorfold; M=/mnt/cache/b70/tensorfold-models
PY=$W/venv/bin/python; RW=$W/main/zig/tools/xpu/rotate_w4.py
BASE=/mnt/vm_8tb/github/b70_ai_things/models/files/qwen3.8-27b/bf16
GPTQ=$M/Qwen3.8-27B-GPTQ-g128-sergiiob-9d189a60
PI=$M/qwen3.8-27b-pi-adapted-bf16
ADA=$M/pi-adapter
run() {  # name bf16dir mode
  echo "=== $1 $(date +%T)"
  systemd-run --user --scope -q -p MemoryMax=60G -- $PY -I $RW $2 $GPTQ $M/$1 $3 || { echo "FAILED $1"; return 1; }
  if [ "$2" = "$PI" ]; then   # pi variants: the adapter's own tokenizer files and chat template
    cp $ADA/chat_template.jinja $ADA/tokenizer.json $ADA/tokenizer_config.json $M/$1/
  fi
  printf 'variant=%s\nbf16 source=%s\ngptq template=%s\nmode=%s\nscript=%s (TensorFold main 9e2a9bd7)\nbuilt=%s\n' $1 $2 $GPTQ $3 $RW "$(date -Is)" > $M/$1/PROVENANCE.txt
  echo "=== done $1 $(date +%T)"
}
run qwen3.8-27b-base-rot-rtn4 $BASE rot
run qwen3.8-27b-base-norot-rtn4 $BASE norot
run qwen3.8-27b-pi-adapted-rot-rtn4 $PI rot
run qwen3.8-27b-pi-adapted-norot-rtn4 $PI norot
