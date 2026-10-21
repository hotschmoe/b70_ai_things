#!/usr/bin/env bash
# Integrated canonical suites, called by merged harness before teardown.
# CPU client only; underlying live server already holds both GPU leases.
set -euo pipefail
R=/mnt/vm_8tb/github/b70_ai_things
B=/mnt/vm_8tb/b70/build/vllm-flashnext
OUT="${F1_OUT_DIR:?F1_OUT_DIR}/lmx"
mkdir -p "$OUT"
export B70_AGENT=D1 B70_CPU_LOCK_TIMEOUT=900
exec "$R/bin/cpu-run" --quiet bash -s -- "$R" "$B" "$OUT" <<'SH'
set -euo pipefail
R=$1
B=$2
OUT=$3
git -C "$B/vllm" rev-parse HEAD > "$OUT/fork-head.txt"
git -C "$B/vllm" diff > "$OUT/fork.patch"
git -C "$B/vllm" ls-files --others --exclude-standard > "$OUT/fork-untracked.txt"
VERSION=$(python3 -c 'import json,os,urllib.request; print(json.load(urllib.request.urlopen(os.environ["F1_BASE"]+"/version"))["version"])')
COMMIT=$(cat "$OUT/fork-head.txt")
python3 - "$OUT/command.txt" <<'PY'
import os,sys
text = (
    'Dirty diagnostic; not frozen. Fork b70/flashnext plus adjacent fork.patch '
    'and fork-untracked.txt. Original O4 decode + F1 prefill recipes plus '
    'recorded candidate overrides. Single-process EP sidecar, 2xB70, '
    'BF16 trunk + ISTA GSQ-RCO IQ3_XXS experts. No speculative config. '
    'vllm serve hf-flashnext --tensor-parallel-size 1 --dtype bfloat16 '
    '--max-model-len '+os.environ.get('MAX_LEN','131072')+' '
    '--max-num-batched-tokens '+os.environ.get('MAX_BATCHED','12288')+' '
    '--max-num-seqs 4 --gpu-memory-utilization 0.98 '
    '--attention-backend FLASH_ATTN --no-enable-prefix-caching '
    '--mamba-cache-mode align --load-format b70_gguf --quantization b70_gguf '
    '--compilation-config FULL_DECODE_ONLY:1,2,3,4,8 '
    '--served-model-name '+os.environ['F1_MODEL']+' '+os.environ['F1_ALIAS']+'; '
    'F1_PYPATH_PRE='+os.environ.get('F1_PYPATH_PRE','')+'; '
    'O4_SERVE_EXTRA='+os.environ.get('O4_SERVE_EXTRA','')
)
assert len(text)<4000
open(sys.argv[1],'w',encoding='ascii').write(text+'\n')
PY
python3 "$R/vllm/flashnext/localmaxxing/lmx_vllm.py" --measure-local \
  --base-url "$F1_BASE" --served-model "$F1_MODEL" --alias "$F1_ALIAS" \
  --hf-id ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF --quantization IQ3_XXS \
  --hardware "$R/vllm/flashnext/localmaxxing/hardware_pair.json" \
  --engine-version "$VERSION" --engine-commit "$COMMIT" \
  --engine-build 'dirty b70/flashnext; adjacent fork.patch+untracked; diagnostic only; not frozen' \
  --command-file "$OUT/command.txt" --mode no-spec --suite all \
  --max-tokens 512 --timeout 600 --out "$OUT" > "$OUT/measure.log" 2>&1
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
for name in ('reasoning-v1','code-v1'):
    directory=out/('no-spec-'+name)
    payload=json.loads((directory/'payload.json').read_text())
    path=directory/'validation.json'
    validation=json.loads(path.read_text())
    validation.update(ranking_eligible=False,diagnostic_only=True,
                      qualification='dirty working tree; gate/health closure required; no frozen or public-valid claim')
    path.write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(dict(suite=name,tok_s=payload['tokSOut'],
                          valid=validation['valid'],submitted=False,
                          ranking_eligible=False)),flush=True)
PY
SH
