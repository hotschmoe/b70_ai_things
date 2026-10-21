#!/usr/bin/env bash
# CPU-only; invoke under bin/cpu-run. Original D3 and BF16 binary stay intact.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
export OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/gdn-ragged
export MAX_JOBS=4
mkdir -p "$OUT/src"
python3 "$REPO/vllm/flashnext/harness/patches/x5_gdn_ragged.py" \
  "$REPO/vllm/flashnext/plugins/b70_gdn_xpu/csrc/gdn_decode.cpp" "$OUT/src/gdn_decode.cpp"
python3 - "$REPO" "$OUT" <<'PY'
from pathlib import Path
import sys
repo, out = map(Path, sys.argv[1:])
text = (repo / 'vllm/flashnext/plugins/b70_gdn_xpu/build.sh').read_text()
text = text.replace('SRC="$ROOT/vllm/flashnext/plugins/b70_gdn_xpu/csrc"', 'SRC="$OUT/src"')
(out / 'build.sh').write_text(text, encoding='ascii')
PY
bash "$OUT/build.sh"
sha256sum "$OUT/libb70_gdn_xpu.so"
