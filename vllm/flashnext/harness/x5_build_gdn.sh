#!/usr/bin/env bash
# Isolated BF16 instantiation of D3; no GPU, run under bin/cpu-run.
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
export OUT=/mnt/vm_8tb/b70/build/vllm-flashnext/x5/gdn-bf16
export MAX_JOBS=4
mkdir -p "$OUT/src"
python3 - "$REPO" "$OUT" <<'PY'
from pathlib import Path
import sys
repo, out = map(Path, sys.argv[1:])
root = repo / 'vllm/flashnext/plugins/b70_gdn_xpu'
src = (root / 'csrc/gdn_decode.cpp').read_text()
src = src.replace('#include <sycl/sycl.hpp>', '#include <sycl/sycl.hpp>\n#include <sycl/ext/oneapi/bfloat16.hpp>')
src = src.replace('sycl::half', 'sycl::ext::oneapi::bfloat16').replace('at::kHalf', 'at::kBFloat16')
src = src.replace(' fp16', ' bf16')
(out / 'src/gdn_decode.cpp').write_text(src)
build = (root / 'build.sh').read_text()
build = build.replace('SRC="$ROOT/vllm/flashnext/plugins/b70_gdn_xpu/csrc"', 'SRC="$OUT/src"')
(out / 'build.sh').write_text(build)
PY
bash "$OUT/build.sh"
sha256sum "$OUT/libb70_gdn_xpu.so"
