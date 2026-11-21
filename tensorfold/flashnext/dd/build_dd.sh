#!/usr/bin/env bash
# Build the frozen TensorFold Flash-Next DD binary (xpu-native, ReleaseFast) exactly as
# tools/flashnext_serve_bench.sh builds it, in a dedicated detached worktree.
# Usage: build_dd.sh [commit]   (default: int-rf2 372bf1b2, the last full quick-gate pass)
set -euo pipefail
host=/mnt/vm_8tb/github/b70_ai_things
commit=${1:-372bf1b2f491b747a310b6c12a1f8323aa5d22ab}
gitdir=/mnt/cache/b70/refs/TensorFold
wt=/mnt/vm_8tb/b70/build/tensorfold-dd/${commit:0:8}
logd=/mnt/vm_8tb/b70/results/hotschmoe-dd/tf-build
mkdir -p "$logd" /mnt/vm_8tb/b70/build/tensorfold-dd
[[ -d $wt ]] || git -C "$gitdir" worktree add --detach "$wt" "$commit"
cd "$wt"
[[ $(git rev-parse HEAD) == $(git -C "$gitdir" rev-parse "$commit^{commit}") ]] || { echo "worktree $wt is not at $commit" >&2; exit 1; }
[[ -z $(git status --porcelain --untracked-files=no) ]] || { echo "worktree $wt has local changes" >&2; exit 1; }
export PATH=/mnt/vm_8tb/b70/build/tensorfold/toolchain/bin:/mnt/vm_8tb/b70/build/tensorfold/toolchain/zig-0.17.0:$PATH
export ZIG_GLOBAL_CACHE_DIR=/mnt/vm_8tb/b70/build/tensorfold/zig-global-cache ZIG_LOCAL_CACHE_DIR=/mnt/vm_8tb/b70/build/tensorfold/zig-local-cache
export B70_CPU_LOCK_TIMEOUT=${B70_CPU_LOCK_TIMEOUT:-5400} B70_AGENT=${B70_AGENT:-hotschmoe-dd-tf}
date -u '+start %F %T UTC'
"$host/bin/cpu-run" systemd-run --user --scope -q -p MemoryMax=90G \
  zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=/mnt/vm_8tb/b70/build/tensorfold/ngen-src -j4
date -u '+end %F %T UTC'
sha256sum zig-out/xpu/bin/tensorfold-native | tee "$logd/${commit:0:8}-binary.sha256"
git rev-parse HEAD > "$logd/${commit:0:8}-head.txt"
