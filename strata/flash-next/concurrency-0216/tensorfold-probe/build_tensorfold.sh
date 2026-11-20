#!/usr/bin/env bash
# CPU-only build of TensorFold (main + pr3603) with zig 0.17.0. No GPU touched.
# Run under: B70_CPU_LOCK_TIMEOUT=0 bin/cpu-run /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/concurrency-0216/tensorfold-probe/build_tensorfold.sh
set -uo pipefail
W=/mnt/vm_8tb/b70/build/tensorfold
export PATH=$W/toolchain/bin:$W/toolchain/zig-0.17.0:$PATH      # toolchain/bin/c++ = 'zig c++' wrapper (no host g++)
export ZIG_GLOBAL_CACHE_DIR=$W/zig-global-cache ZIG_LOCAL_CACHE_DIR=$W/zig-local-cache
NGEN=$W/ngen-src
for tree in main pr3603; do
  cd $W/$tree || exit 1
  echo "=== $tree $(git rev-parse HEAD) $(date +%T)"
  steps="xpu-native xpu-qwen27 xpu-qwen27-tests"
  [ $tree = pr3603 ] && steps="$steps xpu-tp-bench xpu-flashnext"
  for s in $steps; do
    echo "--- zig build $s $(date +%T)"
    systemd-run --user --scope -q -p MemoryMax=90G -- zig build $s -Doptimize=ReleaseFast -Dngen-include=$NGEN -j4 2>&1 | tail -40
    echo "--- exit=${PIPESTATUS[0]} $s $(date +%T)"
  done
done
echo "=== done $(date +%T)"
