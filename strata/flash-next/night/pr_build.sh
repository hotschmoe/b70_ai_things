#!/usr/bin/env bash
# Upstream-PR build of the Strata SYCL engine: checks out <ref> in the src-pr worktree (a clean upstream main clone
# with a copied build-sycl-head), builds incrementally and copies the exe to pr-20261016/builds/strata-<tag>.
# Serialized with flock so several agents can share the one build dir.
# Usage: pr_build.sh <git-ref> <tag> [ninja-target]     (PR_BUILD_DRY=1: ninja -n only)
set -euo pipefail
ref=${1:?git ref}; tag=${2:?tag}; target=${3:-strata}
REPO=/mnt/vm_8tb/github/b70_ai_things
AB=/mnt/vm_8tb/b70/build/strata-ab
WT=$AB/src-pr
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
CACHE=/mnt/vm_8tb/b70/cache/strata-aot-bmg-g31
OUT=/mnt/vm_8tb/b70/results/pr-20261016/builds
mkdir -p "$OUT" "$OUT/tmp" "$OUT/cache"
exec 9>"$AB/src-pr.build.lock"
flock 9
git -C "$WT" checkout -q --detach "$ref"
git -C "$WT" rev-parse HEAD > "$OUT/strata-$tag.head"
ninja_cmd="ninja -j6 $target && cp $target /out/strata-$tag"
[ -n "${PR_BUILD_DRY:-}" ] && ninja_cmd="ninja -n $target | tail -5"
B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=pr-build "$REPO/bin/cpu-run" docker run --rm --network none --cpus 8 --memory 48g \
  -u "1000:$(id -g)" \
  -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
  -e SYCL_CACHE_PERSISTENT=0 -e NEO_CACHE_PERSISTENT=1 -e NEO_CACHE_DIR=/offline-cache \
  -v "$AB":/work -v "$WT":/work/src -v "$OUT":/out -v "$CACHE":/offline-cache \
  --entrypoint bash "$IMAGE" -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /work/src/build-sycl-head && time ($ninja_cmd)" 2>&1 | tail -${PR_BUILD_TAIL:-8}
[ -n "${PR_BUILD_DRY:-}" ] || ls -la "$OUT/strata-$tag"
