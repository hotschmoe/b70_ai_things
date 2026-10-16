#!/usr/bin/env bash
# Night-lab worktrees: isolated reflink copies of src-ep (source + build dir) for parallel agents.
#   nl_wt.sh new <name>            create /mnt/vm_8tb/b70/build/night-wt/<name>/src-ep from the main src-ep
#   nl_wt.sh build <name> <tag>    incremental `ninja strata` inside the image; exe -> builds/strata-<name>-<tag>
#   nl_wt.sh ninja <name> <target> build any ninja target (e.g. a parity/probe executable) in that worktree
set -euo pipefail
REPO=/mnt/vm_8tb/github/b70_ai_things
MAIN=/mnt/vm_8tb/b70/build/strata-ab
WT=/mnt/vm_8tb/b70/build/night-wt
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
CACHE=/mnt/vm_8tb/b70/cache/strata-aot-bmg-g31
OUT=/mnt/vm_8tb/b70/results/night-20261016/builds
cmd=${1:?cmd}; name=${2:?name}
case "$cmd" in
  new)
    mkdir -p "$WT/$name"; [[ ! -e "$WT/$name/src-ep" ]] || { echo "exists: $WT/$name/src-ep"; exit 1; }
    cp -a --reflink=always "$MAIN/src-ep" "$WT/$name/src-ep"
    echo "$WT/$name/src-ep" ;;
  build|ninja)
    target=strata; tag=${3:?tag-or-target}; [[ $cmd == ninja ]] && target=$tag
    mkdir -p "$OUT/tmp-$name" "$OUT/cache-$name"
    copy=""; [[ $cmd == build ]] && copy="&& cp strata /out/strata-$name-$tag"
    B70_CPU_LOCK_TIMEOUT=0 B70_AGENT="wt-$name" "$REPO/bin/cpu-run" docker run --rm --network none --cpus 8 -u "1000:$(id -g)" \
      -e TMPDIR=/out/tmp-$name -e XDG_CACHE_HOME=/out/cache-$name -e SYCL_CACHE_DIR=/out/cache-$name \
      -e SYCL_CACHE_PERSISTENT=0 -e NEO_CACHE_PERSISTENT=1 -e NEO_CACHE_DIR=/offline-cache \
      -v "$WT/$name/src-ep":/work/src-ep -v "$MAIN/llama-pinned":/work/llama-pinned:ro -v "$OUT":/out -v "$CACHE":/offline-cache \
      --entrypoint bash "$IMAGE" -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /work/src-ep/build-sycl-ep && ninja -j6 $target $copy" 2>&1 | tail -${NL_BUILD_TAIL:-8}
    [[ $cmd == build ]] && ls -la "$OUT/strata-$name-$tag" || true ;;
  *) echo "unknown $cmd"; exit 2 ;;
esac
