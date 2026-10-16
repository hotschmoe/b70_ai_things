#!/usr/bin/env bash
# ROOT ONLY after full read. Four required reviewed post-apply SHA arguments.
# No apply/restore/ownership repair/retry, no GPU or cache warm.
set -euo pipefail
[[ $(id -u) == 1000 && $# == 4 ]]
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=/mnt/vm_8tb/github/b70_ai_things
IMAGE=sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2
CACHE=/mnt/vm_8tb/b70/cache/strata-aot-bmg-g31
python3 - "$HERE/expected-applied-source.sha256" "$@" <<'PY'
import re,sys
want=[line.split()[0] for line in open(sys.argv[1])]
assert all(re.fullmatch('[0-9a-f]{64}',x) for x in sys.argv[2:])
assert want==sys.argv[2:],(want,sys.argv[2:])
PY
mkdir "$HERE/build"
mkdir "$HERE/build/tmp" "$HERE/build/cache"
# Existing S129 cache only; do not create, warm, clear or replace it.
[[ -d "$CACHE" ]]
python3 - "$CACHE" "$HERE/build/cache-ownership.json" <<'PY'
import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.stat()
d=dict(path=str(p),uid=s.st_uid,gid=s.st_gid,mode=oct(s.st_mode&511),writable=os.access(p,os.W_OK),symlink=p.is_symlink())
pathlib.Path(sys.argv[2]).write_text(json.dumps(d,indent=2)+'\n')
assert p.is_dir() and s.st_uid==1000 and d['writable'] and not d['symlink'],d
PY
docker image inspect "$IMAGE" > "$HERE/build/image.json"
python3 - "$HERE/build/image.json" "$IMAGE" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))[0]['Id']==sys.argv[2]
PY
cmd=(env B70_CPU_LOCK_TIMEOUT=0 B70_AGENT=S133-C6-native-only "$REPO/bin/cpu-run" env MAX_JOBS=4 docker run --rm --network none --cpus 4 -u "1000:$(id -g)" \
 -e TMPDIR=/out/tmp -e XDG_CACHE_HOME=/out/cache -e SYCL_CACHE_DIR=/out/cache \
 -e SYCL_CACHE_PERSISTENT=0 -e NEO_CACHE_PERSISTENT=1 -e NEO_CACHE_DIR=/out/offline-cache -e NEO_CACHE_STATS=1 \
 -e MAX_JOBS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
 -v /mnt/vm_8tb/b70/build/strata-ab:/work -v "$HERE":/prep:ro -v "$HERE/build":/out \
 -v "$CACHE":/out/offline-cache \
 --entrypoint bash "$IMAGE" /prep/build-container.sh)
printf '%q ' "${cmd[@]}" > "$HERE/build/command.txt"
printf '\n' >> "$HERE/build/command.txt"
set +e
"${cmd[@]}" 2>&1 | tee "$HERE/build/build.log"
rc=("${PIPESTATUS[@]}")
set -e
printf '%s\n' "${rc[0]}" > "$HERE/build/build.exit"
printf '%s\n' "${rc[1]}" > "$HERE/build/tee.exit"
[[ ${rc[0]} == 0 && ${rc[1]} == 0 ]]
