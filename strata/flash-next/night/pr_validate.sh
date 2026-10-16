#!/usr/bin/env bash
# Upstream-PR GPU validation on a 2-card layer split (configs/pr-main-layer-c1.json): one start/bench/stop per line
# of the spec file; fixed prompts (--tag nl) so output SHAs compare across builds. Needs the daily driver stopped.
# Spec lines: <name> <exe> <env-json> [extra-args-json]      e.g.  kreg-on /mnt/.../strata-kreg {"STRATA_MMVQ_KREG":"1"}
# Usage: pr_validate.sh <spec-file> [bench args...]   (default bench: --c 1 --tokens 512 --reps 2 --prefill 4096,32768)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
spec=${1:?spec file}; shift
bench_args=("$@"); [ ${#bench_args[@]} -gt 0 ] || bench_args=(--c 1 --tokens 512 --reps 2 --prefill 4096,32768)
OUT=/mnt/vm_8tb/b70/results/pr-20261016/runs
mkdir -p "$OUT/configs"
if systemctl --user is-active -q hotschmoe-dd; then echo "hotschmoe-dd is running: stop it first" >&2; exit 2; fi
while read -r -u 3 name exe envjson argsjson; do
  [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  cfg=$OUT/configs/$name.json
  python3 - "$HERE/configs/pr-main-layer-c1.json" "$cfg" "$exe" "$envjson" "$OUT/$name.engine.log" "${argsjson:-[]}" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); d['exe'] = sys.argv[3]; d['env'] = {**d['env'], **json.loads(sys.argv[4])}
d['log'] = sys.argv[5]; d['args'] += json.loads(sys.argv[6]); json.dump(d, open(sys.argv[2], 'w'), indent=1)
EOF
  echo "== $name $(date -u +%T) exe=$(basename "$exe") env=$envjson"
  if ! python3 "$HERE/nl.py" start "pr-$name" "$cfg"; then
    echo "START FAILED $name"; python3 "$HERE/nl.py" stop "pr-$name" || true; continue
  fi
  python3 "$HERE/nl.py" bench "pr-$name" --out "$OUT/$name" --tag nl "${bench_args[@]}" || echo "BENCH FAILED $name"
  python3 "$HERE/nl.py" stop "pr-$name" || echo "STOP/HEALTH FAILED $name"
done 3< "$spec"
python3 - "$OUT" <<'EOF'
import json, sys, pathlib
for p in sorted(pathlib.Path(sys.argv[1]).glob('*/bench.json')):
    d = json.loads(p.read_text())
    c1 = [r for r in d.get('decode', []) if r.get('c') == 1]
    print(p.parent.name, 'coh', d.get('coherence_ok'), 'c1 engine', [r['engine_tps'] for r in c1],
          'drafts', [r['drafts'] for r in c1], 'sha', [r['sha'] for r in c1],
          'prefill', [(x.get('target'), x.get('prompt_tps')) for x in d.get('prefill', [])])
EOF
