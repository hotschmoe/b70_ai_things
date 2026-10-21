"""Exact existing-control body; private teardown grace and passive watcher only."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
S=Path(__file__).resolve().parent
CORE=Path('/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/d1_existing_route_q8f')
BASE=Path('/mnt/vm_8tb/b70/build/vllm-flashnext/d1/existing_route_q8f/d1-existing-control-r1/serve.sh')
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('d1_retained_builder',CORE/'make_snapshot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.build(a.out)
path=a.out/'serve.sh';before=path.read_text();anchor='docker stop -t 20 "$NAME" >/dev/null 2>&1 || true'
assert before.count(anchor)==1
replacement='''python3 "D1_PRIVATE/teardown_watch.py" --container "$NAME" --seconds 45 \\
  --out "$OUT/teardown-proc.jsonl" --armed "$OUT/teardown-armed.json" > "$OUT/teardown-watch.log" 2>&1 &
D1_WATCH_PID=$!
for _w in $(seq 1 70); do [ -s "$OUT/teardown-armed.json" ] && break; sleep .1; done
python3 - "$OUT/teardown-armed.json" <<'PY_ARM'
import json,sys
try:ok=json.load(open(sys.argv[1])).get('first_sample_saved')is True
except Exception:ok=False
print('TEARDOWN_FIRST_SAMPLE_SAVED:'+str(ok))
raise SystemExit(0 if ok else 1)
PY_ARM
D1_WATCH_ARM_RC=$?; echo "TEARDOWN_ARM_RC:$D1_WATCH_ARM_RC"
# Stop ALWAYS, including lookup/handshake/sampler failure; no stranded serve.
docker stop -t 45 "$NAME" >/dev/null 2>&1 || true
wait $D1_WATCH_PID; echo "TEARDOWN_WATCH_RC:$?"
'''.replace('D1_PRIVATE',str(S))
final=before.replace(anchor,replacement);path.write_text(final,encoding='ascii')
(a.out/'grace-vs-control.diff').write_text(''.join(difflib.unified_diff(BASE.read_text().splitlines(True),final.splitlines(True),fromfile=str(BASE),tofile=str(path))),encoding='ascii')
meta=json.loads((a.out/'snapshot.json').read_text());meta.update(private_sha256=hashlib.sha256(final.encode()).hexdigest(),
    shutdown_cli='--shutdown-timeout 30 (wrapper SERVE_ARGS_EXTRA)',docker_stop_grace_s=45,
    intervention='same production clients/observer; only shutdown30/Docker45 plus read-only teardown proc watcher')
(a.out/'snapshot.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='ascii')
