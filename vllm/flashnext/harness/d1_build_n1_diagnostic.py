#!/usr/bin/env python3
"""Make a reviewable private merged lifecycle snapshot; CPU-only builder.

No campaign source is changed. Preserve the lifecycle prefix/suffix;
bound only the post-READY client body, keeping teardown/health outside.
"""
import argparse
import hashlib
import json
from pathlib import Path

R = Path("/mnt/vm_8tb/github/b70_ai_things")
D = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
source = R / "vllm/flashnext/harness/f1_merged_serve.sh"
text = source.read_text()
start = "if [ $ready = 1 ]; then\n  awk '/MemAvailable:|AnonPages:/"
if text.count(start) != 1:
    raise SystemExit("merged readiness anchor changed; review required")
begin = text.index(start)
end_anchor = 'echo "BENCH_RC:$BENCH_RC GATE_RC:$GATE_RC CALLBACK_RC:$CALLBACK_RC"'
if text.count(end_anchor) != 1:
    raise SystemExit("merged teardown anchor changed; review required")
end = text.index(end_anchor)
body = text[begin:end]
prefill_anchor = '    F1_MODEL=hotschmoe-dd F1_BASE="http://127.0.0.1:$PORT"'
if body.count(prefill_anchor) != 1:
    raise SystemExit("prefill anchor changed; review required")
body = body.replace(prefill_anchor,
    '    printf "detail\\n" > "$OUT/d1-liveness/arm"\n' + prefill_anchor)
body_path = args.out / "clients.sh"
body_path.write_text('''#!/bin/bash
set -uo pipefail
GREEDY_RC=2; BENCH_RC=2; GATE_RC=2; PROBE_RC=2; CALLBACK_RC=0
trap 'printf "GREEDY_RC=%s\\nBENCH_RC=%s\\nGATE_RC=%s\\nPROBE_RC=%s\\nCALLBACK_RC=%s\\n" "$GREEDY_RC" "$BENCH_RC" "$GATE_RC" "$PROBE_RC" "$CALLBACK_RC" > "$OUT/d1-client-status.env"' EXIT
trap 'CALLBACK_RC=124; exit 124' TERM INT
''' + body)
injected = '''# D1 diagnostic addition: arm only after READY, bound client body only.
D1_CLIENT_RC=2
if [ "$ready" = 1 ]; then
  mkdir -p "$OUT/d1-liveness"
  PORT="$PORT" OUT="$OUT" python3 - <<'D1_IDENTITY'
import json,os,urllib.request
from pathlib import Path
out=Path(os.environ['OUT'])
with urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/server_info?config_format=json',timeout=5) as response:
    info=json.load(response)
(out/'server-info.json').write_text(json.dumps(info,indent=2,ensure_ascii=True)+'\\n')
config=info['vllm_config']
assert 'speculative_config' in config and config['speculative_config'] is None
assert config['model_config']['max_model_len']==131072
assert config['parallel_config']['tensor_parallel_size']==1
(out/'server-info-proof.json').write_text(json.dumps(dict(no_speculation=True,maxlen=131072,tensor_parallel_size=1)))
D1_IDENTITY
  D1_IDENTITY_RC=$?
  printf 'coarse\\n' > "$OUT/d1-liveness/arm"
  D1_ARM_OK=0
  for _d1_poll in $(seq 1 100); do
    if compgen -G "$OUT/d1-liveness/armed-*.json" >/dev/null; then D1_ARM_OK=1; break; fi
    if compgen -G "$OUT/d1-liveness/install-failed-*.txt" >/dev/null; then break; fi
    sleep 0.1
  done
  if [ "$D1_ARM_OK" = 1 ] && [ "$D1_IDENTITY_RC" = 0 ]; then
    export REPO H B OUT PORT ready
    D1_ENGINE_PID=$(docker top "$NAME" -eo pid,comm | awk '$2 ~ /VLLM::EngineCor/ {print $1; exit}')
    D1_PROC_WATCH=""
    if [ -n "$D1_ENGINE_PID" ]; then
      python3 "@D@/d1_proc_watch.py" --pid "$D1_ENGINE_PID" --out "$OUT/d1-liveness/proc" --seconds 130 &
      D1_PROC_WATCH=$!
    fi
    timeout --signal=TERM --kill-after=5 120s bash "@BODY@"
    D1_CLIENT_RC=$?
    D1_CLIENT_RC="$D1_CLIENT_RC" python3 - "$OUT/d1-client-end.json" <<'D1_END'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(wall_ns=time.time_ns(), client_rc=int(os.environ['D1_CLIENT_RC']))))
D1_END
    if [ -f "$OUT/d1-client-status.env" ]; then source "$OUT/d1-client-status.env"; fi
    if [ "$D1_CLIENT_RC" != 0 ]; then CALLBACK_RC=$D1_CLIENT_RC; fi
    if [ -n "$D1_PROC_WATCH" ]; then kill "$D1_PROC_WATCH" 2>/dev/null || true; wait "$D1_PROC_WATCH" 2>/dev/null || true; fi
  else
    echo D1_ARM_FAILED
    CALLBACK_RC=1
    ready=0
  fi
  rm -f "$OUT/d1-liveness/arm"
fi
echo "D1_CLIENT_RC:$D1_CLIENT_RC"
'''.replace("@D@", str(D)).replace("@BODY@", str(body_path))
run = text[:begin] + injected + text[end:]
(args.out / "merged-source.sh").write_text(text)
(args.out / "serve.sh").write_text(run)
manifest = dict(classification="diagnostic snapshot; no GPU executed by builder",
    original_source=str(source), sha256={
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [source, body_path, args.out / "serve.sh", D / "d1_liveness.py",
                  D / "d1_proc_watch.py", D / "d1_first_n1_callback.py",
                  D / "d1_first_n1_callback.sh", D / "d1_classify_liveness.py",
                  D / "d1_n1_diagnostic.sh", Path(__file__)]},
    intended_changes=["coarse arm after READY with handshake", "detail before unchanged F1 client sequence",
                      "120-second client body bound", "exact initial N1 callback only",
                      "external CPU /proc observer", "teardown/health outside body timeout"])
(args.out / "private-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(dict(result="PREPARED_ONLY", out=str(args.out), source_sha256=manifest["sha256"][str(source)])))
