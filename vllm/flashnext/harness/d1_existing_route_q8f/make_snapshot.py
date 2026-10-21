"""Private lifecycle snapshot: replace duplicate clients with bounded phases."""
import argparse
import hashlib
import difflib
import json
from pathlib import Path
S=Path(__file__).resolve().parent
H=Path('/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness')

def build(out):
    source=(H/'f1_merged_serve.sh').read_text()
    start=source.index('GREEDY_RC=2; BENCH_RC=2; GATE_RC=2; PROBE_RC=2; CALLBACK_RC=0')
    end=source.index('docker stop -t 20 "$NAME"',start)
    body='''GREEDY_RC=2; BENCH_RC=2; GATE_RC=2; PROBE_RC=2; CALLBACK_RC=2; CLIENT_RC=2
if [ "$ready" = 1 ]; then
  awk '/MemAvailable:|AnonPages:/{print}' /proc/meminfo > "$OUT/rss-ready.txt"
  python3 "D1_SCRATCH/clients.py" --out "$OUT" --base "http://127.0.0.1:$PORT" > "$OUT/clients-launch.log" 2>&1
  CLIENT_RC=$?
  read -r GREEDY_RC GATE_RC BENCH_RC PROBE_RC CALLBACK_RC < <(python3 - "$OUT" <<'PY_STATUS'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
try:s=json.loads((p/'clients-status.json').read_text())['phases']
except Exception:s={}
def rc(n):
    r=s.get(n,{}).get('rc');return r if type(r)is int else 2
try:
    c=json.loads((p/'gate.json').read_text())['checks']
    mathok=all(c[n]['pass']is True for n in ['identity','greedy_17x19','greedy_23x29'])
except Exception:mathok=False
print(0 if mathok else 1,rc('gate'),rc('bench'),rc('probe'),rc('canonical'))
PY_STATUS
)
fi
echo "GREEDY_RC:$GREEDY_RC"
echo "CLIENT_RC:$CLIENT_RC"
echo "BENCH_RC:$BENCH_RC GATE_RC:$GATE_RC PROBE_RC:$PROBE_RC CALLBACK_RC:$CALLBACK_RC"
'''.replace('D1_SCRATCH',str(S))
    final=source[:start]+body+source[end:]
    out.mkdir(parents=True,exist_ok=True)
    (out/'serve.sh').write_text(final,encoding='ascii')
    (out/'serve.sh').chmod(0o755)
    (out/'serve.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),final.splitlines(True),
        fromfile=str(H/'f1_merged_serve.sh'),tofile=str(out/'serve.sh'))),encoding='ascii')
    (out/'snapshot.json').write_text(json.dumps({'source':str(H/'f1_merged_serve.sh'),
       'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'private_sha256':hashlib.sha256(final.encode()).hexdigest(),
       'changes':'Only postREADY duplicate client block replaced; server/lease/loader/startup/teardown/health/source logic retained'},indent=2)+'\n',encoding='ascii')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out)
