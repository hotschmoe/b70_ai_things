"""CPU-only verify proc field52 and exact private grace intervention."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
S=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('d1_teardown_cpu',S/'teardown_watch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--prepared',type=Path,required=True)
args=parser.parse_args()
output=args.out.resolve();output.mkdir(parents=True,exist_ok=False)
p=output/'cpu-child.jsonl'
assert not p.exists()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.25);raise SystemExit(7)'])
r=m.watch(child.pid,p,seconds=2)
assert child.wait()==7
assert r['last_leader']['state']=='Z'
assert r['last_leader']['exit_code_waitstatus']==7<<8
assert os.waitstatus_to_exitcode(r['last_leader']['exit_code_waitstatus'])==7
for f in [*S.glob('*.py'),S/'window.sh',args.prepared/'serve.sh']:
 f.read_bytes().decode('ascii')
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.sh':subprocess.run(['bash','-n',str(f)],check=True)
source=(S/'window.sh').read_text();private=(args.prepared/'serve.sh').read_text()
assert "SERVE_ARGS_EXTRA='--shutdown-timeout 30'"in source
assert 'docker stop -t 45 "$NAME"'in private
assert 'docker stop -t 20 "$NAME"'not in private
assert 'B70_EP_ROUTE=1 B70_DECODE_HC_Q8F=1'in source # inherited unreachable branch
assert '[[ "$ARM" = control || "$ARM" = routeq8f ]]'in source
assert source.count('bash "$OUT/serve.sh"')==1 # one explicit arm, neverauto-chain
assert private.count('clients.py" --out')==1
assert 'seq 1 70'in private and 'TEARDOWN_FIRST_SAMPLE_SAVED:'in private
assert private.index('TEARDOWN_ARM_RC:')<private.index('docker stop -t 45')<private.index('wait $D1_WATCH_PID')
r={'verdict':'CPU_PASS; no GPU or loaded-runtime shutdown qualification',
   'checks':['proc_field52_natural_child_exit7','no_watcher_signal','positiveCLI30','Dockergrace45',
             'same_control_clients','explicit_arm_no_auto_chain','handshake7s_then_stop_always','ASCII_AST_BASH'],
   'child_witness':r}
(output/'cpu-result.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');print(json.dumps(r))
