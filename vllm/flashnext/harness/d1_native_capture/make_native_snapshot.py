#!/usr/bin/env python3
"""Prepare only: private runner/image/capability and native debug handshake."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

R=Path('/mnt/vm_8tb/github/b70_ai_things')
H=R/'vllm/flashnext/harness'
N=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
subprocess.run(['python3',str(H/'d1_build_n1_diagnostic.py'),'--out',str(args.out)],check=True)
xpu=(H/'xpu-run.sh').read_text()
changes=[('HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',f'HERE={H}'),
         ('IMAGE=vllm-flashnext-xpu:py312','IMAGE=vllm-flashnext-xpu:d1-gdb-r1'),
         ('    --ulimit nofile=1048576:1048576','    --ulimit nofile=1048576:1048576\n    --cap-add SYS_PTRACE')]
for old,new in changes:
    assert xpu.count(old)==1, 'runner anchor changed'
    xpu=xpu.replace(old,new)
(args.out/'xpu-run.sh').write_text(xpu)
(args.out/'xpu-run.sh').chmod(0o755)
overlay=args.out/'native-overlay';overlay.mkdir(exist_ok=True)
monitor=(H/'d1_liveness.py').read_text()
monitor=monitor.replace('import faulthandler','''# Native capture arm: disable BOTH D1 C-timed and Python traceback dumps.
class _NoTracebackDumps:
    @staticmethod
    def dump_traceback_later(*args, **kwargs): pass
    @staticmethod
    def cancel_dump_traceback_later(*args, **kwargs): pass
    @staticmethod
    def dump_traceback(*args, **kwargs): pass
faulthandler = _NoTracebackDumps()
''')
ast.parse(monitor)
(overlay/'d1_liveness.py').write_text(monitor)
serve=(args.out/'serve.sh').read_text()
old='"$H/xpu-run.sh" --card both'
assert serve.count(old)==1
serve=serve.replace(old,f'"{args.out}/xpu-run.sh" --card both')
attach='''  mkdir -p "$OUT/d1-native"
  python3 "@N@/make_gdb_commands.py" --out "$OUT/d1-native"
  D1_ENGINE_HOST=$(docker top "$NAME" -eo pid,comm | awk '$2 ~ /VLLM::EngineCor/ {print $1; exit}')
  D1_ENGINE_NS=""
  if [ -n "$D1_ENGINE_HOST" ]; then D1_ENGINE_NS=$(awk '/^NSpid:/ {print $NF}' "/proc/$D1_ENGINE_HOST/status"); fi
  D1_GDB_READY=0
  D1_GDB_PID=""
  if [ -n "$D1_ENGINE_NS" ]; then
    docker exec --user 0 "$NAME" /usr/bin/gdb --nx --batch -q \
      -iex 'set auto-load off' -iex 'set debuginfod enabled off' -iex 'set print frame-arguments none' --pid "$D1_ENGINE_NS" \
      -x "$OUT/d1-native/capture.gdb" > "$OUT/d1-native/gdb.log" 2>&1 &
    D1_GDB_PID=$!
    for _d1_gdb_wait in $(seq 1 100); do
      if [ -f "$OUT/d1-native/gdb-resumed.json" ]; then D1_GDB_READY=1; break; fi
      kill -0 "$D1_GDB_PID" 2>/dev/null || break
      sleep .1
    done
  fi
'''.replace('@N@',str(N))
anchor="  printf 'coarse\\n' > \"$OUT/d1-liveness/arm\""
assert serve.count(anchor)==1
serve=serve.replace(anchor,attach+anchor)
anchor='  if [ "$D1_ARM_OK" = 1 ] && [ "$D1_IDENTITY_RC" = 0 ]; then'
assert serve.count(anchor)==1
serve=serve.replace(anchor,'  if [ "$D1_ARM_OK" = 1 ] && [ "$D1_IDENTITY_RC" = 0 ] && [ "$D1_GDB_READY" = 1 ]; then')
anchor='wait $SP; SERVE_RC=$?; echo "SERVE_RC:$SERVE_RC"'
assert serve.count(anchor)==1
cleanup='''
if [ -n "${D1_GDB_PID:-}" ]; then
  for _d1_gdb_close in $(seq 1 30); do kill -0 "$D1_GDB_PID" 2>/dev/null || break; sleep .1; done
  kill -0 "$D1_GDB_PID" 2>/dev/null && kill -TERM "$D1_GDB_PID" 2>/dev/null || true
  wait "$D1_GDB_PID"; D1_GDB_RC=$?; echo "D1_GDB_RC:$D1_GDB_RC"
fi
'''
serve=serve.replace(anchor,anchor+'\n'+cleanup)
(args.out/'serve.sh').write_text(serve)
files=[H/'xpu-run.sh',H/'f1_merged_serve.sh',H/'d1_liveness.py',H/'d1_build_n1_diagnostic.py',
       args.out/'xpu-run.sh',args.out/'serve.sh',args.out/'clients.sh',overlay/'d1_liveness.py',
       N/'make_gdb_commands.py',Path(__file__),N/'native_window.sh']
files += [N/name for name in ('Dockerfile.gdb','check_plan.py','check_packages.py',
                              'cpu_attach_check.py','segv_target.c','prepare_cpu.sh','README.md')]
manifest=dict(prepared_only=True,image='vllm-flashnext-xpu:d1-gdb-r1',changes=['private IMAGE/HERE/SYS_PTRACE runner',
    'native GDB attach handshake before clients','both D1 faulthandler paths disabled in private overlay',
    'GDB SIGSEGV PC/bt then pass/continue; SIGTERM pass through','bounded GDB cleanup outside client bound'],
    sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
(args.out/'native-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(prepared_only=True,out=str(args.out))))
