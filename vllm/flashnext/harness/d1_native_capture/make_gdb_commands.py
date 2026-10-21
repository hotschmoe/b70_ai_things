#!/usr/bin/env python3
"""Generate bounded-owner GDB session; does not attach or touch a GPU.

Attach inside the already leased container, root with per-container
CAP_SYS_PTRACE, after READY and before clients. Parent lifecycle owns the
120s client deadline and teardown, so GDB ends when EngineCore terminates.
"""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
ready=str(args.out/'gdb-attached.json')
resumed=str(args.out/'gdb-resumed.json')
# No tensor pretty-printers, auto-loaded Python helpers, external symbol
# downloads, injected target calls or inferior memory writes.
commands='''set pagination off
set confirm off
set print frame-arguments none
set auto-load python-scripts off
set debuginfod enabled off
set breakpoint pending on
handle SIGTERM nostop noprint pass
handle SIGSEGV stop print pass
catch signal SIGSEGV
commands
  silent
  printf "D1_NATIVE_SIGSEGV pc="
  p/x $pc
  p $_siginfo
  info registers
  info sharedlibrary
  thread apply all bt 32
  printf "D1_NATIVE_CAPTURE_END; delivering original SIGSEGV to existing handler\\n"
  continue
end
python
import json,os,time
def d1_on_continue(event):
    if not os.path.exists(@RESUMED@):
        with open(@RESUMED@,'w') as f:
            json.dump(dict(debugger_pid=os.getpid(),wall_ns=time.time_ns(),inferior_pid=gdb.selected_inferior().pid,event='continued'),f)
gdb.events.cont.connect(d1_on_continue)
with open(@READY@,'w') as f:
    json.dump(dict(debugger_pid=os.getpid(),wall_ns=time.time_ns(),inferior_pid=gdb.selected_inferior().pid),f)
end
continue
'''.replace('@READY@',repr(ready)).replace('@RESUMED@',repr(resumed))
(args.out/'capture.gdb').write_text(commands,encoding='ascii')
print(json.dumps(dict(prepared_only=True,path=str(args.out/'capture.gdb'))))
