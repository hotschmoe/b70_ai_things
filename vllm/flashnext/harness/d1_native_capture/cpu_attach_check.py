#!/usr/bin/env python3
"""CPU-only debugger validation, no /dev/dri and no Torch import."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
assert not Path('/dev/dri').exists(), 'CPU-only test must have no GPU device mount'
subprocess.run(['cc','-g','-O0','-fno-omit-frame-pointer',str(root/'segv_target.c'),'-o',str(out/'segv_target')],check=True,timeout=15)
subprocess.run(['python3',str(root/'make_gdb_commands.py'),'--out',str(out)],check=True,timeout=5)
child_log=open(out/'child.log','w');gdb_log=open(out/'gdb.log','w')
child=subprocess.Popen([str(out/'segv_target'),str(out/'go')],stdout=child_log,stderr=subprocess.STDOUT)
debugger=None
started=time.monotonic()
try:
    debugger=subprocess.Popen(['/usr/bin/gdb','--nx','--batch','-q','-iex','set auto-load off',
                               '-iex','set debuginfod enabled off','-iex','set print frame-arguments none',
                               '--pid',str(child.pid),'-x',str(out/'capture.gdb')],stdout=gdb_log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+8
    while not (out/'gdb-resumed.json').exists():
        if debugger.poll() is not None or time.monotonic()>deadline:
            raise RuntimeError('GDB attach/handshake failed; inspect gdb.log')
        time.sleep(.05)
    (out/'go').touch()
    child_rc=child.wait(timeout=5)
    debugger_rc=debugger.wait(timeout=5)
    child_log.flush();gdb_log.flush()
    text=(out/'gdb.log').read_text()
    assert child_rc==-signal.SIGSEGV, 'fatal signal was not preserved'
    assert text.count('D1_NATIVE_SIGSEGV pc=')==2, 'both recoverable and fatal SIGSEGV must be captured'
    assert 'D1_RECOVERABLE_HANDLER_RAN' in (out/'child.log').read_text(), 'existing handler did not run'
    assert '#0' in text and 'main' in text, 'native backtrace missing'
    result=dict(result='PASS',scope='CPU-only synthetic child; no GPU/server',child_rc=child_rc,debugger_rc=debugger_rc,
                elapsed_s=time.monotonic()-started,checks=['per-container SYS_PTRACE/default seccomp attach',
                'two SIGSEGV native PCs and backtraces','recoverable existing handler preserved','fatal SIGSEGV waitstatus preserved',
                'no target function calls/tensor printers/external symbol fetch'])
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:
    for process in (debugger,child):
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=3)
    child_log.close();gdb_log.close()
