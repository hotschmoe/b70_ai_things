#!/usr/bin/env python3
"""Real subprocess OS-pipe FD merge test, no GPU/Docker/model."""
import json,sys,tempfile
from pathlib import Path
from merged_numerical_protocol_v2 import MergedProtocol
from audit_layer0_capture_lifecycle_v1 import audit_text


def main():
 with tempfile.TemporaryDirectory(prefix='merged-numerical-cpu-v2-') as name:
  root=Path(name);engine=root/'engine.py';engine.write_text('''import os,sys,json
# Producer descriptor merge occurs before ANY output, including UR constructor.
os.dup2(1,2)
def out(fd,text):os.write(fd,(text+"\\n").encode("ascii"))
out(1,"<--- urUSMDeviceAlloc(.hContext = 0x111, .size = 7023104, .ppMem = 0x222 (0x1000)) -> UR_RESULT_SUCCESS;")
out(2,"L0Q8 allocation stage=0 pointer=0x1000 bytes=7023104 owner_queue=verifier_cs")
out(1,"READY 2048 numerical")
for line in sys.stdin:
 if line.strip()=="QUIT":break
 if line.startswith("GEN "):
  out(2,'PCL {"event":"begin","pid":9,"request":1}')
  out(1,'PREFIX_DIAG {"event":"selection","pid":9,"request":1}')
  out(2,'SFD request pid=9 request=1 tokens=1 ids=248045')
  out(1,"RESUME 0");out(2,"L0Q8 frame pid=9 request=1 stage=0 layer=0 position=0 token=248045 metadata=CPU_TEST")
  out(1,"T 17");out(2,'PREFIX_DIAG {"event":"finish","pid":9,"request":1}')
  out(1,"LP -0.1 17:-0.1");out(1,"DONE 1 1 0 0 length")
out(2,"L0Q8 release_begin stage=0 pointer=0x1000 bytes=7023104")
out(1,"<--- urUSMFree(.hContext = 0x111, .pMem = 0x1000) -> UR_RESULT_SUCCESS;")
out(2,"L0Q8 release_returned stage=0")
''')
  protocol=MergedProtocol([sys.executable,str(engine)],root,5);record=protocol.request('numerical',[248045],fresh=1,max_new=1);assert record['output_ids']==[17] and record['done'].endswith('length');assert protocol.close()==0
  combined=(root/'engine.combined.log').read_text();checked=audit_text(combined);assert checked['passed'];assert checked['owners'][0]['alloc_line']<checked['owners'][0]['owner_line']<checked['owners'][0]['release_begin_line']<checked['owners'][0]['free_line']<checked['owners'][0]['release_end_line']
  assert any(line.startswith('PCL ') for line in record['stderr']) and any(line.startswith('SFD request ') for line in record['stderr']) and any(line.startswith('L0Q8 frame ') for line in record['stderr'])
  assert 'READY' not in record['stderr'] and all(not line.startswith('L0Q8 ') for line in record['stdout'])
  lines=combined.splitlines();a=next(i for i,l in enumerate(lines) if 'urUSMDeviceAlloc' in l);b=next(i for i,l in enumerate(lines) if l.startswith('L0Q8 allocation'))
  reversed_lines=lines[:];reversed_lines[a],reversed_lines[b]=reversed_lines[b],reversed_lines[a];assert not audit_text('\n'.join(reversed_lines))['passed']
  omitted='\n'.join(line for line in lines if 'urUSMFree' not in line);assert not audit_text(omitted)['passed']
  result={'CONFIG':'actual bounded Python producer duplicates stderr FD tostdout before output; real oneOSpipe reader; synthetic UR only','COMMAND':'python3 strata/flash-next/test_merged_numerical_protocol_cpu_v2.py','RESULT':{'actual_opposite_write_channels_preserved_order':True,'READY_GEN_DONE_protocol_classification':True,'all_request_diagnostic_classes_retained':True,'canonical_single_log_owner_context_free_pass':True,'reversed_and_missingfree_rejected':True},'VERDICT':'PASS realOSpipe transport/chronology fixture only; no GPU/lifecycle model claim'}
  Path(__file__).with_name('merged-numerical-protocol-cpu-receipt-v2.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS realproducerFDmerge canonicalorderedlog, protocol andnegativeURcontrols; noGPU')
if __name__=='__main__':main()
