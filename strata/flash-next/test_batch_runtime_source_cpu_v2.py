#!/usr/bin/env python3
"""Actual draft runtime functions exercised on tiny files and pipes, no engine."""
import copy,io,json,tempfile
from pathlib import Path
from batch_api_trace_v2 import PipeObserver
from run_batch_api_controls_v2 import derived_config
from batch_numerical_protocol_v2 import Roster
from batch_numerical_proofs_v2 import require

def main():
 checks=0
 with tempfile.TemporaryDirectory() as name:
  root=Path(name);base=root/'base';base.mkdir();out=root/'out';out.mkdir()
  cfg={'args':['--max-context','2048','--prefill','64','--batch','0','--batch-groups','1','--prompt-cache','128','--conversation-cache-mib','512','--adapt-every','0'],'env':{'STRATA_STAGE_MIRRORS':'1','STRATA_STAGE_MIRROR_SEGMENT_MIB':'1024','STRATA_SYCL_NATIVE_HC':'1','STRATA_PREFIX_DIAG':'1','ZE_AFFINITY_MASK':'0'},'model_name':'hotschmoe-dd','aliases':['qwen-source29-exact'],'port':1}
  manifest={'schema':1,'runtime':{'args':cfg['args'],'env':cfg['env'],'python_sources':{'serve/server.py':'unchanged'}},'model':{'revision':'immutable'},'tokenizer_files':{'tokenizer.json':'immutable'}}
  (base/'server-config.json').write_text(json.dumps(cfg));(base/'artifact-identity.json').write_text(json.dumps(manifest));before=(base/'server-config.json').read_bytes()
  for slots in (2,4,6):
   for diagnostic in (False,True):
    made,bound=derived_config(base,out,slots,8765,diagnostic)
    assert made['model_name']=='hotschmoe-dd' and made['aliases']!=cfg['aliases'] and 'ctx2048-prefill64-batch'+str(slots) in made['aliases'][0] and made['aliases'][0].endswith('diag24'+('on' if diagnostic else 'off')) and made['parallel']==slots
    assert made['args'][made['args'].index('--batch')+1]==str(slots)
    assert made['env']['STRATA_STAGE_MIRRORS']=='1' and made['env']['STRATA_STAGE_MIRROR_SEGMENT_MIB']=='1024'
    assert made['env']['STRATA_SLOT_OWNER_TRACE']=='1' and made['env']['STRATA_BATCH_FULL_STATE_CHAIN']==made['env']['STRATA_BATCH_PUBLIC_PREFIX']=='0'
    assert made['env']['STRATA_BATCH_FIDELITY_DIAG']==str(int(diagnostic))
    assert bound['runtime']['args']==made['args'] and bound['runtime']['env']=={k:str(v) for k,v in sorted(made['env'].items()) if k.startswith(('STRATA_','SYCL_','ONEAPI_'))}
    assert bound['model']==manifest['model'] and bound['tokenizer_files']==manifest['tokenizer_files'] and bound['runtime']['python_sources']==manifest['runtime']['python_sources'];checks+=1
  assert (base/'server-config.json').read_bytes()==before
 seen=[];raw=io.StringIO('INFO batch_slots=2\nREADY\n');observed=PipeObserver(raw,lambda direction,line:seen.append((direction,line)),'receive')
 assert observed.readline()=='INFO batch_slots=2\n' and list(observed)==['READY\n'];assert seen==[('receive','INFO batch_slots=2'),('receive','READY')]
 target=io.StringIO();sent=[];writer=PipeObserver(target,lambda d,l:sent.append((d,l)),'send');writer.write('BSTOP 0 rid=7 slotgen=1\n');writer.flush();assert target.getvalue()=='BSTOP 0 rid=7 slotgen=1\n' and sent==[('send','BSTOP 0 rid=7 slotgen=1')];checks+=2
 for slots in (2,4,6):
  roster=Roster(slots);roster.consume(f'INFO batch_slots={slots} batch_requested={slots} batch_protocol=2');roster.capacity();roster.info['batch_slots']=slots-1
  try:roster.capacity()
  except ValueError:checks+=1
  else:raise AssertionError('Fallback capacity accepted')
 print(json.dumps({'passed':True,'checks':checks,'scope':'actual Python helper source on tiny config/files/pipes only; no genuine prepare, serving, GPU or original weights'},ensure_ascii=True))
if __name__=='__main__':main()
