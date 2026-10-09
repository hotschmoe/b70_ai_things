#!/usr/bin/env python3
"""CPU synthetic transport/capture rejection controls; no model execution."""
import array,copy,json,math,sys,tempfile
from pathlib import Path
import serial_prefix_qualification as q

def rejected(fn):
 try:fn()
 except (ValueError,IndexError,KeyError):return
 raise AssertionError('Negative control accepted')

with tempfile.TemporaryDirectory(prefix='strata-serial-prefix-cpu-') as name:
 d=Path(name);cap=d/'captures';cap.mkdir()
 logit=cap/'first.f32';logit.write_bytes(array.array('f',[1.0]*248320).tobytes())
 for i in range(48):(cap/('layer%d.f32'%i)).write_bytes(array.array('f',[2.0]*10240).tobytes())
 def fixture(reused=0):
  sel={'event':'selection','stages':2}
  span={'event':'evaluated_span','lo':reused,'hi':3,'prompt_rows':3-reused}
  finish={'event':'finish','prompt_tokens':3,'actual_reused':reused,'candidate_resume':reused,'evaluated_prompt_rows':3-reused,'stage_prompt_rows':(3-reused)*2,'cancelled':False}
  stderr=['PREFIX_DIAG '+json.dumps(x) for x in [sel,span,finish]]+['SFD request pid=9 request=1 tokens=3 ids=11,12,13 shape=4x2560','SFD vector pid=9 request=1 stage=1 lb=24 le=48 phase=first_logits_before_sampler layer=-1 pos=2 token=13 bytes=993280 file=/results/captures/first.f32']
  for i in range(48):stderr.append('SFD vector pid=9 request=1 stage=%d lb=%d le=%d phase=first_window_residual layer=%d pos=2 token=13 bytes=40960 file=/results/captures/layer%d.f32'%(i//24,(i//24)*24,(i//24+1)*24,i,i))
  return {'ids':[11,12,13],'fresh':int(reused==0),'stderr':stderr,'output_ids':[17,18],'LP':['LP -0.1 17:-0.1'],'done':'DONE 2 3 1.0 1.0 stop'}
 a=fixture();b=fixture(2);am=q.extract(a,cap);bm=q.extract(b,cap);assert q.compare_requests(a,b,am,bm)['first_logits']['bitwise_equal']
 # Counter totals alone cannot conceal overlapping or missing token work.
 bad=copy.deepcopy(a);bad['stderr'][1]='PREFIX_DIAG '+json.dumps({'event':'evaluated_span','lo':1,'hi':4,'prompt_rows':3});rejected(lambda:q.extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][3]=bad['stderr'][3].replace('11,12,13','11,12,14');rejected(lambda:q.extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][4]=bad['stderr'][4].replace('request=1','request=2');rejected(lambda:q.extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][4]=bad['stderr'][4].replace('pos=2','pos=1');rejected(lambda:q.extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'].pop();rejected(lambda:q.extract(bad,cap))
 bad=copy.deepcopy(a);bad['output_ids']=[19];rejected(lambda:q.compare_requests(a,bad,am,am))
 bad=copy.deepcopy(a);bad['LP']=['LP -0.2'];rejected(lambda:q.compare_requests(a,bad,am,am))
 bad=copy.deepcopy(a);bad['done']=bad['done'].replace('stop','length');rejected(lambda:q.compare_requests(bad,bad,am,am))
 raw=logit.read_bytes();logit.write_bytes(raw[:-4]);rejected(lambda:q.extract(a,cap));logit.write_bytes(raw)
 values=array.array('f',[1.0]*248320);values[42]=float('nan');logit.write_bytes(values.tobytes());rejected(lambda:q.extract(a,cap));logit.write_bytes(raw)
 changed=cap/'changed.f32';values[42]=1.1;changed.write_bytes(values.tobytes());cm=copy.deepcopy(bm);cm['logits'][0]['path']=str(changed);cm['logits'][0]['sha256']=q.sha(changed);rejected(lambda:q.compare_requests(a,b,am,cm))
 # Real host subprocess pipe/pump/QUIT fixture; it has no SYCL or GPU runtime.
 mock=d/'mock.py';mock.write_text('''import json,sys
print("READY 2048 stop",flush=True)
for line in sys.stdin:
 if line.strip()=="QUIT":break
 if line.startswith("GEN "):
  print('PREFIX_DIAG {"event":"finish"}',file=sys.stderr,flush=True)
  print("RESUME 0",flush=True);print("T 17",flush=True);print("LP -0.1 17:-0.1",flush=True);print("DONE 1 3 0 0 stop",flush=True)
''')
 (d/'ARM').touch();p=q.Protocol([sys.executable,str(mock)],d,5);r=p.request('transport',[11,12,13],fresh=1,max_new=1);assert r['output_ids']==[17] and 'fresh=1 pin=0' in r['command'];assert p.close()==0
 print('PASS CPU: bounded native pipe transport/QUIT, finite full-vocab and48-layer coverage, exact inputs/ownership/spans;11 rejection controls. No model/GPU claim.')
