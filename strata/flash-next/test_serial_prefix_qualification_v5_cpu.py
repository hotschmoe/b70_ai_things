#!/usr/bin/env python3
"""CPU synthetic transport/capture rejection controls; no model execution."""
import array,copy,json,math,sys,tempfile
from pathlib import Path
import serial_prefix_qualification_v5 as q
def extract(request,directory):return q.extract(request,directory,True,True,[(0,24),(24,48)])

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
 a=fixture();b=fixture(2);am=extract(a,cap);bm=extract(b,cap);assert q.compare_requests(a,b,am,bm)['first_logits']['bitwise_equal']
 # Counter totals alone cannot conceal overlapping or missing token work.
 bad=copy.deepcopy(a);bad['stderr'][1]='PREFIX_DIAG '+json.dumps({'event':'evaluated_span','lo':1,'hi':4,'prompt_rows':3});rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][3]=bad['stderr'][3].replace('11,12,13','11,12,14');rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][4]=bad['stderr'][4].replace('request=1','request=2');rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][4]=bad['stderr'][4].replace('pos=2','pos=1');rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'].pop();rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['output_ids']=[19];rejected(lambda:q.compare_requests(a,bad,am,am))
 bad=copy.deepcopy(a);bad['LP']=['LP -0.2'];rejected(lambda:q.compare_requests(a,bad,am,am))
 bad=copy.deepcopy(a);bad['done']=bad['done'].replace('stop','length');rejected(lambda:q.compare_requests(bad,bad,am,am))
 raw=logit.read_bytes();logit.write_bytes(raw[:-4]);rejected(lambda:extract(a,cap));logit.write_bytes(raw)
 values=array.array('f',[1.0]*248320);values[42]=float('nan');logit.write_bytes(values.tobytes());rejected(lambda:extract(a,cap));logit.write_bytes(raw)
 changed=cap/'changed.f32';values[42]=1.1;changed.write_bytes(values.tobytes());cm=copy.deepcopy(bm);cm['logits'][0]['path']=str(changed);cm['logits'][0]['sha256']=q.sha(changed);rejected(lambda:q.compare_requests(a,b,am,cm))
 # Empty/partial activation coverage cannot pass as equality.
 bad=copy.deepcopy(a);bad['stderr']=[x for x in bad['stderr'] if 'first_window_residual' not in x];rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][4]=bad['stderr'][4].replace('stage=1','stage=0');rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr'][5]=bad['stderr'][5].replace('le=24','le=23');rejected(lambda:extract(bad,cap))
 bad=copy.deepcopy(a);bad['stderr']=[x for x in bad['stderr'] if not x.startswith('SFD vector')];rejected(lambda:extract(bad,cap))
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
 print('PASS CPU: bounded native pipe transport/QUIT, finite full-vocab and48-layer coverage, exact inputs/ownership/spans;15 rejection controls. No model/GPU claim.')

args=[]
for flag,value in q.GEOMETRY.items():args += [flag,value]
args.append('--no-prefill-borrow')
assert q.geometry_contract(args)['values']==q.GEOMETRY
other=list(args);q.option(other,'--max-context',8192);q.option(other,'--prefill',128)
rejected(lambda:q.geometry_contract(other))
pair=args+['--layer-split','32','--split-device','1','--trim-stage-weights']
assert q.topology_args(pair)==args
assert q.expected_stage_ranges(args)==[(0,48)] and q.expected_stage_ranges(pair)==[(0,32),(32,48)]
old=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5')
rejected(lambda:q.engine_identity(old,{'engine_receipt':str(old/'receipt.json')}))
assert q.sha(q.ROOT/'strata/flash-next/serial_prefix_qualification.py')=='9356586943ee66a1254bee792428bcfcaefea8df1696570ea63ea7fd5e0de2f9'
assert q.sha(q.ROOT/'strata/flash-next/serial_prefix_qualification_v2.py')=='5a930456b90257274358910eff71c04d3b25a5ea870bebfc2fc533c89c2db21e'
assert q.sha(q.ROOT/'strata/flash-next/patches/0018-sycl-verifier-init-owned-streams.patch')=='a9f382a8861132746475b34b738b723d281f44dc33e83aa8828ae849a87987a3'
print('PASS CPU v3: mandatory48 activation coverage, head/stage ownership, same2048/64 geometry, mismatch rejection, old17 engine refusal, immutableV1/V2 and exact18 patch. No GPU claim.')

assert q.sha(q.ROOT/'strata/flash-next/serial_prefix_qualification_v3.py')=='8fc7e840b126c13903df6119b9a566cb09082c7e69acd0328b03b854495ed155'
one=q.pilot_alias([0],args);two=q.pilot_alias([0,1],pair)
assert one!=two and 'pilotctx2048-pf64-ple65536' in one and 'pilotctx2048-pf64-ple65536' in two
assert 'ctx8192' not in two and two.endswith('cards0_1-layers0_32_48')
assert one.endswith('cards0-layers0_48')
rejected(lambda:q.pilot_alias([0,1],args))
print('PASS CPU v4: explicit pilot geometry/topology aliases, invalid card/stage roster rejected, frozenV3 unchanged.')

actual=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T085407Z-pzbw4jgp')
built=q.engine_identity(actual,{'engine_receipt':str(actual/'receipt.json')})
assert built['build_rc']==0
prepared=q.read(Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/c1-onecard-corrected-streams-prepared-v2/prepared.json'))
q.matched_upload_engine_gate(prepared,actual)
assert q.sha(q.ROOT/'strata/flash-next/serial_prefix_qualification_v4.py')=='30b38bc2fee8f6d8b80d4c83764924c61cf6689ae19a9ad3f589ae706cc65253'
print('PASS CPU v5 actual compiled0018 positive engine_identity and matching genuine full390 prepared gate; V4 unchanged. No GPU execution.')
