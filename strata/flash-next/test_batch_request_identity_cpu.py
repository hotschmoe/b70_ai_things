#!/usr/bin/env python3
"""Reconstruct tracked22; exercise actual API identity/TLS methods CPU-only."""
import ast,hashlib,importlib.util,json,os,subprocess,tempfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
plan=json.loads((ROOT/'strata/flash-next/batch-request-identity-draft-plan.json').read_text());base=Path(plan['base']);patch=ROOT/plan['patch']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Unsupported/stale identity accepted')
with tempfile.TemporaryDirectory(prefix='batch-identity-cpu-') as name:
 w=Path(name)
 for rel,h in plan['base_files'].items():
  assert sha(base/rel)==h;dest=w/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((base/rel).read_bytes())
 assert sha(patch)==plan['patch_sha256'];subprocess.run(['git','apply','--check',str(patch)],cwd=w,check=True);subprocess.run(['git','apply',str(patch)],cwd=w,check=True)
 for rel,h in plan['overlays'].items():assert sha(w/rel)==h
 spec=importlib.util.spec_from_file_location('identity',w/'serve/batch_request_identity.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 identity.strict_request({},None);identity.strict_request(None,None)
 for request in [{'temperature':.1},{'repetition_penalty':1.2},{'frequency_penalty':.5},{'presence_penalty':.5},{'experimental_speed_projection':False},{'logprobs':True}]:reject(lambda r=request:identity.strict_request(r,None))
 reject(lambda:identity.strict_request({},'imagefile'))
 assert identity.tags('BT 0 42 rid=7 slotgen=3',7,3)==(7,3)
 for line in ['BT 0 42','BT 0 42 rid=8 slotgen=3','BT 0 42 rid=7 slotgen=2','BT 0 42 rid=x slotgen=3']:reject(lambda x=line:identity.tags(x,7,3))
 tree=ast.parse((w/'serve/server.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StrataEngine');wanted={'last','_begin_request_identity','_request_keys','_stop_slot_identity','_parse_done','sampling_keys','projection_key'}
 assignments=[n for n in cls.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='RequestField']
 methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in wanted];code=ast.Module(body=[ast.ClassDef(name='Engine',bases=[],keywords=[],body=assignments+methods,decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(code)
 ns={'threading':threading,'RequestField':identity.RequestField,'batch_identity_tags':identity.tags,'EngineDied':RuntimeError};exec(compile(code,'actual-patched-server','exec'),ns);Engine=ns['Engine'];ns['StrataEngine']=Engine;e=Engine();e.batch=2;e.strict_batch_identity=True;e.gen=1;e._request_identity_lock=threading.Lock();e._request_identity_counter=0;e.slot_identity=[None,None];e.sent=[];e._send=e.sent.append
 barrier=threading.Barrier(2);results=[]
 def worker(index):
  e._begin_request_identity();rid=e._tl.rid;e.progress=(index,index+1);e.reused=index;barrier.wait()
  e._parse_done('DONE 2 20 1 2 stop 0 0 5 6 7 8 9 10 15 0 rid=%d slotgen=3'%rid)
  e.slot_identity[index]=(1,rid,3);e._stop_slot_identity(index)
  results.append((index,rid,e.progress,e.reused,dict(e.last),e._request_keys({})))
 threads=[threading.Thread(target=worker,args=(i,)) for i in range(2)]
 for t in threads:t.start()
 for t in threads:t.join()
 assert len(results)==2 and len({r[1] for r in results})==2
 for index,rid,progress,reused,last,keys in results:
  assert progress==(index,index+1) and reused==index and last['request_id']==rid and last['engine_generation']==1 and len(last['segments'])==1 and ' rid='+str(rid) in keys
 assert all(' rid=' in s and ' slotgen=3' in s for s in e.sent)
 e._begin_request_identity();assert e.last['segments']==[];reject(lambda:e._parse_done('DONE 2 20 1 2 stop rid=999 slotgen=3'))
 func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parallel_args');unit=ast.Module(body=[func],type_ignores=[]);ast.fix_missing_locations(unit);fns={'os':os,'PARALLEL_MAX':8};exec(compile(unit,'actual-parallel-args','exec'),fns)
 reject(lambda:fns['parallel_args']({'parallel':2},['--batch','0']));reject(lambda:fns['parallel_args']({'parallel':1},['--batch','2']));assert fns['parallel_args']({'parallel':2},['--batch','2'])==[]
 print('PASS CPU tracked22 source application/hashes, actual TLS last+progress/reuse andsegments, RID/engine/slot identity, unsupportedscope/stale negatives, explicitparallel conflict. No full source compilation/GPU/concurrency qualification.')
