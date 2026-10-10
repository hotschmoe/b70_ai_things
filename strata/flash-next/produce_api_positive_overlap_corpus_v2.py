"""Root-only authentic overlap corpus tokenization; no model inference."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import api_cache_positive_contract_v2 as original
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE_PLAN=HERE/'api-positive-overlap-corpus-source-plan-v2.json'
SEED=HERE/'api-positive-overlap-corpus-seed-v2.json'
RENDER=HERE/'render_api_positive_overlap_corpus_v2.py'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source')
TOKENIZER=Path('/mnt/vm_8tb/b70/models/flashnext-native-source-pack-20261009-v3/tokenizer')
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
read=original.read;sha=original.sha;require=original.require
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
def source_binding():
 p=read(SOURCE_PLAN)
 for n,w in p['files'].items():require(sha(ROOT/n)==w,'Overlap corpus source changed '+n)
 return {'source_plan_sha256':sha(SOURCE_PLAN),'files':p['files']}
def inputs_binding(sdk,tokenizer):
 expected=read(HERE/'current-ple-prompt35-engine-build-plan-v1.json')['expected_patched_source_sha256'];sources={n:sha(Path(sdk)/n) for n in ('tools/strata_tokenizer.py','serve/frontend.py')};require(sources=={n:expected[n] for n in sources},'Exact source35 tokenizer/frontend required')
 files={p.name:sha(p) for p in Path(tokenizer).iterdir() if p.is_file()};require(files==read(HERE/'batch-numerical-case2-source35-v2.json')['tokenizer_sha256'],'Exact original tokenizer/template file roster required')
 return {'sources':sources,'tokenizer_files':files}
def recipe(out,pid):
 out=Path(out).resolve();name='b70-api-positive-overlap-tokenizer-v2-'+str(pid);mounts=[(out/'render.snapshot.py','/harness/render.py',False),(out/'seed.snapshot.json','/seed.json',False),(out/'strata_tokenizer.snapshot.py','/tools/strata_tokenizer.py',False),(out/'frontend.snapshot.py','/serve/frontend.py',False),(TOKENIZER,'/tokenizer',False),(out/'generated','/results',True)]
 cmd=['docker','run','--name',name,'--label','b70.api.positive.overlap.fixture='+sha(SOURCE_PLAN),'--network','none','--read-only','--memory','512m','--memory-swap','512m','--cpus','1','--pids-limit','128','--user','1000:1000','--entrypoint','/usr/bin/env','-w','/harness']
 for src,dst,rw in mounts:cmd+=['-v',str(src)+':'+dst+(':rw' if rw else ':ro')]
 cmd += [IMAGE,'-i','PATH=/usr/bin:/bin','LANG=C','LC_ALL=C','PYTHONDONTWRITEBYTECODE=1','/opt/b70-c1-python/bin/python','/harness/render.py']
 return name,cmd,mounts
def owned(obj,out,pid):
 name,cmd,mounts=recipe(out,pid);cfg=obj['Config'];host=obj['HostConfig']
 require(obj['Name']=='/'+name and obj['Image']==IMAGE and cfg['Image']==IMAGE and cfg['Labels'].get('b70.api.positive.overlap.fixture')==sha(SOURCE_PLAN),'Foreign tokenizer container owner')
 require(cfg['WorkingDir']=='/harness' and cfg['User']=='1000:1000' and cfg['Entrypoint']==['/usr/bin/env'] and cfg['Cmd']==cmd[cmd.index(IMAGE)+1:],'Actual tokenizer entrypoint/command/user differs')
 require(host['NetworkMode']=='none' and host['ReadonlyRootfs'] is True and host['Privileged'] is False and host['Memory']==host['MemorySwap']==512*1024*1024 and host['NanoCpus']==1000000000 and host['PidsLimit']==128 and host['Devices']==[] and host['DeviceRequests'] is None and not host.get('GroupAdd'),'CPU-only memory/network/device profile differs')
 require(sorted((m['Source'],m['Destination'],m['RW'],m['Type']) for m in obj['Mounts'])==sorted((str(s),d,r,'bind') for s,d,r in mounts),'Exact tokenizer-only mounts differ')
 return obj
def fixture_contract(fixture,seed,inputs):
 require(fixture['source_seed_sha256']==sha(SEED) and fixture['producer_sha256']==sha(RENDER) and fixture['overlap_corpus_fixture_generation']==2,'Actual corpus producer/seed changed')
 require(fixture['source_file_sha256']=={'/'+n:v for n,v in inputs['sources'].items()} and fixture['tokenizer_file_sha256']==inputs['tokenizer_files'],'Consumed original tokenizer/template/source changed')
 for key in ('actual_GPU_touch','actual_model_payload_read','actual_CPU_continuation_observed','actual_GPU_positive_overlap_observed','warm_two_row_runtime_qualified','matched_buffer_only_A_B_input_equivalent'):
  require(fixture[key] is False,'Tokenization cannot qualify model/cache/continuation '+key)
 for phase,count in [('warm',2),('target',6)]:
  rows=fixture['fixtures'][phase];require(len(rows)==count and [r['logical_index'] for r in rows]==list(range(count)) and [r['messages'] for r in rows]==seed['messages'][phase],'Complete exact corpus roster required')
  for row in rows:require(type(row['rendered'])is str and row['rendered'].isascii() and 2<len(row['ids'])<1984 and all(type(t)is int and 0<=t<248320 for t in row['ids']),'Actual corpus token/rendered extent/type changed')
  if phase=='target':require(all(len(row['ids'])>64 for row in rows),'Actual target must exceed source prefill64 for later partial-read overlap')
 require([c['target_indices'] for c in seed['candidates']]==[[0,1],[2,3],[4,5]] and [c['selection_order'] for c in seed['candidates']]==[0,1,2],'Preregistered candidate selection order changed')
 return {'actual_CPU_tokenization_only':True,'candidate_selection':'NONE before genuine CPU screen','actual_CPU_continuation_observed':False,'actual_GPU_positive_overlap_observed':False}
def finalized_binding(root):
 root=Path(root).resolve();r=read(root/'receipt.json');binding=source_binding();require(r['passed'] is True and r['return_code']==0 and r['error'] is None and r['removed'] is True and r['source_before']==r['source_after']==binding,'Completed authentic fixture source/terminal proof required')
 name,cmd,_=recipe(root,r['producer_pid']);require(read(root/'command.json')==r['command']==cmd,'Exact original tokenizer command differs');client=r['client_launch_receipt'];require(client['command']==cmd and client['return_code']==0 and client['error'] is None and client['stop_signals']==[] and client['subreaper']=={'subreaper':True,'owner_pid':r['producer_pid']} and client['launch_descendants']['launch_session_empty'] is True and client['launch_descendants']['adopted_children']=={} and client['launch_descendants']['tracking_errors']==[] and client['client_pid']==client['launch_descendants']['producer_pid']==client['launch_descendants']['producer_session'] and r['started_epoch']<=client['client_started_epoch']<=client['client_terminal_epoch']<=r['finished_epoch'],'Original tokenizer Popen/session/owned-terminal receipt changed');owned(r['inspection'],root,r['producer_pid']);s=r['inspection']['State'];require(s['ExitCode']==0 and not s['Running'] and not s['OOMKilled'] and not s['Error'],'Original tokenizer failed terminal')
 current=inputs_binding(SDK,TOKENIZER);require(current==r['inputs_before']==r['inputs_after'],'Current original source/tokenizer changed')
 require(set(r['files'])=={'seed.snapshot.json','render.snapshot.py','strata_tokenizer.snapshot.py','frontend.snapshot.py','command.json','stdout.log','stderr.log','generated/fixtures.json'},'Exact original fixture evidence roster required')
 for n,w in r['files'].items():require(sha(root/n)==w,'Actual preserved tokenizer evidence changed '+n)
 require((root/'seed.snapshot.json').read_bytes()==SEED.read_bytes() and (root/'render.snapshot.py').read_bytes()==RENDER.read_bytes(),'Original fixture producer/seed snapshots differ')
 for n,p in [('strata_tokenizer.snapshot.py',SDK/'tools/strata_tokenizer.py'),('frontend.snapshot.py',SDK/'serve/frontend.py')]:require((root/n).read_bytes()==p.read_bytes(),'Actual source snapshot changed')
 fixture=read(root/'generated/fixtures.json');corpus=fixture_contract(fixture,read(root/'seed.snapshot.json'),current)
 require(source_binding()==binding and inputs_binding(SDK,TOKENIZER)==current and all(sha(root/n)==w for n,w in r['files'].items()),'Fixture source/data changed during readonly admission')
 return fixture,{'root':str(root),'receipt_sha256':sha(root/'receipt.json'),'fixture_sha256':sha(root/'generated/fixtures.json'),'source_binding':binding,'actual_CPU_tokenization_only':True,'corpus_scope':corpus,'warm_two_row_runtime_qualified':False,'matched_buffer_only_A_B_input_equivalent':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();require(not out.exists(),'Fresh fixture directory required');source=source_binding();inputs=inputs_binding(SDK,TOKENIZER);seed=read(SEED);require(seed['schema']=='api-positive-overlap-corpus-seed-v2' and len(seed['messages']['target'])==6 and len(seed['messages']['warm'])==2,'Declared complete corpus required before Docker');from api_positive_overlap_finite_corpus_v2 import binding as finite_binding
 finite_binding();out.mkdir(parents=True);(out/'generated').mkdir()
 for name,path in [('seed.snapshot.json',SEED),('render.snapshot.py',RENDER),('strata_tokenizer.snapshot.py',SDK/'tools/strata_tokenizer.py'),('frontend.snapshot.py',SDK/'serve/frontend.py')]: (out/name).write_bytes(path.read_bytes())
 name,cmd,_=recipe(out,os.getpid());write(out/'command.json',cmd);r={'passed':False,'error':None,'producer_pid':os.getpid(),'started_epoch':time.time(),'source_before':source,'inputs_before':inputs,'corpus_seed_sha256':sha(SEED),'command':cmd}
 try:
  from finite_tokenizer_owned_execution_v2 import execute
  row=execute(cmd,out,lambda obj:owned(obj,out,r['producer_pid']))
  r['client_launch_receipt']=row;r['return_code']=row['return_code'];require(row['return_code']==0 and row['error'] is None and not row['stop_signals'] and not row['launch_descendants']['adopted_children'] and not row['launch_descendants']['tracking_errors'],'Authentic tokenizer execution/launch retirement failed')
 except BaseException as exc:r['error']=type(exc).__name__+': '+str(exc)
 finally:
  try:
   obj=owned(json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0],out,r['producer_pid'])
   if obj['State']['Running']:subprocess.run(['docker','stop','--time','10',name],check=True);obj=owned(json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0],out,r['producer_pid'])
   r['inspection']=obj;subprocess.run(['docker','rm',name],check=True,capture_output=True);r['removed']=not subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True).strip()
  except BaseException as exc:r['error']=r['error'] or type(exc).__name__+': '+str(exc)
 r['source_after']=source_binding();r['inputs_after']=inputs_binding(SDK,TOKENIZER);r['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()};r['finished_epoch']=time.time();
 if r['error'] is None:
  try:
   require(r['source_before']==r['source_after'] and r['inputs_before']==r['inputs_after'] and r['removed'] is True,'Pre/post source or removal changed');state=r['inspection']['State'];require(state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'] and not state['Error'],'Actual tokenizer terminal failed');fixture_contract(read(out/'generated/fixtures.json'),seed,inputs)
  except BaseException as exc:r['error']=type(exc).__name__+': '+str(exc)
 r['passed']=r['error'] is None;write(out/'receipt.json',r)
 finalized_binding(out);print(json.dumps({'passed':True,'fixture':str(out),'GPU_touch':False,'warm_two_row_runtime_qualified':False}));return 0
if __name__=='__main__':raise SystemExit(main())
