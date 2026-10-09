#!/usr/bin/env python3
"""Pair-leased no-weight owner oracle lifecycle. Preparation does not execute GPU."""
import argparse,hashlib,json,os,re,shlex,signal,subprocess,sys,time
from pathlib import Path
import c1_serve_controller as lifecycle
from collect_stage_slot_owner_oracle import collect
ROOT=Path(__file__).resolve().parents[2]
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
PLAN_SHA='6d05aa945891bfcac9f5f6c223029a2e302874e2b8f9b45f0605dedfc765e5f7'
FAULT=re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed',re.I)
sha=lifecycle.sha;read=lifecycle.read;write=lifecycle.write;require=lifecycle.require

def cases(plan):
 expected=[('owner_card0','0',['0:48:0'],82),('owner_card1','1',['0:48:0'],82),('owner_pair32_16','0,1',['0:32:0','32:48:1'],92)]
 got=[(r['id'],r['mask'],r['stages'],r['expected_owner_registrations']) for r in plan['cases']]
 require(got==expected,'Frozen case/device/range/owner roster differs')
 require(plan['bounds']['cells']==64 and plan['bounds']['slots_per_stage']==3 and plan['bounds']['live_device_arena_bytes_max']==1<<30,'Bounded oracle geometry differs')
 return plan['cases']

def validate_build(receipt_path,plan_path):
 r=read(receipt_path);plan=read(plan_path);root=receipt_path.parent;engine_path=Path(r['engine_receipt']);engine=read(engine_path)
 require(r.get('passed') and r.get('build_rc')==0 and r.get('libraries_unchanged') and r.get('devices_exposed') is False and r['image']==IMAGE,'Passing actual no-device oracle build required')
 require(sha(plan_path)==sha(root/'plan.snapshot.json')==r['plan_sha256']==PLAN_SHA,'Frozen oracle plan fingerprint differs')
 require(sha(root/'oracle.cpp')==sha(ROOT/plan['oracle_source'])==r['oracle_source_sha256']==plan['oracle_source_sha256'],'Compiled oracle source differs')
 require(sha(root/'stage-slot-owner-oracle')==r['binary_sha256'],'Oracle binary changed')
 require(sha(engine_path)==r['engine_receipt_sha256'] and engine.get('build_rc')==0 and engine.get('plan_snapshot_unchanged') and engine.get('external_source_unchanged'),'Actual combined engine receipt changed/incomplete')
 require(engine['image']==IMAGE and engine['source_revision']=='fb58e0dbc8399662c0e47c76578c6e878b14f6cf' and engine['ggml_revision']=='3cf03257f219afbe7334045ff7c6a06ac68c627d','Engine dependency/image/source pin differs')
 require(sha(Path(engine['plan_snapshot']))==engine['plan_sha256']==plan['engine_plan_sha256']==sha(ROOT/plan['engine_plan']),'Exact fully linked20 recipe required')
 require(sha(Path(engine['build'])/'CMakeCache.txt')==engine['cmake_cache_sha256'],'Compiled configuration changed')
 for path,h in engine['binary_sha256'].items():require(sha(path)==h,'Engine executable changed')
 expected=read(ROOT/plan['engine_plan'])['expected_patched_source_sha256']
 for rel,h in expected.items():require(sha(Path(engine['source_copy'])/rel)==h==engine['patched_source_sha256'].get(rel),'Consumed20 source differs: '+rel)
 for path,h in r['library_sha256'].items():require(sha(path)==h,'Linked ABI library changed')
 for key in ['collector','builder','logical_free_parser']:
  require(sha(ROOT/plan[key])==plan[key+'_sha256'],'Pinned '+key+' changed')
 cases(plan)
 return r,plan

def can_pass(receipt):
 return (receipt.get('pre_health_passed') is True and receipt.get('post_health_passed') is True and
         receipt.get('kernel_fault_gate_passed') is True and receipt.get('owned_containers_terminal') is True and
         not receipt.get('interrupted') and not receipt.get('errors') and not receipt.get('forced_cleanup') and
         [r['case'] for r in receipt.get('cases',[])]==['owner_card0','owner_card1','owner_pair32_16'] and
         all(r.get('passed') and r['terminal']['ExitCode']==0 and not r['terminal'].get('OOMKilled') and not r['terminal']['Running'] and r.get('removed') for r in receipt['cases']))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oracle-receipt',type=Path,required=True);p.add_argument('--plan',type=Path,default=ROOT/'strata/flash-next/stage-slot-owner-oracle-plan.json');p.add_argument('--output',type=Path,required=True);p.add_argument('--case-timeout',type=int,default=240);p.add_argument('--sentinel-shard',type=Path,help='Optional buffered observation only; no model identity claim');p.add_argument('--leased',action='store_true');a=p.parse_args()
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 lifecycle.leased([0,1]);require(a.case_timeout>0,'Positive case deadline required')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);receipt={'schema':1,'passed':False,'started_epoch':time.time(),'held_cards':[0,1],'cases':[],'errors':[],'interrupted':False,'forced_cleanup':False,'owned_containers_terminal':False,'controller_sha256':sha(Path(__file__)),'scope':'No-weight slot allocation/host staging/context/rollback/resize/private-state/probe and logical frees; not model bytes/math/concurrency/graph retirement'}
 (out/'controller.py').write_bytes(Path(__file__).read_bytes());active=None;stopped=[False];health_started=False
 def save():write(out/'receipt.json',receipt)
 def stop(sig,frame):stopped[0]=True;receipt['interrupted']=True;receipt.setdefault('signals',[]).append({'signal':sig,'epoch':time.time()});save()
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
 save()
 def command(cmd,label,timeout=210):
  write(out/(label+'.command.json'),cmd);print('OWNER stage '+label,flush=True)
  with (out/(label+'.log')).open('w') as log:
   proc=None;error=None;rc=None
   try:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(8,9));rc=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    proc.terminate();error='Deadline exceeded; await cleanup traps'
    while proc.poll() is None:time.sleep(2)
    rc=proc.wait()
   except Exception as e:
    error=str(e)
    if proc:
     proc.terminate()
     while proc.poll() is None:time.sleep(2)
     rc=proc.wait()
  return {'path':str(out/(label+'.log')),'sha256':sha(out/(label+'.log')),'command':cmd,'command_file_sha256':sha(out/(label+'.command.json')),'return_code':rc,'error':error,'supervisor_pid':proc.pid if proc else None}
 def health(stage):
  nonlocal health_started
  health_started=True;started=time.time();rows=[command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],stage+'-strict',180)]
  if rows[0]['return_code']==0 and rows[0]['error'] is None:rows.append(command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],stage+'-compiled-pair',210))
  else:rows.append({'return_code':None,'error':'Compiled pair skipped after failed strict per-card health'})
  for row in rows:
   if not row.get('supervisor_pid'):continue
   pid=row['supervisor_pid'];strict='strict' in Path(row['path']).name;filter_value='name=^/xpu-health-'+str(pid)+'-' if strict else 'name=^/xpu-collective-health-'+str(pid)+'$'
   while True:
    ids=subprocess.check_output(['docker','ps','-aq','--filter',filter_value],text=True,timeout=30).split()
    if not ids:break
    row['error']='Owned health container survived supervisor'
    for identity in ids:
     obj=lifecycle.inspected(identity);prefix='/xpu-health-'+str(pid)+'-' if strict else '/xpu-collective-health-'+str(pid)
     require(obj['Name'].startswith(prefix) and obj['Config']['Image']==HEALTH and (not strict or obj['Config']['Labels'].get('b70.xpu-health','').startswith('xpu-health-'+str(pid)+'-')),'Health cleanup ownership differs')
     clean=command(['docker','rm','-f',identity],stage+'-health-remove-'+identity,90)
     if clean['return_code']!=0:time.sleep(2)
  r={'schema':1,'passed':all(x['return_code']==0 and x['error'] is None for x in rows),'cards':[0,1],'files':[x for x in rows if 'path' in x],'checks':rows,'image':HEALTH,'started_epoch':started,'finished_epoch':time.time(),'source_sha256':{str(path):sha(path) for path in [ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh',ROOT/'bin/xpu-collective-health',ROOT/'bin/xpu-collective-health.py']}}
  write(out/(stage+'-health.json'),r);receipt[stage+'_health_passed']=r['passed'];save();require(r['passed'],stage+' strict+compiled health failed')
 def faults(label):
  r=command(['journalctl','-k','--since','@'+str(int(receipt['started_epoch'])),'--no-pager'],label,30);require(r['return_code']==0 and not FAULT.search(Path(r['path']).read_text(errors='replace')),'Kernel journal unavailable or GPU fault signature')
 def sentinel(label):
  if not a.sentinel_shard:return
  with a.sentinel_shard.open('rb') as f:f.seek(3857879040);data=f.read(4096)
  value=hashlib.sha256(data).hexdigest();write(out/(label+'-sentinel.json'),{'path':str(a.sentinel_shard),'offset':3857879040,'bytes':len(data),'sha256':value,'scope':'Optional buffered recurrence observation; not oracle weight identity'})
  require(len(data)==4096 and value=='2780fef9ce50fa1acbd4bdbf6c311b847395571fcc5e6ddb55898841fbcee90e','Optional known page sentinel changed')
 def capture(name,label):
  obj=lifecycle.inspected(name);require(obj['Config']['Labels'].get('b70.slot-owner.parent')==str(os.getpid()) and obj['Config']['Labels'].get('b70.slot-owner.plan')==PLAN_SHA,'Oracle container ownership differs')
  write(out/(label+'-container.json'),obj)
  with (out/(label+'.log')).open('w') as log:subprocess.run(['docker','logs',name],stdout=log,stderr=subprocess.STDOUT,timeout=30)
  return obj['State']
 def cleanup():
  nonlocal active
  while active:
   try:
    ids=subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+active+'$'],text=True,timeout=30).split()
    if not ids:active=None;break
    label=active.rsplit('-',1)[-1];state=capture(active,'cleanup-'+label)
    if state['Running']:receipt['forced_cleanup']=True;receipt['errors'].append('Owned oracle required termination');save()
    r=command(['docker','rm','-f',active],active+'-remove',90);require(r['return_code']==0,'Owned container removal failed')
   except Exception as e:receipt['errors'].append('Cleanup retry: '+str(e));save();time.sleep(2)
  receipt['owned_containers_terminal']=True;save()
 try:
  built,plan=validate_build(a.oracle_receipt,a.plan);receipt.update(oracle_receipt=str(a.oracle_receipt.resolve()),oracle_receipt_sha256=sha(a.oracle_receipt),engine_receipt=built['engine_receipt'],engine_receipt_sha256=built['engine_receipt_sha256'],oracle_plan_sha256=PLAN_SHA,expected_cases=[x['id'] for x in cases(plan)],collector_sha256=plan['collector_sha256'],logical_free_parser_sha256=plan['logical_free_parser_sha256']);save();health('pre');faults('pre-kernel-journal')
  for profile in cases(plan):
   require(not stopped[0],'Interrupted before case');validate_build(a.oracle_receipt,a.plan);sentinel(profile['id']+'-pre');label=profile['id'];active='slot-owner-'+str(os.getpid())+'-'+label;receipt['owned_containers_terminal']=False;save()
   args=['/oracle/stage-slot-owner-oracle','--cells',str(plan['bounds']['cells']),'--output','/results/'+label+'.json']
   for stage in profile['stages']:args += ['--stage',stage]
   cmd=['docker','run','-d','--name',active,'--label','b70.slot-owner.parent='+str(os.getpid()),'--label','b70.slot-owner.plan='+PLAN_SHA,'--network','none','--device','/dev/dri','--user','1000:1000','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'--group-add',str(os.stat('/dev/dri/card0').st_gid),'--memory','4g','--memory-swap','4g','-v',str(a.oracle_receipt.parent.resolve())+':/oracle:ro','-v',str(out)+':/results','-e','ZE_AFFINITY_MASK='+profile['mask']]
   for k,v in sorted(plan['runtime_environment'].items()):cmd += ['-e',k+'='+v]
   cmd += [IMAGE,'exec '+shlex.join(args)];r=command(cmd,label+'-launch',60);require(r['return_code']==0 and not r['error'],'Oracle launch failed');deadline=time.monotonic()+a.case_timeout
   while True:
    state=lifecycle.inspected(active)['State']
    if not state['Running']:break
    require(not stopped[0] and time.monotonic()<deadline,'Oracle interrupted/deadline exceeded');time.sleep(2)
   state=capture(active,label);case={'case':label,'terminal':state,'passed':False,'removed':False};receipt['cases'].append(case);save()
   require(state['ExitCode']==0 and not state.get('OOMKilled') and not state['Running'],'Oracle did not exit normally')
   result=collect(out/(label+'.json'),out/(label+'.log'),a.plan,label);write(out/(label+'-logical-free.json'),result);case.update(raw=read(out/(label+'.json')),logical_free=result,passed=result['passed']);save();require(result['passed'],'Raw/trace logical owner gate failed')
   sentinel(label+'-post');faults(label+'-kernel-journal');cleanup();case['removed']=True;save()
 except Exception as e:receipt['errors'].append({'type':type(e).__name__,'message':str(e)});save()
 finally:
  cleanup()
  if health_started:
   try:health('post');faults('post-kernel-journal');receipt['kernel_fault_gate_passed']=True
   except Exception as e:receipt['errors'].append('Post-health/journal: '+str(e))
  receipt['passed']=can_pass(receipt);receipt['finished_epoch']=time.time();save()
 print(json.dumps({'passed':receipt['passed'],'receipt':str(out/'receipt.json')}));return 0 if receipt['passed'] else 1
if __name__=='__main__':sys.exit(main())
