#!/usr/bin/env python3
"""Pair-leased parent lifecycle for bounded BATCH_NUMERICAL actual-process arms.
The research agent prepares this source CPU-only. Explicit execution acquires
both cards, performs health/model gates, supervises BATCH_NUMERICAL and retains failures.
"""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,threading
from pathlib import Path
import batch_numerical_execution_v2 as ctrl
from source_page_watchdog_v3 import preserve as preserve_source_pages
ROOT=Path(__file__).resolve().parents[2]
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__))
ENGINE_PLAN='strata/flash-next/integrated-slot-owner-trace-engine-build-plan-v1.json'
ENGINE_PLAN_SHA='94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094'
FAULT=re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed',re.I)
sha=ctrl.sha;read=ctrl.read;write=ctrl.write;require=ctrl.require

def stat_signature(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def full_buffered_identity(lock_path,lock,shards,output,after_epoch):
 """Read all four publisher shards with ordinary buffered reads; no mutation."""
 require(time.time()>=after_epoch,'Full identity scan started before child terminal bound')
 files=[f for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
 require(len(files)==len(shards)==4,'Complete four-shard source identity required')
 receipt={'schema':1,'passed':False,'lock_sha256':sha(lock_path),'model_revision':lock['revision'],'started':time.time(),'finished':None,'rows':[],'read_mode':'ordinary buffered read; no direct IO/cache mutation','after_child_terminal_epoch':after_epoch}
 write(output,receipt)
 for path,want in zip(shards,files):
  row={'path':str(path),'expected_sha256':want['sha256'],'bytes':0,'sha256':None,'stat_before':stat_signature(path),'stat_after':None,'passed':False}
  try:
   h=hashlib.sha256()
   with Path(path).open('rb') as handle:
    for block in iter(lambda:handle.read(8<<20),b''):h.update(block);row['bytes']+=len(block)
   row['sha256']=h.hexdigest();row['stat_after']=stat_signature(path)
   row['passed']=row['bytes']==want['size'] and row['sha256']==want['sha256'] and row['stat_before']==row['stat_after']
  except Exception as e:row['error']=str(e)
  receipt['rows'].append(row);write(output,receipt)
 receipt['finished']=time.time();receipt['passed']=all(r['passed'] for r in receipt['rows']) and len(receipt['rows'])==4
 write(output,receipt);return receipt

def finalizable(parent,child,post_hash):
 collected=child.get('collection_and_teardown_passed') is True or child.get('initial_native_collection_and_teardown_passed') is True
 return (parent.get('child_return_code')==0 and collected and not parent.get('interrupted') and parent.get('owned_containers_terminal') is True and parent.get('forced_cleanup') is False and parent.get('pre_health_passed') is True and parent.get('post_health_passed') is True and parent.get('kernel_fault_gate_passed') is True and not parent.get('errors') and post_hash.get('passed') is True and post_hash.get('started',0)>=max(child.get('finished_epoch',float('inf')),parent.get('post_health_finished_epoch',float('inf'))))

def source_plan_binding():
 require(sha(ROOT/ENGINE_PLAN)==ENGINE_PLAN_SHA,'Frozen numerical combined source plan differs')
 return read(ROOT/ENGINE_PLAN)

def prepared_chain_binding(plan):
 require(sha(Path(ctrl.__file__))==plan['driver_sha256']==BATCH_NUMERICAL_SHA,'Numerical driver/preparation differs')
 return ctrl.manifest_binding(plan)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--group',choices=['arm'],default='arm')
 ap.add_argument('--max-runtime',type=int,default=7200);ap.add_argument('--leased',action='store_true');a=ap.parse_args()
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 ctrl.c1.leased([0,1]);require(a.max_runtime>0,'Positive bounded child deadline required')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);child_dir=out/'child';plan=read(a.plan)
 parent={'schema':1,'passed':False,'started_epoch':time.time(),'cards_held':[0,1],'workload_cards':plan['cards'],'group':a.group,'expected_stage_ranges':ctrl.expected_stage_ranges(plan['args']),'plan':str(a.plan.resolve()),'plan_sha256':sha(a.plan),'wrapper_sha256':sha(Path(__file__)),'controller_sha256':sha(Path(ctrl.__file__)),'errors':[],'interrupted':False,'forced_cleanup':False,'owned_containers_terminal':False,'GPU_execution_scope':'one source29 native/API or bounded serial process arm; full2/4/6/cache/servingmirror/originalmath/latency require separate final joins'}
 (out/'wrapper.py').write_bytes(Path(__file__).read_bytes());(out/'controller.py').write_bytes(Path(ctrl.__file__).read_bytes());write(out/'input-plan.snapshot.json',plan)
 child=None;stopped=[False];stop_sent=False;child_terminal_epoch=None;health_invoked=False
 def save():write(out/'parent-qualification.json',parent)
 def stop(sig,frame):stopped[0]=True;parent['interrupted']=True;parent.setdefault('stop_signals',[]).append({'signal':sig,'epoch':time.time()});save()
 for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
 save()
 def command(cmd,label,timeout=210):
  write(out/(label+'.command.json'),cmd)
  print('PARENT stage '+label,flush=True)
  with (out/(label+'.log')).open('w') as log:
   proc=None;error=None;rc=None
   try:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
    rc=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    # Health shell traps own their containers; do not SIGKILL past those traps.
    proc.terminate();error='Command deadline exceeded; graceful terminal cleanup required'
    while proc.poll() is None:time.sleep(2)
    rc=proc.wait()
   except Exception as e:
    error=str(e)
    if proc is not None:
     proc.terminate()
     while proc.poll() is None:time.sleep(2)
     rc=proc.wait()
  return {'path':str(out/(label+'.log')),'sha256':sha(out/(label+'.log')),'command':cmd,'command_file_sha256':sha(out/(label+'.command.json')),'return_code':rc,'error':error,'supervisor_pid':proc.pid if proc else None}
 def health(stage):
  nonlocal health_invoked
  health_invoked=True;started=time.time();rows=[]
  strict=[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH]
  rows.append(command(strict,stage+'-strict',180))
  if rows[0]['return_code']==0 and rows[0]['error'] is None:
   rows.append(command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],stage+'-compiled-pair',210))
  else:rows.append({'return_code':None,'error':'Compiled pair skipped after failed strict per-card health; recovery required'})
  for row in rows:
   if not row.get('supervisor_pid'):continue
   pid=row['supervisor_pid'];strict_probe='strict' in Path(row['path']).name
   filters=['name=^/xpu-health-'+str(pid)+'-'] if strict_probe else ['name=^/xpu-collective-health-'+str(pid)+'$']
   while True:
    ids=subprocess.check_output(['docker','ps','-aq','--filter',filters[0]],text=True,timeout=30).split()
    if not ids:break
    row['error']='Health container survived supervisor; forced owned cleanup'
    for identity in ids:
     obj=json.loads(subprocess.check_output(['docker','inspect',identity],text=True,timeout=30))[0]
     expected_prefix='/xpu-health-'+str(pid)+'-' if strict_probe else '/xpu-collective-health-'+str(pid)
     require(obj['Name'].startswith(expected_prefix) and obj['Config']['Image']==HEALTH and (not strict_probe or obj['Config']['Labels'].get('b70.xpu-health','').startswith('xpu-health-'+str(pid)+'-')),'Health cleanup ownership differs')
     cleanup=command(['docker','rm','-f',identity],stage+'-health-owned-'+identity,90)
     if cleanup['return_code']!=0:time.sleep(2)
  receipt={'schema':1,'passed':all(r['return_code']==0 and r['error'] is None for r in rows),'cards':[0,1],'files':[r for r in rows if 'path' in r],'checks':rows,'started_epoch':started,'finished_epoch':time.time(),'health_image':HEALTH,'source_sha256':{str(p):sha(p) for p in [ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh',ROOT/'bin/xpu-collective-health',ROOT/'bin/xpu-collective-health.py']},'expected_workload_stage_ranges':parent['expected_stage_ranges']}
  write(out/(stage+'-health.json'),receipt);parent[stage+'_health_passed']=receipt['passed'];parent[stage+'_health_finished_epoch']=receipt['finished_epoch'];save();return out/(stage+'-health.json')
 def faults(label):
  row=command(['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],label,30)
  text=Path(row['path']).read_text(errors='replace');require(row['return_code']==0 and not FAULT.search(text),'Kernel journal unavailable or GPU fault signature')
  return row
 def containers():
  hashes={parent['plan_sha256'],sha(out/'input-plan.snapshot.json')}
  if (child_dir/'plan.snapshot.json').exists():hashes.add(sha(child_dir/'plan.snapshot.json'))
  found={}
  for digest in hashes:
   ids=subprocess.check_output(['docker','ps','-aq','--filter','label=b70.prefix.plan='+digest],text=True,timeout=30).split()
   for identity in ids:
    obj=json.loads(subprocess.check_output(['docker','inspect',identity],text=True,timeout=30))[0]
    require(obj['Config']['Labels'].get('b70.prefix.plan')==digest,'Container plan ownership changed')
    found[identity]=obj
  return found
 def source_watch(label):
  try:return ctrl.verify_model_identity(Path(plan['model_identity']['path']),lock,shards)
  except Exception:
   preserve_source_pages(shards[2],out,label)
   raise
 lock_path=ROOT/'strata/flash-next/model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/f['path'] for f in lock['files'] if f['path'].startswith('UD-Q4_K_XL/')]
 try:
  require(parent['controller_sha256']==plan['driver_sha256']==BATCH_NUMERICAL_SHA,'Immutable BATCH_NUMERICAL controller/source plan differs')
  parent['prepared_chain']=prepared_chain_binding(plan);save()
  require(sha(Path(plan['model_identity']['path']))==plan['model_identity']['sha256'],'Prepared complete model identity receipt changed')
  source_watch('pre');require(not containers(),'Pre-existing same-plan container; refusing duplicate ownership')
  pre=health('pre');require(parent['pre_health_passed'],'Pre-health gate failed');faults('pre-kernel-journal');source_watch('after-pre-health')
  require(not stopped[0],'Interrupted before child launch')
  cmd=[sys.executable,str(Path(ctrl.__file__)),'run','--plan',str(a.plan.resolve()),'--pre-health',str(pre),'--output',str(child_dir)]
  write(out/'child.command.json',cmd)
  with (out/'child-supervisor.log').open('w') as log:
   child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(8,9))
   def forward():
    for line in child.stdout:
     log.write(line);log.flush();print(line.rstrip('\n').encode('ascii','backslashreplace').decode('ascii'),flush=True)
   reader=threading.Thread(target=forward,daemon=True);reader.start()
   parent['child_pid']=child.pid;save();deadline=time.monotonic()+a.max_runtime;stop_at=None
   while child.poll() is None:
    if stopped[0] or time.monotonic()>=deadline:
     if not stop_sent:child.send_signal(signal.SIGTERM);stop_sent=True;stop_at=time.monotonic();parent['errors'].append('Child stop requested by parent interruption/deadline');save()
    elif not stop_sent:
     try:source_watch('during')
     except Exception as e:parent['errors'].append('Active model identity watch: '+str(e));child.send_signal(signal.SIGTERM);stop_sent=True;stop_at=time.monotonic();save()
    if stop_sent and time.monotonic()-stop_at>180:
     child.kill();parent['errors'].append('Child supervisor required SIGKILL; owned containers still require terminal proof');save()
    time.sleep(2)
   parent['child_return_code']=child.wait();reader.join(timeout=10);child_terminal_epoch=time.time();parent['child_terminal_epoch']=child_terminal_epoch;save()
   if child.returncode:parent['errors'].append('BATCH_NUMERICAL child failed '+str(child.returncode))
 except Exception as e:parent['errors'].append(str(e));save()
 finally:
  # Lease descriptors remain open until supervisor and owned containers are terminal.
  if child is not None and child.poll() is None:
   child.send_signal(signal.SIGTERM)
   while child.poll() is None:
    if not stop_sent:stop_sent=True;stop_at=time.monotonic()
    if time.monotonic()-stop_at>180:child.kill();parent['errors'].append('Forced child terminal cleanup')
    time.sleep(2)
   parent['child_return_code']=child.wait();reader.join(timeout=10);child_terminal_epoch=time.time();parent['child_terminal_epoch']=child_terminal_epoch
  foreign=False
  while True:
   try:
    live=containers()
    if live and child is None:
     parent['errors'].append('Pre-existing same-plan containers left untouched; no child owned by this run')
     parent['foreign_plan_containers']=list(live);foreign=True;parent['owned_containers_terminal']=True;break
    if not live:parent['owned_containers_terminal']=True;break
    for identity,obj in live.items():
     require(child is not None and obj['Name'].startswith('/b70-prefix-'+str(child.pid)+'-'),'Same-plan container belongs to another supervisor; lease retained')
     parent['forced_cleanup']=True;parent['errors'].append('Owned container survived child terminal: '+obj['Name']);save()
     with (out/(identity+'-container.log')).open('w') as log:subprocess.run(['docker','logs',identity],stdout=log,stderr=subprocess.STDOUT,timeout=30)
     write(out/(identity+'-state.json'),obj)
     row=command(['docker','rm','-f',identity],identity+'-remove',90);require(row['return_code']==0,'Owned container could not be removed')
   except Exception as e:parent['errors'].append('Cleanup retry: '+str(e));save();time.sleep(2)
  save()
  # Post-health and complete identity are retained even when the child failed.
  post=None
  try:
   require(not foreign,'Post-health skipped while pre-existing foreign model container remains live')
   post=health('post');faults('post-kernel-journal');parent['kernel_fault_gate_passed']=True
  except Exception as e:parent['errors'].append('Post-health/journal: '+str(e))
  child_report=read(child_dir/'report.json') if (child_dir/'report.json').exists() else {}
  after=max(child_terminal_epoch or time.time(),child_report.get('finished_epoch',0));identity=None
  try:
   identity=full_buffered_identity(lock_path,lock,shards,out/'post-model-identity.json',after)
   if not identity['passed']:parent['errors'].append('Complete post-run buffered model identity failed; no mutation/repair attempted')
   source_watch('post')
  except Exception as e:parent['errors'].append('Post-model identity scan: '+str(e))
  if identity and post and finalizable(parent,child_report,identity):
   parent['passed']=True
   parent['scoped_collection_or_serial_arm_qualified']=True
   parent['child_report_sha256']=sha(child_dir/'report.json')
   parent['post_model_identity']={'path':str(out/'post-model-identity.json'),'sha256':sha(out/'post-model-identity.json')}
   parent['source_guard_generation']=3
   parent['polling_continuous_source_identity_qualified']=False
   parent['full_model_math_qualified']=False
   parent['full_concurrent_cache_migration_serving_qualified']=False
   parent['serving_mirror_owner_logical_frees']='UNOBSERVED in source29'
  parent['finished_epoch']=time.time();save()
 print(json.dumps({'passed':parent['passed'],'receipt':str(out/'parent-qualification.json'),'child_report':str(child_dir/'report.json')}))
 return 0 if parent['passed'] else 1
if __name__=='__main__':sys.exit(main())
