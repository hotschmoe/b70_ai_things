#!/usr/bin/env python3
"""Owned synthetic HC35 runtime parent. Root execution only; no fullmodel claim.
The parent holds both-card leases for health. Its physical-card0 leaf inherits
those descriptors and pins ZE_AFFINITY_MASK=0; nested flock acquisition is avoided.
"""
import argparse,hashlib,json,os,re,signal,subprocess,sys,time,shlex,math
from pathlib import Path
import c1_serve_controller_combined_v13 as c1
import compile_hc_projection35_v1 as compiler
import prepare_hc_projection35_leaf_v1 as producer
import hc_projection_arithmetic35_fixture_v1 as fixture
from parse_usm_logical_free_trace import parse_trace,negative_controls
from qualify_serial_prefix_v7 import full_buffered_identity
from source_page_watchdog_v3 import preserve as preserve_source_pages
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
PLAN_SHA= 'dcf4bb0997bdc2a576c51e1656088910f8b7d145ef993b4bedf6b1c6201a6ed4'
FAULT=re.compile(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed',re.I)
sha,read,write,require=c1.sha,c1.read,c1.write,c1.require


def source_binding():
 plan=read(HERE/'hc-projection35-parent-source-plan-v1.json')
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Frozen parent dependency changed '+name)
 return plan['files']


def compile_binding(path):
 path=Path(path).resolve();receipt=read(path);plan=receipt['plan']
 require(receipt.get('passed') is True and receipt.get('errors')==[] and receipt.get('compile_return_code')==0 and receipt.get('post_source_unchanged') is True and receipt.get('container_removed') is True and receipt.get('actual_GPU_run') is False,'Fresh completed compile/owned cleanup prerequisite absent')
 state=receipt.get('container_terminal',{});require(state.get('Running') is False and state.get('ExitCode')==0 and not state.get('OOMKilled') and not state.get('Error'),'Compile container terminal failed')
 require(receipt['controller_sha256']==sha(compiler.__file__) and receipt['plan_sha256']==PLAN_SHA==compiler.PLAN_SHA and sha(path.parent/'plan.snapshot.json')==PLAN_SHA and sha(path.parent/'compile-controller.snapshot.py')==receipt['controller_sha256'],'Actual fresh leaf compiler/plan snapshot association differs')
 require(sha(HERE/'hc-projection35-leaf-compile-plan-v1.json')==PLAN_SHA and producer.prepare(Path(plan['engine_root']))==plan,'Current exact SDK/source/archive/eightELF/leaf inputs changed')
 binary=Path(receipt['binary']).resolve();require(binary==path.parent/'hc_projection_arithmetic35_gpu_v1' and sha(binary)==receipt['binary_sha256'],'Fresh leaf executable identity changed')
 with binary.open('rb') as stream:require(stream.read(4)==b'\x7fELF','Actual fresh leaf ELF required')
 expected=['docker','run','--name',receipt['container'],'--network','none','--user','1000:1000','--entrypoint','/bin/bash','--label','b70.hc35.compile='+PLAN_SHA,'-v',plan['engine_root']+':/sdk:ro','-v',str(Path(plan['leaf_source']).parent)+':/leaf:ro','-v',str(path.parent)+':/out',plan['image'],'-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1\nexec '+shlex.join(plan['compile_argv_inside_pinned_image'])]
 require(receipt.get('command')==expected,'Fresh compiler container command/entrypoint association differs')
 require(all(type(receipt.get(k)) in (int,float) and math.isfinite(receipt[k]) and receipt[k]>0 for k in ('started_epoch','finished_epoch')) and receipt['finished_epoch']>=receipt['started_epoch'],'Compile receipt chronology invalid')
 require(receipt['compile_log_sha256']==sha(path.parent/'compile.log'),'Fresh compile log changed')
 require(receipt['archive_association_limit']==plan['archive_association_limit'] and receipt['model_math_qualified'] is False,'Archive provenance/math scope changed')
 require(plan['environment']=={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_UR_TRACE':'2'} and plan['required_absent_environment']==['STRATA_VERIFY_EAGER'],'Exact physical-card0 normal graph leaf env differs')
 return receipt,{'compile_receipt':str(path),'compile_receipt_sha256':sha(path),'binary':str(binary),'binary_sha256':receipt['binary_sha256'],'plan_sha256':PLAN_SHA,'current_inputs':plan,'archive_association_limit':plan['archive_association_limit']}


def owned_container(obj,name,image,binding):
 return obj.get('Name')=='/'+name and obj.get('Config',{}).get('Image')==image and obj.get('Config',{}).get('Labels',{}).get('b70.hc35.runtime')==binding


def launch_command(receipt,out,name,binding,inputs):
 plan=receipt['plan'];env=dict(plan['environment']);require('STRATA_VERIFY_EAGER' not in env,'EAGER must be absent')
 command=['docker','run','--entrypoint','/bin/bash','--name',name,'--label','b70.hc35.runtime='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(Path(receipt['binary']).parent)+':/out:ro','-v',str(out)+':/results','-v',str(Path(inputs).resolve()/'inputs')+':/inputs:ro']
 for key,value in sorted(env.items()):command+=['-e',key+'='+value]
 command += [plan['image'],'-c','source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1\nif [ "${STRATA_VERIFY_EAGER+x}" ]; then printf "HC35_PARENT_ERROR inherited EAGER present\\n" >&2; exit 64; fi\nexec /out/hc_projection_arithmetic35_gpu_v1 --inputs /inputs --output /results/raw-new']
 return command


def raw_proofs(out,inputs):
 log=out/'leaf.log';raw=fixture.collect(inputs,out/'raw-new',log);trace=parse_trace(log.read_text(),False);trace.update(log_sha256=sha(log),control_smoke=True,lifecycle_qualified_by_this_receipt=False)
 if trace['passed']:trace['negative_controls']=negative_controls(log.read_text(),False);trace['passed']=all(trace['negative_controls'].values())
 require(len(raw['cases'])==13 and sum(row['words'] for row in raw['cases'])==38 and len(raw['negative_controls'])==2,'Complete raw13/38words and2 negatives required')
 write(out/'raw-component-proof.json',raw);write(out/'usm-logical-free-proof.json',trace)
 return {'numerical_and_teardown_passed':raw['numeric_bitwise_passed'] and trace['passed'],'synthetic_component_passed':raw['numeric_bitwise_passed'],'logical_free_passed':trace['passed'],'raw_proof_sha256':sha(out/'raw-component-proof.json'),'usm_proof_sha256':sha(out/'usm-logical-free-proof.json'),'full_model_math_qualified':False,'real_model_inputs_observed':False}


def finalizable(parent,proof,identity):
 return (parent.get('child_return_code')==0 and not parent.get('interrupted') and parent.get('owned_containers_terminal') is True and not parent.get('forced_cleanup') and parent.get('pre_health_passed') is True and parent.get('post_health_passed') is True and parent.get('kernel_fault_gate_passed') is True and parent.get('post_source_unchanged') is True and not parent.get('errors') and proof.get('numerical_and_teardown_passed') is True and identity.get('passed') is True and len(identity.get('rows',[]))==4 and identity.get('started',0)>=max(parent.get('child_terminal_epoch',float('inf')),parent.get('post_health_finished_epoch',float('inf'))))


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--compile-receipt',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-runtime',type=int,default=900);ap.add_argument('--leased',action='store_true');a=ap.parse_args()
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 c1.leased([0,1]);require(a.max_runtime>0,'Positive leaf deadline required');out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 parent={'schema':1,'passed':False,'started_epoch':time.time(),'cards_held':[0,1],'workload_cards':[0],'physical_leaf_card':0,'errors':[],'interrupted':False,'forced_cleanup':False,'owned_containers_terminal':False,'full_model_math_qualified':False,'real_model_inputs_observed':False,'scope':'Owned synthetic HC35 public-projection only; no fullHC/deviceintrinsic/model/fidelity/speed qualification'}
 (out/'parent.py').write_bytes(Path(__file__).read_bytes());child=None;leaf_started=False;stopped=[False];receipt=None;binding=None;name='b70-hc35-runtime-'+str(os.getpid());proof={};identity=None;health_invoked=False;post=None
 def save():write(out/'parent-qualification.json',parent)
 def stop(sig,frame):stopped[0]=True;parent['interrupted']=True;parent.setdefault('stop_signals',[]).append({'signal':sig,'epoch':time.time()});save()
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 save()
 def command(cmd,label,timeout=210):
  write(out/(label+'.command.json'),cmd);print('HC35_PARENT '+label,flush=True)
  with (out/(label+'.log')).open('w') as log:
   proc=None;error=None;rc=None
   try:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(8,9));rc=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    proc.terminate();error='Command deadline exceeded; owned terminal cleanup required'
    while proc.poll() is None:time.sleep(2)
    rc=proc.wait()
   except Exception as error0:
    error=str(error0)
    if proc is not None:
     proc.terminate()
     while proc.poll() is None:time.sleep(2)
     rc=proc.wait()
  return {'path':str(out/(label+'.log')),'sha256':sha(out/(label+'.log')),'command':cmd,'return_code':rc,'error':error,'supervisor_pid':proc.pid if proc else None}
 def health(stage):
  nonlocal health_invoked
  health_invoked=True;started=time.time();rows=[command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],stage+'-strict',180)]
  if rows[0]['return_code']==0 and rows[0]['error'] is None:rows.append(command([str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],stage+'-compiled-pair',210))
  else:rows.append({'return_code':None,'error':'Compiled pair skipped after failed strict percard health'})
  for row in rows:
   if not row.get('supervisor_pid'):continue
   pid=row['supervisor_pid'];strict='strict' in Path(row['path']).name;pattern='name=^/xpu-health-'+str(pid)+'-' if strict else 'name=^/xpu-collective-health-'+str(pid)+'$'
   while True:
    ids=subprocess.check_output(['docker','ps','-aq','--filter',pattern],text=True,timeout=30).split()
    if not ids:break
    row['error']='Health container survived supervisor; forced owned cleanup'
    for value in ids:
     obj=c1.inspected(value);expected='/xpu-health-'+str(pid)+'-' if strict else '/xpu-collective-health-'+str(pid);require(obj['Name'].startswith(expected) and obj['Config']['Image']==HEALTH and (not strict or obj['Config']['Labels'].get('b70.xpu-health','').startswith('xpu-health-'+str(pid)+'-')),'Health cleanup ownership differs');removed=command(['docker','rm','-f',value],stage+'-health-owned-'+value,90)
     if removed['return_code']!=0:time.sleep(2)
  h={'schema':1,'passed':all(row['return_code']==0 and row['error'] is None for row in rows),'cards':[0,1],'health_image':HEALTH,'files':[row for row in rows if 'path' in row],'started_epoch':started,'finished_epoch':time.time()};write(out/(stage+'-health.json'),h);parent[stage+'_health_passed']=h['passed'];parent[stage+'_health_finished_epoch']=h['finished_epoch'];save();return h
 def faults(stage):
  row=command(['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],stage+'-kernel-journal',30);require(row['return_code']==0 and not FAULT.search(Path(row['path']).read_text()),'Kernel journal unavailable or GPU fault signature');return row
 def watch(label):
  try:
   pages=preserve_source_pages(shards[2],out,label);require(pages['passed'],'Both known source pages failed');parent['page_guard_calls']=parent.get('page_guard_calls',0)+1;return pages
  except Exception:
   preserve_source_pages(shards[2],out,label+'-failure');raise
 def exists():return bool(subprocess.check_output(['docker','ps','-aq','--filter','name=^/'+name+'$'],text=True,timeout=30).split())
 lock_path=HERE/'model-lock.json';lock=read(lock_path);shards=[ROOT/lock['destination']/row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
 try:
  parent['source_dependencies']=source_binding();parent['input_manifest']=fixture.input_binding(a.inputs);parent['input_manifest_sha256']=sha(a.inputs/'manifest.json');receipt,binding=compile_binding(a.compile_receipt);parent['compile_binding']=binding;parent['archive_association_limit']=binding['archive_association_limit'];parent['container']=name;parent['image']=receipt['plan']['image'];save();require(not exists(),'Pre-existing named runtime container; untouched')
  watch('pre');pre=health('pre');require(pre['passed'],'Prehealth failed');faults('pre');watch('after-pre-health');require(not stopped[0],'Interrupted before leaf')
  c1.leased([0]);cmd=launch_command(receipt,out,name,binding['compile_receipt_sha256'],a.inputs);write(out/'leaf.command.json',cmd)
  with (out/'leaf.log').open('w') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(8,9));leaf_started=True;parent['child_pid']=child.pid;save();deadline=time.monotonic()+a.max_runtime;sent=False;sent_at=None
   while child.poll() is None:
    if stopped[0] or time.monotonic()>=deadline:
     if not sent:child.terminate();sent=True;sent_at=time.monotonic();parent['errors'].append('Leaf interrupted/deadline');save()
    if sent and time.monotonic()-sent_at>180:child.kill();parent['forced_cleanup']=True;parent['errors'].append('Leaf Docker supervisor needed SIGKILL')
    time.sleep(2)
   parent['child_return_code']=child.wait();parent['child_terminal_epoch']=time.time();save()
   if child.returncode:parent['errors'].append('Leaf exited nonzero '+str(child.returncode))
 except Exception as error:parent['errors'].append(str(error));save()
 finally:
  if child is not None and child.poll() is None:
   child.terminate();deadline=time.monotonic()+180
   while child.poll() is None:
    if time.monotonic()>deadline:child.kill();parent['forced_cleanup']=True
    time.sleep(2)
   parent['child_return_code']=child.wait();parent['child_terminal_epoch']=time.time()
  if leaf_started:
   while True:
    try:
     if not exists():
      parent['errors'].append('Owned container absent before terminal inspection; exit proof unavailable');parent['owned_containers_terminal']=True;break
     obj=c1.inspected(name);require(owned_container(obj,name,receipt['plan']['image'],binding['compile_receipt_sha256']),'Runtime container ownership differs; lease retained')
     if obj['State']['Running']:
      parent['forced_cleanup']=True;parent['errors'].append('Runtime survived supervisor; graceful owned stop');command(['docker','stop','--time','30',name],'leaf-owned-stop',60);continue
     parent['container_terminal']=obj['State'];require(obj['State']['ExitCode']==0 and not obj['State'].get('OOMKilled') and not obj['State'].get('Error'),'Runtime container nonzero/OOM/error')
     row=command(['docker','rm',name],'leaf-owned-remove',60);require(row['return_code']==0 and not exists(),'Owned terminal container removal failed');parent['owned_containers_terminal']=True;break
    except Exception as error:
     parent['errors'].append('Owned cleanup: '+str(error));save()
     if 'container_terminal' in parent:
      if exists():command(['docker','rm',name],'leaf-failed-owned-remove',60)
      if not exists():parent['owned_containers_terminal']=True;break
     time.sleep(2)
  else:parent['owned_containers_terminal']=not exists()
  save()
  if receipt is not None:
   try:
    require(parent['owned_containers_terminal'],'Posthealth refused while named foreign/unretired leaf remains');post=health('post');faults('post');parent['kernel_fault_gate_passed']=True
   except Exception as error:parent['errors'].append('Posthealth/journal: '+str(error))
   try:
    require(parent['owned_containers_terminal'],'Postsource refused while named foreign/unretired leaf remains');after=max(parent.get('child_terminal_epoch',time.time()),post['finished_epoch'] if post else time.time());parent['known_pages_before_hash']=watch('before-full4');identity=full_buffered_identity(lock_path,lock,shards,out/'post-model-identity.json',after);parent['known_pages_after_hash']=watch('after-full4');require(identity['passed'],'New postterminal/posthealth full4 scan failed');parent['post_model_identity_sha256']=sha(out/'post-model-identity.json')
   except Exception as error:parent['errors'].append('Postsource4: '+str(error))
   try:
    require(fixture.input_binding(a.inputs)==parent['input_manifest'] and sha(a.inputs/'manifest.json')==parent['input_manifest_sha256'],'Synthetic input/host admission changed during runtime');require(source_binding()==parent['source_dependencies'],'Runtime parent dependencies changed');_,final_binding=compile_binding(a.compile_receipt);require(final_binding==binding,'SDK/archive/source/ELF/leaf/binary changed during run');parent['post_source_unchanged']=True
   except Exception as error:parent['errors'].append('Final source: '+str(error))
   if leaf_started:
    try:proof=raw_proofs(out,a.inputs);parent['component_proof']=proof;require(proof['numerical_and_teardown_passed'],'Synthetic numerical/free failure retained')
    except Exception as error:parent['errors'].append('Raw component/USM proof: '+str(error))
  parent['passed']=bool(identity and finalizable(parent,proof,identity));parent['finished_epoch']=time.time();save()
 print(json.dumps({'passed':parent['passed'],'receipt':str(out/'parent-qualification.json'),'synthetic_component_passed':proof.get('synthetic_component_passed'),'full_model_math_qualified':False}));return 0 if parent['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
