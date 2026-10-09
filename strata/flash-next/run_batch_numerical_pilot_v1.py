#!/usr/bin/env python3
"""Actual native initial BGEN pilot; parent owns lease/health/stop/full source gate."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import layer0_numerical_qualification_v8 as numerical
from batch_numerical_protocol_v1 import NativeStream
from batch_numerical_prefixes_v1 import serial_jobs
from audit_batch_fidelity_coverage_v2 import coverage
from qualify_c1_serving_combined_v6 import validate_final_source_proof
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent


def require(ok,message):
 if not ok:raise ValueError(message)

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='ascii')

def set_arg(args,key,value):
 if key in args:args[args.index(key)+1]=str(value)
 else:args.extend([key,str(value)])

def prepare(a):
 c1=numerical.c1;require(a.cancel_index is None or 0<=a.cancel_index<a.slots,'Cancel target index outside requested slots');prepared=c1.validate_prepared(a.prepared);engine=Path(prepared['engine_receipt']).parent;numerical.engine_binding(engine,numerical.PLAN_SOURCE_SHA);proof=validate_final_source_proof(a.prepared,prepared);tokens=json.loads(a.tokens.read_text());require(len(tokens['warm'])==len(tokens['target'])==a.slots and a.slots in (2,4,6),'Exactly requested warm/target vectors required')
 for ids in tokens['warm']+tokens['target']:require(1<=len(ids)<=2048 and all(type(t) is int and 0<=t<248320 for t in ids),'Bounded token IDs required')
 require(all(x[0]!=y[0] for x in tokens['warm'] for y in tokens['target']),'Warm complete prefixes must be unrelated to target from token0')
 args=list(c1.read(a.prepared/'server-config.json')['args'])
 for key,value in [('--max-context',2048),('--prefill',64),('--batch',a.slots),('--batch-groups',1),('--prompt-cache',0),('--conversation-cache-mib',0),('--adapt-every',0)]:set_arg(args,key,value)
 require('--mtp' not in args and '--pipeline-windows' not in args,'No MTP/pipeline pilot');env=dict(c1.read(a.prepared/'server-config.json')['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='1',STRATA_BATCH_FIDELITY_ARM='/results/ARM',STRATA_BATCH_FIDELITY_DIR='/results/captures',SYCL_UR_TRACE='2')
 for key in ['STRATA_FIDELITY_DIAG','STRATA_LAYER0_Q8_DIAG','STRATA_VERIFY_EAGER']:env.pop(key,None)
 env['STRATA_BATCH_FULL_STATE_CHAIN']='0';env['STRATA_BATCH_PUBLIC_PREFIX']='0'
 plan={'schema':1,'status':'prepared initial native numerical pilot only; actual GPU/serial/API/cache/migration unqualified','prepared':str(a.prepared.resolve()),'prepared_sha256':c1.sha(a.prepared/'prepared.json'),'engine_root':str(engine),'engine_receipt_sha256':prepared['engine_receipt_sha256'],'source_proof':proof,'slots':a.slots,'cards':prepared['cards'],'args':args,'env':env,'image':prepared['runtime']['image'],'pack':prepared['pack'],'tokens':tokens,'tokens_sha256':c1.sha(a.tokens),'cancel_index':a.cancel_index,'max_new':32,'driver_sha256':c1.sha(Path(__file__)),'strict_unplanned_reuse_policy':'require actual everytarget RESUME/read_from/reread_to zero and complete perstage spans; cache0 does not implyfresh','full_model_math_qualified':False,'complete_cache_or_migration_qualified':False}
 a.output.mkdir(parents=True,exist_ok=False);write(a.output/'plan.json',plan);print('PREPARED native initial pilot; no GPU executed')

def run(a):
 c1=numerical.c1;c1.leased([0,1]);plan=c1.read(a.plan);require(plan['driver_sha256']==c1.sha(Path(__file__)),'Pilot source changed');prepared=c1.validate_prepared(Path(plan['prepared']));require(c1.sha(Path(plan['prepared'])/'prepared.json')==plan['prepared_sha256'] and prepared['engine_receipt_sha256']==plan['engine_receipt_sha256'],'Genuine preparation changed');validate_final_source_proof(Path(plan['prepared']),prepared);numerical.engine_binding(Path(plan['engine_root']),numerical.PLAN_SOURCE_SHA)
 health=c1.read(a.pre_health);require(health['passed'] and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh parent actual prehealth required')
 a.output.mkdir(parents=True,exist_ok=False);(a.output/'captures').mkdir();write(a.output/'plan.snapshot.json',plan);binding=c1.sha(a.output/'plan.snapshot.json');container='b70-batch-pilot-'+str(os.getpid());env=dict(plan['env']);env['STRATA_ARTIFACT_IDENTITY_SHA256']=binding;engine=Path(plan['engine_root']);model=ROOT/c1.read(HERE/'model-lock.json')['destination'];command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(model)+':/model:ro','-v',str(a.output.resolve())+':/results']
 for key,value in sorted(env.items()):command.extend(['-e',key+'='+str(value)])
 command.extend([plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+plan['args'])]);write(a.output/'command.json',command);stream=None;error=None;report=None
 try:
  stream=NativeStream(command,a.output,plan['slots']);warm=[1001+i for i in range(plan['slots'])]
  for i,rid in enumerate(warm):stream.submit(rid,i,plan['tokens']['warm'][i],32)
  stream.drain(warm);stream.arm(a.output/'ARM');targets=[2001+i for i in range(plan['slots'])]
  for i,rid in enumerate(targets):stream.submit(rid,i,plan['tokens']['target'][i],32)
  stream.drain(targets,cancel_rid=targets[plan['cancel_index']] if plan['cancel_index'] is not None else None);trace='\n'.join(stream.lines)
  stages=[(i,lo,hi) for i,(lo,hi) in enumerate(numerical.expected_stage_ranges(plan['args']))];raw=coverage(trace,targets,stages,a.output/'captures')
  require(any(int(e['rows'])==plan['slots'] for e in stream.roster.events[stream.arm_event_index:]),'Actual completed requested-N-row event absent')
  for line in trace.splitlines():
   if line.startswith('SBF resume '):
    f=dict(item.split('=',1) for item in line.split()[2:] if '=' in item)
    if int(f['rid']) in targets:require(all(int(f[k])==0 for k in ['reused','read_from','reread_to']),'Actual target reused unexpectedly; cache0 is not fresh proof')
  jobs=serial_jobs(trace,stream.roster);write(a.output/'serial-jobs.json',jobs);write(a.output/'requests.json',stream.roster.requests);report={'initial_native_raw_coverage_completed':True,'raw':raw,'serial_jobs':jobs,'full_model_math_qualified':False,'actual_serial_comparison_completed':False,'actual_cache_or_migration_qualified':False}
 except BaseException as exc:error=str(exc)
 finally:
  try:rc=stream.close() if stream else None
  except BaseException as exc:rc=None;error=(error+'; ' if error else '')+'close: '+str(exc)
  state=c1.inspected(container)['State'];removed=False
  if not state['Running']:subprocess.run(['docker','rm',container],capture_output=True,text=True,check=True);removed=c1.absent(container)
  result={'passed':False,'initial_native_collection_and_teardown_passed':error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed,'error':error,'report':report,'engine_rc':rc,'state':state,'removed':removed,'finished_epoch':time.time(),'plan_sha256':binding,'source_and_posthealth_fullhash_qualified':False,'scope':'parent must retain lease/watch2pages/stop/health/final4hash then actualserial/fullvectors; no serving/math claim'};write(a.output/'report.json',result)
 return 0 if result['initial_native_collection_and_teardown_passed'] else 1

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--tokens',type=Path,required=True);a.add_argument('--slots',type=int,choices=[2,4,6],required=True);a.add_argument('--cancel-index',type=int);a.add_argument('--output',type=Path,required=True);a=sub.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--pre-health',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a=p.parse_args();return prepare(a) if a.mode=='prepare' else run(a)
if __name__=='__main__':raise SystemExit(main())
