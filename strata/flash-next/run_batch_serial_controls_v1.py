#!/usr/bin/env python3
"""Run actual selected consumed prefixes through same-engine serial13 controls."""
import argparse,json,os,shlex,subprocess,time
from pathlib import Path
import layer0_numerical_qualification_v8 as numerical
from merged_numerical_protocol_v2 import MergedProtocol
from batch_numerical_prefixes_v1 import compare_all
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def run(plan,batch_output,output,pre_health):
 c1=numerical.c1;c1.leased([0,1]);health=c1.read(pre_health);c1.require(health['passed'] and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh actual parent prehealth required');prepared=c1.validate_prepared(Path(plan['prepared']));numerical.validate_final_source_proof(Path(plan['prepared']),prepared);numerical.engine_binding(Path(plan['engine_root']),numerical.PLAN_SOURCE_SHA);jobs=c1.read(batch_output/'serial-jobs.json')['jobs'];c1.require(1<=len(jobs)<=18,'Bounded raw jobs required');output.mkdir(parents=True,exist_ok=False);serial={};error=None
 for group,start in enumerate(range(0,len(jobs),6)):
  directory=output/('serial-'+str(group));directory.mkdir();(directory/'captures').mkdir();(directory/'ARM').write_text('ARM serial comparison\n');args=list(plan['args']);args[args.index('--batch')+1]='0';env=dict(plan['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG='0');engine=Path(plan['engine_root']);binding=c1.sha(batch_output/'plan.snapshot.json');container='b70-batch-serial-'+str(os.getpid())+'-'+str(group);command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(ROOT/c1.read(HERE/'model-lock.json')['destination'])+':/model:ro','-v',str(directory.resolve())+':/results']
  for k,v in sorted(env.items()):command.extend(['-e',k+'='+str(v)])
  command.extend([plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]);c1.write(directory/'command.json',command);protocol=None;rows=[]
  try:
   protocol=MergedProtocol(command,directory)
   for job in jobs[start:start+6]:
    raw=protocol.request('serial-'+str(job['rid'])+'-'+job['role'],job['ids'],fresh=1,max_new=1);meta=numerical.base.extract(raw,directory/'captures',True,True,numerical.expected_stage_ranges(args));key=(job['rid'],job['role']);serial[key,-1]=meta['logits'][0]['path']
    for field in meta['residuals']:serial[key,int(field['layer'])]=field['path']
    rows.append({'job':job,'raw':raw,'meta':meta});c1.write(directory/'requests.json',rows)
  except BaseException as exc:error=str(exc)
  finally:
   try:rc=protocol.close() if protocol else None
   except BaseException as exc:rc=None;error=(error+'; ' if error else '')+str(exc)
   state=c1.inspected(container)['State'];removed=False
   if not state['Running']:subprocess.run(['docker','rm',container],capture_output=True,text=True,check=True);removed=c1.absent(container)
   terminal=error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed;c1.write(directory/'result.json',{'passed':terminal,'engine_rc':rc,'state':state,'removed':removed,'error':error});c1.require(terminal,'Actual serial control/teardown failed')
 # Paths and full roster from the source producer, not a hash-only inferred head.
 import re
 batch={}
 for line in (batch_output/'engine.combined.log').read_text().splitlines():
  if not line.startswith('SBF vector '):continue
  f=dict(re.findall(r'(\w+)=([^ ]+)',line));role='admission' if f['phase'].startswith('admission_') else 'later' if f['phase'].startswith('batch_step_') else 'solo_migration';key=((int(f['rid']),role),int(f['layer']));c1.require(key not in batch,'Duplicate batch vector');batch[key]=str((batch_output/'captures'/Path(f['file']).name).resolve())
 result=compare_all(batch,serial);result.update(source_and_final_posthealth_fullhash_qualified=False,natural_API_completion_or_latency_qualified=False);c1.write(output/'serial-comparison.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--batch-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pre-health',type=Path,required=True);a=p.parse_args();result=run(numerical.read(a.plan),a.batch_output,a.output,a.pre_health);print(json.dumps({'supplied_prefix_native_serial_equal':result['passed'],'full_model_math_qualified':False}));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
