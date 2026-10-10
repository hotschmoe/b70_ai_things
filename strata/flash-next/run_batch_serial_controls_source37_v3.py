#!/usr/bin/env python3
"""Run actual selected consumed prefixes through same-engine serial13 controls."""
import argparse,json,os,shlex,subprocess,time,traceback
from pathlib import Path
import layer0_numerical_qualification_v9 as numerical
from extract_serial_cacheoff_numerical_v1 import extract
from run_batch_serial_cacheoff_v8 import CacheOffMergedProtocol
from serial_prefix_qualification_v8 import expected_stage_ranges
from serial37_selection_v3 import selected_jobs
from merged_numerical_protocol_v2 import MergedProtocol
from batch_numerical_prefixes_v2 import compare_all
from batch_numerical_proofs_v40 import genuine_baseline,engine_binding,providers,source_observers_off
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def run(plan,batch_output,output,pre_health,group_index):
 source_observers_off(plan['env'])
 c1=providers(plan['lane'])[0];c1.leased([0,1]);health=c1.read(pre_health);c1.require(health['passed'] and set(plan['cards'])<=set(health['cards']) and 0<=time.time()-health['finished_epoch']<=300,'Fresh actual parent prehealth required');prepared,binding=genuine_baseline(Path(plan['prepared']),plan['lane'],plan.get('baseline_adjudication_receipt'));engine_binding(Path(plan['engine_root']),plan['lane']);jobs=c1.read(batch_output/'serial-jobs.json')['jobs'];c1.require(1<=len(jobs)<=18,'Bounded raw jobs required');output.mkdir(parents=True,exist_ok=False);c1.write(output/'plan.snapshot.json',plan);serial={};error=None
 require_group=group_index in range((len(jobs)+5)//6);c1.require(require_group,'Actual bounded serial group absent')
 for group,start in [(group_index,group_index*6)]:
  directory=output/('serial-'+str(group));directory.mkdir();(directory/'captures').mkdir();(directory/'ARM').write_text('ARM serial comparison\n');args=list(plan['args']);args[args.index('--batch')+1]='0';env=dict(plan['env']);env.update(STRATA_BATCH_FIDELITY_DIAG='0',STRATA_FIDELITY_DIAG='1',STRATA_FIDELITY_DIAG_ACTIVATIONS='1',STRATA_FIDELITY_DIAG_ARM='/results/ARM',STRATA_FIDELITY_DIAG_DIR='/results/captures',STRATA_PREFIX_DIAG='1',STRATA_PREFIX_LIFECYCLE_DIAG='1',STRATA_PREFIX_DIAG_ARM='/results/ARM',STRATA_LAYER0_Q8_DIAG='0');engine=Path(plan['engine_root']);binding=c1.sha(output/'plan.snapshot.json');container='b70-prefix-'+str(os.getpid())+'-serial'+str(group);command=['docker','run','-i','--name',container,'--label','b70.prefix.plan='+binding,'--network','none','--device','/dev/dri','--user','1000:1000','--memory','105g','--memory-swap','105g','--group-add',str(os.stat('/dev/dri/renderD128').st_gid),'-v',str(engine/'build')+':/build:ro','-v',str(engine/'source')+':/src:ro','-v',plan['pack']+':/pack:ro','-v',str(ROOT/c1.read(HERE/'model-lock.json')['destination'])+':/model:ro','-v',str(directory.resolve())+':/results']
  for k,v in sorted(env.items()):command.extend(['-e',k+'='+str(v)])
  command.extend([plan['image'],'exec 2>&1; cd /src; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; exec '+shlex.join(['/build/strata','--serve']+args)]);c1.write(directory/'command.json',command);protocol=None;rows=[]
  try:
   protocol=CacheOffMergedProtocol(command,directory)
   for job in selected_jobs(plan,jobs):
    raw=protocol.request('serial-'+str(job['rid'])+'-'+job['role'],job['ids'],fresh=1,max_new=1);meta=extract(raw,directory/'captures',args,env,expected_stage_ranges(args));key=(job['rid'],job['role']);serial[key,-1]=meta['logits'][0]['path']
    for field in meta['residuals']:serial[key,int(field['layer'])]=field['path']
    rows.append({'job':job,'raw':raw,'meta':meta});c1.write(directory/'requests.json',rows)
  except BaseException as exc:error=type(exc).__name__+': '+str(exc);failure_traceback=traceback.format_exc()
  finally:
   try:rc=protocol.close() if protocol else None
   except BaseException as exc:rc=None;error=(error+'; ' if error else '')+str(exc)
   state=c1.inspected(container)['State'];removed=False
   if not state['Running']:subprocess.run(['docker','rm',container],capture_output=True,text=True,check=True);removed=c1.absent(container)
   terminal=error is None and rc==0 and state['ExitCode']==0 and not state.get('OOMKilled') and removed;c1.write(directory/'result.json',{'passed':terminal,'engine_rc':rc,'state':state,'removed':removed,'error':error,'failure_traceback':locals().get('failure_traceback')});c1.require(terminal,'Actual serial control/teardown failed')
 # Paths and full roster from the source producer, not a hash-only inferred head.
 import re
 batch={}
 for line in (batch_output/'engine.combined.log').read_text().splitlines():
  if not line.startswith('SBF vector '):continue
  f=dict(re.findall(r'(\w+)=([^ ]+)',line));role='admission' if f['phase'].startswith('admission_') else 'later' if f['phase'].startswith('batch_step_') else 'solo_migration';key=((int(f['rid']),role),int(f['layer']));c1.require(key not in batch,'Duplicate batch vector');batch[key]=str((batch_output/'captures'/Path(f['file']).name).resolve())
 group_keys={((j['rid'],j['role']),l) for j in selected_jobs(plan,jobs) for l in range(-1,48)};c1.require(set(serial)==group_keys,'Actual bounded group serial roster differs');result=compare_all({k:v for k,v in batch.items() if k in group_keys},serial);result['serial_vectors']={str(k):v for k,v in serial.items()};result['group']=group_index;result['groups_required']=(len(jobs)+5)//6;result['finished_epoch']=time.time();result.update(source_and_final_posthealth_fullhash_qualified=False,natural_API_completion_or_latency_qualified=False);c1.write(output/'serial-comparison.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--group-index',type=int,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--batch-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pre-health',type=Path,required=True);a=p.parse_args();result=run(numerical.read(a.plan),a.batch_output,a.output,a.pre_health,a.group_index);print(json.dumps({'supplied_prefix_native_serial_equal':result['passed'],'full_model_math_qualified':False}));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
