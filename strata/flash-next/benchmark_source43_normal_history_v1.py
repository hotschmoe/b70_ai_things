#!/usr/bin/env python3
"""ROOT readonly startup comparison, no model/GPU latency qualification."""
import argparse,hashlib,json,os,signal,statistics,time
from pathlib import Path
import source43_normal_historical_association_v1 as history
from serial37_canonical_json_v3 import canonical,read_unique
HERE=Path(__file__).resolve().parent;PLAN=HERE/'source43-normal-history-benchmark-source-plan-v1.json'
require=history.require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
 before=sha(PLAN)
 for name,want in read_unique(PLAN)['files'].items():require(sha(HERE.parents[1]/name)==want,'Benchmark source changed '+name)
 require(sha(PLAN)==before,'Benchmark sourceplan changed');return before

def current_contract(normal_plan,association):
 import source43_sdk_binding_v1 as sdk
 plan=read_unique(normal_plan);actual=sdk.binding();require(canonical(actual)==canonical(plan['source43']),'Actual current43 SDK differs');path=Path(plan['template_path'])/'server-config.json';require(sha(path)==plan['template_config_sha256'],'Actual current configuration changed');cfg=read_unique(path);match=history.match_current_contract(association,actual,cfg);return {'actual_current_sdk':actual,'config_path':str(path),'config_sha256':sha(path),'normal_intake_sha256':sha(normal_plan),'matched_fields':match,'per_arm_full4_pages_health_and_retirement_still_required':True}

def no_inherited_GPU_leases():
 for fd in Path('/proc/self/fd').iterdir():
  try:target=os.readlink(fd)
  except OSError:continue
  require(target not in('/mnt/vm_8tb/b70/gpu.lock.0','/mnt/vm_8tb/b70/gpu.lock.1'),'CPU comparison must run outside inherited GPU leases')

def run(output,normal_plan,blocks=1,original_call=None,association_call=None,current_call=None):
 no_inherited_GPU_leases();require(type(blocks)is int and 1<=blocks<=2,'Bounded preregistered1/2 ABBA blocks required');output=Path(output);require(not output.exists(),'New benchmark output required');output.mkdir();source=sources();report={'schema':1,'passed':False,'errors':[],'trials':[],'started_epoch':time.time(),'source_plan_sha256':source,'GPU_execution':False,'model_GPU_or_serving_latency_qualified':False,'A_and_B_current_identity_authority_equal':False,'comparison_scope':'historical prerequisite replacement; current SDK/config checked independently, fresh caller/per-arm model gates never waived'}
 def write(): (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii')
 original_call=original_call or(lambda:__import__('validate_source43_normal_model_cli_v1').finalized_binding(history.NORMAL));association_call=association_call or history.binding;current_call=current_call or(lambda a:current_contract(normal_plan,a));write()
 try:
  prepared=association_call();before=current_call(prepared);report['current_contract_before']=before
  for block in range(blocks):
   for arm in('A','B','B','A'):
    index=len(report['trials']);row={'index':index,'block':block,'arm':arm,'started_epoch':time.time(),'elapsed_s':None,'passed':False,'error':None};start=time.monotonic()
    try:
     actual=original_call()if arm=='A'else association_call();row['elapsed_s']=time.monotonic()-start
     projected=actual if arm=='A'else actual['original_public_admission'];require(canonical(projected)==canonical(prepared['original_public_admission']),'Exact original public result equality differs')
     if arm=='B':require(actual['current_model_identity_qualified']is False and actual['current_public_full4_rehashed']is False and actual['current_GPU_health_transferred']is False,'Historical branch cannot acquire current authority')
     path=output/('trial-'+str(index)+'.json');path.write_text(json.dumps(actual,indent=2,ensure_ascii=True,allow_nan=False)+'\n',encoding='ascii');row.update(result_path=str(path),result_sha256=sha(path),original_result_equal=True,passed=True)
    except BaseException as exc:row['elapsed_s']=time.monotonic()-start;row['error']=type(exc).__name__+': '+str(exc);raise
    finally:row['finished_epoch']=time.time();report['trials'].append(row);write()
  after=current_call(association_call());require(canonical(before)==canonical(after),'Independently current SDK/config changed across benchmark');report['current_contract_after']=after;report['median_A_s']=statistics.median(r['elapsed_s']for r in report['trials']if r['arm']=='A');report['median_B_s']=statistics.median(r['elapsed_s']for r in report['trials']if r['arm']=='B');report['all_original_public_results_equal']=True
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:require(sources()==source,'Benchmark source changed after calls')
  except BaseException as exc:report['errors'].append('closing source: '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors']and len(report['trials'])==4*blocks and all(r['passed']for r in report['trials']);write()
 require(report['passed'],'Benchmark failed; actual attempts retained');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--normal-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--blocks',type=int,default=1);a=p.parse_args();run(a.output,a.normal_plan,a.blocks)
