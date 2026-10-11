"""ROOT-only two independently loaded live-identity actors plus fresh full49."""
import argparse,os,sys
from pathlib import Path
import live_cache_identity_controller_v2 as ctrl
from full_cache_shared_plan_snapshot_v9 import Snapshot
from run_full_cache_shared_suite_v9 import child

def prepare(args):
 ctrl.source_binding();out=args.output.resolve();ctrl.require(not out.exists(),'Fresh preregistered two-incarnation identity suite required');out.mkdir();rows=[]
 for index in range(2):
  directory=out/('incarnation'+str(index));ctrl.prepare(argparse.Namespace(prepared=args.prepared,model_identity=args.model_identity,port=args.port+index,output=directory));snapshot=Snapshot(directory/'plan.json');rows.append({'name':'incarnation'+str(index),'plan':str(snapshot.path),'plan_sha256':snapshot.sha256})
 plan={'schema':'live-identity-two-incarnation-v1','source':ctrl.source_binding(),'actors':rows,'native_state_between_actors_allowed':False,'same_original_weights_required':True,'actual_other_weights_loaded':False,'changed_model_full_cache_lifecycle_qualified':False};ctrl.write(out/'suite-plan.json',plan);return plan

def run(path,out,expected=None,leased=False):
 snapshot=Snapshot(path,expected);plan=snapshot.plan;ctrl.require(plan['schema']=='live-identity-two-incarnation-v1'and plan['source']==ctrl.source_binding()and [r['name']for r in plan['actors']]==['incarnation0','incarnation1']and plan['native_state_between_actors_allowed']is False,'Exact two independently loaded identity actors required')
 if not leased:os.execv(str(ctrl.ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,'run','--plan',str(snapshot.path),'--expected-plan-sha256',snapshot.sha256,'--output',str(out),'--leased'])
 ctrl.c1.leased([0,1]);out=Path(out);ctrl.require(not out.exists(),'Fresh actual identity suite output required');out.mkdir();snapshot.write(out/'suite-plan.snapshot.json');report={'passed':False,'actors':[],'actual_other_weights_loaded':False,'changed_model_full_cache_lifecycle_qualified':False,'error':None}
 try:
  import live_cache_identity_fresh40_v2 as fresh
  from validate_live_cache_identity_v2 import finalized_binding
  for row in plan['actors']:
   snapshot.verify();actor_plan=Snapshot(row['plan'],row['plan_sha256']);ctrl.manifest_binding(actor_plan.plan);actor=out/row['name'];command=[sys.executable,str(ctrl.HERE/'qualify_live_cache_identity_v2.py'),'--plan',str(actor_plan.path),'--expected-plan-sha256',actor_plan.sha256,'--output',str(actor),'--leased'];receipt=child(command,out/(row['name']+'.log'));ctrl.write(out/(row['name']+'.receipt.json'),receipt);ctrl.require(receipt['return_code']==0 and receipt['interruption_signals']==[],'Actual old identity actor failed; no later incarnation')
   original=finalized_binding(actor);jobs=fresh.job_roster(fresh.actual_groups(actor));controls=[]
   for index in range(len(jobs['groups'])):
    control_plan=out/(row['name']+'-fresh'+str(index)+'.plan.json');fresh.prepare(argparse.Namespace(cached_parent=actor,group_index=index,output=control_plan));captured=Snapshot(control_plan);control=out/(row['name']+'-fresh'+str(index));cmd=[sys.executable,str(ctrl.HERE/'qualify_live_cache_identity_fresh40_v2.py'),'--plan',str(captured.path),'--expected-plan-sha256',captured.sha256,'--output',str(control),'--leased'];r=child(cmd,out/(row['name']+'-fresh'+str(index)+'.log'));ctrl.write(out/(row['name']+'-fresh'+str(index)+'.receipt.json'),r);ctrl.require(r['return_code']==0 and r['interruption_signals']==[],'Every independently owned fresh49 control required');controls.append(str(control))
   admitted=finalized_binding(actor,fresh_control_roots=controls);ctrl.require(admitted['all_supplied_comparisons_bitwise_equal']and not admitted['fresh_control_parent_and_source_binding_unobserved'],'Actual complete raw49 equality against independent fresh controls required');report['actors'].append({'root':str(actor),'original_actor':receipt,'fresh_control_roots':controls,'closed_binding':admitted})
  owners=[ctrl.read(Path(r['root'])/'child/live-identity-owner.json')for r in report['actors']];reports=[ctrl.read(Path(r['root'])/'child/report.json')for r in report['actors']];ctrl.require(reports[0]['producer_pid']!=reports[1]['producer_pid']and owners[0]['namespace']==owners[1]['namespace'],'Independent frontend incarnations with same locked original identity required');report['passed']=True
 except BaseException as error:report['error']=type(error).__name__+': '+str(error)
 finally:snapshot.verify();report['source_after']=ctrl.source_binding();ctrl.write(out/'suite-report.json',report)
 return report

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--port',type=int,default=28900);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--expected-plan-sha256',required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--leased',action='store_true');a=p.parse_args()
 if a.mode=='prepare':prepare(a);return 0
 return int(not run(a.plan,a.output,a.expected_plan_sha256,a.leased)['passed'])
if __name__=='__main__':raise SystemExit(main())
