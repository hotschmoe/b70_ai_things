"""ROOT ONLY complete declared suite scheduling, never partial full-cache PASS."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import full_cache_shared_runtime_v8 as ctrl
import full_cache_shared_batch0_persisted_v8 as persisted
from full_cache_shared_suite_v8 import declaration,finalized_binding
from full_cache_shared_plan_snapshot_v8 import Snapshot
from serial37_canonical_json_v3 import canonical


def prepare(args):
 ctrl.source_binding();case=ctrl.authentic_case();decl=declaration(case);out=args.output.resolve();ctrl.require(not out.exists(),'Fresh complete suite plan directory required');out.mkdir();(out/'plans').mkdir();actors=[]
 for index,row in enumerate(decl['actors']):
  directory=out/'plans'/row['name'];a=argparse.Namespace(scenario=row['scenario'],capture_pass=row['capture_pass'],prepared=args.prepared,model_identity=args.model_identity,port=args.port+index,output=directory);ctrl.prepare(a);path=directory/'plan.json';snapshot=Snapshot(path);actors.append({'name':row['name'],'plan':str(snapshot.path),'plan_sha256':snapshot.sha256,'root':str(out/'actors'/row['name']),'fresh_control_directory':str(out/'fresh49'/row['name']),'fresh_group_policy':'all_actual_consumed_prefix_groups_of_six','cache_state_from_other_actor_allowed':False})
 serial=out/'plans/persisted.json';persisted.prepare(argparse.Namespace(prepared=args.prepared,model_identity=args.model_identity,port=args.port+len(actors),output=serial));plan={'schema':'source40-full-cache-suite-v3','engine_plan_sha256':ctrl.ENGINE_SHA,'source_binding':ctrl.source_binding(),'declaration':decl,'actor_roots':actors,'serial_persisted_root':str(out/'actors/persisted'),'serial_persisted_plan':str(serial),'serial_persisted_plan_sha256':ctrl.sha(serial),'serial_persisted_fresh_directory':str(out/'fresh49/persisted'),'full_cache_runtime_qualified':False};ctrl.write(out/'suite-plan.json',plan);return plan

def child(command,out):
 """Owned Popen remains under pair lease until actual child+EOF are joined."""
 out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);started=time.time();stop=[];previous={};proc=None
 def interrupted(number,frame):stop.append(number)
 for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous[number]=signal.signal(number,interrupted)
 try:
  with out.open('w') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(8,9));sent=False
   from full_cache_shared_health_handoff_v8 import process
   try:identity=process(proc.pid);identity_error=None
   except BaseException as error:identity=None;identity_error=type(error).__name__+': '+str(error)
   while proc.poll() is None:
    if stop and not sent:proc.send_signal(signal.SIGTERM);sent=True
    # Never SIGKILL across a nested owned launch/retirement. The nested parent
    # retains exclusion and resolves its original containers/child session.
    time.sleep(.1)
   rc=proc.wait()
  return {'command':command,'producer_pid':proc.pid,'producer_identity':identity,'identity_error':identity_error,'started_epoch':started,'finished_epoch':time.time(),'return_code':rc,'interruption_signals':stop,'stdout_sha256':ctrl.sha(out),'actual_child_terminal_and_file_EOF':True}
 finally:
  if proc is not None and proc.poll() is None:
   proc.send_signal(signal.SIGTERM)
   while proc.poll() is None:time.sleep(.1)
   proc.wait()
  for number,handler in previous.items():signal.signal(number,handler)

def run(plan_path,output,leased=False,expected_plan_sha256=None):
 admitted=Snapshot(plan_path,expected_plan_sha256);plan=admitted.plan;ctrl.require(plan['schema']=='source40-full-cache-suite-v3' and plan['declaration']==declaration(ctrl.authentic_case()) and plan['source_binding']==ctrl.source_binding(),'Whole original source/family suite plan required')
 if not leased:os.execv(str(ctrl.ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,'run','--plan',str(admitted.path),'--output',str(output),'--expected-plan-sha256',admitted.sha256,'--leased'])
 ctrl.c1.leased([0,1]);out=Path(output);ctrl.require(not out.exists(),'Fresh suite execution receipt root required');out.mkdir();admitted.write(out/'suite-plan.snapshot.json');report={'passed':False,'all_declared_families_completed':False,'rows':[],'unexecuted_declared':[],'started_epoch':time.time(),'producer_pid':os.getpid(),'producer_interpreter':sys.executable,'producer_interpreter_sha256':ctrl.sha(Path(sys.executable).resolve()),'error':None,'full_cache_runtime_qualified':False};actual=json.loads(json.dumps(plan))
 try:
  for item in actual['actor_roots']+ [{'name':'persisted','plan':actual['serial_persisted_plan'],'plan_sha256':actual['serial_persisted_plan_sha256'],'root':actual['serial_persisted_root'],'fresh_control_directory':actual['serial_persisted_fresh_directory']}]:
   admitted.verify();actor_snapshot=Snapshot(item['plan'],item['plan_sha256']);path=actor_snapshot.path;is_serial=item['name']=='persisted';module=persisted if is_serial else ctrl;wrapper='qualify_full_cache_shared_batch0_persisted_v8.py' if is_serial else 'qualify_full_cache_shared_runtime_v8.py';module.manifest_binding(actor_snapshot.plan);actor_snapshot.verify();command=[sys.executable,str(ctrl.HERE/wrapper),'--plan',str(path),'--output',item['root'],'--expected-plan-sha256',actor_snapshot.sha256,'--leased'];receipt=child(command,out/(item['name']+'.parent.log'));ctrl.write(out/(item['name']+'.parent.receipt.json'),receipt);ctrl.require(receipt['return_code']==0 and receipt['interruption_signals']==[],'Original actor family did not finish normally');from validate_full_cache_shared_suite_v8 import receipt_binding
   receipt_binding(receipt,command,out/(item['name']+'.parent.log'),ctrl.read(Path(item['root'])/'parent-qualification.json'));fresh=__import__('full_cache_shared_batch0_fresh40_v8' if is_serial else 'full_cache_shared_fresh40_v8');groups=fresh.job_roster(fresh.actual_groups(Path(item['root'])))['groups'];controls=[]
   for index in range(len(groups)):
    fresh_dir=Path(item['fresh_control_directory']);fresh_dir.mkdir(parents=True,exist_ok=True);fresh_plan=fresh_dir/('group'+str(index)+'.plan.json');fresh.prepare(argparse.Namespace(cached_parent=Path(item['root']),group_index=index,output=fresh_plan));fresh_snapshot=Snapshot(fresh_plan);root=fresh_dir/('group'+str(index));fresh_wrapper='qualify_full_cache_shared_batch0_fresh40_v8.py' if is_serial else 'qualify_full_cache_shared_fresh40_v8.py';command=[sys.executable,str(ctrl.HERE/fresh_wrapper),'--plan',str(fresh_plan),'--output',str(root),'--expected-plan-sha256',fresh_snapshot.sha256,'--leased'];fresh_snapshot.verify();r=child(command,out/(item['name']+'.fresh'+str(index)+'.log'));ctrl.write(out/(item['name']+'.fresh'+str(index)+'.receipt.json'),r);ctrl.require(r['return_code']==0 and r['interruption_signals']==[],'Every independently owned fresh49 group required');receipt_binding(r,command,out/(item['name']+'.fresh'+str(index)+'.log'),ctrl.read(root/'parent-qualification.json'));controls.append(str(root))
   if is_serial:actual['serial_persisted_fresh_roots']=controls
   else:item['fresh_control_roots']=controls
   report['rows'].append({'name':item['name'],'original_actor_parent':receipt,'actual_fresh_control_roots':controls});ctrl.write(out/'progress.json',report)
  report['closed_family_binding']=finalized_binding(actual);report['all_declared_families_completed']=True;report['passed']=True
 except BaseException as error:report['error']=type(error).__name__+': '+str(error)
 finally:
  completed={r['name'] for r in report['rows']};report['unexecuted_declared']=[r['name'] for r in plan['actor_roots'] if r['name']not in completed]+([] if 'persisted' in completed else ['persisted']);report['finished_epoch']=time.time();report['actual_family_manifest']=actual;report['source_binding_after']=ctrl.source_binding();admitted.verify();report['artifact_bindings']={str(p.relative_to(out)):{'sha256':ctrl.sha(p),'bytes':p.stat().st_size}for p in out.rglob('*')if p.is_file() and p.name!='suite-report.json'};ctrl.write(out/'suite-report.json',report)
 return report

def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True);a=s.add_parser('prepare');a.add_argument('--prepared',type=Path,required=True);a.add_argument('--model-identity',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--port',type=int,default=28800);a=s.add_parser('run');a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--expected-plan-sha256');a.add_argument('--leased',action='store_true');a=p.parse_args()
 if a.mode=='prepare':prepare(a);return 0
 return int(not run(a.plan,a.output,a.leased,a.expected_plan_sha256)['passed'])
if __name__=='__main__':raise SystemExit(main())
