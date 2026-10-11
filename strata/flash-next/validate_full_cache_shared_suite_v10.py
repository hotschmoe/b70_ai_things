"""ROOT READONLY original whole-suite scheduling and closed family receipts."""
import math,stat
from pathlib import Path
import full_cache_shared_runtime_v10 as ctrl
from full_cache_shared_plan_snapshot_v10 import Snapshot
from full_cache_shared_suite_v10 import declaration,finalized_binding
from serial37_canonical_json_v3 import canonical

def receipt_binding(receipt,command,log,parent):
 ctrl.require(receipt['command']==command and type(receipt['producer_pid'])is int and receipt['producer_pid']>0 and type(receipt['return_code'])is int and receipt['return_code']==0 and receipt['interruption_signals']==[] and receipt['actual_child_terminal_and_file_EOF'] is True,'Exact originally owned nested command and normal EOF required')
 ctrl.require(type(parent['parent_pid'])is int and type(parent['parent_start_ticks'])is int and type(receipt['producer_identity'])is dict and type(receipt['producer_identity']['pid'])is int and type(receipt['producer_identity']['start_ticks'])is int and receipt['identity_error'] is None and receipt['producer_pid']==parent['parent_pid'] and receipt['producer_identity']=={'pid':parent['parent_pid'],'start_ticks':parent['parent_start_ticks']},'Original nested Popen must be the exact admitted parent process/incarnation')
 epochs=[receipt['started_epoch'],parent['started_epoch'],parent['finished_epoch'],receipt['finished_epoch']]
 ctrl.require(all(type(v)in(int,float) and math.isfinite(v)for v in epochs) and epochs==sorted(epochs),'Actual nested parent start/terminal must be bracketed by original Popen receipt')
 path=Path(log);ctrl.require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode) and ctrl.sha(path)==receipt['stdout_sha256'],'Original normal parent stdout changed')
 return True

def finalized_suite_binding(root):
 root=Path(root).resolve();report=ctrl.read(root/'suite-report.json');snapshot=Snapshot(root/'suite-plan.snapshot.json');plan=snapshot.plan;before=ctrl.source_binding()
 ctrl.require(ctrl.sha(Path(report['producer_interpreter']).resolve())==report['producer_interpreter_sha256'],'Original suite producer interpreter bytes changed')
 paths={str(p.relative_to(root)) for p in root.rglob('*')if p.is_file() and p.name!='suite-report.json'};ctrl.require(paths==set(report['artifact_bindings']),'Exact original suite artifact roster differs')
 for name,binding in report['artifact_bindings'].items():
  path=root/name;ctrl.require(not path.is_symlink() and path.resolve().is_relative_to(root) and path.stat().st_size==binding['bytes'] and ctrl.sha(path)==binding['sha256'],'Original suite artifact bytes/extent changed')
 ctrl.require(report['passed'] is True and report['all_declared_families_completed'] is True and report['error'] is None and report['unexecuted_declared']==[] and report['full_cache_runtime_qualified'] is False,'Complete original suite scheduling required; no whole-cache promotion')
 ctrl.require(plan['source_binding']==before and plan['declaration']==declaration(ctrl.authentic_case()),'Original current complete suite source/declaration differs')
 actual=report['actual_family_manifest'];normalized=ctrl.read(root/'suite-plan.snapshot.json');ctrl.require(canonical(report['source_binding_after'])==canonical(before),'Original end source proof changed')
 expected_names=[r['name']for r in plan['actor_roots']]+['persisted'];ctrl.require([r['name']for r in report['rows']]==expected_names,'Every original actor scheduling receipt required exactly once')
 for index,item in enumerate(plan['actor_roots']):
  current=actual['actor_roots'][index];ctrl.require(type(current['fresh_control_roots'])is list and current['fresh_control_roots'],'Actual complete independently owned fresh roots required');copy=dict(current);copy.pop('fresh_control_roots');ctrl.require(canonical(copy)==canonical(item),'Only actual fresh control roots can extend declared actor view')
  normalized['actor_roots'][index]['fresh_control_roots']=current['fresh_control_roots']
 normalized['serial_persisted_fresh_roots']=actual['serial_persisted_fresh_roots'];ctrl.require(canonical(normalized)==canonical(actual),'Unexpected suite derivative fields or original declaration mutation')
 for index,name in enumerate(expected_names):
  serial=name=='persisted';item=plan['actor_roots'][index] if not serial else {'plan':plan['serial_persisted_plan'],'plan_sha256':plan['serial_persisted_plan_sha256'],'root':plan['serial_persisted_root']}
  actor_snapshot=Snapshot(item['plan'],item['plan_sha256']);parent=ctrl.read(Path(item['root'])/'parent-qualification.json');ctrl.require(ctrl.sha(Path(item['root'])/'input-plan.snapshot.json')==actor_snapshot.sha256,'Originally declared nested actor plan snapshot differs')
  wrapper='qualify_full_cache_shared_batch0_persisted_v10.py' if serial else 'qualify_full_cache_shared_runtime_v10.py';receipt=ctrl.read(root/(name+'.parent.receipt.json'));command=[report['producer_interpreter'],str(ctrl.HERE/wrapper),'--plan',str(actor_snapshot.path),'--output',item['root'],'--expected-plan-sha256',actor_snapshot.sha256,'--leased'];receipt_binding(receipt,command,root/(name+'.parent.log'),parent);ctrl.require(canonical(receipt)==canonical(report['rows'][index]['original_actor_parent']),'Original individual suite command receipt differs')
  controls=actual['serial_persisted_fresh_roots'] if serial else actual['actor_roots'][index]['fresh_control_roots'];ctrl.require(controls==report['rows'][index]['actual_fresh_control_roots'],'Original fresh actor scheduling list differs')
  for group,control in enumerate(controls):
   path=Path(control);fresh_plan=path.with_name(path.name+'.plan.json');fresh_snapshot=Snapshot(fresh_plan);fresh_parent=ctrl.read(path/'parent-qualification.json');ctrl.require(ctrl.sha(path/'input-plan.snapshot.json')==fresh_snapshot.sha256,'Actual independently prepared fresh input snapshot differs');wrapper='qualify_full_cache_shared_batch0_fresh40_v10.py' if serial else 'qualify_full_cache_shared_fresh40_v10.py';command=[report['producer_interpreter'],str(ctrl.HERE/wrapper),'--plan',str(fresh_snapshot.path),'--output',str(path),'--expected-plan-sha256',fresh_snapshot.sha256,'--leased'];receipt_binding(ctrl.read(root/(name+'.fresh'+str(group)+'.receipt.json')),command,root/(name+'.fresh'+str(group)+'.log'),fresh_parent)
 proof=finalized_binding(actual);ctrl.require(canonical(proof)==canonical(report['closed_family_binding']),'Actual full family recollection changed');snapshot.verify();ctrl.require(ctrl.source_binding()==before,'Current source changed during complete suite reader')
 for name,binding in report['artifact_bindings'].items():ctrl.require(ctrl.sha(root/name)==binding['sha256'],'Original suite artifact changed during full family reader')
 return {'closed_family_binding':proof,'original_complete_scheduling_recollected':True,'full_cache_runtime_qualified':False}
