"""NEW parent sharing exact V5 journal/EOF/source lifecycle, no frozen edits."""
from pathlib import Path
import qualify_batch_api_cache_positive_buffered_v5 as template
import qualify_batch_numerical_v7 as original
import full_cache_shared_runtime_v5 as ctrl
def adapted_source():
 source=template.adapted_source();needle='import batch_api_cache_positive_buffered_v5 as ctrl';ctrl.require(source.count(needle)==1,'Exact current strong lifecycle integration failed');source=source.replace(needle,'import full_cache_shared_runtime_v5 as ctrl')
 replacements={
  "a=ap.parse_args()":"ap.add_argument('--expected-plan-sha256');a=ap.parse_args()",
  "try:preflight_plan=read(a.plan);prepared_chain_binding(preflight_plan)":"try:snapshot=Snapshot(a.plan,a.expected_plan_sha256);preflight_plan=snapshot.plan;prepared_chain_binding(preflight_plan);snapshot.verify()",
  "*sys.argv[1:],'--leased']":"*sys.argv[1:],'--leased','--expected-plan-sha256',snapshot.sha256]",
  "child_dir=out/'child';plan=read(a.plan)":"child_dir=out/'child';snapshot.verify();plan=snapshot.plan",
  "'plan_sha256':sha(a.plan)":"'plan_sha256':snapshot.sha256",
  "write(out/'input-plan.snapshot.json',plan)":"snapshot.write(out/'input-plan.snapshot.json')",
  "'--plan',str(a.plan.resolve()),'--pre-health'":"'--plan',str(out/'input-plan.snapshot.json'),'--expected-plan-sha256',snapshot.sha256,'--pre-health'"}
 for before,after in replacements.items():
  ctrl.require(source.count(before)==1,'Exact source40 captured-plan integration differs '+before);source=source.replace(before,after)
 parent_record="parent={'schema':1,";ctrl.require(source.count(parent_record)==1,'Original actual parent process record point differs');source=source.replace(parent_record,"parent={'parent_pid':os.getpid(),'parent_start_ticks':parent_process_identity(os.getpid())['start_ticks'],'schema':1,")
 offer_point="   parent['child_launch_started_epoch']=time.time();save()\n";ctrl.require(source.count(offer_point)==1,'Exact source40 original launch bracket changed');source=source.replace(offer_point,"   from full_cache_shared_health_handoff_v5 import offer\n   parent['leaf_offer']=offer(out,snapshot.sha256,Path(__file__),Path(ctrl.__file__),min(a.max_runtime,1200),parent['started_epoch']);save()\n"+offer_point)
 poll='   while child.poll() is None:\n    if stopped[0] or time.monotonic()>=deadline:\n';ctrl.require(source.count(poll)==1,'Exact source40 child supervision loop changed')
 hook='''   while child.poll() is None:
    if (out/'leaf-ready.json').is_file() and not (out/'leaf-ack.json').exists() and not stop_sent and not stopped[0]:
     from full_cache_shared_health_handoff_v5 import start_ticks,admit_ready,digest,atomic,process,leases
     from serial37_canonical_json_v3 import read_unique
     ready=read_unique(out/'leaf-ready.json');ticks=start_ticks(child.pid);require(parent['leaf_offer']['parent']==process(os.getpid()) and parent['leaf_offer']['lease_identity']==leases(),'Actual parent/lease offer changed');admit_ready(ready,plan,parent['prepared_chain'],child.pid,ticks,parent['child_launch_started_epoch'],parent['leaf_offer']);parent['leaf_child_start_ticks']=ticks;parent['leaf_ready']=ready;save()
     health_path=health('leaf');require(parent['leaf_health_passed'],'Actual postsemantic strict/compiled health failed')
     row=command(['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],'leaf-kernel-journal',30);row['observed_epoch']=time.time();parent['leaf_kernel_journal']=row;write(out/'leaf-kernel-receipt.json',row);save();require(row['return_code']==0 and row['error'] is None and not FAULT.search(Path(row['path']).read_text()),'Actual postsemantic kernel journal failed')
     require(child.poll() is None and start_ticks(child.pid)==ticks and not stopped[0],'Actual owned waiting child retired or interrupted before ACK')
     ack={'parent':process(os.getpid()),'lease_identity':leases(),'offer_semantic_sha256':digest(parent['leaf_offer']),'schema':3,'kind':'source40_actual_parent_health_ACK','ready_sha256':digest(ready),'producer_pid':child.pid,'producer_start_ticks':ticks,'health_sha256':sha(health_path),'kernel_receipt_sha256':sha(out/'leaf-kernel-receipt.json'),'observed_epoch':time.time()};parent['leaf_ack']=ack;save();atomic(out/'leaf-ack.json',ack)
'''
 source=source.replace(poll,hook+'    if stopped[0] or time.monotonic()>=deadline:\n')
 kill='    if stop_sent and time.monotonic()-stop_at>180:\n';ctrl.require(source.count(kill)==1,'Exact parent active forced-kill boundary changed')
 source=source.replace(kill,"    critical=retirement_guard(child_dir)\n    if stop_sent and time.monotonic()-stop_at>180 and not parent.get('leaf_ack') and not critical.get('retirement_in_progress'):\n")
 final_kill="    if time.monotonic()-stop_at>180:child.kill();parent['errors'].append('Forced child terminal cleanup')";ctrl.require(source.count(final_kill)==1,'Exact final forced-kill boundary changed')
 source=source.replace(final_kill,"    critical=retirement_guard(child_dir)\n    if time.monotonic()-stop_at>180 and not parent.get('leaf_ack') and not critical.get('retirement_in_progress'):child.kill();parent['errors'].append('Forced child terminal cleanup')")
 return 'from full_cache_shared_plan_snapshot_v5 import Snapshot\nfrom full_cache_shared_actor_retirement_v5 import critical_guard as retirement_guard\nfrom full_cache_shared_health_handoff_v5 import process as parent_process_identity\n'+source
def main():
 ctrl.source_binding();source=adapted_source();namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='owned_full_cache_shared_parent_v1',ctrl=ctrl,BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__)));exec(compile(source,str(Path(original.__file__))+'[NEW-fullcache-current-journal-EOF]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
