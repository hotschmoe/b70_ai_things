"""NEW parent sharing exact V5 journal/EOF/source lifecycle, no frozen edits."""
from pathlib import Path
import qualify_batch_api_cache_positive_buffered_v5 as template
import qualify_batch_numerical_v7 as original
import full_cache_shared_runtime_v2 as ctrl
def adapted_source():
 source=template.adapted_source();needle='import batch_api_cache_positive_buffered_v5 as ctrl';ctrl.require(source.count(needle)==1,'Exact current strong lifecycle integration failed');source=source.replace(needle,'import full_cache_shared_runtime_v2 as ctrl')
 replacements={
  "a=ap.parse_args()":"ap.add_argument('--expected-plan-sha256');a=ap.parse_args()",
  "try:preflight_plan=read(a.plan);prepared_chain_binding(preflight_plan)":"try:snapshot=Snapshot(a.plan,a.expected_plan_sha256);preflight_plan=snapshot.plan;prepared_chain_binding(preflight_plan);snapshot.verify()",
  "*sys.argv[1:],'--leased']":"*sys.argv[1:],'--leased','--expected-plan-sha256',snapshot.sha256]",
  "child_dir=out/'child';plan=read(a.plan)":"child_dir=out/'child';snapshot.verify();plan=snapshot.plan",
  "'plan_sha256':sha(a.plan)":"'plan_sha256':snapshot.sha256",
  "write(out/'input-plan.snapshot.json',plan)":"snapshot.write(out/'input-plan.snapshot.json')",
  "'--plan',str(a.plan.resolve()),'--pre-health'":"'--plan',str(out/'input-plan.snapshot.json'),'--expected-plan-sha256',snapshot.sha256,'--pre-health'"}
 for before,after in replacements.items():
  ctrl.require(source.count(before)==1,'Exact source39 captured-plan integration differs '+before);source=source.replace(before,after)
 return 'from full_cache_shared_plan_snapshot_v2 import Snapshot\n'+source
def main():
 ctrl.source_binding();source=adapted_source();namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='owned_full_cache_shared_parent_v1',ctrl=ctrl,BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__)));exec(compile(source,str(Path(original.__file__))+'[NEW-fullcache-current-journal-EOF]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
