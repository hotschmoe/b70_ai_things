"""One independent V6 pair-owned experiment; no matched-mate inference."""
import argparse
from pathlib import Path
import run_source52_p2048_profile_v6 as actor
import validate_source52_p2048_profile_v5 as common
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;HERE=actor.HERE;ROOT=actor.ROOT
PLAN=HERE/'source52-p2048-profile-reader-v6-source-plan-v1.json'
def source_binding():
 for name,want in c.read(PLAN)['files'].items():require(c.sha(ROOT/name)==want,'Single-arm reader source changed '+name)
 return c.sha(PLAN)
def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');phases=list(report['per_arm']);require(len(phases)==1,'Exactly one actual independently owned arm required');phase=phases[0];recipe=c.read(root/phase/'recipe.json');flag=recipe['single_arm_copy_engine'];require(type(flag)is int and flag in(0,1)and phase=='source52-copy%d-device-profile'%flag,'Actual single copy route/phase differs')
 before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Original single experiment artifact roster differs');old=(actor.COPY,actor.PHASES,actor.parent.PHASES,actor.NS['PHASES'])
 try:
  actor.COPY=flag;actor.PHASES=(phase,);actor.parent.PHASES=(phase,);actor.NS['PHASES']=(phase,)
  with common.configured_actor(report,recipe,actor)as active:
   require(active.PHASES==(phase,),'Private public reader must use the genuine single phase');result=boundary_reader_ports(active)['finalized_binding'](root);arm=common.arm_binding(root,report,root/phase,recipe,active);require(arm['copy_engine']==str(flag),'Actual single copy flag differs');comparison=active.contract.equivalence(arm['requests']);require(comparison['matched_trace_arms_qualified']is False and canonical(comparison)==canonical(report['H36_OFF_ON_comparison']),'Original warmup/selected-only comparison differs');proof=active.fresh.binding();require(actor.closure()==report['source_plan_sha256'],'Original single producer source closure differs')
   result.update(producer_admitted=True,reader_source_plan_sha256=source,producer_report_sha256=report_sha,actual52_SDK=proof,source_generation=52,independent_diagnostic_experiments=1,native_processes=1,requests_per_process=2,GEN_cap=8,arms=[arm],copy_engine=flag,original_individual_root_cap_bytes=1<<30,matched_trace_arms_qualified=False,H36_OFF_ON_equivalence_passed=False,H36_coverage_qualified=False,Chrome_CPU_host_lane_recorded=False,clock_alignment_qualified=False,complete_model_operation_coverage_qualified=False,physical_idle_qualified=False,physical_bus_bandwidth_qualified=False,clean_serving_speed_qualified=False,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False,inspector_execution_allowed=True);return result
 finally:
  actor.COPY,actor.PHASES,actor.parent.PHASES,actor.NS['PHASES']=old
  require(source_binding()==source and c.sha(root/'report.json')==report_sha and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing single-arm source/producer/raw roster changed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists()and a.root.resolve()not in a.output.resolve().parents,'New external single-arm public proof required');c.write(a.output,finalized_binding(a.root))
