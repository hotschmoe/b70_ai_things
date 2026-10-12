"""Saved experiment identity reader; safe47 OFF-only construction by default.
Reusable boundary port for future matched controls after actual SDK admission.
No extra whole-model reread, original full-math or speed authority.
"""
import ast,copy
from pathlib import Path
import run_source51_esimd_f16_model_v1 as actor
import validate_source43_bounded_pti_v1 as original
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require

from validate_source47_grouped_f16_model_v2 import boundary_reader_ports

def finalized_binding(root):
 root=Path(root);report=c.read(root/'report.json');actor.set_sdk(report['SDK51_root'],report['SDK51_receipt_sha256']);actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=c.read(root/actor.PHASES[0]/'recipe.json')['tool_runtime'];actor.set_runtime(saved['intake_path'],saved['exact_intake_sha256'])
 result=boundary_reader_ports(actor)['finalized_binding'](root)
 arms=[];observations=[]
 for index,phase in enumerate(actor.PHASES):
  directory=root/phase;recipe=c.read(directory/'recipe.json');require(recipe['bounded_window_schema']==2 and recipe['selected_request_count']==2 and recipe['maximum_total_output_tokens']==128 and all(r['max_new']==64 for r in recipe['requests'])and recipe['engine_generation']==51 and recipe['native_env']['STRATA_VERIFY_MIRROR_STAGE_T1']=='0'and recipe['native_env']['STRATA_PF_ESIMD_F16']==str(index)and recipe['native_argv'][recipe['native_argv'].index('--spec')+1]=='2'and recipe['native_argv'][recipe['native_argv'].index('--pipeline-windows')+1]=='0' and recipe['materialized_mode']==0 and recipe['native_env']['STRATA_PF_GROUPED_F16']=='0'and recipe['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and recipe['native_argv'][recipe['native_argv'].index('--prefill')+1]=='64','Exact safe native47 groupedOFF and traceOFF required');rows=c.read(directory/'requests.json');require(len(rows)==2 and all(len(row['raw']['ids'])==524 for row in rows),'Two fresh long524 GEN64 requests required');views=actor.contract.requests(rows,recipe['requests']);require(canonical(views[0])==canonical(views[1]),'Warmup/selected output repeats differ');arms.append(views)
  require(not list((directory/'host-traces').glob('*')),'H36 disabled but emitted trace');observations.append({'phase':phase,'requests':[{'ordinal':row['ordinal'],'native_DONE':row['raw']['done'],'root_stdout_seconds':row['raw']['stdout_seconds'],'output_ids':row['raw']['output_ids']}for row in rows]})
 if len(actor.PHASES)==2:
  left=c.read(root/actor.PHASES[0]/'recipe.json');right=c.read(root/actor.PHASES[1]/'recipe.json');le=dict(left['native_env']);re=dict(right['native_env']);require(le.pop('STRATA_PF_ESIMD_F16')=='0'and re.pop('STRATA_PF_ESIMD_F16')=='1'and canonical(le)==canonical(re)and left['native_argv']==right['native_argv'] and canonical(left['requests'])==canonical(right['requests']),'Safe baseline matched configuration differs');require(canonical(arms[0])==canonical(arms[1]),'Matched bounded outputs differ')
 proof=actor.fresh.binding();require(actor.closure()==report['source_plan_sha256']and {str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'}==report['artifacts'],'Closing actual source/raw artifacts changed')
 result.update(H36_OFF_ON_equivalence_passed=False,identity_scope='whole_matched_arms',current_public_full4_rehashed=False,saved_campaign_full4_validated=True,source_generation=51,actual51_SDK=proof,original43_runtime_proof_transferred=False,grouped_F16_enabled=False,ESIMD_F16_only_switch=True,matched_arms_outputs_equal=len(arms)==2,native_observations=observations,numeric_first49_observed=False,H36_coverage_qualified=False,graph_coverage_qualified=False,device_coverage_qualified=False,full_model_math_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False);return result
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
