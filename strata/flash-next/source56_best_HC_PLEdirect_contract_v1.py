"""Actual compiled56 HC control; recorded61 reference never becomes same-source proof."""
from pathlib import Path
from serial37_canonical_json_v3 import canonical
import run_source56_hc_prefill_tile4_natural_v1 as natural
c=natural.c;require=natural.require
B17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
BASE61=B17/'source61-ple-clean-abba-v1-run-v1'
PUBLIC61=B17/'source61-ple-clean-abba-v1-public-v1.json'
PUBLIC61_SHA='5092765bfb438c2175d05558eb6ccdb07e877d74a1f3197d15b7fdb2fa8efc78'
REPORT61_SHA='04eb743e5ab5043e5e3f24dabdfeadcac15c0743e683d214e076098ed56f687a'
DIRECT61=('source61-ple-clean-B1-direct','source61-ple-clean-B2-direct')
HC={'STRATA_HC_Q8_TOKEN_TILE':'2','STRATA_HC_PREFILL_COALESCE':'1','STRATA_HC_PREFILL_TILE4':'1'}
OFF=('STRATA_TRACE','STRATA_PREFILL_HOST60','STRATA_BATCH_PREFILL_LAYER_COOPERATE','STRATA_BATCH_PREFILL_SINGLE_WINDOW','STRATA_CRITICAL_PATH_TRACE','STRATA_FIDELITY_DIAG','STRATA_PREFIX_DIAG','STRATA_PREFIX_LIFECYCLE_DIAG','STRATA_BATCH_FIDELITY_DIAG','STRATA_FULL_CACHE_OBSERVER38','STRATA_COPY_ENGINE','STRATA_PF_GROUPED_F16','STRATA_PF_ESIMD_F16')

def baseline61_binding(recipe=None):
 require(c.sha(PUBLIC61)==PUBLIC61_SHA and c.sha(BASE61/'report.json')==REPORT61_SHA,'Closed actual61 baseline reference changed');public=c.read(PUBLIC61);report=c.read(BASE61/'report.json');require(public['passed'] is True and public['bounded61_PLE_clean_ABBA_capture_passed'] is True and public['producer_report_sha256']==REPORT61_SHA and report['passed'] is True,'Current61 baseline not independently closed')
 result=[]
 for phase in DIRECT61:
  path=BASE61/phase/'recipe.json';require(c.sha(path)==report['artifacts'][phase+'/recipe.json'],'Actual61 direct recipe changed');prior=c.read(path);arm=next(x for x in public['arms'] if x['phase']==phase);require(prior['PLE_mode']=='direct' and len(prior['requests'])==len(arm['natural_observations'])==6,'Actual61 six-history direct reference required')
  if recipe is not None:
   require(recipe['native_argv']==prior['native_argv'] and canonical(recipe['requests'])==canonical(prior['requests']),'Common actual runtime arguments/prompts must equal current61 direct')
   env=dict(recipe['native_env']);old=dict(prior['native_env'])
   for name,value in HC.items():require(env.pop(name)==value and name not in old,'Only explicit HC2/1/1 flag additions permitted')
   require(env.pop('STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256')==recipe['producer_binding_sha256'] and old.pop('STRATA_CRITICAL_PATH_TRACE_BINDING_SHA256')==report['source_plan_sha256'],'Disabled diagnostic identity tags must name their actual producers')
   require(canonical(env)==canonical(old),'Unexpected common-config runtime environment delta')
   require(recipe['expert_profile_binding']['sha256']==prior['expert_profile_binding']['sha256'],'Same original profile bytes required')
  result.append(arm['natural_observations'])
 return result

def requests(rows,expected):
 views=natural.requests(rows,expected)
 for prior in baseline61_binding():
  for view,want,old in zip(views,expected,prior):require(want['role']==old['case_id'] and view['output_ids']==old['output_ids'] and view['done_signature'][3]==old['finish'],'Actual56 complete histories differ from current61 B1/B2')
 return views

def equivalence(*arms):
 require(len(arms)==1 and len(arms[0])==6,'One actual56 best-HC direct owner required')
 return dict(bounded_actual56_HC2_1_1_PLEdirect_histories_passed=True,current61_B1_B2_full_histories_equal=True,source56_vs61_difference_explicit=True,same_source_three_flag_AB_qualified=False,scope='Actual56 consumes HC2/1/1; actual61 lacks these kernels. Common CLI/profile/prompts retained; not a same-source flag-only A/B.',original_fullmath_qualified=False,clean_speed_promotion_qualified=False,API_qualified=False,shelf_qualified=False)
