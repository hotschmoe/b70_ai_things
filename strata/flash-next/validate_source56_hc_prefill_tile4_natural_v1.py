"""Current source/raw/native lifecycle admission; natural output timings only."""
import argparse,hashlib,math
from pathlib import Path
import run_source56_hc_prefill_tile4_natural_v1 as actor
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
c=actor.c;require=actor.require

def observations(rows,recipe):
 actor.requests(rows,recipe['requests']);result=[]
 for row,want in zip(rows,recipe['requests']):
  raw=row['raw'];done=raw['done'].split();tokens=[(int(line.split()[1]),stamp)for line,stamp in zip(raw['stdout'],raw['stdout_seconds'])if line.startswith('T ')]
  useful=[(token,stamp)for token,stamp in tokens if token not in(248044,248046)]
  require(useful,'No useful natural output');require(all(token not in(248044,248046)for token,_ in tokens[:-1]),'EOS must terminate the request')
  if done[5]=='stop':require(tokens[-1][0]in(248044,248046),'Stop must be actual configured EOS')
  if done[5]=='length':require(len(tokens)==512,'Length finish must exhaust actual cap')
  result.append({'ordinal':row['ordinal'],'case_id':want['role'],'process_temperature':want['process_temperature'],'input_temperature':want['input_temperature'],'input_tokens':2048,'emitted_tokens':len(tokens),'useful_non_EOS_tokens':len(useful),'natural_output_256_target_met':len(useful)>=256,'finish':done[5],'native_PP_ms':float(done[3]),'native_decode_ms':float(done[4]),'root_client_TTFT_s':useful[0][1],'root_client_completion_s':raw['stdout_seconds'][-1],'root_useful_token_gaps_s':[b[1]-a[1]for a,b in zip(useful,useful[1:])],'output_ids':raw['output_ids'],'decoded_text_SHA_observed':False})
 return result

def finalized_binding(root):
 root=Path(root);report=c.read(root/'report.json');recipe=c.read(root/actor.PHASES[0]/'recipe.json');actor.configure(recipe['natural_settings']);actor.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);actor.set_api(report['API_prepared_root'],report['API_prepared_sha256']);actor.set_case(report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=recipe['tool_runtime'];actor.set_runtime(saved['intake_path'],saved['exact_intake_sha256'])
 before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Original actual raw roster changed')
 try:
  result=boundary_reader_ports(actor)['finalized_binding'](root);arms=[];views=[];recipes=[]
  for phase in actor.PHASES:
   directory=root/phase;current=c.read(directory/'recipe.json');recipes.append(current);rows=c.read(directory/'requests.json');views.append(actor.requests(rows,current['requests']));asset=current['expert_profile_binding'];require(actor.profile_binding(asset['path'],asset['sha256'])==asset,'Current admitted profile changed');arms.append({'phase':phase,'natural_observations':observations(rows,current),'startup_memory_admission':c.read(directory/'memory-admission.json'),'expert_profile':asset,'copy_engine':current['native_env']['STRATA_COPY_ENGINE'],'HC_Q8_token_tile':current['native_env']['STRATA_HC_Q8_TOKEN_TILE'],'HC_prefill_coalesce':current['native_env']['STRATA_HC_PREFILL_COALESCE'],'HC_prefill_tile4':current['native_env']['STRATA_HC_PREFILL_TILE4']})
  actor.matched_recipe_gate(*recipes);actor.equivalence(*views)
  result.update(arms=arms,requested_prefill=recipe['requested_prefill'],resolved_prefill_basis='exact explicit non-auto command plus startup/raw logs; capacity fallback refused',requested_context=4096,source_generation=recipe['engine_generation'],processes=2,requests_per_process=6,matched_experiment=recipe['matched_experiment'],matched_outputs_equal=True,natural_EOS_honored=True,original_fullmath_qualified=False,device_coverage_qualified=False,API_qualified=False,speed_promotion_qualified=False)
  return result
 finally:require(actor.closure()==report['source_plan_sha256']and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing source/raw roster differs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external output required');c.write(a.output,finalized_binding(a.root))
