"""Independent actual52 serial cache reader; preserved original model boundary."""
import argparse,copy,math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_source52_serial_cache_latency_v3 as actor
from validate_source47_grouped_f16_model_v2 import boundary_reader_ports
from serial37_canonical_json_v3 import canonical
c=actor.c;require=actor.require;HERE=actor.HERE;ROOT=actor.ROOT;PLAN=HERE/'source52-serial-cache-reader-source-plan-v1.json'

def source_binding():
 value=c.read(PLAN)
 for name,want in value['files'].items():require(c.sha(ROOT/name)==want,'Serial cache public reader source changed '+name)
 return c.sha(PLAN)

@contextmanager
def configured_actor(report,recipe):
 # Replay the producer's explicit private namespace setup; no __getattr__ assumption,
 # no old actor code/global callback is replaced. All changes are process-local intake.
 p=actor.parent;old_ns=p.NS;old_fixture=(actor.FIXTURE,actor.FIXTURE_SHA)
 try:
  p.NS=actor.NS;actor.FIXTURE=Path(recipe['cache_fixture_path']).resolve();actor.FIXTURE_SHA=recipe['cache_fixture_sha256'];p.configure(recipe['natural_settings']);p.set_sdk(report['SDK_root'],report['SDK_receipt_sha256']);ns=actor.NS;ns['fresh']=p.fresh
  ns['set_api'](report['API_prepared_root'],report['API_prepared_sha256']);ns['set_case'](report['role_input'],report['producer_pid'],report['producer_start_ticks']);saved=recipe['tool_runtime'];ns['set_runtime'](saved['intake_path'],saved['exact_intake_sha256']);ns['_PORTS']=p.base.boundary_ports(ns)
  active=SimpleNamespace(**{**vars(actor),**ns});require(active.__file__==actor.__file__ and active.contract.requests is actor.requests and active.contract.equivalence is actor.equivalence,'Actual serial actor private namespace/contract differs');require(active.fresh is p.fresh and active.fresh.SDK is not None,'Actual source52 configured before reader')
  actor.fixture()
  try:yield active
  finally:require(actor.closure()==report['source_plan_sha256'],'Closing original actor source/fixture changed')
 finally:p.NS=old_ns;actor.FIXTURE,actor.FIXTURE_SHA=old_fixture

def head_binding(root,report,directory,meta,raw):
 heads=[x for x in meta['vectors']if x['phase']=='first_logits_before_sampler'];require(len(heads)==1,'Exactly one final-stage first head required');head=heads[0];path=Path(head['path']);require(path.is_file()and not path.is_symlink()and path.resolve()==path and path.parent==(directory/'captures').resolve()and int(head['bytes'])==248320*4,'Exact full native head path/extent differs');before=c.stat_signature(path);data=path.read_bytes();digest=c.sha(path);require(before==c.stat_signature(path)and len(data)==248320*4 and digest==head['sha256']==report['artifacts'][str(path.relative_to(root))],'Original full head SHA/stat differs');values=np.frombuffer(data,dtype='<f4');require(np.isfinite(values).all(),'Full native head nonfinite');maximum=float(np.max(values));argmax=int(np.argmax(values));emitted=raw['output_ids'][0];require(type(emitted)is int and 0<=emitted<248320 and float(values[emitted])==maximum,'Actual greedy first token is not an argmax of captured full head')
 return {'sha256':digest,'bytes':len(data),'path':str(path),'first_emitted_id':emitted,'argmax_smallest_id':argmax,'argmax_tie_count':int(np.count_nonzero(values==maximum)),'all248320_finite':True,'emitted_is_argmax':True}

def request_binding(root,report,directory,row,want):
 raw=row['raw'];require(type(raw['fresh'])is int and type(raw['pin'])is int and raw['fresh']==want['fresh']and raw['pin']==1536 and raw['cancel_requested']is None,'Typed native fresh/PIN/cancel differs');require(canonical(raw)==canonical(c.read(directory/('request-%d.json'%row['ordinal']))),'Individual original request changed');meta=actor.numeric.extract(copy.deepcopy(raw),directory/'captures',True,False,[(0,24),(24,48)]);ledger=meta['ledger'];require(ledger['actual_reused']==want['expected_reused']and ledger['evaluated_prompt_rows']==len(want['ids'])-want['expected_reused']and ledger['cancelled']is False,'Actual work/reuse ledger differs');life=meta['lifecycle'];require(life['observed']is True and life['body_completion_only']is True,'Actual both-stage entry/return body work required');head=head_binding(root,report,directory,meta,raw);done=raw['done'].split();require(done==raw['stdout'][-1].split()and done[5]in('length','stop'),'Actual DONE differs');useful=[(int(line.split()[1]),stamp)for line,stamp in zip(raw['stdout'],raw['stdout_seconds'])if line.startswith('T ')and int(line.split()[1])not in(248044,248046)];require(useful and 0<len(raw['output_ids'])<=64,'Bounded useful GEN64 required');require(done[5]!='length'or len(raw['output_ids'])==64,'Length finish must exhaust cap64');require(done[5]!='stop'or raw['output_ids'][-1]in(248044,248046),'Stop must be actual configured EOS');require(all(i not in(248044,248046)for i in raw['output_ids'][:-1]),'EOS must terminate native output')
 return {'ordinal':row['ordinal'],'role':want['role'],'prompt_tokens':len(want['ids']),'actual_reused':ledger['actual_reused'],'evaluated_prompt_rows':ledger['evaluated_prompt_rows'],'actual_stage_ranges':[[0,24],[24,48]],'each_stage_exact_prompt_body_coverage':True,'completed_stage_spans':sum(e['complete']for e in life['stage_spans']),'first_head':head,'output_ids':raw['output_ids'],'finish':done[5],'client_TTFT_seconds':useful[0][1],'client_completion_seconds':raw['stdout_seconds'][-1],'client_TEmit_gaps_seconds':[b[1]-a[1]for a,b in zip(useful,useful[1:])],'native_prompt_ms':float(done[3]),'native_decode_ms':float(done[4]),'complete_state_cache_qualified':False}

def finalized_binding(root):
 root=Path(root).resolve();source=source_binding();report=c.read(root/'report.json');report_sha=c.sha(root/'report.json');before={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'};require(before==report['artifacts'],'Original actual serial cache roster differs');recipe=c.read(root/actor.parent.PHASES[0]/'recipe.json')
 try:
  with configured_actor(report,recipe)as active:
   result=boundary_reader_ports(active)['finalized_binding'](root);arms=[];views=[];recipes=[]
   for index,phase in enumerate(active.PHASES):
    directory=root/phase;r=c.read(directory/'recipe.json');recipes.append(r);require(type(r['engine_generation'])is int and r['engine_generation']==52 and r['native_env']['STRATA_COPY_ENGINE']=='0'and r['native_env']['STRATA_PF_GROUPED_F16']=='0'and r['native_env']['STRATA_PF_ESIMD_F16']=='0'and r['native_env']['STRATA_CRITICAL_PATH_TRACE']=='0'and r['PTI_enabled']is False and r['cards']==[0,1]and r['stages']==[[0,0,24],[1,24,48]],'Actual bounded source52 pair/plain compute route required');require(r['native_argv'][r['native_argv'].index('--prompt-cache')+1]=='3'and r['native_argv'][r['native_argv'].index('--conversation-cache-mib')+1]=='0','Exact basic prompt-only cache profile required');rows=c.read(directory/'requests.json');v=actor.requests(rows,r['requests']);views.append(v);observations=[request_binding(root,report,directory,row,want)for row,want in zip(rows,r['requests'])];require([o['actual_reused']for o in observations]==([0]*6 if index==0 else[0,1536,1536,1536,1536,1536]),'Actual complete cold/hit reuse pattern differs');
    for indexes in((0,1,4),(2,3,5)):
     first=observations[indexes[0]]
     for i in indexes[1:]:require(observations[i]['output_ids']==first['output_ids']and observations[i]['finish']==first['finish']and observations[i]['first_head']['sha256']==first['first_head']['sha256'],'Within-arm exact role repeats differ')
    require(not list((directory/'host-traces').glob('*'))and not list((directory/'unitrace').glob('*')),'Disabled heavy observer emitted files');asset=r['expert_profile_binding'];require(actor.parent.profile_binding(asset['path'],asset['sha256'])==asset,'Current source52 expert profile changed');arms.append({'phase':phase,'requests':observations})
   require(recipes[0]['native_argv']==recipes[1]['native_argv']and canonical(recipes[0]['native_env'])==canonical(recipes[1]['native_env']),'Only request fresh state may differ between arms');comparison=actor.equivalence(*views);require(canonical(comparison)==canonical(report['H36_OFF_ON_comparison']),'Original serial cache comparison result differs');proof=active.fresh.binding();result.update(arms=arms,actual52_SDK=proof,source_generation=52,reader_source_plan_sha256=source,producer_report_sha256=report_sha,requests_per_arm=6,native_processes=2,instrumented_cache_diagnostic=True,clean_latency_qualified=False,H36_OFF_ON_equivalence_passed=False,all_first_heads_finite_and_greedy=True,cold_hit_head_bitwise=True,cold_hit_output_IDs_equal=True,bounded_basic_cache_passed=True,full_cache_qualified=False,native_concurrency_qualified=False,original_fullmath_qualified=False,API_qualified=False,shelf_qualified=False,speed_promotion_qualified=False);return result
 finally:require(source_binding()==source and c.sha(root/'report.json')==report_sha and before=={str(p.relative_to(root)):c.sha(p)for p in root.rglob('*')if p.is_file()and p.name!='report.json'},'Closing actual reader/source/raw roster differs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New external public output required');c.write(a.output,finalized_binding(a.root))
