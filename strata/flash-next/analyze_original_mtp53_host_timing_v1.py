"""Closed, admitted host-only timing analysis; no current model scans or GPU authority."""
import argparse,hashlib,json,math,collections
from pathlib import Path
import run_original_mtp53_host_timing_v1 as actor
import original_mtp53_host_timing_intake_v1 as timing
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'original-mtp53-host-timing-analysis-source-plan-v1.json'
RUNTIME_PLAN_SHA='90ddca829a7d3e046cd8057af7a2eff8c380d0f85d329aa4b25cf9c68b13421d'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def guard():
 for path,want in read(PLAN)['files'].items():
  if sha(ROOT/path)!=want:raise ValueError('Exact analysis source changed '+path)
 return sha(PLAN)
def stats(values):
 values=sorted(values);n=len(values)
 return {'count':n,'p50_ms':values[math.ceil(.50*n)-1]if n else None,'p95_ms':values[math.ceil(.95*n)-1]if n else None,'total_ms':sum(values),'max_ms':values[-1]if n else None,'nearest_rank_percentiles':True}
def summarize(actual):
 rows=actual['host_timing']['rows'];phases=collections.defaultdict(list);stages={};output_windows=[]
 for row in rows:
  begin,end=int(row['begin_ns']),int(row['end_ns']);elapsed=(end-begin)/1e6
  bucket=row['phase']+('/device'+row['device']if row['phase'].startswith('target_stage_')else'')
  phases[(row['scope'],bucket)].append(elapsed)
  if row['phase']=='target_stage_run':stages[(int(row['device']),int(row['T']),int(row['position']))]=row
 emitted=1
 for ordinal,window in enumerate(actual['windows'],1):
  keep=int(window['committed']);position=int(window['position'])-keep;T=2 if int(window['remaining'])>=2 else 1
  a,b=stages[(0,T,position)],stages[(1,T,position)]
  a_begin,a_end,b_begin,b_end=map(int,(a['begin_ns'],a['end_ns'],b['begin_ns'],b['end_ns']))
  if a['tid']!=b['tid']or not a_begin<=b_begin<=b_end<=a_end:raise ValueError('Same-thread nested target ownership changed')
  inclusive=(a_end-a_begin)/1e6;child=(b_end-b_begin)/1e6
  ids=actual['ids'][emitted:emitted+keep]
  if len(ids)!=keep:raise ValueError('Actual committed token roster incomplete')
  output_windows.append({'ordinal':ordinal,'position':position,'T':T,'accepted':bool(int(window['accepted'])),'committed':keep,'output_ordinals':list(range(emitted+1,emitted+keep+1)),'output_ids':ids,'source_input_id':int(window['input']),'stage0_inclusive_host_ms':inclusive,'stage1_inclusive_host_ms':child,'stage0_exclusive_host_ms':inclusive-child,'host_span_start_ns':a_begin,'host_span_end_ns':a_end,'tid':int(a['tid']),'device_idle_inferred':False})
  emitted+=keep
 if emitted!=len(actual['ids']):raise ValueError('Actual final committed output count differs')
 categories={}
 for label in('accepted','rejected'):
  selected=[w for w in output_windows if w['accepted']==(label=='accepted')]
  categories[label]={'windows':len(selected),'committed_tokens':sum(w['committed']for w in selected),'target_stage0_inclusive_host_ms':stats([w['stage0_inclusive_host_ms']for w in selected]),'target_stage1_inclusive_host_ms':stats([w['stage1_inclusive_host_ms']for w in selected]),'stage0_exclusive_host_ms':stats([w['stage0_exclusive_host_ms']for w in selected]),'by_actual_target_T':{str(T):stats([w['stage0_inclusive_host_ms']for w in selected if w['T']==T])for T in(1,2)},'conditional_observation_not_causal_acceptance_cost':True}
 return {'scope_phase_statistics':{scope+'/'+phase:stats(values)for(scope,phase),values in sorted(phases.items())},'decode_window_statistics':categories,'decode_windows':output_windows,'first_output_ordinal_is_prompt_final_target_row':True,'host_clock_domain':'std_steady_clock_nanoseconds','nested_totals_additive':False,'stage0_exclusive_means_host_elapsed_outside_nested_stage1':True,'device_idle_bandwidth_calibration_qualified':False,'clean_latency_or_serving_qualified':False}
def analyze(root,expected):
 root=Path(root);report_path=root/'report.json';source=guard();before=sha(report_path)
 if before!=expected:raise ValueError('Exact actual closed admitted timing report required')
 report=read(report_path)
 if report.get('passed')is not True or report.get('errors')!=[]or report.get('source_plan_sha256')!=RUNTIME_PLAN_SHA or report.get('faithful_full512_history_equal')is not True or len(report.get('arm_results',[]))!=1:raise ValueError('Successful full512 protected timing intake required')
 roster=report['artifacts']
 def closed():
  if sha(report_path)!=before or guard()!=source:raise ValueError('Original report/source changed while analyzing')
  for name,want in roster.items():
   path=root/name
   if path.is_symlink()or not path.is_file()or path.resolve().parent!=root.resolve()and root.resolve()not in path.resolve().parents or sha(path)!=want:raise ValueError('Original closed artifact changed '+name)
 closed()
 try:
  arm=report['arm_results'][0]
  if arm['mode']!='on' or arm.get('passed')is not True or arm.get('owned_removed')is not True:raise ValueError('Actual MTP-on normal owned close required')
  directory=root/'on-0-0';stdout=(directory/'native.stdout').read_text();stderr=(directory/'native.stderr').read_text();recipe=read(directory/'recipe.json')
  env=recipe['native_env']
  if env.get('STRATA_ORIGINAL_MTP53_PHASE_TIMING')!='1'or any(env.get(k)!='0'for k in('STRATA_ORIGINAL_MTP53_TEACHER_HEAD_SKIP','STRATA_VERIFY_HC_T2_REUSE','STRATA_EXPERT_PAIR_REUSE')):raise ValueError('Exact instrumented/no-confound policy required')
  prompt=read(root/'tokenizer-fixture.json')['cases'][0]['ids'];actual=actor.natural_rows(stdout,'on',prompt);actual['residency']=actor.residency(stderr);actual['host_timing']=timing.collect(stdout,actual)
  if json.loads(json.dumps(actual))!=arm['actual']:raise ValueError('Fresh original raw/source collector differs from admitted timing result')
  result=summarize(actual);result.update(schema=1,passed=True,report_sha256=before,report_root=str(root),analysis_source_plan_sha256=source,historical_diagnostic_only=True,current_model_health_authority=False)
  return result
 finally:closed()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--report-sha256',required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 if a.output.exists()or not a.output.is_absolute():raise ValueError('New absolute output required')
 result=analyze(a.root,a.report_sha256);a.output.write_text(json.dumps(result,indent=2,ensure_ascii=True)+'\n',encoding='ascii');print(json.dumps({'passed':True,'output':str(a.output),'decode_window_statistics':result['decode_window_statistics']},ensure_ascii=True))
