"""Bounded host-only source-shaped spans; never infer device idle or bandwidth."""
import math,re,collections
import run_original_mtp53_two_stage_v5 as base

def collect(text,actual):
 rows=base.records(text,'MTP53_HOST_PHASE');foot=base.records(text,'MTP53_HOST_PHASE_SCOPE')
 expected={'rows':str(len(rows)),'capacity':'32768','additional_GPU_waits':'0','additional_GPU_queries':'0','additional_GPU_copies':'0','stage_spans_nested':'1','spans_additive':'0','host_only':'1','production_speed_qualified':'0'}
 if not rows or len(rows)>32768 or foot!=[expected]:raise ValueError('Complete bounded host timing scope required')
 allowed={'draft_prepare_submit','draft_record_wait','draft_settle_wait','own_head_read_argmax','target_run','target_commit','target_wait_commit','draft_prior_drain','draft_discard_recovery','final_target_wait_commit','target_prefill_part_run','target_stage_run','target_stage_submit'}
 if [int(r['ordinal'])for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Original completion ordinal roster differs')
 counts=collections.Counter();stage={};parts=[]
 for r in rows:
  begin,end=int(r['begin_ns']),int(r['end_ns']);ms=float(r['wall_ms'])
  if r['phase']not in allowed or r['scope']not in('teacher_prompt','decode')or r['pid']!='1'or int(r['tid'])<=0 or r['success']!='1'or begin<=0 or end<begin or not math.isfinite(ms)or abs(ms-(end-begin)/1e6)>1e-8:raise ValueError('Actual host span value/ownership differs')
  counts[(r['scope'],r['phase'])]+=1
  if r['phase'].startswith('target_stage_'):
   d=int(r['device']);lo,hi=int(r['lb']),int(r['le']);T,p=int(r['T']),int(r['position'])
   if d not in(0,1)or(lo,hi)!=(d*24,(d+1)*24)or not re.fullmatch(r'0x[0-9a-f]+',r['owner']):raise ValueError('Actual target stage span owner differs')
   key=(d,T,p,r['phase'])
   if key in stage:raise ValueError('Duplicate actual target stage span')
   stage[key]=r
  elif r['phase']=='target_prefill_part_run':parts.append(r)
 windows=actual['windows'];expected_runs=[(1,2047)]+[(2 if int(w['remaining'])>=2 else 1,int(w['position'])-int(w['committed']))for w in windows]
 if set(stage)!={(d,T,p,k)for d in(0,1)for T,p in expected_runs for k in('target_stage_run','target_stage_submit')}:raise ValueError('Actual stage spans not matched to completed native windows')
 for T,p in expected_runs:
  a,b=stage[(0,T,p,'target_stage_run')],stage[(1,T,p,'target_stage_run')]
  if not int(a['begin_ns'])<=int(b['begin_ns'])<=int(b['end_ns'])<=int(a['end_ns'])or a['tid']!=b['tid']:raise ValueError('Original recursive same-thread span chronology differs')
  for d in(0,1):
   r,s=stage[(d,T,p,'target_stage_run')],stage[(d,T,p,'target_stage_submit')]
   if r['owner']!=s['owner']or r['tid']!=s['tid']or not int(r['begin_ns'])<=int(s['begin_ns'])<=int(s['end_ns'])<=int(r['end_ns']):raise ValueError('Actual selected submit span not within its source run')
 for phase in('draft_prepare_submit','draft_record_wait','draft_settle_wait'):
  if counts[('teacher_prompt',phase)]!=2048 or counts[('decode',phase)]!=sum(int(w['committed'])for w in windows):raise ValueError('Actual own teacher/catchup phase counts differ')
 for phase in('target_run','target_commit','target_wait_commit'):
  if counts[('teacher_prompt',phase)]!=1 or counts[('decode',phase)]!=len(windows):raise ValueError('Actual target calls/commit/wait counts differ')
 if counts[('decode','own_head_read_argmax')]!=len(windows)or counts[('decode','draft_prior_drain')]!=len(windows)or counts[('decode','final_target_wait_commit')]!=1:raise ValueError('Actual decoder phase ownership differs')
 ends=base.pp_policy.cuts(base.read(base.FIXTURE)['cases'][0]['ids'],base.pp_policy.policy());start=0
 if len(parts)!=len(ends):raise ValueError('Complete actual prompt part spans required')
 for r,end in zip(parts,ends):
  if int(r['device'])!=-1 or int(r['T'])!=end-start or int(r['position'])!=start or r['scope']!='teacher_prompt':raise ValueError('Actual normal-cut whole-part ownership differs')
  start=end
 return {'rows':rows,'row_count':len(rows),'clock_domain':'host_std_steady_clock_nanoseconds','per_phase_counts':{a+'/'+b:v for(a,b),v in counts.items()},'stage_call_count':{'0':len(expected_runs),'1':len(expected_runs)},'nested_spans_additive':False,'physical_device_idle_bandwidth_or_calibration_qualified':False}
