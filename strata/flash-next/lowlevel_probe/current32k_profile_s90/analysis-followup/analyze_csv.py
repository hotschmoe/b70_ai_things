from pathlib import Path
from collections import Counter
import csv,json,heapq,re,time
r=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014');p=r/'current32k-profile-s90-analysis';o=r/'current32k-profile-s90-analysis-followup';s=json.loads((p/'summary.json').read_text());window=s['request_window_epoch_us'];start=window['start_us'];end=window['end_us'];pid0=4294952351;pid1=4294952350
# Extracted timelines only; no raw JSON/trace load.
def ev(row):
 return {'name':row['name'],'pid':int(row['pid']),'tid':int(row['tid']),'start':float(row['epoch_ts_us']),'end':float(row['epoch_ts_us'])+float(row['clipped_us']),'duration_us':float(row['duration_us'])}
gpu0=[];gdn=[];reduce=[];copies0=[]
with(p/'gpu-timeline.csv').open()as f:
 for row in csv.DictReader(f):
  e=ev(row)
  if e['pid']==pid0:
   gpu0.append(e)
   if 'gdn_rec_cols_pipe_kernel'in e['name']:gdn.append(e)
   if'zeCommandListAppendMemoryCopy'in e['name']:copies0.append(e)
  if e['pid']==pid1 and'ep_prefill_reduce_kernel'in e['name']:reduce.append(e)
assert len(gdn)==108 and len(reduce)==144
segments=[]
for e in gpu0:
 if segments and e['start']<=segments[-1]['end']:
  if e['end']>segments[-1]['end']:segments[-1].update(end=e['end'],last=e)
 else:segments.append({'start':e['start'],'end':e['end'],'first':e,'last':e})
gaps=[];cursor=start;previous=None
for seg in segments:
 if seg['start']>cursor:gaps.append({'start':cursor,'end':seg['start'],'previous':previous,'next':seg['first'],'edge':previous is None})
 cursor=max(cursor,seg['end']);previous=seg['last']
if cursor<end:gaps.append({'start':cursor,'end':end,'previous':previous,'next':None,'edge':True})
hostcopies=[];hostwaits=[]
with(p/'host-submit-wait-timeline.csv').open()as f:
 for row in csv.DictReader(f):
  e=ev(row)
  if e['name']=='zeCommandListAppendMemoryCopy':hostcopies.append(e)
  if'Synchronize'in e['name']or'Wait'in e['name']:hostwaits.append(e)
def overlap(a,b):return max(0,min(a['end'],b['end'])-max(a['start'],b['start']))
def union(intervals):
 total=0;last=None
 for a,b in sorted(intervals):
  if last is None or b>last:total+=b-max(a,last if last is not None else a)
  last=max(last if last is not None else b,b)
 return total
copy_union=union([(e['start'],e['end'])for e in hostcopies]);gap_copy_overlap=sum(union([(max(g['start'],h['start']),min(g['end'],h['end']))for h in hostcopies if overlap(g,h)>0])for g in gaps)
def near(e):
 if e is None:return None
 return {'name':e['name'],'pid':e['pid'],'tid':e['tid'],'offset_ms':(e['start']-start)/1000,'duration_us':e['duration_us']}
ranked=[]
for g in sorted(gaps,key=lambda g:g['end']-g['start'],reverse=True)[:15]:
 hs=sorted([h for h in hostcopies if overlap(g,h)>0],key=lambda h:overlap(g,h),reverse=True)
 ranked.append({'offset_ms':(g['start']-start)/1000,'gap_ms':(g['end']-g['start'])/1000,'window_edge':g['edge'],'previous':near(g['previous']),'next':near(g['next']),'host_copy_overlaps':[dict(near(h),overlap_ms=overlap(g,h)/1000)for h in hs[:3]],'overlapping_host_copy_union_ms':union([(max(g['start'],h['start']),min(g['end'],h['end']))for h in hs])/1000,'recorded_GPU0_copy_events_temporally_inside_overlapping_host_calls':sum(1 for c in copies0 if any(overlap(c,h)>0 for h in hs))})
ops=list(csv.DictReader((p/'operations.csv').open()));keyops=[]
for row in ops:
 if row['kind']=='gpu'and any(n in row['operation']for n in ['block_scores_prefill_tiled_s89','block_scores_kernel_925c9d','block_scores_multi_kernel_dea7d4','gdn_rec_cols_pipe_kernel','ep_prefill_reduce_kernel','block_topk']):keyops.append({k:row[k]for k in ['pid','operation','overlapping_count','contained_service_ms']})
# Same-device GDN/reducer order envelopes are motifs, not complete chunk endpoints.
def waves(events,n):
 return [{'source_wave':i+1,'count':len(events[i*n:(i+1)*n]),'first_start_offset_ms':(events[i*n]['start']-start)/1000,'last_end_offset_ms':(events[(i+1)*n-1]['end']-start)/1000,'service_ms':sum(e['duration_us']for e in events[i*n:(i+1)*n])/1000}for i in range(3)]
inputs={}
for f in [p/'summary.json',p/'operations.csv',p/'gpu-timeline.csv',p/'host-submit-wait-timeline.csv',r/'current32k-profile-s90-v2-prep/run/trace/timing.467.csv']:
 st=f.stat();inputs[str(f)]={'bytes':st.st_size,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns}
facts={'status':'EXTRACTED_CSV_ONLY_SOURCE_INFORMED_READINESS_AUDIT','inputs':inputs,'raw_trace_sha256_from_existing_summary':s['raw_sha256'],'request_ms':(end-start)/1000,'gpu0_events':len(gpu0),'gpu0_busy_union_ms':s['devices'][0]['all_recorded_gpu_op_union_ms'],'gpu0_uncovered_ms':sum(g['end']-g['start']for g in gaps)/1000,'gpu0_uncovered_intervals':len(gaps),'gpu1_busy_union_ms':s['devices'][1]['all_recorded_gpu_op_union_ms'],'host_copy_calls_in_request':len(hostcopies),'host_copy_full_duration_sum_ms':sum(e['duration_us']for e in hostcopies)/1000,'host_copy_union_ms':copy_union/1000,'gpu0_recorded_gap_overlap_with_hostcopy_union_ms':gap_copy_overlap/1000,'top15_gpu0_recorded_gaps':ranked,'source_wave_geometry':{'main_tokens':32721,'chunks':[12288,12288,8145],'QSA_layers':12,'GDN_layers':36,'MoE_layers':48,'selection_calls_per_layer':[48,48,32],'expected_prefill_scorer_calls':[576,576,384],'expected_tiled_calls':[0,396,372],'expected_original_prefill_calls':[576,180,12],'observed_tiled':768,'observed_original_all':960,'observed_topk_all':1728,'original_residual_beyond_mainprefill':192,'residual_scope':'Small final prompt handling/capture/decode not independently split; do not label all192 steadydecode without evidence.'},'gpu0_GDN_ordered_36call_envelopes':waves(gdn,36),'gpu1_prefillreduce_ordered_48call_envelopes':waves(reduce,48),'selected_exact_operations':keyops,'clock_drift_us':s['clock']['shift_change_us'],'limits':['No rawJSON reread. Request includes capture/admission/prefill/decode; source3wave envelopes are partial same-device motifs, not exactphase timestamps.','Hostcopy metadata lacks queue/device argument or correlation ID in extractedCSV; overlaps with GPU0 gaps do not assign hostcopy toGPU0 or causality.','Recorded busy/uncovered are not physicalidle/occupancy/removablewait. API duration is not transfer service or P2P traffic.','GPU0/GPU1 clocks never joined; times not summed across devices or nested APIs.']}
(o/'facts.json').write_text(json.dumps(facts,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
print(json.dumps({k:facts[k]for k in ['gpu0_events','gpu0_uncovered_ms','gpu0_uncovered_intervals','host_copy_calls_in_request','host_copy_full_duration_sum_ms','host_copy_union_ms','gpu0_recorded_gap_overlap_with_hostcopy_union_ms']}))
print(json.dumps({'top5_gaps':ranked[:5],'gpu0_gdn_waves':facts['gpu0_GDN_ordered_36call_envelopes'],'gpu1_reduce_waves':facts['gpu1_prefillreduce_ordered_48call_envelopes']},ensure_ascii=True))
