#!/usr/bin/env python3
"""One retained S95 raw read/hash/parse; all events in12 exact anchor intervals."""
from pathlib import Path
import csv,hashlib,json,math,statistics,time
P=Path(__file__).resolve().parent
R=P.parent
G=R/'current-c2-profile-s95-prep/run/analysis/gpu'
def save(name,v):
 with(P/name).open('x',encoding='ascii')as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n')
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def compact(e):return {'name':e['name'],'trace_ts_us':e['ts'],'duration_us':e['dur'],'pid':e['pid'],'tid':e.get('tid'),'id':e.get('args',{}).get('id')}
def end(e):return e['trace_ts_us']+e['duration_us']
def columns(e):
 import re
 if 'native_mmvq_wide'not in e['name']or'WideQ5K'not in e['name']or'{62080; 1; 1}'not in e['name']:return None
 m=re.search(r'dpct_kernel_scalar<(\d+)>',e['name']);return int(m[1])if m else None
def role(e):
 n=e['name']
 if columns(e)==2:return 'C2_head'
 for key,label in [('sampler_greedy_kernel_5b2c3f','sampler'),('row_top_prob_split_kernel_9495b0','top_probability'),('mtp_select_kernel_4611fe','select'),('copy_from_mapped_multi_kernel_66c71e','step_position_copy'),('embed_rows_kernel_284a43','token_embedding'),('norm_127936','embedding_RMS'),('native_quantize_q8_1_kernel_b480d2','embedding_Q8')]:
  if key in n:return label
 return 'UNEXPECTED_RETAINED'
t0=time.monotonic()
for row in json.loads((P/'source-pins.json').read_text())['files']:assert sha(Path(row['path']))==row['sha256'],row['path']
old=json.loads((G/'cycles.json').read_text());primary=old['primary_pid']
assert old['trace_sha256']=='1139c4c8cc9a4b0737a6c57e67a8b717151ec4276714fa355fb48e3cd9ab3583'
cycles=[c for c in old['cycles']if c['index']in range(5,17)]
assert [c['index']for c in cycles]==list(range(5,17))and len({c['index']for c in cycles})==12
intervals=[]
for c in cycles:
 heads=sorted(c['head_events'],key=lambda e:e['trace_ts_us'])
 assert [columns(e)for e in heads]==[2,2,6]and c['structurally_compatible_full_paired_cycle']
 h0,h1=heads[:2]
 selects=[e for e in c['pair_tail_motifs']if'mtp_select'in e['name']and end(h0)<=e['trace_ts_us']<h1['trace_ts_us']]
 assert len(selects)==2 and len({e['id']for e in selects})==2
 selects.sort(key=lambda e:e['trace_ts_us'])
 quants=sorted((e for e in c['pair_tail_motifs']if'native_quantize_q8_1_kernel_b480d2'in e['name']and end(selects[-1])<=e['trace_ts_us']<h1['trace_ts_us']),key=lambda e:e['trace_ts_us'])
 assert quants
 q=quants[0];anchors=[h0,*selects,q]
 assert all(e['pid']==primary for e in anchors)and len({e['id']for e in anchors})==4
 intervals.append({'cycle':c['index'],'start_us':h0['trace_ts_us'],'end_us':end(q),'anchors':{'first_C2':h0,'second_C2_context':h1,'select0':selects[0],'select1':selects[1],'first_following_Q8':q},'events':[]})
assert all(a['end_us']<b['start_us']for a,b in zip(intervals,intervals[1:]))
save('anchors.json',{'status':'TWELVE_EXACT_RETAINED_ANCHOR_INTERVALS','primary_pid':primary,'intervals':[{k:v for k,v in c.items()if k!='events'}for c in intervals],'clock_scope':'Same primary GPU raw trace timestamps only; no shifts/native/peer join'})
trace=Path(old['trace']);expected_bytes=269853744
assert trace.stat().st_size==expected_bytes
print('S108 one raw read/hash/parse begins',flush=True)
read_start=time.monotonic();raw=trace.read_bytes();read_s=time.monotonic()-read_start
hash_start=time.monotonic();digest=hashlib.sha256(raw).hexdigest();hash_s=time.monotonic()-hash_start
assert digest==old['trace_sha256']and len(raw)==expected_bytes
parse_start=time.monotonic();payload=json.loads(raw);del raw;parse_s=time.monotonic()-parse_start
assert isinstance(payload,dict)and isinstance(payload.get('traceEvents'),list)
events=payload['traceEvents'];assert len(events)==old['total_trace_events']==1663171
meta=[e for e in events if e.get('ph')=='M']
proc={e['pid']:e.get('args',{}).get('name','')for e in meta if e.get('name')=='process_name'}
matches=[pid for pid,n in proc.items()if'0:b:0:0 #0'in n]
assert matches==[primary]and proc[primary]==old['process_names'][str(primary)]
scan_start=time.monotonic();primary_complete=0
for ordinal,e in enumerate(events):
 if e.get('ph')!='X'or e.get('cat')!='gpu_op'or e.get('pid')!=primary:continue
 assert isinstance(e.get('ts'),(int,float))and isinstance(e.get('dur'),(int,float))and math.isfinite(e['ts'])and math.isfinite(e['dur'])and e['dur']>=0
 primary_complete+=1;s=e['ts'];f=s+e['dur']
 for window in intervals:
  lo,hi=window['start_us'],window['end_us']
  if(s<hi and f>lo)or(e['dur']==0 and lo<=s<hi):
   window['events'].append({'raw_event_ordinal':ordinal,'crosses_left':s<lo,'crosses_right':f>hi,'event':e})
scan_s=time.monotonic()-scan_start
for c in intervals:
 for key in ['first_C2','select0','select1','first_following_Q8']:
  a=c['anchors'][key];matches=[x for x in c['events']if compact(x['event'])==a]
  assert len(matches)==1,('Anchor mismatch',c['cycle'],key)
 c['events'].sort(key=lambda x:(x['event']['ts'],x['event']['ts']+x['event']['dur'],x['raw_event_ordinal']))
save('bounded-intervals.json',{'status':'PASS_ONE_RAW_READ_HASH_PARSE_ALL_OVERLAPPING_PRIMARY_GPU_EVENTS','trace':str(trace),'trace_bytes':expected_bytes,'trace_sha256':digest,'primary_pid':primary,'primary_device':proc[primary],'intervals':intervals,'filter':'ALL complete primaryGPU events overlapping exact firstC2start->firstfollowingQ8end, including boundary crossers and unexpected names; no name filter'})
save('metadata.json',{'primary_pid':primary,'primary_device':proc[primary],'retained_process_binding_exact':True,'metadata':meta,'scope':'Original raw metadata; selected primary device bound uniquely, no peer clock interpretation'})
expected=['C2_head','sampler','top_probability','sampler','top_probability','select','select','step_position_copy','token_embedding','embedding_RMS','embedding_Q8']
findings=[]
with(P/'ordered-timeline.csv').open('x',encoding='ascii',newline='')as f:
 writer=csv.writer(f);writer.writerow(['cycle','order','source_role_candidate','raw_event_ordinal','pid','tid','id','start_us','duration_us','end_us','crosses_left','crosses_right','name'])
 for c in intervals:
  seq=[role(x['event'])for x in c['events']];consistent=seq==expected and not any(x['crosses_left']or x['crosses_right']for x in c['events'])
  for j,x in enumerate(c['events']):
   e=x['event'];writer.writerow([c['cycle'],j,seq[j],x['raw_event_ordinal'],e['pid'],e.get('tid'),e.get('args',{}).get('id'),e['ts'],e['dur'],e['ts']+e['dur'],x['crosses_left'],x['crosses_right'],e['name']])
  row={'cycle':c['cycle'],'event_count':len(seq),'source_role_sequence':seq,'role_status':'SOURCE_CONSISTENT_ORDINAL_CHAIN'if consistent else'HOLD_AMBIGUOUS_UNEXPECTED_OR_CROSSING_EVENTS','owner_buffer_bindings_independently_observed':False}
  if consistent:
   es=[x['event']for x in c['events']];finish=lambda j:es[j]['ts']+es[j]['dur']
   row['diagnostic_us']={'C2_head_service':es[0]['dur'],'head_end_to_first_select_start':es[5]['ts']-finish(0),'sampler_probability_service_sum':sum(es[j]['dur']for j in [1,2,3,4]),'last_probability_end_to_first_select_start':es[5]['ts']-finish(4),'both_select_service_sum':es[5]['dur']+es[6]['dur'],'second_select_end_to_position_copy_start':es[7]['ts']-finish(6),'second_select_end_to_embedding_start':es[8]['ts']-finish(6),'head_end_to_first_Q8_end':finish(10)-finish(0),'position_copy_embed_RMS_Q8_service_sum':sum(es[j]['dur']for j in [7,8,9,10]),'smallest_signed_adjacent_coverage_gap':min(es[j+1]['ts']-finish(j)for j in range(10))}
  findings.append(row)
status='PASS_SOURCE_CONSISTENT_ORDINAL_CHAIN_ALL12'if all(x['role_status']=='SOURCE_CONSISTENT_ORDINAL_CHAIN'for x in findings)else'HOLD_ROLE_ATTRIBUTION_REVIEW_RAW_EXTRACTION_COMPLETE'
summary=None
if status.startswith('PASS'):
 keys=list(findings[0]['diagnostic_us']);summary={k:{'median':statistics.median(x['diagnostic_us'][k]for x in findings),'min':min(x['diagnostic_us'][k]for x in findings),'max':max(x['diagnostic_us'][k]for x in findings)}for k in keys}
save('findings.json',{'status':status,'cycles':findings,'descriptive_same_trace_nested12_statistics_us':summary,'scope':'Historical instrumented S95 sameGPU sequence only; role labels are source-consistent ordinal, not independently bound buffers/slot owners. No raw ID-gap inference.','no_claims':['No physical idle/traffic/bandwidth or removable-time forecast.','No CPU-steady/GPU or cross-card clock join.','No S106 timing, queue-fed-ahead, clean serving speed or stability result.','Service sums and coverage spans are not interchangeable wall budgets.']})
save('execution.json',{'status':'TERMINAL_EXTRACTION_COMPLETE','raw_read_count':1,'raw_JSON_parse_count':1,'raw_sha256_count_same_in_memory_bytes':1,'raw_trace_bytes':expected_bytes,'raw_trace_sha256':digest,'read_s':read_s,'hash_s':hash_s,'JSON_parse_s':parse_s,'bounded_scan_s':scan_s,'script_total_s':time.monotonic()-t0,'raw_events':1663171,'primary_complete_gpu_events':primary_complete,'bounded_events':sum(len(c['events'])for c in intervals),'role_status':status,'no_GPU_or_new_trace':True})
print(json.dumps({'status':status,'bounded_events':sum(len(c['events'])for c in intervals),'elapsed_s':time.monotonic()-t0,'raw_read_count':1}),flush=True)
