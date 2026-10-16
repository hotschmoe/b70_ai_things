#!/usr/bin/env python3
"""S95 one-parse census, adapted from retained S31; no native steady clock join."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,re,statistics,sys
sys.path.insert(0,"/mnt/vm_8tb/github/b70_ai_things/strata/flash-next")
from summarize_trace import union_us,percentile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('trace',type=Path);p.add_argument('--timebase',type=Path,required=True)
p.add_argument('--start-us',type=float,required=True);p.add_argument('--end-us',type=float,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--primary-match',default='0:b:0:0 #0')
a=p.parse_args();raw=a.trace.read_bytes();trace_sha256=hashlib.sha256(raw).hexdigest();payload=json.loads(raw);del raw;events=payload['traceEvents'] if isinstance(payload,dict) else payload
timebase=json.loads(a.timebase.read_text());shift=timebase.get('trace_to_client_shift_us',timebase.get('selected',{}).get('trace_to_client_shift_us'))
if shift is None:raise RuntimeError('Missing fresh clock correction')
proc={e['pid']:e.get('args',{}).get('name','') for e in events if e.get('ph')=='M' and e.get('name')=='process_name'}
pids=[pid for pid,name in proc.items() if a.primary_match in name]
if len(pids)!=1:raise RuntimeError('Primary process identity ambiguous; inspect metadata and pass exact --primary-match')
primary=pids[0]
xs=[e for e in events if e.get('ph')=='X' and 'ts' in e and 'dur' in e]
def within(e,start,end):return e['ts']>=start and e['ts']+e['dur']<=end
lo=a.start_us-shift;hi=a.end_us-shift
selected=[e for e in xs if within(e,lo,hi)]
gpu=[e for e in selected if e.get('cat')=='gpu_op' and e['pid']==primary]
def head_columns(e):
 n=e.get('name','')
 if 'native_mmvq_wide' not in n or 'WideQ5K' not in n or '{62080; 1; 1}' not in n:return None
 m=re.search(r'dpct_kernel_scalar<(\d+)>',n)
 return int(m[1]) if m else None
heads=sorted((e for e in gpu if head_columns(e)==6),key=lambda e:e['ts']+e['dur'])
def counts(items):return dict(sorted(Counter(e['name'] for e in items).items()))
def l0(items):return [e for e in items if e.get('cat')!='gpu_op' and e.get('name','').startswith('ze')]
def compact(e):return {'name':e['name'],'trace_ts_us':e['ts'],'duration_us':e['dur'],'pid':e['pid'],'tid':e.get('tid'),'id':e.get('args',{}).get('id')}
cycles=[]
for i,(left,right) in enumerate(zip(heads,heads[1:])):
 start=left['ts']+left['dur'];end=right['ts']+right['dur']
 complete=[e for e in selected if within(e,start,end)]
 g=[e for e in complete if e.get('cat')=='gpu_op' and e['pid']==primary]
 h=[e for e in complete if head_columns(e) is not None and e['pid']==primary and e.get('cat')=='gpu_op']
 motif=[e for e in g if any(s in e['name'].lower() for s in ['mtp_select','argmax','quantize_q8'])]
 hc=Counter(head_columns(e) for e in h)
 ep=[e for e in g if 'ep_wait' in e['name']]
 boundary=[e for e in selected if e.get('cat')!='gpu_op' and e['name'].startswith('ze') and e['ts']<end and e['ts']+e['dur']>start and not within(e,start,end)]
 interior=i>0 and i<len(heads)-2
 cold_api=[e for e in selected if e.get('cat')!='gpu_op' and any(x in e.get('name','').lower()for x in ['create','finalize','close']) and e['ts']<end and e['ts']+e['dur']>start]
 compatible=interior and hc[2]==2 and hc[1]==0 and len(ep)==48 and not cold_api
 cycles.append({'index':i,'capture_create_finalize_close_API_overlap':[compact(e)for e in cold_api],'trace_start_us':start,'trace_end_us':end,'epoch_start_us':start+shift,'epoch_end_us':end+shift,'duration_ms':(end-start)/1000,'interior_excludes_first_last_candidate':interior,'full_vocabulary_head_column_counts':dict(hc),'head_events':[compact(e) for e in h],'all_primary_GPU_operations':[{'name':name,'count':sum(e['name']==name for e in g),'service_us':sum(e['dur']for e in g if e['name']==name)}for name in sorted({e['name']for e in g})],'primary_ep_wait_count':len(ep),'primary_ep_wait_names':counts(ep),'pair_tail_motifs':[compact(e) for e in motif],'host_L0_API_counts_complete_intervals':counts(l0(complete)),'host_L0_boundary_crossers':[compact(e) for e in boundary],'structurally_compatible_full_paired_cycle':compatible,'classification':'Measured kernel/API counts; paired-tail attribution remains source-informed and requires motif/native-counter reconciliation.'})
# S136 bounded event retention, not a new parser or phase/tensor classifier.
first=next((c for c in cycles if c['structurally_compatible_full_paired_cycle']),None)
if first is not None:
 timeline=[compact(e) for e in gpu if e['ts']<first['trace_end_us'] and e['ts']+e['dur']>first['trace_start_us']]
 first['bounded_GPU_event_timeline']=timeline if len(timeline)<=6000 else None
 first['bounded_GPU_event_timeline_scope']={'first_source_compatible_cycle':True,'event_count':len(timeline),'cap':6000,'truncated':False,'cap_exceeded_INCONCLUSIVE':len(timeline)>6000,'includes_boundary_crossers':True,'tensor_owner_binding':'UNKNOWN_SOURCE_CONDITIONED_ONLY'}
valid=[c for c in cycles if c['structurally_compatible_full_paired_cycle']]
runs=[]
for c in valid:
 if runs and c['index']==runs[-1][-1]['index']+1:runs[-1].append(c)
 else:runs.append([c])
best=max(runs,key=len) if runs else []
window=None if not best else {'start_us':best[0]['epoch_start_us'],'end_us':best[-1]['epoch_end_us'],'trace_start_us':best[0]['trace_start_us'],'trace_end_us':best[-1]['trace_end_us'],'cycle_indices':[c['index'] for c in best],'cycles':len(best),'timestamp_shift_us':shift,'status':'PROPOSED_INTERIOR_REQUIRES_REVIEW'}
out={'status':'STRUCTURAL_CANDIDATES_REVIEW_REQUIRED' if best else 'INCONCLUSIVE_NO_REPLAY_RANGE_NO_RERUN','trace':str(a.trace),'trace_sha256':trace_sha256,'total_trace_events':len(events),'total_complete_events':len(xs),'selected_complete_events':len(selected),'timebase':str(a.timebase),'timestamp_shift_us':shift,'client_window_us':[a.start_us,a.end_us],'process_names':proc,'primary_pid':primary,'full_vocabulary_head_counts_primary':dict(Counter(head_columns(e) for e in gpu if head_columns(e) is not None)),'host_L0_API_counts_entire_client_window':counts(l0(selected)),'cycles':cycles,'proposed_interior_window':window,'limits':['C2 head alone does not identify draft head: require two C2s between six-row target heads plus MTP-select/argmax motifs and native paired counters.','EP48 is the known source/trace signature, not proof of all layers without inspecting names.','Host API intervals counted only when complete inside a cycle; boundary crossers retained separately. Nested durations are not summed.','No peer-clock events are joined into primary cycles; primary BDF identity comes from trace metadata.','One ON trace cannot establish added submissions/waits versus OFF. No S22 same-binary delta or speed claim.','Create/finalize/close overlap is retained and excludes structural replay candidates; CPU first-observed role evidence remains separate and must be reconciled ordinally.','No native steady_ns-to-trace join is authorized. No fastest-cycle selection or clean latency qualification.']}
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'cycles.json').write_text(json.dumps(out,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
if window:(a.output/'interior-window.json').write_text(json.dumps(window,indent=2)+'\n',encoding='ascii')
print(json.dumps({'status':out['status'],'target6_heads':len(heads),'candidate_cycles':len(cycles),'proposed_interior_cycles':len(best)},ensure_ascii=True))

# The same already loaded events supply operation counts; do not parse raw JSON again.
threads={(e['pid'],e['tid']):e.get('args',{}).get('name','')for e in events if e.get('ph')=='M'and e.get('name')=='thread_name'}
groups=defaultdict(list);devices=defaultdict(list);engines=defaultdict(list)
for e in selected:
 kind='gpu'if e.get('cat')=='gpu_op'else'host';process=proc.get(e['pid'],str(e['pid']));thread=threads.get((e['pid'],e.get('tid')),str(e.get('tid')))
 groups[(kind,process,thread,e['name'])].append(e['dur'])
 if kind=='gpu':devices[process].append(e);engines[(process,thread)].append(e)
rows=[]
for(kind,process,thread,name),vs in groups.items():
 rows.append(dict(kind=kind,process=process,engine_or_thread=thread,operation=name,count=len(vs),total_ms=sum(vs)/1000,mean_us=statistics.mean(vs),median_us=statistics.median(vs),p95_us=percentile(vs,.95),min_us=min(vs),max_us=max(vs)))
rows.sort(key=lambda r:(r['kind'],-r['total_ms']))
if not devices:raise RuntimeError('No GPU operations in instrumented window; no repeat')
with(a.output/'operations.csv').open('x',newline='',encoding='ascii')as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={'status':'ONE_PARSE_CURRENT_OPERATION_CENSUS_REQUIRES_REVIEW','trace_sha256':trace_sha256,'complete_events':len(selected),'total_trace_events':len(events),'total_complete_events':len(xs),'gpu_busy_union_ms':{n:union_us(es)/1000 for n,es in devices.items()},'gpu_engine_union_ms':[{'process':n,'engine':t,'union_ms':union_us(es)/1000}for(n,t),es in engines.items()],'all_cycle_count':len(cycles),'structural_replay_candidate_count':len(valid),'limits':['Per-device/engine service and host spans are not additive wall budgets.','No physical traffic/bandwidth/occupancy/idle or causal removable-gap conclusion.','Kernel name/grid/type/columns do not identify every weight or shared-expert branch.','Actual API invocation counts retain poll/timeout ambiguity.','One instrumented pair; nested cycles are not independent replication.','No native steady clock bridge: reconcile CPU markers ordinally only.']}
(a.output/'census.json').write_text(json.dumps(summary,indent=2,ensure_ascii=True)+'\n',encoding='ascii')

coverage=[]
for process,es in devices.items():
 ordered=sorted(es,key=lambda e:e['ts'])
 end=None;last=None
 for e in ordered:
  if end is not None and e['ts']>end:
   coverage.append({'process':process,'trace_start_us':end,'trace_end_us':e['ts'],'gap_us':e['ts']-end,'previous_operation':last['name'],'next_operation':e['name'],'overlapping_host_L0_API_names':counts([x for x in selected if x.get('cat')!='gpu_op'and x.get('name','').startswith('ze')and x['ts']<e['ts']and x['ts']+x['dur']>end])})
  if end is None or e['ts']+e['dur']>end:end=e['ts']+e['dur'];last=e
coverage.sort(key=lambda x:-x['gap_us'])
(a.output/'coverage-gaps.json').write_text(json.dumps({'scope':'Gaps in collected GPU-operation coverage per device, not measured physical idle or a causal CPU/data-readiness budget. Adjacent names/API overlaps are associations only; devices never joined.','total_count':len(coverage),'largest50':coverage[:50]},indent=2,ensure_ascii=True)+'\n',encoding='ascii')
