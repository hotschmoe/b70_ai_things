#!/usr/bin/env python3
"""CPU-only S90 postcollection census; run once after root TRACE_READY GO.
Reuses summarize_trace percentile/union; reads and hashes the stopped JSON once.
No GPU, subprocess, model, source edits or benchmark invocation.
"""
import argparse,csv,hashlib,json,statistics,sys
from collections import defaultdict
from pathlib import Path
REPO=Path('/mnt/vm_8tb/github/b70_ai_things')
sys.path.insert(0,str(REPO/'strata/flash-next'))
from summarize_trace import percentile,union_us
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('trace',type=Path)
p.add_argument('--run',type=Path,default=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/current32k-profile-s90-prep/run'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists(),'Preserve previous analysis; do not overwrite/reparse'
def read(name):return json.loads((a.run/name).read_text())
w=read('window.json');before=read('timebase-before.json');after=read('timebase-after.json')
assert w['end_us']>w['start_us']
shift=before['trace_to_client_shift_us'];shift_after=after['trace_to_client_shift_us']
assert all(isinstance(v,(int,float))for v in [shift,shift_after])
# Default profiler epoch-BOOTTIME+RAW requires this correction. Collector records actual env/source.
stat0=a.trace.stat();raw=a.trace.read_bytes();digest=hashlib.sha256(raw).hexdigest();data=json.loads(raw);stat1=a.trace.stat()
assert(stat0.st_ino,stat0.st_size,stat0.st_mtime_ns)==(stat1.st_ino,stat1.st_size,stat1.st_mtime_ns)
events=data['traceEvents']if isinstance(data,dict)else data
proc={e['pid']:e.get('args',{}).get('name','')for e in events if e.get('ph')=='M'and e.get('name')=='process_name'}
threads={(e['pid'],e.get('tid')):e.get('args',{}).get('name','')for e in events if e.get('ph')=='M'and e.get('name')=='thread_name'}
start,end=w['start_us'],w['end_us'];groups=defaultdict(list);gpus=defaultdict(list);host=defaultdict(list);crossers=[];selected=[]
for e in events:
 if e.get('ph')!='X'or'ts'not in e or'dur'not in e:continue
 t=e['ts']+shift;stop=t+e['dur']
 if stop<=start or t>=end:continue
 lo,hi=max(start,t),min(end,stop)
 if hi<=lo:continue
 z=dict(e,ts=lo,dur=hi-lo,original_trace_ts_us=e['ts'],original_duration_us=e['dur'],epoch_ts_us=t,boundary_crossing=t<start or stop>end)
 selected.append(z);kind='gpu'if e.get('cat')=='gpu_op'else'host';pid=e['pid'];name=e.get('name','')
 groups[(kind,pid,e.get('tid'),name)].append(z)
 if kind=='gpu':gpus[pid].append(z)
 elif name.startswith(('ze','ur','pi')):host[(pid,e.get('tid'))].append(z)
 if z['boundary_crossing']:crossers.append(z)
assert gpus,'No GPU events: inspect stop/flush/clock alignment, do not rerun collection automatically'
def hint(name):
 n=name.lower()
 if'block_scores_prefill_tiled_s89'in n:return'qsa_tiled_s89'
 if'block_scores'in n:return'qsa_other_scores'
 if'topk'in n:return'topk'
 if'ep_wait'in n:return'ep_wait'
 if'memorycopy'in n or'memcpy'in n:return'recorded_copy'
 if n.startswith('ze'):return'recorded_runtime_command'
 if'dequant'in n or'unpack'in n:return'dequant_unpack'
 if'gemm'in n or'matmul'in n or'mmq'in n or'mmvq'in n:return'matmul_family'
 if'attn'in n or'attention'in n:return'attention_family'
 if'gdn'in n or'delta'in n or'conv1d'in n:return'gdn_conv_family'
 return'other_exact_name_retained'
rows=[]
for(kind,pid,tid,name),items in groups.items():
 vals=[e['dur']for e in items];complete=[e for e in items if not e['boundary_crossing']]
 rows.append({'kind':kind,'pid':pid,'process':proc.get(pid,str(pid)),'tid':tid,'engine_or_thread':threads.get((pid,tid),str(tid)),'operation':name,'source_hint':hint(name),'overlapping_count':len(items),'contained_count':len(complete),'boundary_crossers':len(items)-len(complete),'contained_service_ms':sum(e['original_duration_us']for e in complete)/1000,'clipped_overlap_ms':sum(vals)/1000,'median_clipped_us':statistics.median(vals),'p95_clipped_us':percentile(vals,.95),'first_trace_ts_us':min(e['original_trace_ts_us']for e in items),'last_trace_end_us':max(e['original_trace_ts_us']+e['original_duration_us']for e in items)})
rows.sort(key=lambda r:(r['kind'],-r['clipped_overlap_ms']))
def compact(e):return{'name':e.get('name'),'pid':e['pid'],'tid':e.get('tid'),'trace_ts_us':e['original_trace_ts_us'],'epoch_ts_us':e['epoch_ts_us'],'duration_us':e['original_duration_us'],'clipped_us':e['dur'],'boundary_crossing':e['boundary_crossing']}
devices=[]
for pid,items in gpus.items():
 named=[e for e in items if not e.get('name','').startswith('ze')]
 no_epwait=[e for e in named if'ep_wait'not in e.get('name','').lower()]
 candidates=[e for e in items if'block_scores_prefill_tiled_s89'in e.get('name','')]
 perhint={}
 for category in sorted({hint(e.get('name',''))for e in items}):
  subset=[e for e in items if hint(e.get('name',''))==category]
  perhint[category]={'count':len(subset),'clipped_sum_ms':sum(e['dur']for e in subset)/1000,'recorded_union_ms':union_us(subset)/1000}
 devices.append({'pid':pid,'metadata_process':proc.get(pid,str(pid)),'count':len(items),'all_recorded_gpu_op_union_ms':union_us(items)/1000,'named_kernel_union_ms':union_us(named)/1000,'named_kernel_excluding_ep_wait_union_ms':union_us(no_epwait)/1000,'source_hint_groups':perhint,'actual_s89_candidate_events':{'count':len(candidates),'exact_names':sorted({e['name']for e in candidates}),'contained_service_ms':sum(e['original_duration_us']for e in candidates if not e['boundary_crossing'])/1000,'clipped_sum_ms':sum(e['dur']for e in candidates)/1000,'union_ms':union_us(candidates)/1000}})
hostspans=[]
for(pid,tid),items in host.items():
 for e in items:
  n=e['name']
  if any(t in n for t in ['Synchronize','Wait','Execute','Submit','Append','Capture','Finalize','Instantiate','CommandGraph']):hostspans.append(compact(e))
# First SSE is perf_counter-based relative to send(), whose exact epoch start was not captured.
request=read('p32768-cap32.json')
summary={'status':'DIAGNOSTIC_REQUEST_CENSUS_PHASE_BOUNDARIES_UNQUALIFIED','trace':str(a.trace),'raw_sha256':digest,'raw_bytes':len(raw),'raw_event_count':len(events),'request_window_epoch_us':w,'clock':{'before_shift_us':shift,'after_shift_us':shift_after,'shift_change_us':shift_after-shift,'before_best_bracket_ns':before['selected']['bracket_ns'],'after_best_bracket_ns':after['selected']['bracket_ns'],'method':'Use fresh before shift for summaries; preserve after drift/brackets. Host correction only; no cross-card clock calibration.'},'selected_overlap_events':len(selected),'boundary_crossers':len(crossers),'devices':devices,'host_L0_UR_API_union_ms_by_process_thread':{str((pid,tid)):union_us(items)/1000 for(pid,tid),items in host.items()},'top100_host_submit_wait_spans_by_duration':sorted(hostspans,key=lambda x:x['duration_us'],reverse=True)[:100],'boundary_crosser_examples':[compact(e)for e in crossers[:50]],'request_native_timings':request.get('timings'),'client_ttft_s':request.get('ttft_s'),'limits':['Full-request bracket includes admission/capture/prefill/32decode. Exact native prefill start/end not recorded; no subtraction of native prompt_ms from epoch endpoints.','TTFT includes admission/prefill/possible firstdecode; perf_counter send() origin has no exact epoch sample. Do not label TTFT exact native prefill or derive precise phase endpoints from it.','Capture/finalize API rows are retained, not silently removed; grouping them does not isolate all capture GPU work.','Per-device metadata pid/engine rows kept separate, no cross-card joins or summed device service budget.','Service sums overlap; union covers recorded intervals only, not physical hardware occupancy/idle. Excluding named ep_wait does not prove productive work.','API durations/adjacency are temporal observations, not cause or ready-dependency proof. Copy names/byte labels do not prove physical PCIe/P2P traffic.','Name-based family hints are source-informed, not complete semantic attribution; exact rows/timeline remain authoritative.','One instrumented ON trace cannot establish delta, clean serving speed or causal critical-path savings.']}
a.output.mkdir(parents=True)
def writecsv(name,vals):
 with(a.output/name).open('w',newline='',encoding='ascii')as f:
  if vals:
   writer=csv.DictWriter(f,fieldnames=list(vals[0]));writer.writeheader();writer.writerows(vals)
writecsv('operations.csv',rows)
# Compact complete request timeline permits subsequent source-informed phase selection without raw JSON reparse.
writecsv('gpu-timeline.csv',[compact(e)for items in gpus.values()for e in sorted(items,key=lambda z:z['original_trace_ts_us'])])
writecsv('host-submit-wait-timeline.csv',sorted(hostspans,key=lambda e:e['trace_ts_us']))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
print(json.dumps({'status':summary['status'],'raw_bytes':len(raw),'events':len(events),'selected':len(selected),'devices':len(devices),'candidate_counts':[d['actual_s89_candidate_events']['count']for d in devices],'clock_shift_change_us':shift_after-shift},ensure_ascii=True))
