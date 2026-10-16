#!/usr/bin/env python3
"""Validate complete S32 buffered CPU marker dumps; derive CPU spans, never GPU idle."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json,re,statistics,sys

PREFIX='strata host boundary:'
HEADER=re.compile(r'^strata host boundary: records=(\d+) dropped=(\d+) clock=steady_ns$')
RECORD=re.compile(r'^strata host boundary: w=(\d+) t=(\d+) tag=([a-z0-9_]+) a=(-?\d+) b=(-?\d+)$')
CORE=['window_enter','packed','run_call','verify_enter','stage_enter','stage_capture_begin','stage_capture_ready','stage_inputs_ready','verify_submit_begin','verify_submit_end','verify_wait_begin','verify_wait_end','verify_return','run_return','commit_return','flush_begin','flush_end','window_end']
PAIR=['pair_call','pair_enter','pair_capture_ready','pair_idle_done','pair_inputs_ready','pair_submit_begin','pair_submit_end','pair_poll_begin','pair_wait_begin','pair_wait_done','pair_return','pair_call_return']
KNOWN=set(CORE+PAIR+['stage_ple_begin','stage_ple_end','slot_copy_begin','slot_copy_end'])
SPANS=[('post_enter_to_run','window_enter','run_call'),('run_call_to_return','run_call','run_return'),('stage_capture','stage_capture_begin','stage_capture_ready'),('stage_PLE','stage_ple_begin','stage_ple_end'),('stage_inputs_after_PLE','stage_ple_end','stage_inputs_ready'),('stage_ready_to_verify_submit','stage_inputs_ready','verify_submit_begin'),('verify_submit','verify_submit_begin','verify_submit_end'),('verify_postsubmit_to_wait','verify_submit_end','verify_wait_begin'),('verify_queue_wait','verify_wait_begin','verify_wait_end'),('verify_waitreturn_to_runreturn','verify_wait_end','run_return'),('commit','run_return','commit_return'),('pair_call_to_enter','pair_call','pair_enter'),('pair_capture','pair_enter','pair_capture_ready'),('pair_slots_idle','pair_capture_ready','pair_idle_done'),('pair_input_stage','pair_idle_done','pair_inputs_ready'),('pair_inputs_to_submit','pair_inputs_ready','pair_submit_begin'),('pair_submit','pair_submit_begin','pair_submit_end'),('pair_postsubmit_to_poll','pair_submit_end','pair_poll_begin'),('pair_poll_and_PLE','pair_poll_begin','pair_wait_begin'),('pair_queue_wait','pair_wait_begin','pair_wait_done'),('pair_waitdone_to_return','pair_wait_done','pair_return'),('pair_return_to_caller','pair_return','pair_call_return'),('pair_caller_to_flush','pair_call_return','flush_begin'),('stdout_flush','flush_begin','flush_end'),('post_flush_to_end','flush_end','window_end'),('whole_window','window_enter','window_end')]

def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def stats(v):
 if not v:return None
 s=sorted(v);p=(len(s)-1)*.95;l=int(p);h=min(l+1,len(s)-1)
 return {'count':len(v),'min_ms':s[0],'median_ms':statistics.median(s),'p95_ms':s[l]+(s[h]-s[l])*(p-l),'max_ms':s[-1],'sum_ms':sum(s)}
def parse(lines,errors):
 dumps=[];active=None
 for lineno,line in enumerate(lines,1):
  h=HEADER.fullmatch(line);r=RECORD.fullmatch(line)
  if h:
   if active is not None and len(active['records'])!=active['declared_records']:errors.append(f'line{lineno}: previous dump incomplete')
   active={'index':len(dumps),'header_line':lineno,'declared_records':int(h[1]),'dropped':int(h[2]),'records':[]};dumps.append(active)
   if active['dropped']!=0:errors.append(f'line{lineno}: dropped markers')
  elif r:
   if active is None:errors.append(f'line{lineno}: orphan marker');continue
   if len(active['records'])>=active['declared_records']:errors.append(f'line{lineno}: extra marker beyond header count')
   x={'w':int(r[1]),'t':int(r[2]),'tag':r[3],'a':int(r[4]),'b':int(r[5]),'line':lineno,'dump':active['index']};active['records'].append(x)
   if x['tag'] not in KNOWN:errors.append(f'line{lineno}: unknown marker tag {x["tag"]}')
  elif PREFIX in line:errors.append(f'line{lineno}: malformed/interleaved marker text')
 for d in dumps:
  if len(d['records'])!=d['declared_records']:errors.append(f'dump{d["index"]}: header/actual record count mismatch')
  if any(a['t']>b['t'] for a,b in zip(d['records'],d['records'][1:])):errors.append(f'dump{d["index"]}: timestamp order decreases')
 if not dumps or not any(d['records'] for d in dumps):errors.append('No nonempty complete marker dump')
 return dumps

def one(by,tag,errs,w,required=True):
 values=by.get(tag,[])
 if len(values)!=1:
  if required or values:errs.append(f'window{w}: {tag} requires exactly1, got{len(values)}')
  return None
 return values[0]
def duration(a,b):return (b['t']-a['t'])/1e6

def analyze_window(w,records,errors):
 by=defaultdict(list)
 for r in records:by[r['tag']].append(r)
 out={'window':w,'dump':records[0]['dump'],'records':records,'slot_copy_order':[],'spans_ms':{},'cold_reasons':[]}
 # Legacy explicit zero-packed frames are retained but cannot be decode/replay evidence.
 if len(by.get('packed',[]))==1 and by['packed'][0]['a']==0:
  if [r['tag'] for r in records]!=['window_enter','packed']:errors.append(f'window{w}: malformed empty packed frame')
  out['classification']='empty_packed_frame_not_decode';return out
 req={tag:one(by,tag,errors,w) for tag in CORE}
 if any(x is None for x in req.values()):out['classification']='INVALID';return out
 first,last=req['window_enter'],req['window_end'];out['start_ns']=first['t'];out['end_ns']=last['t']
 if records[0] is not first or records[-1] is not last:errors.append(f'window{w}: positive-window record outside explicit enter/end bounds')
 if len({r['dump'] for r in records})!=1:errors.append(f'window{w}: spans more than one dump')
 seq=[req[t]['t'] for t in CORE]
 if any(a>b for a,b in zip(seq,seq[1:])):errors.append(f'window{w}: required core marker order violated')
 ple=[one(by,t,errors,w,required=False) for t in ['stage_ple_begin','stage_ple_end']]
 if bool(ple[0])!=bool(ple[1]):errors.append(f'window{w}: partial PLE markers')
 if all(ple) and not(req['stage_capture_ready']['t']<=ple[0]['t']<=ple[1]['t']<=req['stage_inputs_ready']['t']):errors.append(f'window{w}: PLE marker order invalid')
 packed=req['packed'];out['packed']={'S':packed['a'],'A':packed['b']};out['stage_hbase']=req['stage_enter']['b']
 for tag in ['run_call','run_return','commit_return']:
  if (req[tag]['a'],req[tag]['b'])!=(packed['a'],packed['b']):errors.append(f'window{w}: {tag} packed shape differs')
 for tag in ['verify_enter','stage_enter','stage_capture_begin','stage_capture_ready','stage_inputs_ready','verify_submit_begin','verify_submit_end','verify_wait_begin','verify_wait_end','verify_return']:
  if req[tag]['a']!=packed['a']:errors.append(f'window{w}: {tag} S differs')
 copies=[];pending=None
 for r in records:
  if r['tag']=='slot_copy_begin':
   if pending is not None:errors.append(f'window{w}: overlapping/nested slot copy begin')
   pending=r
  elif r['tag']=='slot_copy_end':
   if pending is None or (pending['a'],pending['b'])!=(r['a'],r['b']):errors.append(f'window{w}: unmatched slot copy end');pending=None;continue
   if not(req['commit_return']['t']<=pending['t']<=r['t']<=req['flush_begin']['t']):errors.append(f'window{w}: copy outside emission bounds')
   copies.append({'slot':r['a'],'G':r['b'],'start_ns':pending['t'],'end_ns':r['t'],'span_ms':duration(pending,r)});pending=None
 if pending is not None:errors.append(f'window{w}: unterminated slot copy')
 out['slot_copy_order']=copies
 is_pair=bool(by.get('pair_call'))
 if is_pair:
  pr={tag:one(by,tag,errors,w) for tag in PAIR}
  if all(pr.values()):
   if any(a['t']>b['t'] for a,b in zip([pr[t] for t in PAIR],[pr[t] for t in PAIR][1:])):errors.append(f'window{w}: pair marker order violated')
   if not(req['commit_return']['t']<=pr['pair_call']['t']<=pr['pair_call_return']['t']<=req['flush_begin']['t']):errors.append(f'window{w}: pair outside emission bounds')
   if copies and copies[-1]['end_ns']>pr['pair_call']['t']:errors.append(f'window{w}: pair call precedes last slot copy end')
   out['pair']={'argument_T':[pr['pair_enter']['a'],pr['pair_enter']['b']],'effective_T':[pr['pair_capture_ready']['a'],pr['pair_capture_ready']['b']],'max_steps':pr['pair_inputs_ready']['a']}
   for t in ['pair_submit_begin','pair_submit_end','pair_poll_begin','pair_wait_begin','pair_wait_done','pair_return']:
    if pr[t]['a']!=out['pair']['max_steps']:errors.append(f'window{w}: {t} depth differs')
   if [pr['pair_idle_done']['a'],pr['pair_idle_done']['b']]!=out['pair']['effective_T']:errors.append(f'window{w}: pair idle effective widths differ')
 else:
  if any(by.get(tag) for tag in PAIR[1:]):errors.append(f'window{w}: pair markers without pair_call')
 for label,left,right in SPANS:
  a=by.get(left,[]);b=by.get(right,[])
  if len(a)==len(b)==1:
   if b[0]['t']<a[0]['t']:errors.append(f'window{w}: negative {label}')
   out['spans_ms'][label]=duration(a[0],b[0])
 out['adjacent_CPU_spans']=[{'from':a['tag'],'to':b['tag'],'from_a':a['a'],'from_b':a['b'],'to_a':b['a'],'to_b':b['b'],'span_ms':duration(a,b)} for a,b in zip(records,records[1:])]
 pair=out.get('pair');out['pair_only_six_rows_two_slots_two_steps']=bool(pair and packed['a']==6 and packed['b']==2 and len(copies)==2 and {x['slot'] for x in copies}=={0,1} and all(x['G']==3 for x in copies) and pair['argument_T']==[3,3] and pair['max_steps']==2 and all(1<=t<=3 for t in pair['effective_T']) and all(ple))
 out['classification']='complete_pair' if pair else 'complete_nonpair';return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('log',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();own=Path(__file__).resolve().parent
 if not a.output.resolve().is_relative_to(own):ap.error('Output must stay under S32/analysis/')
 a.output.mkdir(parents=True,exist_ok=False);raw=a.log.read_bytes();text=raw.decode('utf-8');errors=[];dumps=parse(text.splitlines(),errors);records=[r for d in dumps for r in d['records']];by_window=defaultdict(list)
 for r in records:
  if r['w']>0:by_window[r['w']].append(r)
 positive_order=[r['w'] for r in records if r['tag']=='window_enter']
 if len(positive_order)!=len(set(positive_order)) or positive_order!=sorted(positive_order):errors.append('Duplicate/nonmonotone window IDs; TLS/thread attribution ambiguous')
 windows=[analyze_window(w,rs,errors) for w,rs in sorted(by_window.items())]
 if any(x['classification']=='INVALID' for x in windows):errors.append('Incomplete decode window')
 validation={'status':'INVALID' if errors else 'COMPLETE_VALIDATED','log':str(a.log),'log_sha256':hashlib.sha256(raw).hexdigest(),'dump_count':len(dumps),'declared_records':sum(d['declared_records'] for d in dumps),'actual_records':len(records),'dropped':sum(d['dropped'] for d in dumps),'errors':errors,'external_precondition':'Root must establish normal process completion; markers have no end-of-file footer/thread IDs.'}
 write(a.output/'validation.json',validation)
 if errors:
  write(a.output/'invalid-dumps.json',dumps);print(json.dumps({'status':'INVALID','errors':errors},ensure_ascii=True));return 2
 seen=set()
 for x in windows:
  by={r['tag']:r for r in x['records']};keys=[];copies=x['slot_copy_order'];packed=x.get('packed',{})
  if packed and len(copies)==packed['A'] and sum(c['G'] for c in copies)==packed['S']:
   keys.append(('target_rows',x.get('stage_hbase'),tuple((c['slot'],c['G']) for c in copies)))
  elif packed:x['cold_reasons'].append('Target row key unknown: not every packed slot survives to a copy marker')
  if x.get('pair'):
   pair=x['pair'];steps=pair['max_steps']
   keys.extend(('pair_pre_round',c,T) for c,T in enumerate(pair['effective_T']))
   keys.extend(('pair_pre_step',c,j) for j in range(1,steps) for c in range(2))
   keys.extend(('pair_tail',j) for j in range(steps))
  for key in keys:
   if key not in seen:x['cold_reasons'].append('First observed capture role/key '+repr(key));seen.add(key)
  x['observed_capture_keys']=[repr(k) for k in keys];x['replay_selected']=x.get('pair_only_six_rows_two_slots_two_steps',False) and not x['cold_reasons']
 cross=[]
 complete=[x for x in windows if 'start_ns' in x]
 for prev,nxt in zip(complete,complete[1:]):
  pb={r['tag']:r for r in prev['records']};nb={r['tag']:r for r in nxt['records']};spans={}
  for left in ['pair_wait_done','pair_return','window_end']:
   for right in ['window_enter','stage_ple_begin','verify_submit_begin']:
    if left in pb and right in nb:
     dt=duration(pb[left],nb[right]);spans[left+'->'+right]=dt
     if dt<0:errors.append('Cross-window timestamps decrease')
  cross.append({'previous_window':prev['window'],'next_window':nxt['window'],'same_dump':prev['dump']==nxt['dump'],'window_ids_consecutive':nxt['window']==prev['window']+1,'both_replay_selected':prev['replay_selected'] and nxt['replay_selected'],'spans_ms':spans,'scope':'CPU ordering. Previous end to next enter includes packing before begin. Cross-dump spans include diagnostic dump/admission/output and cannot be labeled steady handoff.'})
 if errors:
  validation['status']='INVALID';validation['errors']=errors;write(a.output/'validation.json',validation);return 2
 replay=[x for x in windows if x['replay_selected']];roll={name:stats([x['spans_ms'][name] for x in replay if name in x['spans_ms']]) for name,_,_ in SPANS};crossnames={k for x in cross for k in x['spans_ms']};crossroll={k:stats([x['spans_ms'][k] for x in cross if x['both_replay_selected'] and x['same_dump'] and x['window_ids_consecutive'] and k in x['spans_ms']]) for k in sorted(crossnames)}
 report={'status':'COMPLETE_CPU_MARKER_DIAGNOSTIC','validation':validation,'clock':'CPU steady_ns only, no GPU/device clock join','dump_headers':[{k:v for k,v in d.items() if k!='records'} for d in dumps],'window0_admission_records':[r for r in records if r['w']==0],'windows':windows,'cross_windows':cross,'replay_pair_windows':[x['window'] for x in replay],'replay_same_window_span_stats_ms':roll,'replay_same_dump_cross_window_span_stats_ms':crossroll,'cold_selection':'Conservative first-observed capture-role keys, including target row/copy order and per-owner effective pair widths/step/tail roles. No duration threshold. No actual hit/miss flag exists. All cold windows retained.','limitations':['Packing precedes window_enter in corrected S32 patch; enter->packed marker span is not packing duration.','CPU wait spans may contain queue/device waiting. Low CPU span does not prove GPU-idle absence or exclude host contribution to a previous unitrace gap.','Slot-copy begin/end captures the existing wrapper including a standalone draft if pair coordinator is absent; pair-only selection avoids that attribution ambiguity.','No GPU events or saved latency are derived from these markers.','No footer/thread IDs exist; positive records require explicit enter/end and unique monotone IDs, but external root teardown evidence remains required.','Source-equivalence evidence concerns retained work/call ordering; timestamp/buffer overhead may perturb CPU scheduling.']}
 write(a.output/'analysis.json',report)
 print(json.dumps({'status':report['status'],'records':len(records),'windows':len(windows),'pair_six_row_windows':sum(x.get('pair_only_six_rows_two_slots_two_steps',False) for x in windows),'replay_selected':report['replay_pair_windows'],'cross_stats':crossroll},ensure_ascii=True))
 return 0
if __name__=='__main__':raise SystemExit(main())
