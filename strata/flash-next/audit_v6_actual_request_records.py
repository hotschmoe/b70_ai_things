#!/usr/bin/env python3
"""Read-only strengthened V6 actual-record auditor; frozen V6/runtime unchanged."""
import argparse,collections,copy,hashlib,json,re,struct
from pathlib import Path
import serial_prefix_qualification_v6 as frozen


def require(value,message):
 if not value:raise ValueError(message)

def digest(ids):return hashlib.sha256(b''.join(struct.pack('<I',int(i)) for i in ids)).hexdigest()

def fnv(ids):
 value=14695981039346656037
 for token in ids:
  for byte in struct.pack('<I',int(token)):value=((value^byte)*1099511628211)&((1<<64)-1)
 return format(value,'016x')

def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))

def strengthen(request,stage_ranges,diagnostic,activations):
 ids=request['ids'];require(ids and all(0<=int(i)<248320 for i in ids),'Invalid actual source input IDs')
 prefixes=[json.loads(line[len('PREFIX_DIAG '):]) for line in request['stderr'] if line.startswith('PREFIX_DIAG ')]
 selections=[p for p in prefixes if p.get('event')=='selection'];finishes=[p for p in prefixes if p.get('event')=='finish']
 require(len(selections)==len(finishes)==1,'Missing or ambiguous PREFIX_DIAG selection/finish')
 selected,finish=selections[0],finishes[0];identity=(int(selected['pid']),int(selected['request']))
 require(selected['prompt_tokens']==len(ids) and selected['token_fnv_le32']==fnv(ids),'PREFIX_DIAG actual source input hash/token length mismatch')
 for event in prefixes:
  if event.get('event') in {'pin_saved','fresh_all_stage_reset'}:continue # No identity fields in frozen source; descriptive only.
  require((int(event['pid']),int(event['request']))==identity,'PREFIX_DIAG cross-event PID/request mismatch')
 require(finish['prompt_tokens']==len(ids),'PREFIX_DIAG final actual input length mismatch')
 events=[json.loads(line[4:]) for line in request['stderr'] if line.startswith('PCL ')]
 starts=[fields(line) for line in request['stderr'] if line.startswith('SFD request ')]
 vectors=[fields(line) for line in request['stderr'] if line.startswith('SFD vector ')]
 if not diagnostic:
  require(not events and not starts and not vectors,'Flag0 unexpectedly observed PCL/SFD captures')
  return {'passed':True,'scope':'flag0 exact PREFIX input identity only; full logits/lifecycle unobserved','prefix_identity':list(identity),'unbound_descriptive_prefix_events':[e['event'] for e in prefixes if 'pid' not in e],'cancelled':finish['cancelled']}
 begins=[e for e in events if e.get('event')=='begin'];commits=[e for e in events if e.get('event')=='committed_live']
 require(len(begins)==len(commits)==1 and len(starts)==1,'Missing/ambiguous PCL/SFD begin/commit binding')
 begin,commit,start=begins[0],commits[0],starts[0]
 for event in events:require((int(event['pid']),int(event['request']))==identity,'PCL versus PREFIX_DIAG PID/request mismatch')
 require((int(start['pid']),int(start['request']))==identity,'SFD versus PREFIX_DIAG PID/request mismatch')
 require(int(start['tokens'])==len(ids) and [int(t) for t in start['ids'].split(',')]==ids,'SFD actual input token binding mismatch')
 require(begin['tokens']==len(ids) and begin['input_sha256_le32']==digest(ids),'PCL actual source input SHA/token binding mismatch')
 for vector in vectors:
  require((int(vector['pid']),int(vector['request']))==identity,'Raw vector versus actual PREFIX/PCL request mismatch')
  pos=int(vector['pos']);require(0<=pos<len(ids) and int(vector['token'])==ids[pos],'Raw vector actual source position/token mismatch')
 require(not commit['ids_truncated'] and commit['tokens']==len(commit['ids']) and commit['sha256_le32']==digest(commit['ids']),'Committed live prefix ID/digest mismatch')
 require(all(e.get('event')!='skip' for e in events),'Lifecycle quota skipped; observations incomplete')
 spans=[e for e in events if e.get('event')=='stage_span'];enter=collections.Counter();returned=collections.Counter()
 for event in spans:
  bounds=(event['lb'],event['le']);require(bounds in stage_ranges,'Span owner range is absent from actual configuration')
  require(event['device']==stage_ranges.index(bounds),'Span actual device/range association mismatch')
  require(event['phase'] in {'prefill_body','verify_body'},'Unknown stage body phase')
  require(0<=event['lo']<event['hi'],'Invalid or empty emitted actual stage span')
  require(event.get('commit_proven') is False,'Stage body metadata silently claims GPU commit proof')
  key=(event['phase'],event['device'],event['lb'],event['le'],event['lo'],event['hi'])
  (returned if event['complete'] else enter)[key]+=1
 missing=enter-returned;orphan=returned-enter
 require(not orphan,'Returned stage body lacks matching entered span')
 cancelled=bool(finish['cancelled'])
 if not cancelled:
  require(enter==returned,'Noncancelled completed request has unmatched/duplicate entered or returned spans')
  expected=ids+request['output_ids'][:-1]
  require(commit['ids']==expected,'Actual ordinary serial committed token prefix differs from GEN input plus consumed outputs')
  for bounds in stage_ranges:
   covered=sorted(i for key,count in returned.items() if (key[2],key[3])==tuple(bounds) for repeat in range(count) for i in range(max(finish['actual_reused'],key[4]),min(len(ids),key[5])))
   require(covered==list(range(finish['actual_reused'],len(ids))),'Actual stage prompt coverage overlaps or misses rows')
 else:
  require(commit['phase']==request['cancel_requested'],'Cancellation occurred outside requested phase')
 return {'passed':True,'scope':'strengthened observed native serial request/body/raw binding; not complete state or commit proof','prefix_pcl_sfd_identity':list(identity),'unbound_descriptive_prefix_events':[e['event'] for e in prefixes if 'pid' not in e],'actual_input_sha256_le32':digest(ids),'input_tokens':len(ids),'entered_spans':sum(enter.values()),'returned_spans':sum(returned.values()),'exact_enter_return_equal':enter==returned,'cancelled':cancelled,'partial_entries_explicitly_unobserved':[{'key':list(key),'count':count} for key,count in sorted(missing.items())],'cancelled_partial_work_is_zero':False,'full_logits_scope':diagnostic,'activations_scope':activations}


def audit(root):
 report={'CONFIG':'CPU read-only external strengthened audit of actual emerging V6 files; no source/run-file mutation','COMMAND':'python3 strata/flash-next/audit_v6_actual_request_records.py --root RUN --output EXTERNAL_RECEIPT','RESULT':[],'VERDICT':'pending actual completed records','root':str(root),'errors':[],'incomplete_records':[],'complete_model_state_proven':False}
 plan_path=root/'plan.snapshot.json'
 if not plan_path.exists():report['incomplete_records'].append('child plan absent');return report
 plan=frozen.read(plan_path);require(plan['controller_sha256']==frozen.sha(Path(frozen.__file__)),'Frozen actual controller differs')
 report['plan_sha256']=frozen.sha(plan_path);ranges=frozen.expected_stage_ranges(plan['args'])
 for process in sorted(p for p in root.iterdir() if p.is_dir()):
  command=process/'command.json'
  if not command.exists():continue
  args=frozen.read(command);env=dict(word.split('=',1) for i,word in enumerate(args) if i and args[i-1]=='-e');diagnostic=env.get('STRATA_FIDELITY_DIAG')=='1';activations=env.get('STRATA_FIDELITY_DIAG_ACTIVATIONS')=='1'
  for path in sorted(process.glob('*.json')):
   if path.name in {'command.json','requests.json','result.json','continuation-seed.json','continuation-token-receipt.json','keyed-eviction.json'}:continue
   try:request=frozen.read(path)
   except json.JSONDecodeError:report['incomplete_records'].append(str(path));continue
   if not isinstance(request,dict) or 'stderr' not in request or 'ids' not in request:continue
   if request.get('label','').startswith('warm-'):continue
   try:
    require(request.get('done','').startswith('DONE '),'Request has no terminal DONE record')
    # Existing finite/raw/geometry checks are retained. Only an in-memory copy
    # is mutated by frozen extract; no source evidence file is rewritten.
    frozen.extract(copy.deepcopy(request),process/'captures',diagnostic,activations,ranges)
    result=strengthen(request,ranges,diagnostic,activations);result.update(path=str(path),request_sha256=frozen.sha(path),process=process.name,label=request['label']);report['RESULT'].append(result)
   except (ValueError,KeyError,IndexError,TypeError) as error:report['errors'].append({'path':str(path),'error':str(error)})
 report['passed']=bool(report['RESULT']) and not report['errors'] and not report['incomplete_records']
 report['VERDICT']='PASS observed completed records only; run/health/full identity/equivalence may remain pending' if report['passed'] else 'FAILED observed record check' if report['errors'] else 'PENDING completed records'
 report['pending_processes']=[p.name for p in root.iterdir() if p.is_dir() and not (p/'result.json').exists()]
 return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.root)
 require(not a.output.resolve().is_relative_to(a.root.resolve()),'Strengthened evidence must be outside frozen live run tree')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(result['VERDICT'],'records',len(result['RESULT']),'errors',len(result['errors']))
 if result['errors']:raise SystemExit(1)
if __name__=='__main__':main()
