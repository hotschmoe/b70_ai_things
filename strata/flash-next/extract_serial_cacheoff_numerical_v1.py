"""Purpose-specific fresh cacheOFF GEN1 source49 audit; no cache qualification.
Frozen strict numerical/SFD coverage is reused; committed_live flags stay real.
"""
import collections,json,re
import serial_prefix_qualification_v6 as old
import serial_prefix_qualification_v8 as protocol
require=old.require

def flag(args,name):
 require(args.count(name)==1 and args.index(name)+1<len(args),'Exact cacheOFF option required '+name);return args[args.index(name)+1]

def profile(args,env):
 require(flag(args,'--batch')=='0' and flag(args,'--prompt-cache')=='0' and flag(args,'--conversation-cache-mib')=='0','Numerical cacheOFF/serial profile differs')
 require('--mtp' not in args and '--pipeline-windows' not in args and flag(args,'--suffix-draft')=='0' and flag(args,'--lookup-chain')=='0','Unqualified numerical route/decoder proposals')
 require('STRATA_VERIFY_EAGER' not in env and 'STRATA_CKPT_REREAD' not in env and 'STRATA_STATE_HASH' not in env,'Normal graph/no reread/cache state override presence required')
 for key,value in {'STRATA_BATCH_FIDELITY_DIAG':'0','STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1','STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1','STRATA_LAYER0_Q8_DIAG':'0','STRATA_PREFIX30':'0','STRATA_PLE_INPUT33':'0','STRATA_BATCH_FULL_STATE_CHAIN':'0','STRATA_BATCH_PUBLIC_PREFIX':'0'}.items():require(env.get(key,'0')==value,'Exact numerical/cacheOFF source flag differs '+key)
 return True

def extract(request,capture_root,args,env,stage_ranges,allow_legacy_pin_zero=False):
 profile(args,env);ids=request['ids'];require(type(request['fresh']) is int and request['fresh']==1 and request['cancel_requested'] is None and request['stop_sent'] is False,'Fresh numeric-only request required')
 expected=protocol.request_command(ids,1,1,None);legacy=protocol.request_command(ids,1,1,0);require(request['command']==expected and request['pin'] is None or allow_legacy_pin_zero and request['command']==legacy and type(request['pin']) is int and request['pin']==0,'Actual GEN1 fresh1 absentpin required; legacyzero is readonly recovery only')
 done=request['done'].split();require(done[0]=='DONE' and int(done[1])==len(request['output_ids'])==1 and int(done[2])==len(ids) and done[5] in ('length','stop') and len(done)>=15 and int(done[8])==0 and int(done[14])==len(ids),'Actual numeric nout/nread/finish differs')
 result=old.extract_numeric(request,capture_root,True,True,stage_ranges);ledger=result['ledger'];selection=result['selection'];require(ledger['actual_reused']==ledger['candidate_resume']==0 and ledger['evaluated_prompt_rows']==len(ids) and ledger['evaluated_decode_rows']==0 and ledger['generated']==1 and ledger['cancelled'] is False and ledger['finish']==done[5],'Actual numeric fresh prompt/output work differs')
 require(selection['actual_reused']==selection['candidate_resume']==selection['parked_bytes']==selection['parked_entries']==selection['evictions']==selection['checkpoints']==0 and selection['reread'] is False,'Cache-disabled actual selection/reuse differs')
 events=[json.loads(line[4:]) for line in request['stderr'] if line.startswith('PCL ')];begins=[e for e in events if e['event']=='begin'];commits=[e for e in events if e['event']=='committed_live'];require(len(begins)==len(commits)==1,'Numerical begin/complete provenance required');begin,commit=begins[0],commits[0];starts=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in request['stderr'] if line.startswith('SFD request ')];require(len(starts)==1 and int(starts[0]['pid'])==begin['pid'] and int(starts[0]['request'])==begin['request'],'Numerical PCL and SFD process/request differ')
 for event in events:require(event['pid']==begin['pid'] and event['request']==begin['request'] and event['event'] not in ('skip','cache'),'Numerical source identity/quota/cache event differs')
 require(begin['tokens']==len(ids) and begin['input_sha256_le32']==old.le32_digest(ids),'Actual numerical input IDs differ')
 require(commit['phase']=='complete' and commit['finish']==done[5] and commit['published'] is False and commit['chain_updated'] is True and commit['live_reusable'] is False,'Exact cacheOFF committed_live tuple must(false,true,false)')
 require(commit['ids_truncated'] is False and commit['tokens']==len(ids) and commit['ids']==ids and commit['sha256_le32']==old.le32_digest(ids),'Numeric GEN1 complete consumed-prefix IDs differ')
 spans=[event for event in events if event['event']=='stage_span'];key=lambda e:(e['phase'],e['device'],e['lb'],e['le'],e['lo'],e['hi']);entered=collections.Counter(key(e) for e in spans if not e['complete']);returned=collections.Counter(key(e) for e in spans if e['complete'])
 require(entered==returned,'Numeric stage entry/return pairing incomplete')
 for event in spans:
  bounds=(event['lb'],event['le']);require(bounds in stage_ranges and event['device']==stage_ranges.index(bounds),'Numeric actual stage owner differs')
 for bounds in stage_ranges:
  covered=sorted(i for e in spans if e['complete'] and (e['lb'],e['le'])==tuple(bounds) for i in range(max(0,e['lo']),min(len(ids),e['hi'])));require(covered==list(range(len(ids))),'Numeric fullprompt stage spans missing/duplicate')
 result['numerical_cacheOFF_lifecycle']={'begin':begin,'commit':commit,'stage_spans':spans,'cache_events_observed':False,'live_chain_reusable_observed':False,'cache_qualification_granted':False,'historical_legacy_pin_zero_replay':request['command']==legacy,'complete_state_mathematics_qualified':False}
 return result
