"""Actual two-request cached profiling protocol; no profiling arithmetic claims."""
import math
from pathlib import Path
from source61_ple_clean_abba_contract_v1 import c,require,B17,REFERENCES

def requests(rows,expected):
 require(len(rows)==len(expected)==2,'One warmup and one selected cached owner required');result=[]
 reference=c.read(B17/REFERENCES[0][0]);require(c.sha(B17/REFERENCES[0][0])==REFERENCES[0][1],'Recorded first natural history changed');gold=reference['arms'][0]['natural_observations'][0]['output_ids'][:8]
 for row,want in zip(rows,expected):
  raw=row['raw'];rid=want['rid'];fresh=want['fresh'];reuse=want['expected_reused'];require(row['ordinal']==want['ordinal'] and raw['profile_request_rid']==rid and raw['ids']==want['ids'] and len(raw['ids'])==2048 and raw['pin']==1536 and raw['fresh']==fresh and not raw['LP'] and raw['stop_sent'] is False and raw['cancel_requested'] is None,'Exact current cached scope required')
  require(raw['command']=='GEN 8 rid=%d fresh=%d pin=1536 ckpt=1 temperature=0 '%(rid,fresh)+' '.join(map(str,want['ids'])),'Exact actual GEN/RID/cache packet required')
  done=raw['done'].split();require(len(done)==16 and done[0]=='DONE' and int(done[1])==8 and int(done[2])==2048 and int(done[8])==reuse and int(done[14])==2048-reuse and done[5]=='length','Actual output/input/reused/new/finish census differs')
  tokens=[int(x.split()[1]) for x in raw['stdout'] if x.startswith('T ')];require(tokens==raw['output_ids']==gold and len(tokens)==8,'Current fresh/cached eight outputs differ from full natural reference')
  require([x for x in raw['stdout'] if x.startswith('REUSED ')]==['REUSED '+str(reuse)] and [x for x in raw['stdout'] if x.startswith('RESUME ')]==['RESUME '+str(reuse)],'Native observed cache markers differ')
  require(not any(x.startswith(('ERR','FATAL','LP ')) for x in raw['stdout']) and raw['stdout'][-1]==raw['done'],'Native error or trailing protocol data')
  points=raw['profile_RAW_receive_rows'];require([x['line'] for x in points]==raw['stdout'] and all(x['rid']==rid for x in points) and [x['host_RAW_ns'] for x in points]==sorted(x['host_RAW_ns'] for x in points),'Actual RAW protocol clock/owner association differs')
  require(raw['profile_call_RAW_begin_ns']<=raw['profile_send_RAW_before_ns']<=raw['profile_send_RAW_after_ns']<=points[0]['host_RAW_ns']<=points[-1]['host_RAW_ns']<=raw['profile_call_RAW_end_ns'],'Actual packet/receive RAW chronology differs')
  require(len(raw['stdout_seconds'])==len(raw['stdout']) and all(math.isfinite(x) and x>=0 for x in raw['stdout_seconds']),'Original MONOTONIC request timings missing')
  result.append(dict(rid=rid,ids=raw['ids'],output_ids=tokens,finish=done[5],actual_reused=reuse,new_prompt_rows=2048-reuse,RAW_call_begin_ns=raw['profile_call_RAW_begin_ns'],RAW_first_token_receive_ns=next(x['host_RAW_ns'] for x in points if x['line'].startswith('T ')),RAW_DONE_receive_ns=points[-1]['host_RAW_ns'],clock_alignment_qualified=False,full_internal_host_coverage_qualified=False))
 require(result[0]['output_ids']==result[1]['output_ids'],'Actual fresh/cached outputs differ');return result
