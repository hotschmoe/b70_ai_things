"""Saved logical byte/provenance witness plus independent current epoch join."""
import math
from serial37_canonical_json_v3 import canonical
from logical_free_require_case_epoch_v2 import RequireLogicalEpoch

def require(ok,message):
 if not ok:raise ValueError(message)
def binding(proof,epoch,owner_pid,predevice_deadline,postoperation_floor,*,ready_boundary=None,ack_epoch=None):
 require(type(epoch)is RequireLogicalEpoch,'Independent current logical epoch required');RequireLogicalEpoch.implementation_binding(epoch);epoch.evidence._owned();require(proof['passed']is True and type(proof['owner_pid'])is int and proof['owner_pid']==owner_pid and not proof['worker_errors'] and proof['live_guards_reused']is False and proof['saved_result_imported']is False,'Actual saved logical owner/failure/scope differs');require(canonical(proof['worker_runtime'])==canonical(epoch.runtime),'Actual saved/current worker runtime closure differs');w=proof['evidence_byte_witness'];require(w['owner_pid']==owner_pid and w['saved_digest_or_result_imported']is False and w['stat_only_validation']is False and w['live_guards_reuse_allowed']is False,'Actual complete logical byte source scope differs');rows=w['boundaries'];labels=['entry','predevice','postoperation']if ready_boundary is None else ['entry','ready','predevice','postoperation'];require([r['label']for r in rows]==labels,'Exact independent logical complete byte edges required');last=float('-inf')
 for row in rows:
  require(all(type(row[k])in (int,float)and math.isfinite(row[k])for k in ('started_epoch','finished_epoch')) and last<=row['started_epoch']<=row['finished_epoch'],'Logical complete byte chronology invalid');last=row['finished_epoch'];require(canonical(row['rows'])==canonical(epoch.evidence.initial),'Saved/current complete logical source/runtime/input bytes differ')
 seal=rows[-2];require(seal['finished_epoch']<=predevice_deadline and rows[-1]['started_epoch']>=postoperation_floor,'Logical bytes must bracket real device/terminal boundaries')
 if ready_boundary is not None:require(canonical(rows[1])==canonical(ready_boundary) and rows[1]['finished_epoch']<=ack_epoch<=seal['started_epoch'],'Actual logical READY/ACK/fresh reseal association differs')
 require(type(proof['semantic_executions'])is int and proof['semantic_executions']>0 and proof['parse_counts_complete']is True and proof['positive_parse_count']>=proof['semantic_executions'] and proof['negative_parse_count']==3*proof['positive_parse_count'],'Genuine logical successful worker counts absent')
 for row in proof['worker_commands']:
  require(row['worker_passed']is True and type(row['return_code'])is int and row['return_code']==0 and row['parse_counts']=={'positive':1,'negative':3} and row['worker_identity']['parent_pid']==owner_pid and row['worker_identity']['pid']==row['child_pid'] and row['worker_retirement']['process_terminal']is True and row['worker_retirement']['owned_session_empty']is True and row['stdout_stderr_regular_sinks_closed']is True,'Actual logical worker command/owner/terminal differs')
 return {'actual_logical_owner_pid':owner_pid,'complete_byte_boundaries':len(rows),'current_complete_byte_roster_rejoined':True,'live_guards_cached':False,'full_math_quality_or_serving_speed_qualified':False}
