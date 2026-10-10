"""Saved runtime witness structure plus current explicit byte epoch; no load trust."""
import math
from serial37_canonical_json_v3 import canonical
from operation_pack_hash_witness_v2 import PackEpoch,require

def binding(proof,epoch,owner_pid,predevice_deadline,postoperation_floor):
 require(isinstance(epoch,PackEpoch),'Fresh current-process byte epoch required for saved witness join');epoch._owned()
 require(type(proof['owner_pid'])is int and proof['owner_pid']==owner_pid and type(proof['semantic_digest_calls'])is int and proof['semantic_digest_calls']>=1,'Exact runtime witness owner/semantic calls differs')
 for field,value in {'saved_digest_imported':False,'stat_only_validation':False,'between_boundaries_mutation_unobserved':True,'actual_GPU_execution_proven':False,'full_semantic_admission_proven':False,'optimization_integrated_into_runtime':False}.items():require(proof[field]is value,'Witness boundary/scope flag differs '+field)
 rows=proof['boundaries'];require(len(rows)==3 and [r['label'] for r in rows]==['admission_start','predevice_complete_byte_recheck','postoperation_complete_byte_recheck'],'Exact three complete byte boundaries required')
 last=float('-inf')
 for row in rows:
  require(all(type(row[k])in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and last<=row['started_epoch']<=row['finished_epoch'],'Saved witness finite chronology differs');last=row['finished_epoch']
  require(canonical(row['rows'])==canonical(epoch.initial),'Saved expected/current full-byte rows or stat5 differ')
 require(proof['unique_pack_files']==len(epoch.expected) and rows[1]['finished_epoch']<=predevice_deadline and rows[2]['started_epoch']>=postoperation_floor,'Actual witness device/terminal boundary differs')
 return {'owner_pid':owner_pid,'complete_byte_boundaries':3,'unique_pack_files':len(epoch.expected),'current_initial_byte_read_rejoined':True,'current_epoch_must_still_seal_and_finalize':True,'between_boundaries_mutation_unobserved':True}
