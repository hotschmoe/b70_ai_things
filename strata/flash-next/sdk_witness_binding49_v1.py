"""Saved SDK runtime proof joined to fresh current operation bytes; no load trust."""
import math
from operation_sdk37_byte_witness_v1 import SDK37Epoch,require
from serial37_canonical_json_v3 import canonical

def binding(proof,epoch,owner_pid,predevice_deadline,postoperation_floor):
 require(isinstance(epoch,SDK37Epoch),'Fresh current-process SDK37Epoch required');epoch._owned()
 require(type(proof['owner_pid'])is int and proof['owner_pid']==owner_pid and proof['unique_executables']==9 and type(proof['digest_calls'])is int and proof['digest_calls']>=1 and type(proof['ELF_magic_calls'])is int and proof['ELF_magic_calls']>=0,'Actual SDK owner/roster/counter differs')
 for name,value in {'stat_only_validation':False,'saved_digest_imported':False,'between_byte_boundaries_mutation_unobserved':True,'full_source_header_recipe_health_or_runtime_admission_proven':False,'source37_ELF_marker_scan_invented':False,'actual_runtime_integration':False}.items():require(proof[name]is value,'SDK witness scope flag changed '+name)
 boundaries=proof['boundaries'];require(len(boundaries)==3 and [r['label'] for r in boundaries]==['entry','predevice','postoperation'],'Exactly three actual complete SDK byte observations required');last=float('-inf')
 for row in boundaries:
  require(all(type(row[k])in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','finished_epoch')) and last<=row['started_epoch']<=row['finished_epoch'],'SDK byte boundary finite chronology differs');last=row['finished_epoch'];require(canonical(row['rows'])==canonical(epoch.initial) and canonical(row['receipt_inputs'])==canonical(epoch.current_receipts),'Saved/current complete SDK bytes/receipt/stat5 differs')
 require(boundaries[1]['finished_epoch']<=predevice_deadline and boundaries[2]['started_epoch']>=postoperation_floor,'Actual SDK device/terminal byte boundary differs')
 return {'owner_pid':owner_pid,'complete_SDK_byte_boundaries':3,'unique_executables':9,'current_entry_bytes_rejoined':True,'current_epoch_seal_and_post_byte_scan_still_required':True,'between_byte_boundaries_mutation_unobserved':True}
