"""Immutable success-token API for consumers that never expose the parsed ledger."""
from pathlib import Path
import hashlib
from logical_free_immutable_epoch_v2 import LogicalEpoch,require,canonical
HERE=Path(__file__).resolve().parent
SOURCES=('logical_free_require_case_epoch_v1.py','original_upload_logical_requirement_v1.py')
class RequireLogicalEpoch(LogicalEpoch):
 def __init__(self,roster,max_seconds=900):
  paths={Path(p).resolve()for p,d in roster};require(all(HERE/name in paths for name in SOURCES),'Explicit compact consumer/source closure required');super().__init__(roster,max_seconds);self.success_tokens={};self.snapshot_sha={row['path']:row['sha256']for row in self.evidence.initial};self.require_calls=0;self.first_result_isolation_copies=0
 def require_case(self,log,logical,expected_owner_count,*,current_roster):
  self.owned(current_roster);require(not self.failed,'Failed parser epoch cannot reuse success token');require(type(expected_owner_count)is int and expected_owner_count>=0,'Exact typed expected owner count');log=Path(log).resolve();logical=Path(logical).resolve();base=dict(self.base_roster);require(log in base and logical in base and log!=logical,'Declared distinct compact case inputs required');key=canonical({'execution_context':self.context,'runtime':self.runtime_key,'source_bytes':{name:hashlib.sha256(self.evidence.bytes_for(HERE/name)).hexdigest()for name in SOURCES},'log_path':str(log),'log_sha256':self.snapshot_sha[str(log)],'logical_path':str(logical),'logical_sha256':self.snapshot_sha[str(logical)],'require_owners':True,'expected_owner_count':expected_owner_count});self.require_calls+=1
  if key not in self.success_tokens:
   full=self.collect(log,logical,expected_owner_count,current_roster=current_roster);require(full['original_upload_logical_subset_passed']is True,'Complete original logical subset must pass');self.first_result_isolation_copies+=1;self.success_tokens[key]=True
  # There is no shared mutable object to expose. Original complete-result
  # collect() API continues returning independent deep copies, unchanged.
  return True
 def finalize(self,*,current_roster):
  proof=super().finalize(current_roster=current_roster);proof.update(compact_requirement_calls=self.require_calls,first_full_result_isolation_copies=self.first_result_isolation_copies,compact_success_tokens=len(self.success_tokens),large_cached_result_exposed_by_requirement=False);return proof
