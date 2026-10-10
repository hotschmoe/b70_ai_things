"""Unintegrated original upload pure-subset requirement; all live guards stay outside."""
from pathlib import Path
from logical_free_require_case_epoch_v2 import RequireLogicalEpoch,require

def logical_case(epoch,upload_root,case_row,*,current_roster):
 import logical_free_require_case_epoch_v2 as known
 if RequireLogicalEpoch is not known.RequireLogicalEpoch:raise ValueError('Known requirement class alias differs')
 known.RequireLogicalEpoch.implementation_binding(epoch)
 require(type(epoch)is RequireLogicalEpoch,'Explicit compact immutable logical epoch required');RequireLogicalEpoch.implementation_binding(epoch);device=case_row['report'];expected=sum(stage['unique_allocations']+1 for stage in device['stages']);name=case_row['case'];require(type(name)is str and name and '/'not in name and name not in ('.','..'),'Exact case basename required');root=Path(upload_root).resolve();return RequireLogicalEpoch.require_case(epoch,root/(name+'.log'),root/(name+'-logical-free.json'),expected,current_roster=current_roster)
