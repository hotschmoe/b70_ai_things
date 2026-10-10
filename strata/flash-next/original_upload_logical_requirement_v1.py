"""Unintegrated original upload pure-subset requirement; all live guards stay outside."""
from pathlib import Path
from logical_free_require_case_epoch_v1 import RequireLogicalEpoch,require

def logical_case(epoch,upload_root,case_row,*,current_roster):
 require(type(epoch)is RequireLogicalEpoch,'Explicit compact immutable logical epoch required');device=case_row['report'];expected=sum(stage['unique_allocations']+1 for stage in device['stages']);name=case_row['case'];require(type(name)is str and name and '/'not in name and name not in ('.','..'),'Exact case basename required');root=Path(upload_root).resolve();return epoch.require_case(root/(name+'.log'),root/(name+'-logical-free.json'),expected,current_roster=current_roster)
