"""Unintegrated pure-subset call adapter; full source-case and live gates stay outside."""
from pathlib import Path
from logical_free_immutable_epoch_v2 import LogicalEpoch,require

def logical_case(epoch,upload_root,case_row,*,current_roster):
 require(isinstance(epoch,LogicalEpoch),'Explicit same-process immutable byte parser epoch required')
 # Future original_upload_gate successor calls this only after unchanged
 # state/full_source_case_gate/bounds checks. No marker or caller bool replaces them.
 device=case_row['report'];expected=sum(stage['unique_allocations']+1 for stage in device['stages']);root=Path(upload_root).resolve();name=case_row['case'];require(type(name)is str and name and '/'not in name and name not in ('.','..'),'Exact case basename required')
 return epoch.collect(root/(name+'.log'),root/(name+'-logical-free.json'),expected,current_roster=current_roster)
