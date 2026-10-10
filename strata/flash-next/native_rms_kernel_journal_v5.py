"""Rerun producer kernel fault refusal on original current journal text."""
import hashlib,re
from pathlib import Path
PATTERN=r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed'
def reject_faults(text):
 if re.search(PATTERN,text,re.I):raise ValueError('Actual kernel fault signature')
def admit(root,report):
 root=Path(root)
 for stage in ('pre','post'):
  name=stage+'-kernel.log';path=root/name;raw=path.read_bytes();row=report[stage+'_journal'];digest=hashlib.sha256(raw).hexdigest()
  if path.is_symlink() or row['path']!=str(path) or row['sha256']!=digest or report['artifact_sha256'].get(name)!=digest:raise ValueError('Actual original kernel log association differs')
  reject_faults(raw.decode('utf-8'))
 return {'original_pre_post_raw_kernel_fault_refusal_reexecuted':True}
