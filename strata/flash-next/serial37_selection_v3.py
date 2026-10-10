"""Explicit selected job roster; no silent skip or rewritten source job list."""
from pathlib import Path
from batch_numerical_proofs_v40 import require,read,sha
from serial37_canonical_json_v3 import matches_saved,read_unique

def selected_jobs(plan,jobs):
 group=jobs[plan['group_index']*6:plan['group_index']*6+6];indices=plan['actual_serial_selected_indices']
 require(indices==list(range(len(group))) or indices==list(range(1,len(group))) and len(group)==4,'Only complete group or declared remaining-three selection permitted')
 if indices!=list(range(len(group))):
  import adjudicate_serial37_cacheoff_first_v1 as recovery
  path=Path(plan['first_job_adjudication']['path']);saved=read_unique(path);require(sha(path)==plan['first_job_adjudication']['sha256'] and matches_saved(recovery.finalized_binding(),path),'Actual exact readonly first-job recovery changed')
  require(saved['job']==group[0] and saved['engine_receipt_sha256']==plan['engine_receipt_sha256'] and saved['collector_parent_sha256']==plan['batch_parent_sha256'] and saved['original_parent_passed'] is False and saved['new_absent_PIN_scope_qualified'] is False,'Actual first-job source/collector/historical exception association differs')
 else:require(plan['first_job_adjudication'] is None,'Complete fresh group must not borrow recovery')
 require(plan['actual_serial_job_count']==len(indices) and plan['actual_serial_submission_budgets']==[1]*len(indices),'Exact actual remaining submission roster/budget differs')
 return [group[i] for i in indices]
