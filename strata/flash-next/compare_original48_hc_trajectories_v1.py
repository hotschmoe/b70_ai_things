"""Readonly report-only phase error trajectories; no model or numeric qualification."""
import json,hashlib
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(prior,candidate):
 p=read(prior);c=read(candidate)
 if p['errors'] or c['errors'] or p['native_observation_binding']!=c['native_observation_binding']:raise ValueError('Completed identical native target reports required')
 a=p['exploration']['comparisons'];b=c['exploration']['comparisons']
 if set(a)!=set(b) or len(a)!=577:raise ValueError('Exact577 fullprefix4 phase/head roster required')
 rows=[]
 for key in a:
  if a[key]['numeric_gate_assigned'] is not False or b[key]['numeric_gate_assigned'] is not False:raise ValueError('No numeric threshold gate admitted')
  rows.append({'scope':key,'prior_nmse':a[key]['nmse'],'candidate_nmse':b[key]['nmse'],'prior_bitwise_equal':a[key]['bitwise_equal'],'candidate_bitwise_equal':b[key]['bitwise_equal']})
 return {'prior_report_sha256':sha(prior),'candidate_report_sha256':sha(candidate),'rows':rows,'numeric_pass_claim':False,'full_model_math_qualified':False,'threshold_assigned':False}
