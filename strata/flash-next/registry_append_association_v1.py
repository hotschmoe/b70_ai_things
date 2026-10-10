"""Explicit append-only registry association; never reports old globalSHA current."""
import copy,hashlib,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASELINE=HERE/'api-fresh-migration-registry-baseline-v1.yaml'
APPEND=HERE/'api-fresh-migration-registry-append-proposal-v1.yaml'
BASELINE_SHA='56a6dd64cd174f9c55d9ea743c8f396a535c19c73d1a2a016d87b45cfb7b5852'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def association(current,old_global_sha=BASELINE_SHA):
 old=BASELINE.read_bytes();extra=APPEND.read_bytes();raw=Path(current).read_bytes();require(digest(old)==old_global_sha==BASELINE_SHA,'Actual old registry identity mustmatch preserved immutable source')
 require(raw==old+extra,'Only exact reviewed appended aliases admitted; old registry bytes cannotchange')
 ids=lambda b:re.findall(rb'^\s*served_model_id:\s*(\S+)\s*$',b,re.M)
 oldids,newids=ids(old),ids(extra);require(len(newids)==12 and len(set(newids))==12 and not set(oldids)&set(newids),'Exact twelve disjoint new aliases required')
 return {'old_global_sha256':digest(old),'actual_current_global_sha256':digest(raw),'append_sha256':digest(extra),'old_registry_bytes_preserved_exactly':True,'new_aliases':[x.decode('ascii') for x in newids],'old_global_digest_still_current':False,'old_V7_current_globalSHA_gate_passed':False,'old_evidence_requalified':False,'scope':'explicit entry/byte preservation association only; old strict readers remain unchanged'}

def historical_entry_association(plan):
 binding=plan['registry_binding'];require(binding['sha256']==BASELINE_SHA,'Only independently preserved actual56a6 registry identity admitted')
 canonical=HERE.parents[1]/'evals/configs/models.yaml';require(Path(binding['path']).resolve()==canonical.resolve(),'Actual original registry path differs')
 ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',BASELINE.read_text(),re.M);require(ids.count(binding['served_model_id'])==1 and plan['research_alias']==binding['served_model_id'],'Original accepted alias is not in preserved actual old registry')
 return association(canonical,binding['sha256'])

def admit_historical_v7_plan(plan):
 """Explicit NEW registry-only admission view, not original validator success.

 Consumers still must bind actual immutable plan snapshots and all raw/current
 runtime evidence using their normal closed-evidence reader. No producer or
 original plan file receives this view. All original nonregistry gates run.
 """
 import batch_numerical_execution_v7 as frozen
 proof=historical_entry_association(plan);view=copy.deepcopy(plan);view['registry_binding']['sha256']=proof['actual_current_global_sha256'];before=copy.deepcopy(plan)
 binding=frozen.manifest_binding(view);require(plan==before,'Original plan changed during named association admission')
 return {'admission_kind':'NEW_registry_only_V7_view','original_global_gate_passed':False,'original_plan_mutated':False,'changed_view_fields':['registry_binding.sha256'],'accepted_entry_association':proof,'other_frozen_V7_admission_gates':binding,'actual_original_runtime_raw_recollection_still_required':True,'measurement_requalified_by_registry_association':False}
