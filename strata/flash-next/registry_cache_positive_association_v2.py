"""NEW positive alias append association; no old globalSHA gate success claim."""
import copy,re
from pathlib import Path
from registry_append_association_v1 import BASELINE,BASELINE_SHA,APPEND as FRESH_APPEND,digest,require
HERE=Path(__file__).resolve().parent
APPEND=HERE/'api-cache-positive-registry-append-proposal-v2.yaml'
def association(current):
 old=BASELINE.read_bytes();fresh=FRESH_APPEND.read_bytes();extra=APPEND.read_bytes();raw=Path(current).read_bytes();require(digest(old)==BASELINE_SHA,'Independently retained actual56a6 baseline changed')
 require(raw in (old+extra,old+fresh+extra),'Only exact preserved old bytes plus reviewed optionalfresh/positive appends admitted')
 ids=lambda b:re.findall(rb'^\s*served_model_id:\s*(\S+)\s*$',b,re.M)
 require(len(ids(extra))==12 and len(set(ids(extra)))==12 and not set(ids(extra))&(set(ids(old))|set(ids(fresh))),'Exact disjoint twelve positivealiases required')
 return {'actual_current_global_sha256':digest(raw),'old_global_sha256':BASELINE_SHA,'old_bytes_preserved_exactly':True,'positive_append_sha256':digest(extra),'fresh_append_present':raw==old+fresh+extra,'new_positive_aliases':[x.decode('ascii') for x in ids(extra)],'old_V7_global_gate_passed':False,'frozen_fresh_V1_exact_append_gate_passed':False,'old_measurements_requalified':False}

def admit_historical_v7_plan(plan):
 import batch_numerical_execution_v7 as frozen
 binding=plan['registry_binding'];canonical=HERE.parents[1]/'evals/configs/models.yaml';require(Path(binding['path']).resolve()==canonical.resolve() and binding['sha256']==BASELINE_SHA,'Only original independently retained registry binding admitted')
 ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',BASELINE.read_text(),re.M);require(ids.count(binding['served_model_id'])==1 and plan['research_alias']==binding['served_model_id'],'Original accepted entry differs');proof=association(canonical);view=copy.deepcopy(plan);view['registry_binding']['sha256']=proof['actual_current_global_sha256'];result=frozen.manifest_binding(view)
 return {'admission_kind':'NEW_positive_registry_only_V7_view','original_global_gate_passed':False,'changed_view_fields':['registry_binding.sha256'],'accepted_entry_association':proof,'other_frozen_admission_gates':result,'original_runtime_raw_recollection_still_required':True,'measurement_requalified':False}
