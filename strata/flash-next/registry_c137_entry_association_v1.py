"""NEW exact C137 append association; old global gates remain explicitly false."""
import copy,re
from pathlib import Path
from registry_append_association_v1 import BASELINE as OLD56,BASELINE_SHA as OLD56_SHA,digest,require
from registry_cache_positive_association_v2 import APPEND as POSITIVE12
HERE=Path(__file__).resolve().parent
BASELINE=HERE/'registry-e0abd-positive-baseline-v1.yaml';BASELINE_SHA='e0abd69b8b991793cc2c2b78d0522aaec6c185a362309ed47982713f66b4dd28'
DELTA=HERE/'c1-combined-v137-model-registry-proposal.yaml';DELTA_SHA='a5935f170d67870915e0efbbb5de486d25851e53d2cc39603affefe425e62349'
def association(current):
 old=BASELINE.read_bytes();extra=DELTA.read_bytes();raw=Path(current).read_bytes();require(digest(old)==BASELINE_SHA and old==OLD56.read_bytes()+POSITIVE12.read_bytes() and digest(OLD56.read_bytes())==OLD56_SHA,'Exact independently retained positive E0abd/old56 bytes changed');require(digest(extra)==DELTA_SHA and raw==old+extra,'Only exact reviewed four-C137 append admitted; no other delta or old edit')
 ids=lambda b:re.findall(rb'^\s*served_model_id:\s*(\S+)\s*$',b,re.M)
 require(len(ids(extra))==4 and len(set(ids(extra)))==4 and not set(ids(extra))&set(ids(old)),'Exact four disjoint C137 aliases required')
 return {'old56_global_sha256':OLD56_SHA,'old_positive_global_sha256':BASELINE_SHA,'actual_current_global_sha256':digest(raw),'C137_append_sha256':DELTA_SHA,'every_original_entry_byte_and_order_preserved':True,'C137_new_aliases':[x.decode('ascii') for x in ids(extra)],'original_current_global_gates_passed':False,'original_reports_or_plans_mutated':False,'C137_SDK_or_math_proof_transferred':False,'measurements_requalified':False}

def historical_entry(plan,current):
 original=plan['registry_binding'];require(original['sha256'] in (OLD56_SHA,BASELINE_SHA),'Only exact independently retained old56/E0abd original association admitted');old=OLD56.read_bytes() if original['sha256']==OLD56_SHA else BASELINE.read_bytes();ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',old.decode('ascii'),re.M);require(ids.count(original['served_model_id'])==1 and plan['research_alias']==original['served_model_id'],'Original accepted entry absent or changed');require(Path(original['path']).resolve()==Path(current).resolve(),'Exact actual original canonical registry path required');return association(current)

def admit_historical_v7_plan(plan):
 import batch_numerical_execution_v7 as frozen
 proof=historical_entry(plan,HERE.parents[1]/'evals/configs/models.yaml');original=copy.deepcopy(plan);view=copy.deepcopy(plan);view['registry_binding']['sha256']=proof['actual_current_global_sha256'];other=frozen.manifest_binding(view);require(plan==original,'Original actual plan mutation forbidden')
 return {'admission_kind':'NEW_C137_registry_only_V7_view','original_current_global_gate_passed':False,'changed_view_fields':['registry_binding.sha256'],'entry_preservation':proof,'all_other_frozen_V7_admission_gates':other,'actual_original_raw_current_source_health_teardown_recollection_still_required':True,'measurements_requalified':False}
