"""NEW exact C137+FULLbatch37 append association; old global gates remain explicitly false."""
import copy,re
from pathlib import Path
from strict_registry_yaml_v3 import semantic_append
from registry_append_association_v1 import BASELINE as OLD56,BASELINE_SHA as OLD56_SHA,digest,require
from registry_cache_positive_association_v2 import APPEND as POSITIVE12
HERE=Path(__file__).resolve().parent
BASELINE=HERE/'registry-e0abd-positive-baseline-v1.yaml';BASELINE_SHA='e0abd69b8b991793cc2c2b78d0522aaec6c185a362309ed47982713f66b4dd28'
DELTA=HERE/'c1-combined-v137-model-registry-proposal-v2.yaml';DELTA_SHA='08043ca0173fecb07b098c3757fa86a621251a31a78db5e29297e7590c074741'
BATCH_DELTA=HERE/'batch-source37-model-registry-proposal-v2.yaml';BATCH_DELTA_SHA='ded5e8d15c6e2cec36195841eb04dbf1dcd36e28b437145783f59918aaea8fc0'
def association(current):
 old=BASELINE.read_bytes();extra=DELTA.read_bytes();batch=BATCH_DELTA.read_bytes();raw=Path(current).read_bytes();require(digest(old)==BASELINE_SHA and old==OLD56.read_bytes()+POSITIVE12.read_bytes() and digest(OLD56.read_bytes())==OLD56_SHA,'Exact independently retained positive E0abd/old56 bytes changed');require(digest(extra)==DELTA_SHA and digest(batch)==BATCH_DELTA_SHA and raw==old+extra+batch,'Only exact E0abd +C137baseline4 +FULLbatch3712 in declared order admitted; no subset or old edit')
 semantic=semantic_append(old,extra,batch,raw)
 ids=lambda b:re.findall(rb'^\s*served_model_id:\s*(\S+)\s*$',b,re.M)
 require(len(ids(batch))==12 and len(set(ids(batch)))==12 and not set(ids(batch))&(set(ids(old))|set(ids(extra))),'Exact disjoint FULLbatch37 twelvealiases required')
 require(len(ids(extra))==4 and len(set(ids(extra)))==4 and not set(ids(extra))&set(ids(old)),'Exact four disjoint C137 aliases required')
 return {'semantic_registry_preservation':semantic,'old56_global_sha256':OLD56_SHA,'old_positive_global_sha256':BASELINE_SHA,'actual_current_global_sha256':digest(raw),'C137_append_sha256':DELTA_SHA,'batch37_append_sha256':BATCH_DELTA_SHA,'batch37_new_aliases':[x.decode('ascii') for x in ids(batch)],'every_original_entry_byte_and_order_preserved':True,'C137_new_aliases':[x.decode('ascii') for x in ids(extra)],'original_current_global_gates_passed':False,'original_reports_or_plans_mutated':False,'C137_SDK_or_math_proof_transferred':False,'measurements_requalified':False}

def historical_entry(plan,current):
 original=plan['registry_binding'];require(original['sha256'] in (OLD56_SHA,BASELINE_SHA),'Only exact independently retained old56/E0abd original association admitted');old=OLD56.read_bytes() if original['sha256']==OLD56_SHA else BASELINE.read_bytes();ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',old.decode('ascii'),re.M);require(ids.count(original['served_model_id'])==1 and plan['research_alias']==original['served_model_id'],'Original accepted entry absent or changed');require(Path(original['path']).resolve()==Path(current).resolve(),'Exact actual original canonical registry path required');return association(current)

def admit_historical_v7_plan(plan):
 import batch_numerical_execution_v7 as frozen
 proof=historical_entry(plan,HERE.parents[1]/'evals/configs/models.yaml');original=copy.deepcopy(plan);view=copy.deepcopy(plan);view['registry_binding']['sha256']=proof['actual_current_global_sha256'];other=frozen.manifest_binding(view);require(plan==original,'Original actual plan mutation forbidden')
 return {'admission_kind':'NEW_semantic_C137_batch37_registry_only_V7_view','original_current_global_gate_passed':False,'changed_view_fields':['registry_binding.sha256'],'entry_preservation':proof,'all_other_frozen_V7_admission_gates':other,'actual_original_raw_current_source_health_teardown_recollection_still_required':True,'measurements_requalified':False}

def current_entry(binding):
 """Actual new entry-specific association; does not assert old global digest current."""
 canonical=HERE.parents[1]/'evals/configs/models.yaml';require(Path(binding['path']).resolve()==canonical.resolve(),'Exact canonical registry association path required');proof=association(canonical)
 source={OLD56_SHA:OLD56.read_bytes(),BASELINE_SHA:BASELINE.read_bytes(),digest(BASELINE.read_bytes()+DELTA.read_bytes()):BASELINE.read_bytes()+DELTA.read_bytes(),proof['actual_current_global_sha256']:canonical.read_bytes()}
 require(binding['sha256'] in source,'Only independently retained declared historical/current snapshots admitted')
 ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',source[binding['sha256']].decode('ascii'),re.M);current_ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',canonical.read_text(),re.M);require(ids.count(binding['served_model_id'])==current_ids.count(binding['served_model_id'])==1,'Original accepted entry mustremain exactlyonce in actualcurrent reviewed bytes')
 return {'served_model_id':binding['served_model_id'],'original_global_sha256':binding['sha256'],'actual_current_global_sha256':proof['actual_current_global_sha256'],'entry_bytes_and_identity_preserved':True,'original_global_digest_still_current':binding['sha256']==proof['actual_current_global_sha256'],'original_whole_global_gate_passed':False,'append_association':proof,'actual_SDK_source_model_currentproof_still_required':True,'runtime_or_math_qualification_transferred':False}
