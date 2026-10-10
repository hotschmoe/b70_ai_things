"""Exact NEW source40 append; no original digest or predicate is overridden."""
import hashlib
from pathlib import Path
from strict_registry_yaml_v3 import parse
HERE=Path(__file__).resolve().parent
BASE=HERE/'c139-registry-prior165-v1.yaml'
BASE_SHA='86621c71b829c75cc2f7312248932e9a8bb276018db3c0bef77f1fdb58678052'
C140=HERE/'c1-combined-v140-model-registry-proposal-v3.yaml'
SHARED=HERE/'full-cache-shared-source40-registry-proposal-v3.yaml'
C140_SHA='94f8da9093ef6cfd549e795d1d18563cd08ff67f797183ab611614a6ac9069df'
SHARED_SHA='5593ad0004910358a20990f10a3eaba15e05ff533c54d2dddc11d7df2462559a'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rendered_fragment(raw):
 require(raw.endswith(b"\n"),'Complete source list fragment required');return b''.join(b'  '+line if line.strip() else line for line in raw.splitlines(keepends=True))

def association(current,include_shared=False):
 require(type(include_shared)is bool,'Explicit registry scope required');base=BASE.read_bytes();delta=C140.read_bytes();purpose=SHARED.read_bytes();raw=Path(current).read_bytes();require(sha(base)==BASE_SHA and sha(delta)==C140_SHA and sha(purpose)==SHARED_SHA,'Exact retained165/source40/purpose proposal bytes required');require(raw==base+rendered_fragment(delta)+(rendered_fragment(purpose) if include_shared else b''),'Only exact deliberate source40 append is admitted')
 before=parse(base);added=parse(delta);extra=parse(purpose) if include_shared else [];after=parse(raw);require(type(before)is dict and len(before['models'])==165 and type(added)is list and len(added)==4 and len(extra)==(2 if include_shared else 0),'Exact source40 entry rosters required')
 require(list(after)==list(before) and {k:v for k,v in after.items() if k!='models'}=={k:v for k,v in before.items() if k!='models'} and after['models']==before['models']+added+extra and len({r['served_model_id'] for r in after['models']})==len(after['models']),'Every original topkey/entry/order and unique source40 alias must remain exact')
 require(all(r['primary_client_id']=='hotschmoe-dd' for r in added+extra),'Stable primary ID required')
 return {'current_registry_sha256':sha(raw),'prior_registry_sha256':BASE_SHA,'source40_baseline_append_sha256':C140_SHA,'rendered_source40_append_sha256':sha(rendered_fragment(delta)),'purpose_append_sha256':SHARED_SHA if include_shared else None,'prior_models_preserved':165,'actual_models':len(after['models']),'source40_baseline_entries':4,'shared_purpose_entries':len(extra),'original_global_hash_gate_passed':False,'source37_or_C139_runtime_proof_transferred':False,'old165_runtime_consumers_need_separate_explicit_association':True,'new_source40_actual_proof_still_required':True}
