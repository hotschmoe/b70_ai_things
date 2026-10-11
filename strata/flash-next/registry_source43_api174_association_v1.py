"""Preregistered exact173+one alias association; never a global hash view."""
import hashlib
from pathlib import Path
from strict_registry_yaml_v3 import parse
from registry_c140_shared_association_v3 import rendered_fragment
from source43_api_baseline_profile_v1 import ALIAS
HERE=Path(__file__).resolve().parent;PRIOR=HERE/'source43-api-registry173-snapshot-v1.yaml';DELTA=HERE/'source43-api-registry-proposal-v1.yaml'
PRIOR_SHA='d4242070dff786a5977254f1ff0fa4deae6487ec308765aa2b4e5a4607234167'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def proposed_bytes():
 old=PRIOR.read_bytes();require(sha(old)==PRIOR_SHA,'Authentic173 snapshot changed');return old+b'\n'+rendered_fragment(DELTA.read_bytes())
def association(current):
 raw=Path(current).read_bytes();require(raw==proposed_bytes(),'Exact actual173 plus named onealias append required');before=parse(PRIOR.read_bytes());added=parse(DELTA.read_bytes());after=parse(raw)
 require(len(before['models'])==173 and type(added)is list and len(added)==1 and added[0]['served_model_id']==ALIAS and added[0]['primary_client_id']=='hotschmoe-dd'and list(before)==list(after)and{k:v for k,v in before.items()if k!='models'}=={k:v for k,v in after.items()if k!='models'}and after['models']==before['models']+added and len({r['served_model_id']for r in after['models']})==174,'Allold173 rows/topkeys/order/uniquealiases mustremain exact')
 return {'current_registry_sha256':sha(raw),'prior_registry_sha256':PRIOR_SHA,'actual_models':174,'all173_complete_rows_preserved':True,'new_alias':ALIAS,'old173_current_global_hash_gate_passed':False,'metadata_association_only':True,'old_runtime_or_math_proof_transferred':False}
