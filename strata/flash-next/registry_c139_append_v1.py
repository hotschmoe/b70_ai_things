"""NEW exact169 registry association; never changes old global-hash gates."""
from pathlib import Path
import hashlib
from strict_registry_yaml_v3 import parse
HERE=Path(__file__).resolve().parent
BASE=HERE/'c139-registry-prior165-v1.yaml'
BASE_SHA='86621c71b829c75cc2f7312248932e9a8bb276018db3c0bef77f1fdb58678052'
DELTA=HERE/'c1-combined-v139-model-registry-proposal-v1.yaml'
DELTA_SHA='cbe7a3fbc883b84313b3a42c2fc0eb06866a515c5efb19797c9f029875132972'
def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)

def association(current):
 old=BASE.read_bytes();delta=DELTA.read_bytes();raw=Path(current).read_bytes();require(digest(old)==BASE_SHA and digest(delta)==DELTA_SHA,'Exact retained165/C139 list fragment changed');require(raw==old+delta,'Only exact declared C139 append permitted')
 before=parse(old);extra=parse(delta);after=parse(raw)
 require(type(before) is dict and type(extra) is list and len(before['models'])==165 and len(extra)==4 and len(after['models'])==169,'Exact165+4=169 semantic registry required')
 require(list(after)==list(before) and {k:v for k,v in after.items() if k!='models'}=={k:v for k,v in before.items() if k!='models'} and after['models']==before['models']+extra,'Every prior row/order/topkey/value must remain unchanged')
 require(len({r['served_model_id'] for r in after['models']})==169,'Actual169 unique aliases required')
 return {'schema':1,'current_registry_sha256':digest(raw),'prior_registry_sha256':BASE_SHA,'C139_append_sha256':DELTA_SHA,'prior_models_preserved':165,'C139_models_added':4,'actual_models':169,'old_global_gates_currently_qualified':False,'old_source37_runtime_proof_transferred':False,'full_model_math_qualified':False}
