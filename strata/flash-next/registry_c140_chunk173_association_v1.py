"""Exact current173 append side proof; old global registry gates stay false."""
import hashlib
from pathlib import Path
from strict_registry_yaml_v3 import parse
from registry_c140_shared_association_v3 import BASE,BASE_SHA,C140,SHARED,rendered_fragment
HERE=Path(__file__).resolve().parent
PRIOR=HERE/'matched-geometry-registry171-snapshot-v1.yaml'
PRIOR_SHA='fec00c10a7fc29f882217dc8ceb0309616966a13dd81e0cef62da4b0c9c5a50c'
DELTA=HERE/'prefill-chunk64-512-registry-proposal-v1.yaml'
DELTA_SHA='da42e7b08d89f036c5300748573a97619aaeead9de4ddbd7e9bd0cf1d9d4c959'
CURRENT_SHA='d4242070dff786a5977254f1ff0fa4deae6487ec308765aa2b4e5a4607234167'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def association(current,include_shared=True):
 require(include_shared is True,'Exact full shared current173 association required');prior=PRIOR.read_bytes();delta=DELTA.read_bytes();raw=Path(current).read_bytes();require(sha(prior)==PRIOR_SHA and sha(delta)==DELTA_SHA,'Retained171/proposal bytes changed');require(raw==prior+b'\n'+rendered_fragment(delta)and sha(raw)==CURRENT_SHA,'Exact deliberate171 newline+twoalias append required')
 before=parse(prior);added=parse(delta);after=parse(raw);require(type(before)is dict and len(before['models'])==171 and type(added)is list and len(added)==2 and list(after)==list(before)and {k:v for k,v in after.items()if k!='models'}=={k:v for k,v in before.items()if k!='models'}and after['models']==before['models']+added and len({r['served_model_id']for r in after['models']})==173,'Exact original topkeys/whole entries/order/unique aliases required');require(all(r['primary_client_id']=='hotschmoe-dd'for r in added),'Stable primary alias required')
 return {'actual_models':173,'current_registry_sha256':CURRENT_SHA,'prior_registry_sha256':PRIOR_SHA,'declared_two_alias_append_sha256':DELTA_SHA,'all171_complete_rows_preserved':True,'original_global_hash_gate_passed':False,'old171_current_global_hash_gate_passed':False,'metadata_association_only':True,'runtime_or_math_proof_transferred':False}
def baseline_registry_binding(prepared,current):
 record=association(current);before=parse(PRIOR.read_bytes());now=parse(Path(current).read_bytes());aliases={r['served_model_id']for r in parse(C140.read_bytes())};alias=prepared['alias'];require(alias in aliases,'Actual retained C140 alias required');old=[r for r in before['models']if r['served_model_id']==alias];new=[r for r in now['models']if r['served_model_id']==alias];require(len(old)==len(new)==1 and old==new,'Whole actual C140 entry not preserved');digest=prepared['registry_sha256'];require(digest in(PRIOR_SHA,CURRENT_SHA),'Prepared baseline must name exact authentic171 or173 registry')
 return {'actual_current_registry':record,'actual_prepared_registry_sha256':digest,'actual_prepared_models':171 if digest==PRIOR_SHA else 173,'entire_C140_entry_preserved':True,'prepared_digest_or_original_proof_rewritten':False,'old171_current_global_hash_gate_passed':False,'new_GPU_health_authorization_transferred':False}
