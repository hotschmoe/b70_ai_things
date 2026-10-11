"""Future exact174 plus one genuine43 cache alias; never a canonical write."""
import hashlib
from pathlib import Path
from strict_registry_yaml_v3 import parse
from registry_c140_shared_association_v3 import rendered_fragment
import registry_source43_api174_association_v1 as prior
HERE=Path(__file__).resolve().parent
ALIAS='qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-native-hc-ple-mtp0-source43-materializedhalf1-fullstatecache-fp16kv-ctx2048-prefill64-ple65536-seg1024-adapt0-borrow0-gpu0'
DELTA=HERE/'source43-cache-registry-proposal-v1.yaml'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def expected_prior_bytes():return prior.proposed_bytes()
def expected_bytes():return expected_prior_bytes()+b'\n'+rendered_fragment(DELTA.read_bytes())
def association(path,enforce=True):
 require(enforce is True,'Exact full preservation always required')
 raw=Path(path).read_bytes();require(raw==expected_bytes(),'Exact actual174 plus declared source43 cache alias required');old=parse(expected_prior_bytes());new=parse(raw);require(len(old['models'])==174 and len(new['models'])==175 and new['models'][:174]==old['models']and list(old)==list(new)and{k:v for k,v in old.items()if k!='models'}=={k:v for k,v in new.items()if k!='models'}and new['models'][-1]['served_model_id']==ALIAS and new['models'][-1]['primary_client_id']=='hotschmoe-dd','Exact full174 preservation and stable primary required');return {'current_registry_sha256':hashlib.sha256(raw).hexdigest(),'all174_complete_rows_preserved':True,'cache_alias':ALIAS,'actual_models':175,'old174_current_global_gate_passed':False,'old_runtime_or_math_proof_transferred':False}
