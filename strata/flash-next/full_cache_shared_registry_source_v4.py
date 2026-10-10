"""Named exact171 admission alongside unchanged historical source-byte checks."""
import hashlib,json
from pathlib import Path
from registry_c140_shared_association_v3 import association,BASE_SHA,BASE,C140,rendered_fragment
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PRIOR_PLAN=HERE/'full-cache-shared-runtime-source-plan-v3.json'
PRIOR_SHA='b6ea1f6e1ede49c4111cbb5cf35755c1cd7645297efa7fd934f6bb03e9233936'
REGISTRY='evals/configs/models.yaml'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_files(files,root,current_registry):
 require(files.get(REGISTRY)==BASE_SHA,'Exact historical original165 registry declaration required')
 for name,wanted in files.items():
  if name==REGISTRY:continue
  require(sha(Path(root)/name)==wanted,'Unchanged inherited source-byte predicate failed '+name)
 current=association(current_registry,True)
 return {'current_registry_association':current,'historical_registry_declaration_sha256':BASE_SHA,'original_global_hash_gate_passed':False,'historical_registry_digest_substituted':False,'old_runtime_proof_transferred':False}
def inherited_source():
 require(sha(PRIOR_PLAN)==PRIOR_SHA,'Frozen sharedV3 historical ledger changed');plan=json.loads(PRIOR_PLAN.read_bytes());require(len(plan['files'])==626,'Exact inherited source/test/provenance roster required');result=verify_files(plan['files'],ROOT,ROOT/REGISTRY)
 return {'historical_source_plan_sha256':PRIOR_SHA,'all625_nonregistry_source_bytes_rechecked':True,**result}
def prepared169_binding(prepared):
 from strict_registry_yaml_v3 import parse
 expected=hashlib.sha256(BASE.read_bytes()+rendered_fragment(C140.read_bytes())).hexdigest();entries=parse(C140.read_bytes());require(prepared['registry_sha256']==expected and prepared['alias']in [r['served_model_id']for r in entries],'Actual prepared C140169 receipt must match exact baseline165 plus source40four bytes/entry')
 current=association(ROOT/REGISTRY,True)
 return {'actual_prepared_registry_sha256':prepared['registry_sha256'],'actual_prepared_models':169,'actual_current_registry':current,'exact_source40_entries_preserved_across169_to171':True,'original_global_hash_gate_passed':False,'prepared_digest_rewritten':False}
