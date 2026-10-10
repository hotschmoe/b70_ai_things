"""Explicit actual native-type diagnosis artifact prerequisite; no inference proof."""
import hashlib
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from operation_pack_hash_witness_v2 import require
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'paired-native-binding-types48-source-plan-v1.json'
PLAN_SHA='cf26331e4549db30eb6adf42e44d603dabf20d1c73d6af71aa866b2ee6459165'
ACTUAL_RECEIPT_SHA='6ed05cab321e67733043fd22541911d3b320aa0695b57fb4b41dcafc78739930'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def binding(path):
 path=Path(path).resolve();raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==ACTUAL_RECEIPT_SHA,'Exact actual root typeprobe1992 receipt required');value=read_unique(path);require(sha(PLAN)==PLAN_SHA==value['source_plan_sha256'],'Exact corrected explicit epoch type diagnosis source required');source=read_unique(PLAN)
 require(value['source_binding']==source['files'],'Complete actual type diagnosis source roster differs')
 for name,digest in source['files'].items():require(sha(ROOT/name)==digest,'Current type diagnosis source changed '+name)
 require(value['schema']==1 and value['passed']is True and value['typed_canonical_JSON_equal']is True and value['native_python_equal']is False and value['actual_binding_canonical_sha256']==value['saved_binding_canonical_sha256'],'Actual representation-only diagnosis not proven')
 require(value['actual_GPU_touch']is False and value['actual_model_inference']is False and value['original_parent_failure_rewritten']is False,'Original failure/inference scope changed');require(value['native_type_differences'] and all((r.get('actual_type'),r.get('saved_type'))==('tuple','list') or r.get('actual_key_type')=='int' and r.get('saved_key_type')=='str' for r in value['native_type_differences']),'Only actual tuple/list and integer JSON-key representation differences admitted')
 original=Path(value['plan_path']);require(sha(original)==value['plan_sha256'],'Exact originally failed plan changed');require(path.read_bytes()==raw,'Actual type diagnosis receipt changed during admission')
 return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'source_plan_sha256':PLAN_SHA,'original_failed_plan_sha256':value['plan_sha256'],'typed_canonical_JSON_representation_only_proven':True,'actual_GPU_or_model_proof_transferred':False}
