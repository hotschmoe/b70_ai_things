"""Pure original upload logical/free subset on explicit immutable evidence bytes.
Unintegrated; source/model/page/SDK/health/ownership outer guards remain live.
"""
import json
from pathlib import Path
from parse_usm_logical_free_trace import parse_trace,negative_controls
from operation_pack_hash_witness_v2 import require

def validate(snapshot,parameters):
 text=snapshot.bytes_for(parameters['log_path']).decode('utf-8');saved=json.loads(snapshot.bytes_for(parameters['logical_path']));checked=parse_trace(text);require(checked['passed'] and all(negative_controls(text,True).values()),'Chronological context-matched logical-free trace failed')
 require(saved.get('passed') and set(saved.get('negative_controls',{}))=={'missing_free','double_free','failed_free'} and all(saved['negative_controls'].values()),'Logical-free negative controls incomplete');require(all(saved.get(k)==v for k,v in checked.items()),'Saved logical-free evidence differs from raw trace');require(saved['counts']['owners']==parameters['expected_owner_count'] and not saved['live'],'Owned allocation/scratch coverage differs or live ledger nonempty')
 return {'checked':checked,'negative_controls':saved['negative_controls'],'original_upload_logical_subset_passed':True,'outer_source_model_health_or_runtime_gates_proven':False}
