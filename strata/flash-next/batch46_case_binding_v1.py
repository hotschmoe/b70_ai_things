"""Exact authenticated source37 case -> runtime corpus/configuration join."""
from pathlib import Path
from batch_numerical_proofs_v46 import require,read,sha
from serial37_canonical_json_v3 import canonical
from batch46_plan_snapshot_v1 import Snapshot
HERE=Path(__file__).resolve().parent
SPECS={4:('batch-numerical-case4-source37-v7.json','6f0026cdce1e6da98a7219f2f0d71202dec9ae6d3e8d77e6d05926f0e3322670'),6:('batch-numerical-case6-source37-v7.json','3edd28c5d68be5eb7044991596c273b8dc0ff8a9173c552e8bd8ff0d912988a0')}
FIELDS=('tokens','messages','api_token_ids','cancel_index','actual_counter_policy','native_diagnostic_max_new_by_request','api_max_new_by_request','port','slots','paired_two_control')

def authenticated(path,slots,digest=None):
 require(type(slots)is int and slots in SPECS,'Exact declared requested4/bounded6 case required');name,want=SPECS[slots];path=Path(path).resolve()
 require(path==(HERE/name).resolve(),'Exact frozen case path required');snapshot=Snapshot(path);require(snapshot.sha256==want==sha(path) and (digest is None or digest==want),'Current frozen case SHA differs');spec=snapshot.plan
 require(spec['slots']==slots and spec['harness_generation']==46 and spec['source_lane']=='source37','Exact source37 H46 case required');snapshot.verify(spec);return spec

def binding(plan,budget_contract):
 spec=authenticated(plan['spec_path'],plan['slots'],plan['spec_sha256'])
 for key in FIELDS:require(canonical(plan[key])==canonical(spec[key]),'Runtime plan differs authenticated case '+key)
 require(canonical(plan['spec_tokenizer_sha256'])==canonical(spec['tokenizer_sha256']),'Authenticated case tokenizer identity differs')
 budget=budget_contract(plan['kind'],spec['slots'],spec['native_diagnostic_max_new_by_request'],spec['api_max_new_by_request'],plan.get('serial_group_job_count'))
 require(canonical(plan['output_budget_contract'])==canonical(budget) and canonical(plan['max_new_by_request'])==canonical(budget['actual_submission_budget']) and type(plan['max_new'])is int and plan['max_new']==32,'Authenticated case actual transport budget differs')
 return spec
