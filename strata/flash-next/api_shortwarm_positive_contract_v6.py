"""NEW authentic shortwarm input scope, unchanged positive source35 profile."""
from pathlib import Path
import api_cache_positive_contract_v2 as original
import produce_api_shortwarm_tokenizer_fixture_v1 as producer
from positive_semantic_registry_admission_v4 import registry_gate
from api_cache_positive_contract_v2 import PROFILE_ENV,actual_startup,CHECKPOINT_ARGS,lock_checkpoint_defaults,last_live_handoff,actual_phase_legs,alias
require=original.require
FIXTURE_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/api-shortwarm-tokenizer-fixture-v1')
SOURCE_PLAN=original.HERE/'api-buffered-shortwarm-positive-source-plan-v6.json'
def fixture_binding():
 pin=original.read(SOURCE_PLAN)['actual_shortwarm_fixture'];require(pin['root']==str(FIXTURE_ROOT),'Exact authentic shortwarm root differs')
 for n,w in pin['files'].items():require(original.sha(FIXTURE_ROOT/n)==w,'Actual authentic shortwarm evidence changed '+n)
 return producer.finalized_binding(FIXTURE_ROOT)
def profile(plan):
 require(plan.get('shortwarm_case_generation')==1 and plan.get('shortwarm_fixture_root')==str(FIXTURE_ROOT),'Explicit authentic shortwarm case required')
 require(plan.get('api_warm_max_new_by_request')==[32,32] and plan.get('api_max_new_by_request')==[64,64],'Exact shortwarm32/target64 budgets required')
 # Same source-bound profile/counter/checkpoint policy, NEW authentic fixture.
 source=Path(original.__file__).read_text();begin=source.index('def profile(plan):');end=source.index('\ndef alias(',begin);ns=dict(vars(original));ns['fixture_binding']=fixture_binding;exec(compile(source[begin:end],str(original.__file__)+'[NEW-shortwarm-fixture-V6]','exec'),ns);scope=ns['profile'](plan)
 scope.update(shortwarm_case_generation=1,matched_buffer_only_A_B_input_equivalent=False,warm_two_row_runtime_qualified=False)
 return scope
