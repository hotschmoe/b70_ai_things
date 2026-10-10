"""NEW shortwarm case from authentic fixture only; existing V2 rejects this case."""
import argparse,copy
from pathlib import Path
import produce_api_shortwarm_tokenizer_fixture_v1 as fixture
def case_from_fixture(root,port=None):
 rows,binding=fixture.finalized_binding(root);seed=fixture.read(fixture.SEED);case=copy.deepcopy(fixture.read(fixture.HERE/'batch-numerical-case2-source35-v2.json'))
 case.update(schema=3,harness_generation=9,api_cache_positive_generation=2,shortwarm_case_generation=1,source_seed_sha256=fixture.sha(fixture.SEED),authentic_fixture_binding=binding,API_warm_request_policy=seed['warm_request_policy'],API_target_request_policy=seed['target_request_policy'])
 case['messages']={p:[r['messages'] for r in rows['fixtures'][p]] for p in ('warm','target')};case['api_token_ids']={p:[r['ids'] for r in rows['fixtures'][p]] for p in ('warm','target')};case['tokens']['target']=case['api_token_ids']['target'];case['api_max_new_by_request']=seed['target_max_new_by_request'];case['max_new_by_request']=seed['target_max_new_by_request'];case['api_warm_max_new_by_request']=seed['warm_max_new_by_request'];case['actual_counter_policy']=[{k:v for k,v in r.items() if k!='prerequisite'} for r in seed['declared_counter_policy']]
 case.update(actual_cached_state_handoff_qualified=False,warm_two_row_runtime_qualified=False,matched_buffer_only_A_B_input_equivalent=False,full_model_math_qualified=False,genuine_runtime_preparable=False,new_shortwarm_runtime_successor_required=True,native_warm_tokens_scope='Unused inherited native transport corpus; API warm actual IDs above require new shortwarm runtime admission')
 if port is not None:fixture.require(type(port)is int and 1024<=port<=65535,'Bounded exact integer port required');case['port']=port
 return case
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int);a=p.parse_args();case=case_from_fixture(a.fixture_root,a.port);fixture.require(not a.output.exists(),'New case output required');a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='ascii') as f:
  import json
  f.write(json.dumps(case,indent=2,ensure_ascii=True,allow_nan=False)+'\n')
if __name__=='__main__':main()
