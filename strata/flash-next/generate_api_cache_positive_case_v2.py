"""Generate NEW positive2 case solely from authentic CPU tokenizer fixtures."""
import argparse,copy
from pathlib import Path
import api_cache_positive_contract_v2 as contract
from batch_numerical_proofs_v7 import read,write,require,sha
HERE=Path(__file__).resolve().parent

def case_from_fixture(port=None):
 fixture,binding=contract.fixture_binding();case=copy.deepcopy(read(HERE/'batch-numerical-case2-source35-v2.json'));case.update(schema=3,harness_generation=9,api_cache_positive_generation=2,source_seed_sha256=fixture['source_seed_sha256'],authentic_fixture_binding=binding,API_warm_request_policy={'strata_fresh':True},API_target_request_policy={'strata_fresh':False});case['messages']={k:[r['messages'] for r in fixture['fixtures'][k]] for k in ('warm','target')};case['api_token_ids']={k:[r['ids'] for r in fixture['fixtures'][k]] for k in ('warm','target')};case['tokens']['target']=case['api_token_ids']['target'];case['native_warm_tokens_scope']='Inherited immutable native transport corpus is unused in this API-only arm; actual APIwarm IDs above are authentic';case['api_max_new_by_request']=[64,64];case['max_new_by_request']=[64,64];case['actual_counter_policy']=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(2)]+[{'logical_index':1,'role':'solo_migration','values':{'reused':'prompt_minus_one','read_from':'prompt_minus_one','reread_to':-1}}];case.update(actual_cached_state_handoff_qualified=False,initial_zero_counters_runtime_readback_required=True)
 if port is not None:require(type(port) is int and 1024<=port<=65535,'Actual TCP port bounded');case['port']=port
 return case

def generate(output,port=None):
 case=case_from_fixture(port);require(not output.exists(),'NEW case output cannotreplace existing evidence');output.parent.mkdir(parents=True,exist_ok=True);write(output,case);return case

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--port',type=int);a=p.parse_args();generate(a.output,a.port)
if __name__=='__main__':main()
