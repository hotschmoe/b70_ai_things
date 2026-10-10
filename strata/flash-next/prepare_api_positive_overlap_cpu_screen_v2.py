"""Prepare preregistered CPU screening requests from authentic tokenization."""
import argparse
from pathlib import Path
import produce_api_positive_overlap_corpus_v2 as producer

def screen_plan(fixture,binding,seed):
    requests=[]
    for phase in ('warm','target'):
        for row in fixture['fixtures'][phase]:
            requests.append({'phase':phase,'logical_index':row['logical_index'],
                             'messages':row['messages'],'rendered':row['rendered'],
                             'accepted_input_ids':row['ids'],
                             'fresh_process_repeats':2,
                             'request':{'model':'hotschmoe-dd','prompt':row['ids'],
                                        'cache_prompt':False,'return_tokens':True,'stream':False,
                                        'temperature':0,'seed':1234,'n_predict':64,
                                        'repeat_penalty':1,'samplers':['temperature']}})
    return {'schema':'api-positive-overlap-CPU-screen-plan-v2','fixture_binding':binding,
            'source_binding':producer.source_binding(),'requests':requests,
            'candidate_order':seed['candidates'],'minimum_generated_tokens':8,
            'maximum_generated_tokens':64,'natural_EOS_stop_required':True,
            'deterministic_output_ids_and_text_repeat_required':True,
            'all_warm_and_all_six_target_requests_must_be_screened':True,
            'selection_rule':'First declared pair whose two members and both warm requests satisfy the actual continuation predicate; no choice before all screen receipts close',
            'model_identity_requirements':['actual original role/shard roster and current stats',
                'fresh preCPU and postCPU whole-four SHA256',
                'known pages before/after every model phase',
                'actual template bytes and inputIDs equal authentic fixture',
                'original CPU build/supplement/source/library/owned terminal admission',
                'CPU memory/no-swap/start112GiB/live6GiB guards'],
            'CPU_serve_recipe_basis':'NEW purpose-specific screening producer required; frozen qualify_cpu_functional_pilot_v3.py helpers/parameters only, no old functional PASS transfer',
            'actual_CPU_screen_observed':False,'selected_candidate':None,
            'actual_GPU_positive_overlap_observed':False,'actual_full_cache_qualified':False,
            'serial_raw49_for_selected_exact_prefixes_required_later':True,
            'actual_two_target_BGEN_and_client_cancel_required_later':True}

def continuation_predicate(response):
    # Same exact current raw API schema as the actual source-bound V1 runner.
    from qualify_api_positive_overlap_cpu_screen_v2 import continuation_eligible
    tokens=response.get('tokens')
    if type(tokens)is not list or any(type(t)is not int or not 0<=t<248320 for t in tokens):return False
    if not all(k in response for k in ('stop','stop_type','truncated','content')) or type(response['content'])is not str:return False
    return continuation_eligible(response)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();producer.require(not args.output.exists(),'New CPU screening plan path required')
    fixture,binding=producer.finalized_binding(args.fixture)
    plan=screen_plan(fixture,binding,producer.read(producer.SEED))
    producer.write(args.output,plan)
    print('Prepared CPU screening plan only; no model inference or GPU qualification.')
if __name__=='__main__':main()
