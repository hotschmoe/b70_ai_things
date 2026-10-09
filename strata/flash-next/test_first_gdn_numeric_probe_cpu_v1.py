#!/usr/bin/env python3
"""Synthetic capture-admission controls only; no captured/model payloads."""
import copy
import hashlib
import json
from pathlib import Path
from prepare_first_gdn_numeric_probe_v1 import prepare,validate_capture,REQUIRED_BINDINGS


def main():
    here=Path(__file__).resolve().parent;plan=prepare([10,11,12,13,14,15,16,17],here/'original-first-gdn-layer-reference-plan-v1.json')
    assert [c['prefix_tokens'] for c in plan['cases']]==[1,2,4,8];bindings={k:'synthetic-'+k for k in REQUIRED_BINDINGS};case=plan['cases'][3]
    record={**bindings,'gen_ids':case['gen_ids'],'positions':case['positions'],'last_input_token':case['last_input_token'],
        'initial_state':'empty_owned','observed_layer':0,'input_position':7,'residual_shape':[4,2560],'residual_phase':'after_layer0_ffn_hc_write',
        'recurrent_shape':[128,48,128],'conv_shape':[10240,3],'source_sentinel_before_original':True,'source_sentinel_after_original':True,
        'native_q8_packets_independently_verified':True,'transcendental_or_reduction_coverage':'declared_and_independently_validated'}
    assert validate_capture(case,record,bindings)['operator_math_pass'] is False;negatives=[]
    mutations={'wrong_token':('last_input_token',0),'wrong_position':('input_position',6),'wrong_layer':('observed_layer',1),
        'wrong_residual_phase':('residual_phase','before_ffn_write'),'missing_packet_prereq':('native_q8_packets_independently_verified',False),
        'missing_operation_contract':('transcendental_or_reduction_coverage','unobserved'),'stale_state':('initial_state','reused'),
        'wrong_state_layout':('recurrent_shape',[48,128,128]),'changed_sentinel':('source_sentinel_after_original',False)}
    mutations.update({key:(key,'wrong') for key in REQUIRED_BINDINGS})
    for name,(key,value) in mutations.items():
        bad=copy.deepcopy(record);bad[key]=value
        try:validate_capture(case,bad,bindings)
        except ValueError:negatives.append(name)
        else:raise AssertionError('Invalid capture admitted '+name)
    result={'mode':'CPU_NUMERICAL_PROBE_METADATA_ONLY','passed':True,'prefixes':[1,2,4,8],'negative_controls':negatives,
        'driver_source_sha256':hashlib.sha256((here/'prepare_first_gdn_numeric_probe_v1.py').read_bytes()).hexdigest(),
        'actual_payloads_read':False,'gpu_executed':False,'operator_math_pass':False,'full_model_math_qualified':False}
    (here/'first-gdn-numeric-probe-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
