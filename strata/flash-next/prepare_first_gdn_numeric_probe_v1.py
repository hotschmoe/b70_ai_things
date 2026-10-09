#!/usr/bin/env python3
"""CPU-only prefix/capture-contract preparation; never runs model or GPU."""
import argparse
import hashlib
import json
from pathlib import Path

PREFIXES=(1,2,4,8)
REQUIRED_BINDINGS=('model_lock_sha256','inventory_sha256','tokenizer_metadata_sha256',
                   'chat_template_sha256','engine_receipt_sha256','capture_binary_sha256',
                   'runtime_image','source_identity_before_sha256','source_identity_after_sha256')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(token_ids,reference_plan):
    if len(token_ids)!=8 or any(type(i) is not int or not 0<=i<248320 for i in token_ids):raise ValueError('Eight exact model GEN IDs required')
    return {'schema':1,'mode':'FIRST_GDN_NUMERICAL_PROBE_PREPARATION_ONLY','reference_plan':str(Path(reference_plan).resolve()),
        'reference_plan_sha256':sha(reference_plan),'cases':[{'prefix_tokens':n,'gen_ids':token_ids[:n],
            'positions':list(range(n)),'first_prediction_input_position':n-1,'last_input_token':token_ids[n-1],
            'computed_reference_layers':[0],'full_native_forward_layers':48,'initial_state':'empty_owned'} for n in PREFIXES],
        'actual_execution_allowed':False,'gpu_executed':False,'full_model_math_qualified':False}


def validate_capture(case,record,expected_bindings):
    """Metadata admission only, never an arithmetic pass from identity alone."""
    for key in REQUIRED_BINDINGS:
        if not expected_bindings.get(key) or record.get(key)!=expected_bindings[key]:raise ValueError('Capture identity differs: '+key)
    if record.get('gen_ids')!=case['gen_ids'] or record.get('positions')!=case['positions'] or record.get('last_input_token')!=case['last_input_token']:raise ValueError('Capture prefix token/position differs')
    if record.get('initial_state')!='empty_owned' or record.get('observed_layer')!=0 or record.get('input_position')!=case['first_prediction_input_position']:raise ValueError('Capture state/layer boundary differs')
    if record.get('residual_shape')!=[4,2560] or record.get('residual_phase')!='after_layer0_ffn_hc_write':raise ValueError('Capture residual scope differs')
    if record.get('recurrent_shape')!=[128,48,128] or record.get('conv_shape')!=[10240,3]:raise ValueError('Capture physical GDN state layout differs')
    if record.get('source_sentinel_before_original') is not True or record.get('source_sentinel_after_original') is not True:raise ValueError('Source sentinel changed or absent')
    if record.get('native_q8_packets_independently_verified') is not True:raise ValueError('Independent production Q8_1 packet prereq missing')
    if record.get('transcendental_or_reduction_coverage')!='declared_and_independently_validated':raise ValueError('Native operation contract is unqualified')
    return {'capture_metadata_admitted':True,'operator_math_pass':False,'full_model_math_qualified':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--gen-ids',type=Path,required=True);p.add_argument('--reference-plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    ids=json.loads(args.gen_ids.read_bytes());result=prepare(ids,args.reference_plan)
    if args.output.exists():raise ValueError('Preserve existing probe plan')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'prepared':True,'actual_execution_allowed':False,'output':str(args.output)}))


if __name__=='__main__':main()
