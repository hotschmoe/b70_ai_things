#!/usr/bin/env python3
"""CPU-only corrected0023 plan/source/header provenance checks."""
import copy
import hashlib
import json
from pathlib import Path
from validate_layer0_producer_plan_v2 import validate


def main():
    here=Path(__file__).resolve().parent;old=here/'layer0-producer-engine-build-plan-draft-v1.json';new=here/'layer0-producer-engine-build-plan-draft-v2.json'
    v1=json.loads(old.read_bytes());v2=json.loads(new.read_bytes());positive=validate(v2);negative=[]
    try:validate(v1)
    except ValueError as error:
        assert 'Consumed header/final source contradiction' in str(error);negative.append('frozen_v1_consumed_header_mismatch')
    else:raise AssertionError('Frozen contradictory v1 admitted')
    for name,mutate in [('bad_final_header',lambda x:x['added_header_payloads'][-1].__setitem__('sha256','0'*64)),
                        ('duplicate_header',lambda x:x['added_header_payloads'].append(copy.deepcopy(x['added_header_payloads'][0]))),
                        ('false_raw_33',lambda x:x['layer0_capture_contract'].__setitem__('source_value_fields',33)),
                        ('hidden_marked_raw',lambda x:x['layer0_capture_contract'].__setitem__('raw_fused_hidden_observed',True)),
                        ('false_math_qualified',lambda x:x['layer0_capture_contract'].__setitem__('full_model_math_qualified',True))]:
        wrong=copy.deepcopy(v2);mutate(wrong)
        try:validate(wrong)
        except ValueError:negative.append(name)
        else:raise AssertionError('Invalid plan admitted '+name)
    source_path=here/'layer0-producer-source-plan-draft-v2.json';source=json.loads(source_path.read_bytes());assert source['engine_plan_sha256']==hashlib.sha256(new.read_bytes()).hexdigest()
    assert source['patch_sha256']==hashlib.sha256(Path(source['patch']).read_bytes()).hexdigest()
    for path,digest in source['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    result={'mode':'CPU_0023_MANIFEST_CORRECTION_ONLY','passed':True,'positive':positive,'negative_controls':negative,
            'frozen_v1_engine_plan_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'corrected_v2_engine_plan_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
            'corrected_v2_source_plan_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'patch_unchanged':True,
            'gpu_executed':False,'source_sdk_compiled':False,'full_model_math_qualified':False}
    (here/'layer0-producer-plan-correction-cpu-receipt-v2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
