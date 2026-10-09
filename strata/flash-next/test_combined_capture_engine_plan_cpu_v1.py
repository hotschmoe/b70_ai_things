#!/usr/bin/env python3
"""CPU final manifest ledger consistency controls; no SDK/GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
from validate_layer0_producer_plan_v2 import validate


def check(plan,receipt):
    validate(plan)
    if plan['expected_patched_source_sha256']!=receipt['expected_final_source_sha256']:raise ValueError('Final source ledger differs from fresh reconstruction')
    if plan['source_file_sha256']!=receipt['pristine_source_sha256']:raise ValueError('Pristine ledger differs')
    if set(plan['overlay_files'])!=set(plan['expected_patched_source_sha256']):raise ValueError('Final source coverage differs')
    headers={e['path']:e['sha256'] for e in plan['added_header_payloads']}
    for name in ['sycl/include/strata/core/fidelity_observer.hpp','sycl/include/strata/core/fidelity_observer_contract.hpp','sycl/include/strata/core/layer0_numerical_observer.hpp','sycl/include/strata/core/batch_fidelity_contract.hpp','sycl/include/strata/core/batch_fidelity_observer.hpp','sycl/include/strata/kernels/shared_expert.hpp','sycl/include/strata/kernels/iq_kernels.hpp']:
        if headers.get(name)!=plan['expected_patched_source_sha256'].get(name):raise ValueError('Consumed header absent/mismatched')
    helper='serve/batch_request_identity.py'
    if helper not in plan['runtime_python_sources'] or plan['runtime_python_sources'][helper]!=plan['expected_patched_source_sha256'][helper]:raise ValueError('API identity helper unbound')
    needed={'strata','native_expert_parity','iq_multi_parity','native_grouped_parity','shared_expert_parity','native_multi_parity','verify_parity','conversation_snapshot_test'}
    if not needed<=set(plan['build_targets']):raise ValueError('Full ABI executable targets missing')
    if plan.get('full_model_math_qualified') is not False or plan.get('concurrency_qualified') is not False:raise ValueError('Unrun numerical/concurrency claim')
    return True


def main():
    here=Path(__file__).resolve().parent;p=here/'combined-numerical-batch-engine-build-plan-v1.json';r=here/'combined-numerical-batch-source-reconstruction-v1.json';plan=json.loads(p.read_bytes());receipt=json.loads(r.read_bytes());assert check(plan,receipt)
    negatives=[]
    for name,mutate in [('missing_final_path',lambda x:x['expected_patched_source_sha256'].pop('include/strata/prefill/prefill.hpp')),
                        ('old21_observer_payload',lambda x:next(e for e in x['added_header_payloads'] if e['path']=='sycl/include/strata/core/layer0_numerical_observer.hpp').__setitem__('sha256','55db0accf63982572a34d10655bb81ffc5d783fe77cf6b76da855776cf64996e')),
                        ('missing_api_helper',lambda x:x['runtime_python_sources'].pop('serve/batch_request_identity.py')),
                        ('missing_abi_caller',lambda x:x['build_targets'].remove('native_grouped_parity')),
                        ('false_concurrency_claim',lambda x:x.__setitem__('concurrency_qualified',True)),
                        ('false_model_math_claim',lambda x:x.__setitem__('full_model_math_qualified',True))]:
        bad=copy.deepcopy(plan);mutate(bad)
        try:check(bad,receipt)
        except ValueError:negatives.append(name)
        else:raise AssertionError('Bad combined plan admitted '+name)
    result={'mode':'CPU_COMBINED_MANIFEST_ONLY','passed':True,'plan_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reconstruction_receipt_sha256':hashlib.sha256(r.read_bytes()).hexdigest(),'final_files':len(plan['overlay_files']),'pristine_files':len(plan['source_file_sha256']),'consumed_header_payloads':len(plan['added_header_payloads']),'negative_controls':negatives,'sdk_compiled':False,'gpu_executed':False,'full_model_math_qualified':False,'concurrency_qualified':False}
    (here/'combined-numerical-batch-plan-cpu-receipt-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
