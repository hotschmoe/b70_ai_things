#!/usr/bin/env python3
"""CPU-only fresh git-show reconstruction; immutable combined21/23/22v2/24v2 plan."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base_path=HERE/'layer0-producer-engine-build-plan-draft-v2.json';base=json.loads(base_path.read_bytes())
    identity_path=HERE/'batch-request-identity-draft-plan-v2.json';identity=json.loads(identity_path.read_bytes())
    batch_path=HERE/'batch-fidelity-observer-draft-plan-v2.json';batch=json.loads(batch_path.read_bytes())
    final_path=HERE/'combined-numerical-batch-engine-build-plan-v1.json';receipt_path=HERE/'combined-numerical-batch-source-reconstruction-v1.json'
    if final_path.exists() or receipt_path.exists():raise ValueError('Preserve immutable combined generation')
    patches=copy.deepcopy(base['patches'])
    for entry in [{'path':identity['patch'],'sha256':identity['patch_sha256']},batch['patches'][-1]]:
        if any(p['path']==entry['path'] for p in patches):raise ValueError('Duplicate patch')
        patches.append(entry)
    touched=set(base['source_file_sha256'])|set(base['overlay_files'])
    for entry in patches:
        p=REPO/entry['path'];assert sha(p)==entry['sha256']
        for line in p.read_text().splitlines():
            if line.startswith(('+++ b/','--- a/')):touched.add(line[6:])
    pristine={};absent=[];command_rows=[]
    with tempfile.TemporaryDirectory(prefix='combined-source-cpu-') as temporary:
        source=Path(temporary)
        for name in sorted(touched):
            r=subprocess.run(['git','-C',base['source_root'],'show',base['source_revision']+':'+name],capture_output=True)
            if r.returncode:
                # A path introduced by a reviewed patch must be absent at the pin.
                exists=subprocess.run(['git','-C',base['source_root'],'cat-file','-e',base['source_revision']+':'+name],capture_output=True)
                if exists.returncode!=128:raise RuntimeError('Unresolved pristine lookup '+name)
                absent.append(name);continue
            digest=hashlib.sha256(r.stdout).hexdigest();pristine[name]=digest
            if name in base['source_file_sha256']:assert digest==base['source_file_sha256'][name]
            dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.stdout)
        # Entire chain starts at pristinefb58, never the active/patched worktree.
        for entry in patches:
            p=REPO/entry['path']
            for mode in ['--check',None]:
                argv=['git','apply']+([mode] if mode else [])+[str(p)]
                r=subprocess.run(argv,cwd=source,capture_output=True,text=True)
                if r.returncode:raise RuntimeError('Patch failed '+entry['path']+' '+r.stderr)
                command_rows.append({'patch':entry['path'],'check_only':bool(mode),'rc':r.returncode})
        final={name:sha(source/name) for name in sorted(touched) if (source/name).is_file()}
        assert set(base['expected_patched_source_sha256'])<=set(final)
        # Changed final paths must be explained by the two independently reviewed increments.
        final_changed={name for name,digest in final.items() if name in base['expected_patched_source_sha256'] and digest!=base['expected_patched_source_sha256'][name]}
        allowed={'sycl/include/strata/core/verify.hpp','sycl/src/core/verify.cpp','sycl/src/prefill/prefill.cpp','sycl/src/program/generate.cpp','serve/server.py','serve/artifact_identity.py'}
        assert final_changed<=allowed
        for name,digest in batch['source_sha256'].items():
            if name in {'sycl/src/core/verify.cpp'}:continue #0023 changes host producer calls in that same file.
            assert final[name]==digest,(name,final[name],digest)
        for name in ['serve/server.py','serve/artifact_identity.py','serve/batch_request_identity.py']:assert final[name]==identity['overlays'][name]
        for name in ['serve/server.py','serve/artifact_identity.py','serve/batch_request_identity.py']:ast.parse((source/name).read_bytes(),filename=name)
        assert 'batch_request_identity.py' in (source/'serve/artifact_identity.py').read_text()
        header_rows={row['path']:copy.deepcopy(row) for row in base['added_header_payloads']}
        for name in sorted(final):
            if name.endswith('.hpp') and name in absent:
                header_rows[name]={'path':name,'sha256':final[name],'consumed_include':name.split('/include/',1)[-1],'resolution':'SYCL shadow first' if name.startswith('sycl/include/') else 'shared include; no reviewed SYCL shadow'}
        for name,row in header_rows.items():row['sha256']=final[name]
        for name in ['sycl/include/strata/core/batch_fidelity_contract.hpp','sycl/include/strata/core/batch_fidelity_observer.hpp']:
            assert name in header_rows and header_rows[name]['sha256']==batch['source_sha256'][name]
        plan=copy.deepcopy(base);plan.update(status='CPU full reviewed patch-chain reconstruction PASS; no SDK compile/link/GPU/concurrency mathematical qualification',generation='combined20+21+23v2+22v2+24v2-full-SYCL-rebuild',patches=patches,source_file_sha256=pristine,overlay_files=sorted(final),expected_patched_source_sha256=final,added_header_payloads=list(header_rows.values()))
        plan['derived_from_plan']={'path':str(base_path.relative_to(REPO)),'sha256':sha(base_path),'additional_plans':[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in [identity_path,batch_path]]}
        plan['build_targets']=['strata','native_expert_parity','conversation_snapshot_test','iq_multi_parity','native_grouped_parity','native_multi_parity','shared_expert_parity','verify_parity']
        plan['abi_rebuild_contract']={'optional23_observer_args':True,'all_model_engine_prefill_core_cpu_kernel_libraries':'transitive fresh full strata target; no object/library reuse','direct_native_grouped_parity_callers':['native_expert_parity','iq_multi_parity','native_grouped_parity'],'additional_header_and_kernel_callers':['native_multi_parity','shared_expert_parity','verify_parity'],'external_fixture_rule':'standalone old mirror/expert/oracle binaries must rebuild and bind new library SHA; compilation is not GPU qualification'}
        plan['runtime_python_sources']={name:final[name] for name in ['serve/server.py','serve/artifact_identity.py','serve/batch_request_identity.py','tools/strata_tokenizer.py']}
        plan['new_source_contracts']['0022']={'patch_sha256':identity['patch_sha256'],'scope':'request/slot/engine identity, exactcapacity, asyncstop/migration transport; GPU concurrent numerical qualification outstanding'}
        plan['new_source_contracts']['0024']={'patch_sha256':batch['patches'][-1]['sha256'],'scope':'default-off perRID full48residual/head snapshots and completedlogical spans, immutable16graphrosters+GPUrow/layout/input epochs; actualSDK/GPU off/on/math/lifecycle gates outstanding','bounds':batch['bounds']}
        plan['required_before_model_run'] += ['Complete matched runtimePython fingerprint includes0022 batch_request_identity.py helper and all raw node/source headers','New combined binary source390/readiness/C1/health/lifecycle prerequisite, exactINFO2/4/6 capacity before concurrentbatch probes','GPU0024 perRID/full48/rawhead/rowepoch/spans coverage and same-binary diagoff/on equality; logical spans are not GPU durations','33 layer0 fields=31source values plus2DERIVED; rawfusedhiddenUNOBSERVED and complete model math/concurrency remain false']
        plan['concurrency_qualified']=False;plan['full_model_math_qualified']=False
        final_path.write_text(json.dumps(plan,indent=2)+'\n')
        receipt={'schema':1,'mode':'CPU_PRISTINE_FULL_PATCH_CHAIN_RECONSTRUCTION_ONLY','passed':True,'base_revision':base['source_revision'],'image':base['image'],'ggml_revision':base['ggml']['revision'],'base_plan_sha256':sha(base_path),'plan':str(final_path.relative_to(REPO)),'plan_sha256':sha(final_path),'pristine_files':len(pristine),'added_paths':len(absent),'final_files':len(final),'pristine_source_sha256':pristine,'absent_at_pristine_pin':absent,'expected_final_source_sha256':final,'final_header_payloads':list(header_rows.values()),'commands':command_rows,'python_ast_passed':True,'python_helper_fingerprint_present':True,'full_abi_targets':plan['build_targets'],'source_overlay_private_and_removed':True,'sdk_compiled':False,'gpu_executed':False,'concurrency_qualified':False,'full_model_math_qualified':False}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'plan_sha256':sha(final_path),'receipt_sha256':sha(receipt_path),'final_files':len(final),'pristine_files':len(pristine),'added_headers':len(header_rows),'SDK_or_GPU_executed':False},sort_keys=True))


if __name__=='__main__':main()
