#!/usr/bin/env python3
"""Exact immutable integrated source plan reconstruction; no compiler/devices/weights."""
import ast,hashlib,json,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'integrated-slot-owner-trace-engine-build-plan-v1.json'
SOURCE=Path('/mnt/vm_8tb/github/strata')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def output(args):return subprocess.check_output(args,text=True).strip()
plan=json.loads(PLAN.read_text())
assert plan['source_revision']=='fb58e0dbc8399662c0e47c76578c6e878b14f6cf'
assert plan['image']=='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
assert plan['ggml']['revision']=='3cf03257f219afbe7334045ff7c6a06ac68c627d'
assert output(['git','-C',str(SOURCE),'rev-parse','HEAD'])==plan['source_revision']
assert not output(['git','-C',str(SOURCE),'status','--porcelain'])
assert len(plan['build_targets'])==8 and set(plan['build_targets'])=={'strata','native_expert_parity','conversation_snapshot_test','iq_multi_parity','native_grouped_parity','native_multi_parity','shared_expert_parity','verify_parity'}
for item in plan['integrated_source_inputs'].values():assert sha(ROOT/item['path'])==item['sha256']
with tempfile.TemporaryDirectory(prefix='integrated-prefix-source-cpu-') as name:
 source=Path(name)/'source'
 subprocess.run(['git','clone','--local','--no-hardlinks','--no-checkout',str(SOURCE),str(source)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 subprocess.run(['git','-C',str(source),'checkout','--detach',plan['source_revision']],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 for rel,h in plan['source_file_sha256'].items():assert sha(source/rel)==h,rel
 for item in plan['patches']:
  patch=ROOT/item['path'];assert sha(patch)==item['sha256']
  subprocess.run(['git','-C',str(source),'apply','--check',str(patch)],check=True)
  subprocess.run(['git','-C',str(source),'apply',str(patch)],check=True)
 assert set(plan['overlay_files'])==set(plan['expected_patched_source_sha256'])
 for rel,h in plan['expected_patched_source_sha256'].items():assert sha(source/rel)==h,rel
 for item in plan['added_header_payloads']:assert sha(source/item['path'])==item['sha256'],item['path']
 tree=ast.parse((source/'serve/artifact_identity.py').read_text());names=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PYTHON_SOURCES' for target in node.targets))
 assert len(names)==6 and set(names)==set(plan['runtime_python_sources'])
 for rel,h in plan['runtime_python_sources'].items():assert sha(source/rel)==h,rel
 api=(source/'serve/server.py').read_text();assert 'from serve.batch_request_identity import' in api and api.index('from serve.batch_request_identity import')>api.index('ROOT =')
 generate=(source/'sycl/src/program/generate.cpp').read_text()
 for anchor in ['strict native batch requires exact2/4/6','fit!=requested_batch','batch checkpoint host byte capacity','batch donor selection identity is stale','public batch prefix requires resident fixed FP16','cache_prefix_ceiling>0?conversations.best','before_later_rows=1']:
  assert anchor in generate,anchor
 verify=(source/'sycl/src/core/verify.cpp').read_text();assert 'actual rows/positions/tokens disagree' in verify
 arena=(source/'sycl/include/strata/core/slot_session_arena.hpp').read_text();assert 'STRATA_SLOT_OWNER_TRACE' in arena and 'owner_register' in arena and 'trace_side_owners' in arena
 assert 'SlotSessionArena>(dev,int(k),b)' in generate and 'owner->trace_graphs_bound()' in generate and generate.count('owner->trace_graphs_retired()')==2
 assert plan['concurrency_qualified'] is False and plan['full_model_math_qualified'] is False and plan['actual_public_cache_qualified'] is False
 assert all(item['default'].startswith('OFF') for key,item in plan['new_source_contracts'].items() if key in ('0025','0026','0027'))
receipt={'CONFIG':'pristinefb58 + integrated60/29 source patch/header/ABI recipe only','COMMAND':'python3 strata/flash-next/test_slot_owner_engine_plan_cpu_v1.py','RESULT':{'all29_patches':True,'final60_source_hashes':True,'header_payload24_closure':True,'all8_SDK_targets_recipe':True},'VERDICT':'PASS CPU source reconstruction only; SDK/GPU/slots/source/model/concurrency unqualified'}
(HERE/'slot-owner-engine-plan-cpu-receipt-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS CPU pristinefb58 +all29 reviewedpatches checkedorder;60finalsource files/24headerpayloads/sixruntimePython closure/eightfreshABI target recipe;guards intact. No compiler/SDK/GPU/model payload or runtime qualification.')
