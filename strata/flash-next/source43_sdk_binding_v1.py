"""Fresh source43 SDK admission; exact full70 receipt/final source roster."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T120045Z-kfiohqmn')
RECEIPT_SHA='6f6517ac5f6b77e7fd8f0487d62869ef108a490bf6409ebc254bf5e1cfb93bc4'
PLAN=HERE/'native43-engine-build-proposal-v1.json'
PLAN_SHA='17ff9e21276cbab2e802435ec3fcda6f6a1bb57444d53f4fb31156a0615b1b5c'
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
BINARY_NAMES={'strata','native_expert_parity','conversation_snapshot_test','iq_multi_parity','native_grouped_parity','native_multi_parity','shared_expert_parity','verify_parity'}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb')as source:
  for chunk in iter(lambda:source.read(1<<20),b''):digest.update(chunk)
 return digest.hexdigest()
def metadata(receipt,plan,root):
 root=Path(root);require(receipt['image']==IMAGE and receipt['devices_exposed']is False and receipt['build_rc']==0 and type(receipt['build_rc'])is int and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['status'].startswith('built;'),'Fresh complete isolated43 build required');require(receipt['source_revision']=='fb58e0dbc8399662c0e47c76578c6e878b14f6cf'and receipt['ggml_revision']=='3cf03257f219afbe7334045ff7c6a06ac68c627d'and receipt['plan_sha256']==PLAN_SHA,'Exact43 source/ggml/plan required');require(Path(receipt['source_copy'])==root/'source'and Path(receipt['build'])==root/'build'and Path(receipt['plan_snapshot'])==root/'plan.snapshot.json','Exact current SDK root/layout required')
 expected=plan['expected_patched_source_sha256'];recorded=receipt['patched_source_sha256'];require(len(expected)==70 and len(recorded)==70 and set(recorded)<=set(expected)and all(expected[n]==w for n,w in recorded.items()),'Exact actual70/final70 source rows required');require(len(plan['patches'])==43 and len(plan['added_header_payloads'])==32 and len(plan['build_targets'])==8 and len(plan['runtime_python_sources'])==6 and receipt['patches']==plan['patches'],'Exact43 patch/header/target/runtime roster required');require(plan['patches'][-1]['path']=='strata/flash-next/patches/0043-opt-in-owned-pti-native-terminal-hook.patch','Exact new43 final hook patch required')
 binaries=receipt['binary_sha256'];require(len(binaries)==8 and {Path(n).name for n in binaries}==BINARY_NAMES and all(Path(n).parent==root/'build'for n in binaries),'All8 actual source43 binary owners required');return {'actual_receipt_source_rows':70,'independently_required_source_rows':70,'required_earlier_added_headers':32,'all8_fresh_ABI_required':True,'old_source35_41_oracle_or_runtime_transferred':False}
def binding(root=SDK):
 root=Path(root).resolve();require(root==SDK,'Only actual fresh43V3 SDK admitted');require(sha(root/'receipt.json')==RECEIPT_SHA and sha(PLAN)==PLAN_SHA and sha(root/'plan.snapshot.json')==PLAN_SHA,'Exact actual43 receipt/plan bytes required');r=json.loads((root/'receipt.json').read_bytes());plan=json.loads(PLAN.read_bytes());require(json.loads((root/'plan.snapshot.json').read_bytes())==plan,'Actual43 snapshot differs');scope=metadata(r,plan,root)
 source={}
 for n,w in plan['expected_patched_source_sha256'].items():require(sha(root/'source'/n)==w,'Current independently required43 source changed '+n);source[n]=w
 for n,w in r['binary_sha256'].items():require(sha(n)==w,'Current fresh43 binary changed '+n)
 require(sha(root/'build/CMakeCache.txt')==r['cmake_cache_sha256'],'Fresh43 compile cache changed')
 headers={n:source[n]for n in('include/strata/core/verify.hpp','sycl/include/strata/core/verify.hpp','include/strata/core/layer.hpp','include/strata/kernels/native_qsa_indexer.hpp')};require(len(headers)==4,'Actual43 constructor/scratch public headers required')
 return {**scope,'engine_root':str(root),'engine_receipt':str(root/'receipt.json'),'engine_receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'current_all70_source_sha256':source,'current_all8_binary_sha256':r['binary_sha256'],'actual43_public_header_sha256':headers,'ordinary_weight_payload_readback_qualified':False,'materialized_model_or_graph_qualified':False,'full_model_math_qualified':False}

def qualification_source_binding():
 source_plan=HERE/'source43-upload-source-plan-v1.json';row=json.loads(source_plan.read_bytes());root=HERE.parents[1]
 require('evals/configs/models.yaml'not in row['files'],'Upload-only source closure must not invent registry authority')
 for name,wanted in row['files'].items():require(sha(root/name)==wanted,'Frozen upload43 namespace dependency changed '+name)
 return {'source_plan_sha256':sha(source_plan),'files':row['files'],'source43_scope_only':True,'old_SDK_or_oracle_transferred':False}
