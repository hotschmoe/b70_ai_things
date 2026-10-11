"""Fresh source41 SDK admission; independently cover69 despite receipt67."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T052714Z-k9s4qsio')
RECEIPT_SHA='880d5f6074cb4db248e5c2d6b9654b7952c5c7da31f94090ba523375cddca849'
PLAN=HERE/'indexer-materialized41-engine-build-proposal-v3.json'
PLAN_SHA='2f40973fb4c0519c1b99434a820b07dc44ada19a8c4d7fda8aecc11110c7d01a'
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
 root=Path(root);require(receipt['image']==IMAGE and receipt['devices_exposed']is False and receipt['build_rc']==0 and type(receipt['build_rc'])is int and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['status'].startswith('built;'),'Fresh complete isolated41 build required');require(receipt['source_revision']=='fb58e0dbc8399662c0e47c76578c6e878b14f6cf'and receipt['ggml_revision']=='3cf03257f219afbe7334045ff7c6a06ac68c627d'and receipt['plan_sha256']==PLAN_SHA,'Exact41 source/ggml/plan required');require(Path(receipt['source_copy'])==root/'source'and Path(receipt['build'])==root/'build'and Path(receipt['plan_snapshot'])==root/'plan.snapshot.json','Exact current SDK root/layout required')
 expected=plan['expected_patched_source_sha256'];recorded=receipt['patched_source_sha256'];require(len(expected)==69 and len(recorded)==67 and set(recorded)<=set(expected)and all(expected[n]==w for n,w in recorded.items()),'Actual67 receipt must be exact subset of separately required69 source rows');require(len(plan['patches'])==41 and len(plan['added_header_payloads'])==31 and len(plan['build_targets'])==8 and len(plan['runtime_python_sources'])==6 and receipt['patches']==plan['patches'],'Exact41 patch/header/target/runtime roster required');require(plan['patches'][-1]['path']=='strata/flash-next/patches/0041-qsa-indexer-materialized-half-candidate-v3.patch'and plan['patches'][-1]['sha256']=='3e0c3752cb558e8be77109b16c18f032bd7a6170acfc88719f263daff9725940','HeldV1/V2/oldsource40 patch cannot transfer')
 binaries=receipt['binary_sha256'];require(len(binaries)==8 and {Path(n).name for n in binaries}==BINARY_NAMES and all(Path(n).parent==root/'build'for n in binaries),'All8 actual source41 binary owners required');return {'actual_receipt_source_rows':67,'independently_required_source_rows':69,'required_earlier_added_headers':31,'all8_fresh_ABI_required':True,'old_source40_oracle_or_runtime_transferred':False}
def binding(root=SDK):
 root=Path(root).resolve();require(root==SDK,'Only actual fresh41V3 SDK admitted');require(sha(root/'receipt.json')==RECEIPT_SHA and sha(PLAN)==PLAN_SHA and sha(root/'plan.snapshot.json')==PLAN_SHA,'Exact actual41 receipt/plan bytes required');r=json.loads((root/'receipt.json').read_bytes());plan=json.loads(PLAN.read_bytes());require(json.loads((root/'plan.snapshot.json').read_bytes())==plan,'Actual41 snapshot differs');scope=metadata(r,plan,root)
 source={}
 for n,w in plan['expected_patched_source_sha256'].items():require(sha(root/'source'/n)==w,'Current independently required41 source changed '+n);source[n]=w
 for n,w in r['binary_sha256'].items():require(sha(n)==w,'Current fresh41 binary changed '+n)
 require(sha(root/'build/CMakeCache.txt')==r['cmake_cache_sha256'],'Fresh41 compile cache changed')
 headers={n:source[n]for n in('include/strata/core/verify.hpp','sycl/include/strata/core/verify.hpp','include/strata/core/layer.hpp','include/strata/kernels/native_qsa_indexer.hpp')};require(len(headers)==4,'Actual41 constructor/scratch public headers required')
 return {**scope,'engine_root':str(root),'engine_receipt':str(root/'receipt.json'),'engine_receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'current_all69_source_sha256':source,'current_all8_binary_sha256':r['binary_sha256'],'actual41_public_header_sha256':headers,'ordinary_weight_payload_readback_qualified':False,'materialized_model_or_graph_qualified':False,'full_model_math_qualified':False}

def qualification_source_binding():
 source_plan=HERE/'source41-upload-source-plan-v1.json';row=json.loads(source_plan.read_bytes());root=HERE.parents[1]
 require('evals/configs/models.yaml'not in row['files'],'Upload-only source closure must not invent registry authority')
 for name,wanted in row['files'].items():require(sha(root/name)==wanted,'Frozen upload41 namespace dependency changed '+name)
 return {'source_plan_sha256':sha(source_plan),'files':row['files'],'source41_scope_only':True,'old_SDK_or_oracle_transferred':False}
