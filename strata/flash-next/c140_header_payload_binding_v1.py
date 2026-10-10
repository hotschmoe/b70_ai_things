"""Separate original added-header payload provenance from final patched headers."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE37_PLAN=HERE/'hosttrace36-batchstamp37-engine-build-plan-v1.json'
SOURCE37_PLAN_SHA='2e940d51c61abe5366526b926f89dcf9f42bf778feee5d13c9b72aefc2785ace'
SOURCE37_ROOT=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T084541Z-zqrph_y1/source')
NEW_PAYLOADS={'include/strata/core/full_cache_observer38.hpp':HERE/'full_cache_observer38.hpp','include/strata/core/full_cache_memory39.hpp':HERE/'full_cache_memory39.hpp','sycl/include/strata/core/layer3_qsa_target_observer.hpp':HERE/'layer3_qsa_target_observer_v1.hpp'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,message):
 if not value:raise ValueError(message)

def binding(plan,source):
 source=Path(source);require(sha(SOURCE37_PLAN)==SOURCE37_PLAN_SHA,'Original source37 header-input plan changed');old=json.loads(SOURCE37_PLAN.read_bytes());orig={h['path']:h for h in old['added_header_payloads']};require(len(orig)==28 and len(plan['added_header_payloads'])==31,'Exact original28 plus new3 header payloads required');rows=[]
 require([h['path'] for h in plan['added_header_payloads'][:28]]==[h['path'] for h in old['added_header_payloads']],'Original ordered28 payload roster changed')
 for header in plan['added_header_payloads']:
  name=header['path'];expected=plan['expected_patched_source_sha256'].get(name);require(type(expected)is str and len(expected)==64 and sha(source/name)==expected,'Actual final consumed header must match exact final67 source ledger '+name)
  if name in orig:
   require(header==orig[name],'Original payload declaration/order/content changed '+name);payload=SOURCE37_ROOT/name
  else:
   require(name in NEW_PAYLOADS,'Unknown newly added payload '+name);payload=NEW_PAYLOADS[name]
  require(sha(payload)==header['sha256'],'Original independently retained header payload changed '+name)
  rows.append({'path':name,'original_payload_path':str(payload),'original_payload_sha256':header['sha256'],'final_consumed_sha256':expected,'modified_by_later_ordered_patches':expected!=header['sha256']})
 require({r['path'] for r in rows if r['modified_by_later_ordered_patches']}=={'sycl/include/strata/core/batch_fidelity_contract.hpp','sycl/include/strata/core/batch_fidelity_observer.hpp'},'Exact declared source38-modified input header set required')
 return {'original_source37_plan_sha256':SOURCE37_PLAN_SHA,'original_payloads':rows,'older_source37_runtime_proof_transferred':False}
