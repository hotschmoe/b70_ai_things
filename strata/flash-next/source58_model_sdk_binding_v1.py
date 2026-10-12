"""Supplied actual SDK58 receipt pins; no invented build proof or old ABI transfer."""
import hashlib,json
from pathlib import Path
SDK=None;RECEIPT_SHA=None
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native58-paired-prefill-cooperative-engine-build-proposal-v1.json'
PLAN_SHA='92931ba113dd47241774a79f75669375f62bcd23313ee5617ea68ec91e9f6ab7'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 path=Path(path);require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Canonical actual SDK file required');before=path.stat();h=hashlib.sha256()
 with path.open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'SDK file changed during hash');return h.hexdigest()
def configure(root,receipt_sha):
 global SDK,RECEIPT_SHA
 require(type(receipt_sha)is str and len(receipt_sha)==64 and all(c in '0123456789abcdef'for c in receipt_sha),'Actual receipt SHA required');root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical actual SDK58 root required');require(sha(root/'receipt.json')==receipt_sha,'Original actual receipt bytes changed');receipt=json.loads((root/'receipt.json').read_bytes());require(receipt['plan_sha256']==PLAN_SHA,'Receipt must belong to actual58 proposal');SDK=root;RECEIPT_SHA=receipt_sha
 return {'SDK58_root':str(root),'SDK58_receipt_sha256':receipt_sha,'actual_build_bytes_still_rechecked_before_model':True}
def binding():
 require(SDK is not None and RECEIPT_SHA is not None,'Actual SDK58 build not configured; no placeholder proof allowed');require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Exact actual58 plan/receipt required');plan=json.loads(PLAN.read_bytes());receipt=json.loads((SDK/'receipt.json').read_bytes());require(type(receipt['build_rc'])is int and receipt['build_rc']==0 and receipt['devices_exposed']is False and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['plan_sha256']==PLAN_SHA,'Fresh actual58 build/isolation proof required');require(len(plan['expected_patched_source_sha256'])==75 and len(receipt['binary_sha256'])==8,'Fresh75 source/all8 ABI required')
 require(receipt['build_strategy']=='direct52_to58_fresh_all_source_v1' and receipt['fresh_all_source_compilation'] is True and receipt['reused_objects']==[] and receipt['reused_archives']==[] and receipt['parent_unchanged'] is True,'Actual58 fresh header-consumer compilation required')
 require(receipt['external_source_root']==str(Path(plan['source58_parent_actual_sdk'])/'source') and receipt['external_source_identity_kind']=='immutable_actual52_source_snapshot','Honest immutable input-source binding required')
 require(receipt['parent_root']==plan['source58_parent_actual_sdk'] and receipt['parent_receipt_sha256']==plan['source58_parent_actual_receipt_sha256'],'Direct admitted52 parent required')
 require(sha(SDK/'plan.snapshot.json')==PLAN_SHA and sha(SDK/'build'/'fresh-audit.json')==receipt['fresh_dependency_audit_sha256'],'Actual58 source/dependency audit receipt changed')
 audit=json.loads((SDK/'build'/'fresh-audit.json').read_bytes());require(audit==receipt['fresh_dependency_audit'] and audit['passed'] is True and audit['fresh_all_source_compilation'] is True and audit['reused_objects']==[] and audit['reused_archives']==[],'Actual58 fresh audit proof required')
 require(set(Path(p).name for p in receipt['binary_sha256'])==set(plan['build_targets']),'Exact8 current ABI targets required')
 for path,want in audit['consumed_dependency_sha256'].items():
  if path.startswith('/src/'):current=SDK/'source'/path[5:]
  elif path.startswith('/ggml/'):current=SDK/'ggml-source'/path[6:]
  elif path.startswith('/build/'):current=SDK/'build'/path[7:]
  else:
   require(path.startswith(('/opt/intel/','/usr/')) and receipt['image']==plan['image'],'Compiler header outside exact admitted image');continue
  require(sha(current)==want,'Consumed source/header dependency changed '+path)
 for file,key in [('compile_commands.json','compile_commands_sha256'),('fresh-commands.txt','ninja_commands_sha256'),('fresh-dependencies.txt','ninja_dependencies_sha256')]:require(sha(SDK/'build'/file)==audit[key],'Actual compiler/dependency commands changed')
 require(sha(SDK/'build'/'CMakeCache.txt')==receipt['cmake_cache_sha256'],'Actual58 configuration changed')
 for name,want in plan['expected_patched_source_sha256'].items():require(sha(SDK/'source'/name)==want,'Actual57 source changed '+name)
 for name,want in receipt['binary_sha256'].items():require(Path(name).parent==SDK/'build'and sha(name)==want,'Actual57 binary changed '+name)
 require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Closing58 proof changed');return {'source_generation':58,'SDK':str(SDK),'receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'patched_source_files':75,'all_fresh_ABI_binaries':8,'old47_runtime_proof_transferred':False,'model_qualified':False}
