"""Supplied actual SDK61 receipt pins; no invented build proof or old ABI transfer."""
import hashlib,json
from pathlib import Path
SDK=None;RECEIPT_SHA=None
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native61-host-actual-chunk-selection-engine-build-proposal-v1.json'
PLAN_SHA='34d49ee71fcee30b4059c37cb166c20a1356454dc0abe2565f21f2be59ab659e'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 path=Path(path);require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Canonical actual SDK file required');before=path.stat();h=hashlib.sha256()
 with path.open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'SDK file changed during hash');return h.hexdigest()
def configure(root,receipt_sha):
 global SDK,RECEIPT_SHA
 require(type(receipt_sha)is str and len(receipt_sha)==64 and all(c in '0123456789abcdef'for c in receipt_sha),'Actual receipt SHA required');root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical actual SDK61 root required');require(sha(root/'receipt.json')==receipt_sha,'Original actual receipt bytes changed');receipt=json.loads((root/'receipt.json').read_bytes());require(receipt['plan_sha256']==PLAN_SHA,'Receipt must belong to actual61 proposal');SDK=root;RECEIPT_SHA=receipt_sha
 return {'SDK61_root':str(root),'SDK61_receipt_sha256':receipt_sha,'actual_build_bytes_still_rechecked_before_model':True}
def binding():
 require(SDK is not None and RECEIPT_SHA is not None,'Actual SDK61 build not configured; no placeholder proof allowed');require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Exact actual61 plan/receipt required');plan=json.loads(PLAN.read_bytes());receipt=json.loads((SDK/'receipt.json').read_bytes());require(type(receipt['build_rc'])is int and receipt['build_rc']==0 and receipt['devices_exposed']is False and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['plan_sha256']==PLAN_SHA,'Fresh actual61 build/isolation proof required');require(len(plan['expected_patched_source_sha256'])==76 and len(receipt['binary_sha256'])==8,'Current76 source/all8 ABI required')
 require(receipt['build_strategy']=='direct60_to61_isolated_incremental_v1' and receipt['fresh_all_source_compilation'] is False and receipt['CMake_reconfiguration_performed'] is False and receipt['parent_unchanged'] is True,'Honest matched-header incremental61 receipt required')
 require(receipt['parent_root']==plan['source61_parent_actual_sdk'] and receipt['parent_receipt_sha256']==plan['source61_parent_actual_receipt_sha256'] and receipt['external_source_root']==str(Path(plan['source61_parent_actual_sdk'])/'source') and receipt['external_source_identity_kind']=='immutable_actual60_source_snapshot','Direct immutable actual58 parent required')
 require(sha(SDK/'plan.snapshot.json')==PLAN_SHA and sha(SDK/'parent-provenance.json')==receipt['parent_provenance_sha256'],'Actual59 provenance/plan changed')
 require(receipt['owned_normal_exit'] is True and receipt['owned_removed'] is True and receipt['original_foreground_EOF'] is True and receipt['forced_cleanup'] is False and receipt['compiler_return_code']==0 and receipt['compiler_errors']==[],'Actual59 typed owned compiler normal retirement required')
 require(receipt['compiler_resource_bounds']=={'cpus':4,'memory_bytes':16<<30,'memory_swap_bytes':16<<30,'pids_limit':512,'network':'none','NoDri':True},'Actual59 bounded NoDri compiler required')
 for file,key in [('compiler-terminal.json','compiler_terminal_sha256'),('compiler-client-terminal.json','compiler_client_terminal_sha256'),('compiler-removed.json','compiler_removed_sha256')]:require(sha(SDK/file)==receipt[key],'Original compiler retirement artifact changed')
 require(len(receipt['newly_compiled_objects'])==2 and len(receipt['reused_objects_sha256'])==122 and receipt['all8_ELFs_deleted_before_relink'] is True and receipt['copied_config_unchanged'] is True,'Exactly main/Prefill compiled/122 objects reused/all8 fresh links required')
 require({row['path'] for row in receipt['newly_compiled_objects']}=={'CMakeFiles/strata.dir/src/program/generate.cpp.o','CMakeFiles/strata_prefill.dir/src/prefill/prefill.cpp.o'},'Only current main/Prefill objects may change')
 require(set(receipt['updated_archive_sha256'])=={'libstrata_prefill.a'},'Only current Prefill archive may update')
 for name,want in receipt['updated_archive_sha256'].items():require(sha(SDK/'build'/name)==want,'Fresh Prefill archive changed')
 for name,want in receipt['reused_objects_sha256'].items():require(sha(SDK/'build'/name)==want,'Reused current object changed '+name)
 for name,want in receipt['reused_archives_sha256'].items():require(sha(SDK/'build'/name)==want,'Reused current ABI archive changed '+name)
 audit=json.loads((SDK/'build'/'incremental-audit.json').read_bytes());require(audit==receipt['incremental_dependency_audit'] and audit['passed'] is True and audit['source_headers_unchanged'] is True and audit['dependency_objects']==124,'Actual59 full consumed-header/object audit required')
 proof=json.loads((SDK/'parent-provenance.json').read_bytes())
 for row in receipt['newly_compiled_objects']:require(sha(SDK/'build'/row['path'])==row['after_sha256'],'Current fresh main/Prefill object changed')
 require(set(receipt['reused_objects_sha256'])==set(proof['object_sha256'])-{row['path'] for row in receipt['newly_compiled_objects']} and all(want==proof['object_sha256'][p] for p,want in receipt['reused_objects_sha256'].items()) and receipt['reused_archives_sha256']=={p:h for p,h in proof['archive_sha256'].items() if p!='libstrata_prefill.a'},'Exactly authoritative current object/archive reuse required')
 for p,want in proof['source_tree'].items():
  if p not in ('include/strata/prefill/host60.hpp','sycl/src/prefill/prefill.cpp'):require(sha(SDK/'source'/p)==want,'Copied source/header changed '+p)
 for p,want in proof['ggml_tree'].items():require(sha(SDK/'ggml-source'/p)==want,'Copied ggml dependency changed '+p)
 for file in ('CMakeCache.txt','build.ninja','compile_commands.json','compiler-version.txt','packages.txt'):require(sha(SDK/'build'/file)==proof['build_metadata_sha256'][file],'Preserved current compiler config changed '+file)
 require(set(Path(p).name for p in receipt['binary_sha256'])==set(plan['build_targets']),'Exact all8 new ABI targets required')
 require(sha(SDK/'build'/'CMakeCache.txt')==receipt['cmake_cache_sha256'],'Actual59 configuration changed')
 for name,want in plan['expected_patched_source_sha256'].items():require(sha(SDK/'source'/name)==want,'Actual57 source changed '+name)
 for name,want in receipt['binary_sha256'].items():require(Path(name).parent==SDK/'build'and sha(name)==want,'Actual57 binary changed '+name)
 require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Closing61 proof changed');return {'source_generation':61,'SDK':str(SDK),'receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'patched_source_files':76,'all_fresh_ABI_binaries':8,'old47_runtime_proof_transferred':False,'model_qualified':False}
