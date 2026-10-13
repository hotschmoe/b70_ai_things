"""Actual new QSA35/mathidentity SDK receipt; no old62 ABI runtime proof transfer."""
import hashlib,json
from pathlib import Path
SDK=None;RECEIPT_SHA=None
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native62-qsa35-session-math-engine-proposal-v1.json'
PLAN_SHA='69d3e02e2d97aa8fd3921e90da9f48e8b4520cfe3515067b0dac09775b83cd61'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 path=Path(path);require(path.resolve()==path and path.is_file()and not path.is_symlink(),'Canonical actual SDK file required');before=path.stat();h=hashlib.sha256()
 with path.open('rb')as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'SDK file changed during hash');return h.hexdigest()
def configure(root,receipt_sha):
 global SDK,RECEIPT_SHA
 require(type(receipt_sha)is str and len(receipt_sha)==64 and all(c in '0123456789abcdef'for c in receipt_sha),'Actual receipt SHA required');root=Path(root);require(root.is_absolute()and root.resolve()==root,'Canonical actual SDK62 root required');require(sha(root/'receipt.json')==receipt_sha,'Original actual receipt bytes changed');receipt=json.loads((root/'receipt.json').read_bytes());require(receipt['plan_sha256']==PLAN_SHA,'Receipt must belong to actual QSA35 proposal');SDK=root;RECEIPT_SHA=receipt_sha
 return {'SDK62_root':str(root),'SDK62_receipt_sha256':receipt_sha,'actual_build_bytes_still_rechecked_before_model':True}
def binding():
 require(SDK is not None and RECEIPT_SHA is not None,'Actual QSA35 SDK build not configured; no placeholder proof allowed');require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Exact actual QSA35 plan/receipt required');plan=json.loads(PLAN.read_bytes());receipt=json.loads((SDK/'receipt.json').read_bytes());require(type(receipt['build_rc'])is int and receipt['build_rc']==0 and receipt['devices_exposed']is False and receipt['external_source_unchanged']is True and receipt['plan_snapshot_unchanged']is True and receipt['plan_sha256']==PLAN_SHA,'Fresh actual QSA35 build/isolation proof required');require(len(plan['expected_patched_source_sha256'])==76 and len(receipt['binary_sha256'])==8,'Current76 source/all8 ABI required')
 require(receipt['build_strategy']=='direct62_to_qsa35_isolated_incremental_v1' and receipt['fresh_all_source_compilation'] is False and receipt['CMake_reconfiguration_performed'] is False and receipt['parent_unchanged'] is True,'Honest matched-header incremental62 receipt required')
 require(receipt['parent_root']==plan['source62_parent_actual_sdk'] and receipt['parent_receipt_sha256']==plan['source62_parent_actual_receipt_sha256'] and receipt['external_source_root']==str(Path(plan['source62_parent_actual_sdk'])/'source') and receipt['external_source_identity_kind']=='immutable_actual QSA35_source_snapshot','Direct immutable actual62 parent required')
 require(sha(SDK/'plan.snapshot.json')==PLAN_SHA and sha(SDK/'parent-provenance.json')==receipt['parent_provenance_sha256'],'New QSA35 provenance/plan changed')
 require(receipt['owned_normal_exit'] is True and receipt['owned_removed'] is True and receipt['original_foreground_EOF'] is True and receipt['forced_cleanup'] is False and receipt['compiler_return_code']==0 and receipt['compiler_errors']==[],'New QSA35 typed owned compiler normal retirement required')
 require(receipt['compiler_resource_bounds']=={'cpus':4,'memory_bytes':16<<30,'memory_swap_bytes':16<<30,'pids_limit':512,'network':'none','NoDri':True},'New QSA35 bounded NoDri compiler required')
 for file,key in [('compiler-terminal.json','compiler_terminal_sha256'),('compiler-client-terminal.json','compiler_client_terminal_sha256'),('compiler-removed.json','compiler_removed_sha256')]:require(sha(SDK/file)==receipt[key],'Original compiler retirement artifact changed')
 require(len(receipt['newly_compiled_objects'])==2 and len(receipt['reused_objects_sha256'])==122 and receipt['all8_ELFs_deleted_before_relink'] is True and receipt['copied_config_unchanged'] is True,'Exactly two consumers compiled/122 reused/all8 fresh links required')
 require({row['path'] for row in receipt['newly_compiled_objects']}==set(plan['expected_changed_objects']),'All two current observer/identity/header consumers must be compiled')
 require(set(receipt['updated_archive_sha256'])=={'libstrata_engine.a'},'Exactly one current consumer archive rebuilt')
 for name,want in receipt['updated_archive_sha256'].items():require(sha(SDK/'build'/name)==want,'Fresh engine archive changed')
 for name,want in receipt['reused_objects_sha256'].items():require(sha(SDK/'build'/name)==want,'Reused current object changed '+name)
 for name,want in receipt['reused_archives_sha256'].items():require(sha(SDK/'build'/name)==want,'Reused current ABI archive changed '+name)
 audit=json.loads((SDK/'build'/'incremental-audit.json').read_bytes());require(audit==receipt['incremental_dependency_audit'] and audit['passed'] is True and audit['unchanged_source_headers_rechecked'] is True and audit['all_changed_header_consumers_accounted'] is True and audit['dependency_objects']==124,'New QSA35 full consumed-header/object audit required')
 final_audit=json.loads((SDK/'build'/'incremental-final-audit.json').read_bytes());require(final_audit==receipt['final_incremental_dependency_audit'] and final_audit['passed'] is True and final_audit['dependency_objects']==124 and set(audit['changed_objects'])==set(final_audit['changed_objects'])==set(plan['expected_changed_objects']),'Complete final consumed-header/124-object proof required')
 proof=json.loads((SDK/'parent-provenance.json').read_bytes())
 require(all(final_audit['dependency_lists_sha256'][n]==audit['dependency_lists_sha256'][n] for n in receipt['reused_objects_sha256']),'Reused-object dependency list changed')
 for row in receipt['newly_compiled_objects']:require(sha(SDK/'build'/row['path'])==row['after_sha256'],'Current fresh consumer object changed')
 require(set(receipt['reused_objects_sha256'])==set(proof['object_sha256'])-{row['path'] for row in receipt['newly_compiled_objects']} and all(want==proof['object_sha256'][p] for p,want in receipt['reused_objects_sha256'].items()) and receipt['reused_archives_sha256']=={p:h for p,h in proof['archive_sha256'].items() if p not in receipt['updated_archive_sha256']},'Exactly authoritative current object/archive reuse required')
 for p,want in proof['source_tree'].items():
  if p not in plan['source62_changed_files']:require(sha(SDK/'source'/p)==want,'Copied source/header changed '+p)
 for p,want in proof['ggml_tree'].items():require(sha(SDK/'ggml-source'/p)==want,'Copied ggml dependency changed '+p)
 for file in ('CMakeCache.txt','build.ninja','compile_commands.json','compiler-version.txt','packages.txt'):require(sha(SDK/'build'/file)==proof['build_metadata_sha256'][file],'Preserved current compiler config changed '+file)
 require(set(Path(p).name for p in receipt['binary_sha256'])==set(plan['build_targets']),'Exact all8 new ABI targets required')
 require(sha(SDK/'build'/'CMakeCache.txt')==receipt['cmake_cache_sha256'],'New QSA35 configuration changed')
 for name,want in plan['expected_patched_source_sha256'].items():require(sha(SDK/'source'/name)==want,'New QSA35 source changed '+name)
 for name,want in receipt['binary_sha256'].items():require(Path(name).parent==SDK/'build'and sha(name)==want,'New QSA35 binary changed '+name)
 require(sha(PLAN)==PLAN_SHA and sha(SDK/'receipt.json')==RECEIPT_SHA,'Closing62 proof changed');return {'source_generation':62,'source_successor':'qsa35_session_math_v1','QSA35_observer_capability':True,'session_math_identity_version':'v8','SDK':str(SDK),'receipt_sha256':RECEIPT_SHA,'plan_sha256':PLAN_SHA,'patched_source_files':76,'all_fresh_ABI_binaries':8,'old62_runtime_proof_transferred':False,'model_qualified':False}
