"""Root-only current metadata/log byte epoch factory; no model/GPU execution."""
import hashlib,json
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
from logical_free_require_case_epoch_v2 import RequireLogicalEpoch,SOURCES as COMPACT_SOURCES
from logical_free_immutable_epoch_v2 import WORKER,PARSER,PROGRAM_SOURCE_NAMES
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'upload-logical-consumer-ports-source-plan-v2.json'
PLAN_SHA='c75684185ff69f4174c4fc2e7ce1f2025640177246403562f68cb209a313e9a9'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def snapshot_sha(path):
 path=Path(path);require(path.is_file() and not path.is_symlink() and path.absolute()==path.resolve(),'Exact regular nonalias upload byte path required');before=path.stat();fingerprint=lambda row:(row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns);digest=sha(path);require(fingerprint(before)==fingerprint(path.stat()),'Upload bytes changed during complete snapshot');return digest
def sources():
 require(sha(PLAN)==PLAN_SHA,'Exact reviewed immutable parser consumer source plan required');p=read_unique(PLAN)
 for name,digest in p['files'].items():require(sha(ROOT/name)==digest,'Current full parser/consumer source closure changed '+name)
 return p

def for_prepared(prepared_roots,max_seconds=10800):
 p=sources();paths=[Path(root).resolve()/'prepared.json'for root in prepared_roots];require(1<=len(paths)<=4 and len(paths)==len(set(paths)),'Bounded explicit one/pair/selected prepared scope');required={str((HERE/name).relative_to(ROOT))for name in (*PROGRAM_SOURCE_NAMES,*COMPACT_SOURCES,'logical_free_immutable_worker_v2.py','parse_usm_logical_free_trace.py','original_upload_logical_requirement_v2.py','c137_sdk37_logical_ports_v2.py','c137_sdk37_digest_ports_v1.py','c1_serve_controller_combined_v137.py','sdk37_witness_scope_v1.py')};roster={ROOT/name:p['files'][name]for name in required};roster[PLAN]=PLAN_SHA;uploads={};bindings=[];specs={}
 for path in paths:
  prepared=read_unique(path);upload=prepared['upload_lifecycle'];receipt=Path(upload['path']).resolve();require(sha(receipt)==upload['sha256'],'Current expected upload receipt changed');report=read_unique(receipt);require(report['passed']is True and report['runner_generation']==2 and len(report['cases'])==5,'Exact successful five-case upload metadata required');require(sha(upload['oracle'])==upload['oracle_sha256'],'Current oracle receipt changed');bindings.append({'prepared_path':str(path),'prepared_sha256':sha(path),'upload_receipt':str(receipt),'upload_receipt_sha256':sha(receipt),'oracle_receipt':upload['oracle'],'oracle_receipt_sha256':upload['oracle_sha256']})
  roster[path]=sha(path);roster[receipt]=upload['sha256'];roster[Path(upload['oracle']).resolve()]=upload['oracle_sha256']
  for row in report['cases']:
   name=row['case'];require(type(name)is str and name and '/'not in name,'Exact upload case basename required');log=receipt.parent/(name+'.log');logical=receipt.parent/(name+'-logical-free.json');expected=snapshot_sha(log);ledger_sha=snapshot_sha(logical);roster[log]=expected;roster[logical]=ledger_sha;stages=row['report']['stages'];require(type(stages)is list and stages and all(type(stage['unique_allocations'])is int and stage['unique_allocations']>=0 for stage in stages),'Typed actual upload stage allocation counts required');spec={'log_path':str(log.resolve()),'logical_path':str(logical.resolve()),'expected_owner_count':sum(stage['unique_allocations']+1 for stage in stages)};key=(spec['log_path'],spec['logical_path']);require(key not in specs or canonical(specs[key])==canonical(spec),'Conflicting current upload case specification');specs[key]=spec
  uploads[str(receipt)]=upload['sha256']
 # Full logical bytes and actual observed runtime bytes are independently read
 # at entry/predevice/post by the frozen known compact epoch constructor.
 items=list(roster.items());epoch=RequireLogicalEpoch(items,max_seconds);return epoch,items,{'case_specs':list(specs.values()),'prepared_scope':bindings,'upload_receipt_scope':uploads,'source_plan_sha256':PLAN_SHA,'full_live_outer_gates_still_required':True,'actual_GPU_touch':False,'model_weights_read':False}

def case_specs_for_prepared(prepared_roots):
 """Rejoin typed expected owner counts to current authenticated upload metadata."""
 specs={}
 for root in prepared_roots:
  prepared=read_unique(Path(root).resolve()/'prepared.json');upload=prepared['upload_lifecycle'];receipt=Path(upload['path']).resolve();require(sha(receipt)==upload['sha256'],'Current upload case specification receipt changed');report=read_unique(receipt);require(report['passed']is True and type(report['runner_generation'])is int and report['runner_generation']==2 and len(report['cases'])==5,'Current five-case original upload metadata required')
  for row in report['cases']:
   name=row['case'];require(type(name)is str and name and '/'not in name,'Exact upload case basename required');stages=row['report']['stages'];require(type(stages)is list and stages and all(type(stage['unique_allocations'])is int and stage['unique_allocations']>=0 for stage in stages),'Typed actual upload allocation counts required');spec={'log_path':str(receipt.parent/(name+'.log')),'logical_path':str(receipt.parent/(name+'-logical-free.json')),'expected_owner_count':sum(stage['unique_allocations']+1 for stage in stages)};key=(spec['log_path'],spec['logical_path']);require(key not in specs or canonical(specs[key])==canonical(spec),'Conflicting authenticated upload case metadata');specs[key]=spec
 require(specs,'Explicit current upload cases required');return list(specs.values())
