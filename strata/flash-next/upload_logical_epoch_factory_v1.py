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
def sources():
 require(sha(PLAN)==PLAN_SHA,'Exact reviewed immutable parser consumer source plan required');p=read_unique(PLAN)
 for name,digest in p['files'].items():require(sha(ROOT/name)==digest,'Current full parser/consumer source closure changed '+name)
 return p

def for_prepared(prepared_roots,max_seconds=10800):
 p=sources();paths=[Path(root).resolve()/'prepared.json'for root in prepared_roots];require(1<=len(paths)<=4 and len(paths)==len(set(paths)),'Bounded explicit one/pair/selected prepared scope');required={str((HERE/name).relative_to(ROOT))for name in (*PROGRAM_SOURCE_NAMES,*COMPACT_SOURCES,'logical_free_immutable_worker_v2.py','parse_usm_logical_free_trace.py','original_upload_logical_requirement_v2.py','c137_sdk37_logical_ports_v2.py','c137_sdk37_digest_ports_v1.py','c1_serve_controller_combined_v137.py','sdk37_witness_scope_v1.py')};roster={ROOT/name:p['files'][name]for name in required};roster[PLAN]=PLAN_SHA;uploads={};bindings=[]
 for path in paths:
  prepared=read_unique(path);upload=prepared['upload_lifecycle'];receipt=Path(upload['path']).resolve();require(sha(receipt)==upload['sha256'],'Current expected upload receipt changed');report=read_unique(receipt);require(report['passed']is True and report['runner_generation']==2 and len(report['cases'])==5,'Exact successful five-case upload metadata required');require(sha(upload['oracle'])==upload['oracle_sha256'],'Current oracle receipt changed');bindings.append({'prepared_path':str(path),'prepared_sha256':sha(path),'upload_receipt':str(receipt),'upload_receipt_sha256':sha(receipt),'oracle_receipt':upload['oracle'],'oracle_receipt_sha256':upload['oracle_sha256']})
  roster[path]=sha(path);roster[receipt]=upload['sha256'];roster[Path(upload['oracle']).resolve()]=upload['oracle_sha256']
  for row in report['cases']:
   name=row['case'];require(type(name)is str and name and '/'not in name,'Exact upload case basename required');log=receipt.parent/(name+'.log');logical=receipt.parent/(name+'-logical-free.json');ledger=read_unique(logical);expected=ledger['log_sha256'];require(sha(log)==expected,'Current log differs original logical source binding');roster[log]=expected;roster[logical]=sha(logical)
  uploads[str(receipt)]=upload['sha256']
 # Full logical bytes and actual observed runtime bytes are independently read
 # at entry/predevice/post by the frozen known compact epoch constructor.
 items=list(roster.items());epoch=RequireLogicalEpoch(items,max_seconds);return epoch,items,{'prepared_scope':bindings,'upload_receipt_scope':uploads,'source_plan_sha256':PLAN_SHA,'full_live_outer_gates_still_required':True,'actual_GPU_touch':False,'model_weights_read':False}
