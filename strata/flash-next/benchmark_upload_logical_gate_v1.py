"""Root-executed CPU metadata ABBA; no SDK executable, device or model inference."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
import c137_sdk37_digest_ports_v1 as old
import c137_sdk37_logical_ports_v2 as new
from upload_logical_epoch_factory_v1 import for_prepared,sources
from sdk37_witness_scope_v1 import create_epoch
from operation_pack_hash_witness_v2 import for_prepared as pack_for_prepared
ROOT=Path(__file__).resolve().parents[2]
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 with Path(path).open('x')as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
def schedule(rounds):
 require(type(rounds)is int and 1<=rounds<=4,'Bounded actual ABBA rounds required');return [arm for r in range(rounds)for arm in ('A','B','B','A')]
def trial(prepared_root,arm,warm_calls):
 require(arm in ('A','B') and type(warm_calls)is int and 1<=warm_calls<=16,'Exact bounded trial configuration');sources();started=time.time();prepared=read_unique(Path(prepared_root)/'prepared.json');sdk=create_epoch([Path(prepared_root)/'prepared.json'],10800);pack=pack_for_prepared(prepared_root,10800);logical,roster,scope=for_prepared([prepared_root]);setup_finished=time.time();results=[];timings=[];upload=prepared['upload_lifecycle'];args=(upload['path'],upload['oracle'],prepared['engine_receipt'],prepared['pack_receipt'])
 # Both arms authenticate the exact current source37 generation before timing.
 generation=old.combined_generation_gate(Path(prepared['engine_receipt']).parent,sdk_epoch=sdk)
 for i in range(1+warm_calls):
  start=time.perf_counter_ns()
  value=old.upload_gate(*args,sdk_epoch=sdk)if arm=='A'else new.upload_gate(*args,sdk_epoch=sdk,logical_epoch=logical,logical_roster=roster)
  timings.append({'index':i,'operation_first':i==0,'operation_memo_warm':i>0,'nanoseconds':time.perf_counter_ns()-start});results.append(value)
 require(all(canonical(r)==canonical(results[0])for r in results),'Same-arm original result changed');seal_start=time.time();logical.seal_predevice(current_roster=roster);pack.seal_predevice();sdk.seal_predevice();logical_proof=logical.finalize(current_roster=roster);pack_proof=pack.finalize();sdk_proof=sdk.finalize();sources();require(logical_proof['passed']is True,'Complete logical evidence byte/provenance closure failed')
 return {'arm':arm,'started_epoch':started,'setup_finished_epoch':setup_finished,'byte_seal_started_epoch':seal_start,'finished_epoch':time.time(),'source_generation':generation,'result':results[0],'timings':timings,'logical_scope':scope,'logical_witness':logical_proof,'pack_witness':pack_proof,'sdk_witness':sdk_proof,'OS_page_cache_cold_claimed':False,'cache_drop_or_model_inference_or_GPU_execution':False,'serving_latency_or_speed_qualified':False}

def negatives(prepared_root):
 import tempfile
 import parse_usm_logical_free_trace as parser
 from logical_free_require_case_epoch_v2 import RequireLogicalEpoch,SOURCES
 from logical_free_immutable_epoch_v2 import HERE,WORKER,PARSER,PROGRAM_SOURCE_NAMES
 prepared=read_unique(Path(prepared_root)/'prepared.json');upload=Path(prepared['upload_lifecycle']['path']);report=read_unique(upload);case=report['cases'][0];log=upload.parent/(case['case']+'.log');logical=upload.parent/(case['case']+'-logical-free.json');raw=log.read_bytes();text=raw.decode('utf-8').replace('\r\n','\n').replace('\r','\n');want=sum(stage['unique_allocations']+1 for stage in case['report']['stages']);checked=parser.parse_trace(text);require(checked['passed']and checked['counts']['owners']==want,'Current positive owner control required');old_wrong_owner_rejected=checked['counts']['owners']!=want+1;worker_owned=False;mutation_rejected=False;proofs=[]
 # Work only on owned temporary copies, never alter original runtime evidence.
 with tempfile.TemporaryDirectory(prefix='upload-logical-negative-')as tmp:
  root=Path(tmp);copy_log=root/'log';copy_log.write_bytes(raw);copy_ledger=root/'logical';copy_ledger.write_bytes(logical.read_bytes());paths=[copy_log,copy_ledger,WORKER,PARSER,*(HERE/name for name in PROGRAM_SOURCE_NAMES),*(HERE/name for name in SOURCES)];roster=[(path,sha(path))for path in paths];epoch=RequireLogicalEpoch(roster,900)
  try:epoch.require_case(copy_log,copy_ledger,want+1,current_roster=roster)
  except ValueError:worker_owned=epoch.failed
  require(worker_owned,'Wrong owner control was not rejected/latching');epoch.seal_predevice(current_roster=roster);failed=epoch.finalize(current_roster=roster);require(failed['passed']is False,'Caught owner failure published success');proofs.append(failed)
  epoch=RequireLogicalEpoch(roster,900);epoch.require_case(copy_log,copy_ledger,want,current_roster=roster);copy_log.write_bytes(raw+b'changed-copy-control\n')
  try:epoch.seal_predevice(current_roster=roster)
  except ValueError:mutation_rejected=True
  require(mutation_rejected,'Changed owned copy escaped complete-byte seal')
 return {'old_owner_predicate_rejected':old_wrong_owner_rejected,'new_wrong_owner_rejected_and_latched':worker_owned,'owned_copy_byte_mutation_rejected':mutation_rejected,'failed_owner_witness':proofs,'original_files_mutated':False,'model_or_GPU_execution':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=2);p.add_argument('--warm-calls',type=int,default=4);a=p.parse_args();order=schedule(a.rounds);require(not a.output.exists(),'New immutable benchmark output required');a.output.mkdir();errors=[];rows=[];baseline=None;controls=None
 try:
  for index,arm in enumerate(order):
   row=trial(a.prepared,arm,a.warm_calls);write(a.output/('trial-'+str(index)+'.json'),row)
   if baseline is None:baseline=row['result']
   require(canonical(row['result'])==canonical(baseline),'Exact old/new gate result mismatch');rows.append({'path':str((a.output/('trial-'+str(index)+'.json')).resolve()),'sha256':sha(a.output/('trial-'+str(index)+'.json')),'arm':arm})
  controls=negatives(a.prepared)
 except BaseException as exc:errors.append({'type':type(exc).__name__,'error':str(exc)})
 report={'schema':1,'passed':not errors and len(rows)==len(order),'order':order,'trials':rows,'errors':errors,'negative_controls':controls,'prepared':str(a.prepared.resolve()),'prepared_sha256':sha(a.prepared/'prepared.json'),'source_sha256':sha(__file__),'matched_configuration':{'rounds':a.rounds,'warm_calls':a.warm_calls},'CPU_background_isolation_observed':False,'root_quiet_window_required':True,'OS_cache_cold_claimed':False,'actual_GPU_or_model_inference':False,'complete_model_math_or_quality_qualified':False,'serving_latency_or_speed_qualified':False};write(a.output/'report.json',report);print(json.dumps({'passed':report['passed'],'report':str(a.output/'report.json')}));return 0 if report['passed']else 1
if __name__=='__main__':raise SystemExit(main())
