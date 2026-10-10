"""Read-only benchmark recollection; full pack/SDK payload checks are root-only."""
import argparse,base64,hashlib,json,math,time,statistics
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from upload_logical_epoch_factory_v2 import for_prepared,sources
from logical_witness_binding_v2 import binding,command_binding
ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'strata/flash-next/matched-upload-logical-gate-benchmark-source-plan-v2.json'
PLAN_SHA='d321495c608dbcf014d1a5cab378f9b5ce4de155d6b179684d5e3abee9bb0a6a'
SUMMARY_SHA='c59e0a52ef16ea8d7092d3a3f6eb0536f538a8dc61496d3a81b18ba3e293317b'
REPORT_SHA='172ff115da00da97fb95aec646eb852684ad1a8e63c696ab4a79b6d0e90ea7f3'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tree(root):
 result={}
 for p in sorted(Path(root).rglob('*')):
  require(not p.is_symlink(),'Benchmark artifact symlink refused')
  if p.is_file():result[str(p.relative_to(root))]={'sha256':sha(p),'stat5':[p.stat().st_dev,p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns]}
 return result

def source_binding():
 require(sha(PLAN)==PLAN_SHA,'Exact frozen benchmark V2 plan changed');plan=read_unique(PLAN)
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Current benchmark source closure changed '+name)
 sources();return len(plan['files'])
def byte_edges(rows,labels,expected,lower,upper):
 require(type(rows)is list and [r['label']for r in rows]==labels,'Exact complete byte edge roster differs');last=lower
 for row in rows:
  require(all(type(row[k])in (int,float)and math.isfinite(row[k])for k in ('started_epoch','finished_epoch')) and last<=row['started_epoch']<=row['finished_epoch']<=upper,'Complete byte chronology differs');last=row['finished_epoch'];require(canonical(row['rows'])==canonical(expected),'Saved/current complete bytes or stat5 differ')
 return rows

def zero_parser(proof,epoch,owner,lower,upper):
 require(proof['passed']is True and proof['owner_pid']==owner and not proof['worker_errors'] and proof['worker_commands']==[] and proof['parse_counts_complete']is True,'Unused parser epoch is not clean')
 for k in ('semantic_calls','semantic_executions','positive_parse_count','negative_parse_count','compact_requirement_calls','first_full_result_isolation_copies','compact_success_tokens'):require(type(proof[k])is int and proof[k]==0,'Unused parser counter differs '+k)
 require(canonical(proof['worker_runtime'])==canonical(epoch.runtime) and proof['live_guards_reused']is False and proof['saved_result_imported']is False,'Unused parser runtime/scope differs');rows=byte_edges(proof['evidence_byte_witness']['boundaries'],['entry','predevice','postoperation'],epoch.evidence.initial,lower,upper);packet=hashlib.sha256((canonical({'mode':'probe','parser_source_base64':base64.b64encode(epoch.source_raw).decode('ascii')})+'\n').encode('ascii')).hexdigest();command_binding(proof['worker_probe_command'],owner,[epoch.runtime['python_executable'],'-I','-B','-S',str(ROOT/'strata/flash-next/logical_free_immutable_worker_v2.py')],packet,0,lower,rows[0]['started_epoch'])

def trial_metadata(row,arm,warm_calls):
 require(row['arm']==arm and len(row['timings'])==warm_calls+1,'Actual ABBA trial/call roster differs');times=[row[k]for k in ('started_epoch','setup_finished_epoch','byte_seal_started_epoch','finished_epoch')];require(all(type(v)in(int,float)and math.isfinite(v)for v in times)and times==sorted(times),'Trial finite chronology differs')
 for i,t in enumerate(row['timings']):require(type(t['index'])is int and t['index']==i and t['operation_first']is (i==0) and t['operation_memo_warm']is (i>0) and type(t['nanoseconds'])is int and t['nanoseconds']>0,'Actual bounded timed call differs')
 for key in ('OS_page_cache_cold_claimed','cache_drop_or_model_inference_or_GPU_execution','serving_latency_or_speed_qualified'):require(row[key]is False,'Benchmark scope changed '+key)

def finalized_binding(root,*,full_bytes=True):
 require(type(full_bytes)is bool,'Exact full byte scope required');reader_sha=sha(__file__);root=Path(root).resolve();before=tree(root);require(set(before)=={'report.json','root-timing-summary-v1.json',*['trial-'+str(i)+'.json'for i in range(8)]},'Exact original benchmark file tree required');count=source_binding();require(sha(root/'report.json')==REPORT_SHA,'Exact actual benchmark report changed');report=read_unique(root/'report.json');require(report['passed']is True and report['errors']==[] and report['order']==list('ABBAABBA') and canonical(report['matched_configuration'])==canonical({'rounds':2,'warm_calls':4}),'Actual terminal/configuration differs');prepared=Path(report['prepared']);require(sha(prepared/'prepared.json')==report['prepared_sha256'] and sha(ROOT/'strata/flash-next/benchmark_upload_logical_gate_v2.py')==report['source_sha256'],'Original prepared/producer binding changed');logical,roster,scope=for_prepared([prepared]);sdk=pack=None
 if full_bytes:
  from sdk37_witness_scope_v1 import create_epoch
  from operation_pack_hash_witness_v2 import for_prepared as pack_for_prepared
  sdk=create_epoch([prepared/'prepared.json'],10800);pack=pack_for_prepared(prepared,10800)
 baseline=None;generation=None;owner=None;last=0;summaries=[];metric_inputs={arm:{'first':[],'warm':[],'complete':[]}for arm in 'AB'}
 for i,arm in enumerate(report['order']):
  item=report['trials'][i];path=root/('trial-'+str(i)+'.json');require(item['path']==str(path) and item['arm']==arm and sha(path)==item['sha256'],'Original trial file/SHA/order changed');row=read_unique(path);trial_metadata(row,arm,4);require(last<=row['started_epoch'],'Original trial overlap');last=row['finished_epoch'];owner_i=row['logical_witness']['owner_pid'];require(type(owner_i)is int and owner_i>0 and (owner is None or owner_i==owner),'Actual common benchmark owner differs');owner=owner_i
  if baseline is None:baseline=row['result'];generation=row['source_generation']
  require(canonical(row['result'])==canonical(baseline) and canonical(row['source_generation'])==canonical(generation) and canonical(row['logical_scope'])==canonical(scope),'Actual matched result/source/input corpus differs');lw=row['logical_witness'];edges=byte_edges(lw['evidence_byte_witness']['boundaries'],['entry','predevice','postoperation'],logical.evidence.initial,row['started_epoch'],row['finished_epoch']);require(edges[1]['started_epoch']>=row['byte_seal_started_epoch'],'Actual logical seal preceding declaration')
  if arm=='B':
   binding(lw,logical,owner,edges[1]['finished_epoch'],edges[2]['started_epoch'],case_specs=scope['case_specs']);zero_parser(row['independent_saved_worker_admission'],logical,owner,edges[2]['finished_epoch'],row['finished_epoch'])
  else:zero_parser(lw,logical,owner,row['started_epoch'],row['finished_epoch']);require(row['independent_saved_worker_admission']is None,'Old arm falsely imports parser result')
  for key,labels in [('pack_witness',['admission_start','predevice_complete_byte_recheck','postoperation_complete_byte_recheck']),('sdk_witness',['entry','predevice','postoperation'])]:
   witness=row[key];require(type(witness['owner_pid'])is int and witness['owner_pid']==owner and witness['stat_only_validation']is False and witness['saved_digest_imported']is False,'Actual byte owner/scope differs');expected=(pack.initial if key=='pack_witness'else sdk.initial)if full_bytes else witness['boundaries'][0]['rows'];be=byte_edges(witness['boundaries'],labels,expected,row['started_epoch'],row['finished_epoch']);require(be[1]['started_epoch']>=row['byte_seal_started_epoch'],'Actual complete byte seal preceding declaration')
   if key=='pack_witness':require(type(witness['semantic_digest_calls'])is int and witness['semantic_digest_calls']==0 and witness['unique_pack_files']==10,'Pack byte-only scope differs; no semantic consumption claim')
   else:
    require(type(witness['digest_calls'])is int and witness['digest_calls']==13 and witness['ELF_magic_calls']==0 and witness['unique_executables']==9,'Actual SDK gate/count scope differs')
    if full_bytes:require(all(canonical(edge['receipt_inputs'])==canonical(sdk.current_receipts)for edge in be),'SDK actual receipt corpus differs')
  metric_inputs[arm]['first'].append(row['timings'][0]['nanoseconds']/1e9);metric_inputs[arm]['warm'].extend(t['nanoseconds']/1e9 for t in row['timings'][1:]);metric_inputs[arm]['complete'].append(row['finished_epoch']-row['started_epoch'])
  summaries.append({'trial_sha256':item['sha256'],'arm':arm,'worker_executions':lw['semantic_executions'],'warm_nanoseconds':[t['nanoseconds']for t in row['timings'][1:]],'first_nanoseconds':row['timings'][0]['nanoseconds']})
 require(sha(root/'root-timing-summary-v1.json')==SUMMARY_SHA,'Exact root supplemental summary changed');summary=read_unique(root/'root-timing-summary-v1.json');metrics={arm:{'operation_first_median_seconds':statistics.median(v['first']),'repeated_gate_median_seconds':statistics.median(v['warm']),'complete_operation_median_seconds':statistics.median(v['complete']),'trials':len(v['first']),'timed_calls':len(v['first'])+len(v['warm'])}for arm,v in metric_inputs.items()};require(summary['report_sha256']==REPORT_SHA and canonical(summary['metrics'])==canonical(metrics),'Independent raw timing summary recollection differs');require(summary['repeated_gate_median_speed_ratio']==metrics['A']['repeated_gate_median_seconds']/metrics['B']['repeated_gate_median_seconds'] and summary['complete_operation_median_reduction_fraction']==1-metrics['B']['complete_operation_median_seconds']/metrics['A']['complete_operation_median_seconds'] and summary['first_call_is_slower_in_B']is True,'Root summary derived formulas differ');require(all(summary[k]is False for k in ('full_preparation_speed_qualified','GPU_serving_latency_or_speed_qualified','OS_cache_cold_claimed','CPU_background_isolation_observed')),'Actual timing scope broadened')
 neg=report['negative_controls'];require(all(neg[k]is True for k in ('old_owner_predicate_rejected','new_wrong_owner_rejected_and_latched','owned_copy_byte_mutation_rejected'))and neg['original_files_mutated']is False and neg['model_or_GPU_execution']is False,'Original declared negative control differs');failed=neg['failed_owner_witness'];require(len(failed)==1 and failed[0]['passed']is False and failed[0]['worker_errors'] and failed[0]['semantic_executions']==0 and failed[0]['compact_success_tokens']==0,'Caught owner rejection was not retained')
 if full_bytes:
  import c137_sdk37_digest_ports_v1 as old
  import c137_sdk37_logical_ports_v2 as new
  prepared_value=read_unique(prepared/'prepared.json');upload=prepared_value['upload_lifecycle'];args=(upload['path'],upload['oracle'],prepared_value['engine_receipt'],prepared_value['pack_receipt']);require(canonical(old.combined_generation_gate(Path(prepared_value['engine_receipt']).parent,sdk_epoch=sdk))==canonical(generation),'Current source generation changed');require(canonical(old.upload_gate(*args,sdk_epoch=sdk))==canonical(baseline),'Current original live upload predicates/result changed');require(canonical(new.upload_gate(*args,sdk_epoch=sdk,logical_epoch=logical,logical_roster=roster))==canonical(baseline),'Current compact live upload predicates/result changed')
 current_negative=None
 if full_bytes:
  from benchmark_upload_logical_gate_v2 import negatives
  current_negative=negatives(prepared)
 logical.seal_predevice(current_roster=roster);current_logical=logical.finalize(current_roster=roster);current_sdk=current_pack=None
 if full_bytes:
  sdk.seal_predevice();pack.seal_predevice();current_sdk=sdk.finalize();current_pack=pack.finalize()
 source_binding();require(sha(__file__)==reader_sha,'Read-only reader source changed');require(canonical(before)==canonical(tree(root)),'Original benchmark tree bytes/stat changed during recollection');return {'schema':1,'passed':True,'reader_source_sha256':reader_sha,'root_summary_sha256':SUMMARY_SHA,'independent_raw_timing_metrics':metrics,'current_negative_controls':current_negative,'original_negative_temporary_copy_bytes_currently_rejoined':False,'report_sha256':REPORT_SHA,'source_plan_sha256':PLAN_SHA,'current_source_files':count,'trials':summaries,'matched_original_results_recollected':True,'original_tree_unchanged':True,'current_original_logical_bytes_and_saved_worker_provenance_rejoined':True,'current_full_SDK_and_pack_byte_edges_rejoined':full_bytes,'current_logical_witness':current_logical,'current_sdk_witness':current_sdk,'current_pack_witness':current_pack,'pack_witness_semantic_consumption_claimed':False,'actual_GPU_or_model_inference':False,'serving_latency_or_speed_qualified':False,'metadata_only_scope':not full_bytes}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args();value=finalized_binding(a.root,full_bytes=not a.metadata_only)
 with a.output.open('x')as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'passed':value['passed'],'metadata_only_scope':value['metadata_only_scope'],'output':str(a.output)}))
if __name__=='__main__':main()
