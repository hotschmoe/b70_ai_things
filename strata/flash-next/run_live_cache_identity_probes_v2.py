"""ROOT ONLY actual HTTP namespace probes against the already-owned actor."""
import hashlib,json,time,urllib.request,urllib.error
from pathlib import Path
from live_cache_identity_namespace_v2 import namespace_probes,FIELD,require
from live_cache_identity_experiment_v2 import refusal_binding

def records(path):
 raw=Path(path).read_bytes();require(not raw or raw.endswith(b'\n'),'Original complete trace publication required');from serial37_canonical_json_v3 import unique_object,finite_constant
 return [json.loads(line,object_pairs_hook=unique_object,parse_constant=finite_constant)for line in raw.splitlines()if line]
def native_census(path):
 rows=records(path);begins=[r for r in rows if r['kind']=='engine_begin'];ends=[r for r in rows if r['kind']=='engine_end'];require(begins and len(begins)==len(ends),'Actual previously completed priming requests required');owners={(r['engine_pid'],r['engine_generation'])for r in begins+ends};require(len(owners)==1,'One actual original native incarnation required');pid,generation=next(iter(owners));return {'native_pid':pid,'native_generation':generation,'native_begin_count':len(begins),'native_terminal_count':len(ends)}

def run(url,owner,messages,model,native_trace,guard_trace,output):
 import c1_serve_controller_combined_v140_v3 as c1
 c1.leased([0,1]);require(url.startswith('http://127.0.0.1:')and type(messages)is list and messages,'Actual owned loopback actor URL/messages required');out=Path(output);require(not out.exists(),'Fresh actual negative-probe receipt directory required');out.mkdir();result=[]
 for index,probe in enumerate(namespace_probes(owner)):
  before=native_census(native_trace);guard_before=len(records(guard_trace));request={'model':model,'messages':messages,'stream':True,'temperature':0,'max_tokens':1,'frequency_penalty':0,'presence_penalty':0,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'strata_fresh':False,'strata_shared_prefix':{'tokens':272},FIELD:probe['requested_namespace']};raw=json.dumps(request,ensure_ascii=True).encode('ascii');started=time.time();req=urllib.request.Request(url+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=60)as response:raise ValueError('Foreign live namespace unexpectedly admitted: '+str(response.status))
  except urllib.error.HTTPError as error:
   body=error.read();status=error.code
  require(status==400 and probe['expected_refusal']in body.decode('utf-8'),'Actual exact before-cache HTTP refusal required');after=native_census(native_trace);guard=records(guard_trace)[guard_before:];binding=refusal_binding(probe,guard,before,after);row={'probe':probe,'request':request,'actual_url':req.full_url,'request_sha256':hashlib.sha256(raw).hexdigest(),'http_status':status,'response_hex':body.hex(),'response_sha256':hashlib.sha256(body).hexdigest(),'guard_rows':guard,'native_before':before,'native_after':after,'source_refusal_binding':binding,'started_epoch':started,'finished_epoch':time.time(),'full_cache_runtime_qualified':False};(out/(str(index)+'.json')).write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n');result.append(row)
 return result

def recollect(rows,native_events,guard_events):
 import math
 require(type(rows)is list and len(rows)==4,'All preregistered namespace components required');namespace=rows[0]['probe']['requested_namespace'];components=[r['probe']['component']for r in rows];require(len(set(components))==4,'Complete unique component refusals required')
 for row in rows:
  require(row['native_before']==census_at(native_events,row['started_epoch'])and row['native_after']==census_at(native_events,row['finished_epoch']),'Saved refusal counters differ from actual native event prefixes')
  require(row['http_status']==400 and bytes.fromhex(row['response_hex']) and hashlib.sha256(bytes.fromhex(row['response_hex'])).hexdigest()==row['response_sha256'],'Original negative raw HTTP bytes changed');require(row['guard_rows']==[e for e in guard_events if e==row['guard_rows'][0]],'Original unique guard trace marker required');refusal_binding(row['probe'],row['guard_rows'],row['native_before'],row['native_after']);activity=[e for e in native_events if e['kind']in('engine_begin','engine_end')];require(all(type(e['epoch'])in(int,float)and math.isfinite(e['epoch'])for e in activity),'Source-bound original native event wall epochs required; unknown clock cannot prove absence');require(not any(row['started_epoch']<=e['epoch']<=row['finished_epoch']for e in activity),'Original native activity during rejected request')
 return {'all_four_actual_frontend_namespace_refusals':True,'other_weights_loaded':False,'native_cache_model_key_qualified':False,'full_cache_runtime_qualified':False}

def loaded_probes(url,messages,model,native_trace,guard_trace,owner_packet,output):
 from full_cache_live_identity_api_v2 import PROBE_FIELD
 from live_cache_identity_namespace_v2 import KEYS
 import c1_serve_controller_combined_v140_v3 as c1
 c1.leased([0,1]);nonce=hashlib.sha256(Path(owner_packet).read_bytes()).hexdigest();out=Path(output);require(not out.exists(),'Fresh actual loaded-owner attempt receipts required');out.mkdir();rows=[]
 for component in (*KEYS,'authoritative_rebind','native_restart'):
  before=native_census(native_trace);start_guard=len(records(guard_trace));request={'model':model,'messages':messages,'stream':True,'temperature':0,'max_tokens':1,'chat_template_kwargs':{'enable_thinking':False},'reasoning_budget_tokens':0,'strata_fresh':False,'strata_shared_prefix':{'tokens':272},PROBE_FIELD:{'component':component,'nonce':nonce}};raw=json.dumps(request).encode('ascii');started=time.time()
  try:
   with urllib.request.urlopen(urllib.request.Request(url+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'}),timeout=60)as response:raise ValueError('Loaded-owner mutation/rebind unexpectedly admitted '+str(response.status))
  except urllib.error.HTTPError as error:status=error.code;body=error.read()
  expected='LIVE_NAMESPACE_REBIND_REFUSED'if component in('authoritative_rebind','native_restart')else'LIVE_LOADED_OWNER_REFUSED';require(status==400 and expected in body.decode('utf-8'),'Actual authoritative loaded-owner refusal required');after=native_census(native_trace);events=records(guard_trace)[start_guard:];require(len(events)==1 and events[0]['kind']=='live_identity_prepare_refused'and events[0]['original_prepare_invoked']is False and events[0]['native_request_authorized']is False and events[0]['actual_model_weight_data_changed']is False and before==after,'Original owner attempt must not invoke native/cache route');row={'component':component,'actual_request':request,'response_status':status,'response_hex':body.hex(),'started_epoch':started,'finished_epoch':time.time(),'native_before':before,'native_after':after,'original_guard_rows':events,'actual_other_model_weights_loaded':False,'full_cache_runtime_qualified':False};(out/(component+'.json')).write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n');rows.append(row)
 return rows

def recollect_loaded(rows,native_events,guard_events):
 """Recompute each actual metadata-update refusal; no other weights inference."""
 import math
 from live_cache_identity_namespace_v2 import KEYS
 from full_cache_live_identity_api_v2 import PROBE_FIELD
 require(type(rows)is list and [r['component'] for r in rows]==[*KEYS,'authoritative_rebind','native_restart'],'All loaded owner attempts exactly once required')
 for row in rows:
  require(row['native_before']==census_at(native_events,row['started_epoch'])and row['native_after']==census_at(native_events,row['finished_epoch']),'Loaded refusal saved counters differ from actual native source')
  body=bytes.fromhex(row['response_hex']);expected='LIVE_NAMESPACE_REBIND_REFUSED'if row['component']in('authoritative_rebind','native_restart')else'LIVE_LOADED_OWNER_REFUSED'
  require(row['response_status']==400 and expected in body.decode('utf-8')and row['actual_other_model_weights_loaded']is False,'Actual loaded metadata refusal changed')
  events=row['original_guard_rows'];require(len(events)==1 and sum(e==events[0]for e in guard_events)==1,'Unique original loaded guard marker required');event=events[0]
  require(event['kind']=='live_identity_prepare_refused'and event['original_prepare_invoked']is False and event['native_request_authorized']is False and event['loaded_owner_replaced']is False and event['actual_model_weight_data_changed']is False and event['loaded_owner_probe']==row['actual_request'][PROBE_FIELD],'Loaded owner refusal source/request differs')
  attempted=row['component']in KEYS
  require(event['loaded_metadata_mutation_attempted']is attempted and event['loaded_metadata_restored']is True,'Actual metadata mutation/restoration required; missing/false is not observation')
  before=event['actual_loaded_owner_before_binding'];restored=event['restored_actual_loaded_owner_binding'];require(type(before)is dict and before==restored and before['native_incarnation']=={'pid':row['native_before']['native_pid'],'generation':row['native_before']['native_generation']},'Exact original loaded owner/native restored binding differs')
  require(event['prepare_code_sha256_before']==event['prepare_code_sha256_restored'] and type(event['prepare_code_sha256_before'])is str and len(event['prepare_code_sha256_before'])==64,'Original source function restoration differs')
  require(expected in event['error'] and event['error'].split(': ',1)[-1]in body.decode('utf-8'),'Actual guard/HTTP refusal error attribution differs')
  require(set(before)=={'loaded_namespace_sha256','native_incarnation','actual_loaded_metadata_snapshot','whole_weight_mutation_observed'}and before['whole_weight_mutation_observed']is False and set(before['native_incarnation'])=={'pid','generation'}and all(type(v)is int and v>0 for v in before['native_incarnation'].values()),'Typed exact actual owner binding required')
  snapshot=event['attempted_loaded_metadata_snapshot']
  fields={'model_sha256':'artifact_metadata_sha256','tokenizer_sha256':'tokenizer_loaded_metadata_sha256','template_sha256':'template_loaded_source_sha256'}
  if attempted:
   require(type(snapshot)is dict and set(snapshot)=={'metadata','prepare_code_sha256'},'Actual attempted value snapshot missing')
   base=before['actual_loaded_metadata_snapshot'];require(set(snapshot['metadata'])==set(base),'Actual loaded snapshot shape differs')
   changed=[key for key in base if base[key]!=snapshot['metadata'][key]]
   if row['component']=='source_sha256':require(changed==[]and snapshot['prepare_code_sha256']!=event['prepare_code_sha256_before'],'Actual source mutation not observed')
   else:require(changed==[fields[row['component']]]and snapshot['prepare_code_sha256']==event['prepare_code_sha256_before'],'Exact one loaded metadata mutation missing/foreign')
  else:require(snapshot is None,'Rebind/restart refusal cannot claim metadata mutation')
  prior=[e for e in guard_events if e['kind']=='live_identity_prepare_admitted'and e['finished_epoch']<event['started_epoch']]
  later=[e for e in guard_events if e['kind']=='live_identity_prepare_admitted'and e['started_epoch']>event['finished_epoch']]
  require(prior and later,'Original admitted priming and repeat restoration proofs required')
  for observed in (prior[-1],later[0]):require(observed['actual_loaded_owner_binding']==before and observed['prepare_code_sha256']==event['prepare_code_sha256_before']and observed['original_prepare_invoked']is True and observed['loaded_owner_replaced']is False,'Original admitted loaded owner/source before and after refusal differs')
  require(row['native_before']==row['native_after'],'Original native incarnation/cache activity changed during refusal')
  require(all(type(t)in(int,float)and math.isfinite(t)for t in(row['started_epoch'],row['finished_epoch']))and row['started_epoch']<=event['started_epoch']<=event['finished_epoch']<=row['finished_epoch'],'Actual request/guard chronology changed')
  require(not any(e['kind']in('engine_begin','engine_end')and row['started_epoch']<=e['epoch']<=row['finished_epoch'] for e in native_events),'Actual native work during loaded refusal')
 return {'all_four_actual_loaded_metadata_changes_refused':True,'live_authoritative_rebind_refused':True,'actual_other_model_weights_loaded':False,'new_loaded_incarnation_qualified':False}

def census_at(events,epoch):
 """Actual source wall-epoch prefix, never a caller-saved counter authority."""
 selected=[e for e in events if e['kind']in('engine_begin','engine_end')and e['epoch']<epoch]
 begins=[e for e in selected if e['kind']=='engine_begin'];ends=[e for e in selected if e['kind']=='engine_end'];owners={(r['engine_pid'],r['engine_generation'])for r in selected}
 require(begins and len(owners)==1 and len(begins)==len(ends),'Complete original primed native event prefix required');pid,generation=next(iter(owners))
 return {'native_pid':pid,'native_generation':generation,'native_begin_count':len(begins),'native_terminal_count':len(ends)}
