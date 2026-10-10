"""Unintegrated allocation-accounting producer successor contract; API bytes are not residency."""
import math,json

def require(ok,message):
 if not ok:raise ValueError(message)

SPACES=('device_usm','shared_usm','host_usm','host_heap')

def integer(value):return type(value)is int and 0<=value<2**64

def analyze(rows,owner,expected_experts,source_generation):
 """Replay bounded future allocator events, never infer missing physical data."""
 require(type(owner)is dict and set(owner)=={'engine_pid','process_start_ticks','device_uuid','source_generation'} and type(owner['engine_pid'])is int and owner['engine_pid']>0 and integer(owner['process_start_ticks']) and ((type(owner['device_uuid'])is str and owner['device_uuid']) or owner['device_uuid'] is None) and type(source_generation)is int and 41<=source_generation<2**32 and type(owner['source_generation'])is int and owner['source_generation']==source_generation,'Exact future source/process/device owner required')
 require(type(rows)is list and rows and len(rows)<=65536,'Complete bounded original allocation event roster required')
 require(type(expected_experts)is list and ((owner['device_uuid'] is None and not expected_experts) or (owner['device_uuid'] is not None and expected_experts)) and all(type(p)is list and len(p)==2 and type(p[0])is int and 0<=p[0]<48 and type(p[1])is int and 0<=p[1]<512 for p in expected_experts) and len({tuple(p) for p in expected_experts})==len(expected_experts),'Exact nonempty preregistered expert placement roster required')
 expected={tuple(p) for p in expected_experts};observed=set();live={};allocated=set();experts={};high={s:0 for s in SPACES};totals={s:0 for s in SPACES};previous=0;epoch=None
 for row in rows:
  require(type(row)is dict and json.dumps(row['owner'],sort_keys=True,separators=(',',':'))==json.dumps(owner,sort_keys=True,separators=(',',':')) and type(row['sequence'])is int and row['sequence']==previous+1,'Actual allocation owner/sequence missing or changed');previous=row['sequence']
  require(type(row['host_epoch']) in (int,float) and math.isfinite(row['host_epoch']) and (epoch is None or row['host_epoch']>=epoch),'Monotonic finite original host observation epoch required');epoch=row['host_epoch']
  require(type(row['command_context'])is dict and set(row['command_context'])=={'current_main_body_rid','actual_slot_rid','phase'} and all(integer(row['command_context'][k]) for k in ('current_main_body_rid','actual_slot_rid')) and type(row['command_context']['phase'])is str,'Main-body context and actual slot ownership must remain separate')
  require(type(row['stage'])is int and (row['stage']==-1 if owner['device_uuid'] is None else 0<=row['stage']<=1) and type(row['owner_generation'])is int and row['owner_generation']>0 and type(row['queue_context'])is str and row['queue_context'] and type(row['role'])is str and row['role'] and type(row['actual_slot_context_observed'])is bool and row['physical_residency_observed'] is False,'Exact observed API owner/stage/generation/scope required')
  require(integer(row['bytes']) and row['bytes']>0 and integer(row['offset']) and type(row['copy_completion_observed'])is bool and type(row['existing_completion_boundary'])is bool,'Exact typed byte/range/completion fields required')
  kind=row['kind'];allocation=row['allocation_id'];require(type(allocation)is str and allocation,'Unique process-lifetime allocation ID required')
  if kind=='allocation_success':
   require(allocation not in allocated and row['space'] in SPACES and integer(row['bytes']) and row['bytes']>0 and row['api_result']=='success','Successful actual allocator API receipt required')
   require((row['space']=='host_heap')==(owner['device_uuid'] is None),'Host heap and actual device UUID domains must remain distinct')
   require(row['owner_generation'] not in {v['identity'][1] for v in live.values()},'No concurrent allocation owner generation borrowing')
   allocated.add(allocation);live[allocation]={'space':row['space'],'bytes':row['bytes'],'ranges':{},'identity':(row['stage'],row['owner_generation'],row['queue_context'],row['role'])};totals[row['space']]+=row['bytes'];high[row['space']]=max(high[row['space']],totals[row['space']])
  elif kind=='expert_bytes_ready':
   require(allocation in live and (row['stage'],row['owner_generation'],row['queue_context'],row['role'])==live[allocation]['identity'],'Exact immutable allocation range owner required')
   require(row['space']==live[allocation]['space'],'Ready byte space contradicts original allocation')
   require(row['completion_boundary'] in ('host_source_workers_joined','blocking_copy_wait_returned','SYCL_queued_copy_existing_wait_returned'),'Exact existing source completion boundary required')
   require(allocation in live and type(row['layer'])is int and 0<=row['layer']<48 and type(row['expert'])is int and 0<=row['expert']<512 and integer(row['offset']) and integer(row['bytes']) and row['bytes']>0,'Actual live expert byte placement required')
   item=live[allocation];require(item['space'] in ('device_usm','shared_usm','host_usm'),'Expert ready bytes require direct actual backed-USM allocation; virtual/physical-handle mapping lineage is not implemented');require(row['offset']+row['bytes']<=item['bytes'],'Expert range outside actual arena allocation');key=(row['layer'],row['expert']);require(key not in experts,'Expert already assigned; explicit release required')
   require(all(row['offset']+row['bytes']<=offset or offset+size<=row['offset'] for offset,size in item['ranges'].values()),'Overlapping expert slot bytes cannot be independently counted')
   require(row['copy_completion_observed'] is True and row['existing_completion_boundary'] is True,'Submitted copy or added diagnostic barrier cannot prove ready bytes')
   require(key in expected,'Unexpected expert placement outside declared roster');observed.add(key);experts[key]=allocation;item['ranges'][key]=(row['offset'],row['bytes'])
  elif kind=='expert_bytes_released':
   require(allocation in live and (row['stage'],row['owner_generation'],row['queue_context'],row['role'])==live[allocation]['identity'] and row['space']==live[allocation]['space'] and row['completion_boundary'] in ('owning_free_returned','overwrite_after_existing_wait'),'Exact actual range retirement owner/boundary required')
   require(type(row['layer'])is int and 0<=row['layer']<48 and type(row['expert'])is int and 0<=row['expert']<512,'Exact integer released expert owner required');key=(row['layer'],row['expert']);require(allocation in live and live[allocation]['ranges'].get(key)==(row['offset'],row['bytes']),'Released range must equal actual registered extent');require(experts.get(key)==allocation and allocation in live,'Exact expert/arena release ownership required');del experts[key];del live[allocation]['ranges'][key]
  elif kind=='allocation_release_success':
   require(allocation in live and (row['stage'],row['owner_generation'],row['queue_context'],row['role'])==live[allocation]['identity'],'Exact returned free owner required')
   require(row['space']==live[allocation]['space'] and row['bytes']==live[allocation]['bytes'],'Returned free must match original successful API allocation')
   require(allocation in live and not live[allocation]['ranges'] and row['api_result']=='success','Actual release must follow owned expert range retirement');item=live.pop(allocation);totals[item['space']]-=item['bytes']
  else:raise ValueError('Unknown allocation/residency event cannot be admitted')
 require(observed==expected,'Complete actual declared expert placement/copy coverage missing')
 require(not live and not experts and all(v==0 for v in totals.values()),'All original owners must retire before final accounting')
 return {'complete_declared_API_allocation_events':True,'peak_live_API_bytes_by_space':high,'expert_byte_placement_and_copy_completion_observed':bool(expected_experts),'actual_host_cache_payload_API_bytes_observed':owner['device_uuid'] is None,'whole_host_cache_metadata_allocations_qualified':False,'graph_consumer_retirement_independently_observed':False,'physical_residency_or_eviction_qualified':False,'per_expert_physical_residency_qualified':False,'whole_actor_transient_peak_qualified':False,'normal_latency_route_qualified':False,'current_native_producer_or_runtime_qualified':False,'full_cache_runtime_qualified':False}
