"""Read-only observed DMA payload/raw-tick overlap; never host/device calibration."""
import argparse,collections,hashlib,json,re,struct
from pathlib import Path
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def union(items):
 out=[]
 for a,b in sorted(items):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
  else:out.append((a,b))
 return out
def intersect(a,b):
 i=j=total=0
 while i<len(a)and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return total
def inspect(directory):
 directory=Path(directory).resolve();recipe=json.loads((directory/'recipe.json').read_bytes())
 require(recipe['engine_generation']==52 and recipe['PTI_enabled'] and len(recipe['requests'])==2 and all(len(x['ids'])==2048 and x['max_new']in(8,16)for x in recipe['requests']),'Exact bounded current52 recorder required')
 controls=[json.loads((directory/(x+'.json')).read_bytes())for x in('temporal-before-selected','temporal-after-selected')]
 require(all(x['passed'] and x['target']['pid']==1 for x in controls),'Owned successful resume/selected-close controls required')
 lo=controls[0]['clock_after']['last_sequence'];hi=controls[1]['clock_before']['last_sequence'];require(0<lo<hi,'Actual complete selected sequence bounds required')
 clock=directory/'unitrace/clock.1.jsonl';require(clock.stat().st_size<=512<<20,'Existing clock budget exceeded')
 before=sha(clock);devices={};instances={};seq=0;bridges=[];queries=[];selected_queries=[]
 with clock.open('rb')as stream:
  for raw in stream:
   require(raw.endswith(b'\n') and len(raw)<=4096,'Partial/oversized actual clock record')
   row=json.loads(raw);require(row['pid']==1 and row['sequence']==seq+1,'Actual PID/continuous clock sequence differs');seq+=1;k=row['kind']
   if k=='device_origin':
    dev=row['device_handle'];require(dev not in devices and row['timer_frequency_hz']>0 and row['timer_mask']==(1<<row['kernel_valid_bits'])-1,'Device clock origin differs')
    factor=struct.unpack('<d',struct.pack('<Q',row['ns_per_cycle_ieee754_u64']))[0];require(factor==1e9/row['timer_frequency_hz'],'Device cycle conversion differs');devices[dev]=row
   elif k=='global_timestamp_query_bracket':
    require(row['raw_before_ns']<=row['raw_after_ns'],'Query bracket inverted')
    require(row['return_code']==0 and row['returned_pair_present']==1 and row['raw_before_ns']<=row['host_ns']<=row['raw_after_ns'],'Returned clock query outside RAW bracket')
    queries.append(row['raw_after_ns']-row['raw_before_ns'])
    if lo<seq<=hi:selected_queries.append(row['device_handle'])
   elif k=='host_clock_bridge':
    require(row['raw_before_ns']<=row['raw_after_ns'],'Host bridge bracket inverted');bridges.append(row['monotonic_ns']-(row['raw_before_ns']+row['raw_after_ns'])//2)
   elif k=='instance_normalization' and lo<seq<=hi:
    ident=row['instance_id'];dev=row['device_handle'];require(ident not in instances and dev in devices,'Instance/device association differs')
    origin=devices[dev];require(row['timer_frequency_hz']==origin['timer_frequency_hz'] and row['timer_mask']==origin['timer_mask'],'Per-device timer geometry changed')
    a,b=row['raw_kernel_start_ticks'],row['raw_kernel_end_ticks'];require(0<=a<=b<=row['timer_mask'],'Wrap/unordered raw interval; no inferred unwrap')
    instances[ident]=[dev,a,b,row['engine_ordinal'],row['engine_index'],None]
 require(seq>=hi and instances and len(devices)==2 and set(selected_queries)==set(devices),'Two current device origins/query evidence/selected instances required')
 files=sorted((directory/'unitrace').glob('*.json'));require(files and sum(p.stat().st_size for p in files)<1<<30,'Bounded original Chrome outputs required');bindings={str(clock):before};joined=set()
 for p in files:
  bindings[str(p)]=sha(p)
  with p.open('rb')as stream:
   for raw in stream:
    if b'"cat": "gpu_op"'not in raw:continue
    require(len(raw)<=1<<20,'Chrome GPU row too large');event=json.loads(raw.strip().rstrip(b','));ident=int(event['args']['id'])
    if ident in instances:
     require(ident not in joined and event['ph']=='X','Duplicate/noncomplete observed GPU instance');joined.add(ident);instances[ident][-1]=event['name']
 require(joined==set(instances),'Missing actual selected Clock/Chrome instance joins')
 output=[]
 for dev,origin in devices.items():
  copy=[];compute=[];payload=collections.Counter();engines=collections.Counter();all_ticks=[]
  for d,a,b,ordinal,index,name in instances.values():
   if d!=dev:continue
   all_ticks.extend((a,b));engines[(ordinal,index)]+=1
   hit=re.fullmatch(r'zeCommandListAppendMemoryCopy(?:Region|FromContext)?\(([HDMS]2[HDMS])\)\[(\d+)\]',name)
   if hit:copy.append((a,b));payload[hit[1]]+=int(hit[2])
   elif not name.startswith('zeCommandListAppend'):compute.append((a,b))
  require(max(all_ticks)-min(all_ticks)<=origin['timer_mask']//2,'Raw linear epoch ambiguous')
  cp,co=union(copy),union(compute);factor=1e9/origin['timer_frequency_hz']
  output.append({'device_handle':dev,'descriptor_id':origin['descriptor_id'],'uuid_words':[origin['uuid_word0_native_u64'],origin['uuid_word1_native_u64']],'explicit_copy_payload_bytes_by_allocation_direction':dict(payload),'observed_copy_union_ms':sum(b-a for a,b in cp)*factor/1e6,'observed_compute_union_ms':sum(b-a for a,b in co)*factor/1e6,'observed_copy_compute_overlap_ms':intersect(cp,co)*factor/1e6,'engine_ordinal_index_counts':{str(k):v for k,v in engines.items()}})
 for name,want in bindings.items():require(sha(Path(name))==want,'Consumed raw source changed')
 return {'scope':'actual selected-emission sequence slice; observed commands only','raw_bindings':bindings,'selected_sequence_bounds':[lo,hi],'clock_chrome_instances':len(instances),'device_observations':output,'query_bracket_max_ns':max(queries),'observed_RAW_to_MONOTONIC_offset_range_ns':max(bridges)-min(bridges)if bridges else None,'host_device_clock_alignment_qualified':False,'continuous_clock_drift_bound_qualified':False,'full_model_operation_coverage_qualified':False,'physical_bus_bandwidth_qualified':False,'mapped_RAM_kernel_traffic_measured':False,'clean_serving_speed_qualified':False}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New read-only analysis output required');value=inspect(a.directory)
 with a.output.open('x',encoding='ascii')as f:json.dump(value,f,indent=2);f.write('\n')
