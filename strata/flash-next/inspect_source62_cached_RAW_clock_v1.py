"""Readonly recorded API/device RAW joins; never infer full model or sampling coverage."""
import argparse,hashlib,json,math,re,struct
from decimal import Decimal
from pathlib import Path

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(4<<20),b''):h.update(x)
 return h.hexdigest()
def serial_uncertainty_ns(value):
 # Actual source uses std::to_string(double), six decimal places in us.
 return Decimal(str(math.ulp(float(value))))*500+Decimal('0.0005')
def inspect(directory):
 directory=Path(directory);r=json.loads((directory/'recipe.json').read_bytes());owner=json.loads((directory/'host-clock-owner.json').read_bytes());require(r['engine_generation']==62 and r['PTI_enabled'] is True and r['Chrome_CPU_host_lane_recorded'] is True and r['both_arms_H36_enabled'] is False,'Exact current62 API/device profile required')
 require(owner['passed'] is True and owner['client_time_namespace']==owner['native_time_namespace'],'Actual native/client time namespace must match')
 rows=json.loads((directory/'requests.json').read_bytes());selected=rows[1]['raw'];require(selected['profile_request_rid']==2 and selected['fresh']==0 and selected['pin']==1536,'Exact selected cached RID2 required')
 begin=selected['profile_send_RAW_before_ns'];end=next(x['host_RAW_ns'] for x in selected['profile_RAW_receive_rows'] if x['line'].startswith('DONE '));require(begin<end,'Actual selected RAW request interval required')
 receive_ledger=[json.loads(line) for line in (directory/'host-protocol-RAW.jsonl').read_text(encoding='ascii').splitlines()];eof=[x for x in receive_ledger if x['line'] is None];require(len(eof)==1 and eof[0]['host_RAW_ns']>=end,'Actual owned native EOF RAW point required');terminal=eof[0]['host_RAW_ns']
 path=directory/'unitrace/clock.1.jsonl';require(path.stat().st_size<=512<<20,'Existing clock budget exceeded');bindings={str(path):sha(path)};origins={};instances={};queries=[];bridges=[];epoch=None;seq=0;closed=set()
 with path.open('rb') as f:
  for line in f:
   require(len(line)<=4096 and line.endswith(b'\n'),'Clock row extent incomplete');x=json.loads(line);seq+=1;require(x['pid']==1 and x['sequence']==seq,'Exact native PID/continuous clock sequence required');kind=x['kind']
   if kind=='host_clock_origin':
    require(epoch is None,'Unique exported host clock origin required');epoch=x['epoch_origin_ns']
   elif kind=='device_origin':
    dev=x['device_handle'];require(dev not in origins and 1<=x['kernel_valid_bits']<=64 and x['timer_frequency_hz']>0 and x['timer_mask']==(1<<x['kernel_valid_bits'])-1,'Exact device clock geometry required');origins[dev]=x
   elif kind=='global_timestamp_query_bracket':
    require(x['return_code']==0 and x['returned_pair_present']==1 and x['raw_before_ns']<=x['host_ns']<=x['raw_after_ns'],'Actual L0 host timestamp outside original RAW query bracket')
    if begin<=x['host_ns']<=end:queries.append(x)
   elif kind=='host_clock_bridge':bridges.append(x)
   elif kind=='query_sampling_summary':
    require(x['sample_every']==(1 if x['query_site']<=2 else 512) and x['selected_brackets']==(x['original_query_calls'] if x['query_site']<=2 else (1+(x['original_query_calls']-1)//512 if x['original_query_calls'] else 0)) and x['selected_brackets']==x['completed_brackets'] and x['selected_brackets']+x['suppressed_brackets']==x['original_query_calls'],'Clock query sampling close incomplete');closed.add(x['query_site'])
   elif kind=='instance_normalization' and begin<=x['submit_host_ns']<=end:
    ident=x['instance_id'];dev=x['device_handle'];require(ident not in instances and dev in origins,'Duplicate/foreign selected instance');o=origins[dev];mask=o['timer_mask'];freq=o['timer_frequency_hz'];factor=struct.unpack('<d',struct.pack('<Q',x['ns_per_cycle_ieee754_u64']))[0]
    require(x['timer_frequency_hz']==freq and x['timer_mask']==mask and factor==1e9/freq and x['epoch_origin_ns']==epoch==o['epoch_origin_ns'],'Actual instance/device/host geometry changed')
    a=x['raw_kernel_start_ticks']&mask;b=x['raw_kernel_end_ticks']&mask;s=x['submit_device_ticks']&mask
    require(s<a<=b and b-s<mask//2,'Selected equal/wrapped/ambiguous timer epoch refused; original arithmetic not repaired')
    start=x['submit_host_ns']+int(float(a-s)*factor);stop=start+int(float(b-a)*factor)
    require(start==x['normalized_start_host_ns'] and stop==x['normalized_end_host_ns'] and stop<=terminal,'Exact original normalization/selected completion differs')
    require(len(instances)<500000,'Selected instance analysis bound exceeded');instances[ident]=dict(clock=x,name=None,RAW_start_ns=start,RAW_end_ns=stop)
 require(epoch is not None and instances and closed==set(range(1,7)) and set(x['device_handle'] for x in queries)==set(x['clock']['device_handle'] for x in instances.values()),'Current selected query evidence/clock close required for every observed device')
 require(len({(origins[z['clock']['device_handle']]['uuid_word0_native_u64'],origins[z['clock']['device_handle']]['uuid_word1_native_u64']) for z in instances.values()})==2,'Actual two distinct physical device clock origins required');files=sorted((directory/'unitrace').glob('*.json'));require(files and sum(x.stat().st_size for x in files)<=1<<30,'Bounded original Chrome files required');cpu=[];flows={};joined=set();cpu_keys=set()
 for p in files:
  bindings[str(p)]=sha(p)
  with p.open('rb') as f:
   for line in f:
    if b'"cat"' not in line:continue
    require(len(line)<=1<<20,'Chrome record exceeds bound');x=json.loads(line.strip().rstrip(b','),parse_float=Decimal);cat=x.get('cat','')
    if cat=='gpu_op':
     ident=int(x['args']['id'])
     if ident not in instances:continue
     require(ident not in joined and x['ph']=='X','Duplicate/noncomplete selected GPU record');z=instances[ident];expected=Decimal(z['RAW_start_ns']+epoch)/1000
     require(abs(Decimal(str(x['ts']))-expected)*1000<=serial_uncertainty_ns(x['ts']),'GPU Chrome/clock timestamp differs beyond source serialization bound');require(abs(Decimal(str(x['dur']))*1000-Decimal(z['RAW_end_ns']-z['RAW_start_ns']))<=serial_uncertainty_ns(x['dur']),'GPU Chrome/clock duration differs beyond serialization bound');z['name']=x['name'];joined.add(ident)
    elif cat=='cpu_op' and x.get('ph')=='X':
     start=Decimal(str(x['ts']))*1000-epoch;duration=Decimal(str(x['dur']))*1000
     if start+duration<begin or start>end:continue
     require(duration>=0 and len(cpu)<1000000,'CPU duration/analysis bound invalid');cpu_keys.add((str(x['ts']),x.get('pid'),x.get('tid')));cpu.append(dict(name=x['name'],tid=x.get('tid'),RAW_start_ns=str(start),RAW_duration_ns=str(duration),start_serialization_uncertainty_ns=str(serial_uncertainty_ns(x['ts']))))
    elif cat.startswith('Flow_H2D_'):
     m=re.fullmatch(r'Flow_H2D_([0-9]+)_[0-9]+',cat)
     if m and int(m[1]) in instances:
      ident=int(m[1]);require(int(x['id'])==ident and x['ph'] in('s','t'),'Exact selected flow ID/type required');values=flows.setdefault(ident,{});require(x['ph'] not in values,'Duplicate selected flow endpoint');values[x['ph']]=x
 require(joined==set(instances) and cpu and set(flows)==set(instances),'Complete selected Clock/GPU/CPU-submit flow ID joins required; no inferred missing operations')
 for ident,values in flows.items():
  require(set(values)=={'s','t'},'Both actual CPU submit/GPU start endpoints required');x=values['s'];require((str(x['ts']),x.get('pid'),x.get('tid')) in cpu_keys,'Actual flow source lacks original same-thread CPU API call');z=instances[ident];x=values['t'];require(abs(Decimal(str(x['ts']))*1000-epoch-z['RAW_start_ns'])<=serial_uncertainty_ns(x['ts']),'Actual GPU flow sink/clock start differs')
 lanes={}
 for ident,z in instances.items():
  x=z['clock'];key=(x['device_handle'],x['queue_handle'],x['engine_ordinal'],x['engine_index']);v=lanes.setdefault(key,dict(instances=0,observed_device_duration_ns=0,observed_submit_to_start_ns=0,observed_GPU_tail_after_DONE_ns=0,append_in_selected_request_instances=0,explicit_copy_payload_bytes={},sample_actual_names=[]));v['instances']+=1;v['observed_device_duration_ns']+=z['RAW_end_ns']-z['RAW_start_ns'];v['observed_submit_to_start_ns']+=z['RAW_start_ns']-x['submit_host_ns'];v['observed_GPU_tail_after_DONE_ns']+=max(0,z['RAW_end_ns']-max(end,z['RAW_start_ns']));v['append_in_selected_request_instances']+=int(begin<=x['append_host_ns']<=end)
  hit=re.fullmatch(r'zeCommandListAppendMemoryCopy(?:Region|FromContext)?\(([HDMS]2[HDMS])\)\[([0-9]+)\]',z['name'])
  if hit:v['explicit_copy_payload_bytes'][hit[1]]=v['explicit_copy_payload_bytes'].get(hit[1],0)+int(hit[2])
  if len(v['sample_actual_names'])<8 and z['name'] not in v['sample_actual_names']:v['sample_actual_names'].append(z['name'])
 for p,want in bindings.items():require(sha(Path(p))==want,'Actual consumed raw bytes changed')
 return dict(scope='Selected explicit serial RID2 cached1536/remaining512PP/GEN8; observed L0 API/device and RAW stream points',raw_bindings=bindings,selected_request_RAW_bounds_ns=[begin,end],selected_instance_count=len(instances),selected_CPU_API_spans=cpu,queue_lanes=[dict(device_handle=k[0],queue_handle=k[1],engine_ordinal=k[2],engine_index=k[3],**v) for k,v in lanes.items()],device_origins=list(origins.values()),current_query_bracket_max_ns=max(x['raw_after_ns']-x['raw_before_ns'] for x in queries),API_GPU_common_RAW_clock_source_recollected=True,RAW_MONOTONIC_bridge_samples=bridges,RAW_MONOTONIC_continuous_drift_bound_qualified=False,native_stage_handle_mapping_qualified=False,internal_sampling_preparation_coverage_qualified=False,full_model_operation_coverage_qualified=False,physical_bus_bandwidth_qualified=False,clean_serving_latency_qualified=False,receiver_points=selected['profile_RAW_receive_rows'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--producer-public',type=Path,required=True);p.add_argument('--producer-public-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(sha(a.producer_public)==a.producer_public_sha256,'Exact independently admitted producer proof required');admitted=json.loads(a.producer_public.read_bytes());require(admitted['passed'] is True and admitted['bounded62_cached_window_profile_passed'] is True and Path(admitted['root']).resolve()==a.directory.parent.resolve(),'Current independent producer admission required');require(not a.output.exists(),'New analysis artifact required');a.output.write_text(json.dumps(inspect(a.directory),indent=2)+'\n',encoding='ascii')
