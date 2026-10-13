"""Bounded admitted RAW interval unions; observed gaps never assert GPU idle."""
import argparse,json,hashlib,re,math
from decimal import Decimal
from pathlib import Path

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for row in iter(lambda:f.read(4<<20),b''):h.update(row)
 return h.hexdigest()
def union(rows,lo,hi):
 merged=[]
 for a,b in sorted((max(lo,a),min(hi,b)) for a,b in rows if b>lo and a<hi):
  if b<=a:continue
  if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
  else:merged.append([a,b])
 return merged
def extent(rows):return sum(b-a for a,b in rows)
def intersect(a,b):
 i=j=0;total=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return total
def gaps(merged,lo,hi):
 out=[];last=lo
 for a,b in merged:
  if a>last:out.append([last,a])
  last=max(last,b)
 if last<hi:out.append([last,hi])
 return out
def dist(values):
 values=sorted(values)
 def pct(q):return values[max(0,math.ceil(len(values)*q)-1)]/1e6 if values else None
 return dict(count=len(values),P50_ms=pct(.5),P95_ms=pct(.95),P99_ms=pct(.99),max_ms=pct(1),definition='Nearest rank of individual observed instances; overlapping waits are never summed as wall time')
def summarize(analysis,public,directory,analysis_sha,public_sha):
 bindings={str(analysis):sha(analysis),str(public):sha(public)};require(bindings[str(analysis)]==analysis_sha and bindings[str(public)]==public_sha,'Exact admitted analysis/public hashes required');admitted=json.loads(public.read_bytes());d=json.loads(analysis.read_bytes());require(admitted['passed'] is True and admitted['bounded62_cached_window_profile_passed'] is True,'Exact admitted producer required');require(Path(admitted['root']).resolve()==directory.parent.resolve(),'Actual producer root association differs');require(d['API_GPU_common_RAW_clock_source_recollected'] is True and d['full_model_operation_coverage_qualified'] is False,'Actual partial aligned analysis required')
 require(analysis.stat().st_size<128<<20,'Analysis bound exceeded');lo,hi=d['selected_request_RAW_bounds_ns'];instances={};cpus=[(int(Decimal(x['RAW_start_ns'])),int(Decimal(x['RAW_start_ns'])+Decimal(x['RAW_duration_ns']))) for x in d['selected_CPU_API_spans']];require(len(cpus)<=1000000,'CPU span bound exceeded');origin={x['device_handle']:x for x in d['device_origins']}
 clock=directory/'unitrace/clock.1.jsonl';require(clock.stat().st_size<=512<<20,'Clock cap exceeded');bindings[str(clock)]=sha(clock);require(bindings[str(clock)]==d['raw_bindings'][str(clock)],'Admitted clock bytes changed')
 with clock.open('rb') as f:
  for line in f:
   require(len(line)<=4096,'Clock row bound exceeded');x=json.loads(line)
   if x['kind']=='instance_normalization' and lo<=x['submit_host_ns']<=hi:instances[x['instance_id']]=x
 require(len(instances)==d['selected_instance_count']<=500000,'Exact selected instance census differs');names={};chrome=sorted((directory/'unitrace').glob('*.json'));require(sum(p.stat().st_size for p in chrome)<1<<30,'Chrome extent exceeded')
 for p in chrome:
  bindings[str(p)]=sha(p);require(bindings[str(p)]==d['raw_bindings'][str(p)],'Admitted Chrome bytes changed')
  with p.open('rb') as f:
   for line in f:
    require(len(line)<=1<<20,'Chrome row bound exceeded')
    if b'"cat": "gpu_op"' not in line:continue
    x=json.loads(line.strip().rstrip(b','));ident=int(x['args']['id'])
    if ident in instances:require(ident not in names,'Duplicate selected GPU name');names[ident]=x['name']
 require(set(names)==set(instances),'Actual selected GPU names incomplete');device={};waits={};groups={};transfers={}
 for ident,x in instances.items():
  dev=x['device_handle'];a=x['normalized_start_host_ns'];b=x['normalized_end_host_ns'];device.setdefault(dev,[]).append((a,b));waits.setdefault(dev,[]).append(a-x['submit_host_ns']);name=names[ident];key=re.sub(r'\[SIMD.*$','',name);v=groups.setdefault((dev,key),[0,0]);v[0]+=1;v[1]+=b-a
  hit=re.fullmatch(r'zeCommandListAppendMemoryCopy(?:Region|FromContext)?\(([HDMS]2[HDMS])\)\[([0-9]+)\]',name)
  if hit:
   v=transfers.setdefault((dev,hit[1]),dict(instances=0,explicit_command_payload_bytes=0,intervals=[]));v['instances']+=1;v['explicit_command_payload_bytes']+=int(hit[2]);v['intervals'].append((a,b))
 perdevice={dev:union(rows,lo,hi) for dev,rows in device.items()};gpu=union([x for rows in device.values() for x in rows],lo,hi);cpu=union(cpus,lo,hi);overlap=intersect(gpu,cpu);report_devices=[]
 for dev,rows in perdevice.items():
  missed=sorted(gaps(rows,lo,hi),key=lambda p:p[1]-p[0],reverse=True)[:10]
  report_devices.append(dict(device_handle=dev,physical_UUID_words=[origin[dev]['uuid_word0_native_u64'],origin[dev]['uuid_word1_native_u64']],observed_union_ms=extent(rows)/1e6,observed_union_fraction=extent(rows)/(hi-lo),observed_gap_top10=[dict(relative_start_ms=(a-lo)/1e6,duration_ms=(b-a)/1e6,CPU_API_union_in_gap_ms=extent(union(cpus,a,b))/1e6) for a,b in missed],submit_to_start=dist(waits[dev])))
 points=[(lo,'selected_GEN_send')]+[(x['host_RAW_ns'],x['line']) for x in d['receiver_points']];windows=[]
 for (a,label),(b,nextlabel) in zip(points,points[1:]):
  require(lo<=a<=b<=hi,'Actual RAW protocol windows invalid');u=union([x for rows in device.values() for x in rows],a,b);v=union(cpus,a,b)
  windows.append(dict(from_observation=label,to_observation=nextlabel,relative_start_ms=(a-lo)/1e6,wall_ms=(b-a)/1e6,observed_any_device_union_ms=extent(u)/1e6,observed_API_union_ms=extent(v)/1e6,observed_both_union_ms=intersect(u,v)/1e6,perdevice_union_ms={str(dev):extent(union(rows,a,b))/1e6 for dev,rows in device.items()}))
 for p,want in bindings.items():require(sha(Path(p))==want,'Consumed admitted bytes changed')
 return dict(scope='Actual cached1536/new512 GEN8 selected request, partial aligned observations',input_sha256=bindings,selected_wall_ms=(hi-lo)/1e6,observed_any_device_union_ms=extent(gpu)/1e6,observed_API_union_ms=extent(cpu)/1e6,observed_both_union_ms=overlap/1e6,neither_recorded_union_ms=((hi-lo)-extent(gpu)-extent(cpu)+overlap)/1e6,devices=report_devices,protocol_windows=windows,largest_device_work_sums=[dict(device_handle=k[0],name=k[1],instances=v[0],observed_duration_sum_ms=v[1]/1e6) for k,v in sorted(groups.items(),key=lambda kv:kv[1][1],reverse=True)[:24]],explicit_copy_commands=[dict(device_handle=k[0],direction=k[1],instances=v['instances'],explicit_command_payload_bytes=v['explicit_command_payload_bytes'],observed_union_ms=extent(union(v['intervals'],lo,hi))/1e6) for k,v in transfers.items()],critical_path_qualified=False,physical_GPU_idle_qualified=False,physical_bus_bandwidth_qualified=False,clean_latency_qualified=False,internal_sampling_preparation_stage_coverage_qualified=False,limitations='Union coverage is wall time of observed spans. Work sums may overlap. Unobserved gaps include untraced work and are not idle. CPU timestamp serialization uncertainty is source-derived <= original analysis per-row bound. Host API calls are not complete native preparation/sampling. DMA payload is requested bytes, not physical bus traffic.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--public',type=Path,required=True);p.add_argument('--analysis-sha256',required=True);p.add_argument('--public-sha256',required=True);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New output required');a.output.write_text(json.dumps(summarize(a.analysis,a.public,a.directory,a.analysis_sha256,a.public_sha256),indent=2)+'\n',encoding='ascii')
