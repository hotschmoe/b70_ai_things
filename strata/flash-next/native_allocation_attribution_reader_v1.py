"""Bounded future original producer logs; no native runtime authority from CPU."""
import json,re
from pathlib import Path
from native_allocation_attribution_contract_v1 import analyze,require

def unique(pairs):
 out={}
 for k,v in pairs:
  require(k not in out,'Duplicate serialized allocation key');out[k]=v
 return out

def recollect(path,owners,source_generation):
 """Owners/expert lists are preregistered from genuine future startup metadata."""
 require(type(owners)is list and 2<=len(owners)<=3,'Actual host plus one/two device owner roster required')
 require(type(source_generation)is int and 41<=source_generation<2**32,'Actual assigned future source generation required')
 keys={};groups={}
 for item in owners:
  owner=item['owner'];key=json.dumps(owner,sort_keys=True,separators=(',',':'));require(key not in keys,'Duplicate actual owner scope');keys[key]=item;groups[key]=[]
 require(sum(item['owner']['device_uuid'] is None for item in owners)==1,'Exactly one process-owned host heap domain required')
 count=0
 with Path(path).open('rb') as stream:
  for line in stream:
   if line.startswith(b'ALLOC_ATTR_ERROR'):raise ValueError('Actual producer attribution failure')
   if not line.startswith(b'ALLOC_ATTR '):continue
   require(len(line)<=16384 and line.endswith(b'\n'),'Bounded complete original producer JSON line required');count+=1;require(count<=196608,'Bounded all-owner original event roster')
   row=json.loads(line[len(b'ALLOC_ATTR '):],object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')));key=json.dumps(row['owner'],sort_keys=True,separators=(',',':'));require(key in groups,'Foreign process/device/source owner');groups[key].append(row)
 require(all(groups.values()),'Missing actual preregistered owner event scope')
 reports=[]
 for key,item in keys.items():
  rows=groups[key];require({r['role'] for r in rows}==set(item['expected_roles']),'Exact preregistered successful API role coverage required')
  require(all(type(r['stage'])is int and r['stage'] in item['expected_stages'] for r in rows),'Foreign actual stage placement')
  reports.append(analyze(rows,item['owner'],item['expected_experts'],source_generation))
 return {'scoped_owner_reports':reports,'actual_complete_declared_scoped_API_events_recollected':True,'physical_residency_qualified':False,'whole_process_transient_peak_qualified':False,'whole_host_cache_metadata_API_coverage_qualified':False,'per_expert_driver_residency_qualified':False,'native_producer_runtime_qualified':False,'full_cache_goal_qualified':False}
