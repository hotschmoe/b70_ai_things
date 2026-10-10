"""Actual source-owned logical extents, cache capacities and frees; no physical claims."""
import json
from audit_slot_owner_trace_v1 import audit as slot_audit
from audit_stage_mirror_owner_trace_v1 import audit as mirror_audit

def require(ok,msg):
 if not ok:raise ValueError(msg)
def peaks(items):
 events=[]
 for a in items:
  begin=a.get('alloc_line',a.get('line'));end=a['free_line'];require(type(begin) is int and type(end) is int and begin<end and a['bytes']>0,'Actual owning allocation and successful free chronology required');events.extend([(begin,a['bytes']),(end,-a['bytes'])])
 live=maximum=0
 for line,delta in sorted(events):live+=delta;require(live>=0,'Logical live extent underflow');maximum=max(maximum,live)
 require(live==0,'Every owned logical allocation mustbe retired/freed')
 return maximum

def account(trace,stages,slots):
 private=slot_audit(trace,[(stage,slot) for stage,lo,hi in stages for slot in range(slots)]);mirrors=mirror_audit(trace,[(stage,stage,lo,hi) for stage,lo,hi in stages]);require(private['passed'] is True and mirrors['passed'] is True,'Actual owned slots/mirrors/context-matched logicalfree proof required')
 categories={}
 for owner in private['owners']:
  for a in owner['allocations']:categories.setdefault('slot_state:'+a['role'],[]).append(a)
 for owner in mirrors['owners']:
  for a in owner['allocations']:categories.setdefault('expert_mirror:'+a['role'],[]).append(a)
 logical={name:{'actual_registered_allocation_bytes':sum(a['bytes'] for a in items),'actual_peak_live_owned_extent_bytes':peaks(items),'allocations':len(items)} for name,items in categories.items()}
 cache=[]
 for line in trace.splitlines():
  if not line.startswith('PCL '):continue
  row=json.loads(line[4:])
  if row['event']!='cache':continue
  require(row['identity_observed'] is True and row['snapshot_instance']>0 and row['snapshot_bytes']>=0 and row['retained_bytes']>=0 and row['retained_bytes']<=row['budget'] and row['entries']<=row['slots'] and row['stage_descriptor_truncated'] is False,'Actual cache identity/capacity/stage bounds invalid');cache.append(row)
 return {'owned_logical_categories':logical,'cache_actual_events':cache,'actual_peak_recorded_parked_capacity_bytes':max((r['retained_bytes'] for r in cache),default=0),'physical_RSS_device_residency_or_reclamation_qualified':False,'expert_layer_residency_qualified':False,'PCIe_traffic_inferred':False,'scope':'actual registered SLOT/MIRROR owned extents and matching UR frees, plus exact PCL capacity events; logical allocation/capacity only'}
