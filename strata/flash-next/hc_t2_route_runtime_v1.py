"""Actual host graph identity/count joins, no inferred physical kernel/event identity."""
import re
import run_original_mtp53_two_stage_v5 as base
from pathlib import Path
BASE_PLAN_SHA='1144f037b1c03cc4840369af01757a899a3edddf87d584b1c53b98d6b6eb3df5'
def reference(path,expected):
 path=Path(path)
 if base.sha(path)!=expected:raise ValueError('Exact actual closed original V6 history report required')
 value=base.read(path)
 if value.get('passed')is not True or value.get('errors')!=[]or value.get('source_plan_sha256')!=BASE_PLAN_SHA or value.get('selected_case_indices')!=[0]or value.get('repeats_per_arm')!=1:raise ValueError('Actual successful V6 first-case target required')
 for name,want in value['artifacts'].items():
  if base.sha(path.parent/name)!=want:raise ValueError('Historical target byte changed '+name)
 arms=value['arm_results']
 if len(arms)!=2 or any(r.get('passed')is not True or r.get('owned_removed')is not True for r in arms):raise ValueError('Actual original V6 OFF/ON normal retirement required')
 ids=arms[0]['actual']['ids']
 if len(ids)!=512 or arms[1]['actual']['ids']!=ids or any(r['actual']['natural_eos']for r in arms):raise ValueError('Actual same full512 target histories required')
 return {'report_path':str(path),'report_sha256':expected,'ids':ids,'current_model_identity_or_health_authority':False,'historical_comparison_target_only':True}

def counts(text,actual,mode,hc_mode):
 captures=base.records(text,'HC_T2_ROUTE_CAPTURE');closed=base.records(text,'HC_T2_ROUTE_CLOSE');final=base.records(text,'HC_T2_ROUTE_FINAL')
 if len(final)!=1 or final[0]!={'pid':'1','failed':'0','GPU_queries_added':'0','GPU_waits_added':'0'}or not captures or len(captures)>32 or len(captures)!=len(closed):raise ValueError('Actual complete route graph closure/namespacePID1 required')
 key=lambda r:(r['owner'],r['graph']);cap={key(r):r for r in captures};end={key(r):r for r in closed}
 if len(cap)!=len(captures)or len(end)!=len(closed)or cap.keys()!=end.keys():raise ValueError('Original captured graph identity roster differs')
 if mode=='controls':expect=3;first=int(actual['restored'][0]['base_position']);last=int(actual['restored'][4]['base_position'])
 else:
  windows=[r for r in actual['windows']if int(r['remaining'])>=2];expect=len(windows)
  if not expect:raise ValueError('Actual MTP targetT2 windows required')
  first=int(windows[0]['position'])-int(windows[0]['committed']);last=int(windows[-1]['position'])-int(windows[-1]['committed'])
 totals={0:0,1:0};positions={0:[],1:[]}
 for k,c in cap.items():
  r=end[k]
  if any(not re.fullmatch(r'0x[0-9a-f]+',c[name])for name in('owner','graph'))or c['graph_kind']!='SYCL_host_identity'or c['pid']!='1':raise ValueError('Actual native owned host graph IDs required')
  stage=int(c['device']);lo,hi=int(c['lb']),int(c['le'])
  if stage not in(0,1)or(lo,hi)!=(24*stage,24*(stage+1))or int(c['enabled'])!=hc_mode or int(c['eligible'])!=48 or int(c['selected'])!=48*hc_mode:raise ValueError('Target stage HC T2 eligibility/selection differs')
  if any(r[name]!=c[name]for name in c):raise ValueError('Captured/closed immutable graph descriptor differs')
  submitted=int(r['submitted']);completed=int(r['completed'])
  if submitted<0 or submitted!=completed:raise ValueError('Actual submitted graphs not completed')
  totals[stage]+=submitted
  if submitted:positions[stage].extend([int(r['first_pos']),int(r['last_pos'])])
 if totals!={0:expect,1:expect}or any(not p or min(p)!=first or max(p)!=last for p in positions.values()):raise ValueError('Actual T2 stage graph submissions do not match MTP windows')
 return {'actual_T2_windows_per_stage':totals,'recorded_eligible_half_calls_per_graph':48,'HC_reuse_enabled':bool(hc_mode),'actual_selected_graphs_completed':True,'physical_kernel_count_qualified':False}
