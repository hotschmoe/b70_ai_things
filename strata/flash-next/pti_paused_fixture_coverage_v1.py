"""Loaded fixture criteria; observed IDs, not inferred from SYCL graph handles."""
import math
from collections import Counter

def require(ok,msg):
    if not ok:raise ValueError(msg)
def child_coverage(instances,gpu_events,retention):
    require(type(retention)is bool,'Exact declared retention mode required');ids={};gpu={}
    for row in instances:
        require(type(row['instance_id'])is int and row['instance_id']>0 and row['instance_id']not in ids,'Unique actual normalized instance required');ids[row['instance_id']]=row
    for event in gpu_events:
        token=event['args']['id'];require(type(token)is str and token.isdigit()and int(token)not in gpu,'Unique actual GPU trace ID required');gpu[int(token)]=event
    require(set(ids)==set(gpu),'Original complete normalization/Chrome ID association required')
    counts=Counter();selected=[]
    for identifier,event in gpu.items():
        name=event['name']
        for kernel in('PausedRegularFixtureA','PausedRegularFixtureB'):
            if kernel in name:counts[kernel]+=1;selected.append(identifier)
    if retention:require(counts==Counter(PausedRegularFixtureA=1,PausedRegularFixtureB=1),'Actual retained selected graph child roster missing/duplicate')
    else:require(not counts,'Stock paused fixture unexpectedly retained children; negative discrimination unproven')
    require(all(type(ids[i]['raw_kernel_start_ticks'])is int and type(ids[i]['raw_kernel_end_ticks'])is int for i in selected),'Actual selected raw kernel times required')
    return dict(retention_enabled=retention,selected_child_instance_ids=sorted(selected),actual_child_counts=dict(counts),per_instance_clock_alignment_qualified=False,full_model_coverage_qualified=False)

def regular_replay_path(events):
    names=Counter(e['name']for e in events if e.get('ph')=='X'and e.get('cat')=='cpu_op')
    require(names['zeCommandListImmediateAppendCommandListsExp']>=1,'Actual same-UR regular-list replay API absent')
    require(not any('Graph'in name for name in names),'Native ZEGraph route does not qualify this regular-list fixture')
    return dict(actual_regular_list_submission_observed=True,native_ZEGraph_route_observed=False,API_occurrences=dict(names))
