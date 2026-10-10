"""Source39 logical accounting plus explicit external physical-memory gaps."""
from full_cache_shared_metadata_v2 import resource_binding,require

def unsigned(value):return type(value)is int and 0<=value<2**64

def host_observation(sample):
 """A caller must independently join original owned PID/cgroup/raw receipts."""
 require(type(sample)is dict and sample['kind']=='owned_host_memory' and type(sample['host_controller_pid'])is int and sample['host_controller_pid']>0 and type(sample['container_id'])is str and sample['container_id'],'Explicit owned host/container memory observation required')
 fields=('cgroup_memory_current_bytes','cgroup_memory_peak_bytes','cgroup_anon_bytes','cgroup_file_bytes','cgroup_shmem_bytes','cgroup_swap_current_bytes')
 require(all(unsigned(sample[k])for k in fields) and sample['cgroup_memory_current_bytes']<=sample['cgroup_memory_peak_bytes'],'Exact current/peak cgroup byte counts required')
 require(sample['cgroup_swap_current_bytes']==0 and sample['whole_system_physical_peak_qualified'] is False and sample['per_expert_device_residency_qualified'] is False,'Cgroup accounting cannot qualify physical/expert residency or hide swap')
 require(type(sample['raw_receipts'])is list and sample['raw_receipts'] and all(type(r)is dict and type(r['sha256'])is str and len(r['sha256'])==64 and type(r['bytes'])is int and r['bytes']>=0 for r in sample['raw_receipts']),'Original raw host-memory receipts required')
 return {'cgroup_current_bytes':sample['cgroup_memory_current_bytes'],'cgroup_sampled_peak_bytes':sample['cgroup_memory_peak_bytes'],'per_expert_device_residency_qualified':False,'whole_system_physical_peak_qualified':False,'original_raw_receipts_still_need_owned_recollection':True}

def account(source_rows,host_samples=None):
 logical=resource_binding(source_rows);experts=[r for r in source_rows if r['kind']=='expert_backing_metadata'];require(experts,'Actual source39 expert capacity observations required')
 for row in experts:
  require(row['actual_device_residency_qualified'] is False and all(unsigned(row[k])for k in ('mapped_arena_metadata_bytes','live_arena_bytes','full_arena_bytes','full_slots','segment_bytes')) and type(row['segmented'])is bool,'Exact source39 expert backing metadata cannot qualify actual device residency')
 host=[host_observation(r) for r in host_samples] if host_samples is not None else []
 return {'actual_logical_source39_accounting':logical,'actual_expert_capacity_observations':experts,'owned_host_accounting_samples':host,'incoming_held_reusable_overlap_counted_by_source39':True,'whole_process_transient_peak_qualified':False,'physical_host_device_expert_residency_qualified':False,'missing_physical_prerequisites':['original owned PID/cgroup raw recollection' if not host else 'device physical allocation/residency observations','actual per-expert device/host residency attribution','whole actor physical and transient peak coverage'],'full_cache_runtime_qualified':False}
