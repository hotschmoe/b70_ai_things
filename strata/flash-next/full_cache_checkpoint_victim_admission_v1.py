"""Known source policy + actual BEFORE/AFTER victim witness; never invent a log."""
def require(ok,msg):
 if not ok:raise ValueError(msg)
def source_victim(points,cap):
 n=len(points)
 if cap<2 or n<2:return 0
 newest=max(range(n),key=lambda i:points[i]['used'])
 tail=next((i for i in range(1,n) if points[i]['tail'] and not points[i]['pinned'] and i!=newest),None)
 if tail is not None:return tail
 available=[i for i in range(1,n) if not points[i]['pinned']]
 return min(available or list(range(1,n)),key=lambda i:points[i]['used'])

def actual_eviction(before,after):
 require(before['phase']=='before_eviction' and after['phase']=='after_eviction' and all(before[k]==after[k] for k in ('actor','slot','rid','generation','cap')),'Exact actual source actor/owner/cap eviction pair required')
 points=before['points'];require(len(points)>before['cap'] and len(points)>=2 and before['actual_victim']==source_victim(points,before['cap']),'Actual emitted victim differs from independent frozen source policy')
 victim=before['actual_victim'];expected=[dict(p,index=i) for i,p in enumerate(points[:victim]+points[victim+1:])];require(after['points']==expected and after['actual_victim']==-1 and after['chain_capacity_bytes']==before['chain_capacity_bytes']-points[victim]['point_capacity_bytes'],'Actual post-erasure identities/capacity do not match exact victim')
 return {'actual_victim':points[victim],'exact_source_checkpoint_victim_observed':True,'physical_allocator_reclamation_qualified':False,'cache_math_qualified':False,'source35_existing_uninstrumented_runtime_proof':False}
