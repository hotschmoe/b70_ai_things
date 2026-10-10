"""Complete actor replication with bounded source39 selected actual RIDs.
No cache state crosses an actor boundary. RIDs follow the pinned frontend's
zero-based counter increment once per request; concurrent row order is unknown.
"""
RAW49_BYTES=(48*10240+248320)*4
BYTE_LIMIT=64<<20
RID_LIMIT=6

def require(ok,message):
    if not ok:raise ValueError(message)

def actor_passes(phases):
    require(phases and phases[0]['name']=='warm','Each independent actor recreates its warm and complete state family')
    cursor=0;logical=[]
    for phase in phases:
        count=len(phase['rows']);rids=list(range(cursor+1,cursor+count+1));cursor+=count
        # Whole concurrent phases are selected together: assigning individual
        # row indices to request IDs would assume an observed scheduling order.
        if phase['name']!='warm':
            maximum=1 if max(phase['max_new'])==1 else 4
            logical.append({'phase':phase['name'],'rids':rids,
                            'maximum_complete49_groups_per_RID':maximum})
    passes=[];chosen=[];groups=0
    def seal():
        if chosen:
            passes.append({'capture_pass':len(passes),'selected_phases':[r['phase']for r in chosen],
                           'selected_actual_RIDs':[rid for r in chosen for rid in r['rids']],
                           'maximum_complete49_groups':groups,
                           'maximum_raw_bytes':groups*RAW49_BYTES,
                           'entire_actor_phase_roster':[r['name']for r in phases],
                           'actual_total_HTTP_requests':cursor,
                           'all_request_metadata_and_protocol_required':True,
                           'fresh_engine_own_state_required':True,
                           'cached_state_transferred_between_actors':False,
                           'actual_execution_observed':False})
    for item in logical:
        extra=len(item['rids'])*item['maximum_complete49_groups_per_RID']
        require(len(item['rids'])<=RID_LIMIT and extra*RAW49_BYTES<=BYTE_LIMIT,'One intact phase exceeds immutable observer bound')
        if chosen and(sum(len(r['rids'])for r in chosen)+len(item['rids'])>RID_LIMIT or(groups+extra)*RAW49_BYTES>BYTE_LIMIT):
            seal();chosen=[];groups=0
        chosen.append(item);groups+=extra
    seal()
    expected=[rid for item in logical for rid in item['rids']]
    actual=[rid for p in passes for rid in p['selected_actual_RIDs']]
    require(actual==expected and len(actual)==len(set(actual)),'Complete selected request coverage changed')
    return {'passes':passes,'armed_requests_required':len(expected),'total_HTTP_requests':cursor,
            'counter_source':'fresh StrataEngine _request_identity_counter=0; increment once in _begin_request_identity',
            'concurrent_row_to_RID_order_inferred':False,'per_actor_RAW_byte_limit':BYTE_LIMIT,
            'per_actor_selected_RID_limit':RID_LIMIT,'full_cache_runtime_qualified':False}

def observe_phase_rids(phase,work):
    require(len(work)==len(phase['rows']),'Actual full phase request roster missing')
    require(sorted(row['rid']for row in work)==phase['expected_actual_RID_set'],'Actual source request counter/phase set changed')
    require(len({row['rid']for row in work})==len(work),'Duplicate actual request RID')
    return True
