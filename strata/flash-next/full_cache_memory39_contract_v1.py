"""Pure CPU arithmetic/recollection for logical source39 samples only."""
FIELDS=('budget_bytes','parked_bytes','reusable_bytes','held_snapshot_bytes',
        'incoming_estimate_bytes','local_reuse_bytes','allocated_snapshot_bytes')
def logical_sample(**values):
    if set(values)!=set(FIELDS):raise ValueError('Exact logical scope fields required')
    if any(type(v)is not int or not 0<=v<2**64 for v in values.values()):
        raise ValueError('Actual uint64 logical bytes required')
    def add(a,b):
        result=a+b
        if result>=2**64:raise ValueError('Logical byte overflow')
        return result
    cache=add(values['parked_bytes'],values['reusable_bytes'])
    pending=add(values['local_reuse_bytes'],values['allocated_snapshot_bytes'])
    base=add(cache,values['held_snapshot_bytes'])
    allocated=add(base,pending)
    reserved=add(base,max(values['incoming_estimate_bytes'],pending))
    return dict(values,cache_total_bytes=cache,observed_owned_snapshot_bytes=allocated,
                logical_reservation_bytes=reserved,reservation_within_budget=reserved<=values['budget_bytes'])
def recollect(samples):
    peaks={}
    for row in samples:
        if row['kind']!='logical_memory':raise ValueError('Logical memory record required')
        computed=logical_sample(**{k:row[k]for k in FIELDS})
        for k,v in computed.items():
            if type(row[k])is not type(v)or row[k]!=v:raise ValueError('Actual logical sample arithmetic changed '+k)
        key=(row['pid'],row['cache_scope'])
        previous=peaks.get(key,(0,0,0))
        if type(row['event'])is not int or row['event']<=previous[2]:raise ValueError('Scope chronology changed')
        allocated=max(previous[0],computed['observed_owned_snapshot_bytes'])
        reserved=max(previous[1],computed['logical_reservation_bytes'])
        if row['observed_owned_snapshot_peak_bytes']!=allocated or row['logical_reservation_peak_bytes']!=reserved:
            raise ValueError('Sampled logical peak changed')
        for k in ('whole_process_peak_qualified','physical_reclamation_qualified','device_physical_memory_qualified'):
            if row[k]is not False:raise ValueError('Logical samples cannot qualify physical peak')
        if row['incoming_estimate_overlaps_owned_pending']is not True:raise ValueError('Pending/estimate overlap changed')
        peaks[key]=(allocated,reserved,row['event'])
    if not peaks:raise ValueError('No actual logical samples')
    return {'sampled_scope_peaks':[(p,c,a,r)for(p,c),(a,r,e)in peaks.items()],
            'whole_process_peak_qualified':False,'actual_full_cache_qualified':False}

def batch_terminal_binding(rows, rid, slotgen, slot):
    """One actual continued BGEN leg, no HTTP or cache-math authority."""
    state='start';generated=None;window=0;step=False;step_tokens=0;events=[]
    for row in rows:
        if row['kind']!='native_lifetime' or (row['rid'],row['slotgen'],row['slot'])!=(rid,slotgen,slot):
            raise ValueError('Exact actual continued leg ownership required')
        if row['actual_HTTP_client_terminal_qualified']is not False:raise ValueError('Native receipt cannot qualify HTTP terminal')
        if type(row['event'])is not int or(events and row['event']<=events[-1]):raise ValueError('Actual native chronology changed')
        for key in ('generated','batch_window'):
            if type(row[key])is not int or row[key]<0:raise ValueError('Exact nonnegative native count/window required')
        events.append(row['event']);phase=row['phase']
        if phase=='DONE_emitted':
            if state!='start' or row['generated']!=1:raise ValueError('Continued BGEN first-token DONE changed')
            generated=1;state='done'
        elif phase=='BADM_emitted':
            if state!='done' or row['continuation']!=1 or row['generated']!=generated:raise ValueError('Actual BADM1 prerequisite changed')
            state='active'
        elif phase=='BSTEP_before':
            if state!='active' or(step and step_tokens==0)or row['batch_window']<=window or row['generated']!=generated:raise ValueError('Actual BSTEP prerequisite changed')
            window=row['batch_window'];step=True;step_tokens=0
        elif phase=='BT_emitted':
            if state!='active' or not step or row['batch_window']!=window or row['generated']!=generated+1:raise ValueError('Actual BT count/window changed')
            generated+=1;step_tokens+=1
        elif phase=='BSTOP_applied':
            if state!='active' or row['generated']!=generated:raise ValueError('Actual BSTOP ownership/phase changed')
        elif phase=='BDONE_emitted':
            if state!='active' or not step or step_tokens==0 or row['batch_window']!=window or row['generated']!=generated or row['continuation']!=0 or row['finish']not in ('stop','cancel','length'):raise ValueError('Actual BDONE changed')
            state='terminal'
        else:raise ValueError('Unexpected actual native phase')
    if state!='terminal':raise ValueError('Complete actual native terminal missing')
    return {'rid':rid,'slotgen':slotgen,'slot':slot,'generated':generated,'native_terminal_observed':True,
            'actual_HTTP_client_terminal_qualified':False,'actual_full_cache_qualified':False}
