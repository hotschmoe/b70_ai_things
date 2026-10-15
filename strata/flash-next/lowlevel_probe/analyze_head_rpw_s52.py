#!/usr/bin/env python3
"""Host-only S52 full-byte closure; initial timings stay provisional until this passes."""
import hashlib,json,statistics,sys
from pathlib import Path
P=Path(sys.argv[1])
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def check(ok,msg):
    if not ok: raise ValueError(msg)
def close_json(p,v):
    with p.open('x',encoding='ascii') as f:json.dump(v,f,indent=2);f.write('\n')
try:
    arm_names=['A1','B1','B2','A2'];alltimes={2:{1:[],2:[]},6:{1:[],2:[]}}
    evidence={};baseline=P/'A1'
    for arm,rpw in zip(arm_names,[1,2,2,1]):
        d=P/arm;r=json.loads((d/'result.json').read_text())
        check(r=={'status':'PASS','scope':'S52_PRIMITIVE_ONLY','rpw':rpw,'own_graph_cases':4,'timing_batches':6,'cross_process_reference':arm!='A1','serving_qualified':False},f'{arm} result')
        rows=[json.loads(s) for s in (P/(arm+'.log')).read_text().splitlines() if s.startswith('{')]
        ids=[x for x in rows if x.get('type')=='identity'];check(len(ids)==1,f'{arm} identity count')
        ident=ids[0];check((ident['N'],ident['M'],ident['weight_bytes'],ident['weight_fnv'],ident['pci'],ident['rpw'])==(2560,248320,437043200,'da8746c3dcc2469e','0000:0b:00.0',rpw),f'{arm} identity')
        if arm!='A1':
            for field in ['driver','device','pci']:check(ident[field]==evidence['A1']['identity'][field],f'{arm} {field}')
        hashes={}
        for C in [2,6]:
            for changed in ['a','b']:
                names={f'C{C}-input-{changed}.bin':6*2560*4,
                       f'C{C}-quantized-{changed}.bin':C*2880}
                for stage in ['eager','graph']:
                    names[f'C{C}-{stage}-{changed}.bin']=C*248320*4
                    names[f'C{C}-{stage}-{changed}-q8.bin']=C*2880
                for name,n in names.items():
                    f=d/name;check(f.stat().st_size==n,f'{arm}/{name} size')
                    a=f.read_bytes();b=(baseline/name).read_bytes()
                    if a!=b:
                        offset=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
                        close_json(P/'comparison-first-mismatch.json',{'arm':arm,'file':name,'first_byte_offset':offset,'actual':a[offset:offset+16].hex(),'reference':b[offset:offset+16].hex()})
                        raise ValueError(f'STRICT cross-process {arm}/{name} at byte {offset}; full files retained')
                    hashes[name]=hashlib.sha256(a).hexdigest()
                check((d/f'C{C}-eager-{changed}.bin').read_bytes()==(d/f'C{C}-graph-{changed}.bin').read_bytes(),f'{arm} own graph')
                q=(d/f'C{C}-quantized-{changed}.bin').read_bytes()
                for stage in ['eager','graph']:check((d/f'C{C}-{stage}-{changed}-q8.bin').read_bytes()==q,f'{arm} Q8 readonly')
            check((d/f'C{C}-graph-a.bin').read_bytes()!=(d/f'C{C}-graph-b.bin').read_bytes(),f'{arm} changed graph')
            per_call=3 if C==2 else 2
            for graph,nodes in [('single',per_call),('batch20',20*per_call)]:
                s=(d/f'C{C}-{graph}.dot').read_text()
                check(s.count('TYPE = CGExecKernel')==nodes,f'{arm} {C} {graph} node count')
                # DOT contains full captured kernel names on this stack. Establish actual
                # branch, not just env intent; quantizer is the other half of the graph.
                branch='native_mmvq_wide_rpw' if rpw==2 else 'native_mmvq_wide'
                check(branch in s,f'{arm} head branch DOT')
                if rpw==1:check('native_mmvq_wide_rpw' not in s,f'{arm} unexpected RPW graph')
        timings=[x for x in rows if x.get('type')=='batch_timing'];check(len(timings)==6,f'{arm} timing count')
        check({(x['C'],x['round']) for x in timings}=={(c,r) for c in [2,6] for r in range(3)},f'{arm} timing keys')
        for x in timings:
            check(x['rpw']==rpw and x['captured_calls']==20 and x['warm_batch_replays']==3 and x['graph_submissions']==1 and x['begin_waited_before_host_t0'] is True,f'{arm} timing geometry')
            check(x['complete_us']>0 and x['barrier_span_us']>0,f'{arm} timing span')
            alltimes[x['C']][rpw].append(x['barrier_span_us']/20)
        evidence[arm]={'identity':ident,'binary_snapshots':hashes,'timings':timings,'result':r}
    summary={}
    for C,arms in alltimes.items():
        a,b=arms[1],arms[2];summary[C]={'rpw1_us_per_quantize_head_graph_call':a,'rpw2_us_per_quantize_head_graph_call':b,'rpw1_median_us':statistics.median(a),'rpw2_median_us':statistics.median(b),'median_change_percent':100*(statistics.median(b)/statistics.median(a)-1),'rpw1_minmax':[min(a),max(a)],'rpw2_minmax':[min(b),max(b)]}
    close_json(P/'comparison.json',{'status':'PASS_ALL_FULL_BYTE_GATES','process_order':['RPW1','RPW2','RPW2','RPW1'],'timing_scope':'process-balanced primitive graph screen; excludes process init; not same-process paired or serving qualification','evidence':evidence,'timings':summary,'speed_qualified':False})
    print(json.dumps({'status':'PASS_ALL_FULL_BYTE_GATES','timings':summary}));sys.exit(0)
except Exception as e:
    print('S52 comparison failure: '+str(e),file=sys.stderr);sys.exit(5 if isinstance(e,ValueError) else 2)
