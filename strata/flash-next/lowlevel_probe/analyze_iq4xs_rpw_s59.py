#!/usr/bin/env python3
"""Host-only S59 full-byte closure; initial timings stay provisional until this passes."""
import hashlib,json,math,re,statistics,sys
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
    arm_names=['A1','B1','B2','A2'];alltimes={6:{1:[],2:[]}}
    evidence={};baseline=P/'A1'
    for arm,rpw in zip(arm_names,[1,2,2,1]):
        d=P/arm;r=json.loads((d/'result.json').read_text())
        check(r=={'status':'PASS','scope':'S59_PRIMITIVE_ONLY','rpw':rpw,'own_graph_cases':2,'timing_batches':3,'cross_process_reference':arm!='A1','serving_qualified':False},f'{arm} result')
        rows=[json.loads(s) for s in (P/(arm+'.log')).read_text().splitlines() if s.startswith('{')]
        ids=[x for x in rows if x.get('type')=='identity'];check(len(ids)==1,f'{arm} identity count')
        ident=ids[0];check((ident['N'],ident['M'],ident['weight_bytes'],ident['weight_fnv'],ident['pci'],ident['rpw'])==(2560,10240,13926400,'e408de02a955877f','0000:0b:00.0',rpw),f'{arm} identity')
        if arm!='A1':
            for field in ['driver','device','pci']:check(ident[field]==evidence['A1']['identity'][field],f'{arm} {field}')
        hashes={};dot_inventory={}
        for C in [6]:
            for changed in ['a','b']:
                names={f'C{C}-input-{changed}.bin':6*2560*4,
                       f'C{C}-quantized-{changed}.bin':C*2880}
                for stage in ['eager','graph']:
                    names[f'C{C}-{stage}-{changed}.bin']=C*10240*4
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
            per_call=2
            for graph,nodes in [('single',per_call),('batch20',20*per_call)]:
                s=(d/f'C{C}-{graph}.dot').read_text()
                check(s.count('TYPE = CGExecKernel')==nodes,f'{arm} {C} {graph} node count')
                # Inspect every captured name and edge before publishing any timing summary.
                calls=1 if graph=='single' else 20
                names=re.findall(r'NAME = ([^\\\n]+)\\n',s)
                quant=[n for n in names if 'native_quantize_q8_1_kernel' in n]
                proj=[n for n in names if 'native_mmvq_wide' in n]
                check(len(names)==nodes and len(quant)==len(proj)==calls,f'{arm} DOT kernel inventory')
                check(len(re.findall(r'"[^"\n]+" -> "[^"\n]+"',s))==nodes-1,f'{arm} DOT edge count')
                for name in proj:
                    check('WideIQ4XS' in name and 'dpct_kernel_scalarILi6EE' in name,f'{arm} IQ4_XS C6 name')
                    if rpw==1:check('native_mmvq_wide_rpw' not in name,f'{arm} unexpected RPW graph')
                    else:check('native_mmvq_wide_rpw' in name and 'ILi2EE' in name and 'ILi4EE' not in name,f'{arm} RPW2 name')
                dot_inventory[graph]={'sha256':sha(d/f'C{C}-{graph}.dot'),'nodes':nodes,'edges':nodes-1,'quantizer_nodes':len(quant),'projection_nodes':len(proj),'unique_kernel_names':sorted(set(names))}
        snapshots=[x for x in rows if x.get('type')=='checked_snapshot']
        expected_phases={'eager-a','eager-b','graph-a','graph-b','batch-check','timed-r0','timed-r1','timed-r2','final'}
        check(len(snapshots)==9 and {x['phase'] for x in snapshots}==expected_phases,f'{arm} checked phases')
        for x in snapshots:
            check(x['C']==6 and all(x.get(k) is True for k in ['guards','tails','finite','input_readonly']),f'{arm} snapshot structural gates')
            check(x['weight_readonly'] is (not x['phase'].startswith('timed-')),f'{arm} weight readback gates')
        timings=[x for x in rows if x.get('type')=='batch_timing'];check(len(timings)==3,f'{arm} timing count')
        check({(x['C'],x['round']) for x in timings}=={(c,r) for c in [6] for r in range(3)},f'{arm} timing keys')
        for x in timings:
            check(x['rpw']==rpw and x['captured_calls']==20 and x['warm_batch_replays']==3 and x['graph_submissions']==1 and x['begin_waited_before_host_t0'] is True,f'{arm} timing geometry')
            check(all(math.isfinite(x[k]) and x[k]>0 for k in ['complete_us','barrier_span_us']),f'{arm} timing span')
            alltimes[x['C']][rpw].append(x['barrier_span_us']/20)
        evidence[arm]={'identity':ident,'binary_snapshots':hashes,'DOT_inventory':dot_inventory,'timings':timings,'result':r}
    summary={}
    for C,arms in alltimes.items():
        a,b=arms[1],arms[2];summary[C]={'rpw1_us_per_quantize_projection_graph_call':a,'rpw2_us_per_quantize_projection_graph_call':b,'rpw1_median_us':statistics.median(a),'rpw2_median_us':statistics.median(b),'process_medians_us':{arm:statistics.median(x['barrier_span_us']/20 for x in evidence[arm]['timings']) for arm in arm_names},'median_change_percent':100*(statistics.median(b)/statistics.median(a)-1),'rpw1_minmax':[min(a),max(a)],'rpw2_minmax':[min(b),max(b)]}
    close_json(P/'comparison.json',{'status':'PASS_ALL_FULL_BYTE_GATES','process_order':['RPW1','RPW2','RPW2','RPW1'],'timing_scope':'process-balanced primitive graph screen; two processes per variant with three nested 20-call batch means each; init excluded; not same-process paired or serving qualification','evidence':evidence,'timings':summary,'speed_qualified':False})
    print(json.dumps({'status':'PASS_ALL_FULL_BYTE_GATES','timings':summary}));sys.exit(0)
except Exception as e:
    print('S59 comparison failure: '+str(e),file=sys.stderr);sys.exit(5 if isinstance(e,ValueError) else 2)
