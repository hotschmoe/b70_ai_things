#!/usr/bin/env python3
"""ROOT ONLY after exact source/contract reviews. No GPU imports or calls."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import ast,ctypes,hashlib,json,math,resource,struct,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from format_reference import need,sha_file,gamma_up,decode_q6,INF
P=Path(__file__).resolve().parent

def q8_decode(data,ni):
    need(len(data)==8*(ni//32)*36,'complete saved Q8 image')
    dt=np.dtype([('d','<f2'),('sum','<f2'),('q','i1',(32,))]);need(dt.itemsize==36,'Q8 layout')
    a=np.frombuffer(data,dtype=dt).reshape(8,ni//32);d=a['d'].astype(np.float64)
    need(np.isfinite(d).all(),'finite Q8 half scale')
    X=np.ascontiguousarray((d[:,:,None]*a['q'].astype(np.float64)).reshape(8,ni).T)
    return X,{'zero_scales':int((d==0).sum()),'min_nonzero_abs_half':float(np.abs(d[d!=0]).min()) if np.any(d!=0) else None,'max_abs_half':float(np.abs(d).max())}

def numeric_check(values,reference,Aupper,L,K):
    gr=gamma_up(K-1,2**-53)
    factor=np.nextafter(gamma_up(L,2**-24)+gr,INF)
    signed=values-reference;err=np.abs(signed);errup=np.nextafter(err,INF);errup[err==0]=0
    bound=np.nextafter(Aupper*factor,INF);bound[Aupper==0]=0
    bad=errup>bound
    ratio=np.divide(errup,bound,out=np.full_like(errup,np.inf),where=bound!=0);ratio[(errup==0)&(bound==0)]=0
    at=np.unravel_index(int(np.argmax(ratio)),ratio.shape)
    return {'L':L,'outputs':int(bad.size),'violations':int(bad.sum()),'max_abs_error':float(err.max()),'rms_error':float(np.sqrt(np.mean(signed*signed))),'max_error_over_bound':float(ratio.max()) if np.isfinite(ratio.max()) else 'INF','worst_ratio_row':int(at[0]),'worst_ratio_column':int(at[1]),'min_bound':float(bound.min()),'max_bound':float(bound.max()),'first_violations_row_column':np.argwhere(bad)[:16].tolist()}

def main():
    need(len(sys.argv)==3,'usage: check_collection.py CLOSED_RUN_DIR NEW_OUTPUT_DIR')
    run=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]);out.mkdir();start=time.monotonic()
    pins=json.loads((P/'checker-pins-v3.json').read_text());s41=Path(pins['S41']);B=Path(pins['B'])
    need(pins['columns']==[6,1,2,3,4,8] and pins['block_rows']==128,'fixed workload contract')
    need(pins['depths']=={'6144':{'original':15,'packed':18},'2560':{'original':12,'packed':15}},'fixed source depth contract')
    for name,h in pins['frozen_dependencies'].items():need(sha_file(name)==h,'frozen dependency '+name)
    oldpins=json.loads((Path(pins['S43'])/'v2/checker-pins.json').read_text())
    for rel,h in oldpins['production_sources'].items():need(sha_file(B/rel)==h,'unchanged production source '+rel)
    for name,h in pins['local_sources'].items():need(sha_file(P/name)==h,'frozen local source '+name)
    for name,h in pins['production_archive_hashes'].items():need(sha_file(B/'build-sycl-ep'/name)==h,'unchanged production archive '+name)
    build=json.loads((P/'build-closure.json').read_text());source_sha=sha_file(P/pins['collector_source'])
    need(build['source_sha256']==source_sha and sha_file(P/'build/q6_dense_collect')==build['binary_sha256'],'actual collection source/binary identity')
    link=(P/'build-v3.sh').read_text()
    for flag in ('-fp-model=precise','-cl-fp32-correctly-rounded-divide-sqrt','-device bmg-g31','-fsycl-device-code-split=per_kernel'):need(flag in link,'AOT flag '+flag)
    for flag in ('-ffast-math','-cl-fast-relaxed-math','-cl-unsafe-math-optimizations'):need(flag not in link,'unsafe math flag')
    native_rule=(B/'build-sycl-ep/build.ninja').read_text().split('build CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o:',1)[1].split('\n\n',1)[0]
    need('-fp-model=precise' in native_rule and '-ffast-math' not in native_rule,'precise native TU')
    libm=ctypes.CDLL('libm.so.6');libm.fegetround.restype=ctypes.c_int;need(libm.fegetround()==0,'CPU FE_TONEAREST')
    need(np.finfo(np.float64).nmant==52 and np.finfo(np.float32).nmant==23,'IEEE binary formats')
    identity=json.loads((run/'identity.json').read_text())
    need(identity['binary_sha256']==build['binary_sha256'] and identity['physical_card']==1 and identity['mask']=='1' and identity['selector']=='level_zero:*' and identity['numerical_processes']==1,'actual numerical process/binary/pin identity')
    closed=json.loads((run/'closure.json').read_text())
    need(closed['oracle_exit']==closed['pre_health_exit']==closed['post_health_exit']==0 and closed['container_removed'],'normal closed numerical process and health')
    events=[json.loads(line) for line in (run/'oracle.log').read_text().splitlines() if line.startswith('{')]
    final=[e for e in events if e.get('status')=='COLLECTED_UNVALIDATED']
    need(len(final)==1 and events[-1]==final[0] and final[0]['fixtures']==4 and final[0]['column_pairs']==24 and final[0]['own_changed_direct_pairs']==4 and final[0]['numerical_validated'] is False,'complete unvalidated collector footer')
    need(len(events)==89 and len([e for e in events if e.get('type')=='fixture'])==4 and len([e for e in events if e.get('type')=='fixture_COLLECTED'])==4 and len([e for e in events if e.get('type')=='strict_difference'])==32,'all collection frames complete')
    calls=[e for e in events if e.get('type')=='call_observation']
    need(len(calls)==48,'all direct calls recorded')
    fixtures=json.loads((P/'fixture-selection.json').read_text())['fixtures'];need([f['shape'] for f in fixtures]==pins['shapes'],'four exact shapes')
    columns=pins['columns'];raw=run/'snapshots';expected=[]
    need({(e['tensor'],e['phase'],e['C']) for e in calls}=={(f['name'],a,C) for f in fixtures for a in ('original','packed') for C in columns},'each direct call exactly once')
    for fi in range(4):
        expected += [f'f{fi}-C{C}-{arm}.bin' for C in columns for arm in ('original','packed')]
        expected += [f'f{fi}-C6-{arm}-{role}.bin' for arm in ('original','packed') for role in ('direct-changed','graph-original','graph-changed')]
        expected += [f'f{fi}-{n}.bin' for n in ('input-f32','input-q8_1','changed-input-f32','changed-input-q8_1','original-graph-changed-q8_1','packed-graph-changed-q8_1')]
    need(len(set(expected))==96 and sorted(p.name for p in raw.iterdir())==sorted(expected),'exact96 complete artifacts')
    watched=[run/'oracle.log',run/'oracle.stderr',run/'closure.json',run/'identity.json']+[raw/n for n in expected]
    observed_hashes={str(f):sha_file(f) for f in watched}
    (out/'collection-input-hashes.json').write_text(json.dumps(observed_hashes,indent=2)+'\n',encoding='ascii')
    tree=ast.parse((s41/'cpu-reference/reference.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='q6_decode')
    ns={'struct':struct};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen-S41-q6_decode>','exec'),ns)
    oldcpu=json.loads((s41/'cpu-reference/reference.json').read_text());known={e['row']:e for e in oldcpu['selected_model_reads']};oldexact={(e['row'],e['column']):e for e in oldcpu['exact_rational_sanity']}
    need(len(known)==19 and set(oldexact)=={(1,0),(782,1),(2314,5)},'fixed previous rational/row checks')
    all_results=[];total_read=0
    for fi,f in enumerate(fixtures):
        ni,no=f['shape'];need(f['ggml_type']==14 and f['original_bytes']==ni//256*no*210,'Q6 byte geometry')
        rowsize=ni//256*210;prefix=f'f{fi}-'
        header=[e for e in events if e.get('type')=='fixture' and e.get('tensor')==f['name']]
        done=[e for e in events if e.get('type')=='fixture_COLLECTED' and e.get('tensor')==f['name']]
        need(len(header)==len(done)==1,'unique complete fixture frame')
        h,done=header[0],done[0]
        need((h['n_in'],h['n_out'],h['offset'],h['bytes'])==(ni,no,f['absolute_offset'],f['original_bytes']),'actual payload metadata')
        for key in ('own_initial_graph_bitmatch','own_changed_graph_bitmatch','Q8_restore_bitmatch','guards','inputs_unchanged'):need(done[key] is True,'fatal runtime invariant '+key)
        need(done['numerical_validated'] is False and done['columns']==columns,'collector not numeric PASS')
        q0=(raw/(prefix+'input-q8_1.bin')).read_bytes();q1=(raw/(prefix+'changed-input-q8_1.bin')).read_bytes()
        X0,ds0=q8_decode(q0,ni);X1,ds1=q8_decode(q1,ni);activeq=6*(ni//32)*36
        need(q0[:activeq]!=q1[:activeq] and q0[activeq:]==q1[activeq:],'actual changed active6/original unused2 Q8')
        for arm in ('original','packed'):need((raw/(prefix+arm+'-graph-changed-q8_1.bin')).read_bytes()==q1,'actual changed graph Q8 equality')
        for role in ('input-f32','changed-input-f32'):
            z=np.fromfile(raw/(prefix+role+'.bin'),dtype='<f4');need(z.size==8*ni and np.isfinite(z).all(),'saved actual FP32 input bytes/finites')
        need((raw/(prefix+'input-f32.bin')).read_bytes()!=(raw/(prefix+'changed-input-f32.bin')).read_bytes(),'changed FP32 observable')
        if fi==0:
            for n in ('input-q8_1','input-f32'):need(sha_file(raw/(prefix+n+'.bin'))==sha_file(s41/'run-profiled/snapshots'/('f0-'+n+'.bin')),'actual first-fixture S41 input continuity')
        values={};images={}
        for name in [n for n in expected if n.startswith(prefix+'C')]:
            C=int(name.split('-C',1)[1].split('-',1)[0]);data=(raw/name).read_bytes()
            need(len(data)==8*no*4 and data[C*no*4:]==b'\xff'*((8-C)*no*4),'inactive tail/length '+name)
            v=np.frombuffer(data,dtype='<f4',count=C*no).reshape(C,no).T.astype(np.float64)
            need(np.isfinite(v).all(),'active finite '+name);values[name]=v;images[name]=data
        for arm in ('original','packed'):
            need(images[f'{prefix}C6-{arm}-graph-original.bin']==images[f'{prefix}C6-{arm}.bin'],'own initial graph/direct bitmatch')
            need(images[f'{prefix}C6-{arm}-graph-changed.bin']==images[f'{prefix}C6-{arm}-direct-changed.bin'],'own changed graph/direct bitmatch')
        diffs=[e for e in events if e.get('type')=='strict_difference' and e.get('tensor')==f['name']]
        need(len(diffs)==8 and len({(e['role'],e['C']) for e in diffs})==8,'complete strict-difference records')
        for e in diffs:
            C=e['C'];suffix='' if e['role']=='direct' else '-'+e['role'].replace('changed-direct','direct-changed').replace('changed-graph','graph-changed')
            need((e['role']=='direct' and C in columns) or (e['role'] in ('changed-direct','changed-graph') and C==6),'expected difference role')
            av=values[f'{prefix}C{C}-original{suffix}.bin'];bv=values[f'{prefix}C{C}-packed{suffix}.bin'];bits=av.view(np.uint64)!=bv.view(np.uint64)
            first=np.flatnonzero(bits.T.ravel());firstbyte=4*int(first[0]) if first.size else -1
            need(e['bit_different_values']==int(bits.sum()) and e['first_value_byte']==firstbyte and e['numerical_validated'] is False,'strict diff provenance')
        X=np.ascontiguousarray(np.concatenate((X0,X1[:,:6]),axis=1));absX=np.abs(X)
        ref=np.empty((no,14));Au=np.empty_like(ref);gr=gamma_up(ni-1,2**-53);den=np.nextafter(np.float64(1)-gr,-INF)
        q6stats=[];rational=[];verified=set();payload_sha=hashlib.sha256();fnv=0xcbf29ce484222325;read=0
        model=Path(f['file']).resolve();need(model.stat().st_size==47039860096 and 0<=f['absolute_offset']<=model.stat().st_size-f['original_bytes'],'model/row bounds')
        selected=set(known) if fi==0 else {0,1,no-1}
        with model.open('rb') as stream:
            stream.seek(f['absolute_offset'])
            for first in range(0,no,128):
                nr=min(128,no-first);payload=stream.read(nr*rowsize);need(len(payload)==nr*rowsize,'complete bounded row block');read+=len(payload);payload_sha.update(payload)
                for byte in payload:fnv=((fnv^byte)*0x100000001b3)&0xffffffffffffffff
                W,stat=decode_q6(payload,nr,ni);q6stats.append(stat)
                R=W@X;Ahat=np.abs(W)@absX;need(np.isfinite(R).all() and np.isfinite(Ahat).all(),'finite independent FP64 dots/norms')
                U=np.nextafter(Ahat/den,INF);U[Ahat==0]=0;ref[first:first+nr]=R;Au[first:first+nr]=U
                for row in selected:
                    if first<=row<first+nr:
                        local=row-first;rb=payload[local*rowsize:(local+1)*rowsize];scalar=[]
                        for k in range(ni//256):scalar.extend(ns['q6_decode'](rb[k*210:(k+1)*210]))
                        need(np.array_equal(W[local],np.asarray(scalar)),'vector/scalar independent stored-format mapping')
                        if fi==0:need(hashlib.sha256(rb).hexdigest()==known[row]['sha256'],'prior selected actual row SHA')
                        verified.add(row)
                # Two predeclared exact reference checks per fixture: initial/changed c0,row0.
                cases={(0,0):None,(0,8):None}
                if fi==0:cases.update({(r,c):e for (r,c),e in oldexact.items()})
                for (row,c),previous in cases.items():
                    if first<=row<first+nr:
                        exact=Fraction(int(previous['exact_numerator']),int(previous['exact_denominator'])) if previous else sum((Fraction.from_float(float(w))*Fraction.from_float(float(x)) for w,x in zip(W[row-first],X[:,c])),Fraction(0))
                        delta=abs(Fraction.from_float(float(R[row-first,c]))-exact);allowed=np.nextafter(gr*U[row-first,c],INF)
                        need(delta<=Fraction.from_float(float(allowed)),'independent rational reference allowance')
                        rational.append({'row':row,'reference_column':c,'exact_rational':str(exact),'absolute_reference_error_exact':str(delta),'allowed_reference_error':float(allowed)})
        need(read==f['original_bytes'] and f'{fnv:016x}'==h['fnv1a64'] and verified==selected,'payload identity and all scalar spot rows checked');total_read+=read
        need(len(rational)==(5 if fi==0 else 2),'all predeclared exact references')
        cases={};negative={};depths=pins['depths'][str(ni)]
        for name,v in values.items():
            arm=name.split('-',2)[2].split('-',1)[0].removesuffix('.bin');changed=name.endswith(('direct-changed.bin','graph-changed.bin'));C=v.shape[1];sl=slice(8,14) if changed else slice(0,C)
            cases[name]=numeric_check(v,ref[:,sl],Au[:,sl],depths[arm],ni)
        for arm in ('original','packed'):
            v=values[f'{prefix}C6-{arm}.bin'];control=numeric_check(np.roll(v,1,axis=0),ref[:,:6],Au[:,:6],depths[arm],ni)
            need(control['violations']>0,'gross wrong-row control rejected');negative[arm]={'mutation':'one +1 row output roll on directC6, same reference/bound','violations':control['violations'],'outputs':control['outputs'],'gross_sensitivity_only':True}
        np.savez(out/f'fixture{fi}-reference.npz',reference64=ref,A_upper=Au)
        result={'tensor':f['name'],'shape':[ni,no],'outputs_checked':sum(c['outputs'] for c in cases.values()),'cases':cases,'negative_controls':negative,'Fraction_checks':rational,'payload_bytes_read':read,'payload_fnv1a64':f'{fnv:016x}','payload_sha256_newly_recorded':payload_sha.hexdigest(),'scalar_decoder_rows':sorted(verified),'actual_Q8_initial':ds0,'actual_Q8_changed':ds1,'actual_Q6_half_scale_blocks':q6stats,'runtime_structural_checks_inherited':done,'CPU_finiteness_tails_own_graph_identity':True}
        (out/f'fixture{fi}-gate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='ascii');all_results.append(result)
    need(total_read==73113600,'exact total selected payload read')
    for path,h in observed_hashes.items():need(sha_file(path)==h,'collection unchanged during CPU check '+path)
    violations=sum(c['violations'] for f in all_results for c in f['cases'].values());status='PASS_PROPOSED_CPU_BOUND_FOUR_FIXTURES_ONLY' if violations==0 else 'FAIL_PROPOSED_CPU_BOUND_FOUR_FIXTURES'
    final={'status':status,'whole_model_qualified':False,'S41_strict_identity':'FAILED_UNCHANGED','collection_status':'COLLECTED_UNVALIDATED','outputs_checked':sum(f['outputs_checked'] for f in all_results),'violations':violations,'fixture_count':4,'case_count':sum(len(f['cases']) for f in all_results),'model_payload_bytes_read':total_read,'unique_reference_products':1247805440,'positive_norm_products_additional':1247805440,'reference':'Unchanged C8 + changed C6; prefix reuse only after actual Q8/own-graph consistency and full collector closure','identity':{'source_sha256':source_sha,'binary_sha256':build['binary_sha256'],'checker_sha256':sha_file(Path(__file__)),'helpers_sha256':sha_file(P/'format_reference.py'),'pins_sha256':sha_file(P/'checker-pins-v3.json'),'collection_hashes_unchanged':True,'actual_run_identity':identity},'preconditions':{'CPU_FE_TONEAREST':True,'GPU_RNE_basis':'existing precise default FP32/AOT flags; no new device-state readback','finite_actual_half_scales':True,'exact_product_max_significand_bits_conservative':44,'no_overflow_underflow_FTZ':'finite-half2^-24/helper2^-24/output2^-48 grid and K<=6144 range proof; zeros allowed','bound':'gamma_source_depth + gamma_(K-1)(u64), outward A_upper/bounds, A=0 exact y=0'},'resource':{'block_rows':128,'one_fixture_at_a_time':True,'explicit_arrays_budget_bytes':64*1024*1024,'external_BLAS_workspace_not_in_budget':True,'BLAS_thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS')},'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_s':time.monotonic()-start},'limits':['Four selected real matrices/declared columns and changed inputs only, not all52 packed matrices or serving.','Numerical bound contract differs from bitidentity; no production, model-quality, stability or speed qualification.','Negative controls test gross row misassociation only.']}
    (out/'gate.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n',encoding='ascii');print(json.dumps(final));return 0 if violations==0 else 5
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:print('CPU_COLLECTION_CHECK_FAILED: '+str(e),file=sys.stderr);raise SystemExit(2)
