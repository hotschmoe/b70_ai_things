#!/usr/bin/env python3
"""ROOT ONLY after exact model review. CPU-only; never imports GPU libraries."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import ast,ctypes,hashlib,json,math,resource,struct,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
INF=np.float64(np.inf)
def need(ok,message):
    if not ok: raise RuntimeError(message)
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def gamma_up(n,u):
    # n*u and 1-n*u are exactly representable for the fixed counts used here.
    return np.nextafter(np.float64(n*u/(1-n*u)),INF)
def decode_q6(payload,nrows):
    # Storage mapping from artifact/dequant.hpp, NOT either GPU lane/reduction tree.
    dt=np.dtype([('ql','u1',(128,)),('qh','u1',(64,)),('sc','i1',(16,)),('d','<f2')])
    need(dt.itemsize==210 and [dt.fields[n][1] for n in ('ql','qh','sc','d')]==[0,128,192,208],'Q6 layout')
    a=np.frombuffer(payload,dtype=dt).reshape(nrows,24)
    d=a['d'].astype(np.float64)
    need(np.isfinite(d).all(),'nonfinite Q6 half scale')
    w=np.empty((nrows,24,256),dtype=np.float64)
    si=np.arange(32)//16
    for half in range(2):
        lo0=a['ql'][:,:,half*64:half*64+32];lo1=a['ql'][:,:,half*64+32:half*64+64]
        hi=a['qh'][:,:,half*32:half*32+32]
        for part,lo in enumerate((lo0,lo1,lo0,lo1)):
            low=(lo&15) if part<2 else (lo>>4)
            q=((low|(((hi>>(2*part))&3)<<4)).astype(np.int16)-32)
            sc=a['sc'][:,:,half*8+si+part*2].astype(np.float64)
            w[:,:,half*128+part*32:half*128+(part+1)*32]=d[:,:,None]*sc*q
    need(np.isfinite(w).all(),'nonfinite decoded weight')
    nonzero=np.abs(d[d!=0])
    stat={'halves':int(d.size),'zero_scales':int((d==0).sum()),'min_nonzero_abs_half':float(nonzero.min()) if nonzero.size else None,'max_abs_half':float(np.abs(d).max())}
    return w.reshape(nrows,6144),stat

def main():
    need(len(sys.argv)==2,'usage: check_first_c6.py NEW_OUTPUT_DIR')
    out=Path(sys.argv[1]);out.mkdir()  # preserve any previous run
    started=time.monotonic();pins=json.loads((P/'checker-pins.json').read_text());s=Path(pins['S41']);b=Path(pins['B'])
    need(pins['L']=={'original':15,'packed':18},'predeclared source bounds changed')
    need((pins['K'],pins['C'],pins['rows'],pins['block_rows'])==(6144,6,2560,128),'predeclared fixture dimensions')
    for rel,expected in pins['S41_files'].items(): need(sha_file(s/rel)==expected,'S41 hash '+rel)
    for rel,expected in pins['production_sources'].items(): need(sha_file(b/rel)==expected,'source hash '+rel)
    need(sha_file(s/'build/q6_dense_oracle')==pins['original_binary_sha256'],'original binary identity')
    report=json.loads((s/'s41-report.json').read_text());coverage=report['coverage']
    for key in ('guards_finites_unused_tails_pass_at_saved_snapshots','original_weights_unchanged_before_and_after_pack','native_q8_input_unchanged_through_failed_comparison','initial_fp32_input_unchanged_before_pack'):
        need(coverage[key] is True,'inherited runtime check '+key)
    closure=json.loads((s/'run-profiled/closure.json').read_text())
    need(closure['oracle_exit']==5 and closure['pre_health_exit']==closure['post_health_exit']==0 and closure['container_removed'],'original runtime closure')
    need('STRICT original/packed blk.3.attn_output.weight C6 first_byte=4' in (s/'run-profiled/oracle.stderr').read_text(),'original failure reason')
    trace=(s/'run-profiled/unitrace.1.txt').read_text()
    need('launch_q6k_wide<6>' in trace and 'native_mmvq_q6k_wide_a2' in trace and 'native_q6_k_packed_kernel_ca2554, dpct_kernel_scalar<6>, dpct_kernel_scalar<0>' in trace,'actual counted kernel variants')
    native_rule=(b/'build-sycl-ep/build.ninja').read_text().split('build CMakeFiles/strata_kernels.dir/src/kernels/cuda/native_mmvq.dp.cpp.o:',1)[1].split('\n\n',1)[0]
    need('-fp-model=precise' in native_rule and '-ffast-math' not in native_rule,'precise production TU')
    link=(s/'build-v2.sh').read_text()
    for flag in ('-fp-model=precise','-cl-fp32-correctly-rounded-divide-sqrt','-device bmg-g31','-fsycl-device-code-split=per_kernel'):
        need(flag in link,'actual AOT/link flag '+flag)
    for flag in ('-ffast-math','-cl-fast-relaxed-math','-cl-unsafe-math-optimizations'):
        need(flag not in link and flag not in native_rule,'unsafe math flag '+flag)
    libm=ctypes.CDLL('libm.so.6');libm.fegetround.restype=ctypes.c_int
    need(libm.fegetround()==0,'CPU FE_TONEAREST required')
    need(np.finfo(np.float64).nmant==52 and np.finfo(np.float32).nmant==23,'IEEE binary formats')
    # Load only the existing pure scalar format function, not S41 top-level code.
    tree=ast.parse((s/'cpu-reference/reference.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='q6_decode')
    scalar_ns={'struct':struct};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<frozen-S41-q6_decode>','exec'),scalar_ns)
    cpu=json.loads((s/'cpu-reference/reference.json').read_text());cpu_manifest=json.loads((s/'cpu-reference/manifest.json').read_text())
    for name,entry in cpu_manifest['files'].items(): need(sha_file(s/'cpu-reference'/name)==entry['sha256'],'S41 CPU manifest '+name)
    fixture=json.loads((s/'fixture-selection.json').read_text())['fixtures'][0]
    need(fixture['shape']==[6144,2560] and fixture['absolute_offset']==3572274688 and fixture['original_bytes']==12902400 and fixture['ggml_type']==14,'fixture contract')
    raw=s/'run-profiled/snapshots';qbytes=(raw/'f0-input-q8_1.bin').read_bytes()
    need(len(qbytes)==8*192*36,'complete saved native Q8 image')
    dt=np.dtype([('d','<f2'),('sum','<f2'),('q','i1',(32,))]);need(dt.itemsize==36,'Q8 layout')
    q=np.frombuffer(qbytes,dtype=dt).reshape(8,192)[:6]
    d8=q['d'].astype(np.float64);need(np.isfinite(d8).all(),'nonfinite actual Q8 half d')
    X=np.ascontiguousarray((d8[:,:,None]*q['q'].astype(np.float64)).reshape(6,6144).T)
    absX=np.abs(X);need(np.isfinite(X).all(),'nonfinite saved Q8 decode')
    outputs={}
    for arm in ('original','packed'):
        data=(raw/('f0-C6-'+arm+'.bin')).read_bytes()
        need(len(data)==81920 and data[61440:]==b'\xff'*20480,'full inactive poison tail '+arm)
        outputs[arm]=np.frombuffer(data,dtype='<f4',count=15360).reshape(6,2560).T.astype(np.float64)
        need(np.isfinite(outputs[arm]).all(),'active finiteness '+arm)
    # All original half*integer decoded products fit <=44 bits by a conservative
    # count (Q6<=11+8+6; Q8<=11+8); their binary64 product is exact (<53 bits).
    # Half d6,d8 grid2^-24 => FP32 helper grids2^-24/2^-48; finite scales and K
    # bound imply no FP32 subnormal/overflow, including signed cancellation.
    gref=gamma_up(6143,2**-53);den=np.nextafter(np.float64(1)-gref,-INF)
    g={arm:np.nextafter(gamma_up(L,2**-24)+gref,INF) for arm,L in pins['L'].items()}
    refs=np.empty((2560,6));upper=np.empty_like(refs);stats=[];spot_results=[]
    model=Path(fixture['file']).resolve();need(model.stat().st_size==47039860096,'shard size')
    payload_sha=hashlib.sha256();fnv=0xcbf29ce484222325;payload_read=0
    selected_rows={e['row']:e for e in cpu['selected_model_reads']};exact={(e['row'],e['column']):e for e in cpu['exact_rational_sanity']}
    need(len(selected_rows)==len(cpu['selected_model_reads'])==19,'all nineteen unique selected rows')
    need(set(exact)=={(row,column) for column,row in pins['reference_spots']},'fixed rational spots are column,row in pins')
    verified_rows=set()
    with model.open('rb') as f:
        f.seek(fixture['absolute_offset'])
        for first in range(0,2560,128):
            nr=min(128,2560-first);payload=f.read(nr*5040);need(len(payload)==nr*5040,'truncated row block');payload_read+=len(payload);payload_sha.update(payload)
            for byte in payload: fnv=((fnv^byte)*0x100000001b3)&0xffffffffffffffff
            W,stat=decode_q6(payload,nr);stats.append(stat)
            for row,entry in selected_rows.items():
                if first<=row<first+nr:
                    local=row-first;rb=payload[local*5040:(local+1)*5040]
                    need(hashlib.sha256(rb).hexdigest()==entry['sha256'],'known row payload '+str(row))
                    # Reuse frozen scalar CPU decoder; no duplicated quantizer.
                    scalar=[]
                    for k in range(24): scalar.extend(scalar_ns['q6_decode'](rb[k*210:(k+1)*210]))
                    need(np.array_equal(W[local],np.asarray(scalar)),'vector/scalar storage decoder '+str(row))
                    verified_rows.add(row)
            ref=W@X;Ahat=np.abs(W)@absX
            need(np.isfinite(ref).all() and np.isfinite(Ahat).all(),'finite FP64 dots/norms')
            Au=np.nextafter(Ahat/den,INF);Au[Ahat==0]=0
            refs[first:first+nr]=ref;upper[first:first+nr]=Au
            for (row,c),entry in exact.items():
                if first<=row<first+nr:
                    v=Fraction(int(entry['exact_numerator']),int(entry['exact_denominator']))
                    delta=abs(Fraction.from_float(float(ref[row-first,c]))-v)
                    allowed=np.nextafter(gref*Au[row-first,c],INF)
                    need(delta<=Fraction.from_float(float(allowed)),'reference beyond rational error allowance')
                    spot_results.append({'row':row,'column':c,'reference64':float(ref[row-first,c]),'exact_rational':str(v),'reference_abs_error_exact':str(delta),'reference_error_allowance':float(allowed)})
    need(payload_read==12902400 and f'{fnv:016x}'==pins['original_payload_fnv1a64'],'full payload original-oracle FNV identity')
    need(verified_rows==set(selected_rows) and len(verified_rows)==19,'all selected row SHA and scalar decoder checks executed')
    need(len(spot_results)==3,'all fixed Fraction spots')
    summary={};negative={};arrays={'reference64':refs,'A_upper':upper}
    for arm in ('original','packed'):
        signed=outputs[arm]-refs;err=np.abs(signed)
        err_up=np.nextafter(err,INF);err_up[err==0]=0
        bound=np.nextafter(upper*g[arm],INF);bound[upper==0]=0
        ok=err_up<=bound
        ratio=np.divide(err_up,bound,out=np.full_like(err_up,np.inf),where=bound!=0);ratio[(err_up==0)&(bound==0)]=0
        worst=np.unravel_index(int(np.argmax(ratio)),ratio.shape)
        summary[arm]={'L':pins['L'][arm],'outputs':int(ok.size),'violations':int((~ok).sum()),'max_abs_error':float(err.max()),'rms_error':float(np.sqrt(np.mean(signed*signed))),'max_error_over_bound':float(ratio.max()) if np.isfinite(ratio.max()) else 'INF','worst_ratio_row':int(worst[0]),'worst_ratio_column':int(worst[1]),'min_bound':float(bound.min()),'max_bound':float(bound.max())}
        arrays[arm+'_signed_error']=signed;arrays[arm+'_bound']=bound;arrays[arm+'_passes']=ok
        # One predeclared gross-sensitivity control per path. No new reference.
        shifted=np.roll(outputs[arm],1,axis=0);mutant_err=np.abs(shifted-refs)
        mutant_up=np.nextafter(mutant_err,INF);mutant_up[mutant_err==0]=0
        failures=int((mutant_up>bound).sum())
        negative[arm]={'mutation':'np.roll(saved_output,1,axis=rows)','same_unmodified_reference_and_bound':True,'violations':failures,'outputs':15360,'gross_sensitivity_only':True}
        need(failures>0,'wrong-row negative control not rejected '+arm)
    np.savez(out/'full-first-C6-reference.npz',**arrays)
    status='PASS_PROPOSED_CPU_BOUND_FIRST_PAIR_ONLY' if all(v['violations']==0 for v in summary.values()) else 'FAIL_PROPOSED_CPU_BOUND_FIRST_PAIR'
    result={'status':status,'whole_model_qualified':False,'S41_strict_identity':'FAILED_UNCHANGED','gate_policy':'Predeclared L15/L18 source-derived bounds; requires root/model approval before execution; no empirical fitted tolerance','scope':{'tensor':fixture['name'],'rows':2560,'C':6,'outputs_per_path':15360,'products':94371840,'payload_bytes_read':payload_read,'block_rows':128,'no_GPU':True,'no_requantization':True},'summary':summary,'negative_control':negative,'reference_error':{'gamma6143_u64_up':float(gref),'A_upper':'outward Ahat/(1-gamma_ref)','GPU_allowance':'outward (gamma_L32+gamma_ref)*A_upper','exact_product_max_significand_bits_conservative':44,'Fraction_spots':spot_results},'preconditions':{'CPU_FE_TONEAREST':True,'GPU_RNE_basis':'standard precise FP32 existing source/flags; no device rounding-state readback in this CPU stage','Q8_half_scale_finite':True,'Q8_min_nonzero_abs_half':float(np.min(np.abs(d8[d8!=0]))) if np.any(d8!=0) else None,'Q8_max_abs_half':float(np.abs(d8).max()),'Q6_half_scale_blocks':stats,'FP32_normal_overflow_FTZ':'finite binary16 grid/range proof in bound-proof.md; no dumped intermediates','actual_counted_kernel_names':True,'native_TU_flag':'-fp-model=precise','AOT_link_flags':['-device bmg-g31','-cl-fp32-correctly-rounded-divide-sqrt','-fsycl-device-code-split=per_kernel'],'unsafe_fast_math_flags_absent':True},'structural':{'CPU_active_finiteness':True,'CPU_full_inactive_poison_tails':True,'inherited_runtime_guard_checks_before_strict_exit':True,'inherited_input_payload_checks':{k:coverage[k] for k in coverage if 'unchanged' in k},'final_success_only_input_checks_executed':coverage['final_success_only_input_checks_executed'],'new_GPU_guard_measurement':False},'identity':{'pins_sha256':sha_file(P/'checker-pins.json'),'checker_sha256':sha_file(Path(__file__)),'full_payload_fnv1a64':f'{fnv:016x}','full_payload_sha256_newly_recorded':payload_sha.hexdigest(),'prior_selected_row_SHA_checks':len(selected_rows),'source_and_snapshot_hashes_verified':True,'numpy_version':np.__version__},'resource':{'explicit_block_W_bytes':128*6144*8,'W_plus_absW_bytes':2*128*6144*8,'input_plus_abs_input_bytes':2*6144*6*8,'decode_temporaries_conservative_bytes':6*128*24*32*8,'peak_explicit_array_budget_bytes':24*1024*1024,'BLAS_workspace_not_in_budget':True,'process_peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'BLAS_thread_env':{k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS')},'elapsed_s':time.monotonic()-started},'limits':['First real matrix/C6 only; other shapes, columns and changed graphs untested.','Conservative bound compliance is a proposed numerical contract, not proof of bitidentity, all indexing semantics, model quality or serving stability.','No timing/speed conclusion; CPU checker uses existing profiled outputs.']}
    (out/'gate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='ascii')
    print(json.dumps({'status':status,'summary':summary,'elapsed_s':result['resource']['elapsed_s']}))
    return 0 if status.startswith('PASS_') else 5
if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as e:
        print('CPU_REFERENCE_CHECK_FAILED: '+str(e),file=sys.stderr);raise SystemExit(2)
