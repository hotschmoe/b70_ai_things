#!/usr/bin/env python3
# S62 sampled, normal-scale CPU operand reference. No GPU acceptance tolerance.
import array,csv,hashlib,json,math,os,struct,subprocess,time
from fractions import Fraction
from pathlib import Path
P=Path(__file__).resolve().parent;A=Path('/mnt/vm_8tb/b70/build/strata-ab')
E=Path('/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-grouped-gemm-s42-prep/run-clean/exports')
O=P/'cells';O.mkdir();assert os.access(O,os.W_OK)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(n,v):
 with (O/n).open('x',encoding='ascii') as f:json.dump(v,f,indent=2);f.write('\n')
start=time.monotonic();expected=json.loads((E/'export-sha256.json').read_text())['files'];inputs={}
for n in ['weights.bin','actual_gathered.fp16','fixtures.tsv','reference_GU.f32','candidate_GU.f32']:
 f=E/n;v={'bytes':f.stat().st_size,'sha256':sha(f)};assert v==expected[n],n;inputs[str(f)]=v
with (E/'fixtures.tsv').open() as f:fixtures=[{k:int(v) for k,v in x.items()} for x in csv.DictReader(f,delimiter='\t')]
blob=(E/'weights.bin').read_bytes()
def floats(p):
 a=array.array('f');a.frombytes(p.read_bytes());assert a.itemsize==4;return a
s=floats(E/'reference_GU.f32');g=floats(E/'candidate_GU.f32');assert len(s)==len(g)==2237440
eligible={}
def locator(e,col):
 rb=e['up_offset']//640;off=e['blob_offset']+(col&1)*e['up_offset']+(col//2)*rb
 return off,rb
for e in fixtures:
 for col in range(1280):
  off,rb=locator(e,col);size=rb//10
  eligible[e['index'],col]=all(0<(struct.unpack_from('<H',blob,off+k*size)[0]>>10)&31<31 for k in range(10))
selected={};blocked=[]
def add(e,local,reason):
 token,col=divmod(local,1280);flat=e['row_offset']*1280+local
 if not eligible[e['index'],col]:
  blocked.append({'fixture':e['index'],'token':token,'GU_column':col,'flat_index':flat,'reason':reason,'status':'SKIPPED_SUBNORMAL_SCALE_GPU_CONVERSION_UNRESOLVED'});return
 if flat not in selected:
  off,rb=locator(e,col);selected[flat]={'fixture':e['index'],'layer':e['layer'],'expert':e['expert'],'gu_type':e['gu_type'],'down_type':e['down_type'],'token':token,'GU_column':col,'flat_index':flat,'X_row':e['row_offset']+token,'quantized_offset':off,'quantized_row_bytes':rb,'all_raw_scales_normal':True,'reasons':[]}
 selected[flat]['reasons'].append(reason)
# Preregistered sampling: first ten experts cover each real GU/down format pair.
for e in fixtures[:10]:
 for parity in [0,1]:
  col=next(c for c in range(parity,1280,2) if eligible[e['index'],c]);add(e,col,'gate0_or_up0_first_normal_scale_row_per_format_pair')
add(fixtures[0],0,'first_reported_S42_mismatch')
global_worst=(-1,None,None);relative_picks=[]
for e in fixtures:
 off=e['row_offset']*1280;worst=(-1,None);cancel=(math.inf,None);relative=(-1,None)
 for local in range(e['ne']*1280):
  i=off+local;sv,gv=s[i],g[i];assert math.isfinite(sv) and math.isfinite(gv)
  d=abs(sv-gv)
  if d>global_worst[0]:global_worst=(d,e,local)
  if not eligible[e['index'],local%1280]:continue
  if d>worst[0]:worst=(d,local)
  # Small saved scalar magnitude is a selection proxy; actual A/abs(dot) reported later.
  if d>0 and abs(sv)<cancel[0]:cancel=(abs(sv),local)
  r=d/max(abs(sv),abs(gv),1e-300)
  if r>relative[0]:relative=(r,local)
 if worst[1] is not None:add(e,worst[1],'max_abs_interalgorithm_difference_among_normal_scale_rows_in_fixture')
 if cancel[1] is not None:add(e,cancel[1],'min_abs_nonzero_difference_scalar_output_cancellation_candidate_in_fixture')
 # Largest relative interalgorithm difference also targets cancellation, without calling scalar truth.
 if relative[1] is not None and relative[0]>0:relative_picks.append((relative[0],e,relative[1]))
for _,e,local in sorted(relative_picks,key=lambda x:x[0],reverse=True)[:4]:add(e,local,'top4_fixture_max_relative_interalgorithm_difference_normal_scale')
add(global_worst[1],global_worst[2],'global_max_abs_interalgorithm_difference')
cells=list(selected.values());assert 20<=len(cells)<=64
for i,c in enumerate(cells):c['id']=i
write('selection.json',{'selection_rule':'first normal-scale gate/up per ten format pairs; first mismatch; per-expert max abs difference and minimum abs scalar with nonzero difference; top4 fixture-relative differences; global worst if normal scale','cells':cells,'blocked_cells':blocked,'scope':'not random/representative; cancellation proxy selected from GPU values, exact cancellation measured only after reference','GPU_acceptance_policy':'NONE; old S42 strict failure unchanged'})
with (O/'selection.tsv').open('x',encoding='ascii') as f:
 f.write('id\tfixture\tflat_index\tX_row\tGU_column\tgu_type\tquantized_offset\tquantized_row_bytes\n')
 for c in cells:f.write('\t'.join(str(c[k]) for k in ['id','fixture','flat_index','X_row','GU_column','gu_type','quantized_offset','quantized_row_bytes'])+'\n')
write('reference-method.json',{'method':'pinned ggml to_float -> RN fp16 row -> ggml widen; exact integer products/sum on2^-48 dyadic lattice, then one binary64 rounding','exact_half_lattice':'all finite FP16 values are integer multiples2^-24; widened F32 is exact; int(ldexp(w,24))*int(ldexp(x,24)) summed with Python arbitrary-precision integers','summation_error':'zero for exact integer sum; final rounded binary64 error computed exactly as Fraction, not fitted','accuracy_acceptance_threshold':None,'CPU_GPU_operand_identity':'source-derived conditional RNE normal-finite conversion, not empirically measured; subnormal-scale rows excluded','source_assumptions':['S45 GPU/CPU format formulas and codebook identities','GPU half loads/store conversions preserve normal finite values and round output RNE under recorded precise AOT flags','no empirical byte equality to GPU dequantized planes is claimed']})
B=A/'src-ep/build-sycl-ep/ggml/src/libggml-base.a';assert sha(B)=='293516f3e8a101a7e6b732347094047ea905e40204fa5489c7202e0e5c2d0f8d'
pins={str(p):sha(p) for p in [P/'select_cells.cpp',P/'selected_reference.py',B,A/'llama-pinned/ggml/include/ggml.h',A/'llama-pinned/ggml/src/ggml-quants.c',A/'llama-pinned/ggml/src/ggml-impl.h']}
write('preflight.json',{'inputs':inputs,'source_pins':pins,'outputs_writable':True,'maximum_cells':64})
IMAGE='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2'
body='source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1\nset -euo pipefail\ncommand -v icpx > /out/compiler-path.txt\ntest -w /out\nicpx -O2 -std=c++20 -fp-model=precise -ffp-contract=off -frounding-math -I/work/llama-pinned/ggml/include /prep/select_cells.cpp /work/src-ep/build-sycl-ep/ggml/src/libggml-base.a -pthread -ldl -lm -o /out/select_cells > /out/build.log 2>&1\nldd /out/select_cells > /out/linked-libraries.txt\nexec /out/select_cells /evidence /out/selection.tsv /out\n'
cmd=['docker','run','--rm','--network','none','-u',str(os.getuid())+':'+str(os.getgid()),'-v',str(A)+':/work:ro','-v',str(P)+':/prep:ro','-v',str(O)+':/out','-v',str(E)+':/evidence:ro','--entrypoint','bash',IMAGE,'-c',body]
write('command.json',cmd)
with (O/'row-export.json').open('x') as f,(O/'row-export.stderr').open('x') as e:r=subprocess.run(cmd,stdout=f,stderr=e)
write('row-export.exit.json',{'exit':r.returncode});assert r.returncode==0
assert all(x not in (O/'linked-libraries.txt').read_text() for x in ['sycl','ze_loader','mkl'])
w=floats(O/'selected-w.f32');x=floats(O/'selected-x.f32');assert len(w)==len(x)==len(cells)*2560
out=[]
for c in cells:
 k=c['id']*2560;num=0;amag=0
 for wi,xi in zip(w[k:k+2560],x[k:k+2560]):
  a=math.ldexp(wi,24);b=math.ldexp(xi,24);assert a.is_integer() and b.is_integer()
  prod=int(a)*int(b);num+=prod;amag+=abs(prod)
 exact=Fraction(num,1<<48);ref=math.ldexp(float(num),-48);rf=Fraction(ref);sv=s[c['flat_index']];gv=g[c['flat_index']]
 se=Fraction(sv)-exact;ge=Fraction(gv)-exact
 out.append({**c,'reference_FP64':ref,'reference_FP64_hex':ref.hex(),'exact_dot_numerator_on_2pow_minus48_lattice':num,'exact_sum_abs_products_numerator_on_same_lattice':amag,'final_FP64_round_error':float(abs(rf-exact)),'sum_abs_products':math.ldexp(float(amag),-48),'cancellation_A_over_abs_exact_dot':float(Fraction(amag,abs(num))) if num else None,'scalar':sv,'grouped':gv,'scalar_signed_error_to_exact_dot':float(se),'grouped_signed_error_to_exact_dot':float(ge),'closer_to_exact_CPU_operand_dot':'scalar' if abs(se)<abs(ge) else 'grouped' if abs(ge)<abs(se) else 'equal'})
summary={}
for name,key in [('scalar','scalar_signed_error_to_exact_dot'),('grouped','grouped_signed_error_to_exact_dot')]:
 errs=[v[key] for v in out];norm=math.fsum(v['reference_FP64']**2 for v in out)
 summary[name]={'sample_max_abs_error':max(map(abs,errs)),'sample_RMS_error':math.sqrt(math.fsum(e*e for e in errs)/len(errs)),'sample_relative_L2_error':math.sqrt(math.fsum(e*e for e in errs)/norm) if norm else None}
result={'status':'SELECTED_NORMAL_SCALE_CPU_REFERENCE_OBSERVATIONS_ONLY','cells':len(out),'source_format_pairs':sorted({(v['gu_type'],v['down_type']) for v in out}),'first_mismatch_included':any('first_reported_S42_mismatch' in v['reasons'] for v in out),'summary':summary,'closer_counts':{k:sum(v['closer_to_exact_CPU_operand_dot']==k for v in out) for k in ['scalar','grouped','equal']},'observations':out,'blocked_cells':blocked,'whole_wave_qualified':False,'GPU_operand_identity_measured':False,'GPU_accuracy_policy_or_gate':None,'S42_strict_gate':'FAILED_UNCHANGED','serving_or_speed_qualified':False}
write('reference-observations.json',result)
for n,h in pins.items():assert sha(Path(n))==h
write('closure.json',{'status':'CPU_SELECTED_REFERENCE_COMPLETE_CONDITIONAL_OPERAND_PROVENANCE','elapsed_s':time.monotonic()-start,'GPU_touches':0,'source_unchanged':True,'CPU_only_helper_binary_sha256':sha(O/'select_cells'),'reference_result_sha256':sha(O/'reference-observations.json'),'export_bytes':sum((O/n).stat().st_size for n in ['selected-w.fp16','selected-w.f32','selected-x.f32'])})
print(json.dumps({k:result[k] for k in ['status','cells','first_mismatch_included','summary','closer_counts','blocked_cells','GPU_operand_identity_measured','S42_strict_gate']}))
