#!/usr/bin/env python3
"""CPU-only identity/output closure. Called only inside an authorized cpu-run."""
from pathlib import Path
import hashlib,json,sys
mode,out,inputs,prep,build=sys.argv[1],*map(Path,sys.argv[2:])
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def write(p,data):
 with p.open('x',encoding='ascii')as f:json.dump(data,f,indent=2);f.write('\n')
expected=json.loads((prep/'input-pins.json').read_text())
actual={n:record(inputs/n)for n in expected};need(actual==expected,'frozen compact input hashes')
source={}
for l in(prep/'expected-container.sha256').read_text().splitlines():
 h,n=l.split(maxsplit=1);need(sha(Path(n))==h,n);source[n]=h
records={}
for l in(build/'build-sha256.txt').read_text().splitlines():
 h,n=l.split(maxsplit=1);records[n]=h
need(records['/out/reference_s80']==sha(build/'reference_s80'),'actual CPU executable hash')
for n in ['/prep/reference_s80.cpp','/prep/reference_bound.hpp']:need(records[n]==source[n],'source/build hash '+n)
need(not any(x in (build/'linked-libraries.txt').read_text().lower()for x in ['libsycl','libze_loader','libopencl']),'CPU-only dynamic linkage')
need(json.loads((build/'image.json').read_text())[0]['Id']=='sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2','pinned image')
frozen={}
for l in(prep/'expected-frozen.sha256').read_text().splitlines():
 h,n=l.split(maxsplit=1);target=build/n.removeprefix('/out/');need(sha(target)==h,'frozen CPU import '+n);frozen[n]=h
for l in(build/'MKL-files.sha256').read_text().splitlines():
 h,n=l.split(maxsplit=1);need(sha(Path(n))==h,'MKL runtime file '+n)
identity={'frozen_CPU_imports':frozen,'inputs':actual,'source_pins':source,'binary_sha256':records['/out/reference_s80'],'image':'sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2','MKL_build_files':(build/'MKL-files.sha256').read_text(),'compiler':(build/'compiler-version.txt').read_text(),'CPU_only':True}
if mode=='pre':
 write(out/'preflight.json',identity)
else:
 need(mode=='post','mode')
 need(json.loads((out/'preflight.json').read_text())==identity,'pre/post CPU source/input/build identity')
 ref=out/'reference';core=json.loads((ref/'reference-core.json').read_text());schema=json.loads((prep/'schema.json').read_text())
 need(core['status']=='PASS_CPU_REFERENCE_ONLY_GPU_OPERANDS_UNMEASURED','CPU core closure')
 need(core['values']==2237440 and core['threads']==4 and core['max_rss_kib']<1024*1024,'full scope/resource guard')
 for k in ['FE_TONEAREST','selected36_exact_numerators','selected36_E64_conformance','zero_DGEMM_and_zero_negative_control','outside_bound_float_negative_control']:need(core[k]is True,k)
 need(core['FTZ_DAZ']is False and core['GPU_plane_identity_measured']is False and core['ideal_H_down_reference']is False,'scope flags')
 for key,total in [('GU_half_classes',52428800),('down_half_classes',26214400),('input_half_classes',4474880),('raw_GU_scale_classes',204800)]:need(sum(core[key].values())==total,key+' count')
 for key in ['GU_half_classes','input_half_classes']:need(core[key]['subnormal']==core[key]['inf']==core[key]['nan']==0,key+' arithmetic model')
 need(core['raw_GU_scale_classes']['subnormal']==577,'retained raw scale enumeration')
 data={}
 for name,d in schema['files'].items():
  need((ref/name).stat().st_size==d['bytes'],'output size '+name);data[name]=record(ref/name)
 # Complete exact sign-bit transformation, with C-level bytes.translate (no NumPy).
 original=(inputs/'actual_gathered.fp16').read_bytes();expected_changed=bytearray(original)
 expected_changed[1::2]=original[1::2].translate(bytes(x^128 for x in range(256)))
 need((ref/'actual-gathered-changed.fp16').read_bytes()==expected_changed,'full changed half sign-bit identity')
 for name in ['reference-core.json','mkl-version.txt']:data[name]=record(ref/name)
 result={'status':'PASS_CPU_FULL_GU_REFERENCE_GPU_IDENTITY_PENDING','files':data,**identity,'schema':schema,'numeric':core,'MKL_runtime_version':(ref/'mkl-version.txt').read_text().strip(),'changed_input_full_sign_bit_identity':True,'GPU_operand_identity_measured':False,'ideal_H_down_reference':False,'bounds_meaning':'Predeclared S77 empirical full-GU arithmetic target; ordinary FP32 envelope plus explicit conventional FP64 reference uncertainty, not a vendor-guaranteed arithmetic theorem.','down_CPU_plane_role':'diagnostic identity only; no down or H ideal reference'}
 write(ref/'reference.json',result)
 print(json.dumps({'status':result['status'],'data_bytes':sum(x['bytes']for x in data.values()),'GPU_operand_identity_measured':False}))
