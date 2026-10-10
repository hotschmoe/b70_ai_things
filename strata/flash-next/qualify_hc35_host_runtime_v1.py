#!/usr/bin/env python3
"""Reproducible ROOT-only tiny HC host qualification; device/model gates false.
The source agent never executes a compiled helper. Read-only finalized admission
may invoke ldd on the explicitly pinned root-built ELF, not any model/backend.
"""
import argparse,hashlib,json,os,platform,re,struct,subprocess,sys,time
from pathlib import Path
import hc_f32_arithmetic35_host_v2 as host
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PLAN=HERE/'hc35-host-runtime-source-plan-v1.json'
V2_PLAN_SHA='b5eca831ca6725b0674d8e06584e0fbb2c42da551756b9f6d63876ea24cf9aae'
sha,require=host.sha,host.require
read=lambda p:json.loads(Path(p).read_text())
pack=lambda values:struct.pack('<%df'%len(values),*values)

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def signature(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def file_binding(path):
 p=Path(path).resolve();before=signature(p);raw=p.read_bytes();require(signature(p)==before and len(raw)==before[2],'Host dependency changed during read '+str(p))
 return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'stat':before}
def source_binding():
 plan=read(PLAN)
 for path,want in plan['files'].items():require(sha(ROOT/path)==want,'Host supplementary source changed '+path)
 require(sha(HERE/'hc-f32-arithmetic35-host-source-plan-v2.json')==V2_PLAN_SHA,'Exact final V2 plan required')
 host.source_binding();return {'source_plan_sha256':sha(PLAN),'files':plan['files'],'V2_source_plan_sha256':V2_PLAN_SHA}

def build_binding(build_root,helper):
 root=Path(build_root).resolve();helper=Path(helper).resolve();require(helper==root/'hc35-host','Root-owned compiled helper path differs')
 compile=read(root/'compile-receipt.json');analytic=read(root/'analytic-receipt.json');adapter=read(root/'v2-final-adapter-receipt.json')
 require(compile.get('passed') is True and compile['compile_return_code']==0 and compile['source_unchanged'] is True and compile['source_sha256']==sha(HERE/'hc_f32_arithmetic35_host_v1.cpp'),'Fresh unchanged host compile proof required')
 require(compile['helper_sha256']==sha(helper)==analytic['helper_sha256'] and sha(root/'compile-receipt.json')==analytic['compile_receipt_sha256'],'Host compile/analytic/ELF association differs')
 require(analytic.get('passed') is True and len(analytic['rows'])==15 and all(r['passed'] is True for r in analytic['rows']),'Declared15 historical analytic rows required')
 require(adapter.get('passed') is True and len(adapter['rows'])==3 and all(r['passed'] is True and r['receipt']['helper_sha256']==sha(helper) for r in adapter['rows']) and adapter['adapter_sha256']==sha(HERE/'hc_f32_arithmetic35_host_v2.py') and adapter['source_plan_sha256']==V2_PLAN_SHA,'Final V2 adapter history/source differs')
 for receipt in (compile,analytic,adapter):require(receipt.get('GPU_touch') is False and receipt.get('model_payload_read') is False and receipt.get('device_intrinsics_qualified') is False and receipt.get('model_math_qualified') is False,'Host history scope differs')
 require(all(flag in compile['command'][-1] for flag in ('-fno-fast-math','-ffp-contract=off','-frounding-math')),'Host compile strict FP flags absent')
 return {'build_root':str(root),'helper':file_binding(helper),'receipts':{name:file_binding(root/name) for name in ('compile-receipt.json','compile.log','analytic-receipt.json','v2-final-adapter-receipt.json')},'compile_image':compile['image'],'compile_image_does_not_pin_host_runtime':True}

def parse_ldd(text):
 resolved={}
 for line in text.splitlines():
  require('not found' not in line,'Host ELF dependency unresolved')
  match=re.match(r'\s*(\S+)\s+=>\s+(/\S+)\s+\(',line)
  loader=re.match(r'\s*(/\S+)\s+\(',line)
  if match:resolved[match.group(1)]=match.group(2)
  elif loader:resolved['ELF_interpreter']=loader.group(1)
  elif line.strip():require(line.strip().startswith('linux-vdso.so.'),'Unknown ldd dependency format '+line)
 require({'libm.so.6','libc.so.6','libstdc++.so.6','libgcc_s.so.1','ELF_interpreter'}<=set(resolved),'Complete actual ELF dependencies required')
 return resolved

def runtime_binding(helper):
 host.require_gradual_f32()
 result=subprocess.run(['ldd',str(Path(helper).resolve())],check=True,capture_output=True,text=True,timeout=30)
 resolved=parse_ldd(result.stdout);libs={name:file_binding(path) for name,path in sorted(resolved.items())}
 mapped=set()
 for line in Path('/proc/self/maps').read_text().splitlines():
  parts=line.split(maxsplit=5)
  if len(parts)==6 and parts[5].startswith('/') and Path(parts[5]).name=='libm.so.6':mapped.add(parts[5])
 require(len(mapped)==1,'Exact mapped Python ctypes libm path required')
 cpu={}
 for line in Path('/proc/cpuinfo').read_text().splitlines():
  if ':' in line:
   key,value=(x.strip() for x in line.split(':',1))
   if key in ('vendor_id','cpu family','model','model name','stepping','flags'):cpu.setdefault(key,set()).add(value)
 env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT') or k.startswith(('OMP_','MKL_','OPENBLAS_','NUMEXPR_','GLIBC_'))}
 return {'ELF_needed_resolved_libraries':libs,'Python_ctypes_libm':file_binding(next(iter(mapped))),'Python_executable':file_binding(sys.executable),'Python_version':sys.version,'kernel':list(platform.uname()),'CPU_identity':{k:sorted(v) for k,v in sorted(cpu.items())},'CPU_affinity':sorted(os.sched_getaffinity(0)),'execution_environment':env,'rounding':'FE_TONEAREST','behavioral_gradual_input_output_probes_passed':True,'MXCSR_direct_flags_observed':False,'device_intrinsics_qualified':False,'model_math_qualified':False}

def fixture_cases():
 cases=[]
 def add(label,op,a,b,c,expected):cases.append({'label':label,'op':op,'a':a,'b':b,'c':c,'expected':expected})
 add('fused_negative','fma',pack([1+2**-23]),pack([1-2**-23]),pack([-1.]),pack([-2**-46]))
 add('ties_even','fma',pack([1.,1.]),pack([2**-24,2**-24]),pack([1.,1+2**-23]),pack([1.,1+2**-22]))
 tiny=2**-149;add('subnormal','fma',pack([tiny,tiny]),pack([1.,.5]),pack([0.,tiny]),pack([tiny,2*tiny]))
 for n in (320,10240):
  w=[0.]*n;x=[0.]*n;w[-1]=.125;x[-1]=-8.;add('onehot'+str(n),'dot',pack(w),pack(x),None,pack([-1.]))
 x=[0.]*32;x[0]=2**24;x[1]=1.;x[16]=-2**24;add('xor_order_negative','dot',pack([1.]*32),pack(x),None,pack([1.]))
 x=[0.]*96;x[0]=2**24;x[32]=1.;x[64]=-2**24;add('lane_cancellation','dot',pack([1.]*96),pack(x),None,pack([0.]))
 x[32],x[64]=x[64],x[32];add('reorder_negative','dot',pack([1.]*96),pack(x),None,pack([1.]))
 for code in (-128,-127,-1,0,1,127):
  codes=[0]*32;codes[17]=code;x=[0.]*32;x[17]=8.;add('signed'+str(code),'q8dot',struct.pack('<e',.125)+struct.pack('32b',*codes),pack(x),None,pack([float(code)]))
 codes=[0]*32;codes[0]=1;x=[0.]*32;x[0]=2**24;add('half_subnormal','q8dot',struct.pack('<e',2**-24)+struct.pack('32b',*codes),pack(x),None,pack([1.]))
 require(len(cases)==15 and len({c['label'] for c in cases})==15,'Declared exact15 tiny analytic cases differ');return cases

def preserve_fixture(root,case):
 directory=root/'fixtures'/case['label'];directory.mkdir(parents=True,exist_ok=False);row={'label':case['label'],'op':case['op'],'files':{}}
 for key in ('a','b','c','expected'):
  if case[key] is not None:
   path=directory/(key+'.bin');path.write_bytes(case[key]);row['files'][key]=file_binding(path)
 return row

def validate_fixture(root,row,case):
 require(row['label']==case['label'] and row['op']==case['op'],'Fixture identity differs')
 require(set(row['files'])=={k for k in ('a','b','c','expected') if case[k] is not None},'Fixture file roster differs')
 for key,binding in row['files'].items():
  path=Path(binding['path']);require(path.resolve()==(root/'fixtures'/case['label']/(key+'.bin')).resolve() and not path.is_symlink(),'Fixture escaped immutable run')
  require(file_binding(path)==binding and path.read_bytes()==case[key],'Tiny fixture content differs from frozen known answer')

def finalized_binding(run_root,helper,build_root):
 root=Path(run_root).resolve();report=read(root/'report.json');require(report.get('passed') is True and report.get('errors')==[],'Final supplementary host proof failed')
 require(report['producer_sha256']==sha(Path(__file__)) and report['source_binding']==source_binding(),'Final producer/source binding differs')
 require(report['build_binding']==build_binding(build_root,helper),'Current compile/history/library proof changed')
 current=runtime_binding(helper);require(report['runtime_pre']==report['runtime_post']==current,'Actual host runtime dependencies/environment changed')
 cases=fixture_cases();require(len(report['analytic'])==15 and len(report['adapter'])==3,'Final actual15+3 roster differs')
 for row,case in zip(report['analytic'],cases):
  validate_fixture(root,row['fixture'],case);require(row['passed'] is True and row['output_hex']==case['expected'].hex(),'Known-answer analytic bytes differ')
  require(row['command']==[str(Path(helper).resolve()),case['op'],str(len(case['b'])//4),row['fixture']['files']['a']['path'],row['fixture']['files']['b']['path'],row['fixture']['files']['c']['path'] if case['c'] is not None else '-',str(root/'outputs'/(case['label']+'.bin')),host.TAG],'Exact analytic command differs')
  receipt=row['receipt'];require(json.loads(row['stdout'])==receipt and row['return_code']==0 and receipt.get('schema')==1 and receipt.get('op')==case['op'] and receipt.get('elements')==len(case['b'])//4 and receipt.get('output_floats')==len(case['expected'])//4 and receipt.get('rounding')=='FE_TONEAREST' and receipt.get('flush_to_zero') is False and receipt.get('denormals_are_zero') is False and receipt.get('device_intrinsics_qualified') is False and receipt.get('model_math_qualified') is False,'Actual analytic producer receipt differs')
  require(file_binding(root/'outputs'/(case['label']+'.bin'))==row['output'] and bytes.fromhex(row['output_hex'])==(root/'outputs'/(case['label']+'.bin')).read_bytes(),'Preserved actual output changed')
 for row,label in zip(report['adapter'],('fused_negative','xor_order_negative','signed-128')):
  case=next(c for c in cases if c['label']==label);require(row['label']==label and row['passed'] is True and row['output_hex']==case['expected'].hex() and row['receipt']['helper_sha256']==sha(helper) and row['receipt']['input_sha256']==[hashlib.sha256(case[k]).hexdigest() for k in ('a','b','c') if case[k] is not None] and row['receipt']['output_sha256']==hashlib.sha256(case['expected']).hexdigest() and row['receipt']['host_gradual_input_output_probes_passed'] is True,'Final actual adapter result differs')
 require(report.get('device_intrinsics_qualified') is False and report.get('full_HC_qualified') is False and report.get('model_math_qualified') is False and report.get('GPU_touch') is False and report.get('model_payload_read') is False,'Supplementary host scope differs')
 return {'report_sha256':sha(root/'report.json'),'source_binding':report['source_binding'],'build_binding':report['build_binding'],'current_host_runtime':current,'tiny_known_answer_cases':15,'actual_adapter_cases':3,'device_intrinsics_qualified':False,'full_HC_qualified':False,'model_math_qualified':False}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for name in ('helper','build-root','output'):ap.add_argument('--'+name,type=Path,required=True)
 a=ap.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);(root/'outputs').mkdir()
 report={'schema':1,'passed':False,'errors':[],'producer_sha256':sha(Path(__file__)),'exact_argv':list(sys.argv),'Python_executable':str(Path(sys.executable).resolve()),'started_epoch':time.time(),'GPU_touch':False,'model_payload_read':False,'device_intrinsics_qualified':False,'full_HC_qualified':False,'model_math_qualified':False,'analytic':[],'adapter':[]}
 try:
  report['source_binding']=source_binding();report['build_binding']=build_binding(a.build_root,a.helper);report['runtime_pre']=runtime_binding(a.helper);write(root/'report.json',report)
  cases=fixture_cases()
  for case in cases:
   fixture=preserve_fixture(root,case);output=root/'outputs'/(case['label']+'.bin');cmd=[str(a.helper.resolve()),case['op'],str(len(case['b'])//4),fixture['files']['a']['path'],fixture['files']['b']['path'],fixture['files']['c']['path'] if case['c'] is not None else '-',str(output),host.TAG]
   result=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=60);receipt=json.loads(result.stdout);raw=output.read_bytes();require(raw==case['expected'],'Actual helper known answer differs '+case['label']);require(receipt['rounding']=='FE_TONEAREST' and receipt['flush_to_zero'] is False and receipt['denormals_are_zero'] is False and receipt['op']==case['op'] and receipt['elements']==len(case['b'])//4 and receipt['device_intrinsics_qualified'] is False and receipt['model_math_qualified'] is False,'Actual helper rounding/scope differs')
   report['analytic'].append({'fixture':fixture,'command':cmd,'stdout':result.stdout,'stderr':result.stderr,'return_code':result.returncode,'receipt':receipt,'output':file_binding(output),'output_hex':raw.hex(),'passed':True});write(root/'report.json',report)
  for label in ('fused_negative','xor_order_negative','signed-128'):
   case=next(c for c in cases if c['label']==label);raw,receipt=host.run_helper(a.helper,sha(a.helper),case['op'],case['a'],case['b'],case['c']);require(raw==case['expected'],'Actual V2 adapter known answer differs '+label);report['adapter'].append({'label':label,'passed':True,'output_hex':raw.hex(),'receipt':receipt});write(root/'report.json',report)
  report['runtime_post']=runtime_binding(a.helper);require(report['runtime_post']==report['runtime_pre'],'Host runtime libraries/environment changed during controls');require(report['build_binding']==build_binding(a.build_root,a.helper) and report['source_binding']==source_binding(),'Source/ELF/history changed during controls');report['passed']=True
 except Exception as error:report['errors'].append(str(error))
 report['finished_epoch']=time.time();write(root/'report.json',report)
 if report['passed']:
  try:finalized_binding(root,a.helper,a.build_root)
  except Exception as error:report['passed']=False;report['errors'].append('Final read-only admission: '+str(error));write(root/'report.json',report)
 print(json.dumps({'passed':report['passed'],'errors':report['errors'],'report':str(root/'report.json'),'device_intrinsics_qualified':False,'model_math_qualified':False}));return int(not report['passed'])
if __name__=='__main__':raise SystemExit(main())
