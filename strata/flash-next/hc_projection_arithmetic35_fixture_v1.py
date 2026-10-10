"""Synthetic projection corpus/ROOT-only host export and read-only GPU recollection.
No original model inputs or tolerance gate. Source preparation does not execute
host helpers, SYCL, Docker or GPU work; root owns those separate receipts.
"""
import argparse,hashlib,json,re,struct
from pathlib import Path
import hc_f32_arithmetic35_host_v2 as arithmetic
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sha,require=arithmetic.sha,arithmetic.require
pack=lambda values:struct.pack('<%df'%len(values),*values)


def fixtures():
 cases=[]
 def add(name,kind,k,m,t,w,x,negative_of=None):
  cases.append({'id':name,'type':kind,'k':k,'m':m,'t':t,'weights':w,'input':pack(x),'negative_of':negative_of})
 x=[0.]*32;x[0]=2**24;x[1]=1.;x[16]=-2**24;add('xor_order','F32',32,1,1,pack([1.]*32),x)
 wrong=x.copy();wrong[1],wrong[16]=wrong[16],wrong[1];add('xor_order_negative','F32',32,1,1,pack([1.]*32),wrong,'xor_order')
 x=[0.]*96;x[0]=2**24;x[32]=1.;x[64]=-2**24;add('lane_cancel','F32',96,1,1,pack([1.]*96),x)
 x=[0.]*64;x[0]=1.;x[32]=1-2**-23;w=[0.]*64;w[0]=-1.;w[32]=1+2**-23;add('fused_cancel','F32',64,1,1,pack(w),x)
 x=[0.]*64;x[0]=1.;x[32]=2**-24;w=[0.]*128;w[0]=1.;w[32]=1.;w[64]=1+2**-23;w[96]=1.;add('round_midpoints','F32',64,2,1,pack(w),x)
 x=[0.]*32;x[17]=8.;codes=[0]*32;codes[17]=-128;codes2=[0]*32;codes2[17]=127;w=struct.pack('<e',.125)+struct.pack('32b',*codes)+struct.pack('<e',.125)+struct.pack('32b',*codes2);add('q8_signed','Q8_0',32,2,1,w,x)
 bad=bytearray(w);bad[19]=129;add('q8_weight_negative','Q8_0',32,2,1,bytes(bad),x,'q8_signed')
 x=[0.]*32;x[0]=2**24;codes=[0]*32;codes[0]=1;add('q8_half_subnormal','Q8_0',32,1,1,struct.pack('<e',2**-24)+struct.pack('32b',*codes),x)
 x=[0.]*32;x[0]=2**-149;add('f32_subnormal','F32',32,1,1,pack([1.]*32),x)
 # Preregistered LCG seed3535: dyadic synthetic inputs, independent of GPU data.
 state=3535
 def next_int():
  nonlocal state
  state=(1664525*state+1013904223)&0xffffffff;return state
 for kind,k in (('F32',320),('F32',10240),('Q8_0',320),('Q8_0',10240)):
  m,t=3,2;x=[arithmetic.f32((next_int()%8193-4096)/8192) for _ in range(k*t)]
  if kind=='F32':w=pack([arithmetic.f32((next_int()%63-31)/32) for _ in range(k*m)])
  else:
   blocks=[]
   for _ in range(k*m//32):blocks.append(struct.pack('<e',(.03125,.0625,.125,.25)[next_int()%4])+struct.pack('32b',*[next_int()%255-127 for _ in range(32)]))
   w=b''.join(blocks)
  add('heldout_'+kind.lower()+'_'+str(k),kind,k,m,t,w,x)
 return cases


def source_binding():
 plan=json.loads((HERE/'hc-projection-arithmetic35-fixture-source-plan-v1.json').read_text())
 for name,digest in plan['files'].items():require(sha(ROOT/name)==digest,'Frozen GPU projection fixture source changed '+name)
 for row in plan['source35_dependencies']:require(sha(row['path'])==row['sha256'],'Current source35 SDK/library changed '+row['path'])
 return plan


def host_binding(host_run,helper,build_root):
 from qualify_hc35_host_runtime_v1 import finalized_binding
 return finalized_binding(host_run,helper,build_root)


def prepare(output,host_run,helper,build_root):
 plan=source_binding();before=host_binding(host_run,helper,build_root);output=Path(output).resolve();require(not output.exists(),'New synthetic input package required');output.mkdir();(output/'inputs').mkdir();(output/'expected').mkdir();rows=[];tsv=['HC35_PROJECTION_INPUTS_V1']
 for case in fixtures():
  row_bytes=case['k']*4 if case['type']=='F32' else case['k']//32*34;expected=[];row_proofs=[]
  for token in range(case['t']):
   inputs=case['input'][token*case['k']*4:(token+1)*case['k']*4]
   for row in range(case['m']):
    weights=case['weights'][row*row_bytes:(row+1)*row_bytes];raw,receipt=arithmetic.run_helper(helper,arithmetic.sha(helper),'dot' if case['type']=='F32' else 'q8dot',weights,inputs);expected.append(raw);row_proofs.append(receipt)
  files={}
  for role,raw in [('weights',case['weights']),('input',case['input']),('expected',b''.join(expected))]:
   path=(output/('expected' if role=='expected' else 'inputs'))/(case['id']+'.'+role+('.bin' if role!='expected' else '.f32'));path.write_bytes(raw);files[role]={'path':str(path),'sha256':sha(path),'bytes':len(raw)}
  rows.append({**{k:case[k] for k in ('id','type','k','m','t','negative_of')},'files':files,'host_row_proofs':row_proofs});tsv.append('%s %s %d %d %d'%(case['id'],case['type'],case['k'],case['m'],case['t']))
 (output/'inputs/cases.tsv').write_text('\n'.join(tsv)+'\n',encoding='ascii');after=host_binding(host_run,helper,build_root);require(before==after,'Host source/runtime/receipts changed during expected export')
 require(all(rows[i]['files']['expected']['sha256']!=next(row for row in rows if row['id']==rows[i]['negative_of'])['files']['expected']['sha256'] for i in range(len(rows)) if rows[i]['negative_of']),'Preregistered negatives not effective')
 manifest={'schema':1,'synthetic_only':True,'source_plan_sha256':sha(HERE/'hc-projection-arithmetic35-fixture-source-plan-v1.json'),'host_run_root':str(Path(host_run).resolve()),'host_helper':str(Path(helper).resolve()),'host_build_root':str(Path(build_root).resolve()),'host_finalized_binding':before,'cases':rows,'cases_tsv_sha256':sha(output/'inputs/cases.tsv'),'model_math_qualified':False,'device_intrinsics_qualified':False};(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='ascii');return manifest


def input_binding(directory):
 directory=Path(directory).resolve();source_binding();manifest=json.loads((directory/'manifest.json').read_text());require(manifest['synthetic_only'] is True and manifest['model_math_qualified'] is False and manifest['source_plan_sha256']==sha(HERE/'hc-projection-arithmetic35-fixture-source-plan-v1.json'),'Synthetic source manifest scope differs')
 current=host_binding(Path(manifest['host_run_root']),Path(manifest['host_helper']),Path(manifest['host_build_root']));require(current==manifest['host_finalized_binding'],'Fresh finalized host receipt/library/source association changed')
 specs=fixtures();require(len(manifest['cases'])==len(specs) and [row['id'] for row in manifest['cases']]==[case['id'] for case in specs],'Exact preregistered corpus roster differs')
 for row,spec in zip(manifest['cases'],specs):
  require(all(row[k]==spec[k] for k in ('id','type','k','m','t','negative_of')),'Preregistered synthetic geometry/policy changed')
  for role in ('weights','input','expected'):
   field=row['files'][role];path=Path(field['path']);require(path.resolve().parent==directory/('expected' if role=='expected' else 'inputs') and not path.is_symlink() and sha(path)==field['sha256'] and path.stat().st_size==field['bytes'],'Current synthetic raw path/hash/extent changed');raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==field['sha256'],'Synthetic input changed during read')
   if role!='expected':require(raw==spec[role],'Native/model values substituted for synthetic preregistered operands')
  require(row['files']['expected']['bytes']==row['m']*row['t']*4 and len(row['host_row_proofs'])==row['m']*row['t'],'Complete host expected row evidence absent')
  for index,proof in enumerate(row['host_row_proofs']):
   token,weight_row=divmod(index,row['m']);row_bytes=row['k']*4 if row['type']=='F32' else row['k']//32*34;weights=spec['weights'][weight_row*row_bytes:(weight_row+1)*row_bytes];inputs=spec['input'][token*row['k']*4:(token+1)*row['k']*4];expected=Path(row['files']['expected']['path']).read_bytes()[index*4:(index+1)*4];independent=arithmetic.dot32_f32(weights,inputs) if row['type']=='F32' else arithmetic.q8dot32_f32(weights,inputs);require(expected==independent,'Expected row contradicts independently recollected validated F32 source schedule');require(proof['helper_sha256']==current['build_binding']['helper']['sha256'] and proof['input_sha256']==[hashlib.sha256(v).hexdigest() for v in (weights,inputs)] and proof['output_sha256']==hashlib.sha256(expected).hexdigest() and proof.get('host_gradual_input_output_probes_passed') is True and proof.get('model_math_qualified') is False,'Complete expected row/realhost helper association differs')
 tsv='HC35_PROJECTION_INPUTS_V1\n'+'\n'.join('%s %s %d %d %d'%(c['id'],c['type'],c['k'],c['m'],c['t']) for c in specs)+'\n';require((directory/'inputs/cases.tsv').read_text()==tsv and sha(directory/'inputs/cases.tsv')==manifest['cases_tsv_sha256'],'Synthetic execution TSV binding differs')
 return manifest


def collect(inputs,raw_root,log):
 inputs=Path(inputs);raw_root=Path(raw_root).resolve();manifest=input_binding(inputs);text=Path(log).read_text();matches=re.findall(r'^HC35_PROJECTION_CONFIG synthetic=1 device=(.+) queue=(0x[0-9a-f]+) ze_context=(0x[0-9a-f]+) ze_device=(0x[0-9a-f]+) in_order=1 device_intrinsics_qualified=0 model_math_qualified=0$',text,re.M);require(len(matches)==1,'Actual queue/device/context config marker absent/duplicate');device=re.findall(r'^HC35_PROJECTION_DEVICE backend=level_zero vendor=(.+) driver=(.+) affinity=0 selector=level_zero:gpu physical_card_parent_mapping_required=1$',text,re.M);require(len(device)==1,'Actual backend/vendor/driver/selector marker absent/duplicate');results=[]
 for row in manifest['cases']:
  path=raw_root/(row['id']+'.gpu.f32');require(not path.is_symlink() and path.is_file() and path.stat().st_size==row['m']*row['t']*4,'Actual complete GPU raw row extent differs');raw=path.read_bytes();arithmetic.words(raw);expected=Path(row['files']['expected']['path']).read_bytes();require(len(raw)==len(expected),'GPU/host complete extent differs');mark='HC35_PROJECTION_ROW case=%s type=%s k=%d m=%d t=%d bytes=%d'%(row['id'],row['type'],row['k'],row['m'],row['t'],len(raw));require(text.splitlines().count(mark)==1,'Actual raw producer case marker absent/duplicate');differences=[i for i in range(len(raw)//4) if raw[i*4:i*4+4]!=expected[i*4:i*4+4]];results.append({'id':row['id'],'negative_of':row['negative_of'],'gpu_raw_sha256':hashlib.sha256(raw).hexdigest(),'expected_sha256':hashlib.sha256(expected).hexdigest(),'words':len(raw)//4,'differing_words':len(differences),'first_differing_word':differences[0] if differences else None,'bitwise_equal':not differences,'device_subnormal_behavior_qualified':False})
 require(set(path.name for path in raw_root.iterdir())=={row['id']+'.gpu.f32' for row in manifest['cases']},'Foreign/missing raw GPU output files');terminal='HC35_PROJECTION_RESULT cases=%d execution_completed=1 all_owned_allocations_freed=1 numerical_comparison_done=0 device_intrinsics_qualified=0 model_math_qualified=0'%len(results);require(text.splitlines().count(terminal)==1 and 'HC35_PROJECTION_ERROR' not in text,'Actual leaf terminal/error differs')
 negatives=[]
 for result in results:
  if result['negative_of']:
   normal=next(row for row in results if row['id']==result['negative_of']);negatives.append({'id':result['id'],'different_from_control':result['gpu_raw_sha256']!=normal['gpu_raw_sha256']})
 return {'schema':1,'numeric_bitwise_passed':all(row['bitwise_equal'] for row in results) and all(row['different_from_control'] for row in negatives),'cases':results,'negative_controls':negatives,'input_manifest_sha256':sha(inputs/'manifest.json'),'host_finalized_binding':manifest['host_finalized_binding'],'log_sha256':sha(log),'queue_device_source_marker':matches[0],'device_vendor_driver_selector_marker':device[0],'physical_device_identity_independently_qualified':False,'actual_GPU_execution_qualified':False,'parent_health_lease_compile_USM_teardown_source4_qualified':False,'device_intrinsics_qualified':False,'model_math_qualified':False,'tolerance_gate':None,'scope':'raw synthetic public-projection comparison only; genuine fresh leaf build/owned lifecycle still mandatory; mismatch fails without tolerance relaxation'}


def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True);a=sub.add_parser('prepare')
 for n in ('output','host-run-root','host-helper','host-build-root'):a.add_argument('--'+n,type=Path,required=True)
 a=sub.add_parser('collect')
 for n in ('inputs','raw','log','output'):a.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args()
 if a.mode=='prepare':result=prepare(a.output,a.host_run_root,a.host_helper,a.host_build_root)
 else:require(not a.output.exists(),'Preserve prior collector receipt');result=collect(a.inputs,a.raw,a.log);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'synthetic_only':True,'numeric_bitwise_passed':result.get('numeric_bitwise_passed'),'actual_GPU_execution_qualified':False}));return int(a.mode=='collect' and not result['numeric_bitwise_passed'])
if __name__=='__main__':raise SystemExit(main())
