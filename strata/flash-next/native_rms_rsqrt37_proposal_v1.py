"""Source/CPU proposal and saved-independent-input fixture admission, root runtime only."""
import json,hashlib,math,struct
from pathlib import Path
import numpy as np
import native_rms_rsqrt37_hypotheses_v1 as mathref
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'native-rms-rsqrt37-source-plan-v1.json'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T084541Z-zqrph_y1')
OWN=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/owned-first-hc-fidelity-v2-conditional-run1')
OWN_REPORT_SHA='076c1e1069de311ee3effa39988f3a78875d8d618804bedc404d31196cec8081'
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
require=mathref.hc.require
read=lambda p:json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding():
 p=read(PLAN)
 for n,w in p['files'].items():require(sha(ROOT/n)==w,'RMS37 proposal source changed '+n)
 for n,w in p['actual_SDK37_bindings'].items():require(sha(n)==w,'Current SDK37 compiled/source/builder binding changed '+n)
 return {'source_plan_sha256':sha(PLAN),'files':p['files'],'actual_SDK37_bindings':p['actual_SDK37_bindings']}
def builder_flags():
 ninja=(SDK/'build/build.ninja').read_text();needle='build CMakeFiles/strata_kernels.dir/src/kernels/hc_native_composition.cpp.o:';require(ninja.count(needle)==1,'Actual HC Ninja compile target missing/ambiguous');block=ninja.split(needle,1)[1].split('\n\n',1)[0];fields={line.strip().split(' = ',1)[0]:line.strip().split(' = ',1)[1] for line in block.splitlines() if ' = ' in line};require(fields['FLAGS']=='-O3 -DNDEBUG -std=c++20 -fsycl -fsycl-default-sub-group-size=32 -fsycl-device-code-split=per_kernel -fp-model=precise -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Wall -Wextra','Actual HC compile flags changed');require(fields['DEFINES']=='-DSTRATA_SYCL_Q8_HC_BUILT=1 -DSTRATA_VERSION=\\"0.1.41\\"','Actual HC compiled defines changed');rules=(SDK/'build/CMakeFiles/rules.ninja').read_text();require('/opt/intel/oneapi/compiler/2026.1/bin/icpx $DEFINES $INCLUDES $FLAGS' in rules,'Actual SDK builder command changed')
 link=ninja.split('build strata: ',1)[1].split('\n\n',1)[0];link_flags=next(line.split(' = ',1)[1] for line in link.splitlines() if line.strip().startswith('LINK_FLAGS = '));require(link_flags=='-fsycl -Xsycl-target-backend=spir64 -cl-fp32-correctly-rounded-divide-sqrt -fsycl-device-code-split=per_kernel -qmkl=sequential','Actual production SDK device-link flags changed');fields['production_device_LINK_FLAGS']=link_flags;fields['leaf_backend_flag_is_device_link_authority_not_object_FLAG']=True
 return fields

def compile_argv():
 builder_flags();return ['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-DSTRATA_SYCL_Q8_HC_BUILT=1','-DSTRATA_VERSION="0.1.41"','-I/sdk/source/third_party/ggml','-I/sdk/source/sycl/include','-I/sdk/source/include','/leaf/native_rms_rsqrt37_gpu_v1.cpp','/sdk/build/libstrata_kernels.a','/sdk/build/libstrata_core.a','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/native-rms-rsqrt37']

def saved_original_input():
 require(sha(OWN/'report.json')==OWN_REPORT_SHA,'Exact successful independent firstHC report required');r=read(OWN/'report.json');require(r['errors']==[] and r['provenance']['ids']==[248045] and r['provenance']['captured_inputs_used'] is False and r['provenance']['captured_state_or_routes_used'] is False and r['provenance']['full_model_math_qualified'] is False,'Independent firstHC embedding provenance differs')
 identity=r['post_original_identity'];require(identity['complete_four_publisher_hashes_verified'] is True and identity['current_stat_verified'] is True and sha(identity['path'])==identity['sha256'],'Saved original postCPU full4 association differs');proof=read(identity['path']);lock=read(HERE/'model-lock.json');require(proof['lock_sha256']==sha(HERE/'model-lock.json') and proof['model_revision']==lock['revision'],'Saved original publisher lock/revision differs')
 for row in proof['rows']:
  stat=Path(row['path']).stat();require(row['passed'] is True and row['stat_before']==row['stat_after']==[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns],'Current saved original shard stat changed; no payload reread')
 require(proof['passed'] is True and len(proof['rows'])==4 and proof['started']>=r['computation_terminal_epoch'] and r['initial_pages']['passed'] is True and r['final_pages']['passed'] is True,'Saved independent-input source/full4/pages chronology failed')
 arrays={}
 for name,shape in [('embedding_owned',[2560]),('residual_owned',[4,2560])]:
  b=r['owned_arrays'][name];path=OWN/('own-'+name+'.f32');require(b['path']==str(path) and b['shape']==shape and b['encoding']=='LE_F32' and not path.is_symlink() and sha(path)==b['sha256'] and path.stat().st_size==b['bytes']==4*np.prod(shape),'Exact own embedding/residual field association differs');arrays[name]=np.frombuffer(path.read_bytes(),dtype='<f4').reshape(shape).copy();require(np.isfinite(arrays[name]).all(),'Finite original own input required')
 require(np.array_equal(arrays['residual_owned'],np.broadcast_to(arrays['embedding_owned'],(4,2560))),'Own residual must broadcast own original embedding');return arrays['residual_owned'],{'original_report_sha256':OWN_REPORT_SHA,'original_post_full4':identity,'input_sha256':sha(OWN/'own-residual_owned.f32'),'captured_native_operand_used':False,'fresh_model_payload_read':False,'synthetic_norm':True,'synthetic_down_up':True,'full_model_math_qualified':False}

def prepare(output):
 source=source_binding();flags=builder_flags();r,binding=saved_original_input();variants=mathref.variants(r);out=Path(output);require(not out.exists(),'New RMS fixture directory required');out.mkdir(parents=True);(out/'residual.f32').write_bytes(r.tobytes());hyp={}
 for label,fields in variants.items():
  hyp[label]={}
  for name,raw in fields.items():
   path=out/(label+'-'+name+'.f32');path.write_bytes(raw);hyp[label][name]={'path':str(path.resolve()),'sha256':sha(path),'bytes':len(raw),'encoding':'LE_F32[4]'}
 record={'schema':1,'source_binding':source,'original_saved_input_binding':binding,'builder_flags':flags,'compile_argv':compile_argv(),'hypotheses':hyp,'owned_residual_path':str((out/'residual.f32').resolve()),'full_model_math_qualified':False,'actual_GPU_execution':False,'actual_RMS_observed':False};(out/'fixture.json').write_text(json.dumps(record,indent=2,ensure_ascii=True)+'\n');return record

def compare_routes(directory):
 directory=Path(directory);fields={'actual_hc_rs':16,'actual_hc_xn_normones':40960,'separate_square_sum_shadow':16,'separate_argument_shadow':16};rows=[]
 for route in range(3):
  row={}
  for name,n in fields.items():
   p=directory/('r'+str(route))/(name+'.f32');raw=p.read_bytes();require(len(raw)==n and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Actual finite complete native RMS output differs');row[name]=raw
  rows.append(row)
 require(rows[0]==rows[1]==rows[2],'Actual direct/samegraph replay raw output mismatch');require(np.all(np.frombuffer(rows[0]['actual_hc_rs'],dtype='<f4')>0),'Actual positive RMS reciprocal required');return {'routes':3,'bitwise_direct_graph_repeat_equal':True,'native_HC_RS_observed':True,'shadow_not_internal_argument_witness':True,'full_model_math_qualified':False}
