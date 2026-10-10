"""Canonical synthetic HC corpus and read-only observations, no model oracle."""
import hashlib,json,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE_PLAN=HERE/'hc-composition-arithmetic35-fixture-source-plan-v1.json'
N=2560;H=4;D=10240;L=320
FIELDS={'rs':H,'inject':H,'norm_square_sum_shadow':H,'norm_argument_shadow':H,'rs_shadow':H,'xn':D,'gate':D,'residual_after_pending':D,'standalone_write':D,'xn_shadow':D,'exp_gate_shadow':D,'post_silu_low':L,'pre_silu_secondary_projection':L,'exp_low_shadow':L,'silu_shadow':L,'mixed':N,'mix_separate_shadow':N,'mix_fma_shadow':N,'fp_control_shadow':2}
ACTUAL={'rs','inject','xn','gate','residual_after_pending','standalone_write','post_silu_low','mixed'}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,row):Path(path).write_text(json.dumps(row,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def fixtures():return [{'id':'analytic_t1','t':1,'seed':0},{'id':'analytic_t2','t':2,'seed':0},{'id':'heldout_t1','t':1,'seed':177},{'id':'heldout_t2','t':2,'seed':319}]
def corpus_counts():
 cases=fixtures();return {'cases':len(cases),'frames':len(cases)*9,'output_floats':sum(c['t'] for c in cases)*9*sum(FIELDS.values()),'fields_per_frame':len(FIELDS),'negative_cases':2}
def count_binding(plan):
 require(plan['computed_corpus_counts']==corpus_counts(),'Canonical HC composition count differs before health/launch');return corpus_counts()
def source_binding():
 plan=json.loads(SOURCE_PLAN.read_text())
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Frozen HC composition fixture changed '+name)
 return plan
def host_binding():
 # Host library identity for later intrinsic-candidate analysis, not a device
 # exp/rsqrt accuracy proof. This admission never executes the compiled helper.
 from conditional_hc_projection35_v1 import host_admission,HOST_ROOT,HOST_RUNTIME_ROOT
 return host_admission(HOST_ROOT/'hc35-host',HOST_ROOT,HOST_RUNTIME_ROOT)
def f32(values):return np.asarray(values,dtype='<f4').tobytes()
def q8(rows,cols,seed):
 blocks=rows*(cols//32);index=np.arange(blocks*32,dtype=np.int64).reshape(blocks,32);codes=np.zeros_like(index,dtype=np.int8) if seed==0 else (((index*17+seed)%7)-3).astype(np.int8);scale=np.full(blocks,2**-12,dtype='<f2')
 if seed:scale[np.arange(blocks)%5==0]=2**-24
 raw=np.empty((blocks,34),dtype=np.uint8);raw[:,:2]=scale.view(np.uint8).reshape(blocks,2);raw[:,2:]=codes.view(np.uint8);return raw.tobytes()
def case_data(case):
 t,seed=case['t'],case['seed'];require(case in fixtures(),'Only canonical synthetic case admitted')
 if not seed:
  residual=np.ones((t,D),dtype='<f4');residual[1:]=-1.;norm=np.ones(D,dtype='<f4');inject=np.zeros((H,D),dtype='<f4')
 else:
  i=np.arange(t*D,dtype=np.uint64);r=((i*1664525+1013904223+seed)&np.uint64(0xffffffff));residual=(((r>>16).astype(np.int64)-32768)/32768.).astype('<f4').reshape(t,D);residual[:,0]=10000.;residual[:,31]=-10000.;residual[:,17]=2**-149
  norm=np.where(np.arange(D)%2,1+2**-20,.5).astype('<f4');inject=(((np.arange(H*D)*13+seed)%11-5)/4096.).astype('<f4').reshape(H,D)
 bo=(((np.arange(t*N)*3)%13-6)/32.).astype('<f4');previous=np.tile(np.asarray([-4.,0.,4.,1.],dtype='<f4'),t)
 return {'residual.f32':f32(residual),'norm.f32':f32(norm),'down.q8_0':q8(L,D,seed),'up.q8_0':q8(D,L,seed+1 if seed else 0),'inject.f32':f32(inject),'bo.f32':f32(bo),'previous.f32':f32(previous),'fp_control.f32':f32(np.tile([2**-126,.5,2**-149,2**24],t))}
def prepare(output):
 source_binding();host=host_binding();out=Path(output).resolve();require(not out.exists(),'New synthetic package required');inputs=out/'inputs';inputs.mkdir(parents=True);rows=[]
 header='HC35_COMPOSITION_INPUTS_V1\n'+''.join(c['id']+' '+str(c['t'])+'\n' for c in fixtures());(inputs/'cases.tsv').write_text(header,encoding='ascii')
 for case in fixtures():
  data=case_data(case);bindings={}
  for field,raw in data.items():
   path=inputs/(case['id']+'.'+field);path.write_bytes(raw);bindings[field]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  rows.append(dict(case,inputs=bindings))
 require(host_binding()==host,'Current host library binding changed during export')
 manifest={'schema':1,'source_plan_sha256':sha(SOURCE_PLAN),'producer_sha256':sha(__file__),'computed_corpus_counts':corpus_counts(),'cases':rows,'host_library_prerequisite':host,'synthetic_inputs_only':True,'original_or_captured_model_inputs_used':False,'device_intrinsics_qualified':False,'normal_model_graph_qualified':False,'full_model_math_qualified':False};write(out/'manifest.json',manifest);input_binding(out);return manifest
def input_binding(package):
 source_binding();root=Path(package).resolve();m=json.loads((root/'manifest.json').read_text());count_binding(m);require(m['source_plan_sha256']==sha(SOURCE_PLAN) and m['producer_sha256']==sha(__file__) and m['synthetic_inputs_only'] is True and m['original_or_captured_model_inputs_used'] is False,'Canonical synthetic source association differs');require(m['host_library_prerequisite']==host_binding(),'Current host library admission differs')
 require([(c['id'],c['t'],c['seed']) for c in m['cases']]==[(c['id'],c['t'],c['seed']) for c in fixtures()],'Exact synthetic case roster differs');inputs=root/'inputs';require(not inputs.is_symlink(),'Synthetic input directory symlink refused');expected={'cases.tsv'}
 require((inputs/'cases.tsv').read_text()=='HC35_COMPOSITION_INPUTS_V1\n'+''.join(c['id']+' '+str(c['t'])+'\n' for c in fixtures()),'Exact native manifest changed')
 for saved,case in zip(m['cases'],fixtures()):
  data=case_data(case);require(set(saved['inputs'])==set(data),'Exact synthetic input fields required')
  for field,raw in data.items():
   path=inputs/(case['id']+'.'+field);expected.add(path.name);require(not path.is_symlink() and path.stat().st_size==len(raw) and path.read_bytes()==raw and saved['inputs'][field]=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},'Actual canonical synthetic bytes changed '+path.name)
 require({p.name for p in inputs.iterdir()}==expected,'Unexpected/missing synthetic input file');return m
def relationships(fields,apply):
 eq=lambda a,b:fields[a]==fields[b]
 return {'actual_rs_equals_shadow':eq('rs','rs_shadow'),'actual_xn_equals_shadow':eq('xn','xn_shadow'),'actual_postSiLU_equals_shadow':eq('post_silu_low','silu_shadow'),'pending_equals_standalone':not apply or eq('residual_after_pending','standalone_write')}
def collect(package,raw_root,log):
 manifest=input_binding(package);root=Path(raw_root).resolve();text=Path(log).read_text();require('HC35_COMPOSITION_RESULT cases=4 frames=36 execution_completed=1 all_owned_allocations_freed=1 numerical_comparison_done=0 device_intrinsics_qualified=0 normal_model_graph_qualified=0 model_math_qualified=0' in text,'Actual complete native result missing')
 require('HC35_COMPOSITION_DEVICE backend=level_zero ' in text and 'affinity=0 selector=level_zero:gpu' in text and 'HC35_COMPOSITION_FP_CONFIG flags=' in text,'Actual device/FP configuration missing');frames=[];cases=[];expected_dirs=set();total=0;relations=[];negative=[]
 for case in fixtures():
  by={};words=0
  for apply in range(3):
   for route in range(3):
    label=case['id']+'-a'+str(apply)+'-r'+str(route);expected_dirs.add(label);directory=root/label;require(directory.is_dir() and not directory.is_symlink(),'Actual bounded frame directory missing');fields={};bindings={}
    line='HC35_COMPOSITION_FRAME case='+case['id']+' tokens='+str(case['t'])+' apply='+str(int(apply!=0))+' alias='+str(int(apply==2))+' route='+str(route)+' graph_replay='+str(route)+' fields=19';require(text.splitlines().count(line)==1,'Actual unique frame/route marker differs')
    require({p.name for p in directory.iterdir()}=={f+'.f32' for f in FIELDS},'Complete exact19 field outputs required')
    for field,n in FIELDS.items():
     path=directory/(field+'.f32');before=path.stat();raw=path.read_bytes();after=path.stat();sig=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns];require(not path.is_symlink() and sig(before)==sig(after) and len(raw)==case['t']*n*4 and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Raw current extent/finite/stat changed '+field);fields[field]=raw;bindings[field]={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'provenance':'actual_compiled_HC_buffer' if field in ACTUAL else 'secondary_actual_compiled_projection' if field=='pre_silu_secondary_projection' else 'separate_shadow_source_arithmetic'};words+=len(raw)//4
    checks=relationships(fields,apply!=0);relations.append(checks);frames.append({'case':case['id'],'t':case['t'],'apply':apply,'route':route,'bindings':bindings,'self_consistency':checks,'device_FMA_gradual_shadow_probe_observation':{'exact_input_output_bytes_equal':fields['fp_control_shadow']==f32(np.tile([2**-127,2**-125],case['t'])),'device_general_FP_mode_qualified':False},'mix_observation':{'separate_bytes_equal':fields['mixed']==fields['mix_separate_shadow'],'fused_bytes_equal':fields['mixed']==fields['mix_fma_shadow'],'candidate_selected_as_policy':False}});by[apply,route]=fields
  for apply in range(3):
   for route in (1,2):require(by[apply,route]==by[apply,0],'Actual graph/replay differs from queue execution for identical synthetic inputs')
  for route in range(3):require(by[1,route]==by[2,route],'Actual alias/noalias applied composition differs')
  cases.append({'id':case['id'],'words':words});total+=words
  if not negative:
   original=by[1,0];bad=dict(original);bad['rs']=bytes([original['rs'][0]^1])+original['rs'][1:];negative.append({'name':'changed_actual_rs_rejected','passed':not relationships(bad,True)['actual_rs_equals_shadow']});bad=dict(original);bad['standalone_write']=bytes([original['standalone_write'][0]^1])+original['standalone_write'][1:];negative.append({'name':'changed_actual_write_rejected','passed':not relationships(bad,True)['pending_equals_standalone']})
 require({p.name for p in root.iterdir()}==expected_dirs,'Unexpected actual raw frame');require(total==corpus_counts()['output_floats'],'Actual corpus float count differs')
 return {'cases':cases,'frames':frames,'actual_output_floats':total,'negative_controls':negative,'numeric_bitwise_passed':all(all(r.values()) for r in relations) and all(r['passed'] for r in negative),'numeric_gate_scope':'synthetic source self-consistency/replay/alias only; never best-mix candidate or intrinsic/model accuracy','computed_corpus_counts':corpus_counts(),'input_manifest_sha256':sha(Path(package)/'manifest.json'),'device_intrinsics_qualified':False,'normal_model_graph_qualified':False,'full_HC_model_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output);print(json.dumps({'prepared':str(a.output),'counts':corpus_counts(),'GPU_executed':False}))
if __name__=='__main__':main()
