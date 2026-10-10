"""Preregistered eager device operations at the unchanged separate argument."""
import hashlib,json
from pathlib import Path
import numpy as np
import native_rms_rsqrt37_proposal_v1 as original
import qualify_native_rms_rsqrt37_v7 as prior
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PLAN=HERE/'native-rms-rsqrt37-device-ops-source-plan-v1.json'
OPERATIONS={'device_rsqrt_shadow':'sycl::rsqrt(x)','device_native_rsqrt_shadow':'sycl::native::rsqrt(x)','device_recip_sqrt_shadow':'1.0f/sycl::sqrt(x)'}
FIELDS={'actual_hc_rs':16,'actual_hc_xn_normones':40960,'separate_square_sum_shadow':16,'separate_argument_shadow':16,**{k:16 for k in OPERATIONS}}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def derivation():
 s=(HERE/'native_rms_rsqrt37_callable_v7.cpp').read_text();s=s.replace('class RmsArgumentShadow37;','class RmsArgumentShadow37;\nclass RmsDeviceOperationShadow37;')
 s=s.replace('auto* sum=mem.floats(H);auto* argument=mem.floats(H);','auto* sum=mem.floats(H);auto* argument=mem.floats(H);\n auto* device_rsqrt=mem.floats(H);auto* device_native_rsqrt=mem.floats(H);auto* device_recip_sqrt=mem.floats(H);')
 s=s.replace('}).wait_and_throw();auto dir=', '}).wait_and_throw();\n  // Same separately observed argument; these are eager leaf candidates, not HC internals.\n  q.parallel_for<RmsDeviceOperationShadow37>(sycl::range<1>(H),[=](sycl::id<1> index){const size_t c=index[0];const float x=argument[c];device_rsqrt[c]=sycl::rsqrt(x);device_native_rsqrt[c]=sycl::native::rsqrt(x);device_recip_sqrt[c]=1.0f/sycl::sqrt(x);}).wait_and_throw();auto dir=')
 s=s.replace('dump(q,dir,"separate_argument_shadow",argument,H);','dump(q,dir,"separate_argument_shadow",argument,H);dump(q,dir,"device_rsqrt_shadow",device_rsqrt,H);dump(q,dir,"device_native_rsqrt_shadow",device_native_rsqrt,H);dump(q,dir,"device_recip_sqrt_shadow",device_recip_sqrt,H);').replace('fields=4 synthetic_norm=1','fields=7 synthetic_norm=1')
 require((HERE/'native_rms_rsqrt37_device_ops_v1.cpp').read_bytes()==s.encode('ascii'),'Device candidate source differs from exact declared additions')
 wrapper=(HERE/'native_rms_rsqrt37_owned_entry_v7.cpp').read_bytes().replace(b'#include "native_rms_rsqrt37_callable_v7.cpp"',b'#include "native_rms_rsqrt37_device_ops_v1.cpp"');require((HERE/'native_rms_rsqrt37_device_ops_entry_v1.cpp').read_bytes()==wrapper,'Metadata wrapper changed beyond include')
 return {'actual_HC_call_and_original_shadow_unchanged':True,'candidates_eager_outside_HC_graph':True,'candidate_source_sha256':sha(s.encode('ascii')),'metadata_wrapper_include_only':True}
def source_binding():
 row=json.loads(PLAN.read_bytes())
 for n,w in row['files'].items():require(sha((ROOT/n).read_bytes())==w,'Device operation source changed '+n)
 require(prior.sha(prior.PLAN)=='265ebe47c63285ee0ec987051a692a696a5a53b0045626babf3a2fa607e57835','Original V7 source plan changed');prior.source_binding();return {'source_plan_sha256':sha(PLAN.read_bytes()),'derivation':derivation(),'operations':OPERATIONS,'source_native_rsqrt_compilation_observed':False}
def compile_argv():
 argv=original.compile_argv();argv[argv.index('/leaf/native_rms_rsqrt37_gpu_v1.cpp')]='/leaf/native_rms_rsqrt37_device_ops_entry_v1.cpp';return argv
def prior_binding(root):
 root=Path(root).resolve();binding=prior.finalized_binding(root);require(binding['actual_RMS_observed'] is True and binding['full_model_math_qualified'] is False,'Actual closed V7 prerequisite failed');return {'root':str(root),'finalized_binding':binding,'captured_fields_used_as_inputs':False,'prior_fields_output_targets_only':True}
def compare_outputs(output,original_output):
 output=Path(output);original_output=Path(original_output);routes=[]
 for route in range(3):
  row={}
  for field,n in FIELDS.items():
   raw=(output/f'r{route}'/(field+'.f32')).read_bytes();v=np.frombuffer(raw,dtype='<f4');require(len(raw)==n and np.isfinite(v).all(),'Complete finite operation field required '+field);row[field]=raw
   if field not in OPERATIONS:require(raw==(original_output/f'r{route}'/(field+'.f32')).read_bytes(),'Unchanged HC/original shadow target differs '+field)
  require(np.all(np.frombuffer(row['separate_argument_shadow'],dtype='<f4')>0),'Exact positive source argument required');routes.append(row)
 require(routes[0]==routes[1]==routes[2],'Direct HC/replays with eager device operations must repeat bitwise')
 return {'all_routes_bitwise':True,'original_HC_and_four_fields_bitwise':True,'source_expressions':OPERATIONS,'comparisons':{field:{'bytes_equal_actual_HC_RS':routes[0][field]==routes[0]['actual_hc_rs'],'sha256':sha(routes[0][field]),'raw_u32':np.frombuffer(routes[0][field],dtype='<u4').tolist()} for field in OPERATIONS},'actual_HC_RS_u32':np.frombuffer(routes[0]['actual_hc_rs'],dtype='<u4').tolist(),'separate_argument_u32':np.frombuffer(routes[0]['separate_argument_shadow'],dtype='<u4').tolist(),'internal_HC_argument_observed':False,'device_instruction_lowering_observed':False,'tolerance_gate':None,'full_model_math_qualified':False}
