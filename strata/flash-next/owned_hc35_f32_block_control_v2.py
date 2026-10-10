"""Independent token-driven layer0 HC block control, never a device oracle.

Only original embeddings/weights enter math. Host expf/rsqrt candidates and mix
contraction alternatives remain explicit, unqualified hypotheses.
"""
import ctypes,hashlib,json,math,struct
from pathlib import Path
import numpy as np
import hc_f32_arithmetic35_host_v2 as host
from full48_owned_composition_storage_v1 import OriginalStorageProjector
from full48_owned_composition_storage_v2 import derive_schedule,bind_dispatch_source,PLAN
from independent_q8_1_activation_v1 import encode

HERE=Path(__file__).resolve().parent
SOURCE_PLAN=HERE/'owned-hc35-f32-block-control-source-plan-v2.json'
N=2560;H=4;L=320;BLOCK=35;COLUMNS=tuple(range(BLOCK*32,(BLOCK+1)*32))
EPSILON=host.f32(1e-6);EPSILON_HEX='bd378635'
_lib=ctypes.CDLL('libm.so.6')
for _name in ('sqrtf','expf'):
 _op=getattr(_lib,_name);_op.argtypes=[ctypes.c_float];_op.restype=ctypes.c_float

def require(ok,message):
 if not ok:raise ValueError(message)
def raw(values):return np.asarray(values,dtype='<f4').tobytes()
def dot(weights,values):return struct.unpack('<f',host.dot32_f32(raw(weights),raw(values)))[0]
def rms_argument(values,eps):
 """Source square-FMA lanes, XOR tree, F32 division and F32 epsilon addition."""
 values=list(values);require(len(values)>0 and len(values)%32==0 and len(values)<=N,'Bounded RMS block32 required')
 total=dot(values,values);argument=host.f32(host.f32(total/len(values))+host.f32(eps))
 require(argument>0 and math.isfinite(argument),'Finite positive RMS argument required')
 return {'square_sum_f32':total,'mean_f32':host.f32(total/len(values)),'rsqrt_argument_f32':argument}
def rsqrt_candidate(argument):
 host.require_gradual_f32();value=host.f32(1/float(_lib.sqrtf(argument)));require(math.isfinite(value),'Host reciprocal sqrt candidate nonfinite');return value
def sigmoid_candidate(value):
 require(math.isfinite(value),'Finite sigmoid candidate input required');host.require_gradual_f32();exponential=float(_lib.expf(host.f32(-value)))
 # Overflow for a finite negative sigmoid argument has a finite limiting result.
 denominator=host.f32(1+exponential) if math.isfinite(exponential) else math.inf
 return host.f32(1/denominator)
def mixed_candidates(normalized,gates):
 require(np.shape(normalized)==np.shape(gates) and len(normalized)==H,'Four owned HC streams required')
 separate=[];fused=[]
 for column in range(len(normalized[0])):
  a=b=0.
  for stream in range(H):
   x=host.f32(normalized[stream][column]);s=sigmoid_candidate(host.f32(gates[stream][column]))
   a=host.f32(a+host.f32(x*s));b=host.host_fma(x,s,b)
  separate.append(host.f32(a/H));fused.append(host.f32(b/H))
 return {'host_expf_separate_mix':np.asarray(separate,dtype='<f4'),'host_expf_fused_mix':np.asarray(fused,dtype='<f4')}
def source_binding(source_root):
 plan=json.loads(SOURCE_PLAN.read_text())
 for name,want in plan['files'].items():require(host.sha(HERE.parents[1]/name)==want,'Frozen own HC control source changed '+name)
 host.source_binding();dispatch=bind_dispatch_source(source_root);engine=json.loads(PLAN.read_text());kernels={}
 for name in ('sycl/src/kernels/hc_native_composition.cpp','sycl/src/kernels/hc_native_projection.cpp'):
  path=Path(source_root)/name;require(host.sha(path)==engine['expected_patched_source_sha256'][name],'Source35 HC kernel binding changed');kernels[name]=host.sha(path)
 return {'dispatch':dispatch,'kernels':kernels,'device_arithmetic_observed':False}

def epsilon_contract(provider):
 row=provider.reader.files[0]['metadata']['qwen4exp.attention.layer_norm_rms_epsilon']
 require(type(row) is dict and type(row['type']) is int and row['type']==6 and type(row['value']) is float,'Exact scalar GGUF FLOAT32 epsilon type6 required')
 value=row['value'];require(value==EPSILON and struct.pack('<f',value).hex()==EPSILON_HEX,'Exact source F32 epsilon value/bytes differs')
 digest=hashlib.sha256(struct.pack('<f',value)).hexdigest();require(row['encoded_value_sha256']==digest,'Pinned scalar F32 metadata encoded bytes differ')
 require(provider.source_contract['original_rms_epsilon']==value and provider.source_contract['effective_expert_scale']==1.,'Original RMS/source contract differs')
 return {'metadata_type':6,'epsilon':value,'epsilon_LE_F32_hex':EPSILON_HEX,'encoded_value_sha256':digest}

class OwnedHcBlockControl:
 def __init__(self,provider,identity_sha256,args,env,source_root,host_admission=None):
  self.p=provider;self.identity=identity_sha256;self.args=tuple(args);self.env=dict(env);self.source_root=Path(source_root);self.host_admission=host_admission
  derive_schedule([19],self.args,self.env)
  self.eps=EPSILON;self.epsilon_binding={'epsilon':EPSILON,'epsilon_LE_F32_hex':EPSILON_HEX,'synthetic_only':not provider.actual_source}
  if provider.actual_source:
   require(provider.source_identity_sha256==identity_sha256 and callable(host_admission),'Actual original identity and current host admission required')
   self.epsilon_binding=epsilon_contract(provider);self.eps=self.epsilon_binding['epsilon']
  self.binding=source_binding(source_root);self.projector=OriginalStorageProjector(provider,64<<20)
  for name,kind,shape in [('token_embd.weight','Q8_0',(N,248320)),('blk.0.hc_attn_norm.weight','F32',(H*N,)),('blk.0.hc_attn_down.weight','Q8_0',(H*N,L)),('blk.0.hc_attn_up.weight','Q8_0',(L,H*N)),('blk.0.hc_attn_inject.weight','F32',(H*N,H))]:self.projector.require(name,kind,shape)
 def project_rows(self,name,indices,operand):
  indices=list(indices);require(len(indices)*len(operand)*8<=64<<20,'Original decoded tile bound exceeded')
  weights=np.asarray(self.p.rows(name,indices),dtype=np.float64);require(weights.shape==(len(indices),len(operand)) and np.isfinite(weights).all(),'Original bounded row shape/finite differs')
  return np.asarray([dot(row,operand) for row in weights],dtype='<f4')
 def tokens(self,token_ids):
  ids=list(token_ids);schedule=derive_schedule(ids,self.args,self.env);before=self.host_admission() if self.p.actual_source else None;require(source_binding(self.source_root)==self.binding,'Source changed before own HC work')
  norm=self.projector.vector('blk.0.hc_attn_norm.weight').reshape(H,N);rows=[]
  for position,token in enumerate(ids):
   embedding=np.asarray(self.p.rows('token_embd.weight',[token])[0],dtype='<f4');require(embedding.shape==(N,) and np.isfinite(embedding).all(),'Original embedding shape/finite differs')
   residual=np.broadcast_to(embedding,(H,N)).copy();rms=[rms_argument(r,self.eps) for r in residual];rs=np.asarray([rsqrt_candidate(r['rsqrt_argument_f32']) for r in rms],dtype='<f4')
   # Source multiplication order is (R * norm) * rs, with two F32 stores.
   normalized=np.asarray(np.asarray(residual*norm,dtype='<f4')*rs[:,None],dtype='<f4');down=self.project_rows('blk.0.hc_attn_down.weight',range(L),normalized.reshape(-1));low=np.asarray([host.f32(host.f32(x/H)*sigmoid_candidate(host.f32(x/H))) for x in down],dtype='<f4')
   indices=[stream*N+column for stream in range(H) for column in COLUMNS];gates=self.project_rows('blk.0.hc_attn_up.weight',indices,low).reshape(H,32);inject=self.project_rows('blk.0.hc_attn_inject.weight',range(H),normalized.reshape(-1));mixed=mixed_candidates(normalized[:,COLUMNS],gates)
   rows.append({'position':position,'token':token,'embedding_owned':embedding,'rms_owned':rms,'rsqrt_host_candidate':rs,'normalized_owned_candidate':normalized,'down_owned_candidate':down,'post_silu_owned_candidate':low,'gate_owned_candidate':gates,'inject_owned_candidate':inject,'mixed_block_owned_candidates':mixed,'q81_block_owned_candidates':{name:encode(value.tobytes(),1,32) for name,value in mixed.items()}})
  require(source_binding(self.source_root)==self.binding,'Source changed during own HC work')
  if self.p.actual_source:require(self.host_admission()==before,'Current host runtime changed during own HC work')
  return {'control_generation':2,'epsilon_contract':self.epsilon_binding,'ids':ids,'source_identity_sha256':self.identity,'rows':rows,'declared_schedule':schedule,'columns':list(COLUMNS),'scope':'Independent original embedding layer0 HC block35 only; no GDN/FFN/fullmodel','captured_inputs_used':False,'captured_state_or_routes_used':False,'source_FMA_XOR_schedule_used':True,'RMS_division_add_device_rounding_qualified':False,'rsqrt_candidate':'host sqrtf then F32 reciprocal; NOT sycl::rsqrt proof','exp_candidate':'host expf with F32 add/divide; NOT ordinary sycl::exp proof','mix_candidates':'separate product/add and fused FMA; compiler contraction unobserved','actual_original_payload_used':self.p.actual_source,'device_intrinsics_qualified':False,'nativebitwise_qualified':False,'full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_gate':None}
