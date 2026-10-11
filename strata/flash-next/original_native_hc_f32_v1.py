"""Independent original-operand HC source schedule; no captured inputs/state.

HC Q8_0 weights are decoded once then dotted with UNQUANTIZED F32 operands.
This is distinct from other Q8_0 x Q8_1 d0*d1 integer-dot consumers. Host
sqrtf/expf are explicit candidates, not a proof of SYCL intrinsic rounding.
"""
import ctypes,math,struct
import numpy as np
import hc_f32_arithmetic35_host_v2 as scalar
from full48_owned_composition_storage_v2 import RouteAwareFull48OwnedComposition
from independent_q8_1_activation_v1 import encode
F32=np.dtype('<f4')
_lib=ctypes.CDLL('libm.so.6')
for _name in('sqrtf','expf'):
 _op=getattr(_lib,_name);_op.argtypes=[ctypes.c_float];_op.restype=ctypes.c_float

def require(ok,message):
 if not ok:raise ValueError(message)
def f32(x):return np.asarray(x,dtype=F32)
def finite(x,shape):
 x=f32(x);require(x.shape==shape and np.isfinite(x).all(),'Independent F32 operand shape/finite differs');return x

def dot32(weights,operand):
 w=f32(weights).reshape(-1);x=f32(operand).reshape(-1)
 require(w.shape==x.shape and 0<len(x)<=10240 and len(x)%32==0 and np.isfinite(w).all()and np.isfinite(x).all(),'Source32 projection geometry/finite differs')
 # Existing pure CPU primitive: lane FMA and simultaneous XOR16..1 updates.
 return struct.unpack('<f',scalar.dot32_f32(w.tobytes(),x.tobytes()))[0]

def rsqrt_candidate(argument):
 scalar.require_gradual_f32();argument=scalar.f32(argument);require(argument>0 and math.isfinite(argument),'Positive independent RMS argument required')
 return scalar.f32(1.0/float(_lib.sqrtf(argument)))
def sigmoid_candidate(value):
 scalar.require_gradual_f32();x=scalar.f32(value);require(math.isfinite(x),'Finite independent sigmoid argument required');e=float(_lib.expf(scalar.f32(-x)));den=scalar.f32(1+e)if math.isfinite(e)else math.inf
 return scalar.f32(1/den)

def normalize_streams(residual,norm,epsilon):
 r=f32(residual);require(r.ndim==2 and r.shape[0]==4 and 0<r.shape[1]<=2560 and r.shape[1]%32==0,'Four source HC streams required');n=finite(norm,r.shape);require(np.isfinite(r).all(),'Independent residual nonfinite');eps=scalar.f32(epsilon);require(math.isfinite(eps)and eps>0,'Positive F32 epsilon required')
 sums=f32([dot32(row,row)for row in r]);arguments=f32(f32(sums/r.shape[1])+eps);scales=f32([rsqrt_candidate(value)for value in arguments]);xn=f32(f32(r*n)*scales[:,None])
 return {'normalized':xn,'square_sum':sums,'rms_argument':arguments,'rms_scale':scales}

def project_original_rows(projector,name,operand):
 shape=projector.p.shape(name);require(len(shape)==2 and shape[0]%32==0 and shape[0]<=10240,'Original bounded HC matrix required');x=finite(operand,(shape[0],));count=shape[1];out=np.empty(count,dtype=F32);step=projector.tile//(shape[0]*8);require(step>0,'Original HC decoded row tile exceeded')
 for at in range(0,count,step):
  size=min(step,count-at);rows=finite(projector.p.rows(name,range(at,at+size)),(size,shape[0]));out[at:at+size]=[dot32(row,x)for row in rows]
 return out

def hc_read_original(projector,residual,stem,epsilon,inject=True):
 require(type(inject)is bool,'Explicit HC inject branch required');r=finite(residual,(4,2560));projector.require(stem+'norm.weight','F32',(10240,));projector.require(stem+'down.weight','Q8_0',(10240,320));projector.require(stem+'up.weight','Q8_0',(320,10240))
 if inject:projector.require(stem+'inject.weight','F32',(10240,4))
 norm=finite(projector.vector(stem+'norm.weight').reshape(4,2560),(4,2560));result=normalize_streams(r,norm,epsilon);xn=result['normalized'];down=project_original_rows(projector,stem+'down.weight',xn.reshape(-1));scaled=f32(down/4);low=f32([scalar.f32(float(x)*sigmoid_candidate(float(x)))for x in scaled]);gate=project_original_rows(projector,stem+'up.weight',low).reshape(4,2560)
 total=np.zeros(2560,dtype=F32)
 # Source HCmix is separate rounded product then rounded sequential sum.
 # The old refinement's fused FMA mix is not silently selected.
 for stream in range(4):total=f32(total+f32(xn[stream]*f32([sigmoid_candidate(float(x))for x in gate[stream]])))
 result.update(down=down,low_silu=low,gate=gate,mixed=f32(total/4))
 if inject:result['inject']=project_original_rows(projector,stem+'inject.weight',xn.reshape(-1))
 return result

def hc_write_original(residual,block,inject):
 r=finite(residual,(4,2560));b=finite(block,(2560,));i=finite(inject,(4,));out=np.empty_like(r)
 for stream in range(4):
  gain=scalar.f32(2*sigmoid_candidate(scalar.f32(float(i[stream])/4)))
  out[stream]=np.frombuffer(scalar.fma_f32(b.tobytes(),f32(np.full(2560,gain)).tobytes(),r[stream].tobytes()),dtype=F32)
 return out

class OwnedOriginalNativeHcF32(RouteAwareFull48OwnedComposition):
 """Only HC arithmetic changes; all48 original weights and own states remain."""
 def hc_read(self,residual,stem,inject=True):return hc_read_original(self.projector,residual,stem,self.g.eps,inject)
 def hc_write(self,residual,block,inject):return hc_write_original(residual,block,inject)
 def tokens(self,token_ids):
  result=super().tokens(token_ids);result.update(lane='independent_original48_native_HC_F32_candidate_v1',HC_lane_FMA_XOR32=True,HC_activation_quantized_Q81=False,HC_injection_original_F32=True,HC_mix_separate_product_add=True,HC_write_FMA=True,host_sqrtf_expf_device_intrinsics_qualified=False,non_HC_primitives_changed=False,captured_inputs_used=False,captured_states_or_selected_ids_used=False,full_model_math_qualified=False,tolerance_gate=None);return result

def owned_first_packet(mixed):
 value=finite(mixed,(2560,));return encode(value.tobytes(),1,2560)
