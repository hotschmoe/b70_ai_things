"""Independent source-derived PLE postops storage audit, not native reducer oracle.
No source weight provider. Synthetic/captured-computed inputs only for diagnosis.
"""
import numpy as np
from native_storage_first_gdn_estimate_v1 import f32
EPS=float(np.float32(1e-6))

def source_stage_estimate(key,value,residual,gammas,history,channel_taps):
 key=f32(key);value=f32(value);residual=f32(residual);history=f32(history);weights=f32(channel_taps)
 if key.shape!=residual.shape or key.shape!=(4,2560) or value.shape!=(2560,) or history.shape!=(9,10240) or weights.shape!=(10240,4) or any(np.shape(g)!=(4,2560) for g in gammas):raise ValueError('Actual source PLE logical geometry differs')
 def norm(x,gamma):
  # Materialized F32 products and norm stores. FP64 sum/CPU sqrt/exp estimate
  # still does NOT emulate native subgroup/FMA reduction or native intrinsics.
  rs=f32(1/np.sqrt(f32(np.mean(f32(x*x),axis=1,keepdims=True))+EPS))
  return f32(f32(x*rs)*gamma)
 kn=norm(key,gammas[0]);qn=norm(residual,gammas[1]);score=f32(f32(np.sum(f32(kn*qn),axis=1))*np.float32(1/np.sqrt(np.float32(2560))))
 gate=f32(1/(1+np.exp(-f32(np.sign(score)*np.sqrt(np.maximum(np.abs(score),EPS))))));gated=f32(value[None,:]*gate[:,None]);normalized=norm(gated,gammas[2]);flat=normalized.reshape(-1)
 # Source physical history[c*9+3*k] equals logical history[3*k,c].
 terms=[history[0],history[3],history[6],flat];summation=f32(terms[0]*weights[:,0])
 for tap in range(1,4):summation=f32(summation+f32(terms[tap]*weights[:,tap]))
 activated=f32(summation/(1+np.exp(-summation)));result=f32(f32(residual+gated)+activated.reshape(4,2560))
 return {'key_norm':kn,'query_norm':qn,'gate':gate,'gated':gated,'normalized':normalized,'convolved_raw':summation.reshape(4,2560),'conv_activation':activated.reshape(4,2560),'result':result,'next_history':np.concatenate([history[1:],flat[None]]),'native_reducer_or_intrinsic_qualified':False}
