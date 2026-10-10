"""Preregistered source argument and RS hypotheses; no native values accepted."""
import math,struct
from fractions import Fraction
import numpy as np
import owned_hc35_f32_block_control_v2 as hc

def rounded_rsqrt(argument):
 """Exact rational midpoint decision for correctly rounded F32 1/sqrt(F32 x).
 libm only proposes a bracket. Every bracket/midpoint decision is exact rational.
 """
 hc.require(type(argument) is float and argument>0 and math.isfinite(argument) and hc.host.f32(argument)==argument,'Positive exact F32 argument required');x=Fraction(argument);guess=hc.host.f32(1/math.sqrt(argument));bits=struct.unpack('<I',struct.pack('<f',guess))[0];candidates=[struct.unpack('<f',struct.pack('<I',b))[0] for b in range(max(1,bits-4),min(0x7f800000,bits+5))]
 exact=[v for v in candidates if Fraction(v)**2*x==1]
 if exact:return exact[0]
 below=[v for v in candidates if Fraction(v)**2*x<1];above=[v for v in candidates if Fraction(v)**2*x>1];hc.require(below and above,'Exact bounded reciprocal sqrt bracket unavailable');lo,hi=max(below),min(above);mid=(Fraction(lo)+Fraction(hi))/2;decision=mid*mid*x-1
 if decision>0:return lo
 if decision<0:return hi
 return lo if struct.unpack('<I',struct.pack('<f',lo))[0]%2==0 else hi

def variants(residual):
 r=np.asarray(residual,dtype='<f4');hc.require(r.shape==(4,2560) and np.isfinite(r).all(),'Own original residual geometry/finite required');result={}
 source_args=[hc.rms_argument(row,hc.EPSILON)['rsqrt_argument_f32'] for row in r]
 # Frozen prior: F32 square, FP64 mean, F32 mean then F32 epsilon argument.
 legacy_args=[hc.host.f32(hc.host.f32(float(np.mean(np.asarray(row*row,dtype='<f4'),dtype=np.float64)))+hc.EPSILON) for row in r]
 for label,args in [('source_FMA_XOR',source_args),('legacy_square_FP64_mean',legacy_args)]:
  result[label]={'argument':np.asarray(args,dtype='<f4').tobytes(),'host_sqrtf_reciprocal':np.asarray([hc.rsqrt_candidate(float(a)) for a in args],dtype='<f4').tobytes(),'mathematically_rounded_rsqrt':np.asarray([rounded_rsqrt(float(a)) for a in args],dtype='<f4').tobytes(),'legacy_FP64_sqrt_reciprocal':np.asarray([hc.host.f32(1/math.sqrt(float(a))) for a in args],dtype='<f4').tobytes()}
 return result
