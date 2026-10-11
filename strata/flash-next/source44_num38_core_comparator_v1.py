"""NUM38 numerical helpers; no actor admission, tolerance or math authority."""
import ast,copy,ctypes,math
from pathlib import Path
import numpy as np
import source43_num10_original_reference_v3 as original

require=original.require
NAMES=('gdn_core_scaled','gdn_rms_square_sum','gdn_rms_argument','gdn_rms_scale','gdn_rsqrt128')

def fp64_core_targets(core,epsilon):
    core=np.asarray(core,dtype=np.float64)
    require(core.shape==(48,128) and np.isfinite(core).all() and type(epsilon)is float and math.isfinite(epsilon) and epsilon>0,'Original owned FP64 geometry/epsilon required')
    square_sum=np.sum(core*core,axis=1);argument=square_sum/128.0+epsilon
    return dict(zip(NAMES,(core,square_sum,argument,1.0/np.sqrt(argument),np.array([1.0/math.sqrt(128.0)],dtype=np.float64))))

def primary_owned(model):
    """Run unchanged original equations, retaining their already-owned core."""
    node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='independent_first'))
    require(isinstance(node.body[-1],ast.Return) and ast.unparse(node.body[-1].value)=='fields','Exact owned original return required')
    node.body[-1:-1]=ast.parse("fields.update(fp64_core_targets(gdn['core'],float(model.g.eps)))").body
    namespace={**vars(original),'fp64_core_targets':fp64_core_targets}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[NUM38-owned-FP64-core-retention]','exec'),namespace)
    return namespace['independent_first'](model)

def comparisons(owned,native):
    require(set(NAMES)<=set(owned) and set(NAMES)<=set(native),'All five actual NUM38 targets required')
    return {'lane':'unchanged original FirstGdnLayer FP64 diagnostic, extended owned fields','captured_inputs_used':False,'comparisons':{n:original.metrics(owned[n],native[n])for n in NAMES},'existing_F32_full48_reference_replaced':False,'numeric_gate_assigned':False,'full_model_math_qualified':False}

def _fmaf():
    # CPU-only implementation of the explicit source FMA; not SYCL/native
    # subnormal or compiler-fastmath equivalence. No device query is made.
    library=ctypes.CDLL('libm.so.6');fn=library.fmaf
    fn.argtypes=(ctypes.c_float,ctypes.c_float,ctypes.c_float);fn.restype=ctypes.c_float
    return fn

def _conditional_core(normalized,beta,actual_inverse_s,S,HK,HV):
    require(all(type(n)is int and n>0 for n in(S,HK,HV))and S%4==0 and HV%HK==0,'Typed four-group diagnostic geometry required')
    h=np.asarray(normalized,dtype=np.float32);b=np.asarray(beta,dtype=np.float32)
    require(h.shape==(2*HK*S+HV*S,)and b.shape==(HV,)and np.isfinite(h).all()and np.isfinite(b).all()and np.all((b>=0)&(b<=1)),'Finite normalized QKV/beta input required')
    inverse=np.float32(actual_inverse_s);require(np.isfinite(inverse)and inverse>0,'Actual positive finite captured inverse-S required')
    q=h[:HK*S].reshape(HK,S);k=h[HK*S:2*HK*S].reshape(HK,S);v=h[2*HK*S:].reshape(HV,S)
    delta=np.multiply(np.subtract(v,np.float32(0),dtype=np.float32),b[:,None],dtype=np.float32)
    state=np.add(np.float32(0),np.multiply(k[np.arange(HV)%HK].T[:,:,None],delta[None,:,:],dtype=np.float32),dtype=np.float32)
    out=np.empty((HV,S),dtype=np.float32);fma=_fmaf();rpg=S//4
    for head in range(HV):
        for col in range(S):
            partial=[]
            for group in range(4):
                value=0.0
                for row in range(group*rpg,(group+1)*rpg):value=fma(float(state[row,head,col]),float(q[head%HK,row]),value)
                partial.append(np.float32(value))
            total=np.float32(np.float32(np.float32(partial[0]+partial[1])+partial[2])+partial[3])
            out[head,col]=np.float32(total*inverse)
    require(np.isfinite(out).all(),'Finite conditional core required');return out

def conditional_core(normalized,beta,before,actual_inverse_s,native_core):
    require(len(before)==3145728 and not any(before),'Actual recurrent before-state must be bytezero including sign bits')
    own=_conditional_core(normalized,beta,actual_inverse_s,128,16,48)
    return {'lane':'conditional captured normalized QKV/beta and observed inverse-S; own zero recurrence','captured_native_inputs_used_for_local_diagnostic':True,'captured_state_used':False,'comparison':original.metrics(own,native_core),'CPU_libm_fmaf_used':True,'device_intrinsic_subnormal_or_fastmath_equivalence_qualified':False,'existing_F32_full48_reference_replaced':False,'numeric_gate_assigned':False,'full_model_math_qualified':False}
