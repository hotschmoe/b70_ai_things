"""Explicit hashed CPU rows-dot backend; scalar candidate remains unchanged."""
import ast,copy,ctypes,hashlib,os,stat
from pathlib import Path
import numpy as np
import original_native_hc_f32_v1 as scalar
require=scalar.require

class RowsDot32:
 def __init__(self,path,expected_sha256):
  self.path=Path(path);require(self.path.is_absolute()and self.path.resolve()==self.path and not any(p.is_symlink()for p in(self.path,*self.path.parents)),'Exact canonical CPU ELF path required');self.expected=expected_sha256;self.recheck();self.lib=ctypes.CDLL(str(self.path));self.recheck();abi=self.lib.hc_rows_dot32_abi_v1;abi.restype=ctypes.c_uint;require(abi()==1,'Exact opt-in CPU ABI required');self.fn=self.lib.hc_rows_dot32_v1;pointer=ctypes.POINTER(ctypes.c_float);self.fn.argtypes=[pointer,pointer,pointer,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_int];self.fn.restype=ctypes.c_int;self.calls=0
 def recheck(self):
  before=self.path.stat();require(not self.path.is_symlink()and stat.S_ISREG(before.st_mode)and 0<before.st_size<=8<<20,'Regular bounded opt-in CPU ELF required');raw=self.path.read_bytes();after=self.path.stat();signature=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  require(signature(before)==signature(after)and raw[:4]==b'\x7fELF'and hashlib.sha256(raw).hexdigest()==self.expected,'Exact current opt-in CPU ELF hash/stat required')
 def rows(self,weights,input,paired=False):
  require(type(paired)is bool,'Explicit paired/shared mode required');w=np.ascontiguousarray(weights,dtype='<f4');x=np.ascontiguousarray(input,dtype='<f4');require(w.ndim==2,'Rows matrix required');m,k=w.shape;require(1<=m<=10240 and 32<=k<=10240 and k%32==0 and w.nbytes<=64<<20 and x.shape==((m,k)if paired else(k,))and np.isfinite(w).all()and np.isfinite(x).all(),'Exact finite CPU tile/operand dimensions required');self.recheck();out=np.empty(m,dtype='<f4');pointer=ctypes.POINTER(ctypes.c_float);rc=self.fn(w.ctypes.data_as(pointer),x.ctypes.data_as(pointer),out.ctypes.data_as(pointer),k,m,int(paired));require(rc==0 and np.isfinite(out).all(),'CPU rows-dot environment/input failure rc='+str(rc));self.recheck();self.calls+=1;return out
 def project(self,projector,name,operand):
  shape=projector.p.shape(name);require(len(shape)==2,'Original HC matrix required');x=scalar.finite(operand,(shape[0],));step=projector.tile//(shape[0]*8);require(step>0,'Original row tile exceeded');out=np.empty(shape[1],dtype='<f4')
  for at in range(0,shape[1],step):
   n=min(step,shape[1]-at);w=scalar.finite(projector.p.rows(name,range(at,at+n)),(n,shape[0]));out[at:at+n]=self.rows(w,x)
  return out
 def normalize(self,residual,norm,epsilon):
  r=np.ascontiguousarray(residual,dtype='<f4');require(r.ndim==2 and r.shape[0]==4 and 0<r.shape[1]<=2560 and r.shape[1]%32==0,'Four source HC streams required');n=scalar.finite(norm,r.shape);eps=scalar.scalar.f32(epsilon);require(np.isfinite(eps)and eps>0,'Positive independent epsilon required');sums=self.rows(r,r,True);arguments=scalar.f32(scalar.f32(sums/r.shape[1])+eps);scales=scalar.f32([scalar.rsqrt_candidate(value)for value in arguments]);return {'normalized':scalar.f32(scalar.f32(r*n)*scales[:,None]),'square_sum':sums,'rms_argument':arguments,'rms_scale':scales}
 def hc_read_original(self,projector,residual,stem,epsilon,inject=True):
  # Only two explicit backend calls change. Original source types, own operands,
  # nonlinear candidates, separate HCmix and packet computation are untouched.
  node=copy.deepcopy(next(n for n in ast.parse(Path(scalar.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='hc_read_original'));ns={**vars(scalar),'project_original_rows':self.project,'normalize_streams':self.normalize};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[opt-in-original-CPU-bulk-only]','exec'),ns);return ns['hc_read_original'](projector,residual,stem,epsilon,inject)

class OwnedOriginalNativeHcBulkF32(scalar.OwnedOriginalNativeHcF32):
 def __init__(self,*args,rows_dot_elf,rows_dot_sha256,**kwargs):
  self.rows_backend=RowsDot32(rows_dot_elf,rows_dot_sha256);super().__init__(*args,**kwargs)
 def hc_read(self,residual,stem,inject=True):return self.rows_backend.hc_read_original(self.projector,residual,stem,self.g.eps,inject)
 def tokens(self,token_ids):
  self.rows_backend.recheck();result=super().tokens(token_ids);self.rows_backend.recheck();result.update(explicit_CPU_rows_dot32_ELF=str(self.rows_backend.path),explicit_CPU_rows_dot32_ELF_sha256=self.rows_backend.expected,CPU_rows_dot32_calls=self.rows_backend.calls,scalar_reference_monkeypatched=False,host_rsqrt_expf_device_intrinsics_qualified=False,full_model_math_qualified=False);return result
