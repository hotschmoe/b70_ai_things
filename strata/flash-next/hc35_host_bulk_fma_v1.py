"""Bounded new CPU bulk FMA/XOR driver, no device or model discovery."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
import hc_f32_arithmetic35_host_v2 as scalar
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE_PLAN=HERE/'hc35-host-bulk-fma-source-plan-v1.json'
TAG='source35_hc_bulk_fma_v1';TILE=64<<20
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_binding():
 plan=json.loads(SOURCE_PLAN.read_text())
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Frozen bulk FMA source changed '+name)
 scalar.source_binding();return plan['files']
def contract(op,k,m,mode,a,b,c=None):
 require(type(k) is int and type(m) is int and 1<=k<=10240 and 1<=m<=10240 and m*k*4<=TILE,'Bounded bulk HC tile required')
 require(op in ('matrixdot','fma') and mode in ('shared','paired'),'Exact bulk operation/mode required');require(op!='matrixdot' or k%32==0 and c is None,'Block32 matrix operation required');require(op!='fma' or mode=='paired' and c is not None,'Paired FMA exact three inputs required')
 sizes=[m*k*4,(k if mode=='shared' else m*k)*4]+([m*k*4] if c is not None else [])
 for raw,n in zip([a,b]+([c] if c is not None else []),sizes):require(isinstance(raw,bytes) and len(raw)==n and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Finite exact LE F32 bulk extent differs')
 return (m if op=='matrixdot' else m*k)*4
def reference_matrixdot(k,m,mode,a,b):
 contract('matrixdot',k,m,mode,a,b);w=np.frombuffer(a,dtype='<f4').reshape(m,k);x=np.frombuffer(b,dtype='<f4').reshape(1 if mode=='shared' else m,k)
 return b''.join(scalar.dot32_f32(row.tobytes(),x[0 if mode=='shared' else i].tobytes()) for i,row in enumerate(w))
def validate_receipt(receipt,op,k,m,mode,extent):
 require(receipt['schema']==2 and receipt['op']==op and receipt['K']==k and receipt['M']==m and receipt['mode']==mode and receipt['output_floats']*4==extent and receipt['rounding']=='FE_TONEAREST' and receipt['flush_to_zero'] is False and receipt['denormals_are_zero'] is False and receipt['device_intrinsics_qualified'] is False and receipt['model_math_qualified'] is False,'Actual bulk helper geometry/mode/scope differs')
def run_bulk(helper,helper_sha,op,k,m,mode,a,b,c=None):
 source_binding();scalar.require_gradual_f32();helper=Path(helper).resolve();require(sha(helper)==helper_sha and helper.read_bytes()[:4]==b'\x7fELF','Fresh qualified bulk ELF required');extent=contract(op,k,m,mode,a,b,c)
 with tempfile.TemporaryDirectory(prefix='hc35-bulk-') as tmp:
  root=Path(tmp);paths=[root/'a.bin',root/'b.bin',root/'c.bin'];out=root/'out.bin'
  for path,raw in zip(paths,[a,b]+([c] if c is not None else [])):path.write_bytes(raw)
  argv=[str(helper),op,str(k),str(m),mode,str(paths[0]),str(paths[1]),str(paths[2]) if c is not None else '-',str(out),TAG];result=subprocess.run(argv,check=True,capture_output=True,text=True,timeout=300);receipt=json.loads(result.stdout);raw=out.read_bytes()
 validate_receipt(receipt,op,k,m,mode,extent);require(len(raw)==extent and np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Finite exact bulk output required');scalar.require_gradual_f32();require(sha(helper)==helper_sha,'Bulk executable changed')
 return raw,{'helper_sha256':helper_sha,'operation':op,'K':k,'M':m,'mode':mode,'input_sha256':[hashlib.sha256(v).hexdigest() for v in [a,b]+([c] if c is not None else [])],'output_sha256':hashlib.sha256(raw).hexdigest(),'helper_receipt':receipt,'host_behavioral_gradual_probes_passed':True,'device_intrinsics_qualified':False,'full_model_math_qualified':False,'tolerance_gate':None}
