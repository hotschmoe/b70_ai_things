"""Separate V2 F32 source-schedule prototype with mandatory gradual probes. Frozen original reference is untouched.
No exp/rsqrt/fullHC/model oracle; arbitrary native values never enter fullmodel.
"""
import ctypes,hashlib,json,math,struct,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
TAG='source35_hc_f32_v1'
_lib=ctypes.CDLL('libm.so.6');_fma=_lib.fmaf;_fma.argtypes=[ctypes.c_float]*3;_fma.restype=ctypes.c_float
_getround=_lib.fegetround;_getround.restype=ctypes.c_int

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def words(raw):
 require(isinstance(raw,bytes) and len(raw)%4==0,'LE_F32 bytes required');out=[v[0] for v in struct.iter_unpack('<f',raw)];require(all(math.isfinite(v) for v in out),'Finite F32 operand required');return out

def require_gradual_f32():
 # Behavioral guards are per call/thread, never cached across rounding changes.
 # Normal operands -> subnormal result exercises output flushing; subnormal
 # operand -> normal result exercises input flushing. This does not query MXCSR.
 require(_getround()==0,'Host FE_TONEAREST required')
 require(float(_fma(2**-126,.5,0.))==2**-127,'Host gradual F32 output required; FTZ behavior refused')
 require(float(_fma(2**-149,2**24,0.))==2**-125,'Host gradual F32 input required; DAZ/input-flush behavior refused')
 return True

def host_fma(a,b,c):
 require_gradual_f32();value=float(_fma(a,b,c));require(math.isfinite(value),'Nonfinite host FMA output');return value

def dot32_f32(weights,input_bytes):
 require_gradual_f32();w,x=words(weights),words(input_bytes);require(len(w)==len(x) and len(x)%32==0 and 0<len(x)<=10240,'Bounded block32 projection required');lanes=[0.]*32
 for lane in range(32):
  for column in range(lane,len(x),32):lanes[lane]=host_fma(w[column],x[column],lanes[lane])
 for mask in (16,8,4,2,1):lanes=[f32(lanes[lane]+lanes[lane^mask]) for lane in range(32)]
 require(math.isfinite(lanes[0]),'Nonfinite reduction');return struct.pack('<f',lanes[0])

def q8dot32_f32(weight_bytes,input_bytes):
 require_gradual_f32();x=words(input_bytes);require(len(x)%32==0 and 0<len(x)<=10240 and len(weight_bytes)==len(x)//32*34,'Q8_0 row extent differs');w=[]
 for at in range(0,len(weight_bytes),34):
  scale=struct.unpack('<e',weight_bytes[at:at+2])[0];require(math.isfinite(scale),'Nonfinite original half scale')
  w.extend(f32(scale*code) for code in struct.unpack('32b',weight_bytes[at+2:at+34]))
 return dot32_f32(struct.pack('<%df'%len(w),*w),input_bytes)

def fma_f32(a_bytes,b_bytes,c_bytes):
 require_gradual_f32();a,b,c=words(a_bytes),words(b_bytes),words(c_bytes);require(len(a)==len(b)==len(c) and 0<len(a)<=10240,'Elementwise FMA shape differs');return struct.pack('<%df'%len(a),*[host_fma(x,y,z) for x,y,z in zip(a,b,c)])

def source_binding():
 plan=json.loads((HERE/'hc-f32-arithmetic35-host-source-plan-v2.json').read_text())
 for name,digest in plan['files'].items():require(sha(HERE.parents[1]/name)==digest,'Frozen host prototype source changed '+name)
 for row in plan['source35_dependencies']:require(sha(row['path'])==row['sha256'],'Current source35 kernel/library binding changed '+row['path'])
 return plan

def run_helper(helper,helper_sha,op,a_bytes,b_bytes,c_bytes=None):
 """Root-only explicit fresh compiled-helper use; input bytes supplied by caller.
 Calls do not discover/read model weights or launch any SYCL/device code.
 """
 source_binding();require_gradual_f32();helper=Path(helper).resolve();require(sha(helper)==helper_sha and helper.read_bytes()[:4]==b'\x7fELF','Pinned fresh host ELF identity required')
 expected={'dot':dot32_f32,'q8dot':q8dot32_f32,'fma':fma_f32};require(op in expected,'Unimplemented intrinsic/composition operation')
 count=len(words(b_bytes));require((op=='fma')==(c_bytes is not None),'Exact FMA/projection input scope differs')
 with tempfile.TemporaryDirectory(prefix='hc35-host-') as directory:
  root=Path(directory);a=root/'a.bin';b=root/'b.bin';c=root/'c.bin';out=root/'out.bin';a.write_bytes(a_bytes);b.write_bytes(b_bytes)
  if c_bytes is not None:c.write_bytes(c_bytes)
  cmd=[str(helper),op,str(count),str(a),str(b),str(c) if c_bytes is not None else '-',str(out),TAG];result=subprocess.run(cmd,check=True,capture_output=True,text=True,timeout=60);receipt=json.loads(result.stdout)
  require(receipt.get('rounding')=='FE_TONEAREST' and receipt.get('flush_to_zero') is False and receipt.get('denormals_are_zero') is False and receipt.get('device_intrinsics_qualified') is False and receipt.get('model_math_qualified') is False and receipt.get('op')==op and receipt.get('elements')==count,'Host helper rounding/scope contract differs')
  require_gradual_f32();raw=out.read_bytes();require(len(raw)==(count if op=='fma' else 1)*4 and all(math.isfinite(v) for v in words(raw)),'Fresh host output shape/finite differs')
  require(sha(helper)==helper_sha,'Host executable changed during call')
 return raw,{'helper_sha256':helper_sha,'input_sha256':[hashlib.sha256(v).hexdigest() for v in (a_bytes,b_bytes,c_bytes) if v is not None],'output_sha256':hashlib.sha256(raw).hexdigest(),'host_receipt':receipt,'host_gradual_input_output_probes_passed':True,'host_direct_MXCSR_flags_observed':False,'implementation_arithmetic_qualified':False,'device_intrinsics_qualified':False,'model_math_qualified':False,'tolerance_gate':None}
