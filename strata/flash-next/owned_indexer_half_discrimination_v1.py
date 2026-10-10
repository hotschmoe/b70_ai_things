"""Preregistered half conversion comparison; no model/captured operand inputs."""
import hashlib,re
from pathlib import Path
import numpy as np
from owned_layer3_qsa_contract_v1 import require
import native_rms_rsqrt37_proposal_v1 as p
from owned_layer3_qsa_runtime_contract_v4 import qsa_builder_flags
HERE=Path(__file__).resolve().parent
CPP=HERE/'owned_indexer_half_discrimination_gpu_v1.cpp'
FIELDS={'expression.f32':('<f4',2048),'materialized.f32':('<f4',2048),'materialized.f16':('<f2',1024),'input.f32':('<f4',2048)}
def sha(raw):return hashlib.sha256(raw).hexdigest()
def original_input(raw):
 require(type(raw)is bytes and len(raw)==2048,'Exact independently owned four128 raw F32 rows required')
 x=np.frombuffer(raw,dtype='<f4').copy();require(np.isfinite(x).all() and (np.abs(x)<=65504).all(),'Finite FP16 bounded own inputs required')
 return x
def preregistered(raw):
 x=original_input(raw);half=x.astype('<f2');return {'identity_F32':x.tobytes(),'FP16_RNE':half.tobytes(),'FP16_RNE_widened_F32':half.astype('<f4').tobytes()}
def compile_argv():
 argv=p.compile_argv();source="/leaf/native_rms_rsqrt37_gpu_v1.cpp";require(argv.count(source)==1,'Original source flag recipe requires exact leaf source')
 argv[argv.index(source)]='/leaf/owned_indexer_half_discrimination_gpu_v1.cpp';argv[-1]='/out/owned-indexer-half-discrimination';return argv

def source_binding():
 flags=p.builder_flags();qsa=qsa_builder_flags(p.SDK,flags)
 src=p.SDK/'source/sycl/src/kernels/cuda/native_qsa_indexer.dp.cpp';text=src.read_text()
 tokens=''.join(text.split());expression='sycl::vec<sycl::half,1>(sycl::vec<float,1>(raw_step[d]).convert<sycl::half,sycl::rounding_mode::rte>()[0]).convert<float,sycl::rounding_mode::automatic>()[0]'
 require(expression in tokens,'Exact consumed source37 append half-roundtrip expression required')
 return {'SDK':str(p.SDK),'indexer_source':str(src),'indexer_source_sha256':sha(src.read_bytes()),'production_object_flags':qsa,'baseline_HC_flags':flags,'leaf_argv':compile_argv(),'source_expression_variable_only_adaptation':True,'runtime_ready':False,'actual_compiler_lowering_observed':False,'full_model_math_qualified':False}

def recollect(root,raw,log):
 root=Path(root);expected=preregistered(raw)
 require((root/'consumed-raw.f32').read_bytes()==raw,'Exact actual consumed independently owned input echo required')
 require(not re.search(r'HALF37_ERROR|OWNED_ERROR',log),'Actual helper error marker')
 require(len(re.findall(r'^HALF37_DEVICE backend=level_zero affinity=0 fp16=1 vendor=.+ driver=.+ name=.+$',log,re.M))==1,'Exact actual device/capability marker required')
 frames=re.findall(r'^HALF37_FRAME route=(\d+) graph_replay=(\d+) fields=4 own_input_restored=1 values=512$',log,re.M)
 require(frames==[('0','0'),('1','1'),('2','2')],'Exact direct/two-replay marker roster required')
 require(log.count('HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0')==1,'Actual graph/owned retirement marker required')
 rows={}
 for field,(dtype,size) in FIELDS.items():
  values=[]
  for route in range(3):
   path=root/('route'+str(route))/field;require(path.is_file() and not path.is_symlink(),'Regular actual raw field required');b=path.read_bytes();require(len(b)==size and np.isfinite(np.frombuffer(b,dtype=dtype)).all(),'Actual finite full raw field extent required');values.append(b)
  require(values[0]==values[1]==values[2],'Actual full raw direct/replay equality required')
  rows[field]={'sha256':sha(values[0]),'bytes':size,'equals_identity_F32':values[0]==expected['identity_F32'],'equals_FP16_RNE_widened_F32':values[0]==expected['FP16_RNE_widened_F32'],'equals_FP16_RNE_bytes':values[0]==expected['FP16_RNE'],'direct_two_replays_bitwise':True}
 require(rows['input.f32']['equals_identity_F32'],'All actual input values must remain original')
 return {'fields':rows,'actual_compiler_lowering_observed':False,'tolerance_gate':None,'candidate_reference_modified':False,'full_model_math_qualified':False}
