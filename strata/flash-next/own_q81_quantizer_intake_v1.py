"""Bounded own-input intake; future fresh SDK43 authority remains separate."""
import hashlib,os,stat,struct,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from independent_q8_1_activation_v1 import encode,NATIVE
HERE=Path(__file__).resolve().parent;IDS=(248045,8678,198,15666)
ANCHOR=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T052714Z-k9s4qsio/source')
FILES=('sycl/src/kernels/cuda/native_mmvq.dp.cpp','sycl/include/strata/kernels/native_mmvq.hpp','sycl/include/strata/kernels/q8_1_finite.hpp')
PLAN=HERE/'own-q81-quantizer-intake-source-plan-v1.json'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sig(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def consume(path,limit,expected=None):
 p=Path(path);require(p.is_absolute()and not any(x.is_symlink()for x in(p,*p.parents))and p.is_file(),'Exact nonaliased raw file required');before=sig(p.stat());require(before[2]<=limit,'Bounded file before read required')
 with p.open('rb')as f:
  require(sig(os.fstat(f.fileno()))==before,'Raw descriptor changed');raw=f.read(limit+1);require(sig(os.fstat(f.fileno()))==before,'Raw descriptor changed during read')
 require(len(raw)==before[2]and sig(p.stat())==before,'Raw path changed during read');digest=hashlib.sha256(raw).hexdigest();require(expected is None or digest==expected,'Consumed raw SHA differs')
 require(p.read_bytes()==raw and sig(p.stat())==before,'Raw reread changed');return raw,{'path':str(p),'bytes':len(raw),'sha256':digest,'stat5':before}
def source_binding():
 plan=read_unique(PLAN);root=HERE.parents[1]
 for name,want in plan['files'].items():consume(root/name,64<<20,want)
 return hashlib.sha256(PLAN.read_bytes()).hexdigest()
def candidate_quantizer_semantics(source):
 """Equality of these three files is not SDK43 ABI/build/runtime admission."""
 plan=read_unique(PLAN);rows={}
 for name,want in plan['source41_quantizer_anchor'].items():_,rows[name]=consume(Path(source)/name,4<<20,want)
 return {'exact_quantizer_source_rows':rows,'actual_fresh43_build_or_ABI_admitted':False,'runtime_qualified':False}
def extract(root,own,identity,role):
 root=Path(root);require(role in('hc_mixed','output_gated'),'Declared own ordinary quantizer role required');records=own['records'];require(len(records)==4,'Exact four own tokens required');arrays=[];bindings=[]
 for pos,row in enumerate(records):
  require(type(row['position'])is int and row['position']==pos and type(row['token'])is int and row['token']==IDS[pos]and row['route']=='verifier'and type(row['predecessor_cells'])is int and row['predecessor_cells']==pos and row['captured_inputs_or_states_used']is False,'Exact token/layer/accept predecessor mapping required')
  if role=='hc_mixed':binding=row['own_HC_mixed_input'];shape=[2560];wanted='blk.3.hc_attn_mixed'
  else:
   projected=[p for p in row['projections']if p['role']=='blk.3.attn_output.weight'];require(len(projected)==1,'Exact output projection role required');binding=projected[0]['own_input_snapshot'];shape=[6144];wanted='blk.3.attn_output.weight.own_input'
  p=Path(binding['path']);require(p.parent==root and binding['shape']==shape and binding['role']==wanted and type(binding['position'])is int and binding['position']==pos and binding['source_identity_sha256']==identity and binding['actual_original_provider']is True and binding['captured_operand_used']is False and binding['snapshot_is_math_input']is False and binding['encoding']=='LE_F32','Exact own source role/identity/shape required')
  raw,observed=consume(p,shape[0]*4,binding['sha256']);require(len(raw)==shape[0]*4==binding['bytes']and all(abs(x)!=float('inf')and x==x for x in struct.unpack('<%df'%shape[0],raw)),'Exact finite own operand extent required');arrays.append(raw);bindings.append(observed)
 return b''.join(arrays),bindings

def prepare(producer_root,out,role='output_gated'):
 """Root-only: complete original predicates run before/after, never saved PASS."""
 import indexer_materialized41_masked_inputs_v2 as original
 source=source_binding();owner=os.getpid();started=time.time();before=original.producer_binding(producer_root)
 work=read_unique(Path(producer_root)/'reference-work/work-report.json');parent=read_unique(Path(producer_root)/'report.json');require(canonical(work)==canonical(parent['reference_phase']['work']),'Exact own producer/work binding required');own=work['own_layer3_projection'];identity=original.sha(parent['original_work_config']['model_identity']);raw,bindings=extract(own['root'],own,identity,role);cols=6144 if role=='output_gated'else 2560;packet=encode(raw,4,cols,NATIVE)
 require(os.getpid()==owner and canonical(original.producer_binding(producer_root))==canonical(before)and source_binding()==source,'Original operation/source changed during intake');out=Path(out);out.mkdir(parents=True,exist_ok=False)
 for name,value in(('own-input.f32',raw),('cpu-native.q81',packet),('source-plan.snapshot.json',PLAN.read_bytes())):
  with(out/name).open('xb')as f:f.write(value)
 import json
 record={'schema':1,'producer_pid':owner,'started_epoch':started,'finished_epoch':time.time(),'original_current_binding':before,'rows':4,'cols':cols,'layer':3,'role':role,'accepted_ids':list(IDS),'windows':[[0,2],[2,1],[3,1]],'source_operands':bindings,'raw_sha256':hashlib.sha256(raw).hexdigest(),'cpu_packet_sha256':hashlib.sha256(packet).hexdigest(),'source_plan_sha256':source,'captured_operands_used':False,'reference_operands_replaced':False,'actualSDK43_build_or_runtime_admitted':False,'full_model_math_qualified':False}
 (out/'input-binding.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');return record
