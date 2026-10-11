"""Fixed original input fixture and exact native output recollection."""
import hashlib,json,os,re
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import canonical,read_unique
import own_q81_quantizer_intake_v1 as intake
import indexer_materialized43_masked_inputs_v2 as original
import own_q81_quantizer43_operation_v2 as operation
from independent_q8_1_activation_v1 import encode,NATIVE
require=intake.require
producer_binding=operation.producer_binding

def config(role):
 require(role in('hc_mixed','output_gated'),'Declared own role required');return ('OWN_Q81_43_V1 rows=4 cols=%d role=%s ids=own4\n'%(2560 if role=='hc_mixed'else 6144,role)).encode('ascii').ljust(64,b' ')
def prepare(producer_root,out,role='output_gated'):
 record=intake.prepare(producer_root,out,role);(Path(out)/'config.txt').write_bytes(config(role));return record

def fixture_binding(root):
 root=Path(root);operation.register_fixture(root);require(not any(p.is_symlink()for p in(root,*root.parents))and root.is_absolute(),'Original fixture alias refused');require({p.name for p in root.iterdir()}=={'own-input.f32','cpu-native.q81','source-plan.snapshot.json','input-binding.json','config.txt'}and all(p.is_file()and not p.is_symlink()for p in root.iterdir()),'Exact own fixture roster required');record=read_unique(root/'input-binding.json');require(record['schema']==1 and record['captured_operands_used']is False and record['reference_operands_replaced']is False and record['accepted_ids']==list(intake.IDS)and record['windows']==[[0,2],[2,1],[3,1]],'Exact original accepted role history required');require((root/'source-plan.snapshot.json').read_bytes()==intake.PLAN.read_bytes()and record['source_plan_sha256']==intake.source_binding(),'Actual intake source snapshot changed');prior_root=record['original_current_binding']['root'];prior=producer_binding(prior_root);require(canonical(prior)==canonical(record['original_current_binding']),'Current independent producer changed');work=read_unique(Path(prior_root)/'reference-work/work-report.json');parent=read_unique(Path(prior_root)/'report.json');require(canonical(work)==canonical(parent['reference_phase']['work']),'Own current work/report differs');own=work['own_layer3_projection'];identity=original.sha(parent['original_work_config']['model_identity']);raw,rows=intake.extract(own['root'],own,identity,record['role']);require(rows==record['source_operands']and record['rows']==4 and record['cols']in(2560,6144),'Actual original operand bindings changed');actual,_=intake.consume(root/'own-input.f32',len(raw),record['raw_sha256']);packet=encode(raw,4,record['cols'],NATIVE);got,_=intake.consume(root/'cpu-native.q81',len(packet),record['cpu_packet_sha256']);cfg,_=intake.consume(root/'config.txt',64);require(actual==raw and got==packet and cfg==config(record['role']),'Exact own raw/CPU packets/config required')
 # Inherited owned lifecycle expects these names; values come from the same
 # current original call, not an admission copied from saved PASS.
 normalized={**record,'own_producer_root':prior_root,'own_producer_binding':prior}
 return {'root':str(root),'fixture_sha256':hashlib.sha256((root/'input-binding.json').read_bytes()).hexdigest(),'record':normalized}

def native_trace(path):
 with Path(path).open('rb')as f:
  text=f.read((4<<20)+1)
 require(len(text)<=4<<20,'Bounded native trace required');text=text.decode('ascii');markers=[l for l in text.splitlines(keepends=True)if l.startswith('OWNQ81_')];require(all(l.endswith('\n')for l in markers),'Complete marker lines required');lines=[l[:-1]for l in markers];require(len(lines)==7 and re.fullmatch(r'OWNQ81_DEVICE backend=level_zero affinity=0 fp16=1 vendor=.+ driver=.+ name=.+',lines[0])and re.fullmatch(r'OWNQ81_FP_CONFIG flags=(?:[0-9]+,)+ observed_device_flags_only=1',lines[1])and re.fullmatch(r'OWNQ81_CONFIG rows=4 cols=(2560|6144) source_function_unchanged=1 shadow_not_internal=1',lines[2]),'Exact device/config marker roster required');mismatches=[]
 for r,line in enumerate(lines[3:6]):
  m=re.fullmatch(r'OWNQ81_FRAME route=%d graph_replay=%d fields=5 mismatch_blocks=(\d+) input_restored=1 packet_zero_restored=1'%(r,r),line);require(m is not None,'Exact direct/replay marker required');mismatches.append(int(m.group(1)))
 require(lines[-1]=='OWNQ81_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 shadow_not_internal=1 full_model_math_qualified=0'and 'OWNQ81_ERROR'not in text and 'OWNED_ERROR'not in text,'Actual normal marker retirement required');return {'mismatch_blocks':mismatches,'frames':3,'shadow_not_internal':True}

def bytes_exact(p,size):return intake.consume(Path(p),size)[0]
def compare_routes(root,fixture,prior):
 root=Path(root);require(not any(p.is_symlink()for p in(root,*root.parents)),'Native output alias refused');require({p.name for p in root.iterdir()}=={'consumed-raw.f32','consumed-cpu.q81','route0','route1','route2'},'Exact output root roster required');record=fixture['record'];require(canonical(prior)==canonical(record['own_producer_binding']),'Current producer prior differs');raw=bytes_exact(Path(fixture['root'])/'own-input.f32',record['rows']*record['cols']*4);expected=encode(raw,4,record['cols'],NATIVE);require(bytes_exact(root/'consumed-raw.f32',len(raw))==raw and bytes_exact(root/'consumed-cpu.q81',len(expected))==expected,'Exact consumed original/CPU echoes required');trace=native_trace(root.parents[1]/'runtime.log');routes=[]
 for r in range(3):
  d=root/('route'+str(r));require(d.is_dir()and not d.is_symlink()and{p.name for p in d.iterdir()}=={'packet_before.q81','input.f32','packet.q81','mismatch_blocks.i32','shadow.f32'},'Exact route fields required');actual=bytes_exact(d/'packet.q81',len(expected));require(bytes_exact(d/'input.f32',len(raw))==raw and not any(bytes_exact(d/'packet_before.q81',len(expected))),'Restored original input/zero output required');ids=[b for b in range(len(expected)//36)if actual[b*36:(b+1)*36]!=expected[b*36:(b+1)*36]];require(len(ids)==trace['mismatch_blocks'][r],'Actual mismatch marker/count differs');selected=bytes_exact(d/'mismatch_blocks.i32',len(ids)*4);require(np.frombuffer(selected,dtype='<i4').tolist()==ids,'Exact actual mismatch block selection required');shadow=bytes_exact(d/'shadow.f32',len(ids)*35*4);require(np.isfinite(np.frombuffer(shadow,dtype='<f4')).all(),'Finite diagnostic shadow required');details=[]
  for j,b in enumerate(ids):
   a=np.frombuffer(shadow[j*140:(j+1)*140],dtype='<f4');block=np.frombuffer(raw[b*128:(b+1)*128],dtype='<f4');amax=np.max(np.abs(block));div=np.float32(amax/np.float32(127));details.append({'block':b,'device_separate_shadow_amax_F32':float(a[0]),'device_separate_shadow_d32_F32':float(a[1]),'device_separate_shadow_raw_sum_F32':float(a[2]),'device_separate_shadow_quotients_F32':a[3:].tolist(),'actual_packet_stored_d_F16_hex':actual[b*36:b*36+2].hex(),'cpu_packet_stored_d_F16_hex':expected[b*36:b*36+2].hex(),'cpu_amax_F32':float(amax),'cpu_divisor_F32':float(div),'shadow_not_internal_witness':True})
  routes.append({'input_sha256':hashlib.sha256(raw).hexdigest(),'packet_sha256':hashlib.sha256(actual).hexdigest(),'packet_bytes':len(actual),'cpu_packet_exact':actual==expected,'mismatch_blocks':ids,'shadow_sha256':hashlib.sha256(shadow).hexdigest(),'shadow_bytes':len(shadow),'mismatch_details':details})
 require(routes[0]==routes[1]==routes[2],'Every native direct/replay byte hash required');return {'routes':routes,'direct_two_replays_bitwise':True,'cpu_native_packet_exact':routes[0]['cpu_packet_exact'],'shadow_not_native_internal':True,'tolerance_gate':None,'full_model_math_qualified':False,'performance_qualified':False}

operation_witness=operation.operation_witness
admit_operation_witness=operation.admit_operation_witness

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--producer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['hc_mixed','output_gated'],default='output_gated');a=p.parse_args();r=prepare(a.producer_root,a.output,a.role);print(json.dumps({'root':str(a.output),'raw_sha256':r['raw_sha256'],'full_model_math_qualified':False}))
