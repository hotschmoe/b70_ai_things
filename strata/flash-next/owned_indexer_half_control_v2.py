"""Strict own-source fixture and full raw diagnostic recollection."""
import hashlib,re,json
from pathlib import Path
import numpy as np
from serial37_canonical_json_v3 import read_unique,canonical
from owned_layer3_qsa_contract_v1 import require
import owned_layer3_qsa_control_v4 as own
import owned_indexer_half_discrimination_v1 as proposal

def sha(raw):return hashlib.sha256(raw).hexdigest()
def consume(path,size):
 path=Path(path);require(path.is_file() and not path.is_symlink(),'Regular confined raw input required');before=path.stat();raw=path.read_bytes();require(len(raw)==size and path.stat()==before and path.read_bytes()==raw and path.stat()==before,'Actual consumed raw extent/stat/bytes changed');return raw

def prepare(original,output):
 from qualify_owned_indexer_half_control_v2 import source_binding,PLAN
 before=own.fixture_binding(original);raw=consume(Path(before['root'])/'raw.f32',2048);proposal.original_input(raw)
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
 with (out/'raw.f32').open('xb') as f:f.write(raw)
 binding=source_binding();snapshot=PLAN.read_bytes();(out/'source-plan.snapshot.json').write_bytes(snapshot)
 record={'schema':2,'original_own_fixture_binding':before,'own_producer_root':before['record']['own_producer_root'],'own_producer_binding':before['record']['own_producer_binding'],'raw_sha256':sha(raw),'raw_bytes':2048,'source_plan_sha256':sha(snapshot),'source_binding':binding,'captured_operands_or_values_used':False,'full_model_math_qualified':False}
 require(own.fixture_binding(original)==before and consume(Path(before['root'])/'raw.f32',2048)==raw,'Original own inputs changed during extraction')
 (out/'input-binding.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');return record

def fixture_binding(root):
 from qualify_owned_indexer_half_control_v2 import source_binding,PLAN
 root=Path(root).resolve();require(not root.is_symlink(),'Confined own fixture required');record=read_unique(root/'input-binding.json')
 require(type(record['schema']) is int and record['schema']==2 and record['captured_operands_or_values_used'] is False and record['full_model_math_qualified'] is False,'Exact own fixture generation/scope required')
 require({p.name for p in root.iterdir()}=={'raw.f32','input-binding.json','source-plan.snapshot.json'} and all(not p.is_symlink() and p.is_file() for p in root.iterdir()),'Exact regular own fixture roster required')
 require((root/'source-plan.snapshot.json').read_bytes()==PLAN.read_bytes() and record['source_plan_sha256']==sha(PLAN.read_bytes()) and canonical(record['source_binding'])==canonical(source_binding()),'Exact current fixture/source snapshot required')
 original=own.fixture_binding(record['original_own_fixture_binding']['root']);require(canonical(original)==canonical(record['original_own_fixture_binding']) and record['own_producer_root']==original['record']['own_producer_root'] and canonical(record['own_producer_binding'])==canonical(original['record']['own_producer_binding']),'Current genuine own producer/fixture associations required')
 raw=consume(root/'raw.f32',2048);require(record['raw_bytes']==2048 and record['raw_sha256']==sha(raw) and raw==consume(Path(original['root'])/'raw.f32',2048),'Exact saved independently original-projected raw required');proposal.original_input(raw)
 return {'root':str(root),'fixture_sha256':sha((root/'input-binding.json').read_bytes()),'record':record}

def native_trace(path):
 text=Path(path).read_text();lines=text.splitlines(keepends=True);markers=[l for l in lines if l.startswith('HALF37_')];require(all(l.endswith('\n') for l in markers),'Complete final native marker newline required');m=[l.rstrip('\n') for l in markers]
 require(len(m)==6 and re.fullmatch(r'HALF37_DEVICE backend=level_zero affinity=0 fp16=1 vendor=.+ driver=.+ name=.+',m[0]),'Exact device/marker inventory required')
 require(re.fullmatch(r'HALF37_FP_CONFIG flags=(?:[0-9]+,)+ observed_device_flags_only=1 compiler_lowering_unobserved=1',m[1]),'Actual device FP flags required')
 require(m[2:5]==['HALF37_FRAME route=%d graph_replay=%d fields=4 own_input_restored=1 values=512'%(r,r) for r in range(3)] and m[5]=='HALF37_RESULT frames=3 graph_retired=1 owned_allocations_freed=1 compiler_lowering_qualified=0 full_model_math_qualified=0','Exact ordered direct/replays/retirement required')
 require(not re.search(r'HALF37_ERROR|OWNED_ERROR|RMS37_ERROR',text),'Actual native error refused');return {'frames':3,'fields':4,'lowering_cause_qualified':False}

def compare_routes(root,fixture,prior):
 root=Path(root);raw=consume(Path(fixture['root'])/'raw.f32',2048);native_trace(root.parents[1]/'runtime.log')
 require(canonical(prior)==canonical(fixture['record']['own_producer_binding']),'Exact original producer prior required')
 require({p.name for p in root.iterdir()}=={'consumed-raw.f32','route0','route1','route2'},'Exact complete native output roster required');require(all(not p.is_symlink() for p in root.iterdir()),'No native output symlinks')
 expected=proposal.preregistered(raw);require(consume(root/'consumed-raw.f32',2048)==raw,'Exact consumed own input echo required');report={}
 for r in range(3):
  route=root/('route'+str(r));require(route.is_dir() and {p.name for p in route.iterdir()}==set(proposal.FIELDS),'Exact complete route raw roster required');values={}
  for field,(dtype,size) in proposal.FIELDS.items():
   b=consume(route/field,size);require(np.isfinite(np.frombuffer(b,dtype=dtype)).all(),'Finite native raw values required');values[field]=b
  require(values['input.f32']==raw,'Actual own input restored on every route')
  widened=np.frombuffer(values['materialized.f16'],dtype='<f2').astype('<f4').tobytes();require(widened==values['materialized.f32'],'Observable actual F16 store/load bytes must rejoin widened F32')
  report[str(r)]={n:{'sha256':sha(b),'bytes':len(b),'identity_F32_match':b==expected['identity_F32'],'host_FP16_RNE_F32_match':b==expected['FP16_RNE_widened_F32'],'host_FP16_RNE_bytes_match':b==expected['FP16_RNE']} for n,b in values.items()}
 require(report['0']==report['1']==report['2'],'All raw direct/two replay bytes bitwise required')
 return {'fields':report['0'],'direct_two_replays_bitwise':True,'actual_materialized_F16_to_F32_join_verified':True,'compiler_lowering_cause_qualified':False,'candidate_reference_modified':False,'tolerance_gate':None,'full_model_math_qualified':False}
