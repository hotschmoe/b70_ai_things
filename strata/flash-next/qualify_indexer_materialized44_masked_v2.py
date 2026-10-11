#!/usr/bin/env python3
"""Deliberate thin successor to frozen owned half lifecycle; root execution only."""
import argparse,ast,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'indexer-materialized44-masked-source-plan-v2.json'
SDK=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261011T165803Z-5xbo5_f0')
IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7'
ORIGINAL=HERE/'qualify_owned_indexer_half_control_v2.py'
ORIGINAL_SHA='ee25c64a5589b9faf770fce1334ebd8131774bf64f5fb50f7e40d621616f2a3c'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 from serial37_canonical_json_v3 import read_unique
 return read_unique(p)

def inside_runtime(helper,output,maps=None):
 # Standalone stdlib-only bootstrap: /harness contains only this mounted file.
 text=subprocess.check_output(['/usr/bin/ldd',str(helper)],text=True);require('not found'not in text,'Runtime ELF dependency unresolved');paths={word for line in text.splitlines()for word in line.split()if word.startswith('/')}
 if maps:
  for line in Path(maps).read_text().splitlines():
   cols=line.split(maxsplit=5)
   if len(cols)==6 and cols[5].startswith('/')and '.so'in Path(cols[5]).name:paths.add(cols[5])
 require(paths,'Runtime dependency roster missing');rows={str(Path(p).resolve()):{'sha256':sha(Path(p).resolve()),'bytes':Path(p).stat().st_size}for p in sorted(paths)};Path(output).write_text(json.dumps({'helper_sha256':sha(helper),'libraries':rows,'mapped_GPU_libraries_included':maps is not None,'environment':dict(os.environ),'actual_model_payload_read':False},indent=2)+'\n')

def builder():
 import native_rms_rsqrt37_proposal_v1 as original
 source=Path(original.__file__).read_bytes();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='builder_flags');ns={**vars(original),'SDK':SDK};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[actual44-builder-flags]','exec'),ns);return ns['builder_flags']()
def leaf_argv():
 fields=builder();require(fields['FLAGS'].startswith('-O3 -DNDEBUG'),'Actual flags required')
 import native_rms_rsqrt37_proposal_v1 as old
 node=next(n for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='compile_argv');namespace={**vars(old),'builder_flags':builder};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[actual44-leaf-compile-argv]','exec'),namespace);argv=namespace['compile_argv']();argv[argv.index('/leaf/native_rms_rsqrt37_gpu_v1.cpp')]='/leaf/indexer_materialized44_masked_gpu_v1.cpp';argv[argv.index('/out/native-rms-rsqrt37')]='/out/indexer-materialized44-masked';return argv

def source_binding():
 from source44_sdk_binding_v1 import binding as sdk_binding
 sdk_proof=sdk_binding();row=read(PLAN)
 for name,w in row['files'].items():require(sha(ROOT/name if not Path(name).is_absolute()else name)==w,'Current materialized source changed '+name)
 for name,w in row['actual_consumed_source_bindings'].items():require(sha(name)==w,'Actual SDK44 consumed source/archive changed '+name)
 from owned_layer3_qsa_runtime_contract_v4 import qsa_builder_flags
 fields=builder();flags=qsa_builder_flags(SDK,fields)
 return {'actual44_sdk':sdk_proof,'source_plan_sha256':sha(PLAN),'files':row['files'],'actual_consumed_source_bindings':row['actual_consumed_source_bindings'],'actual_builder_flags':fields,'indexer_builder_flags':flags,'actual_leaf_argv':leaf_argv(),'source_model_semantic_fix_qualified':False}

class Baseline:
 @staticmethod
 def finalized_binding(root,adjudication=None):
  require(adjudication is None,'Historical baseline adjudication cannot transfer')
  from admit_source_upload44_v1 import finalized_binding
  root=Path(root).resolve();record=read(root/'prerequisite.json');proof=finalized_binding(record['upload_receipt'],record['oracle_receipt'],record['pack_receipt']);require(proof==record['actual_fresh44_upload_binding'],'Fresh44 upload/source/ABI prerequisite changed')
  return {'engine_receipt':str(SDK/'receipt.json'),'cards':[0],'derived_from_actual_upload_only':True,'current_model_baseline_qualified':False},{'baseline_root':str(root),'prerequisite_path':str(root/'prerequisite.json'),'prerequisite_sha256':sha(root/'prerequisite.json'),'actual_fresh44_upload_binding':proof,'model_baseline_or_constructor_proof_transferred':False}
def modules():
 from types import SimpleNamespace
 import serial37_stdout_capture_v2 as capture
 from source_page_watchdog_v3 import guard,preserve
 from run_source_upload_oracle_full_v2 import full_buffered_identity
 p=SimpleNamespace(SDK=SDK,IMAGE=IMAGE)
 return p,capture,guard,preserve,full_buffered_identity,Baseline

def fixture_binding(root):
 from indexer_materialized44_masked_inputs_v2 import fixture_binding as admit
 return admit(root)
def compare_all_variants(root,fixture,prior):
 from indexer_materialized44_masked_inputs_v2 import own_inputs
 from indexer_materialized44_masked_control_v1 import recollect
 raw,gamma=own_inputs(prior['root']);return {'OFF':recollect(Path(root)/'mode0',raw,gamma,0),'ON':recollect(Path(root)/'mode1',raw,gamma,1),'normal_model_graph_qualified':False,'full_model_math_qualified':False}

def derived():
 import qualify_owned_indexer_half_control_v2 as old
 require(sha(ORIGINAL)==ORIGINAL_SHA,'Frozen original lifecycle source changed')
 selected=('observed_container','docker_recipe','expected_recipes','command_binding','runtime_receipt_binding','terminal_receipt_binding','terminal_binding','chronology','finalized_binding','main')
 source=ORIGINAL.read_text()
 source=source.replace('owned-indexer-half37','indexer-materialized44-masked').replace('b70-half37','b70-idx44mask').replace('b70.half37.plan','b70.idxmat44.plan')
 source=source.replace('from owned_layer3_qsa_control_v4 import producer_binding as admit_prior','from indexer_materialized44_masked_inputs_v2 import producer_binding as admit_prior').replace('from owned_indexer_half_control_v2 import native_trace','from indexer_materialized44_masked_control_v1 import native_trace')
 source=source.replace('/out/indexer-materialized44-masked --raw /inputs/raw.f32 --output /out/native-output --maps-prefix /out/half37','env STRATA_QSA_INDEXER_MATERIALIZED_HALF=0 /out/indexer-materialized44-masked --inputs /inputs --output /out/native-output/mode0 --mode 0 --maps-prefix /out/idx44mode0; env STRATA_QSA_INDEXER_MATERIALIZED_HALF=1 /out/indexer-materialized44-masked --inputs /inputs --output /out/native-output/mode1 --mode 1 --maps-prefix /out/idx44mode1; cat /out/idx44mode0.after /out/idx44mode1.after > /out/idx44mask.after')
 source=source.replace('/out/half37.after','/out/idx44mask.after')

 source=source.replace("r['schema']==2","r['schema']==1").replace("'schema':2","'schema':1")
 source=source.replace("report['finished_epoch']=time.time()","from indexer_materialized44_masked_inputs_v2 import operation_witness;write(out/'producer-operation-witness.json',operation_witness());report['producer_operation_witness_sha256']=sha(out/'producer-operation-witness.json');report['finished_epoch']=time.time()")
 source=source.replace("report['run_command']=command(recipe,'runtime',180","require(admit_prior(args.own_producer_root,role='predevice')==prior_proof,'Current complete source bytes changed before device');report['run_command']=command(recipe,'runtime',180")
 source=source.replace("report['run_command']=command(recipe,'runtime',180","require(0<=__import__('time').time()-report['pre_health']['finished_epoch']<=300,'Actual helper prehealth older than300s');report['run_command']=command(recipe,'runtime',180")
 source=source.replace("report['run_terminal']=terminal_owned(name,'runtime');terminal=time.time()","report['run_terminal']=terminal_owned(name,'runtime');require(admit_prior(args.own_producer_root,role='postrun')==prior_proof,'Current complete source bytes changed after runtime');terminal=time.time()")
 source=source.replace("admit_prior(args.own_producer_root)==prior_proof,'Actual prior V7 changed during owned runtime'","admit_prior(args.own_producer_root,role='postproof')==prior_proof,'Actual prior V7 changed during owned runtime'")
 nodes=[copy.deepcopy(n)for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in selected];require({n.name for n in nodes}==set(selected),'Exact inherited function roster required')
 namespace={**vars(old),**globals()};exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),str(ORIGINAL)+'[materialized44-explicit-port]','exec'),namespace)
 return namespace

def finalized_binding(root):
 binding=derived()['finalized_binding'](root)
 from indexer_materialized44_masked_inputs_v2 import admit_operation_witness
 admit_operation_witness(root,read(Path(root)/'report.json'));return binding
def main():
 if '--inside-runtime'in sys.argv:
  p=argparse.ArgumentParser();p.add_argument('--inside-runtime',type=Path,required=True);p.add_argument('--binding-output',type=Path,required=True);p.add_argument('--maps',type=Path);a=p.parse_args();inside_runtime(a.inside_runtime,a.binding_output,a.maps);return 0
 if '--leased'not in sys.argv:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 return derived()['main']()
if __name__=='__main__':raise SystemExit(main())
