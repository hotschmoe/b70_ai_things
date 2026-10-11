"""New43 compile/upload association; no prior oracle/SDK/current model claim."""
import hashlib,json,re
from pathlib import Path
from source43_sdk_binding_v1 import binding,require,SDK,IMAGE,sha,qualification_source_binding
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native-source-upload-plan-full-source43-v1.json'
PLAN_SHA='e2691ab4752627a2a8de8a58ca0ed07acfd42ff3158d6b2919fdb38a20f2b84f'
def layout_markers(log):
 rows=[]
 with Path(log).open('rb')as stream:
  while True:
   raw=stream.readline((4<<20)+1)
   if not raw:break
   require(len(raw)<=4<<20 and raw.endswith(b'\n'),'Bounded complete original upload line required')
   require(not raw.startswith(b'UPLOAD43_LAYOUT_ERROR'),'Actual43 layout/mode refusal')
   if not raw.startswith(b'UPLOAD43_LAYOUT '):continue
   require(len(raw)<=4096,'Bounded ABI marker required');words=raw[:-1].decode('ascii').split();fields={}
   for word in words[1:]:
    key,value=word.split('=',1);require(key not in fields and value.isdigit(),'Unique integer actual43 ABI layout fields required');fields[key]=int(value)
   require(set(fields)=={'sizeof_Verifier','align_Verifier','sizeof_QsaBuffers','align_QsaBuffers','half_offset','expanded_offset','idx_raw_offset','mode','source_generation'}and fields['mode']==0 and fields['source_generation']==43 and fields['sizeof_Verifier']>0 and fields['align_Verifier']>0 and fields['sizeof_QsaBuffers']>0 and fields['align_QsaBuffers']>0 and fields['half_offset']<fields['expanded_offset']<fields['idx_raw_offset']<fields['sizeof_QsaBuffers'],'Actual compiled43 public layout/disabled scope differs');rows.append(fields)
 require(len(rows)==1,'Exactly one actual43 layout marker required');return rows[0]

def elf_magic(path):
 with Path(path).open('rb')as elf:return elf.read(4)==b'\x7fELF'
def admission(oracle_receipt):
 qualification_source_binding();require(sha(PLAN)==PLAN_SHA,'Fresh43 upload plan changed');sdk=binding();plan=json.loads(PLAN.read_bytes());path=Path(oracle_receipt).resolve();r=json.loads(path.read_bytes());root=path.parent;require(r['passed']is True and type(r['build_rc'])is int and r['build_rc']==0 and r['libraries_unchanged']is True and r['image']==IMAGE and r['devices_exposed']is False,'Actual new43 isolated oracle compile required');require(Path(r['engine_receipt']).resolve()==SDK/'receipt.json'and r['engine_receipt_sha256']==sdk['engine_receipt_sha256']and r['plan_sha256']==PLAN_SHA==sha(root/'plan.snapshot.json')and json.loads((root/'plan.snapshot.json').read_bytes())==plan,'OldSDK/oracle/upload plan cannot transfer');require(r['oracle_source_sha256']==plan['oracle_source_sha256']==sha(root/'oracle.cpp')and sha(root/'source-upload-oracle')==r['binary_sha256'],'Actual new43 oracle source/ELF differs');require(elf_magic(root/'source-upload-oracle'),'Actual43 oracle ELF required');require({Path(n).name for n in r['library_sha256']}=={'libstrata_engine.a','libstrata_prefill.a','libstrata_spec.a','libstrata_kernels.a','libstrata_core.a','libstrata_kernels_cpu.a','libggml-cpu.a','libggml-base.a'}and len(r['library_sha256'])==8 and all(Path(n).is_relative_to(SDK/'build')and sha(n)==w for n,w in r['library_sha256'].items()),'All8 current actual43 link libraries required')
 command=r['command'];engine=json.loads((SDK/'receipt.json').read_bytes());require(command==compile_recipe(engine,root),'Exact actual43 source/header/library compile recipe required');require(command[:3]==['docker','run','--rm']and '--device'not in command and '--gpus'not in command and command[command.index('--network')+1]=='none'and IMAGE in command,'Actual no-device isolated41 compile recipe required');require('-I/src/sycl/include -I/src/include'in command[-1],'Actual43 SYCL headers must precede canonical headers')
 return {'actual43_sdk':sdk,'actual_fresh43_oracle_receipt':str(path),'actual_fresh43_oracle_receipt_sha256':sha(path),'actual_oracle_source_sha256':r['oracle_source_sha256'],'actual_oracle_binary_sha256':r['binary_sha256'],'actual_upload_qualified':False,'old_constructor_layout_or_runtime_transferred':False,'full_model_math_qualified':False}

def finalized_binding(upload_receipt,oracle_receipt,pack_receipt):
 """All original upload predicates plus explicit new43 SDK/layout association."""
 import ast,copy
 import c1_serve_controller_combined_v140_v3 as old
 sdk=binding();compile_binding=admission(oracle_receipt);runner=HERE/'run_source_upload43_oracle_v1.py';source=Path(old.__file__).read_text();source=source.replace("report.get('runner_generation') == 2","report.get('runner_generation') == 43").replace("'runner_generation': 2","'runner_generation': 43").replace('strata/flash-next/run_source_upload_oracle_full_v2.py','strata/flash-next/run_source_upload43_oracle_v1.py');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='strict_v2_upload_provenance');namespace={**vars(old),'UPLOAD_RUNNER_V2_SHA':sha(runner)};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[source43-original-upload-provenance]','exec'),namespace)
 strengthened=namespace['strict_v2_upload_provenance'](upload_receipt,oracle_receipt,SDK/'receipt.json');original=old.original_upload_gate(upload_receipt,oracle_receipt,SDK/'receipt.json',pack_receipt);report=json.loads(Path(upload_receipt).read_bytes());require(report['oracle_plan_sha256']==PLAN_SHA and len(report['cases'])==5,'Actual new43 exact five-case upload required');layouts={row['case']:layout_markers(Path(upload_receipt).parent/(row['case']+'.log'))for row in report['cases']};require(len(layouts)==5 and len({json.dumps(v,sort_keys=True)for v in layouts.values()})==1,'All5 profiles must use identical freshly compiled43 public layout')
 require(binding()==sdk,'Current43 ABI/header/source proof changed during readonly admission');return {'actual43_sdk':sdk,'actual_fresh43_compile':compile_binding,'actual_original_upload_predicates':original,'actual_strict_source_lifecycle':strengthened,'actual_all5_compiled43_layouts':layouts,'actual_source390_upload_qualified':True,'ordinary_weight_payload_readback_qualified':False,'materialized_model_or_graph_qualified':False,'old_constructor_or_source40_proof_transferred':False,'full_model_math_qualified':False}

def compile_recipe(engine,root):
 import shlex
 root=Path(root);build=Path(engine['build']);source=Path(engine['source_copy']);ggml=Path(engine['ggml_source']);libraries=[build/('libstrata_'+name+'.a')for name in('engine','prefill','spec','kernels','core','kernels_cpu')]+[build/'ggml/src/libggml-cpu.a',build/'ggml/src/libggml-base.a']
 compiler=['icpx','-std=c++20','-O2','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-DSTRATA_NATIVE_EXPERTS=1','-DSTRATA_SYCL_Q8_HC_BUILT=1','-I/src/sycl/include','-I/src/include','-I/src/third_party/ggml','-I/ggml/ggml/include','/work/oracle.cpp','-Wl,--start-group']+[str(p).replace(str(build),'/engine')for p in libraries]+['-Wl,--end-group','-lze_loader','-lcrypto','-qmkl=sequential','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/work/source-upload-oracle']
 return ['docker','run','--rm','--network','none','--user','1000:1000','-v',str(source)+':/src:ro','-v',str(build)+':/engine:ro','-v',str(ggml)+':/ggml:ro','-v',str(root)+':/work',IMAGE,'exec '+shlex.join(compiler)]
