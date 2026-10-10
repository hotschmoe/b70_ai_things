"""Translate six genuine saved post-link modules, then observed binary-to-text."""
import hashlib,shlex
from pathlib import Path
from serial37_canonical_json_v3 import read_unique
import qualify_owned_indexer_half_control_v2 as q
import half37_compiler_lowering_recipe_v2 as inventory
CAPTURE=Path('/mnt/vm_8tb/b70/build/half37-save-temps-v1-20261010')
CAPTURE_SHA='db27e681ce9572c5322027a2575b26b4078687c01d2e02373aaa78c26e21e231'
PROBES=Path('/mnt/vm_8tb/b70/build/half37-driver-probes-v3-20261010')
STEM='owned_indexer_half_discrimination_gpu_v2-sycl-spir64-unknown-unknown'
MODULES={21:'_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI16Half37ExpressionEE',24:'_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI11Half37StoreEE',25:'_ZTSN4sycl3_V16detail19__pf_kernel_wrapperI10Half37LoadEE',33:'_ZTS16Half37Expression',36:'_ZTS11Half37Store',38:'_ZTS10Half37Load'}

def consume(path,limit):
 path=Path(path);q.require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'Bounded regular original code artifact required');before=path.stat();raw=path.read_bytes();q.require(len(raw)==before.st_size and path.stat()==before and path.read_bytes()==raw and path.stat()==before,'Actual consumed code bytes changed');return raw

def original_modules(root=CAPTURE):
 root=Path(root).resolve();q.require(q.sha(root/'capture-binding.json')==CAPTURE_SHA,'Exact successful save-temps binding changed');record=read_unique(root/'capture-binding.json');q.require(record['passed'] is True and record['helper_executed'] is False and record['original_executed_ELF_unchanged'] is True and record['actual_runtime_JIT_ISA_observed'] is False and record['compiler_image']==q.modules()[0].IMAGE,'Exact fresh-recompile-only artifact scope required');expected=q.leaf_argv();q.require(record['actual_argv']==[expected[0],'-save-temps=obj',*expected[1:-1],'/out/half37-save-temps'],'All original compile/link options required');rows=[]
 for index,symbol in MODULES.items():
  name=STEM+'_'+str(index);raw=consume(root/(name+'.sym'),4096);bc=consume(root/(name+'.bc'),1<<20);q.require(raw.decode('ascii').splitlines()==[symbol] and bc[:4]==b'BC\xc0\xde','Exact six saved kernel symbols/LLVM bitcode required')
  for suffix,value in (('.sym',raw),('.bc',bc)):q.require(record['artifact_sha256'][name+suffix]==hashlib.sha256(value).hexdigest(),'Exact successful compiler artifact SHA required')
  rows.append({'index':index,'symbol':symbol,'bc':name+'.bc','bc_sha256':hashlib.sha256(bc).hexdigest(),'bytes':len(bc),'sym_sha256':hashlib.sha256(raw).hexdigest()})
 return {'capture_binding_sha256':CAPTURE_SHA,'root':str(root),'modules':rows,'original_runtime_JIT_ISA_observed':False}

def translator_options(log):
 rows=[]
 for line in consume(log,128<<10).decode('ascii').splitlines():
  try:argv=shlex.split(line)
  except ValueError:continue
  tool=inventory.INSTALLED+'llvm-spirv'
  if tool in argv and '--out-ext=spv' in argv:rows.append(argv[argv.index(tool)+1:])
 q.require(len(rows)==1 and rows[0][:1]==['-o'] and len(rows[0])>4,'Exactly one actual complete saved-driver translator phase required');options=rows[0][2:-1];q.require(all(arg.startswith('-') for arg in options) and '-spirv-max-version=1.5' in options and any(arg.startswith('-spirv-ext=')for arg in options),'Actual translator options required without guessed defaults');return options

def recipes(output,pid):
 q.require(type(pid)is int and pid>0,'Actual root PID required');inventory.inventory_binding();binding=original_modules();options=translator_options(PROBES/'save-temps-driver-phases.log');help=consume(PROBES/'llvm-spirv-help.log',128<<10).decode('ascii');q.require('--to-text' in help and 'Convert input SPIR-V binary to internal textual format' in help,'Actual supported binary-to-text tool option required');out=Path(output).resolve();p,*_=q.modules();mounts=[(CAPTURE,'/code','ro'),(out,'/out','rw')];env={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'};rows=[]
 for item in binding['modules']:
  label='module-'+str(item['index']);binary='/out/'+label+'.spv';text='/out/'+label+'.spt';translate=[inventory.INSTALLED+'llvm-spirv',*options,'/code/'+item['bc'],'-o',binary];dump=[inventory.INSTALLED+'llvm-spirv','--to-text',binary,'-o',text]
  rows.append({'module':item,'translation_argv':translate,'text_argv':dump,'command':q.docker_recipe('b70-half37-text-'+str(pid)+'-'+str(item['index']),p.IMAGE,mounts,env,q.SETVARS_PREFIX+shlex.join(translate)+'; exec '+shlex.join(dump))})
 return {'original_saved_modules':binding,'actual_original_translator_options':options,'commands':rows,'new_translation_not_original_executed_embedded_SPIRV':True,'actual_kernel_or_GPU_execution_requested':False,'actual_IR_or_ISA_observed':False,'full_model_math_qualified':False}
