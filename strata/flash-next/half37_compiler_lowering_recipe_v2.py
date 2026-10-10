"""Observed-tool successor. Probe first; no assumed IR/save-temps support."""
import hashlib,re,shlex
from pathlib import Path
import qualify_owned_indexer_half_control_v2 as q
from serial37_canonical_json_v3 import read_unique
INVENTORY=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/half37-compiler-absolute-tool-inventory-v1.json')
INVENTORY_SHA='7166be308a157ee6bdac43d3d524ece574369d6d7af63448cdd732b4235ef10d'
INSTALLED='/opt/intel/oneapi/compiler/2026.1/bin/compiler/'
NAMES=('clang++','clang-offload-bundler','clang-offload-extract','llvm-objcopy','llvm-spirv','sycl-post-link')
def inventory_binding():
 q.require(q.sha(INVENTORY)==INVENTORY_SHA,'Original actual absolute-tool metadata changed');row=read_unique(INVENTORY);q.require(row['compiler_image']==q.modules()[0].IMAGE and row['compiler_root']=='/opt/intel/oneapi/compiler/2026.1' and row['devices_exposed'] is False and row['compiler_or_GPU_execution'] is False,'Exact observed compiler metadata scope required')
 expected={'/opt/intel/oneapi/compiler/2026.1/bin/icpx',*(INSTALLED+n for n in NAMES)};tools={item['path']:item for item in row['actual_tool_files']};q.require(len(tools)==len(row['actual_tool_files'])==7 and set(tools)==expected,'Exact seven actual observed absolute tool files required')
 for item in tools.values():q.require(type(item['bytes'])is int and item['bytes']>0 and re.fullmatch('[0-9a-f]{64}',item['sha256']) and item['resolved'].startswith('/opt/intel/oneapi/compiler/2026.1/'),'Observed tool ELF/path/size record required')
 return {'path':str(INVENTORY),'sha256':INVENTORY_SHA,'tools':tools,'other_tool_locations_not_inventoried':True}

def probe_argv():
 inventory_binding();original=q.leaf_argv()
 return {'icpx-help':[original[0],'--help'],'icpx-hidden-help':[original[0],'--help-hidden'],'clang-help':[INSTALLED+'clang++','--help'],
         'actual-driver-phases':[original[0],'-###',*original[1:]],
         'save-temps-driver-phases':[original[0],'-###','-save-temps=obj',*original[1:]],
         **{name+'-help':[INSTALLED+name,'--help'] for name in NAMES if name!='clang++'}}

def save_temps_argv(probe_root,pid):
 """Admit original saved probe recipe/receipt/log/inspection before compile."""
 root=Path(probe_root).resolve();label='save-temps-driver-phases';expected=recipes(root,pid)[label];log=root/(label+'.log');row=read_unique(root/(label+'.receipt.json'))
 q.command_binding(row,log,root/(label+'.command.json'),root/(label+'.receipt.json'))
 q.require(row['command']==expected,'Actual probe command differs from exact original flags/owned recipe')
 obj=read_unique(root/(label+'.inspection.json'));q.terminal_binding(obj['State']);q.observed_container(obj,expected[expected.index('--name')+1],q.modules()[0].IMAGE,expected)
 text=log.read_text();q.require('IR output is not supported' not in text and 'error:' not in text.lower() and '-cc1' in text and '-fsycl-is-device' in text,'Actual supported device phase must be present')
 for token in ('-cl-fp32-correctly-rounded-divide-sqrt','llvm-spirv','sycl-post-link'):q.require(token in text,'Original backend/tool phase authority absent '+token)
 original=q.leaf_argv();return [original[0],'-save-temps=obj',*original[1:-1],'/out/half37-save-temps']

def device_frontend(dryrun_log):
 """Inspect the actual phase; do not run an invented clang lowering command."""
 text=Path(dryrun_log).read_text();rows=[]
 for line in text.splitlines():
  try:argv=shlex.split(line)
  except ValueError:continue
  if '-cc1' in argv and '-fsycl-is-device' in argv:rows.append(argv)
 q.require(len(rows)==1,'Exactly one actual device frontend phase required');argv=rows[0]
 q.require(argv[0] in (INSTALLED+'clang',INSTALLED+'clang++') and '/leaf/owned_indexer_half_discrimination_gpu_v2.cpp' in argv and '-O3' in argv,'Exact actual source/compiler/O3 phase required')
 return {'actual_device_cc1_argv':argv,'frontend_phase_only':True,'actual_link_backend_options_not_equated_with_frontend':True,'not_yet_executed_or_original_runtime_IR':True}

def recipes(output,pid):
 p,*_=q.modules();out=Path(output).resolve();q.require(type(pid)is int and pid>0,'Actual root PID required');mounts=[(p.SDK/'source','/sdk/source','ro'),(p.SDK/'build','/sdk/build','ro'),(q.HERE,'/leaf','ro'),(out,'/out','rw')];env={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'}
 return {name:q.docker_recipe('b70-half37-probe-'+str(pid)+'-'+name,p.IMAGE,mounts,env,q.SETVARS_PREFIX+'exec '+shlex.join(argv))for name,argv in probe_argv().items()}

def downstream_probes(original_elf):
 inventory_binding();return {'original_ELF_offload_extract_help':[INSTALLED+'clang-offload-extract','--help'],'original_ELF_objcopy_help':[INSTALLED+'llvm-objcopy','--help'],'original_executed_ELF':original_elf,'original_ELF_extract_argv_pending_observed_help':True,'SPIRV_reverse_to_LLVM_argv_pending_observed_llvm_spirv_help':True,'ISA_recipe_pending_actual_tool_help_target_and_runtime_IGC_identity':True,'fresh_save_temps_recompile_is_not_original_executed_ELF':True,'actual_original_runtime_JIT_ISA_saved':False}
