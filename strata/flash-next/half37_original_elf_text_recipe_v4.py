"""Actual original embedded SPIRV text; ISA probe stays distinct and unsupported."""
import shlex
from pathlib import Path
import half37_original_elf_spirv_v4 as original
import half37_compiler_lowering_recipe_v2 as inventory
import qualify_owned_indexer_half_control_v2 as q
PROBES=Path('/mnt/vm_8tb/b70/build/half37-driver-probes-v3-20261010')
def text_recipes(extracted,output,pid):
 q.require(type(pid)is int and pid>0,'Actual root PID required');binding=original.finalized_binding(extracted);inventory.inventory_binding();help_raw=original.consume(PROBES/'llvm-spirv-help.log',128<<10);q.require(original.digest(help_raw)=='634b491859022e32fbbd70f36121002b64ee44aab2e19cb04554c70df122ad82','Exact observed tool help required');help=help_raw.decode('ascii');q.require('--to-text' in help and 'Convert input SPIR-V binary to internal textual format' in help,'Actual supported original binary text option required');p,*_=q.modules();rows=[]
 for index in original.MODULES:
  argv=[inventory.INSTALLED+'llvm-spirv','--to-text','/images/module-'+str(index)+'.spv','-o','/out/module-'+str(index)+'.spt'];command=q.docker_recipe('b70-half37-original-text-'+str(pid)+'-'+str(index),p.IMAGE,[(Path(extracted).resolve(),'/images','ro'),(Path(output).resolve(),'/out','rw')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'},q.SETVARS_PREFIX+'exec '+shlex.join(argv));rows.append({'index':index,'argv':argv,'command':command})
 return {'original_extraction_binding':binding,'commands':rows,'input_is_original_executed_embedded_SPIRV':True,'new_translation_from_postlink_bitcode':False,'actual_JIT_ISA_observed':False,'actual_kernel_or_model_execution_requested':False}

def offline_IGC_inventory_shell():
 # Actual paths/versions/ELFs/dependencies must be joined to the ORIGINAL C388
 # runtime-after mapped library record before any offline artifact claim.
 return q.SETVARS_PREFIX+'for p in /usr/bin/ocloc /usr/local/bin/ocloc; do if [[ -x "$p" ]]; then readlink -f "$p"; sha256sum "$p"; ldd "$p"; "$p" --help; fi; done; '

def offline_IGC_inventory_recipe(output,pid):
 q.require(type(pid)is int and pid>0,'Actual root PID required');return q.docker_recipe('b70-half37-runtime-igc-help-'+str(pid),q.RUNTIME_IMAGE,[(Path(output).resolve(),'/out','rw')],{'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_CACHE_PERSISTENT':'0'},offline_IGC_inventory_shell())
