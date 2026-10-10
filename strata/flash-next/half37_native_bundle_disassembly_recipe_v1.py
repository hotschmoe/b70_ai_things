"""Supported offline disassembly of exact NEW queried native ELF module bytes."""
from pathlib import Path
from collect_half37_native_bundle_v1 import collect,sha,require
TOOL='/usr/bin/ocloc-26.22.1'
TOOL_SHA='7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HELP=Path('/mnt/vm_8tb/b70/build/half37-runtime-igc-supported-help-v1-20261010/capture-binding.json')
HELP_SHA='51de76d2aa410358204242d1458c1ce6cd72be601bf5ae0d966e0b5be5609232'
TARGET='0xe223'
def recipes(root,compile_root,output):
 require(sha(HELP)==HELP_SHA,'Actual supported help changed');proof=collect(root,compile_root);output=Path(output).absolute();require(not output.exists(),'New disassembly output root required')
 jobs=[]
 for module in proof['modules']:
  require(len(module['kernel_names'])==1,'Exact one named observed module required');job=output/('module-'+str(module['index']))
  jobs.append({'input':module['path'],'input_sha256':module['sha256'],'input_bytes':module['bytes'],'kernel_name':module['kernel_names'][0],'output':str(job),'image':IMAGE,'tool':TOOL,'tool_sha256':TOOL_SHA,'argv_inside': [TOOL,'disasm','-file','/input/native.bin','-dump','/out/dump','-device',TARGET],'mounts':[[module['path'],'/input/native.bin','ro'],[str(job),'/out','rw']],'runtime_GPU_replay_requested':False,'actual_loaded_bundle_native_binary':True,'actual_direct_launch_module_proven':False,'historical_original_JIT_ISA_observed':False,'scope':'Offline decoding of saved NEW queried bundle code. Current host target, no compilation or original-runtime options identity claim.'})
 return {'schema':1,'bundle_binding':proof,'supported_help_sha256':HELP_SHA,'jobs':jobs,'root_owned_lease_terminal_removal_required':True,'module_bytes_and_tool_SHA_required_before_after':True,'actual_tool_execution_by_proposal':False}
