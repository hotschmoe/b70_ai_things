"""Source-only compiler artifact recipe; never execute the diagnostic or GPU."""
import hashlib,shlex
from pathlib import Path
import qualify_owned_indexer_half_control_v2 as q
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'owned_indexer_half_discrimination_gpu_v2.cpp'
ORIGINAL_PLAN_SHA='52dbb91d9886e4685bd733b064166c202c6f8ad7bbbaa167a6a82a2f2fa3861b'
KERNELS=('Half37Expression','Half37Store','Half37Load')

def flags():
 q.require(q.sha(q.PLAN)==ORIGINAL_PLAN_SHA,'Frozen actual half source changed');binding=q.source_binding();actual=q.leaf_argv()
 q.require(actual.count('/leaf/owned_indexer_half_discrimination_gpu_v2.cpp')==1 and actual[0]=='/opt/intel/oneapi/compiler/2026.1/bin/icpx','Exact original compiler/source required')
 return {'original_compile_argv':actual,'original_source_sha256':q.sha(SOURCE),'source_binding':binding,'device_link_flags_remain_separate_authority':True}

def llvm_argv():
 original=flags()['original_compile_argv'];q.require(original[-2:]==['-o','/out/owned-indexer-half37'],'Exact original leaf output authority required')
 remove={'/sdk/build/libstrata_kernels.a','/sdk/build/libstrata_core.a','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt'}
 q.require(remove<=set(original),'Original link-only roster differs');argv=[v for v in original[:-2] if v not in remove]
 return argv+['-fsycl-device-only','-S','-emit-llvm','-o','/out/half37-device.ll']

def tool_inventory_shell():
 # Actual paths/versions/hashes must be recorded by root, never assumed present.
 names=('llvm-as','llvm-spirv','spirv-dis','llvm-readelf','llvm-objcopy','clang-offload-extract','clang-offload-bundler','ocloc')
 return q.SETVARS_PREFIX+' /opt/intel/oneapi/compiler/2026.1/bin/icpx --version; '+''.join('if command -v '+n+' >/dev/null 2>&1; then command -v '+n+'; else echo TOOL_UNAVAILABLE_'+n+'; fi; 'for n in names)

def recipes(output,pid):
 p,*_=q.modules();out=Path(output).resolve();q.require(type(pid)is int and pid>0,'Exact root producer PID required')
 argv=llvm_argv();mounts=[(p.SDK/'source','/sdk/source','ro'),(p.SDK/'build','/sdk/build','ro'),(HERE,'/leaf','ro'),(out,'/out','rw')];env={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu'}
 inventory=q.docker_recipe('b70-half37-tools-'+str(pid),p.IMAGE,mounts,env,tool_inventory_shell())
 llvm=q.docker_recipe('b70-half37-ir-'+str(pid),p.IMAGE,mounts,env,q.SETVARS_PREFIX+'/opt/intel/oneapi/compiler/2026.1/bin/icpx --version; exec '+shlex.join(argv))
 return {'inventory_command':inventory,'LLVM_command':llvm,'LLVM_argv':argv,'original_math_FLAGS_unchanged':True,'new_artifact_only_driver_flags':['-fsycl-device-only','-S','-emit-llvm'],'actual_original_executed_ELF_unchanged':True,'actual_new_compilation_observed':False,'actual_IR_or_ISA_observed':False}

def downstream(llvm_as,llvm_spirv,spirv_dis,readelf,helper_elf):
 # Paths are root-observed tool identities, whose versions/ELFs/libraries must
 # be pinned to compiler39992; missing tools refuse this step.
 tools=[llvm_as,llvm_spirv,spirv_dis,readelf]
 q.require(all(type(t)is str and t.startswith('/') for t in tools),'Actual discovered absolute tool roster required')
 return {'fresh_source_LLVM_to_SPIRV_commands':[[llvm_as,'/out/half37-device.ll','-o','/out/half37-device.bc'],[llvm_spirv,'/out/half37-device.bc','-o','/out/half37-device.spv'],[spirv_dis,'/out/half37-device.spv','-o','/out/half37-device.spvasm']],
         'original_executed_ELF_section_inventory':[readelf,'--sections','--wide',helper_elf],
         'original_ELF_embedded_SPIRV_extraction_recipe_pending_actual_section_inventory':True,
         'actual_runtime_JIT_ISA_saved':False,'offline_ISA_cannot_be_called_original_runtime_ISA':True,
         'offline_ISA_recipe_pending_actual_ocloc_help_and_device_target_and_runtime_IGC_identity':True}
