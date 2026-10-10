"""Original SPIRV/current host offline target; historical runtime proof only."""
import hashlib,json,re,shlex
from pathlib import Path
import half37_original_elf_spirv_v4 as original
import half37_runtime_ocloc_probe_recipe_v1 as inventory
import qualify_owned_indexer_half_control_v2 as q
from serial37_canonical_json_v3 import canonical,read_unique
HERE=Path(__file__).resolve().parent
IMAGES=Path('/mnt/vm_8tb/b70/build/half37-original-elf-extraction-v4-20261010')
PUBLICATION_SHA='4d63475831fa62f477a9a769a4ce53c327d02ec80c4da698a73407c600f1cca1'
BYTE_JOIN=Path('/mnt/vm_8tb/b70/build/half37-original-image-current-byte-join-v1.json')
BYTE_JOIN_SHA='18a5df8a933a13cd9f7eb84f71c2ad2eb1b6032b61caa6ada5761a9083e581f7'
HELP=Path('/mnt/vm_8tb/b70/build/half37-runtime-igc-supported-help-v1-20261010')
HELP_SHA='51de76d2aa410358204242d1458c1ce6cd72be601bf5ae0d966e0b5be5609232'
CENSUS=Path('/mnt/vm_8tb/b70/build/half37-runtime-igc-library-census-v1-20261010')
CENSUS_SHA='f16dc3e7f10a0635a4cb2f890218560334826b5beb43bacc0accc84cafb3382d'
TARGET='0xe223'
OPTION='-cl-fp32-correctly-rounded-divide-sqrt'
require=q.require

def sha(path):return original.digest(original.consume(path,4<<20))
def captured(root,expected):
 require(sha(root/'capture-binding.json')==expected,'Exact actual tool capture changed');value=read_unique(root/'capture-binding.json');require(value['passed']is True and value['runtime_image']==q.RUNTIME_IMAGE and value['original_runtime_JIT_ISA_observed']is False,'Actual offline inventory/image/scope differs')
 for name,want in value['artifact_sha256'].items():require(Path(name).name==name and sha(root/name)==want,'Current original tool artifact differs '+name)
 for file in sorted(root.glob('module-*.receipt.json')):
  label=file.name[:-len('.receipt.json')];row=read_unique(file);command=read_unique(root/(label+'.command.json'));q.command_binding(row,root/(label+'.log'),root/(label+'.command.json'),file);obj=read_unique(root/(label+'.inspection.json'));q.terminal_binding(obj['State']);q.observed_container(obj,command[command.index('--name')+1],q.RUNTIME_IMAGE,command)
 return value

def evidence_binding():
 inventory.inventory_binding();captured(HELP,HELP_SHA);captured(CENSUS,CENSUS_SHA);compile_help=original.consume(HELP/'module-compile-help.log',128<<10).decode('ascii');disasm_help=original.consume(HELP/'module-disasm-help.log',128<<10).decode('ascii');require(all(t in compile_help for t in ('-spirv_input','-out_dir','-output','-options','-64','The hexadecimal value represents device ID.'))and all(t in disasm_help for t in ('-file','-dump','-device')),'Exact observed offline options required');census=read_unique(CENSUS/'original-library-census.json');require(census['passed']is True and census['original_report_sha256']==original.REPORT_SHA and census['offline_tool_library_loading_or_actual_JIT_ISA_proven']is False and all(census['current_original_runtime_library_rows'][p]['sha256']==r['sha256']and census['current_original_runtime_library_rows'][p]['bytes']==r['bytes'] for p,r in inventory.REQUIRED_LIBRARIES.items()),'Actual original current library census differs');return {'help_sha256':HELP_SHA,'census_sha256':CENSUS_SHA,'original_libraries_current_bytes_censused':True,'ldd_cannot_prove_IGC_dynamic_loading':True}

def current_host_target():
 # This is present-host metadata, deliberately NOT original native/JIT PCI identity.
 rows=[]
 for render,pci in (('renderD128','0000:0b:00.0'),('renderD129','0000:44:00.0')):
  device=(Path('/sys/class/drm')/render/'device').resolve();vendor=(device/'vendor').read_text().strip();identifier=(device/'device').read_text().strip();require(device.name==pci and vendor=='0x8086' and identifier==TARGET,'Exact current host PCI target metadata changed');rows.append({'render':render,'PCI':pci,'vendor':vendor,'device':identifier})
 return {'rows':rows,'target_hex':TARGET,'current_host_metadata_only':True,'original_runtime_PCI_or_JIT_target_association_observed':False}

def input_binding():
 require(sha(BYTE_JOIN)==BYTE_JOIN_SHA and sha(IMAGES/'extraction.json')==PUBLICATION_SHA,'Exact historical publication/current-byte join changed');record=read_unique(IMAGES/'extraction.json');require(record['schema']==4 and record['original_binding']['helper_sha256']==original.HELPER_SHA and record['original_binding']['report_sha256']==original.REPORT_SHA and record['original_binding']['current_full_runtime_requalification_performed']is True,'Historical publication source scope differs');raw=original.consume(original.ROOT/'build/owned-indexer-half37',original.MAX_ELF);require(original.digest(raw)==original.HELPER_SHA,'Current original ELF differs');images=original.elf_images(raw);require({p.name for p in IMAGES.iterdir()}=={'extraction.json',*('module-'+str(i)+'.spv'for i in original.MODULES)},'Exact original six-image publication roster required')
 for index,image in images.items():
  expected={k:v for k,v in image.items()if k not in ('raw','SPIRV')};expected['file']='module-'+str(index)+'.spv';require(canonical(record['images'][str(index)])==canonical(expected)and original.consume(IMAGES/expected['file'],1<<20)==image['raw'],'Current original ELF symbol/SPIRV/publication association changed')
 require(original.consume(original.ROOT/'build/owned-indexer-half37',original.MAX_ELF)==raw,'Original ELF changed during bounded byte join');require(OPTION in ' '.join(record['original_binding']['actual_compile_argv']),'Original recorded device backend option absent');return {'publication_sha256':PUBLICATION_SHA,'original_helper_sha256':original.HELPER_SHA,'current_original_six_symbol_byte_join':True,'historical_full_runtime_admission_recorded':True,'full_current_runtime_requalification_repeated':False,'original_runtime_JIT_ISA_observed':False}

def compile_argv(index):
 require(type(index)is int and index in original.MODULES,'Exact declared original image index required');return [inventory.TOOL,'compile','-spirv_input','-file','/images/module-'+str(index)+'.spv','-device',TARGET,'-64','-out_dir','/out','-output','module-'+str(index),'-options',OPTION]
def env(index):return {'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_CACHE_PERSISTENT':'0','LD_DEBUG':'libs,files','LD_DEBUG_OUTPUT':'/out/module-'+str(index)+'-loader'}
def compile_recipes(output,pid):
 require(type(pid)is int and pid>0,'Exact root producer PID required');evidence=evidence_binding();inputs=input_binding();target=current_host_target();out=Path(output).resolve();commands={}
 for index in original.MODULES:
  argv=compile_argv(index);guard='printf "%s  %s\\n" '+shlex.quote(inventory.TOOL_SHA)+' '+shlex.quote(inventory.TOOL)+' | sha256sum --check --strict; ';commands[str(index)]=q.docker_recipe('b70-half37-offline-compile-'+str(pid)+'-'+str(index),q.RUNTIME_IMAGE,[(IMAGES,'/images','ro'),(out,'/out','rw')],env(index),q.SETVARS_PREFIX+guard+'exec '+shlex.join(argv))
 return {'commands':commands,'input_binding':inputs,'actual_tool_evidence':evidence,'target':target,'offline_current_target_compile_only':True,'original_JIT_ISA_or_lowering_cause_qualified':False,'kernel_or_model_execution_requested':False,'actual_loaded_IGC_witness_pending':True,'original_runtime_JIT_target_inferred_from_B70_name':False,'all_original_runtime_backend_options_identical':False,'omitted_original_target_conditional_option':'-ftarget-register-alloc-mode=pvc:-ze-intel-enable-auto-large-GRF-mode','conditional_PVC_option_mapping_to_current_e223_proven':False}

def disasm_recipes(compile_manifest,output,pid):
 """No invented output naming: root provides exact actual observed binary roster."""
 require(type(pid)is int and pid>0,'Exact root producer PID required');evidence_binding();input_binding();target=current_host_target();require(type(compile_manifest)is dict and set(compile_manifest)=={str(i)for i in original.MODULES},'Actual complete six offline output manifests required');out=Path(output).resolve();commands={}
 for key,row in compile_manifest.items():
  require(type(row)is dict and set(row)=={'path','sha256','bytes','input_sha256','target_hex'}and row['target_hex']==TARGET and row['input_sha256']==sha(IMAGES/('module-'+key+'.spv')),'Actual offline binary/input/target association differs');path=Path(row['path']);raw=original.consume(path,8<<20);require(type(row['bytes'])is int and len(raw)==row['bytes'] and original.digest(raw)==row['sha256'] and raw[:4]==b'\x7fELF','Exact observed offline Zebin ELF required');argv=[inventory.TOOL,'disasm','-file','/input/'+path.name,'-dump','/out/module-'+key,'-device',TARGET];commands[key]=q.docker_recipe('b70-half37-offline-disasm-'+str(pid)+'-'+key,q.RUNTIME_IMAGE,[(path,'/input/'+path.name,'ro'),(out,'/out','rw')],env(int(key)),q.SETVARS_PREFIX+'exec '+shlex.join(argv))
 return {'commands':commands,'current_target':target,'offline_current_target_disassembly_only':True,'actual_runtime_JIT_ISA_observed':False,'loaded_library_match_is_not_JIT_cause_authority':True}
