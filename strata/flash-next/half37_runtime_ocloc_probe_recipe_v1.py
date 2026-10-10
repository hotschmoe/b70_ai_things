"""Observed offline ocloc help/census only; no original runtime JIT authority."""
import hashlib,json,shlex,stat,subprocess
from pathlib import Path
REPORT_SHA='5e66575dcf49c66fd1118c9af14e118217fa672ea5ed42217e506cba1b110cc3'
ORIGINAL_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-indexer-half-control-v2-run-v1')
def require(ok,message):
 if not ok:raise ValueError(message)
INVENTORY=Path('/mnt/vm_8tb/b70/build/half37-runtime-igc-inventory-v4-20261010')
INVENTORY_SHA='08551dba10357b9ff74f0d39c6c113e0141b46e970af6d1d86f1480e9c1a52ff'
TOOL='/usr/bin/ocloc-26.22.1'
TOOL_SHA='7a418c1e7dfc62b6fb3895bea111647fa3e1ff600bb69f6be2a3114b90c41eac'
REQUIRED_LIBRARIES={
 '/usr/local/lib/libigc.so.2.36.3+1781103806':{'sha256':'aebe3597ade3ede16ba59d0522085579f1e70fa4f3ffd07a2a0dbbf1d2197269','bytes':95845480},
 '/usr/local/lib/libigdfcl.so.2.36.3+1781103806':{'sha256':'e962be41ce5ee1560064e4f3987f3def2340c1e85d8aa43b83804570fa2bc402','bytes':2448520}}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def stat5(row):return [row.st_dev,row.st_ino,row.st_size,row.st_mtime_ns,row.st_ctime_ns]
def inventory_binding():
 import qualify_owned_indexer_half_control_v2 as q
 from half37_original_elf_spirv_v4 import consume
 from serial37_canonical_json_v3 import read_unique
 require(digest(consume(INVENTORY/'capture-binding.json',1<<20))==INVENTORY_SHA,'Exact observed runtime ocloc inventory required');binding=read_unique(INVENTORY/'capture-binding.json');require(binding['passed']is True and binding['runtime_image']==q.RUNTIME_IMAGE and binding['source_plan_sha256']=='a674e0f070259b717ba65d67f4e22750750a1adcb629265e03b8ef0ad2c3f2ed','Observed inventory image/source differs')
 for name,want in binding['artifact_sha256'].items():require(Path(name).name==name and digest(consume(INVENTORY/name,4<<20))==want,'Original observed inventory bytes differ '+name)
 row=read_unique(INVENTORY/'module-inventory.receipt.json');command=read_unique(INVENTORY/'module-inventory.command.json');q.command_binding(row,INVENTORY/'module-inventory.log',INVENTORY/'module-inventory.command.json',INVENTORY/'module-inventory.receipt.json');obj=read_unique(INVENTORY/'module-inventory.inspection.json');q.terminal_binding(obj['State']);q.observed_container(obj,command[command.index('--name')+1],q.RUNTIME_IMAGE,command)
 log=consume(INVENTORY/'module-inventory.log',128<<10).decode('ascii');require(log.startswith(TOOL+'\n'+TOOL_SHA+'  /usr/bin/ocloc\n') and "Use 'ocloc <command> --help'"in log and all(word in log for word in ('compile','disasm','query','ids','--version')),'Actual resolved ocloc path/SHA/help command support differs');return {'inventory_sha256':INVENTORY_SHA,'tool':TOOL,'tool_sha256':TOOL_SHA,'runtime_image':q.RUNTIME_IMAGE,'original_JIT_or_same_IGC_linkage_proven':False}
def original_library_specs():
 from half37_original_elf_spirv_v4 import consume
 raw=consume(ORIGINAL_ROOT/'report.json',4<<20);require(digest(raw)==REPORT_SHA,'Exact original executed half report changed');report=json.loads(raw);require(report['passed']is True and report['errors']==[],'Original runtime did not pass');libraries=report['runtime_binding_after']['libraries'];require(all(libraries.get(path)==row for path,row in REQUIRED_LIBRARIES.items()),'Actual original mapped IGC library rows differ');return dict(REQUIRED_LIBRARIES)
def probe_argv():
 return {'compile-help':[TOOL,'compile','--help'],'disasm-help':[TOOL,'disasm','--help'],'ids-help':[TOOL,'ids','--help'],'query-help':[TOOL,'query','--help'],'version':[TOOL,'--version']}
def recipes(output,pid):
 import qualify_owned_indexer_half_control_v2 as q
 require(type(pid)is int and pid>0,'Exact positive root producer PID required');inventory=inventory_binding();libraries=original_library_specs();out=Path(output).resolve();env={'ZE_AFFINITY_MASK':'0','ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','SYCL_CACHE_PERSISTENT':'0'};guard='test "$(readlink -f /usr/bin/ocloc)" = '+shlex.quote(TOOL)+'; printf "%s  %s\\n" '+shlex.quote(TOOL_SHA)+' '+shlex.quote(TOOL)+' | sha256sum --check --strict; '
 commands={name:q.docker_recipe('b70-half37-ocloc-'+str(pid)+'-'+name,q.RUNTIME_IMAGE,[(out,'/out','rw')],env,q.SETVARS_PREFIX+guard+'exec '+shlex.join(argv))for name,argv in probe_argv().items()}
 commands['original-library-census']=q.docker_recipe('b70-half37-ocloc-'+str(pid)+'-original-library-census',q.RUNTIME_IMAGE,[(out,'/out','rw'),(Path(__file__).resolve(),'/harness/strata/flash-next/probe.py','ro')],env,q.SETVARS_PREFIX+'exec /opt/b70-c1-python/bin/python /harness/strata/flash-next/probe.py --inside-library-census /out/original-library-census.json')
 return {'commands':commands,'expected_original_library_bytes':libraries,'observed_inventory':inventory,'device_target_selected':None,'offline_compile_or_disassembly_requested':False,'native_helper_or_model_execution_requested':False,'original_runtime_JIT_ISA_observed':False,'offline_tool_shares_actual_original_JIT_library_loading_proven':False,'original_runtime_IGC_current_byte_census_pending':True}
def inside_library_census(output):
 """ROOT container only: bounded current ELF bytes, never dlopen or a kernel."""
 source=Path(__file__);source_raw=source.read_bytes();source_sha=digest(source_raw);rows={}
 for name,expected in REQUIRED_LIBRARIES.items():
  path=Path(name);require(path.is_file() and not path.is_symlink() and path.absolute()==path.resolve(),'Exact original runtime library regular path required');before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'] and before.st_size<=128<<20,'Exact bounded original IGC ELF extent required');h=hashlib.sha256();magic=None
  with path.open('rb')as stream:
   while True:
    part=stream.read(1<<20)
    if not part:break
    if magic is None:magic=part[:4]
    h.update(part)
  after=path.stat();require(stat5(before)==stat5(after) and magic==b'\x7fELF' and h.hexdigest()==expected['sha256'],'Actual original runtime IGC ELF bytes/stat differs');rows[name]={'sha256':h.hexdigest(),'bytes':after.st_size,'stat5':stat5(after),'ELF_magic':magic.hex()}
 tool_raw=None;tool=Path(TOOL);require(tool.is_file() and not tool.is_symlink() and tool.resolve()==tool and tool.stat().st_size<=16<<20,'Exact bounded observed ocloc ELF required');tool_before=tool.stat();tool_raw=tool.read_bytes();require(stat5(tool.stat())==stat5(tool_before) and tool_raw[:4]==b'\x7fELF' and digest(tool_raw)==TOOL_SHA,'Current observed ocloc executable differs');command=['ldd',TOOL];completed=subprocess.run(command,capture_output=True,text=True,timeout=30);require(type(completed.returncode)is int and completed.returncode==0 and not completed.stderr and 'not found'not in completed.stdout,'Current ocloc runtime dependency resolution failed');paths={Path(word).resolve()for line in completed.stdout.splitlines()for word in line.split()if word.startswith('/')};require(paths,'Current observed ocloc dependency roster absent');dependencies={}
 for path in sorted(paths):
  before=path.stat();require(path.is_file() and not path.is_symlink() and before.st_size<=128<<20,'Bounded regular current ocloc dependency required');h=hashlib.sha256()
  with path.open('rb')as stream:
   magic=stream.read(4);require(magic==b'\x7fELF','Actual resolved ocloc ELF dependency required');h.update(magic)
   while True:
    raw=stream.read(1<<20)
    if not raw:break
    h.update(raw)
  after=path.stat();require(stat5(before)==stat5(after),'Actual ocloc dependency changed during census');dependencies[str(path)]={'sha256':h.hexdigest(),'bytes':after.st_size,'stat5':stat5(after)}
 require(source.read_bytes()==source_raw,'Mounted probe source changed during census');value={'schema':1,'passed':True,'current_probe_source_sha256':source_sha,'ocloc_executable':{'path':TOOL,'sha256':TOOL_SHA,'bytes':len(tool_raw)},'ocloc_ldd_command':command,'ocloc_ldd_stdout':completed.stdout,'ocloc_ldd_stdout_sha256':digest(completed.stdout.encode()),'ocloc_resolved_dependency_ELFs':dependencies,'ldd_is_not_dynamic_IGC_dlopen_or_JIT_map_authority':True,'current_original_runtime_library_rows':rows,'original_report_sha256':REPORT_SHA,'dlopen_or_GPU_or_model_execution':False,'offline_tool_library_loading_or_actual_JIT_ISA_proven':False}
 with Path(output).open('x')as f:f.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
 return value
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--inside-library-census',type=Path,required=True);args=parser.parse_args();inside_library_census(args.inside_library_census)
