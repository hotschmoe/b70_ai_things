"""Explicit fresh runtime intake; unfilled build is refused, never old ELF reuse."""
import hashlib,os
from pathlib import Path
from serial37_canonical_json_v3 import read_unique,canonical
from source41_pti_raw_budget_v1 import require
HERE=Path(__file__).resolve().parent;SOURCE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-clock-budget-source-v1');HEADER='tools/unitrace/src/pti_unitrace_clock_export_v1.h';HEADER_SHA='1042a101816113325090f8ed4f94b973affce958dd5d1b64a89f5f2a262081c7'
OPTIONS=['--chrome-call-logging','--chrome-kernel-logging','--chrome-device-logging','--chrome-event-buffer-size','16384','--output-dir-path','/results/unitrace']
CLOCK_ENV={'UNITRACE_CLOCK_METADATA_PATH':'/results/unitrace/clock.{pid}.jsonl','UNITRACE_HOST_CLOCK_BRIDGE':'1'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def regular(path,expected,limit=512<<20):
    path=Path(path);require(path.is_file()and not path.is_symlink()and path.resolve()==path and path.stat().st_size<=limit,'Fresh regular runtime input required');before=path.stat();digest=sha(path);after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)and digest==expected,'Fresh runtime bytes changed');return {'path':str(path),'sha256':digest,'bytes':after.st_size}

def inherited_env(env):
    require(not any(k.startswith('UNITRACE_')for k in env)and 'LD_PRELOAD'not in env,'Inherited profiler/preload overrides forbidden')
    require(not any(k.startswith('UNITRACE_')for k in os.environ)and 'LD_PRELOAD'not in os.environ,'Parent inherited profiler/preload overrides forbidden')

def binding(path):
    path=Path(path).resolve();value=read_unique(path);require(value['runtime_ready']is True,'Fresh clock-budget runtime intake remains unfilled; launch refused')
    require(type(value['schema'])is int and value['schema']==1 and value['source_root']==str(SOURCE)and value['expected_header_sha256']==HEADER_SHA and type(value['metadata_cap_bytes'])is int and value['metadata_cap_bytes']==512<<20,'Exact fresh512MiB clock-budget source required');regular(SOURCE/HEADER,HEADER_SHA,1<<20)
    receipt=regular(value['build_receipt_path'],value['build_receipt_sha256'],8<<20);build=read_unique(receipt['path']);require(build['passed']is True and type(build['build_rc'])is int and build['build_rc']==0 and build['error']is None and build['GPU_devices_exposed']is False and build['normal_owned_removed']is True and build['clock_source_root']==str(SOURCE)and build['clock_source_plan_sha256']==sha(HERE/'pti-unitrace-clock-budget-source-plan-v1.json')and build['all_patched_source_bytes_checked_before_after']is True,'Actual fresh build/source/owned receipt required')
    install=Path(value['install_root']);tool=regular(install/'bin/unitrace',value['unitrace_sha256']);injection=regular(install/'lib/libunitrace_tool.so',value['injection_sha256'])
    for file in(tool,injection):
        with Path(file['path']).open('rb')as stream:require(stream.read(4)==b'\x7fELF','Fresh profiler ELF required')
    dep=regular(value['dependency_receipt_path'],value['dependency_receipt_sha256'],16<<20);dependencies=read_unique(dep['path']);require(dependencies['passed']is True and dependencies['image']==build['image']and dependencies['unitrace_sha256']==tool['sha256']and dependencies['injection_sha256']==injection['sha256'],'Actual same-image profiler dependency census required')
    require(dependencies['files'],'Complete actual DSO byte census required')
    for row in dependencies['files']:
        require(type(row['original_container_path'])is str and row['original_container_path'].startswith('/')and type(row['exported_path'])is str,'Original container DSO path plus actual exported host artifact required');regular(row['exported_path'],row['sha256'])
    device=regular(value['actual_device_clock_metadata_path'],value['actual_device_clock_metadata_sha256'],8<<20);properties=read_unique(device['path']);require(properties['passed']is True and properties['ZE_AFFINITY_MASK']=='0'and canonical(properties['clock_properties'])==canonical(value['actual_device_clock_properties']),'Actual masked GPU0 clock metadata required')
    return {'intake_path':str(path),'intake_sha256':sha(path),'build':receipt,'tool':tool,'injection':injection,'dependencies':dep,'device_metadata':device,'clock_properties':value['actual_device_clock_properties'],'install_root':str(install),'image':build['image'],'fresh512MiB_clock_budget':True}
