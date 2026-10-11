#!/usr/bin/env python3
"""Pure source validator/recipe emitter; never builds or queries a device."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/mnt/vm_8tb/b70/build/pti-unitrace-source-6c0d6b0b-v1')
COMMIT='6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
MANIFEST=ROOT/'strata/flash-next/pti-unitrace-source-manifest-v1.json'
MANIFEST_SHA='7593e5381a7373cebbe861b0ffb464afaa5374db0c122972d638c39849b82a11'
ICXX='/opt/intel/oneapi/compiler/2026.1/bin/icpx'
ICXX_SHA='98054d858efe1c6fd8b7e8c6ae5df014c7aa7c6333c1881998ebf621ef3b51dc'
LEVEL_ZERO='d3b3efbdaa27a0aef9a9812cc8fa4260557c1e7c'
COMPUTE_RUNTIME='050536ff3b6f830199ec99cff8ffcac3d804c40a'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_binding():
 require(sha(MANIFEST)==MANIFEST_SHA,'Source manifest changed');m=json.loads(MANIFEST.read_bytes());require(m['commit']==COMMIT and Path(m['source_root'])==SOURCE,'Exact fresh source root required')
 require(subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==COMMIT,'PTI source revision differs')
 tracked=set(subprocess.check_output(['git','-C',str(SOURCE),'ls-files','-z']).decode().split('\0'))-{''};require(tracked==set(m['tracked_files']),'Whole tracked source roster differs')
 for name,row in m['tracked_files'].items():
  p=SOURCE/name
  if 'symlink_target'in row:require(p.is_symlink()and p.readlink().as_posix()==row['symlink_target'],'Source symlink changed')
  else:require(p.is_file()and not p.is_symlink()and p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'Source bytes changed '+name)
 return MANIFEST_SHA
def recipe(output):
 out=Path(output);require(out.is_absolute()and not out.exists(),'New absolute build output required')
 name='b70-pti-unitrace-build-'+str(os.getpid())
 shell='''set -e
if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi
test "$(sha256sum /opt/intel/oneapi/compiler/2026.1/bin/icpx | cut -d' ' -f1)" = 98054d858efe1c6fd8b7e8c6ae5df014c7aa7c6333c1881998ebf621ef3b51dc
/opt/intel/oneapi/compiler/2026.1/bin/icpx --version > /out/compiler-version.txt
cmake --version > /out/cmake-version.txt
cmake -S /src/tools/unitrace -B /out/build -GNinja -DCMAKE_BUILD_TYPE=Release -DONEAPI_COMPILER_HOME=/opt/intel/oneapi/compiler/2026.1 -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icx -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.1/bin/icpx -DBUILD_WITH_L0=1 -DBUILD_WITH_OPENCL=0 -DBUILD_WITH_MPI=0 -DBUILD_WITH_OMP=0 -DBUILD_WITH_ITT=0 -DBUILD_WITH_XPTI=0 -DBUILD_WITH_PERFETTO=OFF -DCMAKE_INSTALL_PREFIX=/out/install
cmake --build /out/build --parallel 2 --target unitrace unitrace_tool
test "$(git -C /out/build/level-zero rev-parse HEAD)" = d3b3efbdaa27a0aef9a9812cc8fa4260557c1e7c
test "$(git -C /out/build/compute-runtime rev-parse HEAD)" = 050536ff3b6f830199ec99cff8ffcac3d804c40a
cmake --install /out/build
'''
 return ['docker','run','--name',name,'--label','b70.pti.source='+MANIFEST_SHA,'--network','bridge','--user',str(os.getuid())+':'+str(os.getgid()),'--cpus','2','--memory','4g','--memory-swap','4g','--pids-limit','512','--entrypoint','/bin/bash','-v',str(SOURCE)+':/src:ro','-v',str(out)+':/out:rw',IMAGE,'-c',shell]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();binding=source_binding();print(json.dumps({'source_manifest_sha256':binding,'command':recipe(a.output),'actual_build_executed':False,'GPU_devices_exposed':False,'root_must_create_new_output_and_capture_command_receipt_inspection_removal':True},indent=2,ensure_ascii=True))
if __name__=='__main__':main()
