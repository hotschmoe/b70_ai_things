"""Root-owned NoDri RMS3 compile; existing bounded HC owner discipline."""
import argparse,json,os,shlex,sys,time
from pathlib import Path
from hc_rms3_owned_command_v1 import command,run_owned,require,sha,write
import prepare_hc_rms3_operator_v2 as source
ROOT=source.ROOT;HERE=source.HERE;IMAGE='sha256:39992d7072aa0557f4e3a5faf7782f83bab8fe4882e9d8fcf3e660d3de50e9e7';PLAN=HERE/'hc-rms3-explicit-sqrt-divide-lifecycle-source-plan-v2.json'
def binding():
 before=sha(PLAN)
 for n,h in json.loads(PLAN.read_text())['files'].items():require(sha(ROOT/n)==h,'RMS3 lifecycle source changed '+n)
 require(sha(PLAN)==before,'RMS3 lifecycle plan changed');return before

def admission(prepared,expected_recipe):
 s=binding();require(prepared.is_absolute()and prepared.resolve()==prepared and prepared.is_dir(),'Canonical actual prepared root required');rp=prepared/'compile-recipe.json';require(rp.is_file()and not rp.is_symlink()and sha(rp)==expected_recipe,'Exact actual source-prepared recipe required');r=json.loads(rp.read_text());require(r['source_plan_sha256']==sha(source.HERE/'hc-rms3-explicit-sqrt-divide-source-plan-v2.json')and r['compiled_or_GPU_executed']is False and r['original_public_layout_changed']is False and r['observer_default_OFF']is True,'Exact isolated source-only recipe required');pins={rp:expected_recipe,PLAN:s}
 require(set(r['files'])=={'fixture.cpp','hc_rms3_generated_v2.inc','hc_native_composition.hpp'},'Complete actual source-prepared roster required')
 for n,h in r['files'].items():
  p=prepared/n;require(p.is_file()and not p.is_symlink()and sha(p)==h,'Prepared source changed '+n);pins[p]=h
 require((prepared/'fixture.cpp').read_bytes()==(ROOT/'kernels/hc_rms3_operator_fixture_v2.cpp').read_bytes(),'Frozen actual fixture mismatch');require((prepared/'hc_rms3_generated_v2.inc').read_text()==source.extract((source.SDK61/'source/sycl/src/kernels/hc_native_composition.cpp').read_text()),'Actual HcNorm exact extraction mismatch')
 sdk=source.SDK61;receipt=json.loads((sdk/'receipt.json').read_text());require(receipt['image']==IMAGE and receipt['devices_exposed']is False and receipt['build_rc']==0,'Actual original compiler image receipt required')
 for p in(sdk/'receipt.json',sdk/'build/compiler-version.txt',sdk/'build/packages.txt'):require(p.is_file()and not p.is_symlink(),'Actual compiler/package regular bytes required');pins[p]=sha(p)
 # The recipe's exact original production flag list is owned by source_prepare;
 # reconstruct without creating output directories or invoking any compiler.
 expected=['/opt/intel/oneapi/compiler/2026.1/bin/icpx','-O3','-DNDEBUG','-std=c++20','-fsycl','-fsycl-default-sub-group-size=32','-fsycl-device-code-split=per_kernel','-fp-model=precise','-I/inputs','/inputs/fixture.cpp','/usr/lib/x86_64-linux-gnu/libze_loader.so','-Xsycl-target-backend=spir64','-cl-fp32-correctly-rounded-divide-sqrt','-o','/out/hc-rms3'];require(r['compiler_argv']==expected,'Exact production-matched isolated compiler flags required');return r,pins

def compile_shell(recipe):
 return '\n'.join(['set -Eeo pipefail','test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi','/opt/intel/oneapi/compiler/2026.1/bin/icpx --version > /out/compiler-version.txt','dpkg-query -W > /out/packages.txt','cmp /sdk/build/compiler-version.txt /out/compiler-version.txt','cmp /sdk/build/packages.txt /out/packages.txt',shlex.join(recipe['compiler_argv']),'ldd /out/hc-rms3 > /out/ldd.txt','! grep -q "not found" /out/ldd.txt'])+'\n'
def main(a):
 recipe,pins=admission(a.prepared,a.recipe_sha256)
 require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Inherited card0 FD8 required even for root NoDri compile');out=a.output;require(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical compile output required');out.mkdir();write(out/'recipe.snapshot.json',recipe);(out/'compile.sh').write_text(compile_shell(recipe),encoding='ascii');mounts=[str(a.prepared)+':/inputs:ro',str(source.SDK61)+':/sdk:ro',str(out)+':/out:rw'];argv,expected=command(IMAGE,'b70-hc-rms3-build-'+str(os.getpid()),'b70.hc.rms3.build',mounts);argv+=['--entrypoint','/bin/bash',IMAGE,'-lc','exec bash /out/compile.sh'];report={'schema':1,'passed':False,'errors':[],'started_epoch':time.time(),'image':IMAGE,'devices_exposed':False,'model_payload_read':False,'recipe_sha256':a.recipe_sha256,'source_plan_sha256':sha(PLAN),'compile_inputs':{str(p):h for p,h in pins.items()},'compiled_or_GPU_executed':True,'GPU_operator_executed':False}
 try:
  owned=run_owned(argv,out,expected,timeout=300);report['owned']=owned;report['errors'].extend(owned['errors']);require(owned['passed'],'Actual owned NoDri compiler failed');binary=out/'hc-rms3';require(binary.is_file()and not binary.is_symlink(),'Fresh actual regular ELF required');require(sha(out/'compiler-version.txt')==pins[source.SDK61/'build/compiler-version.txt']and sha(out/'packages.txt')==pins[source.SDK61/'build/packages.txt'],'Actual image compiler/package changed');report['outputs']={'hc-rms3':sha(binary)}
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  for p,h in pins.items():
   try:require(sha(p)==h,'Closing original input changed '+str(p))
   except BaseException as exc:report['errors'].append('input-close: '+str(exc))
  try:require(binding()==report['source_plan_sha256'],'Closing lifecycle source changed')
  except BaseException as exc:report['errors'].append('source-close: '+str(exc))
  report.update(passed=not report['errors'],finished_epoch=time.time(),raw_sha256={p.name:sha(p)for p in out.iterdir()if p.is_file()and p.name!='receipt.json'});write(out/'receipt.json',report)
 print(json.dumps({'passed':report['passed'],'errors':report['errors'],'receipt':str(out/'receipt.json')},ensure_ascii=True));return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--recipe-sha256',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
