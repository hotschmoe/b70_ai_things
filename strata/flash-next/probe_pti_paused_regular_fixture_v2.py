#!/usr/bin/env python3
"""Root-only leased fixture controller, using the original strict retirement utilities."""
import argparse,ast,hashlib,json,os,signal,time
from pathlib import Path
import probe_unitrace_indexer40_v5 as prior
import pti_paused_fixture_runtime_v1 as runtime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BUILD=Path('/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-c388-build-v1')
INSTALL=BUILD/'install'
PROPERTY_SHA='ca646c4245e20fb44d6bcbc5a4a214f578d8f1b44f2b1e7ae2c68f4477aac289'
BUILD_SHA='f52f9a974ff8a8c3f3327bd6b9ddab48025fda1c14ada8628225528d81551789'
IMAGE=prior.IMAGE
require=prior.require

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def source_binding():
 plan=HERE/'pti-paused-fixture-runtime-source-plan-v2.json';value=json.loads(plan.read_bytes())
 for name,want in value['files'].items():require(sha(ROOT/name)==want,'Current source changed '+name)
 require(sha(BUILD/'receipt.json')==BUILD_SHA,'Actual fresh tool build receipt changed');row=json.loads((BUILD/'receipt.json').read_bytes());require(row['passed']is True and row['errors']==[]and row['normal_owned_removed']is True,'Fresh build not qualified')
 proposal=json.loads((HERE/'pti-paused-regular-list-tool-build-proposal-v1.json').read_bytes());tool_source=Path('/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-source-v1')
 for name,want in proposal['expected_full_source_sha256'].items():require(sha(tool_source/name)==want,'Current fresh tool source changed '+name)
 require(sha(INSTALL/'bin/unitrace')==runtime.TOOL_SHA and sha(INSTALL/'lib/libunitrace_tool.so')==runtime.LIB_SHA,'Actual new installed ELF changed');return sha(plan)

def artifact_roster(root):
 result={};total=0
 for path in sorted(Path(root).rglob('*')):
  require(not path.is_symlink(),'Original artifact aliases refused')
  if path.is_file()and path!=Path(root)/'report.json':
   raw,binding=runtime.bound(path,32<<20);total+=len(raw);require(total<=64<<20 and len(result)<1000,'Original fixture artifact bound exceeded');result[str(path.relative_to(root))]=binding
 return result

def utility(recipe,compile_only=False):
 namespace={**vars(prior),'recipe':recipe,'IMAGE':IMAGE,'INSTALL':INSTALL}
 # Original full creator/state/cleanup/health predicates in a private scope.
 for name in('owned','arm_run'):
  node=next(n for n in ast.parse(Path(prior.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name==name)
  for n in ast.walk(node):
   if isinstance(n,ast.Constant)and type(n.value)is str:n.value=n.value.replace('b70.unitrace.indexer40','b70.pti.paused.fixture').replace('b70-unitrace-idx40-','b70-pti-paused-fixture-')
  exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[strict-owned-utilities]','exec'),namespace)
 if compile_only:
  original_owned=namespace['owned']
  node=next(n for n in ast.parse(Path(prior.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='owned')
  # Exact original config/resources; only the no-device domain differs.
  text=ast.unparse(node).replace('b70.unitrace.indexer40','b70.pti.paused.fixture')
  text=text.replace("h['GroupAdd'] == [command[command.index('--group-add') + 1]]","not h.get('GroupAdd') and not h.get('DeviceCgroupRules') and not h.get('CapAdd')")
  text=text.replace("h['Devices'] == [{'PathOnHost': '/dev/dri', 'PathInContainer': '/dev/dri', 'CgroupPermissions': 'rwm'}]","not h['Devices']")
  require(text!=ast.unparse(node)and "command.index('--group-add')"not in text and 'PathOnHost'not in text,'Exact compile-only no-device derivation required')
  exec(compile(text,__file__+'[no-device-owned-recipe]','exec'),namespace)
 return namespace

def recipe_for(fixture,fixture_sha,properties):
 def recipe(arm,out,name,source):
  gid=str(Path('/dev/dri/renderD128').stat().st_gid)
  command=['docker','create','--name',name,'--label','b70.pti.paused.fixture='+source,'--network','none','--device','/dev/dri','--group-add',gid,'--user',str(os.getuid())+':'+str(os.getgid()),'--memory','4g','--memory-swap','4g','--cpus','2','--pids-limit','512','--entrypoint','/bin/bash']
  for p,target,mode in((fixture,'/leaf/fixture','ro'),(properties,'/properties.json','ro'),(ROOT,'/harness','ro'),(out,'/results','rw'),(INSTALL,'/pti','ro')):command+=['-v',str(p)+':'+target+':'+mode]
  for k,v in sorted({'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu','ZE_AFFINITY_MASK':'0','SYCL_CACHE_PERSISTENT':'0','SYCL_UR_TRACE':'0'}.items()):command+=['-e',k+'='+v]
  shell='set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; export PYTHONPATH=/harness/strata/flash-next; exec /opt/b70-c1-python/bin/python /harness/strata/flash-next/pti_paused_fixture_runtime_v1.py --retention '+arm+' --fixture-sha256 '+fixture_sha+' --properties /properties.json'
  return command+[IMAGE,'-c',shell]
 return recipe

def compile_recipe(properties):
 def recipe(arm,out,name,source):
  command=['docker','create','--name',name,'--label','b70.pti.paused.fixture='+source,'--network','none','--user',str(os.getuid())+':'+str(os.getgid()),'--memory','4g','--memory-swap','4g','--cpus','2','--pids-limit','512','--entrypoint','/bin/bash']
  for path,target,mode in((HERE/'pti_paused_regular_list_fixture_v2.cpp','/leaf/fixture.cpp','ro'),(properties,'/properties.json','ro'),(ROOT,'/harness','ro'),(out,'/results','rw'),(INSTALL,'/pti','ro')):command+=['-v',str(path)+':'+target+':'+mode]
  shell='set -e; if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; /opt/intel/oneapi/compiler/2026.1/bin/icpx --version; exec /opt/intel/oneapi/compiler/2026.1/bin/icpx -fsycl -O2 -std=c++17 -fsycl-targets=spir64 /leaf/fixture.cpp -o /results/fixture'
  return command+[IMAGE,'-c',shell]
 return recipe

def health(out,stage):
 row=prior.run_command([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--card','0','--img',prior.HEALTH],out,stage+'-strict',180)
 health_text([row]);return [row]

def health_text(rows):
 require(type(rows)is list and len(rows)==1,'Exactly one card0 health receipt required')
 import re
 raw=Path(rows[0]['stderr']['path']).read_text();require('card 0: OK'in raw and 'xpu-health: HEALTHY (cards 0)'in raw and not re.search(r'HEALTH_FAIL|WEDGED|INCONCLUSIVE',raw),'Actual card0 health failed')

def main():
 p=argparse.ArgumentParser();p.add_argument('--leased',action='store_true');p.add_argument('--properties',type=Path,required=True);p.add_argument('--properties-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 def interrupted(signum,frame):
  for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
  raise InterruptedError('Owned fixture interrupted '+str(signum))
 for sig in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
 require(a.leased,'Invoke bin/gpu-run --card 0 and --leased');os.fstat(8)
 require(a.properties_sha256==PROPERTY_SHA,'Exact original API-property receipt bytes required')
 for f,want in((a.properties,a.properties_sha256),):require(f.is_absolute()and f.resolve()==f and f.is_file()and sha(f)==want,'Actual compiled fixture/properties bytes changed')
 require(a.output.is_absolute()and a.output.parent.resolve()==a.output.parent and not a.output.exists(),'Fresh canonical output required');source=source_binding();a.output.mkdir();started=time.time();report={'passed':False,'errors':[],'started_epoch':started,'source_plan_sha256':source,'properties_path':str(a.properties),'properties_sha256':a.properties_sha256,'build_receipt_sha256':BUILD_SHA,'producer_pid':os.getpid()}
 try:
  report['compile_terminal']=utility(compile_recipe(a.properties),compile_only=True)['arm_run']('compile',a.output/'build',source)
  a.fixture=a.output/'build/fixture';a.fixture_sha256=sha(a.fixture);report.update(fixture_path=str(a.fixture),fixture_sha256=a.fixture_sha256,compile_source_sha256=sha(HERE/'pti_paused_regular_list_fixture_v2.cpp'))
  report['pre_health']=health(a.output,'pre');report['pre_kernel']=prior.kernel(a.output,'pre',started)
  u=utility(recipe_for(a.fixture,a.fixture_sha256,a.properties));report['arms']={}
  for arm in('off','on'):
   report['arms'][arm]=u['arm_run'](arm,a.output/arm,source)
   row=json.loads((a.output/arm/'inside-receipt.json').read_bytes());require(row['passed']is True and row['retention_enabled']is(arm=='on'),'Actual loaded coverage failed')
  require(source_binding()==source and sha(a.fixture)==a.fixture_sha256 and sha(a.properties)==a.properties_sha256,'Closing source/fixture changed');report['passed']=True
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  try:report['post_health']=health(a.output,'post');report['post_kernel']=prior.kernel(a.output,'post',started)
  except BaseException as exc:report['passed']=False;report['errors'].append('Closing '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['full_model_coverage_qualified']=False;report['clock_alignment_qualified']=False;report['artifacts']=artifact_roster(a.output);write(a.output/'report.json',report)
 require(report['passed'],'Loaded fixture failed; original raw retained')
if __name__=='__main__':main()
