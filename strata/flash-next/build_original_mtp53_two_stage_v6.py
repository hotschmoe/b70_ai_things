"""Root-owned direct NoDri paired original MTP V6 build; no GPU or model qualification."""
import argparse,collections,hashlib,json,os,shlex,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-publisher-mtp53-two-stage-proposal-v6.json'
PLAN_SHA='eaf1239278657f964ba537a56d778c1ac9cabfc0997684f34afad4a88026f470'
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def regular(path):
 path=Path(path);require(path.is_file()and not path.is_symlink(),'Actual regular input required '+str(path));return path

def archive_members(before,after,members):
 require(all(before.count(name)==1 for name in members),'Exactly one original each HC member required')
 require(after==[name for name in before if name not in members],'Only named HC members may be removed')
 return True

def shell(recipe):
 command=recipe['compile_commands']
 require(len(command)==1 and command[0]['name']=='original-mtp53-two-stage','Single original paired native ELF required')
 rows=['set -Eeo pipefail','test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi','/opt/intel/oneapi/compiler/2026.1/bin/icpx --version > /out/compiler-version.txt','dpkg-query -W > /out/packages.txt','cmp /sdk/build/compiler-version.txt /out/compiler-version.txt','cmp /sdk/build/packages.txt /out/packages.txt',shlex.join(command[0]['argv'])]
 return '\n'.join(rows)+'\n'

def absent(stderr,name):
 if not isinstance(stderr,bytes)or len(stderr)>4096:return False
 try:text=stderr.decode('ascii').strip().lower()
 except UnicodeDecodeError:return False
 return any(text==prefix+'no such '+kind+': '+name.lower()for prefix in('','error: ','error response from daemon: ')for kind in('container','object'))

def main(args):
 require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Original card0 FD8 lease required')
 require(sha(PLAN)==PLAN_SHA,'Frozen paired V6 source proposal changed');plan=read(PLAN)
 for name,want in plan['own_source_sha256'].items():require(sha(ROOT/name)==want,'Frozen paired source changed '+name)
 prepared=args.compile_root;require(prepared.is_absolute()and prepared.resolve()==prepared,'Canonical compile root required')
 recipe_path=regular(prepared/'compile-recipe.json');require(sha(recipe_path)==args.recipe_sha256,'Exact actual prepared compile recipe required');recipe=read(recipe_path)
 require(recipe['two_stage_source_plan_sha256']==PLAN_SHA and recipe['NoDri_required']is True and recipe['No_network_required']is True,'Exact V6 NoDri recipe required')
 sdk=Path(recipe['actual52']['SDK']);require(recipe['mounts']=={'SDK_RO':str(sdk),'overlay_RO':str(prepared/'overlay'),'output_RW':str(prepared/'output')},'Exact original compile mounts required')
 require(sha(sdk/'receipt.json')==recipe['actual52']['receipt_sha256'],'Actual SDK receipt changed');sdk_receipt=read(sdk/'receipt.json')
 require(sdk_receipt['build_rc']==0 and sdk_receipt['devices_exposed']is False and sdk_receipt['image']==recipe['compiler_image'],'Actual original NoDri compiler image required')
 inputs={sdk/'receipt.json':sha(sdk/'receipt.json'),recipe_path:sha(recipe_path),PLAN:PLAN_SHA,Path(__file__).resolve():sha(Path(__file__).resolve())}
 for name,want in sdk_receipt['patched_source_sha256'].items():
  p=regular(sdk/'source'/name);require(sha(p)==want,'Actual SDK source changed '+name);inputs[p]=want
 for name,dest in plan['new_overlay_sources'].items():
  p=regular(prepared/'overlay'/dest);require(sha(p)==plan['own_source_sha256'][name],'Actual compiled overlay changed '+name);inputs[p]=sha(p)
 main=regular(prepared/'overlay/sycl/src/program/original_mtp53_two_stage_diagnostic.cpp');require(sha(main)==recipe['native_MAIN_two_stage_sha256'],'Actual paired MAIN changed');inputs[main]=sha(main)
 policy=regular(prepared/'overlay/sycl/include/strata/kernels/original_mtp53_reference_pp_policy.hpp');require(sha(policy)==recipe['generated_prefix_policy_header_sha256']==plan['generated_policy_header_sha256'],'Actual reference policy header changed');inputs[policy]=sha(policy)
 require(recipe['normal_reference_recipe_sha256']==plan['normal_reference_recipe_sha256']and sha(recipe['normal_reference_recipe_path'])==plan['normal_reference_recipe_sha256']and recipe['normal_prefix_policy']==plan['normal_reference_prefix_policy'],'Exact actual reference metadata/policy required')
 argv=recipe['compile_commands'][0]['argv'];require(argv[0]=='/opt/intel/oneapi/compiler/2026.1/bin/icpx'and argv[-2:]==['-o','/out/original-mtp53-two-stage'],'Actual original compiler/leaf output required')
 for item in argv:
  if item.startswith('/sdk/')and item.endswith('.a'):
   p=regular(sdk/item[len('/sdk/'):]);inputs[p]=sha(p)
 for name in('compiler-version.txt','packages.txt','CMakeFiles/rules.ninja','build.ninja'):
  p=regular(sdk/'build'/name);inputs[p]=sha(p)
 for folder in('sycl/include','include','third_party/ggml','sycl/src/kernels/cuda'):
  for p in sorted((sdk/'source'/folder).rglob('*')):
   if p.is_file():inputs[p]=sha(regular(p))
 out=prepared/'output';require(out.is_dir()and not out.is_symlink()and not any(out.iterdir()),'Original prepared output must be empty; preserve prior attempts')
 script=shell(recipe);(out/'compile.sh').write_text(script,encoding='ascii');write(out/'recipe.snapshot.json',recipe)
 name='b70-mtp-pair-v6-build-'+str(os.getpid());label='b70.mtp.pair.v6.build';owner=str(os.getpid());mounts=[str(sdk)+':/sdk:ro',str(prepared/'overlay')+':/overlay:ro',str(out)+':/out:rw']
 command=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','2','--memory','6g','--memory-swap','6g','--pids-limit','256','--user',str(os.getuid())+':'+str(os.getgid())]
 for mount in mounts:command+=['-v',mount]
 command+=['--entrypoint','/bin/bash',recipe['compiler_image'],'-lc','exec bash /out/compile.sh'];write(out/'command.json',command)
 report={'schema':1,'passed':False,'started_epoch':time.time(),'devices_exposed':False,'model_payload_read':False,'recipe_sha256':sha(recipe_path),'source_plan_sha256':PLAN_SHA,'compiler_image':recipe['compiler_image'],'command':command,'original_inputs_sha256':{str(p):h for p,h in inputs.items()},'fresh_compile_commands':recipe['compile_commands'],'compile_rc':None,'builder_source_sha256':sha(Path(__file__).resolve()),'compile_script_sha256':sha(out/'compile.sh'),'errors':[],'owned_removed':False,'GPU_operator_executed':False};proc=None;terminal=None;signals=[]
 def on_signal(number,frame):signals.append(number)
 saved={n:signal.signal(n,on_signal)for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def inspect():
  result=subprocess.run(['docker','inspect',name],capture_output=True,timeout=30)
  if result.returncode:
   require(absent(result.stderr,name),'Original container inspection ambiguous');return None
  rows=json.loads(result.stdout);require(len(rows)==1,'Exact original container inspection required');row=rows[0];config=row['Config'];host=row['HostConfig']
  require(config['Labels'].get(label)==owner and config['Image']==recipe['compiler_image']and config['Entrypoint']==['/bin/bash']and config['Cmd']==command[-2:]and config['User']==str(os.getuid())+':'+str(os.getgid()),'Original compiler ownership/command changed')
  require(sorted(host['Binds'])==sorted(mounts)and host['NetworkMode']=='none'and not host.get('Devices')and not host.get('DeviceRequests')and host['Privileged']is False,'Exact NoDri bind/device/network contract changed')
  require(all(type(host[k])is int and host[k]==v for k,v in{'NanoCpus':2000000000,'Memory':6<<30,'MemorySwap':6<<30,'PidsLimit':256}.items()),'Typed original compiler bounds changed');require(type(row['RestartCount'])is int and row['RestartCount']==0,'Original compiler restarted');return row
 try:
  with(out/'compile.stdout').open('wb')as so,(out/'compile.stderr').open('wb')as se:
   proc=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True,pass_fds=(8,));write(out/'client.json',{'pid':proc.pid,'started_epoch':time.time()});deadline=time.monotonic()+900
   while proc.poll()is None:
    if signals or time.monotonic()>deadline or sum(p.stat().st_size for p in out.iterdir()if p.is_file())>512<<20:
     raise RuntimeError('Compile interrupted/deadline/output bound')
    time.sleep(.25)
   require(proc.returncode==0,'NoDri compiler failed; raw stdout/stderr retained')
 except BaseException as error:report['errors'].append(type(error).__name__+': '+str(error))
 finally:
  # Unknown states retain the parent/FD8 lease until original owned terminal+absence+client EOF.
  while True:
   try:
    row=inspect()
    if row and row['State']['Running']:
     report['errors'].append('Owned compiler forced cleanup');subprocess.run(['docker','stop','--time','20',row['Id']],check=True,capture_output=True,timeout=40);continue
    if proc and proc.poll()is None:time.sleep(.25);continue
    if row:
     terminal=row;write(out/'terminal.json',row);subprocess.run(['docker','rm',row['Id']],check=True,capture_output=True,timeout=30);continue
    report['owned_removed']=True
    if proc:proc.wait();report['return_code']=proc.returncode;write(out/'client-terminal.json',{'pid':proc.pid,'return_code':proc.returncode,'original_EOF':True,'terminal_epoch':time.time()})
    break
   except BaseException as error:
    message='Retirement '+type(error).__name__+': '+str(error)
    if message not in report['errors']and len(report['errors'])<64:report['errors'].append(message)
    try:write(out/'retirement-unresolved.json',{'passed':False,'ownership_retained':True,'name':name,'owner':owner,'error':message,'epoch':time.time()})
    except BaseException:pass
    time.sleep(1)
  if signals:report['errors'].append('Signals latched '+repr(signals))
  try:
   require(proc is not None and proc.returncode==0 and terminal is not None,'Original normal compiler terminal required');state=terminal['State'];require(type(state['ExitCode'])is int and state['ExitCode']==0 and state['OOMKilled']is False and state['Error']==''and state['Paused']is False and state['Restarting']is False and state['Dead']is False and type(state['Pid'])is int and state['Pid']==0,'Original compiler abnormal terminal')
   report['outputs']={'original-mtp53-two-stage':sha(regular(out/'original-mtp53-two-stage'))};report['compile_rc']=proc.returncode
   report['compiler_version_sha256']=sha(out/'compiler-version.txt');report['packages_sha256']=sha(out/'packages.txt')
   require(report['compiler_version_sha256']==inputs[sdk/'build/compiler-version.txt']and report['packages_sha256']==inputs[sdk/'build/packages.txt'],'Actual image compiler/packages changed')
  except BaseException as error:report['errors'].append('Build closure '+type(error).__name__+': '+str(error))
  for p,want in inputs.items():
   try:require(sha(regular(p))==want,'Original input changed '+str(p))
   except BaseException as error:report['errors'].append('Input closure '+str(error))
  if sum(p.stat().st_size for p in out.iterdir()if p.is_file())>512<<20:report['errors'].append('Final observed compile output budget exceeded')
  for number,handler in saved.items():signal.signal(number,handler)
  report['finished_epoch']=time.time();report['passed']=not report['errors'];report['raw_sha256']={p.name:sha(p)for p in out.iterdir()if p.is_file()and p.name not in('receipt.json',)};write(out/'receipt.json',report)
 print(json.dumps({'passed':report['passed'],'errors':report['errors'],'receipt':str(out/'receipt.json')},ensure_ascii=True));return 0 if report['passed']else 1
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--compile-root',type=Path,required=True);parser.add_argument('--recipe-sha256',required=True);raise SystemExit(main(parser.parse_args()))
