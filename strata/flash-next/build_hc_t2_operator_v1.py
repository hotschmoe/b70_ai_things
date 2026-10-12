"""Root-owned NoDri HC T2 object/leaf build; no GPU or model qualification."""
import argparse,collections,hashlib,json,os,shlex,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'hc-t2-operator-compile-source-plan-v1.json'
PLAN_SHA='dc3c5ab0c9c3879d4f6bed73e1fc6052fab7a91c7ed7e838135297649c241de3'
RECIPE_SHA='744722155e94d8e8c83cd176601d08dd63087f70e01a828a34544e800bcbe4df'
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
 filt=recipe['archive_filter'];members=filt['remove_members']
 require(members==['hc_native_projection.cpp.o','hc_native_composition.cpp.o'],'Exact original HC members required')
 check="import pathlib,sys; rows=pathlib.Path('/out/original-members.txt').read_text().splitlines(); names="+repr(members)+"; sys.exit(0 if all(rows.count(x)==1 for x in names) else 'exact single HC members missing')"
 filtered="import pathlib,sys; a=pathlib.Path('/out/original-members.txt').read_text().splitlines(); b=pathlib.Path('/out/filtered-members.txt').read_text().splitlines(); names="+repr(members)+"; sys.exit(0 if b==[x for x in a if x not in names] else 'unexpected archive change')"
 rows=['set -Eeo pipefail','test ! -e /dev/dri','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh > /out/setvars.log 2>&1; fi','/opt/intel/oneapi/compiler/2026.1/bin/icpx --version > /out/compiler-version.txt','dpkg-query -W > /out/packages.txt','cmp /sdk/build/compiler-version.txt /out/compiler-version.txt','cmp /sdk/build/packages.txt /out/packages.txt','ar t '+shlex.quote(filt['original'])+' > /out/original-members.txt',shlex.join(['python3','-c',check]),shlex.join(filt['copy_command']),shlex.join(filt['remove_command']),'ar t '+shlex.quote(filt['copy'])+' > /out/filtered-members.txt',shlex.join(['python3','-c',filtered])]
 rows.extend(shlex.join(command['argv'])for command in recipe['commands']);rows.append(shlex.join(recipe['link_argv']))
 return '\n'.join(rows)+'\n'

def absent(stderr,name):
 if not isinstance(stderr,bytes)or len(stderr)>4096:return False
 try:text=stderr.decode('ascii').strip().lower()
 except UnicodeDecodeError:return False
 return any(text==prefix+'no such '+kind+': '+name.lower()for prefix in('','error: ','error response from daemon: ')for kind in('container','object'))

def main(args):
 require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Original card0 FD8 lease required')
 require(sha(PLAN)==PLAN_SHA,'Frozen HC T2 compile plan changed');plan=read(PLAN)
 for name,want in plan['files'].items():require(sha(ROOT/name)==want,'Frozen compile source changed '+name)
 prepared=args.prepared;require(prepared.is_absolute()and prepared.resolve()==prepared,'Canonical prepared root required')
 recipe_path=regular(prepared/'compile-recipe.json');require(sha(recipe_path)==RECIPE_SHA,'Exact actual prepared compile recipe required');recipe=read(recipe_path);require(recipe['source_plan_sha256']==PLAN_SHA and recipe['bounds']=={'NoDri':True,'network_none':True,'cpus':2,'memory_gib':4},'Exact frozen NoDri recipe required')
 require(recipe['candidate_proposal_sha256']==plan['candidate_proposal_sha256'],'Candidate source proposal changed')
 sdk=Path(recipe['actual52']['SDK']);require(recipe['mounts']=={'SDK_RO':str(sdk),'candidate_RO':str(prepared/'source'),'fixture_RO':str(prepared/'fixture.cpp'),'output_RW':str(prepared/'output')},'Exact original prepared mounts required')
 require(sha(sdk/'receipt.json')==recipe['actual52']['receipt_sha256'],'Actual original SDK receipt changed');sdk_receipt=read(sdk/'receipt.json')
 require(sdk_receipt['build_rc']==0 and sdk_receipt['devices_exposed']is False and sdk_receipt['image']==recipe['compiler_image'],'Actual original NoDri compiler image required')
 require(sha(sdk/'build/build.ninja')==recipe['production_ninja_sha256'],'Actual production Ninja changed')
 for name,want in recipe['source_object_sha256'].items():require(sha(regular(prepared/'source'/name))==want,'Actual candidate input changed '+name)
 require(sha(regular(prepared/'fixture.cpp'))==recipe['fixture_sha256'],'Actual fixture changed')
 inputs={sdk/'receipt.json':sha(sdk/'receipt.json'),recipe_path:sha(recipe_path),PLAN:PLAN_SHA,Path(__file__).resolve():sha(Path(__file__).resolve())}
 for name,want in sdk_receipt['patched_source_sha256'].items():
  p=regular(sdk/'source'/name);require(sha(p)==want,'Actual SDK source changed '+name);inputs[p]=want
 for name,want in recipe['original_archive_sha256'].items():
  p=regular(sdk/'build'/name);require(sha(p)==want,'Original archive changed '+name);inputs[p]=want
 for name in('compiler-version.txt','packages.txt','CMakeFiles/rules.ninja'):
  p=regular(sdk/'build'/name);inputs[p]=sha(p)
 for folder in('sycl/include','include','third_party/ggml','sycl/src/kernels/cuda'):
  for p in sorted((sdk/'source'/folder).rglob('*')):
   if p.is_file():inputs[p]=sha(regular(p))
 for name,want in recipe['source_object_sha256'].items():inputs[prepared/'source'/name]=want
 inputs[prepared/'fixture.cpp']=recipe['fixture_sha256']
 out=prepared/'output';require(out.is_dir()and not out.is_symlink()and not any(out.iterdir()),'Original prepared output must be empty; preserve prior attempts')
 script=shell(recipe);(out/'compile.sh').write_text(script,encoding='ascii');write(out/'recipe.snapshot.json',recipe)
 name='b70-hc-t2-build-'+str(os.getpid());label='b70.hc.t2.build';owner=str(os.getpid());mounts=[str(sdk)+':/sdk:ro',str(prepared/'source')+':/candidate:ro',str(prepared/'fixture.cpp')+':/fixture.cpp:ro',str(out)+':/out:rw']
 command=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','2','--memory','4g','--memory-swap','4g','--pids-limit','256','--user',str(os.getuid())+':'+str(os.getgid())]
 for mount in mounts:command+=['-v',mount]
 command+=['--entrypoint','/bin/bash',recipe['compiler_image'],'-lc','exec bash /out/compile.sh'];write(out/'command.json',command)
 report={'schema':1,'passed':False,'started_epoch':time.time(),'devices_exposed':False,'model_payload_read':False,'recipe_sha256':sha(recipe_path),'source_plan_sha256':PLAN_SHA,'compiler_image':recipe['compiler_image'],'command':command,'original_inputs_sha256':{str(p):h for p,h in inputs.items()},'fresh_compile_commands':recipe['commands'],'fresh_link_argv':recipe['link_argv'],'builder_source_sha256':sha(Path(__file__).resolve()),'compile_script_sha256':sha(out/'compile.sh'),'errors':[],'owned_removed':False,'GPU_operator_executed':False};proc=None;terminal=None;signals=[]
 def on_signal(number,frame):signals.append(number)
 saved={n:signal.signal(n,on_signal)for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def inspect():
  result=subprocess.run(['docker','inspect',name],capture_output=True,timeout=30)
  if result.returncode:
   require(absent(result.stderr,name),'Original container inspection ambiguous');return None
  rows=json.loads(result.stdout);require(len(rows)==1,'Exact original container inspection required');row=rows[0];config=row['Config'];host=row['HostConfig']
  require(config['Labels'].get(label)==owner and config['Image']==recipe['compiler_image']and config['Entrypoint']==['/bin/bash']and config['Cmd']==command[-2:]and config['User']==str(os.getuid())+':'+str(os.getgid()),'Original compiler ownership/command changed')
  require(sorted(host['Binds'])==sorted(mounts)and host['NetworkMode']=='none'and not host.get('Devices')and not host.get('DeviceRequests')and host['Privileged']is False,'Exact NoDri bind/device/network contract changed')
  require(all(type(host[k])is int and host[k]==v for k,v in{'NanoCpus':2000000000,'Memory':4<<30,'MemorySwap':4<<30,'PidsLimit':256}.items()),'Typed original compiler bounds changed');require(type(row['RestartCount'])is int and row['RestartCount']==0,'Original compiler restarted');return row
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
   archive_members((out/'original-members.txt').read_text().splitlines(),(out/'filtered-members.txt').read_text().splitlines(),recipe['archive_filter']['remove_members'])
   outputs=[v['name']for v in recipe['commands']]+['libstrata_kernels.without_hc.a','hc-t2-operator'];report['outputs']={name:sha(regular(out/name))for name in outputs}
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
 parser=argparse.ArgumentParser();parser.add_argument('--prepared',type=Path,required=True);raise SystemExit(main(parser.parse_args()))
