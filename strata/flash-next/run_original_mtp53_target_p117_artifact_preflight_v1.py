"""Protected root NoDri artifact validator with cacheV2-compatible actual receipt."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
import prepare_original_mtp53_target_p117_artifact_v1 as artifact
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PLAN=HERE/'original-mtp53-target-p117-preflight-source-plan-v1.json'
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
MARKER='MTP53_ARTIFACT_RUNTIME_PREFLIGHT_PASS modes=2 actual_custom_ELF=1 exact_native_config=1 actual_python_source_versions=1 tokenizer_files=5 no_GPU=1 whole_GGUF_read=0'
def need(v,s):
 if not v:raise ValueError(s)
def sha(p):return artifact.sha(p)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def absent(stderr,name):
 if type(stderr)is not bytes or len(stderr)>4096:return False
 try:text=stderr.decode('ascii').strip().lower()
 except UnicodeDecodeError:return False
 return any(text==prefix+'no such '+kind+': '+name.lower()for prefix in('','error: ','error response from daemon: ')for kind in('container','object'))
def validate_terminal(row,command):
 name=command[command.index('--name')+1];key,owner=command[command.index('--label')+1].split('=',1);cfg=row['Config'];host=row['HostConfig']
 need(row['Name']=='/'+name and cfg['Labels'].get(key)==owner and cfg['Image']==IMAGE and cfg['Entrypoint']==['/bin/bash']and cfg['Cmd']==command[-2:]and cfg['User']==command[command.index('--user')+1],'Exact original preflight owner/command required')
 mounts=[command[i+1]for i,x in enumerate(command)if x=='-v'];need(type(host['Binds'])is list and all(type(x)is str for x in host['Binds'])and sorted(host['Binds'])==sorted(mounts)and all(x.endswith(':ro')for x in mounts),'Exact readonly bind multiset required')
 need(host['Privileged']is False and host['NetworkMode']=='none'and not host.get('Devices')and not host.get('DeviceRequests')and not host.get('DeviceCgroupRules')and not host.get('GroupAdd')and not host.get('CapAdd'),'Exact NoDri unprivileged domain required')
 need(all(type(host[k])is int and host[k]==v for k,v in {'NanoCpus':2000000000,'Memory':2<<30,'MemorySwap':2<<30}.items())and type(row['RestartCount'])is int and row['RestartCount']==0,'Typed CPU2/RAM2 runtime bounds required')
 return row

def main(a):
 plan=read(PLAN)
 for p,h in plan['files'].items():need(sha(ROOT/p)==h,'Preflight source changed '+p)
 root=a.artifacts;need(root.is_absolute()and root.resolve()==root,'Canonical prepared artifact directory required')
 bindings=artifact.admit(root,a.artifact_binding_sha256,read(root/'runtime-intake.json'));recipe=read(root/'preflight-command.json');command=recipe['argv']
 need(command[:2]==['docker','run']and '--device'not in command and '--device-cgroup-rule'not in command and '--gpus'not in command and '--privileged'not in command and '--detach'not in command and '-d'not in command,'Declared foreground NoDri invocation required')
 need(recipe['artifact_binding_sha256']==a.artifact_binding_sha256 and recipe['devices_exposed']is False and command[-3]==IMAGE and '--cpus'in command and command[command.index('--cpus')+1]=='2'and command[command.index('--memory')+1]=='2g'and command[command.index('--memory-swap')+1]=='2g','Exact prepared bounded C388 command required')
 out=a.output;need(out.is_absolute()and out.resolve()==out and not out.exists(),'New canonical preflight result root required');out.mkdir()
 name=command[command.index('--name')+1];proc=None;terminal=None;removed=False;forced=False;attempted=False;signals=[]
 receipt={'schema':1,'passed':False,'started_epoch':time.time(),'errors':[],'source_plan_sha256':sha(PLAN),'artifact_binding_sha256':a.artifact_binding_sha256,'runtime_image':IMAGE,'devices_exposed':False,'owned_removed':False,'original_EOF':False,'return_code':None,'native_pack_payloads_hashed_by_root':True,'GGUF_payload_read':False,'GPU_executed':False,'command_path':str(out/'command.json'),'stdout_path':str(out/'stdout.txt'),'stderr_path':str(out/'stderr.txt'),'terminal_path':str(out/'terminal.json')}
 write(out/'command.json',command)
 inputs={root/'binding.json':a.artifact_binding_sha256,root/'preflight-command.json':sha(root/'preflight-command.json'),root/'runtime-intake.json':sha(root/'runtime-intake.json')};inputs.update({Path(p):h for p,h in bindings['host_bindings'].items()})
 for r in bindings['records'].values():inputs[Path(r['manifest'])]=r['manifest_sha256'];inputs[Path(r['runtime'])]=r['runtime_sha256']
 def error(e):
  s=str(e)
  if s not in receipt['errors']and len(receipt['errors'])<64:receipt['errors'].append(s)
 def inspect():
  r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=20)
  if r.returncode:need(absent(r.stderr,name),'Unknown original Docker state; ownership retained');return None
  rows=json.loads(r.stdout);need(type(rows)is list and len(rows)==1,'Exact single original Docker inspection required');return validate_terminal(rows[0],command)
 saved={n:signal.signal(n,lambda number,frame:signals.append(number))for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 try:
  need(inspect()is None,'Preflight owned name must be absent before launch')
  with(out/'stdout.txt').open('wb')as so,(out/'stderr.txt').open('wb')as se:
   attempted=True;proc=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True);write(out/'client.json',{'pid':proc.pid,'session':os.getpgid(proc.pid),'started_epoch':time.time()});deadline=time.monotonic()+180
   while proc.poll()is None:
    if signals or time.monotonic()>deadline or (out/'stdout.txt').stat().st_size+(out/'stderr.txt').stat().st_size>4<<20:raise RuntimeError('Preflight signal/deadline/output failure')
    time.sleep(.1)
   need(proc.returncode==0,'Actual API artifact validator failed; raw output retained')
 except BaseException as e:error(type(e).__name__+': '+str(e))
 finally:
  if attempted:
   while True:
    try:
     row=inspect()
     if row and row['State']['Running']:
      forced=True;error('Forced original preflight cleanup');r=subprocess.run(['docker','stop','--time','10',row['Id']],capture_output=True,timeout=30);need(r.returncode==0,'Original stop unresolved');continue
     if proc is None:
      # An ambiguous attempted Popen has no successful EOF receipt. Keep
      # ownership until root settles it; never release on name absence alone.
      raise RuntimeError('Attempted launch has no returned creator; ownership retained')
     if proc.poll()is None:time.sleep(.1);continue
     if row:
      terminal=row
      try:write(out/'terminal.json',row)
      except BaseException as e:error('Terminal evidence IO '+str(e))
      r=subprocess.run(['docker','rm',row['Id']],capture_output=True,timeout=20);need(r.returncode==0,'Original removal unresolved')
      try:write(out/'removal.json',{'id':row['Id'],'return_code':r.returncode,'stdout':r.stdout.decode('ascii',errors='replace'),'stderr':r.stderr.decode('ascii',errors='replace'),'epoch':time.time()})
      except BaseException as e:error('Removal evidence IO '+str(e))
      continue
     proc.wait();removed=True;break
    except BaseException as e:
     error('Retirement '+type(e).__name__+': '+str(e))
     try:write(out/'retirement-pending.json',{'error':str(e),'name':name,'ownership_retained':True,'epoch':time.time()})
     except BaseException:pass
     time.sleep(1)
  if signals:error('Signals latched '+str(signals))
  try:
   need(proc is not None and proc.returncode==0 and terminal is not None and removed and not forced,'Original normal preflight EOF/terminal/removal required')
   state=terminal['State'];need(type(state['ExitCode'])is int and state['ExitCode']==0 and type(state['Pid'])is int and state['Pid']==0 and all(state[k]is False for k in ['Running','Paused','Restarting','Dead','OOMKilled'])and state['Error']=='','Typed normal NoDri terminal required')
   need((out/'stdout.txt').read_text('ascii').strip()==MARKER and (out/'stderr.txt').read_bytes()==b'','Exact actual validator marker/cleanstderr required')
   need((out/'stdout.txt').stat().st_size+(out/'stderr.txt').stat().st_size<=4<<20,'Closing preflight output bound')
   receipt.update(owned_removed=True,original_EOF=True,return_code=proc.returncode,forced_cleanup=False)
  except BaseException as e:error('Completion '+str(e))
  for p,h in inputs.items():
   try:need(sha(p)==h,'Closing exact artifact dependency changed '+str(p))
   except BaseException as e:error('Binding closure '+str(e))
  for p,h in plan['files'].items():
   try:need(sha(ROOT/p)==h,'Closing preflight source changed')
   except BaseException as e:error('Source closure '+str(e))
  try:artifact.admit(root,a.artifact_binding_sha256,read(root/'runtime-intake.json'))
  except BaseException as e:error('Artifact closure '+str(e))
  for number,handler in saved.items():signal.signal(number,handler)
  receipt['finished_epoch']=time.time();receipt['passed']=not receipt['errors']
  for key in ('command','stdout','stderr','terminal'):
   p=Path(receipt[key+'_path']);receipt[key+'_sha256']=sha(p)if p.is_file()else None
  write(out/'receipt.json',receipt)
 print(json.dumps({'passed':receipt['passed'],'errors':receipt['errors'],'receipt':str(out/'receipt.json')},ensure_ascii=True));return 0 if receipt['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--artifact-binding-sha256',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
