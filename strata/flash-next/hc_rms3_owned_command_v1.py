"""Scoped RMS3 Docker owner/EOF retirement, derived from closed HC T2 lifecycle."""
import json,os,signal,subprocess,time
from pathlib import Path
from build_hc_t2_operator_v1 import absent,require,sha,write

def inspect_owned(name,command,expected):
 r=subprocess.run(['docker','inspect',name],capture_output=True,timeout=20)
 if r.returncode:require(absent(r.stderr,name),'Unknown original container state; lease retained');return None
 rows=json.loads(r.stdout);require(type(rows)is list and len(rows)==1,'Exact owned inspection required');row=rows[0];c=row['Config'];h=row['HostConfig']
 require(c['Labels'].get(expected['label'])==expected['owner']and c['Image']==expected['image']and row['Image']==expected['image']and c['Entrypoint']==['/bin/bash']and c['Cmd']==command[-2:]and c['User']==expected['user']and c['WorkingDir']=='/','Exact original image/command/owner required')
 require(type(row['RestartCount'])is int and row['RestartCount']==0,'Original process restarted');require(h['NetworkMode']=='none'and h['Privileged']is False and h['PidMode']==''and h.get('DeviceRequests')in(None,[])and h['Init']is False,'Original isolated process settings differ')
 require(type(h['Binds'])is list and len(h['Binds'])==len(expected['mounts'])and sorted(h['Binds'])==sorted(expected['mounts']),'Exact original bind bijection required')
 require(all(type(h[k])is int and h[k]==v for k,v in{'NanoCpus':2000000000,'Memory':4<<30,'MemorySwap':4<<30,'PidsLimit':256}.items()),'Typed original bounded resources required')
 devices=h.get('Devices')or[];require(len(devices)==(1 if expected['GPU']else 0),'Unexpected original devices')
 if expected['GPU']:require(devices==[{'PathOnHost':'/dev/dri','PathInContainer':'/dev/dri','CgroupPermissions':'rwm'}],'Exact fullDRI exposure required')
 require(sorted(h.get('GroupAdd')or[])==sorted(expected['groups']),'Exact original supplemental groups required')
 for k,v in expected.get('env',{}).items():require(c['Env'].count(k+'='+v)==1 and len([e for e in c['Env']if e.startswith(k+'=')])==1,'Exact original runtime flag changed '+k)
 return row

def run_owned(command,out,expected,timeout=300,raw_bound=8<<20):
 # Caller holds FD8 throughout unknown-state retries, including NoDri compilation.
 name=expected['name'];proc=None;attempted=False;terminal=None;errors=[];signals=[];removed=False
 saved={n:signal.signal(n,lambda number,frame:signals.append(number))for n in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
 def error(msg):
  if msg not in errors and len(errors)<64:errors.append(msg)
 write(out/'command.json',command);started=time.time()
 try:
  with(out/'stdout').open('wb')as so,(out/'stderr').open('wb')as se:
   attempted=True;proc=subprocess.Popen(command,stdout=so,stderr=se,start_new_session=True,pass_fds=(8,));write(out/'client.json',{'pid':proc.pid,'session':os.getpgid(proc.pid),'started_epoch':time.time()});deadline=time.monotonic()+timeout
   while proc.poll()is None:
    if signals or time.monotonic()>deadline or sum((out/n).stat().st_size for n in('stdout','stderr'))>raw_bound:raise RuntimeError('Original signal/deadline/raw output failure')
    time.sleep(.25)
   require(proc.returncode==0,'Original command returned '+str(proc.returncode))
 except BaseException as exc:error(type(exc).__name__+': '+str(exc))
 finally:
  if attempted:
   while True:
    try:
     row=inspect_owned(name,command,expected)
     if row and row['State']['Running']:
      error('Forced original cleanup');subprocess.run(['docker','stop','--time','20',row['Id']],check=True,capture_output=True,timeout=40);continue
     if proc and proc.poll()is None:time.sleep(.25);continue
     if row:
      terminal=row;write(out/'terminal.json',row);subprocess.run(['docker','rm',row['Id']],check=True,capture_output=True,timeout=20);continue
     removed=True
     if proc:proc.wait()
     break
    except BaseException as exc:
     error('Retirement '+type(exc).__name__+': '+str(exc))
     try:write(out/'retirement-pending.json',{'lease_retained':True,'owner':expected['owner'],'name':name,'error':str(exc)[:4096],'epoch':time.time()})
     except BaseException:pass
     time.sleep(1)
  if signals:error('Signals latched '+repr(signals))
  try:
   require(proc is not None and proc.returncode==0 and terminal is not None and removed,'Normal original process+EOF+absence required');s=terminal['State'];require(type(s['ExitCode'])is int and s['ExitCode']==0 and s['Running']is False and s['Paused']is False and s['Restarting']is False and s['Dead']is False and s['OOMKilled']is False and s['Error']==''and type(s['Pid'])is int and s['Pid']==0,'Typed original normal terminal required')
   require(sum((out/n).stat().st_size for n in('stdout','stderr'))<=raw_bound,'Final raw output bound exceeded')
  except BaseException as exc:error('Terminal '+str(exc))
  for n,h in saved.items():signal.signal(n,h)
 receipt={'schema':1,'passed':not errors,'errors':errors,'started_epoch':started,'finished_epoch':time.time(),'command_sha256':sha(out/'command.json'),'return_code':proc.returncode if proc else None,'original_client_PID':proc.pid if proc else None,'original_EOF':bool(proc and proc.poll()is not None),'owned_removed':removed,'original_terminal_ID':terminal['Id']if terminal else None,'stdout_sha256':sha(out/'stdout')if(out/'stdout').exists()else None,'stderr_sha256':sha(out/'stderr')if(out/'stderr').exists()else None};write(out/'owned-receipt.json',receipt);return receipt

def command(image,name,label,mounts,GPU=False):
 owner=str(os.getpid());user=str(os.getuid())+':'+str(os.getgid());groups=[];argv=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','2','--memory','4g','--memory-swap','4g','--pids-limit','256','--init=false','--workdir','/','--user',user]
 if GPU:
  argv+=['--device','/dev/dri'];groups=[str(g)for g in sorted({Path('/dev/dri/card0').stat().st_gid,Path('/dev/dri/renderD128').stat().st_gid})]
  for g in groups:argv+=['--group-add',g]
 for mount in mounts:argv+=['-v',mount]
 return argv,{'name':name,'label':label,'owner':owner,'user':user,'image':image,'mounts':mounts,'groups':groups,'GPU':GPU}
