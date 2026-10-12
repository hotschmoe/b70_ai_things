"""Root-run one-card full HC T2 fixture; no model mounts or model qualification."""
import argparse,hashlib,json,os,re,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUNTIME='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
PLAN=Path(__file__).resolve().parent/'hc-t2-operator-runtime-source-plan-v1.json'
HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
MARKER='HC_T2_OPERATOR_PASS pending=2 replays=4 fields=7 borrowed_tail_exact=1 model_qualified=0 speed_qualified=0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,message):
 if not v:raise ValueError(message)
def write(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
def absent(stderr,name):
 text=stderr.decode('ascii',errors='replace').strip().lower()
 return any(text==prefix+'no such '+kind+': '+name.lower()for prefix in('','error: ','error response from daemon: ')for kind in('object','container'))
def main(a):
 require(os.path.samefile('/proc/self/fd/8','/mnt/vm_8tb/b70/gpu.lock.0'),'Card0 lease FD8 required')
 plan=json.loads(PLAN.read_bytes())
 for path,want in plan['files'].items():require(sha(ROOT/path)==want,'Runtime source changed '+path)
 receipt=a.build/'receipt.json';binary=a.build/'hc-t2-operator';recipe=a.build.parent/'compile-recipe.json'
 require(sha(receipt)==a.build_receipt_sha256 and sha(binary)==a.elf_sha256,'Actual completed build receipt/ELF required')
 build=json.loads(receipt.read_bytes());require(build.get('passed')is True and build.get('devices_exposed')is False,'Successful actual NoDri compile required')
 require(build['source_plan_sha256']==sha(ROOT/'strata/flash-next/hc-t2-operator-compile-source-plan-v1.json')and build['recipe_sha256']==sha(recipe),'Exact fresh HC object recipe required')
 require(a.output.is_absolute()and a.output.resolve()==a.output and not a.output.exists(),'New canonical output required');out=a.output;out.mkdir()
 report={'schema':1,'started_epoch':time.time(),'passed':False,'errors':[],'build_receipt_sha256':a.build_receipt_sha256,'elf_sha256':a.elf_sha256,'card':0,'synthetic_only':True,'model_payload_read':False,'paired_model_qualified':False,'speed_qualified':False}
 status={'signal':None,'retiring':False};proc=None;row_saved=None;removed=False;forced=False;attempted=False
 def error(message):
  if message not in report['errors']and len(report['errors'])<64:report['errors'].append(message)
 def signals(number,frame):status['signal']=number
 for number in(signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(number,signals)
 name='b70-hc-t2-'+str(os.getpid());owner=str(os.getpid());label='b70.hct2.owner'
 def inspect():
  result=subprocess.run(['docker','inspect',name],capture_output=True,timeout=20)
  if result.returncode:
   require(absent(result.stderr,name),'Docker state unknown; lease retained');return None
  row=json.loads(result.stdout)[0]
  require(row['Config']['Labels'].get(label)==owner and row['Config']['Image']==RUNTIME,'Foreign/changed owner; lease retained')
  return row
 def checked(argv,phase,timeout=180):
  write(out/(phase+'.command.json'),argv);started=time.time()
  with(out/(phase+'.stdout')).open('wb')as stdout,(out/(phase+'.stderr')).open('wb')as stderr:result=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=timeout)
  write(out/(phase+'.result.json'),{'started_epoch':started,'finished_epoch':time.time(),'return_code':result.returncode,'stdout_sha256':sha(out/(phase+'.stdout')),'stderr_sha256':sha(out/(phase+'.stderr'))});require(result.returncode==0,phase+' returned '+str(result.returncode))
 try:
  checked(['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'],'pre-kernel')
  checked([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'pre-health')
  require(status['signal']is None,'Signal before launch')
  command=['docker','run','--name',name,'--label',label+'='+owner,'--network','none','--cpus','2','--memory','4g','--memory-swap','4g','--user',str(os.getuid())+':'+str(os.getgid()),'--device','/dev/dri/card0','--device','/dev/dri/renderD128']
  for gid in sorted({Path('/dev/dri/card0').stat().st_gid,Path('/dev/dri/renderD128').stat().st_gid}):command+=['--group-add',str(gid)]
  command+=['-e','ZE_AFFINITY_MASK=0','-e','ONEAPI_DEVICE_SELECTOR=level_zero:gpu','-e','SYCL_CACHE_PERSISTENT=0','-e','SYCL_UR_TRACE=0','-e','STRATA_HC_Q8_TOKEN_TILE=0','-v',str(a.build)+':/harness:ro','--entrypoint','/bin/bash',RUNTIME,'-lc','if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; exec /harness/hc-t2-operator']
  write(out/'native.command.json',command)
  with(out/'native.stdout').open('wb')as stdout,(out/'native.stderr').open('wb')as stderr:
   attempted=True;proc=subprocess.Popen(command,stdout=stdout,stderr=stderr,start_new_session=True);write(out/'client.json',{'pid':proc.pid,'session':os.getpgid(proc.pid),'started_epoch':time.time()});deadline=time.monotonic()+180
   while proc.poll()is None:
    total=(out/'native.stdout').stat().st_size+(out/'native.stderr').stat().st_size
    if status['signal']is not None or time.monotonic()>deadline or total>4<<20:
     error('Signal/deadline/output failure');break
    time.sleep(.25)
   if proc.returncode is not None:require(proc.returncode==0,'Native return '+str(proc.returncode))
 except BaseException as e:error(type(e).__name__+': '+str(e))
 finally:
  if attempted:
   status['retiring']=True;attempts=0
   while True:
    attempts+=1
    try:
     row=inspect()
     require(proc is not None,'Original creator identity unavailable; lease retained')
     if row and row['State']['Running']:
      forced=True;error('Forced owned cleanup');subprocess.run(['docker','stop','--time','20',row['Id']],check=True,capture_output=True,timeout=40);continue
     # An early absent-name result cannot retire a still-running original Docker creator.
     if proc and proc.poll()is None:time.sleep(.25);continue
     if row:
      row_saved=row
      try:write(out/'terminal.json',row)
      except Exception as e:error('Terminal evidence IO '+str(e))
      subprocess.run(['docker','rm',row['Id']],check=True,capture_output=True,timeout=20);continue
     removed=True
     if proc:proc.wait()
     break
    except Exception as e:
     error('Retirement '+str(e))
     try:write(out/'retirement-pending.json',{'attempts':attempts,'error':str(e)[:4096],'lease_retained':True,'original_name':name})
     except Exception:pass
     time.sleep(1)
   status['retiring']=False
  if status['signal']is not None:error('Requested signal '+str(status['signal']))
  report.update({'forced_cleanup':forced,'owned_removed':removed,'original_client_return_code':proc.returncode if proc else None,'original_foreground_EOF':bool(proc and proc.poll()is not None),'retirement_attempts':attempts if attempted else 0})
  try:
   require(row_saved is not None and removed and not forced and proc is not None and proc.returncode==0,'Actual normal original process/container retirement required')
   state=row_saved['State'];require(type(state['ExitCode'])is int and state['ExitCode']==0 and state['Running']is False and state['Paused']is False and state['Restarting']is False and state['Dead']is False and state['OOMKilled']is False and state['Error']==''and type(state['Pid'])is int and state['Pid']==0,'Typed normal Docker terminal required')
   require((out/'native.stdout').stat().st_size+(out/'native.stderr').stat().st_size<=4<<20,'Final output bound exceeded')
   text=(out/'native.stdout').read_text(encoding='ascii');require(text.count(MARKER)==1,'Exact complete HC fixture marker required');report['complete_fields_and_tail_marker']=True
  except BaseException as e:error('Completion '+str(e))
  try:checked([str(ROOT/'bin/xpu-health'),'--card','0','--img',HEALTH],'post-health');report['post_health_passed']=True
  except BaseException as e:error('Post-health '+str(e))
  try:
   checked(['journalctl','-k','--since','@'+str(int(report['started_epoch'])),'--no-pager'],'post-kernel');text=(out/'post-kernel.stdout').read_text();require(not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error',text,re.I),'Fresh kernel fault');report['kernel_window_passed']=True
  except BaseException as e:error('Kernel '+str(e))
  try:require(sha(receipt)==a.build_receipt_sha256 and sha(binary)==a.elf_sha256,'Closing build/ELF changed')
  except BaseException as e:error('Closing build '+str(e))
  for path,want in plan['files'].items():
   try:require(sha(ROOT/path)==want,'Closing runtime source changed '+path)
   except BaseException as e:error('Closing source '+str(e))
  report['source_plan_sha256']=sha(PLAN);report['finished_epoch']=time.time();report['passed']=not report['errors'];write(out/'report.json',report);(out/'controller.snapshot.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(report,ensure_ascii=True));return 0 if report['passed']else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--build-receipt-sha256',required=True);p.add_argument('--elf-sha256',required=True);p.add_argument('--output',type=Path,required=True);raise SystemExit(main(p.parse_args()))
