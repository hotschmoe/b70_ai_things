#!/usr/bin/env python3
"""Root CPU-only same-recipe retry with passive metadata under one exclusion lease."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
import observe_cpu_swap_attribution_v3 as observer
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
OBS_SHA='0b4fa9e0b5ae483a3705064442e368a8dba5a79c7958b60ff28b90c6725db77b'
def require(ok,message):
 if not ok:raise ValueError(message)
def command(output):
 return [sys.executable,str(HERE/'qualify_api_positive_overlap_cpu_screen_v1.py'),'--build-root','/mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010','--source-receipt','/mnt/vm_8tb/b70/build/flashnext-cpu-source-v2-20261010/source-receipt.json','--fixture',str(BASE/'api-positive-overlap-authentic-corpus-v1'),'--output',str(output),'--leased']
def retire(proc,label,report):
 if proc is None:return
 if proc.poll() is None:
  try:proc.send_signal(signal.SIGTERM)
  except ProcessLookupError:pass
 try:rc=proc.wait(timeout=30)
 except subprocess.TimeoutExpired:
  report['errors'].append(label+' normal join exceeded30s; exclusion retained until actual terminal')
  rc=proc.wait()
 report.setdefault('owned_retirement',{})[label]={'pid':proc.pid,'return_code':rc,'terminal_epoch':time.time(),'terminal_confirmed':proc.poll() is not None}
 require(proc.poll() is not None,'Owned process not terminal '+label)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-source-plan-sha256',required=True);a=p.parse_args()
 source_plan=HERE/'cpu-overlap-observed-retry-source-plan-v2.json';source_raw=source_plan.read_bytes();require(observer.sha(source_plan)==a.expected_source_plan_sha256,'Exact reviewed retry source plan required');closure=observer.read_unique(source_plan)['files']
 for name,digest in closure.items():require(observer.sha(ROOT/name)==digest,'Current retry source closure differs '+name)
 require(not a.output.exists(),'New retry orchestration root required')
 for card in (0,1):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Inherited pair CPU exclusion lease required')
 require(observer.sha(HERE/'cpu-swap-attribution-source-plan-v3.json')==OBS_SHA,'Reviewed observer source plan changed')
 for name,digest in observer.read_unique(HERE/'cpu-swap-attribution-source-plan-v3.json')['files'].items():require(observer.sha(ROOT/name)==digest,'Current observer prerequisite changed '+name)
 observer.source_binding();a.output=a.output.resolve();a.output.mkdir();report={'schema':1,'started_epoch':time.time(),'passed':False,'errors':[],'actual_GPU_touch':False,'memory_guards_changed':False,'inference_settings_changed':False}
 idle=a.output/'idle';screen=a.output/'screen';watch=a.output/'observe';fd=(8,9)
 idle_cmd=[sys.executable,str(HERE/'observe_cpu_swap_attribution_v3.py'),'idle','--output',str(idle),'--seconds','30']
 report['idle_command']=idle_cmd
 cpu=None;monitor=None;idle_proc=None;stopping=[]
 def stop(sig,frame):
  stopping.append(sig)
  for proc in (idle_proc,cpu):
   if proc and proc.poll() is None:
    try:proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:pass
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 try:
  with (a.output/'idle.log').open('w') as log:
   idle_proc=subprocess.Popen(idle_cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=fd);report['idle_return_code']=idle_proc.wait()
  require(report['idle_return_code']==0 and not stopping,'Fresh passive idle observation failed/interrupted');report['idle_binding']=observer.finalized_binding(idle)
  cpu_cmd=command(screen);report['screen_command']=cpu_cmd
  with (a.output/'screen.log').open('w') as clog,(a.output/'observe.log').open('w') as olog:
   cpu=subprocess.Popen(cpu_cmd,stdout=clog,stderr=subprocess.STDOUT,pass_fds=fd);report['producer_pid']=cpu.pid
   obs_cmd=[sys.executable,str(HERE/'observe_cpu_swap_attribution_v3.py'),'observe','--output',str(watch),'--screen-root',str(screen),'--producer-pid',str(cpu.pid),'--idle-receipt',str(idle/'report.json'),'--seconds','7200'];report['observer_command']=obs_cmd
   monitor=subprocess.Popen(obs_cmd,stdout=olog,stderr=subprocess.STDOUT,pass_fds=fd)
   while cpu.poll() is None:
    require(monitor.poll() is None,'Passive observer terminated while model screen still running');time.sleep(1)
   report['screen_return_code']=cpu.returncode;report['screen_terminal_epoch']=time.time()
   if monitor.poll() is None:monitor.send_signal(signal.SIGTERM)
   report['observer_return_code']=monitor.wait(timeout=30)
  report['observer_binding']=observer.finalized_binding(watch)
  require(report['screen_return_code']==0 and report['observer_return_code']==0 and not stopping,'CPU screen or observer failed/interrupted')
  report['screen_binding']=observer.screen.finalized_binding(screen)
  require(source_plan.read_bytes()==source_raw,'Retry source plan changed during operation')
  for name,digest in closure.items():require(observer.sha(ROOT/name)==digest,'Final retry source closure differs '+name)
  report['passed']=True
 except BaseException as exc:
  report['errors'].append(type(exc).__name__+': '+str(exc))
 finally:
  # Screen owns all Docker/model cleanup. Keep exclusion lease until it exits.
  for proc,label in ((cpu,'screen'),(monitor,'observer'),(idle_proc,'idle')):retire(proc,label,report)
  if report['errors']:report['passed']=False
  report.update(finished_epoch=time.time(),stop_signals=stopping,causal_swap_attribution_qualified=False,model_math_qualified=False,API_overlap_qualified=False)
  observer.write_new(a.output/'report.json',report)
 print(json.dumps({'passed':report['passed'],'report':str(a.output/'report.json')}));return 0 if report['passed']else 1
if __name__=='__main__':raise SystemExit(main())
