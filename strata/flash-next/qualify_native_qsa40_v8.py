#!/usr/bin/env python3
"""Owned source40 OFF/ON model parent; fresh semanticREADY->healthACK."""
import argparse,os,signal,subprocess,sys,threading,time,traceback
from pathlib import Path
import native_qsa40_runtime_v8 as d
from native_rms_phase_supervisor_v3 import supervise
from serial37_stdout_capture_v2 import forward,completed
from batch54_health_handshake_v1 import pid_identity,write_new
from source_page_watchdog_v3 import preserve
from run_source_upload_oracle_full_v2 import full_buffered_identity as full4
from native_rms_kernel_journal_v5 import reject_faults
ROOT=d.ROOT;HERE=d.HERE;HEALTH='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067';require=d.require;read=d.read;write=d.write;sha=d.sha

def health(out,label):
 rows=[]
 for name,argv,timeout in [('strict-health',[str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'),'--img',HEALTH],210),('compiled-health',[str(ROOT/'bin/xpu-collective-health'),'--img',HEALTH,'--p2p','0','--timeout','180'],240)]:rows.append(command(out,argv,label+'-'+name,timeout))
 return {'passed':True,'finished_epoch':time.time(),'rows':rows,'image':HEALTH}
def command(out,argv,label,timeout):
 write(out/(label+'.command.json'),argv);row=supervise(argv,out/(label+'.log'),timeout);row['command_sha256']=sha(out/(label+'.command.json'));write(out/(label+'.receipt.json'),row);require(row['passed'],'Actual command failed '+label);return row

def journal(out,label,started):
 row=command(out,['journalctl','-k','--since','@'+str(int(started)),'--no-pager'],label+'-kernel',30);reject_faults((out/(label+'-kernel.log')).read_text());return row

def cleanup(directory,plan,pid,on):
 name='b70-qsa40-'+str(pid)+'-'+('on'if on else'off')
 if d.c.absent(name):return False
 result=read(directory/'command.json');expected,_,_=d.command(plan,directory,pid,on,os.stat('/dev/dri/renderD128').st_gid);require(result==expected,'Failure cleanup exact owned command differs');obj=d.c.inspected(name);d.inspection(obj,expected,plan);write(directory/'parent-cleanup-inspection.json',obj)
 owned_id=obj['Id']
 if obj['State']['Running']:subprocess.run(['docker','stop','--time','15',name],check=True,capture_output=True,timeout=30)
 if d.c.absent(name):return True
 obj=d.c.inspected(name);require(obj['Id']==owned_id,'Owned name was reused during cleanup; refusal');d.inspection(obj,expected,plan);require(not obj['State']['Running'],'Failure cleanup unresolved owned engine');subprocess.run(['docker','rm',name],check=True,capture_output=True);require(d.c.absent(name),'Owned failure removal missing');return True

def run_child(out,plan_path,plan,on,started,timeout):
 label='on'if on else'off';directory=out/label;directory.mkdir();argv=[sys.executable,str(HERE/'native_qsa40_runtime_v8.py'),'child','--plan',str(plan_path),'--output',str(directory)]+(['--on']if on else[]);write(out/(label+'-child.command.json'),argv);status={'eof':False,'error':None};begin=time.time();forced=False;error=None
 with(out/(label+'-child.log')).open('w')as log:
  try:proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,pass_fds=(8,9))
  except BaseException as exc:
   write(out/(label+'-child-launch-failure.json'),{'command':argv,'started_epoch':begin,'finished_epoch':time.time(),'error':type(exc).__name__+': '+str(exc),'actual_child_started':False});raise
  reader=threading.Thread(target=forward,args=(proc.stdout,log,lambda _:None,status),daemon=True);reader.start();deadline=time.monotonic()+timeout
  try:
   while not(directory/'READY.json').exists():require(proc.poll()is None and time.monotonic()<deadline,'Actual child semantic READY failed/deadline');time.sleep(.1)
   ready=read(directory/'READY.json');require(ready['pid']==pid_identity(proc.pid)and ready['parent']==pid_identity(os.getpid())and ready['plan_sha256']==sha(plan_path)and ready['source_plan_sha256']==d.closure(),'Actual owned child semantic READY differs');h=health(out,label+'-pre');j=journal(out,label+'-pre',started);ack={'ready_sha256':sha(directory/'READY.json'),'plan_sha256':sha(plan_path),'health_finished_epoch':h['finished_epoch'],'health_sha256':None,'journal_sha256':j['sha256'],'ack_epoch':time.time()};write(out/(label+'-pre-health.json'),h);ack['health_sha256']=sha(out/(label+'-pre-health.json'));write_new(directory/'ACK.json',ack)
   rc=proc.wait(timeout=max(.001,deadline-time.monotonic()))
  except BaseException as exc:
   error=type(exc).__name__+': '+str(exc);forced=True
   if proc.poll()is None:proc.terminate()
   try:cleanup(directory,plan,proc.pid,on)
   except BaseException as failure:error+='; cleanup refused: '+type(failure).__name__+': '+str(failure)
   try:rc=proc.wait(timeout=30)
   except subprocess.TimeoutExpired:
    # Never SIGKILL the child while an original launch can still create a GPU
    # container. Its controlled handler owns attach retirement and cleanup.
    while proc.poll()is None:
     print('QSA40 failure retains leases pending controlled child/launch retirement',flush=True)
     try:rc=proc.wait(timeout=30)
     except subprocess.TimeoutExpired:pass
    rc=proc.wait()
  reader.join(10)
  if reader.is_alive():
   forced=True;error=(error+'; 'if error else'')+'child stdout drain deadline'
   try:cleanup(directory,plan,proc.pid,on)
   except BaseException as failure:error+='; cleanup refused: '+str(failure)
   reader.join(10)
  if reader.is_alive():
   # A retained inherited writer is a failure. Keep the leased parent alive
   # with an explicit failure sidecar until its actual owned stream retires.
   write(out/(label+'-drain-failure.json'),{'error':error,'child_pid':proc.pid,'leases_retained':True})
   while reader.is_alive():
    print('QSA40 failure retains leases pending owned stdout EOF',flush=True);reader.join(30)
  proc.stdout.close()
 cap=completed(reader,status,out/(label+'-child.log'));row={'child_pid':proc.pid,'child_started_epoch':begin,'child_terminal_epoch':time.time(),'command':argv,'command_sha256':sha(out/(label+'-child.command.json')),'return_code':rc,'forced_cleanup':forced,'error':error,'stdout_capture':cap,'ready':ready if 'ready'in locals()else None,'pre_health':h if'h'in locals()else None,'pre_journal':j if'j'in locals()else None};write(out/(label+'-child.receipt.json'),row);require(rc==0 and not forced and error is None and cap['passed'],'Actual owned child/EOF failed');return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--one-card-receipt',type=Path);ap.add_argument('--max-runtime',type=int,default=10800);ap.add_argument('--leased',action='store_true');a=ap.parse_args();require(1<=a.max_runtime<=21600,'Bounded child deadline required')
 if not a.leased:os.execv(str(ROOT/'bin/gpu-run'),['gpu-run',sys.executable,__file__,*sys.argv[1:],'--leased'])
 for card in(0,1):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Exact inherited pair leases before semantic admission required')
 plan=read(a.plan);d.manifest(plan)
 if plan['cards']==[0,1]:
  require(a.one_card_receipt is not None,'Same source40 observer onecard prerequisite required');from validate_native_qsa40_v8 import finalized_binding;one=finalized_binding(a.one_card_receipt);require(one['cards']==[0]and one['engine_receipt_sha256']==plan['candidate_binding']['prepared']['engine_receipt_sha256'],'Actual sameengine observer onecard proof required')

 for card in(0,1):require(os.path.samefile('/proc/self/fd/'+str(8+card),'/mnt/vm_8tb/b70/gpu.lock.'+str(card)),'Exact inherited pair leases required')
 signal.signal(signal.SIGTERM,lambda *_:(_ for _ in()).throw(InterruptedError('Owned parent terminated')));out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.snapshot.json',plan);write(out/'source-plan.snapshot.json',read(d.PLAN));(out/'parent.snapshot.py').write_bytes(Path(__file__).read_bytes());started=time.time();write(out/'parent-start.json',{'started_epoch':started});lock=read(HERE/'model-lock.json');shards=[ROOT/lock['destination']/x['path']for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')];report={'schema':1,'producer_pid':os.getpid(),'producer_identity':pid_identity(os.getpid()),'producer_interpreter':sys.executable,'producer_interpreter_sha256':sha(sys.executable),'started_epoch':started,'plan_sha256':sha(out/'plan.snapshot.json'),'source_plan_sha256':d.closure(),'errors':[],'passed':False,'cards':plan['cards'],'phases':{},'full_model_math_qualified':False,'captured_operands_used':False,'forced_cleanup':False}
 try:
  report['pre_pages']=preserve(shards[2],out,'pre');report['pre_full4']=full4(HERE/'model-lock.json',lock,shards,out/'pre-full4.json',time.time());report['pre_full4_after_pages']=preserve(shards[2],out,'pre-after-full4');require(report['pre_full4']['passed']and report['pre_pages']['passed']and report['pre_full4_after_pages']['passed'],'Actual pre-source4/pages failed')
  for on in(False,True):report['phases']['on'if on else'off']=run_child(out,out/'plan.snapshot.json',plan,on,started,a.max_runtime)
  from native_qsa40_reader_v8 import arm,comparisons
  off=arm(out/'off',plan,report['phases']['off'],False,report['plan_sha256']);on=arm(out/'on',plan,report['phases']['on'],True,report['plan_sha256']);report['comparison']=comparisons(off,on)
  from native_qsa40_localization_v8 import join
  from native_qsa40_input33_v8 import verify
  identity=plan['current_identity'];report['original_input33']={'off':verify(off['source33'],identity),'on':verify(on['source33'],identity)}
  report['localization']=join(plan['own_producer_root'],on['targets'],plan['own_producer_binding']['model_identity_association']);write(out/'numerical-comparison.json',report['comparison']);write(out/'localization.json',report['localization']);write(out/'original-input33.json',report['original_input33']);report['model_computation_terminal_epoch']=time.time()
 except BaseException as exc:report['errors'].append(type(exc).__name__+': '+str(exc));report['failure_traceback']=traceback.format_exc()
 finally:
  for label in ('off','on'):
   receipt=out/(label+'-child.receipt.json')
   if receipt.exists():
    from native_qsa40_owned_launch_v8 import fence
    while not fence(out/label):
     print('QSA40 failure retains leases pending original attach/session fence',flush=True);time.sleep(30)
    pid=read(receipt)['child_pid'];name='b70-qsa40-'+str(pid)+'-'+label
    while not d.c.absent(name):
     try:cleanup(out/label,plan,pid,label=='on');report['errors'].append('Failure-side owned cleanup required '+name)
     except BaseException as exc:
      report['errors'].append('Cleanup refusal retains leases: '+str(exc));write(out/'failed-owner-pending.json',{'errors':report['errors'],'name':name,'leases_retained':True});print('QSA40 failure retains leases for unresolved exact owner '+name,flush=True);time.sleep(30)
  report['GPU_terminal_epoch']=time.time()
  try:
   report['post_health']=health(out,'post');report['post_journal']=journal(out,'post',started);report['before_post_full4_pages']=preserve(shards[2],out,'post-before-full4');report['post_full4']=full4(HERE/'model-lock.json',lock,shards,out/'post-full4.json',max(report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch']));report['post_pages']=preserve(shards[2],out,'post-after-full4');d.manifest(plan)
  except BaseException as exc:report['errors'].append('post-proof: '+type(exc).__name__+': '+str(exc))
  report['finished_epoch']=time.time();report['passed']=not report['errors']and'localization'in report;report['artifact_sha256']={str(p.relative_to(out)):sha(p)for p in sorted(out.rglob('*'))if p.is_file()and p.name!='report.json'};write(out/'report.json',report)
 if report['passed']:
  from validate_native_qsa40_v8 import finalized_binding
  finalized_binding(out)
 return int(not report['passed'])
if __name__=='__main__':raise SystemExit(main())
