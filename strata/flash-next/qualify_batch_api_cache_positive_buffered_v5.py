"""NEW journal patch to pinned original V7 lifecycle; all original file bytes kept."""
from pathlib import Path
import batch_api_cache_positive_buffered_v5 as ctrl
import qualify_batch_numerical_v7 as original
TEMPLATE=Path(original.__file__).resolve()
OLD=''' def faults(label):
  row=command(['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],label,30)
  text=Path(row['path']).read_text(errors='replace');require(row['return_code']==0 and not FAULT.search(text),'Kernel journal unavailable or GPU fault signature')
  return row
'''
NEW=''' def faults(label):
  row=command(['journalctl','-k','--since','@'+str(int(parent['started_epoch'])),'--no-pager'],label,30)
  row['observed_epoch']=time.time();parent.setdefault('kernel_journal_rows',{})[label]=row;parent['journal_admission_generation']=5;save()
  text=Path(row['path']).read_text(errors='replace');require(row['return_code']==0 and row['error'] is None and not FAULT.search(text),'Kernel journal unavailable/error or GPU fault signature')
  return row
'''
OLD_FORWARD="""   def forward():
    for line in child.stdout:
     log.write(line);log.flush();print(line.rstrip('\\n').encode('ascii','backslashreplace').decode('ascii'),flush=True)
"""
NEW_FORWARD="""   parent['child_stdout_reader']={'completed':False,'error':None,'eof_epoch':None}
   def forward():
    try:
     for line in child.stdout:
      log.write(line);log.flush();print(line.rstrip('\\n').encode('ascii','backslashreplace').decode('ascii'),flush=True)
     parent['child_stdout_reader'].update(completed=True,eof_epoch=time.time())
    except BaseException as exc:parent['child_stdout_reader'].update(error=type(exc).__name__+': '+str(exc),eof_epoch=time.time())
"""
JOIN="reader.join(timeout=10);child_terminal_epoch=time.time();parent['child_terminal_epoch']=child_terminal_epoch"
NEW_JOIN="reader.join(timeout=10);require(not reader.is_alive() and parent['child_stdout_reader']['completed'] is True and parent['child_stdout_reader']['error'] is None,'Actual child stdout EOF/drain failed');parent['child_stdout_log_sha256']=sha(out/'child-supervisor.log');child_terminal_epoch=time.time();parent['child_terminal_epoch']=child_terminal_epoch"
def adapted_source():
 source=TEMPLATE.read_text();ctrl.require(source.count(OLD)==1 and source.count(OLD_FORWARD)==1 and source.count(JOIN)==2,'Exact journal/EOF integration failed');source=source.replace(OLD,NEW).replace(OLD_FORWARD,NEW_FORWARD).replace(JOIN,NEW_JOIN);import_line='import batch_numerical_execution_v7 as ctrl';ctrl.require(source.count(import_line)==1,'Exact controller integration failed');source=source.replace(import_line,'import batch_api_cache_positive_buffered_v5 as ctrl')
 binding="parent['post_model_identity']={'path':str(out/'post-model-identity.json'),'sha256':sha(out/'post-model-identity.json')}"
 replacement="parent['post_model_identity']={'path':str(out/'post-model-identity.json'),'sha256':sha(out/'post-model-identity.json'),'started':identity['started'],'finished':identity['finished']}"
 ctrl.require(source.count(binding)==1,'Exact copied identity epoch integration failed');source=source.replace(binding,replacement)
 command_start=" def command(cmd,label,timeout=210):\n"
 command_result="return {'path':str(out/(label+'.log')),'sha256':sha(out/(label+'.log'))"
 launch="   child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,pass_fds=(8,9))"
 ctrl.require(source.count(command_start)==1 and source.count(command_result)==1 and source.count(launch)==1,'Exact actual command/launch timing integration failed')
 source=source.replace(command_start,command_start+"  command_started_epoch=time.time()\n").replace(command_result,"return {'started_epoch':command_started_epoch,'finished_epoch':time.time(),'path':str(out/(label+'.log')),'sha256':sha(out/(label+'.log'))")
 source=source.replace(launch,"   parent['child_launch_started_epoch']=time.time();save()\n"+launch)
 return source[:source.index("if __name__=='__main__':")]
def main():
 ctrl.require(ctrl.sha(TEMPLATE)==ctrl.read(ctrl.SOURCE_PLAN)['files'][str(TEMPLATE.relative_to(ctrl.ROOT))],'Pinned original lifecycle changed');source=adapted_source();namespace=dict(vars(original));namespace.update(__file__=__file__,__name__='owned_buffered_parent_v5',ctrl=ctrl,BATCH_NUMERICAL_SHA=ctrl.sha(Path(ctrl.__file__)));exec(compile(source,str(TEMPLATE)+'[exact-journal-EOF-V4]', 'exec'),namespace);return namespace['main']()
if __name__=='__main__':raise SystemExit(main())
