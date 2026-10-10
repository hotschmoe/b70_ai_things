"""Root-only exact owned actor retirement; unknown ownership retains exclusion."""
import signal,time
from pathlib import Path
from full_cache_shared_history_v2 import require

def critical_guard(root):
 """A missing/partial marker never authorizes killing an ACKed actor."""
 import json
 path=Path(root)/'actor-retirement-state.json'
 if not path.exists():return {}
 try:
  require(not path.is_symlink() and path.is_file(),'Original actor marker must be regular')
  value=json.loads(path.read_bytes());require(type(value)is dict and type(value['retirement_in_progress'])is bool,'Typed actual actor critical state required')
  return value
 except BaseException as error:return {'retirement_in_progress':True,'marker_read_unknown':type(error).__name__}


def retire(name,owned,run,absent,pause,write,root,deadline=180):
 root=Path(root);obj=owned();stop_row=None
 if obj['State']['Running'] is True:
  row=run(['docker','kill','--signal','TERM',name],check=False,capture_output=True,timeout=30);stop_row={'command':['docker','kill','--signal','TERM',name],'return_code':row.returncode,'stdout':row.stdout.decode('ascii','backslashreplace') if type(row.stdout)is bytes else row.stdout,'stderr':row.stderr.decode('ascii','backslashreplace') if type(row.stderr)is bytes else row.stderr};write(root/'owned-stop-receipt.json',stop_row);require(type(row.returncode)is int and row.returncode==0,'Actual exact-owned actor stop failed')
 limit=time.monotonic()+deadline
 while True:
  obj=owned();state=obj['State'];write(root/'owned-latest-inspection.json',obj)
  if state['Running'] is False:break
  require(time.monotonic()<limit,'Exact owned actor remains live; exclusion cannot end');pause(1)
 require(type(state['ExitCode'])is int and type(state['OOMKilled'])is bool and type(state['Error'])is str,'Actual typed owned terminal state required');write(root/'owned-terminal-inspection.json',obj)
 command=['docker','rm',name];row=run(command,check=False,capture_output=True,timeout=30);write(root/'owned-removal-receipt.json',{'command':command,'return_code':row.returncode});require(type(row.returncode)is int and row.returncode==0,'Actual owned terminal removal not proved')
 # Once the original object is removed, retry only authoritative absence.
 # Re-running owned()/rm would discard the successful original removal and
 # could target a subsequently reused name. Unknown/False never means absent.
 absence_failures=[]
 while True:
  try:
   require(absent(name) is True,'Original successful removal lacks trustworthy current absence')
   break
  except BaseException as error:
   if len(absence_failures)<64:absence_failures.append(type(error).__name__+': '+str(error))
   write(root/'owned-absence-recovery.json',{'original_terminal':obj,'original_removal':{'command':command,'return_code':row.returncode},'failures':absence_failures,'normal_qualification':False});pause(.1)
 return {'state':state,'removed':True,'owned_actor_actually_terminal':True,'normal_terminal':not absence_failures and state['ExitCode']==0 and state['OOMKilled'] is False and state['Error']=='','absence_recovery_failures':absence_failures,'unknown_ownership_or_deadline_as_absence':False,'normal_qualification_from_cleanup_only':False}


def original_signals(handler):
 previous={}
 for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):previous[number]=signal.signal(number,handler)
 return previous

def restore_signals(previous):
 for number,handler in previous.items():signal.signal(number,handler)

def retain_until_settled(name,owned,run,absent,pause,write,root):
 """ROOT ONLY: no failed inspection/timeout permits releasing inherited leases."""
 root=Path(root);ledger={'schema':3,'retirement_in_progress':True,'owned_terminal_and_removed':False,'failures':[],'failure_count':0,'interruption_signals':[]};previous=original_signals(lambda number,frame:ledger['interruption_signals'].append(number))
 try:
  write(root/'actor-retirement-state.json',ledger)
  while True:
   try:
    result=external_recovery(root,name,absent) if (root/'external-owned-terminal-removal.json').exists() else retire(name,owned,run,absent,pause,write,root);ledger.update(retirement_in_progress=False,owned_terminal_and_removed=True,result=result);write(root/'actor-retirement-state.json',ledger);return result,ledger
   except BaseException as error:
    ledger['failure_count']+=1
    if len(ledger['failures'])<64:ledger['failures'].append(type(error).__name__+': '+str(error))
    ledger['last_failure']=type(error).__name__+': '+str(error);write(root/'actor-retirement-state.json',ledger);pause(.1)
 finally:restore_signals(previous)

def saved_binding(root,report,name,read):
 root=Path(root);ledger=read(root/'actor-retirement-state.json');require(ledger==report['actor_retirement'] and type(ledger['schema'])is int and ledger['schema']==3 and ledger['retirement_in_progress'] is False and ledger['owned_terminal_and_removed'] is True and type(ledger['failure_count'])is int and ledger['failure_count']==0 and ledger['failures']==[] and ledger['interruption_signals']==[],'Original actor retirement recovery/interruption cannot qualify normal closure');result=ledger['result'];terminal=read(root/'owned-terminal-inspection.json');require(result['state']==terminal['State']==report['state'] and result['removed'] is True and result['owned_actor_actually_terminal'] is True and result['normal_terminal'] is True and result['absence_recovery_failures']==[] and not(root/'owned-absence-recovery.json').exists() and result['unknown_ownership_or_deadline_as_absence'] is False and result['normal_qualification_from_cleanup_only'] is False,'Original actual terminal/removal scope differs')
 removal=read(root/'owned-removal-receipt.json');require(removal['command']==['docker','rm',name] and type(removal['return_code'])is int and removal['return_code']==0,'Original actual removal receipt differs')
 stop=root/'owned-stop-receipt.json'
 if stop.exists():
  row=read(stop);require(row['command']==['docker','kill','--signal','TERM',name] and type(row['return_code'])is int and row['return_code']==0,'Original exact owned stop receipt differs')
 return {'actual_normal_terminal_and_owned_removal_recollected':True,'recovery_or_cleanup_as_normal_qualification':False}

def external_recovery(root,name,absent):
 """ROOT-only explicit terminal/removal proof; never normal qualification."""
 import json
 from serial37_canonical_json_v3 import canonical
 from full_cache_shared_runtime_v5 import c1
 from full_cache_shared_memory_capture_v5 import recipe_binding
 root=Path(root);path=root/'external-owned-terminal-removal.json'
 require(path.is_file()and not path.is_symlink(),'Explicit owned external recovery receipt absent')
 value=json.loads(path.read_bytes());command=json.loads((root/'launch.command.json').read_bytes());owner=json.loads((root/'actor-owner.json').read_bytes())
 require(value['schema']==1 and type(value['schema'])is int and canonical(value['actor_owner'])==canonical(owner) and value['command']==command and value['name']==name and value['normal_qualification']is False,'Foreign/normal-promoted recovery refused')
 image=command[-3];terminal=value['terminal_inspection'];recipe_binding(terminal,command,image)
 labels=dict(command[i+1].split('=',1)for i,v in enumerate(command)if v=='--label');require(all(terminal['Config']['Labels'].get(k)==v for k,v in labels.items()),'Exact original ownership labels required')
 require(terminal['Name']=='/'+name and terminal['Id']==value['container_id'] and terminal['State']['Running']is False,'Exact originally owned terminal required')
 removal=value['removal_receipt'];require(removal['command']==['docker','rm',value['container_id']]and type(removal['return_code'])is int and removal['return_code']==0 and removal['error']is None,'Actual original exact-ID removal required')
 c1.leased([0,1]);require(absent(name)is True,'Current authoritative absence after exact-ID removal required')
 return {'state':terminal['State'],'removed':True,'owned_actor_actually_terminal':True,'normal_terminal':False,'external_recovery':value,'normal_qualification_from_cleanup_only':False,'unknown_ownership_or_deadline_as_absence':False,'absence_recovery_failures':[]}
