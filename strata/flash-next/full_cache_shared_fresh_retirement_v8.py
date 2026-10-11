"""ROOT ONLY retain fresh constructor handles and join exact owned retirement."""
import time
from pathlib import Path
from full_cache_shared_actor_retirement_v8 import retain_until_settled,original_signals,restore_signals

def original_protocol(cls,command,output):
 """Allocate before __init__: a readiness exception must not lose its Popen."""
 protocol=cls.__new__(cls)
 return protocol

def controlled_signals(result):
 result['interruption_signals']=[]
 def interrupted(number,frame):
  result['interruption_signals'].append(number)
  raise KeyboardInterrupt('Original fresh actor controlled stop '+str(number))
 return original_signals(interrupted)

def finish(protocol,name,owned,run,absent,write,output,grace=90):
 interruptions=[];previous=original_signals(lambda number,frame:interruptions.append(number))
 write(Path(output)/'actor-retirement-state.json',{'schema':3,'retirement_in_progress':True,'owned_terminal_and_removed':False,'interruption_signals':interruptions})
 try:
  rc,result,ledger,errors=_finish(protocol,name,owned,run,absent,write,output,grace)
  ledger['interruption_signals']+=interruptions;ledger['original_protocol_CLI_and_pipe_EOF_joined']=True;ledger['retirement_in_progress']=False
  write(Path(output)/'actor-retirement-state.json',ledger)
  return rc,result,ledger,errors
 finally:restore_signals(previous)

def _finish(protocol,name,owned,run,absent,write,output,grace):
 """Unknown inspection never turns into absence or releases inherited leases."""
 errors=[];rc=None;deadline=time.monotonic()+grace
 process=getattr(protocol,'p',None)
 if process is not None and process.poll() is None:
  try:protocol.send('QUIT')
  except BaseException as error:errors.append(type(error).__name__+': '+str(error))
  while process.poll() is None and time.monotonic()<deadline:time.sleep(.1)
  if process.poll() is None:errors.append('Original fresh QUIT deadline expired; exact-owned retirement required')
 result,ledger=retain_until_settled(name,owned,run,absent,time.sleep,write,output)
 ledger['retirement_in_progress']=True;write(Path(output)/'actor-retirement-state.json',ledger)
 # Exact container terminal/removal is insufficient by itself: preserve the
 # original Docker CLI and producer pipe until actual wait and pump EOF.
 if process is not None:
  rc=process.wait()
  if process.stdin is not None:
   try:process.stdin.close()
   except BaseException as error:errors.append('original stdin close: '+type(error).__name__+': '+str(error))
  for thread in getattr(protocol,'threads',[]):thread.join()
  for role in ('stdout','stderr'):
   stream=getattr(process,role,None)
   if stream is not None:
    try:stream.close()
    except BaseException as error:errors.append('original '+role+' close: '+type(error).__name__+': '+str(error))
 if hasattr(protocol,'err'):
  try:protocol.err.close()
  except BaseException as error:errors.append('original merged sink close: '+type(error).__name__+': '+str(error))
 Path(output,'engine.protocol.log').write_text('\n'.join(getattr(protocol,'stdout',[]))+'\n',encoding='ascii')
 if process is None:errors.append('Original protocol launch handle was never observed')
 return rc,result,ledger,errors
