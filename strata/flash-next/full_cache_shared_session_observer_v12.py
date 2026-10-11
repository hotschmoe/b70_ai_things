"""Root producer hooks original batch0 session method; no command or state edits."""
import threading

def install(server,emit,local,counter,lock,active_calls):
 original=server.StrataEngine.session_file
 def session_file(self,action,path):
  if getattr(self,'batch',0)!=0:raise ValueError('Purpose persisted observer is batch0 only')
  with lock:
   if active_calls:raise ValueError('Original serial FIFO session overlapped a generating request')
   counter[0]+=1;call=counter[0];active_calls.add(call)
  local.call=call;owner={'engine_pid':self.proc.pid,'engine_generation':self.gen};emit(dict(owner,kind='session_begin',call=call,action=action,path=path));result=None;error=None
  try:
   result=original(self,action,path);return result
  except server.SessionRefused as exc:
   error={'kind':exc.kind,'published':exc.published,'message':str(exc)};raise
  except BaseException as exc:
   error={'type':type(exc).__name__,'message':str(exc)};raise
  finally:
   if self.gen!=owner['engine_generation']:error=error or {'type':'EngineGenerationChanged','message':'Session changed native incarnation'}
   emit(dict(owner,kind='session_end',call=call,action=action,path=path,result=result,error=error));local.call=None
   with lock:active_calls.discard(call)
 server.StrataEngine.session_file=session_file
 return {'original_session_method_passed_through':True,'persisted_command_or_state_modified':False}
