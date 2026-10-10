#!/usr/bin/env python3
"""Source40 output-only observation of original EOS background waiter."""
import hashlib, inspect, json, os, queue, threading, types
from pathlib import Path

SERVER_SHA256='3166f70fa013b905adf713dba404a2eb91a4f3381842dafb542af98cf978d241'
MAX_WAITERS=128
MAX_GETS=4096

def require(ok,message):
 if not ok:raise ValueError(message)

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')).hexdigest()
def cell(value):return (lambda:value).__closure__[0]
def code_shape(code):
 return (code.co_code,code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars,code.co_argcount,code.co_kwonlyargcount,code.co_flags,tuple(code_shape(x) if isinstance(x,types.CodeType) else x for x in code.co_consts))
def find_code(code,name):
 for item in code.co_consts:
  if isinstance(item,types.CodeType):
   if item.co_name==name:return item
   found=find_code(item,name)
   if found is not None:return found
 return None

class Observer:
 def __init__(self,emit,context):self.emit=emit;self.context=context;self.lock=threading.Lock();self.waiters=0;self.gets=0;self.errors=[];self.threads=[];self.waiter_calls={};self.waiter_gets={}
 def record(self,row):
  try:self.emit(row)
  except BaseException as exc:
   with self.lock:self.errors.append(type(exc).__name__+': '+str(exc))
 def failed(self,message):
  with self.lock:self.errors.append(message)
 def proof(self,calls=None):
  if calls is not None:require(type(calls)is list and len(set(calls))==len(calls) and all(type(x)is int and x>0 for x in calls),'Explicit unique phase call roster required')
  with self.lock:
   selected=set(self.waiter_calls) if calls is None else {wid for wid,call in self.waiter_calls.items() if call in calls}
   threads=[{'waiter':wid,'thread_ident':thread.ident,'thread_native_id':thread.native_id,'started':thread.ident is not None,'alive':thread.is_alive(),'daemon':thread.daemon} for wid,thread in self.threads if wid in selected]
   return {'calls':None if calls is None else sorted(calls),'threads':threads,'all_waiter_threads_retired':len(threads)==len(selected) and all(x['started'] and not x['alive'] for x in threads),'schema':1,'server_sha256':SERVER_SHA256,'waiters':len(selected),'queue_returns':sum(self.waiter_gets.get(wid,0) for wid in selected),'errors':list(self.errors),'observer_passed':not self.errors,'changes_frontend_or_native_control':False}
 def factory(self,original_thread,**kwargs):
  target=kwargs.get('target')
  try:
   require(isinstance(target,types.FunctionType) and target.__name__=='wait','Only exact original waiter target supported')
   cells=dict(zip(target.__code__.co_freevars,target.__closure__ or ()))
   require({'q','self','slot','stopped_identity','busy','held'}<=set(cells),'Actual waiter closure differs')
   values={k:v.cell_contents for k,v in cells.items()};engine=values['self'];identity=values['stopped_identity'];slot=values['slot'];ctx=self.context(engine)
   require(type(ctx)is dict and set(ctx)=={'call'} and type(ctx['call'])is int and ctx['call']>0,'Exact API call context required')
   require(type(identity)is tuple and len(identity)==3 and all(type(x)is int and x>0 for x in identity),'Strict actual slot identity required')
   require(type(slot)is int and slot>=0 and kwargs.get('daemon')is True,'Original daemon waiter/slot required')
   with self.lock:
    require(self.waiters<MAX_WAITERS,'Bounded waiter quota exceeded');self.waiters+=1;wid=self.waiters;self.waiter_calls[wid]=ctx['call'];self.waiter_gets[wid]=0
   owner={'call':ctx['call'],'waiter':wid,'engine_pid':engine.proc.pid,'engine_generation':identity[0],'rid':identity[1],'slot_generation':identity[2],'slot':slot,'server_pid':os.getpid(),'queue_object':id(values['q']),'busy_object':id(values['busy']),'held_object':id(values['held']),'server_sha256':SERVER_SHA256}
   require(type(owner['engine_pid'])is int and owner['engine_pid']>0,'Actual native PID required')
   self.record({'kind':'eos_waiter_created',**owner})
   observer=self;raw=values['q']
   class QueueObserver:
    def get(proxy,*args,**kw):
     line=raw.get(*args,**kw)
     with observer.lock:
      observer.gets+=1;observer.waiter_gets[wid]+=1;bounded=observer.gets<=MAX_GETS
     if not bounded:observer.failed('Bounded queue observation quota exceeded')
     elif line is None or (type(line)is str and len(line)<=4096):observer.record({'kind':'eos_waiter_queue_return',**owner,'line':line,'thread_ident':threading.get_ident(),'thread_native_id':threading.get_native_id()})
     else:observer.failed('Queue return unsupported/bounded string')
     return line
   replacement=tuple(cell(QueueObserver()) if name=='q' else cells[name] for name in target.__code__.co_freevars)
   observed=types.FunctionType(target.__code__,target.__globals__,target.__name__,target.__defaults__,replacement)
   require(code_shape(observed.__code__)==code_shape(target.__code__),'Original target code changed')
   def run():
    error=None;self.record({'kind':'eos_waiter_enter',**owner,'thread_ident':threading.get_ident(),'thread_native_id':threading.get_native_id()})
    try:return observed()
    except BaseException as exc:error={'type':type(exc).__name__,'message':str(exc)};raise
    finally:
     try:
      busy=values['busy'][slot];held=list(values['held'][slot]);self.record({'kind':'eos_waiter_exit',**owner,'thread_ident':threading.get_ident(),'thread_native_id':threading.get_native_id(),'current_engine_generation':engine.gen,'current_engine_pid':engine.proc.pid,'current_slot_identity':list(engine.slot_identity[slot]) if engine.slot_identity[slot] is not None else None,'same_current_busy_array':engine.slot_busy is values['busy'],'same_current_queue':engine.slot_q[slot] is raw,'error':error,'busy':busy,'held_ids':held,'held_sha256':digest(held),'same_original_queue':id(raw)==owner['queue_object'],'same_original_busy_array':id(values['busy'])==owner['busy_object']})
     except BaseException as exc:self.failed('Exit observation failed: '+type(exc).__name__)
   prepared={**kwargs,'target':run}
  except BaseException as exc:
   self.failed('Waiter install/identity failed: '+type(exc).__name__+': '+str(exc))
   # Observation failures never change original frontend/thread execution.
   return original_thread(**kwargs)
  thread=original_thread(**prepared)
  with self.lock:self.threads.append((wid,thread))
  return thread

def install(server,emit,context):
 """Return observer proof handle. Caller must retain proof at owned terminal."""
 source=Path(server.__file__).read_bytes();require(hashlib.sha256(source).hexdigest()==SERVER_SHA256,'Exact source40 server required')
 original=server.StrataEngine._release_slot_when_done
 expected=find_code(compile(source,str(server.__file__),'exec'),'_release_slot_when_done');require(expected is not None and code_shape(expected)==code_shape(original.__code__),'Original release method code changed')
 require(original.__globals__.get('threading') is threading,'Original threading global required')
 observer=Observer(emit,context)
 class LocalThreading:
  def __getattr__(self,name):return getattr(threading,name)
  def Thread(self,*args,**kwargs):
   if args:observer.failed('Unexpected positional Thread args');return threading.Thread(*args,**kwargs)
   return observer.factory(threading.Thread,**kwargs)
 globals_copy={**original.__globals__,'threading':LocalThreading()}
 replacement=types.FunctionType(original.__code__,globals_copy,original.__name__,original.__defaults__,original.__closure__);replacement.__kwdefaults__=original.__kwdefaults__
 require(code_shape(replacement.__code__)==code_shape(original.__code__),'Original method code changed')
 server.StrataEngine._release_slot_when_done=replacement
 return observer
