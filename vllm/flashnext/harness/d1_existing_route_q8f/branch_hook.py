"""Default-off, one-return native-op observation. Stdlib before worker setup.

No tensor inspection, synchronization, dispatch replacement, graph changes,
or GPU work. After a successful native call its namespace entry is restored
immediately; the wrapper is absent from subsequent inference Python calls.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time

_TARGETS={'b70_o4_xpu':('b70_o4',('w8_gemv','w8_down_act','w8_up_collapse')),
          'b70_ep_xpu.ops':('b70_ep',('moe_ep_route',))}
_RUNNER='vllm.v1.worker.gpu.model_runner'
_INSTALLED=False
_LOCK=threading.Lock()
_SEEN=set()

def _emit(event,target,**fields):
    # Payload fields are exclusively internally constructed scalar metadata.
    # Never stringify/repr arbitrary args, results, exception payloads/tensors.
    try:
        directory=os.environ.get('B70_D1_BRANCH_DIR','')
        if not directory:return
        row=dict(event=event,target=target,pid=os.getpid(),native_tid=threading.get_native_id(),
                 monotonic_ns=time.monotonic_ns(),**fields)
        raw=(json.dumps(row,ensure_ascii=True,sort_keys=True)+'\n').encode('ascii')
        fd=os.open(str(Path(directory)/('observer-'+str(os.getpid())+'.jsonl')),
                   os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
        try:os.write(fd,raw)
        finally:os.close(fd)
    except Exception:pass

def _observe_ops(namespace,names):
    torch=sys.modules.get('torch')
    if torch is None:return
    try:owner=getattr(torch.ops,namespace)
    except Exception:return
    for name in names:
        target=namespace+'.'+name
        try:original=getattr(owner,name)
        except Exception:continue
        if getattr(original,'_d1_branch_once',False) or target in _SEEN:continue
        def make(original,name,target):
            def call(*args,**kwargs):
                try:value=original(*args,**kwargs)
                except BaseException as exc:
                    try:_emit('op_raise',target,error_type=type(exc).__name__)
                    except Exception:pass
                    raise
                # All observation failures must preserve the original value.
                try:
                    with _LOCK:
                        if target not in _SEEN:
                            _SEEN.add(target)
                            _emit('op_return',target)
                        if getattr(owner,name,None) is call:setattr(owner,name,original)
                except Exception:pass
                return value
            call._d1_branch_once=True
            return call
        try:setattr(owner,name,make(original,name,target))
        except Exception:pass

def _patch(module):
    if module.__name__ in _TARGETS:
        original=getattr(module,'load',None)
        if original is None or getattr(original,'_d1_branch_load',False):return
        ns,names=_TARGETS[module.__name__]
        @functools.wraps(original)
        def load(*args,**kwargs):
            result=original(*args,**kwargs)
            try:_observe_ops(ns,names)
            except Exception:pass
            return result
        load._d1_branch_load=True
        module.load=load
        if getattr(module,'_LOADED',False):_observe_ops(ns,names)
    elif module.__name__==_RUNNER:
        cls=module.GPUModelRunner
        original=cls.capture_model
        if getattr(original,'_d1_branch_capture',False):return
        @functools.wraps(original)
        def capture(self,*args,**kwargs):
            result=original(self,*args,**kwargs)
            try:
                profile=kwargs.get('profile_only',False)
                _emit('capture_return','GPUModelRunner.capture_model',
                      profile_only=profile if type(profile)is bool else None)
                if profile is False:cls.capture_model=original
            except Exception:pass
            return result
        capture._d1_branch_capture=True
        cls.capture_model=capture

class _Loader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):
        fn=getattr(self.original,'create_module',None)
        return fn(spec) if fn else None
    def exec_module(self,module):
        self.original.exec_module(module)
        try:_patch(module)
        except Exception:_emit('hook_error',module.__name__)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in _TARGETS and fullname!=_RUNNER:return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.loader:spec.loader=_Loader(spec.loader)
        return spec

def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get('B70_D1_BRANCH_OBSERVE')!='1':return
    for name in (*_TARGETS,_RUNNER):
        if name in sys.modules:
            try:_patch(sys.modules[name])
            except Exception:_emit('hook_error',name)
    sys.meta_path.insert(0,_Finder())
    _INSTALLED=True
