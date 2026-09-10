"""Opt-in startup Python boundary trace. Never reads tensor values or waits XPU."""
import collections
import functools
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import threading
import time

EXPECTED = {'runner': '4e0e1aee778bb1f690b393061fc36736f390c7519dc93f41c776f154b25e8261', 'communicator': '5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d'}
_local = threading.local()
_ids = itertools.count()
_fd = None
_events = 0
MRV1_RUNNER_SHA256 = '7299b4cfabc447b15b66a5fcaf5bb858a0783fc46340d561f292f5f96e99e789'
PROFILE_IMAGES = {
    'phase': 'sha256:0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077',
    'phase-mrv1': 'sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1',
}


def emit(event):
    global _fd, _events
    limit = int(os.environ.get('B70_TP_HOST_TRACE_MAX_EVENTS', '100000'))
    if _events > limit:
        return
    if _fd is None:
        directory = Path(os.environ['B70_TP_HOST_TRACE_DIR'])
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _fd = os.open(str(directory / ('host-%d.jsonl' % os.getpid())),
                      os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    if _events == limit:
        event = {'event': 'truncated', 'limit': limit, 'counts_incomplete': True}
    _events += 1
    row = dict(event, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
    os.write(_fd, (json.dumps(row, ensure_ascii=True, default=str)+'\n').encode('ascii'))


def verify(cls, kind):
    profile = os.environ.get('B70_TP_HOST_TRACE_PROFILE', 'phase')
    if profile not in PROFILE_IMAGES:
        raise RuntimeError('Unreviewed TP host trace profile: '+profile)
    path = Path(inspect.getfile(cls))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = MRV1_RUNNER_SHA256 if kind == 'runner' and profile == 'phase-mrv1' else EXPECTED[kind]
    if digest != expected:
        raise RuntimeError('TP host trace source mismatch: '+str(path))
    emit({'event': 'installed', 'kind': kind, 'source_sha256': digest,
          'profile': profile, 'expected_image': PROFILE_IMAGES[profile],
          'image_verification': 'external lifecycle required; source hash checked here'})


def metadata(value):
    # shape/dtype/device/stride are host-side Tensor metadata, never values.
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device), 'stride': list(value.stride())}
    if isinstance(value, (tuple, list)):
        return [metadata(item) for item in value[:16]]
    return None


def install_runner(cls):
    verify(cls, 'runner')
    original = cls._dummy_run
    signature = inspect.signature(original)
    @functools.wraps(original)
    def traced(self, *args, **kwargs):
        from vllm.distributed.parallel_state import get_tp_group
        group = get_tp_group()
        bound = signature.bind(self, *args, **kwargs); bound.apply_defaults()
        fields = {k: v for k,v in bound.arguments.items() if k != 'self'}
        if not group.use_custom_op_call:
            raise RuntimeError('Host trace requires opaque custom-op collective route')
        scope = {'id': next(_ids), 'rank': group.rank_in_group,
                 'world_size': group.world_size, 'group': group.unique_name,
                 'use_custom_op_call': group.use_custom_op_call,
                 'arguments': fields, 'phase': ('profile' if fields.get('is_profile') else 'capture' if fields.get('is_graph_capturing') else 'warmup_or_replay'),
                 'counts': collections.Counter(), 'shape_counts': collections.Counter()}
        previous = getattr(_local, 'scope', None); _local.scope = scope
        try:
            emit({'event': 'dummy_enter', **scope, 'counts': {}})
            result = original(self, *args, **kwargs)
            emit({'event': 'dummy_return', **scope, 'outputs': metadata(result),
                  'semantics': 'Python return only; not device completion'})
            return result
        except BaseException as exc:
            emit({'event': 'dummy_exception', **scope, 'exception': type(exc).__name__})
            raise
        finally:
            _local.scope = previous
    cls._dummy_run = traced


def install_communicator(cls):
    verify(cls, 'communicator')
    for name in ('all_reduce','all_gather','all_gatherv','reduce_scatter',
                 'reduce_scatterv','gather','broadcast'):
        if not hasattr(cls, name):
            continue
        original = getattr(cls, name)
        def wrap(original, name):
            @functools.wraps(original)
            def traced(self, *args, **kwargs):
                scope = getattr(_local, 'scope', None)
                if scope is None:
                    return original(self, *args, **kwargs)
                call = next(_ids)
                detail = bool(scope['arguments'].get('is_profile'))
                comm = {key:getattr(self,key,None) for key in ('unique_name','world_size','rank_in_group','global_rank')}
                if comm['unique_name'] != scope['group']:
                    scope['counts']['non_tp_communicator_skipped'] += 1
                    return original(self,*args,**kwargs)
                meta = metadata(args[0] if args else kwargs.get('input_'))
                scope['shape_counts'][json.dumps({'op':name,'input':meta},default=str,sort_keys=True)] += 1
                scope['counts'][name+'_enter'] += 1
                if detail: emit({'event': 'collective_enter','scope': scope['id'], 'call': call,
                      'rank': scope['rank'], 'communicator':comm, 'op': name,
                      'input': meta})
                try:
                    result = original(self, *args, **kwargs)
                except BaseException as exc:
                    scope['counts'][name+'_exception'] += 1
                    emit({'event':'collective_exception','scope':scope['id'],
                          'call':call,'rank':scope['rank'],'communicator':comm,'op':name,
                          'exception':type(exc).__name__})
                    raise
                scope['counts'][name+'_return'] += 1
                if detail: emit({'event':'collective_return','scope':scope['id'],'call':call,
                      'rank':scope['rank'],'communicator':comm,'op':name,'output':metadata(result),
                      'semantics':'Python return, existing backend wait unchanged'})
                return result
            return traced
        setattr(cls, name, wrap(original, name))


def install():
    # Called by existing sitecustomize HookLoader only after runner module import.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
    import vllm.distributed.parallel_state as parallel_state
    import vllm.distributed.communication_op as communication_op
    for module, expected in ((parallel_state, 'c6d65b96a260ea0d779da1073d1799392813eee74c9ed44753dfa56f4e7a9e4c'),
                             (communication_op, '3c3f8ac60db38ece891d39dc1e2f6f1947ad438685893d2e5777d072320bd1b8')):
        digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError('TP host trace route source mismatch: '+module.__name__)
        emit({'event':'route_source','module':module.__name__,'source_sha256':digest})
    from vllm.distributed.device_communicators import base_device_communicator as base_comm
    digest=hashlib.sha256(Path(base_comm.__file__).read_bytes()).hexdigest()
    if digest != '9377a17525b57abc2a362df26db0d7792c20c4573b34b42c6207cd6a99883d3b':
        raise RuntimeError('TP host trace base communicator source mismatch')
    emit({'event':'route_source','module':base_comm.__name__,'source_sha256':digest})
    install_communicator(XpuCommunicator)
    install_runner(GPUModelRunner)
