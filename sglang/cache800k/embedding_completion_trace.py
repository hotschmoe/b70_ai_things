"""Diagnostic device-wide fences around embedding reduction; not a serving fix."""
import itertools
import json
import os
import time

_sequence = itertools.count()


def emit(stage, tensor, rank, sequence, route=None, completed=False):
    record = dict(stage=stage, rank=rank, sequence=sequence, route=route,
                  pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                  shape=list(tensor.shape), stride=list(tensor.stride()),
                  dtype=str(tensor.dtype), device=str(tensor.device),
                  bytes=tensor.numel() * tensor.element_size(),
                  device_completion_observed=completed,
                  completion_scope='device-wide preceding work' if completed else None)
    print('B70_EMBED_TRACE ' + json.dumps(record, ensure_ascii=True), flush=True)


def fence(stage, tensor, rank, sequence, route=None):
    import torch
    if tensor.device.type != 'xpu':
        raise ValueError('completion control requires an XPU tensor')
    emit(stage + '_sync_entry', tensor, rank, sequence, route)
    torch.xpu.synchronize(tensor.device)
    emit(stage + '_sync_return', tensor, rank, sequence, route, completed=True)


def producer(tensor, rank):
    sequence = next(_sequence)
    emit('producer_host_return', tensor, rank, sequence)
    return sequence


def reduce_call(function, tensor, rank, sequence, route):
    # The pre-fence includes ALL prior device work, not just embedding production.
    fence('pre_collective', tensor, rank, sequence, route)
    emit('collective_host_entry', tensor, rank, sequence, route)
    result = function(tensor)
    emit('collective_host_return', result, rank, sequence, route)
    fence('post_collective', result, rank, sequence, route)
    return result
