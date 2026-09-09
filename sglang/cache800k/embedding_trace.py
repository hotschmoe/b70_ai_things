"""Host metadata only; a returned call is NOT proof of device completion."""
import itertools
import json
import os
import time

_sequence = itertools.count()


def emit(stage, tensor, rank, sequence, route=None):
    record = dict(stage=stage, rank=rank, sequence=sequence, route=route,
                  pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                  shape=list(tensor.shape), stride=list(tensor.stride()),
                  dtype=str(tensor.dtype), device=str(tensor.device),
                  bytes=tensor.numel() * tensor.element_size(),
                  device_completion_observed=False)
    print('B70_EMBED_TRACE ' + json.dumps(record, ensure_ascii=True), flush=True)


def producer(tensor, rank):
    sequence = next(_sequence)
    emit('producer_host_return', tensor, rank, sequence)
    return sequence


def reduce_call(function, tensor, rank, sequence, route):
    emit('collective_host_entry', tensor, rank, sequence, route)
    result = function(tensor)
    emit('collective_host_return', result, rank, sequence, route)
    return result
