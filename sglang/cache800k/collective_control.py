#!/usr/bin/env python3
"""Fresh-process subgroup diagnostic, not a loaded-model reproduction.

GPU invocation requires the campaign's leased lifecycle and an outer timeout.
This module imports no Torch and touches no device until main is called.
"""

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time


def check_environment(expected, actual):
    differences = {key: [value, actual.get(key)] for key, value in expected.items()
                   if actual.get(key) != value}
    if differences:
        raise RuntimeError('environment mismatch: ' + json.dumps(differences))
    if actual.get('CCL_TOPO_P2P_ACCESS') != '0':
        raise RuntimeError('this control requires P2P off')


def loaded_libraries():
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith('/'):
            path = parts[5]
            if any(name in Path(path).name for name in
                   ('libccl', 'libsycl', 'libur_', 'libze_', 'libtorch', 'libc10')):
                paths.add(path)
    return {path: hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()
            for path in sorted(paths)}


def fenced_reduce(tensor, group, synchronize, reduce, emit):
    emit('pre_fence_entry')
    synchronize()
    emit('pre_fence_return')
    emit('collective_host_entry')
    reduce(tensor, group=group)
    emit('collective_host_return_not_completion')
    emit('post_fence_entry')
    synchronize()
    emit('post_fence_return')


def make_input(torch, rows, rank, iteration, device):
    # Small exactly representable integers permit a strict FP16 SUM oracle.
    base = torch.arange(rows * 5120, dtype=torch.int32, device=device)
    return (base.remainder_(31).to(torch.float16).reshape(rows, 5120)
            + rank * 3 + iteration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment-reference', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.environment_reference.read_text())
    check_environment(reference['env'], os.environ)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or rank not in (0, 1) or local_rank != rank:
        raise RuntimeError('requires two local ranks mapped to XPU 0 and 1')
    import torch
    import torch.distributed as dist

    def emit(stage, **metadata):
        print(json.dumps(dict(stage=stage, rank=rank, pid=os.getpid(),
                              monotonic_ns=time.monotonic_ns(), **metadata)), flush=True)

    emit('identity', torch_version=torch.__version__, torch_path=torch.__file__,
         environment={key: os.environ.get(key) for key in reference['env']},
         scope='fresh process; synthetic arange/add producer; no model allocations')
    torch.xpu.set_device(local_rank)
    timeout = timedelta(seconds=45)
    dist.init_process_group(backend='xccl', timeout=timeout)
    group = dist.new_group(ranks=[0, 1], backend='xccl', timeout=timeout)
    emit('group_ready', backend='xccl', group_ranks=[0, 1], group_is_world=False)
    for sequence, (rows, iteration) in enumerate([(4, 0), (2048, 0),
                                                 (2048, 1), (2048, 2)]):
        tensor = make_input(torch, rows, rank, iteration, f'xpu:{local_rank}')
        metadata = dict(sequence=sequence, shape=list(tensor.shape),
                        stride=list(tensor.stride()), dtype=str(tensor.dtype),
                        device=str(tensor.device), bytes=tensor.numel() * tensor.element_size(),
                        allocation='fresh arange/remainder/cast/reshape/add output')
        emit('producer_host_return', **metadata)
        # Record actual GPU-process mapped library bytes, not CPU-import identity.
        emit('mapped_libraries_before_reduce', libraries=loaded_libraries(), **metadata)
        fenced_reduce(tensor, group, torch.xpu.synchronize, dist.all_reduce,
                      lambda stage: emit(stage, **metadata))
        expected = make_input(torch, rows, 0, iteration, f'xpu:{local_rank}') * 2 + 3
        if not torch.equal(tensor, expected):
            raise RuntimeError(f'SUM mismatch sequence={sequence}')
        emit('numerical_pass', **metadata)
    emit('mapped_libraries_after_reduce', libraries=loaded_libraries())
    # On failure let the outer lifecycle terminate/recover; no blocking cleanup
    # in an exception handler that might conceal the first failure.
    dist.destroy_process_group(group)
    dist.destroy_process_group()
    emit('control_pass', reductions=4, limitation='no loaded-model qualification')


if __name__ == '__main__':
    main()
