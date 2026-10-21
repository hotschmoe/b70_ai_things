"""Pure CPU oracle validation for the graph-shaped packed selector."""
import sys
from pathlib import Path
import numpy as np
import torch
from oracle import CANARY, cases, reference
from selector import PackedTopk

torch.set_num_threads(2)
count = 0
for name, bits, n, width in cases():
    cols = max(2051, len(bits))
    rows = np.zeros((4, cols), dtype=np.uint32)
    rows[:, :len(bits)] = bits
    step = torch.zeros((4, 4), dtype=torch.int32)
    # Keep the original logical n clamp when score vectors were padded.
    nn = min(n, len(bits))
    step[:2, 1] = nn
    step[:2, 3] = width
    ids = torch.full((4,2051), CANARY, dtype=torch.int32)
    selector = PackedTopk(cols, 2051, 'cpu')
    scores = torch.from_numpy(rows.view(np.float32))
    selector(scores, ids, step)
    expected = np.full((4,2051), CANARY, dtype=np.int32)
    expected[:2] = reference(bits, nn, width)
    np.testing.assert_array_equal(ids.numpy(), expected, err_msg=name)
    count += 1

# Actual serving score-column budget with active n well below score width.
rng = np.random.default_rng(20261021)
for n in (4096,16384,32768):
    bits = rng.integers(0,2**32,(4,147456),dtype=np.uint32)
    step = torch.zeros((4,4),dtype=torch.int32)
    step[0,1] = n; step[0,3] = 2048
    step[1,1] = n+1; step[1,3] = 2051
    ids = torch.full((4,2051),CANARY,dtype=torch.int32)
    select = PackedTopk(147456,2051,'cpu')
    select(torch.from_numpy(bits.view(np.float32)),ids,step)
    expected = np.full((4,2051),CANARY,dtype=np.int32)
    for row in (0,1):
        expected[row] = reference(bits[row],int(step[row,1]),int(step[row,3]))
    np.testing.assert_array_equal(ids.numpy(),expected)
    count += 1
print(f'TORCH_PACKED_SELECTOR_CPU_PASS cases={count} signed_map=exact padded=147456 inactive_canaries=exact',flush=True)
