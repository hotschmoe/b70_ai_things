"""Actual Q4 trunk Q8 data: numeric, batch invariance, graphs and latency."""
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from gguf import GGUFReader
import b70_gguf_xpu
import b70_x5_xpu

torch.xpu.set_device(0)
helper = ctypes.CDLL('/mnt/vm_8tb/b70/build/vllm-flashnext/x5/eventless-lazy/libx5_eventless_helper.so', mode=ctypes.RTLD_GLOBAL)
b70_gguf_xpu.load()
b70_x5_xpu.load()
out = Path(os.environ['X5_Q8_OUT'])
reader = GGUFReader('/mnt/cache/b70/devmodel/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf')
tensors = {t.name: t for t in reader.tensors}
weights = []
for names in [('blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight'), ('blk.0.ssm_out.weight',)]:
    parts = []
    for name in names:
        t = tensors[name]
        assert int(t.tensor_type) == 8
        n, k = reversed(tuple(int(v) for v in t.shape))
        parts.append(torch.from_numpy(np.array(t.data, copy=True)).reshape(n, k//32*34))
    packed = torch.cat(parts).contiguous().to('xpu')
    n = packed.shape[0]
    blocks = packed.reshape(n, k//32, 34)
    codes = blocks[:, :, 2:].contiguous().view(torch.int8).reshape(n, k)
    scales = blocks[:, :, :2].contiguous().view(torch.float16).reshape(n, k//32)
    weights.append((names, packed, codes, scales, n, k))
torch.xpu.synchronize()
(out/'ready').touch()
deadline = time.monotonic() + 7200
while not (out/'go').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('X5 Q8 quiet trigger')
    time.sleep(.2)
results = []
for names, packed, codes, scales, n, k in weights:
    base = None
    for m in [1, 2, 3, 4, 8]:
        torch.manual_seed(701)
        x = torch.randn((1, k), dtype=torch.bfloat16, device='xpu').repeat(m, 1).contiguous()
        reference = torch.ops.b70_gguf.gemv(x, packed, 8, n, k)
        actual = torch.ops.b70_x5.q8_linear(x, codes, scales, True)
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=1e-5)
        y = torch.ops.b70_x5.q8_linear(x, codes, scales)
        assert torch.equal(y, actual.to(torch.bfloat16)), 'BF16 epilogue differs'
        cpu = y.cpu()
        if base is None:
            base = cpu[0].clone()
        assert torch.equal(cpu, base.expand_as(cpu)), 'M-dependent output'
        timings = {}
        for label, fn in [('old', lambda: torch.ops.b70_gguf.gemv(x, packed, 8, n, k).to(torch.bfloat16)),
                          ('new', lambda: torch.ops.b70_x5.q8_linear(x, codes, scales))]:
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                captured = fn()
            for _ in range(16): graph.replay()
            torch.xpu.synchronize()
            start, end = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            start.record()
            for _ in range(64): graph.replay()
            end.record(); torch.xpu.synchronize()
            timings[label] = start.elapsed_time(end)/64
            if label == 'new':
                assert torch.equal(captured.cpu(), cpu), 'graph replay differs'
        results.append(dict(names=names, m=m, n=n, k=k,
                            max_abs=(actual-reference).abs().max().item(), timing_ms=timings,
                            batch_exact=True, graph_exact=True))
        print(json.dumps(results[-1], ensure_ascii=True), flush=True)
(out/'result.json').write_text(json.dumps(results, indent=2, ensure_ascii=True)+'\n')
(out/'done').touch()
print('X5_Q8_PROBE_NORMAL_EXIT', flush=True)
