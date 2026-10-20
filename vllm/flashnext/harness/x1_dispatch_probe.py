"""Actual-weight check of the decode/prefill quantization boundary."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from vllm.models.qwen4_exp.gguf.ops_interface import moe

torch.xpu.set_device(0)
torch.ops.load_library(os.environ["B70_GGUF_SO"])
directory = Path("/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS")
readers = [GGUFReader(str(p)) for p in sorted(directory.glob("*.gguf"))]
tensors = {t.name: t for r in readers for t in r.tensors}
weights, kinds = [], []
for which in ("gate", "up", "down"):
    tensor = tensors[f"blk.0.ffn_{which}_exps.weight"]
    k, n, e = map(int, tensor.shape)
    raw = np.asarray(tensor.data).view(np.uint8).reshape(e, -1)[:16]
    weights.append(torch.from_numpy(np.array(raw, copy=True)).to("xpu"))
    kinds.append(int(tensor.tensor_type))
    if which == "gate":
        hidden, inter = k, n
torch.manual_seed(31)
x = torch.randn(17, hidden, device="xpu", dtype=torch.bfloat16)
ids = torch.arange(170, device="xpu", dtype=torch.int32).reshape(17, 10) % 16
ids[:, 1::3] = -1
router = torch.softmax(torch.randn(17, 10, device="xpu"), dim=-1)


def run(m):
    return moe(x[:m], *weights, *kinds, router[:m], ids[:m], 16, inter)


for canonical in ("0", "1"):
    os.environ["B70_MTP_CANONICAL_MOE"] = canonical
    seven = run(7)
    for m in (9, 17):
        full = run(m)
        unequal = int((seven != full[:7]).sum())
        maximum = float((seven.float() - full[:7].float()).abs().max())
        print(
            json.dumps(
                dict(canonical=canonical, rows=m, unequal=unequal, max_abs=maximum)
            ),
            flush=True,
        )
        if canonical == "1":
            assert unequal == 0
    if canonical == "1":
        reference = run(17)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            captured = run(17)
        graph.replay()
        torch.testing.assert_close(captured, reference, atol=0, rtol=0)
print("X1_CANONICAL_Q8_DISPATCH_OK", flush=True)
