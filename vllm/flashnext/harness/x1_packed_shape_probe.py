"""Actual target packed weights: row-count and graph replay equivalence."""

import json
import os
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader


def difference(a, b):
    return dict(
        unequal=int((a != b).sum()),
        max_abs=float((a.float() - b.float()).abs().max()),
        bf16_unequal=int((a.bfloat16() != b.bfloat16()).sum()),
        bf16_max_abs=float((a.bfloat16().float() - b.bfloat16().float()).abs().max()),
    )


def emit(op, a, b, **extra):
    print(json.dumps(dict(op=op, **extra, **difference(a, b))), flush=True)


torch.ops.load_library(
    os.environ.get(
        "B70_GGUF_SO",
        "/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/"
        "b70_gguf_xpu/b70_gguf_xpu/_C.o3.so",
    )
)
torch.manual_seed(31)
directory = Path("/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS")
readers = [GGUFReader(str(p)) for p in sorted(directory.glob("*.gguf"))]
tensors = {t.name: t for r in readers for t in r.tensors}
device = "xpu:0"
seen = set()
for tensor in tensors.values():
    shape, kind = tuple(map(int, tensor.shape)), int(tensor.tensor_type)
    if len(shape) != 2 or kind in (0, 1, 30) or kind in seen:
        continue
    k, n = shape
    if max(k, n) > 16384 or k % 256:
        continue
    seen.add(kind)
    w = torch.from_numpy(np.array(tensor.data, copy=True)).view(torch.uint8).to(device)
    x = torch.randn(6, k, device=device, dtype=torch.bfloat16)
    one = torch.cat([torch.ops.b70_gguf.gemv(row[None], w, kind, n, k) for row in x])
    for m in (2, 3, 6):
        batch = torch.ops.b70_gguf.gemv(x[:m], w, kind, n, k)
        emit("target_dense", batch, one[:m], tensor=tensor.name, qtype=kind, rows=m)
    del w, x, one

for layer in (0, 1, 3, 8, 12, 24, 36, 47):
    weights, types = [], []
    for component in ("gate", "up", "down"):
        tensor = tensors[f"blk.{layer}.ffn_{component}_exps.weight"]
        k, n, e = map(int, tensor.shape)
        # Keep sixteen actual experts; no generated quant bytes or dequant.
        raw = np.asarray(tensor.data).view(np.uint8).reshape(e, -1)[:16]
        weights.append(torch.from_numpy(np.array(raw, copy=True)).to(device))
        types.append(int(tensor.tensor_type))
        if component == "gate":
            hidden, inter = k, n
    ids = torch.arange(60, device=device, dtype=torch.int32).reshape(6, 10) % 16
    ids[:, 1::3] = -1
    router = torch.full((6, 10), 0.1, device=device)
    empty = torch.empty(0, device=device, dtype=torch.uint8)
    x = torch.randn(6, hidden, device=device, dtype=torch.bfloat16)

    def moe(value, local_ids, local_router, weights=weights):
        return torch.ops.b70_gguf.moe_layer(
            value,
            weights[0],
            types[0],
            weights[1],
            types[1],
            weights[2],
            types[2],
            local_ids,
            local_router,
            empty,
            0,
            empty,
            0,
            empty,
            0,
            inter,
            hidden,
        )

    one = torch.cat(
        [moe(x[j : j + 1], ids[j : j + 1], router[j : j + 1]) for j in range(6)]
    )
    for m in (2, 3, 6):
        batch = moe(x[:m], ids[:m], router[:m])
        emit("target_moe", batch, one[:m], layer=layer, qtypes=types, rows=m)
    expected = moe(moe(x, ids, router).bfloat16(), ids, router)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        actual = moe(moe(x, ids, router).bfloat16(), ids, router)
    for replay in range(3):
        graph.replay()
        emit("target_moe_graph_two_calls", actual, expected, layer=layer, replay=replay)
    del graph, weights, x, one, expected, actual
torch.xpu.synchronize()
print("X1_TARGET_PACKED_SHAPE_DONE", flush=True)
