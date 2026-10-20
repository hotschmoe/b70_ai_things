"""BF16 D3 port: numerical and replay gate before model serving."""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import vllm_xpu_kernels._xpu_C  # noqa: F401

import d3_probe as d3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/b70_gdn_xpu"))
import b70_gdn_xpu

torch.xpu.set_device(0)  # xpu-run --card 1 exposes only physical card 1.
b70_gdn_xpu.register()
rows = []
for seed in range(3):
    torch.manual_seed(seed)
    q, ba, c0, s0, w, a, dt, nw = d3.make_gdn(1, 3, 4, seed)
    q, ba, c0, w, dt = [x.bfloat16() for x in (q, ba, c0, w, dt)]
    cu = torch.tensor([0, 1], dtype=torch.int32, device="xpu")
    slot = torch.tensor([0], dtype=torch.int32, device="xpu")
    has = torch.ones(1, dtype=torch.bool, device="xpu")
    cs, ss = c0.clone(), s0.clone()
    cf, sf = c0.clone(), s0.clone()
    core_s = torch.empty((1, 32, 128), dtype=torch.bfloat16, device="xpu")
    z = torch.empty_like(core_s)
    core_f, out = torch.empty_like(core_s), torch.empty_like(core_s)

    def stock():
        cs.copy_(c0)
        ss.copy_(s0)
        d3.stock_nonspec(core_s, z, q, ba, cs, ss, w, a, dt, cu, slot, has)

    def fused():
        cf.copy_(c0)
        sf.copy_(s0)
        torch.ops.b70_gdn.decode_fused(core_f, out, q, ba, cf, sf, w, a, dt,
                                      nw, cu, slot, has, None, None, 1e-6)

    stock()
    fused()
    expected = (core_s.float() * torch.rsqrt(core_s.float().square().mean(-1, keepdim=True) + 1e-6)
                * nw * F.silu(z.float())).bfloat16()
    torch.xpu.synchronize()
    cosine = F.cosine_similarity(expected.float().flatten(), out.float().flatten(), dim=0).item()
    row = dict(seed=seed, core=d3.stats(core_s, core_f), out=d3.stats(expected, out),
               ssm=d3.stats(ss, sf), conv_exact=torch.equal(cs, cf), cosine=cosine)
    print(json.dumps(row), flush=True)
    assert row["conv_exact"] and cosine > 0.99998, row
    torch.testing.assert_close(core_f, core_s, rtol=0.02, atol=1e-4)
    # D3's retained FP16 probe already reports state differences up to
    # 5.85e-5. BF16 scalar rounding differs too; log the full error and
    # require 1e-4 absolute / 0.5 percent relative, not bit equality.
    torch.testing.assert_close(sf, ss, rtol=0.005, atol=1e-4)
    reference = out.clone()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        fused()
    for _ in range(32):
        graph.replay()
    torch.xpu.synchronize()
    row["replay_exact"] = torch.equal(reference, out)
    assert row["replay_exact"]
    del graph
    rows.append(row)
print(json.dumps(dict(passed=True, rows=rows), ensure_ascii=True), flush=True)
Path(os.environ["X5_PROBE_OUT"]).write_text(json.dumps(rows, indent=2) + "\n")
