"""BF16 fused GDN verification versus the same fused one-token recurrence."""
import json
import os
import sys
from pathlib import Path

import torch
import d3_probe as d3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'plugins/b70_gdn_xpu'))
import b70_gdn_xpu

torch.xpu.set_device(0)
b70_gdn_xpu.register()
rows = []
for width in (3, 4):
    for accepted in range(1, width + 1):
        for seed in range(3):
            torch.manual_seed(seed)
            q, ba, c0, s0, w, a, dt, nw = d3.make_gdn(width, width + 2, width + 1, seed)
            q, ba, c0, w, dt = [x.bfloat16() for x in (q, ba, c0, w, dt)]
            cs, ss = c0.clone(), s0.clone()
            cd = c0[:1, accepted-1:accepted+2].contiguous().clone()
            sd = s0[accepted-1:accepted].clone()
            cu = torch.tensor([0, width], dtype=torch.int32, device='xpu')
            slots = torch.arange(width, dtype=torch.int32, device='xpu').reshape(1, width)
            tokens = torch.arange(width, dtype=torch.int32, device='xpu')
            acc = torch.tensor([accepted], dtype=torch.int32, device='xpu')
            has = torch.ones(1, dtype=torch.bool, device='xpu')
            cu1 = torch.tensor([0, 1], dtype=torch.int32, device='xpu')
            slot1 = torch.zeros(1, dtype=torch.int32, device='xpu')
            core = torch.empty((width, 32, 128), dtype=torch.bfloat16, device='xpu')
            out = torch.empty_like(core)
            cores, outs, states = [], [], []
            for t in range(width):
                cr, ot = torch.empty_like(core[:1]), torch.empty_like(out[:1])
                torch.ops.b70_gdn.decode_fused(cr, ot, q[t:t+1], ba[t:t+1], cd, sd,
                    w, a, dt, nw, cu1, slot1, has, None, None, 1e-6)
                cores.append(cr)
                outs.append(ot)
                states.append(sd.clone())
            expected_core, expected_out = torch.cat(cores), torch.cat(outs)

            def spec():
                cs.copy_(c0)
                ss.copy_(s0)
                torch.ops.b70_gdn.decode_fused(core, out, q, ba, cs, ss,
                    w, a, dt, nw, cu, slots, has, tokens, acc, 1e-6)

            spec()
            torch.xpu.synchronize()
            row = dict(width=width, accepted=accepted, seed=seed,
                core_exact=torch.equal(core, expected_core),
                out_exact=torch.equal(out, expected_out),
                states_exact=torch.equal(ss[:width], torch.cat(states)),
                final_conv_exact=torch.equal(cs[:1, -3:], cd),
                core_maxabs=(core.float()-expected_core.float()).abs().max().item(),
                state_maxabs=(ss[:width]-torch.cat(states)).abs().max().item())
            assert all(row[k] for k in ('core_exact', 'out_exact', 'states_exact', 'final_conv_exact')), row
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                spec()
            for _ in range(8):
                graph.replay()
            torch.xpu.synchronize()
            row['replay_exact'] = torch.equal(out, expected_out)
            assert row['replay_exact'], row
            del graph
            rows.append(row)
            print(json.dumps(row), flush=True)
Path(os.environ['X5_PROBE_OUT']).write_text(json.dumps(rows, indent=2) + '\n')
print('X5_GDN_SPEC_PROBE_PASS', len(rows), flush=True)
