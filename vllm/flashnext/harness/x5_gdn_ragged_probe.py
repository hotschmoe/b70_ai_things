"""Exact compact-ragged recurrence, rejection carry, paged strides and replay."""
import json
import os
from pathlib import Path
import runpy
import torch
import d3_probe as d3

outdir = Path(os.environ['X5_RAGGED_OUT'])
os.environ['X5_PROBE_OUT'] = str(outdir / 'stock-numeric.json')
# Preserve the stock approximation gate and its original tolerances too.
runpy.run_path(str(Path(__file__).with_name('x5_gdn_probe.py')))
torch.xpu.set_device(0)
cases = [(3, (3, 1)), (3, (1, 3)), (3, (2, 2)), (3, (1, 1)),
         (3, (0, 3)), (3, (3, 0)), (3, (3, 1, 2)),
         (4, (4, 1)), (4, (1, 4)), (4, (3, 2)), (4, (2, 3)),
         (4, (4, 3)), (4, (3, 4)), (4, (2, 4)), (4, (4, 2)),
         (4, (1, 1)), (4, (0, 4)), (4, (4, 0)), (4, (4, 2, 1))]
results = []
for width, initial_lengths in cases:
    for seed in range(2):
        nseq = len(initial_lengths)
        nslots = nseq * width + 1
        torch.manual_seed(100 + seed)
        _, _, c0, s0, w, a, dt, nw = d3.make_gdn(1, width + 2, nslots, seed)
        c0, w, dt = [x.bfloat16() for x in (c0, w, dt)]
        slot = torch.arange(nseq * width, dtype=torch.int32, device='xpu').reshape(nseq, width)
        lengths = list(initial_lengths)
        accepted = [1 + (seed + i) % width for i in range(nseq)]
        for i, length in enumerate(lengths):
            if length == 0 and seed == 0:
                slot[i].fill_(-1)
        for step in range(2):
            ntok = sum(lengths)
            q = torch.randn(ntok, 12288, dtype=torch.bfloat16, device='xpu')
            ba = torch.randn(ntok, 64, dtype=torch.bfloat16, device='xpu')
            order = torch.randperm(ntok).tolist()
            token_idx = torch.tensor(order, dtype=torch.int32, device='xpu')
            starts = [0]
            for length in lengths:
                starts.append(starts[-1] + length)
            cu = torch.tensor(starts, dtype=torch.int32, device='xpu')
            acc = torch.tensor(accepted, dtype=torch.int32, device='xpu')
            has = torch.ones(nseq, dtype=torch.bool, device='xpu')
            core = torch.empty((ntok, 32, 128), dtype=torch.bfloat16, device='xpu')
            output = torch.empty_like(core)
            expected_core, expected_out = torch.zeros_like(core), torch.zeros_like(output)
            expected_c, expected_s = c0.clone(), s0.clone()
            slot_cpu = slot.cpu().tolist()
            for seq, length in enumerate(lengths):
                if not length:
                    continue
                first = slot_cpu[seq][0]
                cs = c0[first:first+1, accepted[seq]-1:accepted[seq]+2].contiguous().clone()
                ss = s0[slot_cpu[seq][accepted[seq]-1]:slot_cpu[seq][accepted[seq]-1]+1].clone()
                cu1 = torch.tensor([0, 1], dtype=torch.int32, device='xpu')
                slot1 = torch.zeros(1, dtype=torch.int32, device='xpu')
                ids = order[starts[seq]:starts[seq+1]]
                for t, row in enumerate(ids):
                    cr, ot = torch.empty_like(core[:1]), torch.empty_like(output[:1])
                    torch.ops.b70_gdn.decode_fused(cr, ot, q[row:row+1], ba[row:row+1],
                        cs, ss, w, a, dt, nw, cu1, slot1, has[:1], None, None, 1e-6)
                    expected_core[row].copy_(cr[0])
                    expected_out[row].copy_(ot[0])
                    expected_s[slot_cpu[seq][t]].copy_(ss[0])
                expected_c[first, :2].copy_(c0[first, accepted[seq]:accepted[seq]+2])
                expected_c[first, 2:2+length].copy_(q[ids, :8192])
                assert torch.equal(expected_c[first, length-1:length+2], cs[0])

            cb = (width + 2) * 8192 * 2
            sb = 32 * 128 * 128 * 4
            stride = ((cb + sb + 4095) // 4096 + 1) * 4096
            page = torch.full((nslots, stride), 90, dtype=torch.uint8, device='xpu')
            actual_c = page[:, :cb].view(torch.bfloat16).reshape(nslots, width+2, 8192)
            actual_s = page[:, cb:cb+sb].view(torch.float32).reshape(nslots, 32, 128, 128)
            assert actual_c.stride(0) > (width+2) * 8192
            assert actual_s.stride(0) > 32 * 128 * 128

            def spec():
                actual_c.copy_(c0)
                actual_s.copy_(s0)
                torch.ops.b70_gdn.decode_fused(core, output, q, ba, actual_c, actual_s,
                    w, a, dt, nw, cu, slot, has, token_idx, acc, 1e-6)

            spec()
            torch.xpu.synchronize()
            row = dict(width=width, lengths=list(lengths), seed=seed, step=step,
                accepted=list(accepted), core_exact=torch.equal(core, expected_core),
                out_exact=torch.equal(output, expected_out),
                state_exact=torch.equal(actual_s, expected_s),
                conv_exact=torch.equal(actual_c, expected_c),
                padding_exact=bool(torch.all(page[:, cb+sb:] == 90).item()))
            assert all(row[k] for k in ('core_exact', 'out_exact', 'state_exact', 'conv_exact', 'padding_exact')), row
            graph = torch.xpu.XPUGraph()
            with torch.xpu.graph(graph):
                spec()
            for _ in range(8):
                graph.replay()
            torch.xpu.synchronize()
            row['replay_exact'] = (torch.equal(core, expected_core)
                and torch.equal(output, expected_out) and torch.equal(actual_s, expected_s)
                and torch.equal(actual_c, expected_c)
                and bool(torch.all(page[:, cb+sb:] == 90).item()))
            assert row['replay_exact'], row
            del graph
            results.append(row)
            print(json.dumps(row), flush=True)
            c0, s0 = actual_c.clone(), actual_s.clone()
            # Commit different lengths, then verify another ragged step.
            accepted = [1 + (seed + i) % max(1, length) for i, length in enumerate(lengths)]
            lengths = [length % width + 1 if length else 0 for length in lengths]
(outdir / 'ragged.json').write_text(json.dumps(results, indent=2) + '\n')
print('X5_GDN_RAGGED_NORMAL_EXIT_PASS', len(results), flush=True)
