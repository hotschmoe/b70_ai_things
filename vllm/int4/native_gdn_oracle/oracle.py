#!/usr/bin/env python3
"""Full native XPU GDN contract oracle. GPU mode needs an owned outer lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path

NATIVE_SHA = '271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f'
IMAGE = 'sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-xpu', action='store_true')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--tp-size', type=int, choices=[1, 2], default=2)
    p.add_argument('--kv-layout', choices=['fp16', 'fp8'], default='fp8')
    p.add_argument('--accepted-pattern', default='1,2,3,4')
    p.add_argument('--output', type=Path)
    return p


def geometry(tp, kv):
    kh, vh, d = 16 // tp, 48 // tp, 128
    conv_dim = (2 * kh + vh) * d
    page = (832 * 4096 if kv == 'fp16' else 1600 * 2048) // tp
    return dict(kh=kh, vh=vh, dim=d, conv_dim=conv_dim, page=page,
                conv_bytes=6 * conv_dim * 2, ssm_bytes=vh * d * d * 4)


def make_parameters(torch, rng, g):
    weights = (torch.randn((g['conv_dim'], 4), generator=rng) * .2).half()
    alog = torch.full((g['vh'],), -1.0, dtype=torch.float32)
    bias = (torch.randn((g['vh'],), generator=rng) * .05).half()
    return weights, alog, bias


def cpu_reference(torch, F, qkv, ba, weights, alog, bias, conv, state, accepted, g):
    """Independent CPU convolution and delta rule; no backend/custom operators."""
    kh, vh, d = g['kh'], g['vh'], g['dim']
    qsz, vsz = kh * d, vh * d
    q, k, v, z = qkv.split([qsz, qsz, vsz, vsz], dim=-1)
    raw = torch.cat([q, k, v], dim=-1)
    window = conv[accepted - 1:accepted + 2].clone()
    # Fixed width4 chronological convolution with FP32 arithmetic, FP16 result.
    joined = torch.cat([window, raw], dim=0).float()
    conv_out = []
    for t in range(4):
        x = (joined[t:t + 4].T * weights.float()).sum(-1)
        conv_out.append(F.silu(x).half().float())
    packed = torch.stack(conv_out)
    q, k, v = packed.split([qsz, qsz, vsz], dim=-1)
    q = q.reshape(4, kh, d); k = k.reshape(4, kh, d)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / (d ** 0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(3, dim=1); k = k.repeat_interleave(3, dim=1)
    v = v.reshape(4, vh, d)
    b, a = ba.float().chunk(2, dim=-1)
    decay = torch.exp(-torch.exp(alog)[None, :] * F.softplus(a + bias))
    beta = torch.sigmoid(b)
    outputs, states = [], []
    state = state.float().clone()
    for t in range(4):
        state = state * decay[t, :, None, None]
        memory = torch.einsum('hvk,hk->hv', state, k[t])
        delta = (v[t] - memory) * beta[t, :, None]
        state = state + delta[:, :, None] * k[t, :, None, :]
        outputs.append(torch.einsum('hvk,hk->hv', state, q[t]).half())
        states.append(state.clone())
    new_conv = torch.cat([window[1:], raw], dim=0)
    return torch.stack(outputs), z.reshape(4, vh, d), new_conv, torch.stack(states)


def main():
    args = parser().parse_args()
    counts = [int(s) for s in args.accepted_pattern.split(',')]
    assert counts and all(1 <= n <= 4 for n in counts)
    assert 1 <= args.steps <= 256
    g = geometry(args.tp_size, args.kv_layout)
    if not args.run_xpu:
        print(json.dumps(dict(image=IMAGE, native_sha256=NATIVE_SHA, geometry=g,
                              steps=args.steps, accepted_pattern=counts,
                              mode='CPU plan only; --run-xpu requires outer selected-card lifecycle')))
        return 0
    assert args.output is not None and not args.output.exists(), 'fresh result required'
    import os
    assert os.environ.get('PYTORCH_ALLOC_CONF') == 'expandable_segments:True', 'production allocator required'
    assert Path('/dev/dri').exists(), 'GPU execution requires device-pinned lifecycle'
    assert os.environ.get('ZE_AFFINITY_MASK') in ('0', '1'), 'selected physical-card pin required'
    import torch
    import torch.nn.functional as F
    import vllm_xpu_kernels._xpu_C as native
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == NATIVE_SHA
    schema = str(torch.ops._xpu_C.gdn_attention.default._schema)
    assert 'num_accepted_tokens' in schema and 'num_spec_decodes' in schema
    torch.set_num_threads(2)
    torch.xpu.set_device(0)
    rng = torch.Generator(device='cpu').manual_seed(172335)
    def rand(shape, scale=.1):
        return (torch.randn(shape, generator=rng) * scale).half()
    device = 'xpu:0'; slots = 12; table_cpu = torch.tensor([[7, 3, 9, 2]], dtype=torch.int32)
    indices = table_cpu[0].tolist()
    weights, alog, bias = make_parameters(torch, rng, g)
    weights_x = weights.to(device); alog_x = alog.to(device); bias_x = bias.to(device)
    initial_conv = rand((slots, 6, g['conv_dim']))
    initial_ssm = rand((slots, g['vh'], 128, 128), .01).float()
    def allocate(padded):
        page = g['page'] if padded else g['conv_bytes'] + g['ssm_bytes']
        raw = torch.full((slots, page), 0x5a, dtype=torch.uint8, device=device)
        conv = raw[:, :g['conv_bytes']].view(torch.float16).view(slots, 6, g['conv_dim'])
        ssm = raw[:, g['conv_bytes']:g['conv_bytes'] + g['ssm_bytes']].view(torch.float32).view(slots, g['vh'], 128, 128)
        conv.copy_(initial_conv); ssm.copy_(initial_ssm)
        assert conv.stride(0) * 2 == ssm.stride(0) * 4 == page
        return raw, conv, ssm
    packed = allocate(False); hybrid = allocate(True)
    pointers = {name:storage[i].data_ptr() for name,storage,i in [('packed_conv',packed,1),('packed_ssm',packed,2),('hybrid_conv',hybrid,1),('hybrid_ssm',hybrid,2)]}
    print(json.dumps(dict(event='pointer_range',allocator=os.environ['PYTORCH_ALLOC_CONF'],pointers_hex={k:hex(v) for k,v in pointers.items()},int64_representable=all(0<=v<2**63 for v in pointers.values()))),flush=True)
    assert all(0<=v<2**63 for v in pointers.values())
    qs = torch.tensor([0, 4], dtype=torch.int32, device=device)
    token_ids = torch.arange(4, dtype=torch.int32, device=device)
    table = table_cpu.to(device)
    all_rows = []
    def error(a, b):
        a = a.float(); b = b.float(); diff = a - b
        return dict(max_abs=float(diff.abs().max()), relative_l2=float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(b).clamp_min(1e-8)))
    def invoke(storage, qkv, ba, acc):
        raw, conv, ssm = storage
        out = torch.full((4, g['vh'], 128), float('nan'), dtype=torch.float16, device=device)
        z = torch.full_like(out, float('nan'))
        torch.ops._xpu_C.gdn_attention(out, z, qkv, ba, 16, 48, 128, 128,
            conv, ssm, weights_x, None, 'silu', alog_x, bias_x,
            0, 0, 1, None, None, torch.empty(0, dtype=torch.int32, device=device), None,
            qs, token_ids, table, acc, 4, args.tp_size, True)
        return out.cpu(), z.cpu()
    # Reference tracks independent state across steps; additionally compare a
    # local one-step reference seeded from each native snapshot to localize drift.
    ref_conv = initial_conv.clone(); ref_ssm = initial_ssm.clone()
    for step in range(args.steps):
        accepted = counts[step % len(counts)]
        qkv_cpu = rand((4, (2 * g['kh'] + 2 * g['vh']) * 128))
        ba_cpu = rand((4, 2 * g['vh']))
        qkv_x = qkv_cpu.to(device); ba_x = ba_cpu.to(device)
        acc = torch.tensor([accepted], dtype=torch.int32, device=device)
        before_conv = hybrid[1].cpu(); before_ssm = hybrid[2].cpu(); before_raw = hybrid[0].cpu()
        ref = cpu_reference(torch, F, qkv_cpu, ba_cpu, weights, alog, bias,
                            ref_conv[indices[0]], ref_ssm[indices[accepted - 1]], accepted, g)
        local = cpu_reference(torch, F, qkv_cpu, ba_cpu, weights, alog, bias,
                              before_conv[indices[0]], before_ssm[indices[accepted - 1]], accepted, g)
        op, zp = invoke(packed, qkv_x, ba_x, acc)
        oh, zh = invoke(hybrid, qkv_x, ba_x, acc)
        after_conv = hybrid[1].cpu(); after_ssm = hybrid[2].cpu(); after_raw = hybrid[0].cpu()
        ref_conv[indices[0]] = ref[2]
        for j, idx in enumerate(indices): ref_ssm[idx] = ref[3][j]
        inactive_conv = [i for i in range(slots) if i != indices[0]]
        inactive_ssm = [i for i in range(slots) if i not in indices]
        active_states = after_ssm[indices]
        err_out = error(oh, ref[0]); err_state = error(active_states, ref[3])
        checks = dict(finite=bool(torch.isfinite(oh).all() and torch.isfinite(active_states).all()),
            z_exact=torch.equal(zh, ref[1]), conv_window_exact=torch.equal(after_conv[indices[0]], ref[2]),
            inactive_conv_exact=torch.equal(after_conv[inactive_conv], before_conv[inactive_conv]),
            inactive_ssm_exact=torch.equal(after_ssm[inactive_ssm], before_ssm[inactive_ssm]),
            page_padding_exact=torch.equal(after_raw[:, g['conv_bytes'] + g['ssm_bytes']:], before_raw[:, g['conv_bytes'] + g['ssm_bytes']:]),
            stride_out_exact=torch.equal(op, oh), stride_z_exact=torch.equal(zp, zh),
            stride_conv_exact=torch.equal(packed[1].cpu(), after_conv),
            stride_ssm_exact=torch.equal(packed[2].cpu(), after_ssm),
            output_math=err_out['max_abs'] <= .005 and err_out['relative_l2'] <= .03,
            state_math=err_state['max_abs'] <= .005 and err_state['relative_l2'] <= .03)
        row = dict(step=step, accepted=accepted, checks=checks, output_error=err_out,
                   state_error=err_state, local_output_error=error(oh, local[0]),
                   local_state_error=error(active_states, local[3]))
        all_rows.append(row)
        print(json.dumps(row), flush=True)
        if not all(checks.values()): break
    verdict = dict(image=IMAGE, native_sha256=NATIVE_SHA, schema=schema,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry=g, physical_state_columns=indices,
        state_strides_bytes=dict(packed_conv=packed[1].stride(0)*2,packed_ssm=packed[2].stride(0)*4,hybrid_conv=hybrid[1].stride(0)*2,hybrid_ssm=hybrid[2].stride(0)*4),
        requested_steps=args.steps, completed_steps=len(all_rows),
        accepted_pattern=counts, rows=all_rows,
        passed=len(all_rows) == args.steps and all(all(r['checks'].values()) for r in all_rows),
        scope='direct full native op only; no model/GEMM/TPcollective/scheduler/graph/cancellation coverage')
    args.output.write_text(json.dumps(verdict, indent=2) + '\n')
    return 0 if verdict['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
