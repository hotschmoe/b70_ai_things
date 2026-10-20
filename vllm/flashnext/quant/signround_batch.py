"""Batched SignRound on fused expert tensors.

Uses Intel AutoRound's own pieces, not a reimplementation of the grid:

  * auto_round.data_type.int.quant_tensor_sym  (symmetric int QDQ + STE)
  * auto_round...sign_sgd.SignSGD              (param -= lr * sign(grad))
  * SignRoundConfig lr = 1/iters, LinearLR down to 0
  * WrapperLinear minmax clamp to (0, 1), best-MSE restore

The fused Qwen3.5-MoE expert parameter is one 3D tensor, not an nn.Linear,
so AutoRound's block walker would either skip it or run a Python loop over
256 experts. This module batches that same QDQ + SignSGD update.

Loss for a routed expert is the expert-block output (down(silu(gate)*up)),
which is the block MSE AutoRound minimizes, restricted to that expert.
Attention projections use their own linear-output MSE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_AR = Path("/mnt/vm_8tb/b70/build/vllm-flashnext/auto-round")
if _AR.is_dir() and str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))

from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD  # noqa: E402
from auto_round.data_type.int import quant_tensor_sym  # noqa: E402

from .pack_gptq import GROUP, maxq_for  # noqa: E402


def _flatten_rows(weight: torch.Tensor) -> torch.Tensor:
    # Detach: only v / min_scale / max_scale are trained. A grad-connected
    # weight (a Parameter view, or an index of one) makes the second step
    # backward through an already-freed node.
    return weight.detach().reshape(-1, weight.shape[-1]).float()


def _codes_and_scale(qdq: torch.Tensor, scale: torch.Tensor, bits: int, group_size: int):
    """qdq is the original weight shape. scale is [n_rows, 1] from quant_tensor_sym."""
    n_rows_k = qdq.reshape(-1, group_size)
    signed = torch.round(n_rows_k.float() / scale.float()).clamp(-maxq_for(bits), maxq_for(bits) - 1)
    signed = signed.reshape(qdq.shape).to(torch.int8)
    n = qdq.shape[-2] if qdq.ndim >= 2 else qdq.shape[0]
    # scale rows follow the flattened [leading * N, K/group] order of a [..., N, K] tensor
    # when the last dim is K. For [N, K], n_rows = N * groups, view as [N, groups].
    groups = qdq.shape[-1] // group_size
    leading = signed.numel() // (groups * qdq.shape[-1])
    # signed shape is original. Scale view: quant_tensor_sym flattens to [-1, group].
    scale_nk = scale.view(-1, groups).to(torch.float16)  # [rows_of_N, groups] where rows_of_N = N if 2D
    del n, leading
    return signed, scale_nk


def signround_linear(
    weight_nk: torch.Tensor,
    inputs_tk: torch.Tensor,
    bits: int,
    iters: int,
    group_size: int = GROUP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """SignRound one Linear.

    weight_nk: [N, K] on device. inputs_tk: [T, K] on device.
    Returns qdq [N, K] float32, signed codes [N, K] int8, scale [groups, N] fp16,
    final best loss, and the RTN (v=0) loss on the same inputs.
    """
    if iters < 1:
        raise ValueError("SignRound iters must be >= 1 (RTN is iters=0 and is not this path)")
    device = weight_nk.device
    flat = _flatten_rows(weight_nk)
    n_rows, k = flat.shape
    groups = k // group_size
    v = torch.nn.Parameter(torch.zeros(n_rows * groups, group_size, device=device))
    min_scale = torch.nn.Parameter(torch.ones(n_rows * groups, device=device))
    max_scale = torch.nn.Parameter(torch.ones(n_rows * groups, device=device))
    lr = 1.0 / iters
    opt = SignSGD(
        [
            {"params": [v], "lr": lr},
            {"params": [min_scale, max_scale], "lr": lr},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=iters)
    target = F.linear(inputs_tk, weight_nk.float()).detach()
    best = None
    best_loss = float("inf")
    rtn_loss = None
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        qdq, scale, _maxq = quant_tensor_sym(
            flat,
            bits=bits,
            group_size=group_size,
            v=v,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=torch.float32,
        )
        pred = F.linear(inputs_tk, qdq.view_as(weight_nk))
        loss = F.mse_loss(pred, target)
        if step == 0:
            rtn_loss = float(loss.detach())
        # Save the parameters that produced this loss, before the step moves them.
        val = float(loss.detach())
        if val < best_loss:
            best_loss = val
            best = (v.detach().clone(), min_scale.detach().clone(), max_scale.detach().clone())
        loss.backward()
        opt.step()
        sched.step()
        min_scale.data.clamp_(0.0, 1.0)
        max_scale.data.clamp_(0.0, 1.0)
    assert best is not None and rtn_loss is not None
    v.data.copy_(best[0])
    min_scale.data.copy_(best[1])
    max_scale.data.copy_(best[2])
    with torch.no_grad():
        qdq, scale, _maxq = quant_tensor_sym(
            flat,
            bits=bits,
            group_size=group_size,
            v=v,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=torch.float32,
        )
    qdq = qdq.view_as(weight_nk).detach()
    signed, scale_ng = _codes_and_scale(qdq, scale, bits, group_size)
    scale_gn = scale_ng.T.contiguous()
    return qdq, signed, scale_gn, best_loss, rtn_loss


def signround_mlp(
    gate_nk: torch.Tensor,
    up_nk: torch.Tensor,
    down_nk: torch.Tensor,
    inputs_tk: torch.Tensor,
    bits: int,
    iters: int,
    group_size: int = GROUP,
) -> dict:
    """Joint SignRound of gate, up, down. Loss is the MLP output MSE.

    Each projection keeps its own v / min_scale / max_scale. bits is one
    width for all three (a shared expert, or one routed expert).
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    device = gate_nk.device
    x = inputs_tk
    params = []
    state = []
    for name, w in (("gate", gate_nk), ("up", up_nk), ("down", down_nk)):
        flat = _flatten_rows(w)
        n_rows, k = flat.shape
        groups = k // group_size
        v = torch.nn.Parameter(torch.zeros(n_rows * groups, group_size, device=device))
        min_scale = torch.nn.Parameter(torch.ones(n_rows * groups, device=device))
        max_scale = torch.nn.Parameter(torch.ones(n_rows * groups, device=device))
        state.append((name, w, flat, v, min_scale, max_scale))
        params.append(v)
        params.append(min_scale)
        params.append(max_scale)
    lr = 1.0 / iters
    opt = SignSGD([{"params": params, "lr": lr}], lr=lr)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=iters)

    def _qdq(flat, v, min_scale, max_scale, bits_):
        qdq, scale, _mq = quant_tensor_sym(
            flat,
            bits=bits_,
            group_size=group_size,
            v=v,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=torch.float32,
        )
        return qdq, scale

    with torch.no_grad():
        g = F.linear(x, gate_nk.float())
        u = F.linear(x, up_nk.float())
        target = F.linear(F.silu(g) * u, down_nk.float())

    best = None
    best_loss = float("inf")
    rtn_loss = None
    for step in range(iters):
        opt.zero_grad(set_to_none=True)
        qdqs = []
        for name, w, flat, v, min_scale, max_scale in state:
            qdq, scale = _qdq(flat, v, min_scale, max_scale, bits)
            qdqs.append(qdq.view_as(w))
        g = F.linear(x, qdqs[0])
        u = F.linear(x, qdqs[1])
        pred = F.linear(F.silu(g) * u, qdqs[2])
        loss = F.mse_loss(pred, target)
        if step == 0:
            rtn_loss = float(loss.detach())
        val = float(loss.detach())
        if val < best_loss:
            best_loss = val
            best = [(s[3].detach().clone(), s[4].detach().clone(), s[5].detach().clone()) for s in state]
        loss.backward()
        opt.step()
        sched.step()
        for _name, _w, _flat, _v, min_scale, max_scale in state:
            min_scale.data.clamp_(0.0, 1.0)
            max_scale.data.clamp_(0.0, 1.0)
    assert best is not None and rtn_loss is not None
    out = {}
    with torch.no_grad():
        for (name, w, flat, v, min_scale, max_scale), saved in zip(state, best):
            v.data.copy_(saved[0])
            min_scale.data.copy_(saved[1])
            max_scale.data.copy_(saved[2])
            qdq, scale = _qdq(flat, v, min_scale, max_scale, bits)
            qdq = qdq.view_as(w).detach()
            signed, scale_ng = _codes_and_scale(qdq, scale, bits, group_size)
            out[name] = {
                "qdq": qdq,
                "signed": signed,
                "scale_gn": scale_ng.T.contiguous(),
            }
    out["best_loss"] = best_loss
    out["rtn_loss"] = rtn_loss
    return out


def _subsample_rows(x: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if x.shape[0] <= max_tokens:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, max_tokens).long()
    return x[idx]


def _rtn_codes(weight_enk: torch.Tensor, bits: int, group_size: int):
    """v = 0 symmetric codes. weight [E, N, K] -> signed [E, N, K], scale [E, G, N], qdq [E, N, K]."""
    e, n, k = weight_enk.shape
    groups = k // group_size
    flat = weight_enk.reshape(e * n, k).float()
    with torch.no_grad():
        qdq, scale, _mq = quant_tensor_sym(
            flat, bits=bits, group_size=group_size, v=0, scale_dtype=torch.float32
        )
    qdq = qdq.view(e, n, k)
    signed = torch.round(qdq.reshape(-1, group_size).float() / scale.float()).clamp(
        -maxq_for(bits), maxq_for(bits) - 1
    )
    signed = signed.reshape(e, n, k).to(torch.int8)
    scale_egn = scale.reshape(e, n, groups).to(torch.float16).permute(0, 2, 1).contiguous()
    return qdq, signed, scale_egn


def signround_experts_batched(
    gate_up: torch.Tensor,
    down: torch.Tensor,
    token_list: list[torch.Tensor],
    bits: int,
    iters: int,
    group_size: int = GROUP,
    max_tokens: int = 256,
    expert_batch: int = 32,
) -> dict:
    """SignRound routed experts in batches. gate_up [E, 2I, H], down [E, H, I].

    Experts that share one bit width are independent, so chunking the batch
    does not change the update. An expert with zero calibration tokens is
    left at RTN (v = 0) and counted in n_rtn_empty. It is not run for one
    SignSGD step.

    qdq_* are bf16 on CPU (the dtype the next block forward will store).
    signed_* are int8 [E, N, K]. scale_* are fp16 [E, groups, N].
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    e, two_i, h = gate_up.shape
    inter = two_i // 2
    device = gate_up.device
    qdq_gu = torch.empty(e, two_i, h, dtype=torch.bfloat16)
    qdq_dn = torch.empty(e, h, inter, dtype=torch.bfloat16)
    signed_g = torch.empty(e, inter, h, dtype=torch.int8)
    signed_u = torch.empty(e, inter, h, dtype=torch.int8)
    signed_d = torch.empty(e, h, inter, dtype=torch.int8)
    groups_h = h // group_size
    groups_i = inter // group_size
    scale_g = torch.empty(e, groups_h, inter, dtype=torch.float16)
    scale_u = torch.empty(e, groups_h, inter, dtype=torch.float16)
    scale_d = torch.empty(e, groups_i, h, dtype=torch.float16)
    n_rtn = 0
    n_tuned = 0
    loss_acc = 0.0
    rtn_acc = 0.0
    tok_acc = 0
    sq_err = 0.0
    sq_w = 0.0

    for start in range(0, e, expert_batch):
        end = min(e, start + expert_batch)
        chunk_tokens = token_list[start:end]
        counts = []
        for t in chunk_tokens:
            if t is None or t.shape[0] == 0:
                counts.append(0)
            else:
                counts.append(int(min(t.shape[0], max_tokens)))
        n_rtn += sum(c == 0 for c in counts)
        n_tuned += sum(c > 0 for c in counts)
        gu = gate_up[start:end]
        dn = down[start:end]
        if not any(counts):
            qg, sg, scg = _rtn_codes(gu[:, :inter], bits, group_size)
            qu, su, scu = _rtn_codes(gu[:, inter:], bits, group_size)
            qd, sd, scd = _rtn_codes(dn, bits, group_size)
            qdq_gu[start:end, :inter] = qg.to(torch.bfloat16).cpu()
            qdq_gu[start:end, inter:] = qu.to(torch.bfloat16).cpu()
            qdq_dn[start:end] = qd.to(torch.bfloat16).cpu()
            signed_g[start:end] = sg.cpu()
            signed_u[start:end] = su.cpu()
            signed_d[start:end] = sd.cpu()
            scale_g[start:end] = scg.cpu()
            scale_u[start:end] = scu.cpu()
            scale_d[start:end] = scd.cpu()
            sq_err += float(
                (qg.float() - gu[:, :inter].float()).pow(2).sum()
                + (qu.float() - gu[:, inter:].float()).pow(2).sum()
                + (qd.float() - dn.float()).pow(2).sum()
            )
            sq_w += float(gu.float().pow(2).sum() + dn.float().pow(2).sum())
            continue

        tmax = max(counts)
        ee = end - start
        x = torch.zeros(ee, tmax, h, device=device, dtype=torch.float32)
        mask = torch.zeros(ee, tmax, device=device, dtype=torch.float32)
        for i, t in enumerate(chunk_tokens):
            c = counts[i]
            if c == 0:
                continue
            rows = _subsample_rows(t.detach(), max_tokens).to(device=device, dtype=torch.float32)
            x[i, :c] = rows
            mask[i, :c] = 1.0

        gate = gu[:, :inter].detach().float()
        up = gu[:, inter:].detach().float()
        dn_f = dn.detach().float()

        def _params(weight: torch.Tensor):
            ee_, n, k = weight.shape
            groups = k // group_size
            v = torch.nn.Parameter(torch.zeros(ee_ * n * groups, group_size, device=device))
            min_scale = torch.nn.Parameter(torch.ones(ee_ * n * groups, device=device))
            max_scale = torch.nn.Parameter(torch.ones(ee_ * n * groups, device=device))
            return v, min_scale, max_scale

        pg, pu, pd = _params(gate), _params(up), _params(dn_f)
        opt = SignSGD([{"params": [p for triple in (pg, pu, pd) for p in triple], "lr": 1.0 / iters}], lr=1.0 / iters)
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=iters)

        def _qdq(weight, packed):
            ee_, n, k = weight.shape
            flat = weight.reshape(ee_ * n, k)
            qdq, scale, _mq = quant_tensor_sym(
                flat,
                bits=bits,
                group_size=group_size,
                v=packed[0],
                min_scale=packed[1],
                max_scale=packed[2],
                scale_dtype=torch.float32,
            )
            return qdq.view(ee_, n, k), scale

        with torch.no_grad():
            g0 = torch.bmm(x, gate.transpose(1, 2))
            u0 = torch.bmm(x, up.transpose(1, 2))
            target = torch.bmm(F.silu(g0) * u0, dn_f.transpose(1, 2))

        best = None
        best_loss = float("inf")
        rtn_loss = None
        denom = (mask.sum() * h).clamp(min=1.0)
        for step in range(iters):
            opt.zero_grad(set_to_none=True)
            qg, _sg = _qdq(gate, pg)
            qu, _su = _qdq(up, pu)
            qd, _sd = _qdq(dn_f, pd)
            g = torch.bmm(x, qg.transpose(1, 2))
            u = torch.bmm(x, qu.transpose(1, 2))
            pred = torch.bmm(F.silu(g) * u, qd.transpose(1, 2))
            diff = (pred - target) * mask.unsqueeze(-1)
            loss = diff.square().sum() / denom
            if step == 0:
                rtn_loss = float(loss.detach())
            val = float(loss.detach())
            if val < best_loss:
                best_loss = val
                best = tuple(tuple(p.detach().clone() for p in triple) for triple in (pg, pu, pd))
            loss.backward()
            opt.step()
            sched.step()
            for triple in (pg, pu, pd):
                triple[1].data.clamp_(0.0, 1.0)
                triple[2].data.clamp_(0.0, 1.0)
        assert best is not None and rtn_loss is not None
        for triple, saved in zip((pg, pu, pd), best):
            for p, s in zip(triple, saved):
                p.data.copy_(s)
        with torch.no_grad():
            qg, sg = _qdq(gate, pg)
            qu, su = _qdq(up, pu)
            qd, sd = _qdq(dn_f, pd)
            signed_g[start:end] = _signed_from(qg, sg, bits, group_size).cpu()
            signed_u[start:end] = _signed_from(qu, su, bits, group_size).cpu()
            signed_d[start:end] = _signed_from(qd, sd, bits, group_size).cpu()
            scale_g[start:end] = _scale_egn(sg, end - start, inter, groups_h).cpu()
            scale_u[start:end] = _scale_egn(su, end - start, inter, groups_h).cpu()
            scale_d[start:end] = _scale_egn(sd, end - start, h, groups_i).cpu()
            qdq_gu[start:end, :inter] = qg.to(torch.bfloat16).cpu()
            qdq_gu[start:end, inter:] = qu.to(torch.bfloat16).cpu()
            qdq_dn[start:end] = qd.to(torch.bfloat16).cpu()
            sq_err += float((qg - gate).pow(2).sum() + (qu - up).pow(2).sum() + (qd - dn_f).pow(2).sum())
            sq_w += float(gate.pow(2).sum() + up.pow(2).sum() + dn_f.pow(2).sum())
        tok = int(sum(counts))
        if tok:
            loss_acc += best_loss * tok
            rtn_acc += rtn_loss * tok
            tok_acc += tok
        del pg, pu, pd, opt, x, mask, gate, up, dn_f, best

    rel = (sq_err / sq_w) ** 0.5 if sq_w else None
    return {
        "qdq_gate_up": qdq_gu,
        "qdq_down": qdq_dn,
        "signed_gate": signed_g,
        "signed_up": signed_u,
        "signed_down": signed_d,
        "scale_gate": scale_g,
        "scale_up": scale_u,
        "scale_down": scale_d,
        "n_tuned": n_tuned,
        "n_rtn_empty": n_rtn,
        "mean_best_loss": (loss_acc / tok_acc) if tok_acc else None,
        "mean_rtn_loss": (rtn_acc / tok_acc) if tok_acc else None,
        "rel_l2": rel,
    }


def _signed_from(qdq: torch.Tensor, scale: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    e, n, k = qdq.shape
    signed = torch.round(qdq.reshape(-1, group_size).float() / scale.float()).clamp(
        -maxq_for(bits), maxq_for(bits) - 1
    )
    return signed.reshape(e, n, k).to(torch.int8)


def _scale_egn(scale: torch.Tensor, e: int, n: int, groups: int) -> torch.Tensor:
    # quant_tensor_sym scale is [E*N*groups, 1] in row-major group order.
    return scale.reshape(e, n, groups).to(torch.float16).permute(0, 2, 1).contiguous()


# Back-compat name used by earlier notes.
signround_expert_stack = signround_experts_batched
