"""GPTQ / auto_round:auto_gptq packing for symmetric WNA16.

Layout (verified against the local Ornith GPTQ int4 checkpoint and against
intel/auto-round auto_round/utils/bit_packing.py):

  * Logical weight is HF Linear order [N, K] (out, in).
  * Groups run along K, group_size 128. Scale is fp16 [K/group, N].
  * Symmetric signed codes use AutoRound's grid
    maxq = 2 ** (bits - 1), codes in [-maxq, maxq - 1].
    int4 [-8, 7], int3 [-4, 3], int2 [-2, 1].
  * Stored unsigned code q = signed + maxq. Dequant is (q - maxq) * scale.
  * qweight is int32 [K * bits / 32, N]. Pack axis is K.
  * Bit stream is little-endian over blocks of 32 values (AutoRound
    pack_bitstream). For bits in {2, 4, 8} that is the familiar
    32/bits values per int32, lowest bits = lowest K index.
    For bits == 3, 32 values occupy 3 int32s and a value may straddle
    a word. K must be a multiple of 32.
  * qzeros is int32 [K/group, N * bits / 32], packed along N with the
    same bit stream. Symmetric AutoGPTQ stores (maxq - 1) in every lane
    because the historical CUDA kernel adds 1. The mathematical zero
    subtracted at dequant time is maxq, which is also what the XPU
    oneDNN symmetric int4 path uses (scalar 8 for int4). N must be a
    multiple of 32.
  * g_idx is int32 [K], g_idx[i] = i // group_size (desc_act false).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_AR = Path("/mnt/vm_8tb/b70/build/vllm-flashnext/auto-round")
if _AR.is_dir() and str(_AR) not in sys.path:
    sys.path.insert(0, str(_AR))

from auto_round.utils.bit_packing import pack_bitstream, packed_dim_size, unpack_bitstream  # noqa: E402

GROUP = 128
PACK_BLOCK = 32


def maxq_for(bits: int) -> int:
    return 1 << (bits - 1)


def stored_zero(bits: int) -> int:
    """Value packed into qzeros. Kernel that does zero+1 recovers maxq."""
    return maxq_for(bits) - 1


def quantize_sym_codes(
    weight_nk: torch.Tensor,
    scale_gn: torch.Tensor,
    bits: int,
    group_size: int = GROUP,
) -> torch.Tensor:
    """Signed codes [N, K] from weight [N, K] and scale [G, N]."""
    n, k = weight_nk.shape
    maxq = maxq_for(bits)
    scale = scale_gn.to(weight_nk.dtype).T.unsqueeze(-1)  # [N, G, 1]
    grouped = weight_nk.reshape(n, k // group_size, group_size)
    signed = torch.round(grouped / scale).clamp(-maxq, maxq - 1)
    return signed.reshape(n, k).to(torch.int8)


def pack_linear(
    signed_nk: torch.Tensor,
    scale_gn: torch.Tensor,
    bits: int,
    group_size: int = GROUP,
) -> dict[str, torch.Tensor]:
    """Pack one Linear. signed_nk [N, K] int, scale_gn [K/group, N] float."""
    n, k = signed_nk.shape
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    if k % PACK_BLOCK != 0 or n % PACK_BLOCK != 0:
        raise ValueError(f"K={k} and N={n} must both be multiples of {PACK_BLOCK}")
    maxq = maxq_for(bits)
    if int(signed_nk.min()) < -maxq or int(signed_nk.max()) > maxq - 1:
        raise ValueError(f"signed codes outside [{-maxq}, {maxq - 1}] for bits={bits}")
    unsigned = (signed_nk.to(torch.int32) + maxq).T.contiguous()  # [K, N]
    qweight = pack_bitstream(unsigned, bits, dim=0).contiguous()
    expect_rows = packed_dim_size(k, bits)
    if qweight.shape != (expect_rows, n):
        raise RuntimeError(f"qweight {tuple(qweight.shape)} != {(expect_rows, n)}")
    zeros = torch.full((k // group_size, n), stored_zero(bits), dtype=torch.int32)
    qzeros = pack_bitstream(zeros, bits, dim=1).contiguous()
    g_idx = torch.arange(k, dtype=torch.int32) // group_size
    scales = scale_gn.to(torch.float16).contiguous()
    if scales.shape != (k // group_size, n):
        raise RuntimeError(f"scales {tuple(scales.shape)} != {(k // group_size, n)}")
    return {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "g_idx": g_idx,
    }


def unpack_qweight(qweight: torch.Tensor, bits: int, k: int, n: int) -> torch.Tensor:
    """Unsigned codes [K, N]."""
    got = unpack_bitstream(qweight, bits, dim=0)
    if got.shape != (k, n):
        raise RuntimeError(f"unpacked {tuple(got.shape)} != {(k, n)}")
    return got


def dequant_linear(packed: dict[str, torch.Tensor], bits: int, group_size: int = GROUP) -> torch.Tensor:
    """Return bf16-ready float weight [N, K] using mathematical zero = maxq.

    Ignores the packed qzeros bias-1 lanes on purpose. Symmetric consumers
    subtract maxq (8 / 4 / 2), matching oneDNN's symmetric int4 scalar.
    """
    qweight = packed["qweight"]
    scales = packed["scales"]
    n = qweight.shape[1]
    k = int(scales.shape[0]) * group_size
    unsigned = unpack_qweight(qweight, bits, k, n).to(torch.float32)
    signed = unsigned - maxq_for(bits)
    scale = scales.to(torch.float32).T.repeat_interleave(group_size, dim=1)  # [N, K]
    return signed.T * scale


def _manual_pack_aligned(unsigned_kn: torch.Tensor, bits: int) -> torch.Tensor:
    """Reference pack for bits that divide 32. unsigned [K, N], low K first."""
    k, n = unsigned_kn.shape
    pf = 32 // bits
    words = torch.zeros(k // pf, n, dtype=torch.int32)
    u = unsigned_kn.to(torch.int32)
    for j in range(pf):
        words |= (u[j::pf] & ((1 << bits) - 1)) << (bits * j)
    return words


def self_check(gptq_root: str | None = None, bf16_shard: str | None = None) -> dict:
    """Roundtrip bits 2/3/4 and, if paths are given, byte-match a real int4 expert."""
    report: dict = {"roundtrip": {}, "real_int4": None}
    for bits in (2, 3, 4):
        gen = torch.Generator().manual_seed(bits)
        n, k = 64, 256
        maxq = maxq_for(bits)
        signed = torch.randint(-maxq, maxq, (n, k), generator=gen, dtype=torch.int16).to(torch.int8)
        scale = torch.rand(k // GROUP, n, generator=gen) * 0.02 + 1e-4
        packed = pack_linear(signed, scale, bits)
        unsigned = unpack_qweight(packed["qweight"], bits, k, n)
        back = (unsigned.to(torch.int32) - maxq).T.to(torch.int8)
        if not torch.equal(back, signed):
            raise AssertionError(f"roundtrip failed bits={bits}")
        if bits in (2, 4, 8):
            manual = _manual_pack_aligned((signed.to(torch.int32) + maxq).T.contiguous(), bits)
            if not torch.equal(manual, packed["qweight"]):
                raise AssertionError(f"aligned pack mismatch bits={bits}")
        rec = dequant_linear(packed, bits)
        # scales are stored fp16; compare against that rounding, not the fp32 source.
        scale16 = scale.to(torch.float16).float()
        grouped = signed.reshape(n, k // GROUP, GROUP).float() * scale16.T.unsqueeze(-1)
        ref = grouped.reshape(n, k)
        if not torch.allclose(rec, ref, rtol=0, atol=0):
            err = (rec - ref).abs().max().item()
            raise AssertionError(f"dequant mismatch bits={bits} max_abs={err}")
        report["roundtrip"][bits] = {
            "qweight": list(packed["qweight"].shape),
            "qzeros": list(packed["qzeros"].shape),
            "scales": list(packed["scales"].shape),
        }

    if gptq_root and bf16_shard:
        import json
        import os

        from safetensors import safe_open

        idx = json.load(open(os.path.join(gptq_root, "model.safetensors.index.json")))
        key = "model.language_model.layers.0.mlp.experts.0.down_proj"
        fn = idx["weight_map"][key + ".qweight"]
        with safe_open(os.path.join(gptq_root, fn), framework="pt") as f:
            qw = f.get_tensor(key + ".qweight")
            qz = f.get_tensor(key + ".qzeros")
            sc = f.get_tensor(key + ".scales")
        unsigned = unpack_qweight(qw, 4, 512, 2048)
        signed = (unsigned.to(torch.int32) - 8).T.to(torch.int8)
        repacked = pack_linear(signed, sc.float(), 4)
        if not torch.equal(repacked["qweight"], qw):
            raise AssertionError("repack of real int4 qweight is not byte-identical")
        if not torch.equal(repacked["qzeros"], qz):
            raise AssertionError("repack of real int4 qzeros is not byte-identical")
        report["real_int4"] = {
            "qweight_match": True,
            "qzeros_match": True,
            "signed_minmax": [int(signed.min()), int(signed.max())],
        }
    return report
