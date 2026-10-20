"""Pack and unpack b70lb1 weights. See plugins/LOWBIT_FORMAT.md."""

from __future__ import annotations

import torch

GROUP = 128


def plane2_stride(k: int) -> int:
    if k % GROUP != 0:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    payload = k // 4
    return (payload + 63) & ~63


def plane_hi_stride(k: int, bits: int) -> int:
    if bits == 2:
        return 0
    if k % GROUP != 0:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    if bits == 3:
        payload = k // 8
    elif bits == 4:
        payload = k // 4
    else:
        raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
    return (payload + 63) & ~63


def quantize_rtn(
    weight: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric RTN along K. weight is [N, K] float.

    Returns int8 codes [N, K] and bf16 scales [N, K/128]. The positive
    bound is maxq-1, matching AutoRound quant_tensor_rtn_sym.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be [N, K]")
    n, k = weight.shape
    if k % GROUP != 0:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    if bits not in (2, 3, 4):
        raise ValueError(bits)
    maxq = 1 << (bits - 1)
    grouped = weight.float().reshape(n, k // GROUP, GROUP)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-8)
    scale = amax / maxq
    codes = (
        torch.round(grouped / scale.unsqueeze(-1))
        .clamp(-maxq, maxq - 1)
        .to(torch.int8)
    )
    return codes.reshape(n, k), scale.to(torch.bfloat16)


def pack_codes(
    codes: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pack int8 two's-complement codes [N, K] into plane2 and plane_hi."""
    if codes.dtype != torch.int8:
        raise ValueError("codes must be int8")
    n, k = codes.shape
    s2 = plane2_stride(k)
    plane2 = torch.zeros(n, s2, dtype=torch.uint8, device=codes.device)
    low = (codes.to(torch.int16) & 3).reshape(n, k // 4, 4)
    shifts2 = torch.tensor([0, 2, 4, 6], dtype=torch.int16, device=codes.device)
    packed2 = (low << shifts2).sum(dim=-1).to(torch.uint8)
    plane2[:, : k // 4] = packed2
    if bits == 2:
        return plane2, None
    shi = plane_hi_stride(k, bits)
    plane_hi = torch.zeros(n, shi, dtype=torch.uint8, device=codes.device)
    if bits == 3:
        hib = ((codes.to(torch.int16) >> 2) & 1).reshape(n, k // 8, 8)
        shifts1 = torch.arange(8, dtype=torch.int16, device=codes.device)
        packed_hi = (hib << shifts1).sum(dim=-1).to(torch.uint8)
        plane_hi[:, : k // 8] = packed_hi
    elif bits == 4:
        hib = ((codes.to(torch.int16) >> 2) & 3).reshape(n, k // 4, 4)
        packed_hi = (hib << shifts2).sum(dim=-1).to(torch.uint8)
        plane_hi[:, : k // 4] = packed_hi
    else:
        raise ValueError(bits)
    return plane2, plane_hi


def unpack_codes(
    plane2: torch.Tensor,
    plane_hi: torch.Tensor | None,
    bits: int,
    k: int,
) -> torch.Tensor:
    """Inverse of pack_codes. Returns int8 codes [N, K]."""
    n = plane2.shape[0]
    raw2 = plane2[:, : k // 4].to(torch.int16)
    parts = torch.stack(
        [
            raw2 & 3,
            (raw2 >> 2) & 3,
            (raw2 >> 4) & 3,
            (raw2 >> 6) & 3,
        ],
        dim=-1,
    ).reshape(n, k)
    if bits == 2:
        codes_u = parts
    elif bits == 3:
        if plane_hi is None:
            raise ValueError("int3 requires plane_hi")
        raw1 = plane_hi[:, : k // 8].to(torch.int16)
        hib = torch.stack([(raw1 >> b) & 1 for b in range(8)], dim=-1).reshape(n, k)
        codes_u = parts | (hib << 2)
    elif bits == 4:
        if plane_hi is None:
            raise ValueError("int4 requires plane_hi")
        rawh = plane_hi[:, : k // 4].to(torch.int16)
        hib = torch.stack(
            [(rawh >> 0) & 3, (rawh >> 2) & 3, (rawh >> 4) & 3, (rawh >> 6) & 3],
            dim=-1,
        ).reshape(n, k)
        codes_u = parts | (hib << 2)
    else:
        raise ValueError(bits)
    # Sign-extend the low `bits` of codes_u.
    shift = 8 - bits
    return ((codes_u << shift).to(torch.int8) >> shift)


def dequant(
    plane2: torch.Tensor,
    plane_hi: torch.Tensor | None,
    scale: torch.Tensor,
    bits: int,
    k: int,
) -> torch.Tensor:
    """Float dequant [N, K] using the bf16 scale rounded back to fp32."""
    codes = unpack_codes(plane2, plane_hi, bits, k).float()
    n = codes.shape[0]
    scale_f = scale.float().reshape(n, k // GROUP, 1)
    return (codes.reshape(n, k // GROUP, GROUP) * scale_f).reshape(n, k)


def quantize_and_pack(
    weight: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    codes, scale = quantize_rtn(weight, bits)
    plane2, plane_hi = pack_codes(codes, bits)
    return plane2, plane_hi, scale


def unpack_gptq_unsigned(qweight: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """AutoGPTQ little-endian bitstream to unsigned codes.

    qweight is int32 [K * bits / 32, N], as in plugins/LOWBIT_FORMAT.md.
    Returns int16 [N, K] in [0, 2**bits).
    """
    if bits not in (2, 3, 4):
        raise ValueError(bits)
    if k % GROUP != 0:
        raise ValueError(f"K={k} is not a multiple of {GROUP}")
    rows, _n = qweight.shape
    if rows * 32 != k * bits:
        raise ValueError(
            f"qweight rows {rows} * 32 != K {k} * bits {bits}"
        )
    u = qweight.to(torch.int64) & 0xFFFFFFFF
    idx = torch.arange(k, device=qweight.device)
    word = (idx * bits) // 32
    off = (idx * bits) % 32
    mask = (1 << bits) - 1
    w0 = u.index_select(0, word)
    unsigned = (w0 >> off.unsqueeze(1)) & mask
    straddle = off + bits > 32
    if bool(straddle.any()):
        nxt = (word + 1).clamp(max=rows - 1)
        w1 = u.index_select(0, nxt)
        nlow = (32 - off).unsqueeze(1)
        mixed = ((w0 >> off.unsqueeze(1)) | (w1 << nlow)) & mask
        unsigned = torch.where(straddle.unsqueeze(1), mixed, unsigned)
    return unsigned.transpose(0, 1).contiguous().to(torch.int16)


def gptq_to_b70lb1(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Repack B1 auto_gptq tensors into the kernel layout.

    qweight: int32 [K * bits / 32, N]
    scales: float [K / 128, N] (fp16 in the checkpoint, any float dtype here)
    Signed codes are unsigned - 2**(bits-1). That is maxq, not the qzeros lane.
    Returns plane2, plane_hi, scale bf16 [N, K/128].
    """
    maxq = 1 << (bits - 1)
    unsigned = unpack_gptq_unsigned(qweight, bits, k)
    codes = (unsigned - maxq).to(torch.int8)
    plane2, plane_hi = pack_codes(codes, bits)
    if scales.shape != (k // GROUP, qweight.shape[1]):
        raise ValueError(
            f"scales {tuple(scales.shape)} != {(k // GROUP, qweight.shape[1])}"
        )
    scale = scales.transpose(0, 1).contiguous().to(torch.bfloat16)
    return plane2, plane_hi, scale


def _signed_codes_batch(qweight: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    """Unsigned GPTQ bitstream to signed int8 codes [C, N, K].

    qweight is int32 [C, K * bits / 32, N]. int3 may straddle int32 words.
    int2 and int4 stay inside one word, so they skip the second gather.
    """
    if bits not in (2, 3, 4):
        raise ValueError(bits)
    c, rows, n = qweight.shape
    if rows * 32 != k * bits:
        raise ValueError(f"qweight rows {rows} * 32 != K {k} * bits {bits}")
    u = qweight.to(torch.int64) & 0xFFFFFFFF
    idx = torch.arange(k, device=qweight.device)
    word = (idx * bits) // 32
    off = (idx * bits) % 32
    mask = (1 << bits) - 1
    w0 = u[:, word, :]
    shift = off.view(1, k, 1)
    unsigned = (w0 >> shift) & mask
    if bits == 3:
        nxt = (word + 1).clamp(max=rows - 1)
        w1 = u[:, nxt, :]
        nlow = (32 - off).view(1, k, 1)
        mixed = ((w0 >> shift) | (w1 << nlow)) & mask
        straddle = (off + bits > 32).view(1, k, 1)
        unsigned = torch.where(straddle, mixed, unsigned)
    maxq = 1 << (bits - 1)
    signed = (unsigned - maxq).to(torch.int8)
    return signed.permute(0, 2, 1).contiguous()


def gptq_experts_to_b70lb1(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    k: int,
    chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Repack routed experts without flattening K across the expert axis.

    qweight: int32 [E, K * bits / 32, N]
    scales: float [E, K / 128, N]
    Returns plane2 [E, N, stride2], plane_hi or None, scale bf16 [E, N, K/128].
    chunk bounds the int64 unpack temporary. The default targets about 128MB.
    """
    if qweight.ndim != 3:
        raise ValueError(f"expert qweight rank {qweight.ndim} != 3")
    e, _rows, n = qweight.shape
    if scales.shape != (e, k // GROUP, n):
        raise ValueError(
            f"expert scales {tuple(scales.shape)} != {(e, k // GROUP, n)}"
        )
    s2 = plane2_stride(k)
    shi = plane_hi_stride(k, bits)
    plane2 = torch.empty(e, n, s2, dtype=torch.uint8, device=qweight.device)
    plane_hi = None
    if bits != 2:
        plane_hi = torch.empty(e, n, shi, dtype=torch.uint8, device=qweight.device)
    scale_out = torch.empty(
        e, n, k // GROUP, dtype=torch.bfloat16, device=qweight.device
    )
    bytes_per = max(k * n * 8, 1)
    step = chunk if chunk is not None else max(1, min(e, (128 << 20) // bytes_per))
    for start in range(0, e, step):
        end = min(e, start + step)
        codes = _signed_codes_batch(qweight[start:end], bits, k)
        packed2, packed_hi = pack_codes(codes.reshape(-1, k), bits)
        c = end - start
        plane2[start:end].copy_(packed2.view(c, n, s2))
        if plane_hi is not None:
            if packed_hi is None:
                raise RuntimeError("int3/int4 pack produced no plane_hi")
            plane_hi[start:end].copy_(packed_hi.view(c, n, shi))
        scale_out[start:end].copy_(
            scales[start:end].transpose(1, 2).contiguous().to(torch.bfloat16)
        )
    return plane2, plane_hi, scale_out


def gptq_experts_mixed_to_b70lb1(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    expert_bits: list[int],
    k: int,
    storage_bits: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Repack experts that do not share a bit width.

    qweight is int32 [E, K * storage_bits / 32, N]. Shorter experts occupy
    the leading rows; the tail is ignored. plane_hi is allocated at
    storage_bits and left zero for int2 experts. plane2 stride does not
    depend on bits.
    """
    if qweight.ndim != 3:
        raise ValueError(f"expert qweight rank {qweight.ndim} != 3")
    e, rows, n = qweight.shape
    if len(expert_bits) != e:
        raise ValueError(f"expert_bits {len(expert_bits)} != E {e}")
    expect = k * storage_bits // 32
    if rows != expect:
        raise ValueError(f"qweight rows {rows} != storage {expect}")
    if scales.shape != (e, k // GROUP, n):
        raise ValueError(
            f"expert scales {tuple(scales.shape)} != {(e, k // GROUP, n)}"
        )
    s2 = plane2_stride(k)
    plane2 = torch.zeros(e, n, s2, dtype=torch.uint8, device=qweight.device)
    plane_hi = None
    if storage_bits != 2:
        shi = plane_hi_stride(k, storage_bits)
        plane_hi = torch.zeros(e, n, shi, dtype=torch.uint8, device=qweight.device)
    scale_out = torch.empty(
        e, n, k // GROUP, dtype=torch.bfloat16, device=qweight.device
    )
    groups: dict[int, list[int]] = {}
    for index, bits in enumerate(expert_bits):
        if bits not in (2, 3, 4) or bits > storage_bits:
            raise ValueError(f"expert {index} bits {bits} storage {storage_bits}")
        groups.setdefault(int(bits), []).append(index)
    for bits, indexes in groups.items():
        need = k * bits // 32
        ix = torch.tensor(indexes, dtype=torch.long)
        packed2, packed_hi, packed_scale = gptq_experts_to_b70lb1(
            qweight.index_select(0, ix)[:, :need].contiguous(),
            scales.index_select(0, ix).contiguous(),
            bits,
            k,
        )
        plane2.index_copy_(0, ix, packed2)
        if packed_hi is not None:
            if plane_hi is None:
                raise RuntimeError("int2 storage cannot hold a hi plane")
            plane_hi.index_copy_(0, ix, packed_hi)
        scale_out.index_copy_(0, ix, packed_scale)
    return plane2, plane_hi, scale_out
