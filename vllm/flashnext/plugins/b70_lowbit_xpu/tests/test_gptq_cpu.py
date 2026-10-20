"""CPU checks for the GPTQ -> b70lb1 repack. No GPU, no lease.

CONFIG: Ornith expert-0 down_proj from the int4 mixedcal-v2 checkpoint,
plus pack_gptq synthetic int2/int3/int4 and a 2-expert fused w13 stack.
COMMAND: venv python tests/test_gptq_cpu.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FLASHNEXT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FLASHNEXT))

from b70_lowbit_xpu.format import (  # noqa: E402
    gptq_experts_to_b70lb1,
    gptq_to_b70lb1,
    unpack_codes,
)
from b70_lowbit_xpu.gptq_import import import_pack_gptq  # noqa: E402

dequant_linear, maxq_for, pack_linear, quantize_sym_codes, unpack_qweight = (
    import_pack_gptq()
)

ORNITH = (
    Path("/mnt/vm_8tb/github/b70_ai_things/models/files")
    / "ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2"
)
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS {name} {detail}", flush=True)
    else:
        FAILS.append(name)
        print(f"  FAIL {name} {detail}", flush=True)


def synthetic(bits: int) -> None:
    gen = torch.Generator().manual_seed(100 + bits)
    n, k = 64, 256
    maxq = maxq_for(bits)
    signed = torch.randint(-maxq, maxq, (n, k), generator=gen, dtype=torch.int16)
    signed = signed.to(torch.int8)
    scale = (torch.rand(k // 128, n, generator=gen) * 0.02 + 1e-4).to(torch.float16)
    packed = pack_linear(signed, scale, bits)
    plane2, plane_hi, scale_b = gptq_to_b70lb1(packed["qweight"], packed["scales"], bits, k)
    codes = unpack_codes(plane2, plane_hi, bits, k)
    check(f"synthetic b{bits} codes", torch.equal(codes, signed), f"shape={tuple(codes.shape)}")
    # bf16 scale is the stored kernel scale. Compare codes, not the float product.
    check(
        f"synthetic b{bits} scale",
        torch.equal(scale_b, scale.transpose(0, 1).contiguous().to(torch.bfloat16)),
    )


def experts_match_linear(bits: int) -> None:
    gen = torch.Generator().manual_seed(200 + bits)
    e, n, k = 3, 64, 256
    maxq = maxq_for(bits)
    planes = []
    his = []
    scales = []
    qws = []
    scs = []
    for _ in range(e):
        signed = torch.randint(-maxq, maxq, (n, k), generator=gen, dtype=torch.int16).to(
            torch.int8
        )
        scale = (torch.rand(k // 128, n, generator=gen) * 0.02 + 1e-4).to(torch.float16)
        packed = pack_linear(signed, scale, bits)
        p2, hi, sc = gptq_to_b70lb1(packed["qweight"], packed["scales"], bits, k)
        planes.append(p2)
        his.append(hi)
        scales.append(sc)
        qws.append(packed["qweight"])
        scs.append(packed["scales"])
    qw = torch.stack(qws, 0)
    sc = torch.stack(scs, 0)
    p2, hi, scale = gptq_experts_to_b70lb1(qw, sc, bits, k, chunk=2)
    check(f"experts b{bits} plane2", torch.equal(p2, torch.stack(planes, 0)))
    if bits == 2:
        check(f"experts b{bits} plane_hi", hi is None and his[0] is None)
    else:
        check(f"experts b{bits} plane_hi", torch.equal(hi, torch.stack(his, 0)))
    check(f"experts b{bits} scale", torch.equal(scale, torch.stack(scales, 0)))


def fused_w13(bits: int) -> None:
    """Gate then up along N, the RoutedExperts w13 layout at TP=1."""
    gen = torch.Generator().manual_seed(300 + bits)
    k, i = 256, 64
    maxq = maxq_for(bits)

    def one(n: int):
        signed = torch.randint(-maxq, maxq, (n, k), generator=gen, dtype=torch.int16).to(
            torch.int8
        )
        scale = (torch.rand(k // 128, n, generator=gen) * 0.02 + 1e-4).to(torch.float16)
        return pack_linear(signed, scale, bits)

    gate = one(i)
    up = one(i)
    qw = torch.cat([gate["qweight"], up["qweight"]], dim=1).unsqueeze(0)
    sc = torch.cat([gate["scales"], up["scales"]], dim=1).unsqueeze(0)
    p2, hi, scale = gptq_experts_to_b70lb1(qw, sc, bits, k)
    g2, ghi, gsc = gptq_to_b70lb1(gate["qweight"], gate["scales"], bits, k)
    u2, uhi, usc = gptq_to_b70lb1(up["qweight"], up["scales"], bits, k)
    check(f"w13 b{bits} plane2", torch.equal(p2[0], torch.cat([g2, u2], 0)))
    if bits == 2:
        check(f"w13 b{bits} hi", hi is None)
    else:
        check(f"w13 b{bits} hi", torch.equal(hi[0], torch.cat([ghi, uhi], 0)))
    check(f"w13 b{bits} scale", torch.equal(scale[0], torch.cat([gsc, usc], 0)))


def ornith_down() -> None:
    from safetensors import safe_open

    index = json.loads((ORNITH / "model.safetensors.index.json").read_text())
    key_w = "model.language_model.layers.0.mlp.experts.0.down_proj.qweight"
    key_s = "model.language_model.layers.0.mlp.experts.0.down_proj.scales"
    shard = ORNITH / index["weight_map"][key_w]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        qweight = handle.get_tensor(key_w)
        scales = handle.get_tensor(key_s)
    k = int(scales.shape[0]) * 128
    ref = dequant_linear({"qweight": qweight, "scales": scales}, 4)
    plane2, plane_hi, scale_b = gptq_to_b70lb1(qweight, scales, 4, k)
    codes = unpack_codes(plane2, plane_hi, 4, k).float()
    got = (
        codes.reshape(codes.shape[0], k // 128, 128)
        * scale_b.float().unsqueeze(-1)
    ).reshape_as(ref)
    # bf16 scale rounding is the only gap versus fp16 dequant.
    rel = (got - ref).norm() / ref.norm()
    check(
        "ornith down codes vs dequant",
        float(rel) < 1e-2,
        f"rel_l2={float(rel):.6f} qweight={tuple(qweight.shape)} K={k}",
    )
    # Exact integer codes: unsigned - 8, compared through pack_gptq unpack.
    unsigned = unpack_qweight(qweight, 4, k, qweight.shape[1])
    signed = (unsigned.to(torch.int32) - 8).T.to(torch.int8)
    check("ornith down exact codes", torch.equal(unpack_codes(plane2, plane_hi, 4, k), signed))


def config_rules() -> None:
    from b70_lowbit_xpu.quant_config import B70LowbitConfig

    ornith = json.loads((ORNITH / "config.json").read_text())["quantization_config"]
    cfg = B70LowbitConfig.from_config(ornith)
    check("ornith pack", cfg.pack == "gptq", cfg.pack)
    prefix = "language_model.model.layers.0"
    check("ornith attn skip", cfg._skipped(prefix + ".self_attn.q_proj"))
    check("ornith linear_attn skip", cfg._skipped(prefix + ".linear_attn.in_proj_qkvz"))
    check("ornith router skip", cfg._skipped(prefix + ".mlp.gate"))
    check("ornith shared skip", cfg._skipped(prefix + ".mlp.shared_expert.down_proj"))
    check("ornith shared_gate skip", cfg._skipped(prefix + ".mlp.shared_expert_gate"))
    check("ornith lm_head skip", cfg._skipped("language_model.lm_head"))
    experts = prefix + ".mlp.experts"
    check("ornith experts kept", not cfg._skipped(experts))
    check("ornith experts bits", cfg._bits_for(experts) == 4, str(cfg._bits_for(experts)))
    # B2 bench contract still holds on a b70lb1 dict.
    bench = B70LowbitConfig.from_config(
        {
            "bits": 3,
            "group_size": 128,
            "sym": True,
            "pack": "b70lb1",
            "modules_to_not_convert": ["lm_head"],
            "extra_bits": {"experts": 2},
        }
    )
    check("bench experts bits", bench._bits_for("model.layers.0.mlp.experts") == 2)
    check("bench q bits", bench._bits_for("model.layers.0.self_attn.q_proj") == 3)
    check("bench lm_head", bench._skipped("lm_head"))
    check("bench gate_proj", not bench._skipped("model.layers.0.mlp.experts.gate_proj"))
    gate = B70LowbitConfig.from_config(
        {
            "bits": 3,
            "group_size": 128,
            "sym": True,
            "pack": "b70lb1",
            "modules_to_not_convert": ["gate", "lm_head"],
        }
    )
    check("bench gate", gate._skipped("model.layers.0.mlp.gate"))
    check("bench gate_proj kept", not gate._skipped("model.layers.0.mlp.experts.0.gate_proj"))

    b1_path = Path(
        "/mnt/cache/b70/devmodel/quant/qwen36-w3a16-g128-experts-w4a16-trunk/quantization_config.json"
    )
    if b1_path.is_file():
        b1 = B70LowbitConfig.from_config(json.loads(b1_path.read_text()))
        check("b1 pack", b1.pack == "gptq")
        check("b1 expert bits", b1._bits_for(experts) == 3, str(b1._bits_for(experts)))
        check("b1 q bits", b1._bits_for(prefix + ".self_attn.q_proj") == 4)
        check("b1 router skip", b1._skipped(prefix + ".mlp.gate"))
        check("b1 shared bits", b1._bits_for(prefix + ".mlp.shared_expert.down_proj") == 4)
        check("b1 shared_gate skip", b1._skipped(prefix + ".mlp.shared_expert_gate"))
        # gate_proj must not inherit the router skip.
        check(
            "b1 gate_proj not router",
            not b1._skipped(prefix + ".mlp.shared_expert.gate_proj"),
        )


def main() -> int:
    print("CONFIG cpu gptq repack", flush=True)
    for bits in (2, 3, 4):
        synthetic(bits)
        experts_match_linear(bits)
        fused_w13(bits)
    ornith_down()
    config_rules()
    if FAILS:
        print("VERDICT fail", FAILS, flush=True)
        return 1
    print("VERDICT pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
