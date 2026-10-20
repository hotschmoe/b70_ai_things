"""CPU checks for mixed int3/int2 expert bits. No GPU, no lease.

CONFIG: qwen36 mixed checkpoint extra_config plus expert_bits.json,
and a synthetic GPTQ stack whose int2 rows sit in an int3-sized buffer
with a garbage tail.
COMMAND: image python tests/test_mixed_cpu.py
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
    gptq_experts_mixed_to_b70lb1,
    gptq_to_b70lb1,
)
from b70_lowbit_xpu.gptq_import import import_pack_gptq  # noqa: E402
from b70_lowbit_xpu.quant_config import B70LowbitConfig  # noqa: E402

_dequant, maxq_for, pack_linear, _quantize, _unpack = import_pack_gptq()

MIXED = Path(
    "/mnt/cache/b70/devmodel/quant/qwen36-mixed-int3-int2-experts-w4a16-trunk"
)
BITS_JSON = Path("/mnt/cache/b70/devmodel/quant/_work/qwen36/expert_bits.json")
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  PASS {name} {detail}", flush=True)
    else:
        FAILS.append(name)
        print(f"  FAIL {name} {detail}", flush=True)


def bit_table() -> None:
    table = json.loads(BITS_JSON.read_text())
    cfg = B70LowbitConfig.from_config(
        json.loads((MIXED / "config.json").read_text())["quantization_config"]
    )
    check("mixed pack", cfg.pack == "gptq", cfg.pack)
    check("mixed default bits", cfg.bits == 4, str(cfg.bits))
    hist = {2: 0, 3: 0}
    uniform_layers = 0
    # The live module prefix inserts .model. before layers.
    vllm_prefix = "model.language_model.model.layers.{}.mlp.experts"
    ckpt_prefix = "model.language_model.layers.{}.mlp.experts"
    for layer, row in enumerate(table):
        found = []
        for expert, bits in enumerate(row):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                got = cfg.expert_proj_bits(vllm_prefix.format(layer), expert, proj)
                alt = cfg.expert_proj_bits(ckpt_prefix.format(layer), expert, proj)
                if got != bits or alt != bits:
                    check(
                        "expert bit",
                        False,
                        f"L{layer} E{expert} {proj} table={bits} vllm={got} ckpt={alt}",
                    )
                    return
            found.append(bits)
            hist[bits] = hist.get(bits, 0) + 1
        if len(set(found)) == 1:
            uniform_layers += 1
    check("hist", hist == {2: 7573, 3: 2667}, str(hist))
    check("layers", len(table) == 40 and len(table[0]) == 256)
    # L0 expert 0 is int2 in the table. The catch-all regex is int3, so a
    # prefix miss would report 3 here.
    check(
        "L0 E0 is int2 not the regex",
        cfg.expert_proj_bits(vllm_prefix.format(0), 0, "gate_proj") == 2,
    )
    check(
        "L0 E1 is int3",
        cfg.expert_proj_bits(vllm_prefix.format(0), 1, "down_proj") == 3,
    )
    print(f"  uniform_layers={uniform_layers} mixed_layers={40 - uniform_layers}", flush=True)
    # Trunk stays the top-level bits. The expert regex must not claim it.
    trunk = "model.language_model.model.layers.0.self_attn.q_proj"
    check("trunk q bits", cfg._bits_for(trunk) == 4, str(cfg._bits_for(trunk)))
    check("router skip", cfg._skipped("model.language_model.model.layers.0.mlp.gate"))


def mixed_repack() -> None:
    gen = torch.Generator().manual_seed(7)
    e, n, k = 4, 32, 256
    expert_bits = [2, 3, 2, 3]
    storage = 3
    rows = k * storage // 32
    qweight = torch.empty(e, rows, n, dtype=torch.int32)
    scales = torch.empty(e, k // 128, n, dtype=torch.float16)
    refs = []
    for index, bits in enumerate(expert_bits):
        maxq = maxq_for(bits)
        signed = torch.randint(-maxq, maxq, (n, k), generator=gen, dtype=torch.int16).to(
            torch.int8
        )
        scale = (torch.rand(k // 128, n, generator=gen) * 0.02 + 1e-4).to(torch.float16)
        packed = pack_linear(signed, scale, bits)
        need = k * bits // 32
        qweight[index].fill_(0x7EADBEEF)
        qweight[index, :need].copy_(packed["qweight"])
        scales[index].copy_(packed["scales"])
        refs.append(gptq_to_b70lb1(packed["qweight"], packed["scales"], bits, k))
    plane2, plane_hi, scale_b = gptq_experts_mixed_to_b70lb1(
        qweight, scales, expert_bits, k, storage
    )
    check("mixed plane_hi allocated", plane_hi is not None)
    for index, bits in enumerate(expert_bits):
        p2, hi, sc = refs[index]
        check(f"e{index} b{bits} plane2", torch.equal(plane2[index], p2))
        check(f"e{index} b{bits} scale", torch.equal(scale_b[index], sc))
        if bits == 2:
            check(f"e{index} hi zero", torch.count_nonzero(plane_hi[index]).item() == 0)
        else:
            check(f"e{index} b{bits} hi", torch.equal(plane_hi[index], hi))


def main() -> int:
    print("CONFIG mixed expert bits cpu", flush=True)
    if not BITS_JSON.is_file() or not (MIXED / "config.json").is_file():
        print("VERDICT fail missing checkpoint", flush=True)
        return 1
    bit_table()
    mixed_repack()
    if FAILS:
        print("VERDICT fail", FAILS, flush=True)
        return 1
    print("MIXED_BITS_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
