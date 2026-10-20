"""Plumbing GPTQ checkpoint: re-RTN Ornith int4 experts to int3 or int2.

Reads the Ornith GPTQ int4 mixedcal-v2 shards. Each routed-expert
projection is dequantized with pack_gptq.dequant_linear (zero = maxq)
and packed again at --bits. Every other tensor is copied. This is
int4-then-RTN, not a fresh bf16 RTN. It exists so b70_lowbit can serve
int3/int2 before B1's SignRound export covers all 40 layers.

The written quantization_config.bits is --bits. Dynamic excludes are
unchanged, so attention, shared experts, router, mtp, vision, embed,
and lm_head stay the dense tensors from the int4 checkpoint.

Does not modify vllm/flashnext/quant/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

PLUGIN = Path("/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/plugins/b70_lowbit_xpu")
sys.path.insert(0, str(PLUGIN))

from b70_lowbit_xpu.gptq_import import import_pack_gptq  # noqa: E402

dequant_linear, maxq_for, pack_linear, quantize_sym_codes, _unpack = import_pack_gptq()
GROUP = 128

DEFAULT_SRC = (
    "/mnt/vm_8tb/github/b70_ai_things/models/files/"
    "ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2"
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def rtn_pack(weight_nk: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    n, k = weight_nk.shape
    if k % GROUP != 0 or n % 32 != 0:
        raise ValueError(f"rtn shape N={n} K={k} is not packable")
    maxq = maxq_for(bits)
    grouped = weight_nk.float().reshape(n, k // GROUP, GROUP)
    amax = grouped.abs().amax(dim=-1).clamp_min(1e-8)
    scale_gn = (amax / maxq).transpose(0, 1).contiguous().to(torch.float16)
    signed = quantize_sym_codes(weight_nk, scale_gn, bits)
    return pack_linear(signed, scale_gn, bits)


def expert_prefix(key: str) -> str | None:
    marker = ".mlp.experts."
    if marker not in key:
        return None
    for suffix in (".qweight", ".qzeros", ".scales", ".g_idx"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def convert_shard(src: Path, dst: Path, bits: int) -> tuple[int, int]:
    from safetensors import safe_open
    from safetensors.torch import save_file

    grouped: dict[str, dict[str, torch.Tensor]] = {}
    plain: dict[str, torch.Tensor] = {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            prefix = expert_prefix(key)
            if prefix is None:
                plain[key] = handle.get_tensor(key)
                continue
            slot = key[len(prefix) + 1 :]
            grouped.setdefault(prefix, {})[slot] = handle.get_tensor(key)
    converted = 0
    for prefix, parts in grouped.items():
        if "qweight" not in parts or "scales" not in parts:
            raise RuntimeError(f"{prefix} is split across shards; this script assumes one shard")
        weight = dequant_linear(parts, 4)
        packed = rtn_pack(weight, bits)
        for slot, tensor in packed.items():
            plain[prefix + "." + slot] = tensor
        converted += 1
        if converted % 256 == 0:
            log(f"  {src.name}: {converted} experts repacked")
    save_file(plain, dst)
    return converted, len(plain)


def write_config(src: Path, dst: Path, bits: int) -> None:
    cfg_path = src / "config.json"
    cfg = json.loads(cfg_path.read_text())
    q = cfg.get("quantization_config")
    if not isinstance(q, dict):
        raise RuntimeError("source config has no quantization_config")
    q["bits"] = bits
    q["sym"] = True
    q["desc_act"] = False
    q["quant_method"] = "gptq"
    q["checkpoint_format"] = "gptq"
    cfg["quantization_config"] = q
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    qcfg = src / "quantize_config.json"
    if qcfg.is_file():
        extra = json.loads(qcfg.read_text())
        extra["bits"] = bits
        extra["sym"] = True
        extra["desc_act"] = False
        (dst / "quantize_config.json").write_text(json.dumps(extra, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.out)
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if dst.exists() and any(dst.glob("*.safetensors")):
        raise SystemExit(f"refusing to overwrite existing shards in {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    log(f"CONFIG rtn bits={args.bits} src={src} out={dst} shards={len(shards)}")
    total = 0
    for name in shards:
        log(f"COMMAND shard {name}")
        converted, ntensor = convert_shard(src / name, dst / name, args.bits)
        total += converted
        log(f"RESULT {name} experts={converted} tensors={ntensor}")
    write_config(src, dst, args.bits)
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    for path in src.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name in ("config.json", "quantize_config.json", "model.safetensors.index.json"):
            continue
        target = dst / path.name
        if not target.exists():
            shutil.copy2(path, target)
    log(f"VERDICT wrote {dst} expert_projections={total} bits={args.bits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
