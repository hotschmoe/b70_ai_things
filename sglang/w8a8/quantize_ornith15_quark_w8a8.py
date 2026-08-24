#!/usr/bin/env python3
"""Stream an Ornith BF16 checkpoint into Sglang's proven Quark W8A8 layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_REPO = "shisa-ai/Ornith-1.5-35B-A3B-MTP"
SOURCE_REVISION = "779a91ed5b7597bc6db383d9fffb4343b67892ea"
MTP_SHA256 = "73c6e839971fff3c6d78dbcb6a15895bbab340a2898e98aa6943070751de712e"


@dataclass
class ErrorStats:
    source_sq: float = 0.0
    error_sq: float = 0.0
    elements: int = 0
    max_abs_error: float = 0.0

    def update(self, source: torch.Tensor, quantized: torch.Tensor, scale: torch.Tensor) -> None:
        restored = quantized.float() * scale.float().unsqueeze(-1)
        delta = restored - source.float()
        self.source_sq += float(torch.sum(source.float().square()).item())
        self.error_sq += float(torch.sum(delta.square()).item())
        self.elements += source.numel()
        self.max_abs_error = max(self.max_abs_error, float(delta.abs().max().item()))

    def as_dict(self) -> dict[str, float | int]:
        relative_l2 = (self.error_sq / self.source_sq) ** 0.5 if self.source_sq else 0.0
        rmse = (self.error_sq / self.elements) ** 0.5 if self.elements else 0.0
        return {
            "elements": self.elements,
            "relative_l2": relative_l2,
            "rmse": rmse,
            "max_abs_error": self.max_abs_error,
        }


def quark_config() -> dict:
    weight = {
        "ch_axis": 0,
        "dtype": "int8",
        "group_size": None,
        "is_dynamic": False,
        "is_scale_quant": False,
        "mx_element_dtype": None,
        "observer_cls": "PerChannelMinMaxObserver",
        "qscheme": "per_channel",
        "round_method": "round",
        "scale_calculation_mode": None,
        "scale_format": None,
        "scale_type": "float",
        "symmetric": True,
    }
    inputs = {
        **weight,
        "ch_axis": 1,
        "is_dynamic": True,
    }
    return {
        "algo_config": None,
        "exclude": [
            "lm_head",
            "re:.*embed_tokens.*",
            "re:.*\\.mlp\\.gate$",
            "re:.*shared_expert_gate$",
            "re:.*linear_attn.*",
            "re:.*visual\\..*",
            "re:.*mtp\\..*",
        ],
        "export": {
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
            "pack_method": "order",
            "weight_format": "real_quantized",
            "weight_merge_groups": None,
        },
        "global_quant_config": {
            "bias": None,
            "input_tensors": inputs,
            "output_tensors": None,
            "target_device": None,
            "weight": weight,
        },
        "kv_cache_post_rope": False,
        "kv_cache_quant_config": {},
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "quant_method": "quark",
        "quant_mode": "eager_mode",
        "softmax_quant_spec": None,
        "version": "b70-rtn-1",
    }


def is_dense_text_weight(name: str, tensor: torch.Tensor) -> bool:
    if not name.startswith("model.language_model.layers."):
        return False
    if not name.endswith(".weight") or tensor.ndim != 2 or not tensor.is_floating_point():
        return False
    excluded = (
        ".linear_attn.",
        ".mlp.gate.weight",
        ".shared_expert_gate.weight",
        ".conv1d.weight",
    )
    return not any(piece in name for piece in excluded)


def quantize_rows(
    weight: torch.Tensor,
    device: torch.device,
    row_chunk: int,
    stats: ErrorStats,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(weight.shape)
    matrix = weight.reshape(-1, shape[-1])
    qparts: list[torch.Tensor] = []
    sparts: list[torch.Tensor] = []
    for offset in range(0, matrix.shape[0], row_chunk):
        source = matrix[offset : offset + row_chunk].to(device=device, dtype=torch.float32)
        scale = source.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny) / 127.0
        quantized = torch.round(source / scale.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
        stats.update(source, quantized, scale)
        qparts.append(quantized.cpu())
        sparts.append(scale.cpu())
    quantized = torch.cat(qparts).reshape(shape).contiguous()
    scales = torch.cat(sparts).reshape(shape[:-1]).contiguous()
    return quantized, scales


def expand_experts(
    name: str,
    tensor: torch.Tensor,
    device: torch.device,
    row_chunk: int,
    stats: ErrorStats,
) -> Iterator[tuple[str, torch.Tensor]]:
    prefix = name.rsplit(".experts.", 1)[0] + ".experts"
    quantized, scales = quantize_rows(tensor, device, row_chunk, stats)
    if name.endswith(".experts.gate_up_proj"):
        intermediate = tensor.shape[1] // 2
        for expert in range(tensor.shape[0]):
            for projection, rows in (("gate_proj", slice(0, intermediate)), ("up_proj", slice(intermediate, None))):
                base = f"{prefix}.{expert}.{projection}"
                yield base + ".weight", quantized[expert, rows].clone().contiguous()
                yield base + ".weight_scale", scales[expert, rows].clone().contiguous()
    elif name.endswith(".experts.down_proj"):
        for expert in range(tensor.shape[0]):
            base = f"{prefix}.{expert}.down_proj"
            yield base + ".weight", quantized[expert].clone().contiguous()
            yield base + ".weight_scale", scales[expert].clone().contiguous()
    else:
        raise ValueError(f"unsupported packed expert tensor: {name}")


def copy_auxiliary_files(source: Path, output: Path) -> None:
    for path in source.iterdir():
        if not path.is_file() or path.name.endswith(".safetensors"):
            continue
        if path.name in {"model.safetensors.index.json", "README.md", "config.json"}:
            continue
        shutil.copy2(path, output / path.name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    source: Path,
    output: Path,
    device_name: str,
    row_chunk: int,
    expected_mtp_sha256: str | None = MTP_SHA256,
) -> dict:
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    source_index = json.loads(index_path.read_text())
    source_map = source_index["weight_map"]
    source_files = sorted(set(source_map.values()))
    missing = [name for name in source_files if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source shards: {missing}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mtp_path = source / "model-mtp.safetensors"
    if expected_mtp_sha256 is not None:
        if not mtp_path.is_file():
            raise FileNotFoundError(f"trained MTP sidecar missing: {mtp_path}")
        actual_mtp_sha256 = file_sha256(mtp_path)
        if actual_mtp_sha256 != expected_mtp_sha256:
            raise ValueError(
                f"trained MTP SHA256 mismatch: expected {expected_mtp_sha256}, got {actual_mtp_sha256}"
            )
    else:
        actual_mtp_sha256 = file_sha256(mtp_path) if mtp_path.is_file() else None

    if device_name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested but torch.xpu is unavailable")
        device = torch.device("xpu:0")
    else:
        device = torch.device(device_name)

    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    copy_auxiliary_files(source, partial)

    output_map: dict[str, str] = {}
    total_bytes = 0
    stats = ErrorStats()
    counts = {"dense": 0, "expert_weights": 0, "bf16": 0, "shards": 0}

    for shard_name in source_files:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(source / shard_name, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if name.startswith("model.language_model.layers.") and (
                    name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj")
                ):
                    for out_name, out_tensor in expand_experts(name, tensor, device, row_chunk, stats):
                        tensors[out_name] = out_tensor
                        counts["expert_weights"] += int(out_name.endswith(".weight"))
                elif is_dense_text_weight(name, tensor):
                    quantized, scales = quantize_rows(tensor, device, row_chunk, stats)
                    tensors[name] = quantized
                    tensors[name + "_scale"] = scales
                    counts["dense"] += 1
                else:
                    tensors[name] = tensor.contiguous()
                    counts["bf16"] += int(tensor.is_floating_point())
        save_file(tensors, partial / shard_name, metadata={"format": "pt"})
        for name, tensor in tensors.items():
            output_map[name] = shard_name
            total_bytes += tensor.numel() * tensor.element_size()
        counts["shards"] += 1
        print(f"shard {counts['shards']}/{len(source_files)} -> {shard_name} tensors={len(tensors)}", flush=True)
        del tensors
        if device.type == "xpu":
            torch.xpu.empty_cache()

    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = quark_config()
    (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    output_index = {"metadata": {"total_size": total_bytes}, "weight_map": output_map}
    (partial / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2) + "\n")

    contract = {
        "format": "quark-compatible-w8a8-int8",
        "method": "rtn-per-output-channel-symmetric",
        "activation": "dynamic-per-token-int8-at-serve",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "mtp_sha256": actual_mtp_sha256,
        "device": str(device),
        "row_chunk": row_chunk,
        "counts": counts,
        "quant_error": stats.as_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (partial / "b70-artifact-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    (partial / "README.md").write_text(
        "# Ornith-1.5-35B-A3B W8A8 RTN + trained MTP\n\n"
        "B70 research artifact. See b70-artifact-contract.json and "
        "docs/20260824_pi_terminalbench_model_selection.md.\n"
    )
    partial.rename(output)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="xpu", choices=("xpu", "cpu"))
    parser.add_argument("--row-chunk", type=int, default=8192)
    args = parser.parse_args()
    if args.row_chunk < 1:
        parser.error("--row-chunk must be positive")
    contract = convert(args.source.resolve(), args.output.resolve(), args.device, args.row_chunk)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
