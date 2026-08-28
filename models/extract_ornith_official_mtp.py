#!/usr/bin/env python3
"""Extract the official Ornith native MTP suffix into the local 19-key form."""

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


OFFICIAL_REPO = "ornith-ai/Ornith-1.5-35B-A3B"
OFFICIAL_REVISION = "10fbf86fed7ecee4a061f8b499a618f46001cac1"
OFFICIAL_SHARD = "model-00016-of-00016.safetensors"
OFFICIAL_SHARD_SIZE = 4_378_994_104
OFFICIAL_SHARD_SHA256 = (
    "63d592b6c5efb743e521fc406282353671f5f8d60540dd6017074613b5143f9f"
)
HEADER_BYTES = 97_720
MTP_DATA_OFFSET = 2_689_614_848
MTP_ABSOLUTE_OFFSET = HEADER_BYTES + MTP_DATA_OFFSET
MTP_SUFFIX_BYTES = OFFICIAL_SHARD_SIZE - MTP_ABSOLUTE_OFFSET
RAW_MTP_TENSORS = 785
FUSED_MTP_TENSORS = 19


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_header(path):
    raw = path.read_bytes()
    if len(raw) != HEADER_BYTES:
        raise ValueError(f"header size {len(raw)} != {HEADER_BYTES}")
    header_length = struct.unpack("<Q", raw[:8])[0]
    if header_length + 8 != HEADER_BYTES:
        raise ValueError(
            f"encoded header size {header_length + 8} != {HEADER_BYTES}"
        )
    return json.loads(raw[8:].decode("utf-8"))


def tensor_from_suffix(mapped, spec):
    if spec["dtype"] != "BF16":
        raise ValueError(f"unexpected dtype: {spec['dtype']}")
    start, end = spec["data_offsets"]
    relative = start - MTP_DATA_OFFSET
    shape = tuple(spec["shape"])
    count = 1
    for dimension in shape:
        count *= dimension
    if end - start != count * 2:
        raise ValueError(f"byte count mismatch for shape {shape}")
    if relative < 0 or relative + count * 2 > MTP_SUFFIX_BYTES:
        raise ValueError(f"tensor outside MTP suffix: offsets={start,end}")
    return torch.frombuffer(
        mapped,
        dtype=torch.bfloat16,
        count=count,
        offset=relative,
    ).reshape(shape)


def validate_source_checkpoint(path):
    sidecar = path / "model-mtp.safetensors"
    index_path = path / "model.safetensors.index.json"
    if not sidecar.is_file() or not index_path.is_file():
        raise ValueError(f"source checkpoint lacks sidecar/index: {path}")
    with safe_open(sidecar, framework="pt", device="cpu") as source:
        keys = set(source.keys())
    if len(keys) != FUSED_MTP_TENSORS or not all(k.startswith("mtp.") for k in keys):
        raise ValueError(f"unexpected source sidecar key contract: {len(keys)} keys")
    index = json.loads(index_path.read_text(encoding="ascii"))
    mapped = {
        key: value
        for key, value in index["weight_map"].items()
        if key.startswith("mtp.")
    }
    if set(mapped) != keys or set(mapped.values()) != {"model-mtp.safetensors"}:
        raise ValueError("source index does not map the 19 MTP keys to the sidecar")


def stage_checkpoint(source, destination, sidecar_path, contract):
    if destination.exists():
        raise FileExistsError(f"output checkpoint already exists: {destination}")
    temporary = destination.with_name(destination.name + ".staging")
    if temporary.exists():
        raise FileExistsError(f"staging directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    for source_path in source.iterdir():
        if source_path.name == "model-mtp.safetensors":
            continue
        if source_path.is_file():
            os.link(source_path, temporary / source_path.name)
    shutil.copyfile(sidecar_path, temporary / "model-mtp.safetensors")
    (temporary / "official-mtp-contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.rename(destination)


def main():
    args = parse_args()
    if args.suffix.stat().st_size != MTP_SUFFIX_BYTES:
        raise ValueError(
            f"suffix size {args.suffix.stat().st_size} != {MTP_SUFFIX_BYTES}"
        )
    validate_source_checkpoint(args.source_checkpoint)
    header = load_header(args.header)
    raw_keys = sorted(key for key in header if key.startswith("mtp."))
    if len(raw_keys) != RAW_MTP_TENSORS:
        raise ValueError(f"raw MTP key count {len(raw_keys)} != {RAW_MTP_TENSORS}")
    offsets = sorted(tuple(header[key]["data_offsets"]) for key in raw_keys)
    if offsets[0][0] != MTP_DATA_OFFSET or offsets[-1][1] != (
        OFFICIAL_SHARD_SIZE - HEADER_BYTES
    ):
        raise ValueError("MTP tensors do not span the expected shard suffix")
    if any(left[1] != right[0] for left, right in zip(offsets, offsets[1:])):
        raise ValueError("MTP tensor data is not contiguous")

    sidecar_path = args.suffix.with_name("model-mtp-official-10fbf86.safetensors")
    if sidecar_path.exists():
        raise FileExistsError(f"sidecar output already exists: {sidecar_path}")
    with args.suffix.open("rb") as suffix_file:
        mapped = mmap.mmap(suffix_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            tensors = {}
            expert_prefix = "mtp.layers.0.mlp.experts."
            ordinary_keys = [key for key in raw_keys if not key.startswith(expert_prefix)]
            for key in ordinary_keys:
                tensors[key] = tensor_from_suffix(mapped, header[key])

            down = torch.empty((256, 2048, 512), dtype=torch.bfloat16)
            gate_up = torch.empty((256, 1024, 2048), dtype=torch.bfloat16)
            for expert in range(256):
                prefix = f"{expert_prefix}{expert}."
                down[expert].copy_(
                    tensor_from_suffix(mapped, header[prefix + "down_proj.weight"])
                )
                gate_up[expert, :512].copy_(
                    tensor_from_suffix(mapped, header[prefix + "gate_proj.weight"])
                )
                gate_up[expert, 512:].copy_(
                    tensor_from_suffix(mapped, header[prefix + "up_proj.weight"])
                )
            tensors[expert_prefix + "down_proj"] = down
            tensors[expert_prefix + "gate_up_proj"] = gate_up
            if len(tensors) != FUSED_MTP_TENSORS:
                raise ValueError(f"fused key count {len(tensors)} != {FUSED_MTP_TENSORS}")
            save_file(tensors, sidecar_path, metadata={"format": "pt"})
            del tensors
            del down
            del gate_up
        finally:
            mapped.close()

    with safe_open(sidecar_path, framework="pt", device="cpu") as result:
        result_keys = sorted(result.keys())
        result_shapes = {key: list(result.get_tensor(key).shape) for key in result_keys}
    sidecar_sha256 = sha256_file(sidecar_path)
    suffix_sha256 = sha256_file(args.suffix)
    contract = {
        "format": "ornith-official-native-mtp-19-key-sidecar-v1",
        "official_repo": OFFICIAL_REPO,
        "official_revision": OFFICIAL_REVISION,
        "official_shard": OFFICIAL_SHARD,
        "official_shard_bytes": OFFICIAL_SHARD_SIZE,
        "official_shard_sha256": OFFICIAL_SHARD_SHA256,
        "downloaded_absolute_byte_range": [MTP_ABSOLUTE_OFFSET, OFFICIAL_SHARD_SIZE],
        "downloaded_suffix_bytes": MTP_SUFFIX_BYTES,
        "downloaded_suffix_sha256": suffix_sha256,
        "source_checkpoint": str(args.source_checkpoint),
        "sidecar_bytes": sidecar_path.stat().st_size,
        "sidecar_sha256": sidecar_sha256,
        "tensor_count": len(result_keys),
        "tensor_shapes": result_shapes,
    }
    stage_checkpoint(
        args.source_checkpoint,
        args.output_checkpoint,
        sidecar_path,
        contract,
    )
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
