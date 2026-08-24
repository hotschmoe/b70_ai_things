#!/usr/bin/env python3
"""Materialize and audit the C4 GDN out_proj-only INT8 candidate.

The base checkpoint supplies every tensor except the 48 target GDN out_proj
weights. Those INT8 weights and scales are copied byte-for-byte from the
already-built combined GDN candidate. Unchanged auxiliary files are hardlinked.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import struct
from pathlib import Path


DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I32": 4}
SHARD = "model.safetensors"
INDEX = "model.safetensors.index.json"
NOTE = "GDN_OUT_PROJ_INT8_NOTE.txt"
HARDLINK_FILES = (
    "model-visual.safetensors",
    "model-mtp.safetensors",
    "config.json",
    "generation_config.json",
    "recipe.yaml",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def tensor_bytes(entry):
    return product(entry["shape"]) * DTYPE_BYTES[entry["dtype"]]


def read_header(path):
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = struct.unpack("<Q", raw)[0]
        return json.loads(handle.read(length)), 8 + length


def pack_header(header):
    payload = json.dumps(header, separators=(",", ":")).encode("ascii")
    payload += b" " * ((8 - len(payload) % 8) % 8)
    return struct.pack("<Q", len(payload)) + payload


def require(entry, name, dtype, shape):
    if entry is None:
        raise ValueError(f"missing tensor: {name}")
    actual_shape = [int(value) for value in entry["shape"]]
    if entry["dtype"] != dtype or actual_shape != list(shape):
        raise ValueError(
            f"bad tensor contract for {name}: got {entry['dtype']} "
            f"{actual_shape}, expected {dtype} {list(shape)}"
        )
    return tensor_bytes(entry)


def tensor_payload_equal(
    left_path, left_start, left_entry, right_path, right_start, right_entry
):
    left_offset = left_start + int(left_entry["data_offsets"][0])
    right_offset = right_start + int(right_entry["data_offsets"][0])
    remaining = tensor_bytes(left_entry)
    if remaining != tensor_bytes(right_entry):
        return False
    chunk_size = 16 << 20
    with left_path.open("rb") as left, right_path.open("rb") as right:
        left.seek(left_offset)
        right.seek(right_offset)
        while remaining:
            size = min(chunk_size, remaining)
            if left.read(size) != right.read(size):
                return False
            remaining -= size
    return True


def layer_contract(base_config):
    text = base_config["text_config"]
    layers = [
        index
        for index, layer_type in enumerate(text["layer_types"])
        if layer_type == "linear_attention"
    ]
    hidden = int(text["hidden_size"])
    qkv_rows = (
        2
        * int(text["linear_num_key_heads"])
        * int(text["linear_key_head_dim"])
        + int(text["linear_num_value_heads"])
        * int(text["linear_value_head_dim"])
    )
    z_rows = (
        int(text["linear_num_value_heads"])
        * int(text["linear_value_head_dim"])
    )
    if len(text["layer_types"]) != 64 or len(layers) != 48:
        raise ValueError("unexpected Qwen3.6 layer census")
    if (hidden, qkv_rows, z_rows) != (5120, 10240, 6144):
        raise ValueError("unexpected Qwen3.6 GDN dimensions")
    return layers, hidden, qkv_rows, z_rows


def is_target_out(name, layers):
    prefix = "model.language_model.layers"
    return any(
        name == f"{prefix}.{layer}.linear_attn.out_proj.weight"
        for layer in layers
    )


def materialize(base_dir, combined_dir, out_dir):
    if (out_dir / SHARD).exists() or (out_dir / INDEX).exists():
        raise ValueError(f"refusing to overwrite existing candidate: {out_dir}")
    base_config = json.loads((base_dir / "config.json").read_text(encoding="ascii"))
    layers, hidden, _qkv_rows, z_rows = layer_contract(base_config)
    base_header, base_start = read_header(base_dir / SHARD)
    combined_header, combined_start = read_header(combined_dir / SHARD)
    entries = sorted(
        ((name, entry) for name, entry in base_header.items() if name != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    )
    new_header = {}
    plan = []
    offset = 0
    target_count = 0
    for name, base_entry in entries:
        sources = [(base_dir / SHARD, base_start, name, base_entry)]
        if is_target_out(name, layers):
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            weight_entry = combined_header.get(name)
            scale_entry = combined_header.get(scale_name)
            require(weight_entry, name, "I8", (hidden, z_rows))
            require(scale_entry, scale_name, "BF16", (hidden, 1))
            sources = [
                (combined_dir / SHARD, combined_start, name, weight_entry),
                (combined_dir / SHARD, combined_start, scale_name, scale_entry),
            ]
            target_count += 1
        for source_path, data_start, source_name, source_entry in sources:
            size = tensor_bytes(source_entry)
            new_header[source_name] = {
                "dtype": source_entry["dtype"],
                "shape": source_entry["shape"],
                "data_offsets": [offset, offset + size],
            }
            start, end = source_entry["data_offsets"]
            if end - start != size:
                raise ValueError(f"non-contiguous tensor size for {source_name}")
            plan.append((source_path, data_start + start, size))
            offset += size
    if target_count != 48:
        raise ValueError(f"expected 48 target out_proj weights, found {target_count}")
    new_header["__metadata__"] = base_header.get("__metadata__", {"format": "pt"})

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".{SHARD}.partial"
    chunk_size = 64 << 20
    try:
        with temporary.open("wb") as output:
            output.write(pack_header(new_header))
            handles = {}
            try:
                for source_path, absolute_start, size in plan:
                    handle = handles.get(source_path)
                    if handle is None:
                        handle = source_path.open("rb")
                        handles[source_path] = handle
                    handle.seek(absolute_start)
                    remaining = size
                    while remaining:
                        payload = handle.read(min(chunk_size, remaining))
                        if not payload:
                            raise IOError(f"short read from {source_path}")
                        output.write(payload)
                        remaining -= len(payload)
            finally:
                for handle in handles.values():
                    handle.close()
        os.replace(temporary, out_dir / SHARD)
    finally:
        if temporary.exists():
            temporary.unlink()

    base_index = json.loads((base_dir / INDEX).read_text(encoding="ascii"))
    candidate_index = copy.deepcopy(base_index)
    for layer in layers:
        root = f"model.language_model.layers.{layer}.linear_attn.out_proj"
        candidate_index["weight_map"][f"{root}.weight_scale"] = SHARD
    saved = 1_509_457_920
    candidate_index["metadata"]["total_size"] = (
        int(base_index["metadata"]["total_size"]) - saved
    )
    (out_dir / INDEX).write_text(
        json.dumps(candidate_index, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    for filename in HARDLINK_FILES:
        source = base_dir / filename
        destination = out_dir / filename
        if destination.exists():
            raise ValueError(f"refusing to replace existing file: {destination}")
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    (out_dir / NOTE).write_text(
        "Qwen3.6-27B W8A8 SQ-GPTQ with only the 48 target GDN out_proj weights\n"
        "stored as RTN symmetric INT8 plus BF16 per-output-channel scales. The\n"
        "INT8 tensors are copied byte-for-byte from w8a8-sqgptq-gdnint8. All\n"
        "GDN in_proj_qkv, in_proj_z, in_proj_b, and in_proj_a weights remain\n"
        "BF16 from the base checkpoint. SGLang requires the dedicated corrected\n"
        "compressed-tensors config overlay; the hardlinked config.json is stale.\n",
        encoding="ascii",
    )


def audit(base_dir, combined_dir, candidate_dir, quant_path):
    base_config = json.loads((base_dir / "config.json").read_text(encoding="ascii"))
    candidate_config = json.loads(
        (candidate_dir / "config.json").read_text(encoding="ascii")
    )
    if base_config != candidate_config:
        raise ValueError("candidate config.json must be the unchanged base artifact")
    layers, hidden, qkv_rows, z_rows = layer_contract(base_config)
    base_header, _ = read_header(base_dir / SHARD)
    combined_header, combined_start = read_header(combined_dir / SHARD)
    candidate_header, candidate_start = read_header(candidate_dir / SHARD)
    base_names = set(base_header) - {"__metadata__"}
    candidate_names = set(candidate_header) - {"__metadata__"}
    expected_scales = {
        f"model.language_model.layers.{layer}.linear_attn.out_proj.weight_scale"
        for layer in layers
    }
    if candidate_names != base_names | expected_scales:
        raise ValueError("candidate tensor-name set is not base plus 48 out_proj scales")

    prefix = "model.language_model.layers"
    out_bytes_base = 0
    out_bytes_candidate = 0
    for layer in layers:
        root = f"{prefix}.{layer}.linear_attn"
        out_name = f"{root}.out_proj.weight"
        scale_name = f"{root}.out_proj.weight_scale"
        out_bytes_base += require(base_header.get(out_name), out_name, "BF16", (hidden, z_rows))
        out_bytes_candidate += require(
            candidate_header.get(out_name), out_name, "I8", (hidden, z_rows)
        )
        out_bytes_candidate += require(
            candidate_header.get(scale_name), scale_name, "BF16", (hidden, 1)
        )
        for name in (out_name, scale_name):
            if not tensor_payload_equal(
                candidate_dir / SHARD,
                candidate_start,
                candidate_header[name],
                combined_dir / SHARD,
                combined_start,
                combined_header[name],
            ):
                raise ValueError(f"candidate/combined tensor bytes mismatch: {name}")
        for projection, shape in (
            ("in_proj_qkv", (qkv_rows, hidden)),
            ("in_proj_z", (z_rows, hidden)),
            ("in_proj_b", (48, hidden)),
            ("in_proj_a", (48, hidden)),
        ):
            name = f"{root}.{projection}.weight"
            require(candidate_header.get(name), name, "BF16", shape)
            if f"{root}.{projection}.weight_scale" in candidate_header:
                raise ValueError(f"BF16 projection unexpectedly contains scale: {name}")

    saved = out_bytes_base - out_bytes_candidate
    if saved != 1_509_457_920:
        raise ValueError(f"unexpected checkpoint saving: {saved}")
    tp2_per_layer = hidden * z_rows - (hidden * (z_rows // 2) + 2 * hidden)
    # BF16 bytes for the row-parallel half-K equal hidden * full-K.
    tp2_saved = tp2_per_layer * len(layers)
    if tp2_saved != 754_483_200:
        raise ValueError(f"unexpected TP=2 saving: {tp2_saved}")

    quant = json.loads(quant_path.read_text(encoding="ascii"))
    expected_ignore = {
        "lm_head",
        r"re:.*linear_attn\.in_proj_qkv$",
        r"re:.*linear_attn\.in_proj_z$",
        r"re:.*linear_attn\.in_proj_b$",
        r"re:.*linear_attn\.in_proj_a$",
        r"re:.*visual.*",
        r"re:.*mtp.*",
    }
    if set(quant.get("ignore") or []) != expected_ignore:
        raise ValueError("out_proj-only quantization ignore mismatch")
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("candidate must use compressed-tensors metadata")
    if quant["config_groups"]["group_0"].get("targets") != ["Linear"]:
        raise ValueError("candidate target must be exactly Linear")

    for filename in HARDLINK_FILES:
        if not (candidate_dir / filename).samefile(base_dir / filename):
            raise ValueError(f"unchanged file is not hardlinked: {filename}")
    index = json.loads((candidate_dir / INDEX).read_text(encoding="ascii"))
    expected_total = 34_419_251_680
    if int(index["metadata"]["total_size"]) != expected_total:
        raise ValueError("candidate index total_size mismatch")
    overlay = copy.deepcopy(base_config)
    overlay["quantization_config"] = quant
    report = {
        "status": "PASS",
        "base": str(base_dir),
        "combined_source": str(combined_dir),
        "candidate": str(candidate_dir),
        "target_gdn_int8_weights": 48,
        "target_gdn_int8_scales": 48,
        "checkpoint_bytes_saved": saved,
        "tp2_runtime_bytes_saved_per_rank": tp2_saved,
        "candidate_index_total_size": expected_total,
        "quantization_ignore": quant["ignore"],
        "bf16_gdn_projections": ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"],
        "int8_gdn_projections": ["out_proj"],
    }
    return overlay, report


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=repo / "models/files/qwen3.6-27b/w8a8-sqgptq",
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdn-out-proj-int8",
    )
    parser.add_argument(
        "--quantization-config",
        type=Path,
        default=repo
        / "sglang/configs/qwen36_w8a8_sqgptq_gdn_out_proj_rtn_quantization.json",
    )
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--overlay-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_dir.resolve()
    combined = args.combined_dir.resolve()
    candidate = args.candidate_dir.resolve()
    if args.materialize:
        materialize(base, combined, candidate)
    overlay, report = audit(
        base, combined, candidate, args.quantization_config.resolve()
    )
    args.overlay_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.overlay_out.write_text(
        json.dumps(overlay, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(
        "C4_GDN_OUT_PROJ_AUDIT PASS "
        f"weights={report['target_gdn_int8_weights']} "
        f"saved_gib={report['checkpoint_bytes_saved'] / 2**30:.3f} "
        f"saved_gib_per_rank={report['tp2_runtime_bytes_saved_per_rank'] / 2**30:.3f}"
    )


if __name__ == "__main__":
    main()
