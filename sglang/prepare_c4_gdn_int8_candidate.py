#!/usr/bin/env python3
"""Audit the existing GDN-RTN checkpoint and materialize a serving config.

This tool never rewrites model weights. The source checkpoint deliberately
keeps its original, stale compressed-tensors ignore list because it was first
created for zml. SGLang needs a full config.json whose quantization metadata
describes the three INT8 GDN projections accurately.
"""

import argparse
import copy
import json
import struct
from pathlib import Path


DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I8": 1,
    "I32": 4,
}


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def read_safetensors_header(path):
    with path.open("rb") as handle:
        header_bytes = handle.read(8)
        if len(header_bytes) != 8:
            raise ValueError(f"short safetensors header: {path}")
        length = struct.unpack("<Q", header_bytes)[0]
        return json.loads(handle.read(length))


def load_tensor_metadata(model_dir):
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="ascii"))
    headers = {}
    metadata = {}
    for name, shard_name in index["weight_map"].items():
        if shard_name not in headers:
            headers[shard_name] = read_safetensors_header(model_dir / shard_name)
        entry = headers[shard_name].get(name)
        if entry is None:
            raise ValueError(f"index/header mismatch for {name} in {shard_name}")
        metadata[name] = entry
    return index, metadata


def tensor_bytes(entry):
    dtype = entry["dtype"]
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported dtype in audit: {dtype}")
    return product(entry["shape"]) * DTYPE_BYTES[dtype]


def require_tensor(metadata, name, dtype, shape):
    entry = metadata.get(name)
    if entry is None:
        raise ValueError(f"missing tensor: {name}")
    actual_shape = [int(value) for value in entry["shape"]]
    if entry["dtype"] != dtype or actual_shape != list(shape):
        raise ValueError(
            f"bad tensor contract for {name}: "
            f"got {entry['dtype']} {actual_shape}, expected {dtype} {list(shape)}"
        )
    return tensor_bytes(entry)


def audit(base_dir, candidate_dir, quant_path):
    base_config = json.loads((base_dir / "config.json").read_text(encoding="ascii"))
    candidate_config = json.loads(
        (candidate_dir / "config.json").read_text(encoding="ascii")
    )
    if base_config != candidate_config:
        raise ValueError("candidate config.json must remain identical to the base artifact")

    text_config = base_config["text_config"]
    layer_types = text_config["layer_types"]
    linear_layers = [
        index for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    ]
    if len(layer_types) != 64 or len(linear_layers) != 48:
        raise ValueError(
            f"unexpected layer census: total={len(layer_types)} linear={len(linear_layers)}"
        )

    hidden = int(text_config["hidden_size"])
    qkv_rows = (
        2
        * int(text_config["linear_num_key_heads"])
        * int(text_config["linear_key_head_dim"])
        + int(text_config["linear_num_value_heads"])
        * int(text_config["linear_value_head_dim"])
    )
    z_rows = (
        int(text_config["linear_num_value_heads"])
        * int(text_config["linear_value_head_dim"])
    )
    value_dim = z_rows
    if (hidden, qkv_rows, z_rows, value_dim) != (5120, 10240, 6144, 6144):
        raise ValueError(
            "unexpected Qwen3.6-27B GDN dimensions: "
            f"hidden={hidden} qkv={qkv_rows} z={z_rows} value={value_dim}"
        )

    base_index, base_metadata = load_tensor_metadata(base_dir)
    candidate_index, candidate_metadata = load_tensor_metadata(candidate_dir)
    prefix = "model.language_model.layers"
    base_projection_bytes = 0
    candidate_projection_bytes = 0
    checked = []
    for layer in linear_layers:
        root = f"{prefix}.{layer}.linear_attn"
        for projection, shape in (
            ("in_proj_qkv", (qkv_rows, hidden)),
            ("in_proj_z", (z_rows, hidden)),
            ("out_proj", (hidden, value_dim)),
        ):
            weight = f"{root}.{projection}.weight"
            scale = f"{root}.{projection}.weight_scale"
            base_projection_bytes += require_tensor(
                base_metadata, weight, "BF16", shape
            )
            candidate_projection_bytes += require_tensor(
                candidate_metadata, weight, "I8", shape
            )
            candidate_projection_bytes += require_tensor(
                candidate_metadata, scale, "BF16", (shape[0], 1)
            )
            if scale in base_metadata:
                raise ValueError(f"base unexpectedly contains GDN scale: {scale}")
            checked.append(weight)

        require_tensor(candidate_metadata, f"{root}.in_proj_b.weight", "BF16", (48, hidden))
        require_tensor(candidate_metadata, f"{root}.in_proj_a.weight", "BF16", (48, hidden))
        require_tensor(candidate_metadata, f"{root}.conv1d.weight", "BF16", (10240, 1, 4))
        require_tensor(candidate_metadata, f"{root}.A_log", "BF16", (48,))
        require_tensor(candidate_metadata, f"{root}.dt_bias", "BF16", (48,))

    if len(checked) != 144:
        raise ValueError(f"expected 144 target GDN INT8 weights, found {len(checked)}")

    quantization = json.loads(quant_path.read_text(encoding="ascii"))
    ignores = quantization.get("ignore")
    expected_ignores = {
        "lm_head",
        r"re:.*linear_attn\.in_proj_ba$",
        r"re:.*visual.*",
        r"re:.*mtp.*",
    }
    if set(ignores or []) != expected_ignores:
        raise ValueError(
            f"candidate quantization ignore mismatch: {ignores}; "
            f"expected {sorted(expected_ignores)}"
        )
    if quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("candidate must use compressed-tensors metadata")
    group = quantization["config_groups"]["group_0"]
    if group.get("targets") != ["Linear"]:
        raise ValueError("candidate quantization target must be exactly Linear")
    weights = group["weights"]
    activations = group["input_activations"]
    if not (
        weights.get("num_bits") == 8
        and weights.get("strategy") == "channel"
        and weights.get("symmetric") is True
        and activations.get("num_bits") == 8
        and activations.get("strategy") == "token"
        and activations.get("dynamic") is True
        and activations.get("symmetric") is True
    ):
        raise ValueError("candidate quantization scheme is not symmetric W8A8")

    expected_saved = 5_534_416_896
    saved = base_projection_bytes - candidate_projection_bytes
    if saved != expected_saved:
        raise ValueError(f"unexpected target GDN byte saving: {saved} != {expected_saved}")
    # Column-parallel qkv/z shard both weights and output-channel scales. The
    # row-parallel out_proj shards its contracting dimension but replicates the
    # per-output scale. Therefore exact TP=2 residency is slightly less than
    # simply dividing checkpoint savings by two.
    qkvz_rows = qkv_rows + z_rows
    tp2_saved_per_layer = (
        qkvz_rows * hidden
        - (qkvz_rows * hidden // 2 + qkvz_rows)
        + hidden * value_dim
        - (hidden * value_dim // 2 + 2 * hidden)
    )
    tp2_saved_per_rank = tp2_saved_per_layer * len(linear_layers)
    if tp2_saved_per_rank != 2_766_962_688:
        raise ValueError(
            f"unexpected TP=2 GDN residency saving: {tp2_saved_per_rank}"
        )

    unchanged_files = ("model-mtp.safetensors", "model-visual.safetensors")
    unchanged_same_inode = {}
    for filename in unchanged_files:
        base_file = base_dir / filename
        candidate_file = candidate_dir / filename
        if not base_file.is_file() or not candidate_file.is_file():
            raise ValueError(f"missing unchanged artifact: {filename}")
        if base_file.stat().st_size != candidate_file.stat().st_size:
            raise ValueError(f"unchanged artifact size mismatch: {filename}")
        unchanged_same_inode[filename] = base_file.samefile(candidate_file)

    overlay = copy.deepcopy(base_config)
    overlay["quantization_config"] = quantization
    report = {
        "status": "PASS",
        "base": str(base_dir),
        "candidate": str(candidate_dir),
        "target_linear_attention_layers": linear_layers,
        "target_gdn_int8_weights": len(checked),
        "target_gdn_int8_scales": len(checked),
        "base_projection_bytes": base_projection_bytes,
        "candidate_projection_bytes": candidate_projection_bytes,
        "checkpoint_bytes_saved": saved,
        "tp2_runtime_bytes_saved_per_rank": tp2_saved_per_rank,
        "tp2_runtime_shapes": {
            "in_proj_qkvz": [hidden, (qkv_rows + z_rows) // 2],
            "out_proj": [value_dim // 2, hidden],
            "in_proj_ba_bf16": [hidden, 48],
        },
        "base_index_total_size": int(base_index["metadata"]["total_size"]),
        "candidate_index_total_size": int(
            candidate_index["metadata"]["total_size"]
        ),
        "quantization_ignore": ignores,
        "unchanged_artifact_same_inode": unchanged_same_inode,
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
        "--candidate-dir",
        type=Path,
        default=repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdnint8",
    )
    parser.add_argument(
        "--quantization-config",
        type=Path,
        default=repo / "sglang/configs/qwen36_w8a8_sqgptq_gdnrtn_quantization.json",
    )
    parser.add_argument("--overlay-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    overlay, report = audit(
        args.base_dir.resolve(),
        args.candidate_dir.resolve(),
        args.quantization_config.resolve(),
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
        "C4_GDN_AUDIT PASS "
        f"weights={report['target_gdn_int8_weights']} "
        f"saved_gib={report['checkpoint_bytes_saved'] / 2**30:.3f} "
        f"saved_gib_per_rank={report['tp2_runtime_bytes_saved_per_rank'] / 2**30:.3f}"
    )


if __name__ == "__main__":
    main()
