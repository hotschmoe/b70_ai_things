#!/usr/bin/env python3
"""Materialize and audit the C4 GDN qkvz-only INT8 candidate.

The base checkpoint supplies every tensor except the 48 in_proj_qkv and 48
in_proj_z target-GDN weights. Those INT8 weights and scales are copied
byte-for-byte from the validated combined GDN candidate. Unchanged auxiliary
files are hardlinked. This tool performs no GPU work.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import prepare_c4_gdn_out_proj_candidate as common


SHARD = common.SHARD
INDEX = common.INDEX
NOTE = "GDN_QKVZ_INT8_NOTE.txt"
HARDLINK_FILES = common.HARDLINK_FILES
CHECKPOINT_BYTES_SAVED = 4_024_958_976
TP2_BYTES_SAVED_PER_RANK = 2_012_479_488
CANDIDATE_TOTAL_SIZE = 31_903_750_624


def target_projection(name, layers):
    prefix = "model.language_model.layers"
    for layer in layers:
        root = f"{prefix}.{layer}.linear_attn"
        for projection in ("in_proj_qkv", "in_proj_z"):
            if name == f"{root}.{projection}.weight":
                return projection
    return None


def materialize(base_dir: Path, combined_dir: Path, out_dir: Path):
    if (out_dir / SHARD).exists() or (out_dir / INDEX).exists():
        raise ValueError(f"refusing to overwrite existing candidate: {out_dir}")
    base_config = json.loads((base_dir / "config.json").read_text(encoding="ascii"))
    layers, hidden, qkv_rows, z_rows = common.layer_contract(base_config)
    expected_shapes = {
        "in_proj_qkv": (qkv_rows, hidden),
        "in_proj_z": (z_rows, hidden),
    }
    base_header, base_start = common.read_header(base_dir / SHARD)
    combined_header, combined_start = common.read_header(combined_dir / SHARD)
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
        projection = target_projection(name, layers)
        if projection is not None:
            scale_name = name.removesuffix(".weight") + ".weight_scale"
            weight_entry = combined_header.get(name)
            scale_entry = combined_header.get(scale_name)
            common.require(weight_entry, name, "I8", expected_shapes[projection])
            common.require(
                scale_entry,
                scale_name,
                "BF16",
                (expected_shapes[projection][0], 1),
            )
            sources = [
                (combined_dir / SHARD, combined_start, name, weight_entry),
                (combined_dir / SHARD, combined_start, scale_name, scale_entry),
            ]
            target_count += 1
        for source_path, data_start, source_name, source_entry in sources:
            size = common.tensor_bytes(source_entry)
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
    if target_count != 96:
        raise ValueError(f"expected 96 target qkv/z weights, found {target_count}")
    new_header["__metadata__"] = base_header.get("__metadata__", {"format": "pt"})

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / f".{SHARD}.partial"
    try:
        with temporary.open("wb") as output:
            output.write(common.pack_header(new_header))
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
                        payload = handle.read(min(64 << 20, remaining))
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
        root = f"model.language_model.layers.{layer}.linear_attn"
        for projection in ("in_proj_qkv", "in_proj_z"):
            candidate_index["weight_map"][f"{root}.{projection}.weight_scale"] = SHARD
    candidate_index["metadata"]["total_size"] = (
        int(base_index["metadata"]["total_size"]) - CHECKPOINT_BYTES_SAVED
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
        "Qwen3.6-27B W8A8 SQ-GPTQ with only the 48 target GDN in_proj_qkv\n"
        "and 48 target GDN in_proj_z weights stored as RTN symmetric INT8 plus\n"
        "BF16 per-output-channel scales. INT8 tensors are copied byte-for-byte\n"
        "from w8a8-sqgptq-gdnint8. GDN out_proj, in_proj_b, and in_proj_a remain\n"
        "BF16 from the base checkpoint. SGLang requires the dedicated corrected\n"
        "compressed-tensors config overlay; the hardlinked config.json is stale.\n",
        encoding="ascii",
    )


def audit(base_dir: Path, combined_dir: Path, candidate_dir: Path, quant_path: Path):
    base_config = json.loads((base_dir / "config.json").read_text(encoding="ascii"))
    candidate_config = json.loads(
        (candidate_dir / "config.json").read_text(encoding="ascii")
    )
    if base_config != candidate_config:
        raise ValueError("candidate config.json must be the unchanged base artifact")
    layers, hidden, qkv_rows, z_rows = common.layer_contract(base_config)
    base_header, _ = common.read_header(base_dir / SHARD)
    combined_header, combined_start = common.read_header(combined_dir / SHARD)
    candidate_header, candidate_start = common.read_header(candidate_dir / SHARD)
    base_names = set(base_header) - {"__metadata__"}
    expected_scales = {
        f"model.language_model.layers.{layer}.linear_attn.{projection}.weight_scale"
        for layer in layers
        for projection in ("in_proj_qkv", "in_proj_z")
    }
    if set(candidate_header) - {"__metadata__"} != base_names | expected_scales:
        raise ValueError("candidate tensor-name set is not base plus 96 qkv/z scales")

    base_bytes = 0
    candidate_bytes = 0
    prefix = "model.language_model.layers"
    for layer in layers:
        root = f"{prefix}.{layer}.linear_attn"
        for projection, shape in (
            ("in_proj_qkv", (qkv_rows, hidden)),
            ("in_proj_z", (z_rows, hidden)),
        ):
            name = f"{root}.{projection}.weight"
            scale_name = f"{root}.{projection}.weight_scale"
            base_bytes += common.require(base_header.get(name), name, "BF16", shape)
            candidate_bytes += common.require(
                candidate_header.get(name), name, "I8", shape
            )
            candidate_bytes += common.require(
                candidate_header.get(scale_name), scale_name, "BF16", (shape[0], 1)
            )
            for tensor_name in (name, scale_name):
                if not common.tensor_payload_equal(
                    candidate_dir / SHARD,
                    candidate_start,
                    candidate_header[tensor_name],
                    combined_dir / SHARD,
                    combined_start,
                    combined_header[tensor_name],
                ):
                    raise ValueError(
                        f"candidate/combined tensor bytes mismatch: {tensor_name}"
                    )
        for projection, shape in (
            ("out_proj", (hidden, z_rows)),
            ("in_proj_b", (48, hidden)),
            ("in_proj_a", (48, hidden)),
        ):
            name = f"{root}.{projection}.weight"
            common.require(candidate_header.get(name), name, "BF16", shape)
            if f"{root}.{projection}.weight_scale" in candidate_header:
                raise ValueError(f"BF16 projection unexpectedly contains scale: {name}")

    saved = base_bytes - candidate_bytes
    if saved != CHECKPOINT_BYTES_SAVED:
        raise ValueError(f"unexpected checkpoint saving: {saved}")
    qkvz_rows = qkv_rows + z_rows
    tp2_saved = (
        qkvz_rows * hidden - (qkvz_rows * hidden // 2 + qkvz_rows)
    ) * len(layers)
    if tp2_saved != TP2_BYTES_SAVED_PER_RANK:
        raise ValueError(f"unexpected TP=2 saving: {tp2_saved}")

    quant = json.loads(quant_path.read_text(encoding="ascii"))
    expected_ignore = {
        "lm_head",
        r"re:.*linear_attn\.out_proj$",
        r"re:.*linear_attn\.in_proj_b$",
        r"re:.*linear_attn\.in_proj_a$",
        r"re:.*visual.*",
        r"re:.*mtp.*",
    }
    if set(quant.get("ignore") or []) != expected_ignore:
        raise ValueError("qkvz-only quantization ignore mismatch")
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("candidate must use compressed-tensors metadata")
    group = quant["config_groups"]["group_0"]
    if group.get("targets") != ["Linear"]:
        raise ValueError("candidate target must be exactly Linear")
    if not (
        group["weights"].get("num_bits") == 8
        and group["weights"].get("strategy") == "channel"
        and group["weights"].get("symmetric") is True
        and group["input_activations"].get("num_bits") == 8
        and group["input_activations"].get("strategy") == "token"
        and group["input_activations"].get("dynamic") is True
        and group["input_activations"].get("symmetric") is True
    ):
        raise ValueError("candidate quantization scheme is not symmetric W8A8")

    for filename in HARDLINK_FILES:
        if not (candidate_dir / filename).samefile(base_dir / filename):
            raise ValueError(f"unchanged file is not hardlinked: {filename}")
    index = json.loads((candidate_dir / INDEX).read_text(encoding="ascii"))
    if int(index["metadata"]["total_size"]) != CANDIDATE_TOTAL_SIZE:
        raise ValueError("candidate index total_size mismatch")
    overlay = copy.deepcopy(base_config)
    overlay["quantization_config"] = quant
    report = {
        "status": "PASS",
        "base": str(base_dir),
        "combined_source": str(combined_dir),
        "candidate": str(candidate_dir),
        "target_gdn_int8_weights": 96,
        "target_gdn_int8_scales": 96,
        "checkpoint_bytes_saved": saved,
        "tp2_runtime_bytes_saved_per_rank": tp2_saved,
        "candidate_index_total_size": CANDIDATE_TOTAL_SIZE,
        "quantization_ignore": quant["ignore"],
        "bf16_gdn_projections": ["out_proj", "in_proj_b", "in_proj_a"],
        "int8_gdn_projections": ["in_proj_qkv", "in_proj_z"],
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
        default=repo / "models/files/qwen3.6-27b/w8a8-sqgptq-gdn-qkvz-int8",
    )
    parser.add_argument(
        "--quantization-config",
        type=Path,
        default=repo
        / "sglang/configs/qwen36_w8a8_sqgptq_gdn_qkvz_rtn_quantization.json",
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
        "C4_GDN_QKVZ_AUDIT PASS "
        f"weights={report['target_gdn_int8_weights']} "
        f"saved_gib={report['checkpoint_bytes_saved'] / 2**30:.3f} "
        f"saved_gib_per_rank={report['tp2_runtime_bytes_saved_per_rank'] / 2**30:.3f}"
    )


if __name__ == "__main__":
    main()
