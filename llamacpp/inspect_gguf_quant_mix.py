#!/usr/bin/env python3
"""Inspect pinned GGUF identity, metadata, tensor quant mix, and Q4_K coverage."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_GGUF_PY = Path("/mnt/vm_8tb/b70/llama.cpp/gguf-py")


def json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_category(name: str) -> str:
    if ".nextn" in name or name.startswith("mtp."):
        return "mtp"
    if name.startswith("token_embd"):
        return "embedding"
    if name.startswith("output"):
        return "output_head"
    if ".ffn_" in name:
        return "mlp"
    if ".attn_" in name:
        return "attention"
    if any(token in name for token in (".ssm_", ".conv1d", ".linear_attn")):
        return "gdn"
    if "norm" in name:
        return "norm"
    return "other"


def summarize(path: Path, gguf_py: Path, with_sha256: bool) -> dict[str, Any]:
    sys.path.insert(0, str(gguf_py))
    try:
        from gguf import GGUFReader  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"cannot import GGUFReader from {gguf_py}: {exc}")

    reader = GGUFReader(path)
    types: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"tensors": 0, "elements": 0, "bytes": 0}
    )
    categories: dict[str, dict[str, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: {"tensors": 0, "elements": 0, "bytes": 0}
        )
    )
    tensor_types: dict[str, str] = {}
    for tensor in reader.tensors:
        quant = tensor.tensor_type.name
        category = tensor_category(tensor.name)
        tensor_types[tensor.name] = quant
        for bucket in (types[quant], categories[category][quant]):
            bucket["tensors"] += 1
            bucket["elements"] += int(tensor.n_elements)
            bucket["bytes"] += int(tensor.n_bytes)

    q4k_pairs = []
    q4k_partial_pairs = []
    block_ids = sorted(
        {
            name.split(".")[1]
            for name in tensor_types
            if name.startswith("blk.") and len(name.split(".")) > 2
        },
        key=lambda item: int(item) if item.isdigit() else 10**9,
    )
    for block_id in block_ids:
        gate = f"blk.{block_id}.ffn_gate.weight"
        up = f"blk.{block_id}.ffn_up.weight"
        if gate not in tensor_types and up not in tensor_types:
            continue
        record = {
            "block": block_id,
            "gate": tensor_types.get(gate),
            "up": tensor_types.get(up),
        }
        if record["gate"] == "Q4_K" and record["up"] == "Q4_K":
            q4k_pairs.append(record)
        else:
            q4k_partial_pairs.append(record)

    metadata_keys = (
        "general.architecture",
        "general.name",
        "general.file_type",
        "general.quantization_version",
        "general.size_label",
        "qwen35.block_count",
        "qwen35.context_length",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
    )
    metadata = {}
    for key in metadata_keys:
        field = reader.fields.get(key)
        if field is not None:
            metadata[key] = json_value(field.contents())

    total_elements = sum(item["elements"] for item in types.values())
    total_tensor_bytes = sum(item["bytes"] for item in types.values())
    result = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path) if with_sha256 else None,
        "gguf": {
            "endian": reader.endianess.name,
            "data_offset": int(reader.data_offset),
            "metadata_count": len(reader.fields),
            "tensor_count": len(reader.tensors),
            "metadata": metadata,
        },
        "tensor_totals": {
            "elements": total_elements,
            "bytes": total_tensor_bytes,
            "effective_bits_per_element": (
                8.0 * total_tensor_bytes / total_elements if total_elements else None
            ),
        },
        "types": dict(sorted(types.items())),
        "categories": {
            category: dict(sorted(values.items()))
            for category, values in sorted(categories.items())
        },
        "q4k_dense_swiglu_coverage": {
            "complete_gate_up_pairs": len(q4k_pairs),
            "non_q4k_or_partial_pairs": len(q4k_partial_pairs),
            "complete": q4k_pairs,
            "non_q4k_or_partial": q4k_partial_pairs,
        },
        "mtp_tensor_names": sorted(
            name for name in tensor_types if tensor_category(name) == "mtp"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--gguf-py", type=Path, default=DEFAULT_GGUF_PY)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-sha256", action="store_true")
    parser.add_argument("--expect", action="append", default=[], metavar="PATH:SIZE:SHA256")
    args = parser.parse_args()

    expected = {}
    for item in args.expect:
        path_text, size_text, digest = item.rsplit(":", 2)
        expected[str(Path(path_text).resolve())] = (int(size_text), digest.lower())

    records = []
    passed = True
    for model in args.models:
        record = summarize(model, args.gguf_py, not args.no_sha256)
        pin = expected.get(record["path"])
        if pin is not None:
            pin_ok = record["size_bytes"] == pin[0] and record["sha256"] == pin[1]
            record["identity_pin"] = {
                "expected_size_bytes": pin[0],
                "expected_sha256": pin[1],
                "passed": pin_ok,
            }
            passed = passed and pin_ok
        records.append(record)

    output = {"passed": passed, "models": records}
    text = json.dumps(output, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="ascii")
    else:
        print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
