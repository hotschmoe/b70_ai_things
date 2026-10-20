"""Small reads from the Flash-Next GGUF shards.

GGUFReader memory-maps the file. The PLE table is 26.8 GiB; callers copy
one 90-byte row at a time and never materialise the table.
"""

from __future__ import annotations

import os

import numpy as np

SHARD1 = os.environ.get(
    "FLASHNEXT_GGUF_SHARD1",
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
)
SHARD2 = os.environ.get(
    "FLASHNEXT_GGUF_SHARD2",
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_S/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
)


def open_reader(path: str):
    import gguf

    return gguf.GGUFReader(path, "r")


def _scalar(item):
    """Numpy scalars become Python ints or floats. Floats stay floats."""
    if hasattr(item, "item"):
        item = item.item()
    if isinstance(item, bool):
        return int(item)
    if isinstance(item, int) and not isinstance(item, bool):
        return int(item)
    if isinstance(item, float):
        return float(item)
    return item


def field_ints(reader, key: str) -> list[int] | None:
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, (bytes, str)):
        return None
    if not isinstance(value, list):
        value = [value]
    out = []
    for item in value:
        item = _scalar(item)
        if isinstance(item, float):
            if item != int(item):
                return None
            item = int(item)
        if not isinstance(item, int):
            return None
        out.append(item)
    return out


def field_value(reader, key: str):
    field = reader.get_field(key)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_scalar(item) for item in value]
    return _scalar(value)


def index_tensors(reader) -> dict:
    return {tensor.name: tensor for tensor in reader.tensors}


def to_f32(tensor) -> np.ndarray:
    """Dequant one tensor to fp32 in numpy order (ne0 was fastest)."""
    from gguf.quants import dequantize

    raw = np.ascontiguousarray(tensor.data)
    dense = dequantize(raw, tensor.tensor_type)
    return np.ascontiguousarray(np.asarray(dense, dtype=np.float32))


def tensor_meta(tensor) -> dict:
    return {
        "name": tensor.name,
        "type": str(tensor.tensor_type).split(".")[-1],
        "ggml_ne": [int(v) for v in tensor.shape],
        "data_shape": [int(v) for v in tensor.data.shape],
        "n_bytes": int(tensor.n_bytes),
    }


def iq4_row_bytes(table, row: int) -> np.ndarray:
    """One packed IQ4_NL row. ``table.data`` is (n_rows, 90) for ne [160, N]."""
    data = table.data
    if data.ndim != 2:
        raise ValueError(f"PLE table ndim {data.ndim} shape {data.shape}")
    row = int(row)
    if row < 0 or row >= data.shape[0]:
        raise IndexError(f"PLE row {row} outside {data.shape[0]}")
    return np.ascontiguousarray(data[row])
