"""Bit-compare host dequant against llama.cpp gguf-py on real GGUF rows.

Q2_0 has no gguf-py dequant class. That type is checked against an independent
transcription of ggml dequantize_row_q2_0 in this file.
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

GGUF_PY = "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py"
sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402
from gguf import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "b70_gguf_xpu" / "libdequant_cpu.so"
SHARD = (
    "/mnt/cache/b70/qwen3.8-flash-next/ista-gsq-rco/IQ3_XXS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
)

WANTED = {
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    42: "Q2_0",
}


def q2_0_rows(raw: np.ndarray, k: int) -> np.ndarray:
    """ggml dequantize_row_q2_0: 64 elems / 18 bytes, v = (q - 1) * fp16_scale."""
    rows = raw.reshape(-1, raw.shape[-1])
    out = np.empty((rows.shape[0], k), np.float32)
    for b in range(k // 64):
        block = rows[:, b * 18:(b + 1) * 18]
        scale = block[:, 0:2].copy().view(np.float16).reshape(-1).astype(np.float32)
        qs = block[:, 2:]
        for e in range(64):
            q = (qs[:, e // 4] >> ((e % 4) * 2)) & 3
            out[:, b * 64 + e] = (q.astype(np.float32) - 1.0) * scale
    return out


def logical_k(tensor) -> int:
    # ggml ne[0] is the contiguous row length.
    return int(tensor.shape[0])


def sample_rows(tensor, n_rows: int) -> tuple[np.ndarray, int]:
    data = np.asarray(tensor.data)
    flat = data.reshape(-1, data.shape[-1])
    take = min(n_rows, flat.shape[0])
    # Copy so the memmap page is real and the C++ side sees stable bytes.
    return np.ascontiguousarray(flat[:take]), take


def main() -> int:
    lib = ctypes.CDLL(str(LIB))
    lib.b70_dequant_f32_cpu.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.b70_row_bytes_cpu.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.b70_row_bytes_cpu.restype = ctypes.c_int

    paths = [SHARD]
    dev = Path("/mnt/cache/b70/devmodel")
    if dev.is_dir():
        paths.extend(str(p) for p in dev.rglob("*.gguf"))

    found: dict[int, tuple[str, str]] = {}
    failures = 0
    for path in paths:
        if not Path(path).is_file():
            print(f"MISSING {path}")
            continue
        print(f"OPEN {path}")
        reader = gguf.GGUFReader(path)
        seen_here = set()
        for tensor in reader.tensors:
            tid = int(tensor.tensor_type)
            if tid not in WANTED or tid in seen_here:
                continue
            seen_here.add(tid)
            k = logical_k(tensor)
            raw, nrows = sample_rows(tensor, 2)
            if raw.shape[-1] == 0 or nrows == 0:
                continue
            rb = lib.b70_row_bytes_cpu(tid, k)
            if rb != raw.shape[-1]:
                print(f"FAIL {WANTED[tid]} row_bytes cpp={rb} file={raw.shape[-1]} name={tensor.name} k={k}")
                failures += 1
                found[tid] = (path, tensor.name)
                continue
            out = np.empty((nrows, k), np.float32)
            qptr = raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            optr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            lib.b70_dequant_f32_cpu(tid, qptr, nrows, k, optr)
            if tid == 42:
                ref = q2_0_rows(raw, k)
                note = "q2_0-local"
            else:
                ref = np.asarray(dequantize(raw, GGMLQuantizationType(tid)), dtype=np.float32)
                note = "gguf-py"
            if ref.shape != out.shape:
                print(f"FAIL {WANTED[tid]} shape cpp={out.shape} ref={ref.shape} {tensor.name}")
                failures += 1
            else:
                mismatch = int(np.count_nonzero(out.view(np.uint32) != ref.view(np.uint32)))
                max_abs = float(np.max(np.abs(out - ref))) if out.size else 0.0
                status = "OK" if mismatch == 0 else "FAIL"
                if mismatch:
                    failures += 1
                print(
                    f"{status} {WANTED[tid]} vs {note} tensor={tensor.name} "
                    f"rows={nrows} k={k} elems={out.size} mismatches={mismatch} max_abs={max_abs:.3e}"
                )
            found[tid] = (path, tensor.name)
        del reader

    missing = [WANTED[t] for t in WANTED if t not in found]
    if missing:
        print("MISSING_TYPES " + ",".join(missing))
        failures += len(missing)
    print(f"VERDICT failures={failures} types={len(found)}/{len(WANTED)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
