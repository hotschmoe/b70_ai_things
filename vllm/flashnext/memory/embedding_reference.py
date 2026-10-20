#!/usr/bin/env python3
"""CPU-only small-row reference for the packed token embedding."""
import argparse
import sys
import numpy as np

sys.path.insert(0, "/mnt/vm_8tb/b70/build/strata-ab/llama-pinned/gguf-py")
import gguf
from gguf.quants import dequantize

p = argparse.ArgumentParser()
p.add_argument("shard")
p.add_argument("output")
args = p.parse_args()
reader = gguf.GGUFReader(args.shard, "r")
table = next(t for t in reader.tensors if t.name == "token_embd.weight")
ids = np.array([0, 1, 17, 19, 323, 4096, 20000, 248044], dtype=np.int64)
raw = np.array(table.data.reshape(int(table.shape[1]), -1)[ids], copy=True)
dense = np.ascontiguousarray(dequantize(raw, table.tensor_type), dtype=np.float32)
bits = dense.view(np.uint32)
bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
np.savez(args.output, ids=ids, bf16=bf16)
print(f"EMBED_REFERENCE_OK rows={len(ids)} values={bf16.size} qtype={int(table.tensor_type)}")
