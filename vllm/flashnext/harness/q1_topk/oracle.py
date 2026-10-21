"""Unsigned integer-key oracle; no torch/XPU import."""
import numpy as np

CANARY = -123456789

def packed(bits):
    bits = np.asarray(bits, dtype=np.uint32)
    order = bits ^ np.where(bits & np.uint32(0x80000000),
                            np.uint32(0xffffffff), np.uint32(0x80000000))
    return (order.astype(np.uint64) << np.uint64(32)) | (
        np.uint64(0xffffffff) ^ np.arange(bits.size, dtype=np.uint64))

def reference(bits, n, width, cap=2051):
    n = max(0, min(n, len(bits)))
    width = max(0, min(width, n, cap))
    out = np.full(cap, CANARY, dtype=np.int32)
    if width:
        keys = packed(bits[:n])
        # Full integer sort is independent of radix threshold selection.
        out[:width] = np.sort(np.argsort(keys)[-width:]).astype(np.int32)
    return out

def radix_reference(bits, n, width, cap=2051):
    n = max(0, min(n, len(bits)))
    width = max(0, min(width, n, cap))
    out = np.full(cap, CANARY, dtype=np.int32)
    if not width:
        return out
    keys = packed(bits[:n])
    prefix = decided = 0
    need = width
    for bit in range(63, -1, -1):
        bitv = 1 << bit
        cnt = int(np.count_nonzero(((keys & np.uint64(decided)) == prefix)
                                   & ((keys & np.uint64(bitv)) != 0)))
        decided |= bitv
        if cnt >= need:
            prefix |= bitv
        else:
            need -= cnt
    ids = np.flatnonzero(keys >= np.uint64(prefix)).astype(np.int32)
    assert len(ids) == width
    out[:width] = ids
    return out

def cases():
    rng = np.random.default_rng(20261021)
    special = np.array([0, 0x80000000, 0x3f800000, 0xbf800000,
                        0x7f800000, 0xff800000, 0x7fc00001, 0x7fffffff,
                        0xffc00001, 0xffffffff, 1, 0x80000001,
                        0x7f7fffff, 0xff7fffff], dtype=np.uint32)
    for n in (1, 2, 257, 4096, 16384, 32768):
        for kind in ("random_bits", "ties_extremes"):
            bits = (rng.integers(0, 2**32, n, dtype=np.uint32)
                    if kind == "random_bits" else np.resize(special, n))
            for width in (0, 1, min(n, 2048), 2051, n + 11):
                yield f"{n}-{kind}-{width}", bits, n, width
    yield "negative_n", special, -1, 10
    yield "negative_width", special, len(special), -1
    yield "clamped_n", special, len(special) + 100, 2051

def selftest():
    count = 0
    for _, bits, n, width in cases():
        np.testing.assert_array_equal(reference(bits, n, width),
                                      radix_reference(bits, n, width))
        count += 1
    # Equal scores prefer lower token IDs, then emit ascending ID order.
    assert reference(np.zeros(8, dtype=np.uint32), 8, 3)[:3].tolist() == [0, 1, 2]
    print(f"CPU_ORACLE_PASS cases={count} ties=lower_id NaNs=raw_bit_order", flush=True)

if __name__ == "__main__":
    selftest()
