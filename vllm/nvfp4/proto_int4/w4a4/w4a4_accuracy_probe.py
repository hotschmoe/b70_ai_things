#!/usr/bin/env python3
# w4a4_accuracy_probe.py -- CPU fake-quant W4A4 accuracy design for qwen3.6-27b.
#
# THE QUESTION this answers: is W4A4 (int4 weights x int4 ACTIVATIONS) usable on
# B70 at all, and does a QuaRot-style Hadamard rotation recover the accuracy that
# naive int4 activations destroy? W4A4's risk is ACCURACY, not speed -- the s4xs4
# DPAS kernel is already proven (proto_int4/). So we settle accuracy first.
#
# Method (numpy only; no torch/scipy needed):
#   Y = X @ W^T for a real qwen3.6-27b linear (gate_proj by default, bf16 slice).
#   X = synthetic activations: Gaussian + 1% heavy outliers (the realistic A4
#   killer -- attention/MLP activations have channel outliers).
#   We compare fp32 Y against dequantized-quantized Y for:
#     W16A16  (sanity, ~0)         W8A8   (int8 both)
#     W4A16   (int4 weight only)   W4A4   (int4 both, NO rotation)
#     W4A4+H  (int4 both, block-Hadamard rotation = parameter-free QuaRot R)
#   Metrics: RMS relerr, cosine, SNR(dB), max abs err, and the activation outlier
#   ratio (max/median |x|) BEFORE vs AFTER rotation to show the mechanism.
#
# Rotation math (exact in fp): R block-diagonal Hadamard, symmetric orthogonal
# (H=H^T, H@H=I). Y = (X H)(W H)^T = X (H H^T) W^T = X W^T. Quantize X H and W H
# instead of X, W -- the Hadamard mixes each block so per-channel outliers are
# spread across the block, shrinking the activation quant range. This is exactly
# the ONLINE Hadamard a real W4A4 kernel would run on the fast path (block size =
# power of 2, e.g. 256), so the probe result maps 1:1 onto the kernel plan.
#
# Usage:
#   python3 w4a4_accuracy_probe.py                 # real gate_proj slice
#   W4A4_LAYER=down_proj python3 w4a4_accuracy_probe.py
#   W4A4_HAD=128 python3 w4a4_accuracy_probe.py    # Hadamard block size
#   W4A4_RANDOM=1 python3 w4a4_accuracy_probe.py   # skip model, random weights

import os, json, struct, math
import numpy as np

BF16 = "models/files/qwen3.6-27b/bf16"
IDX  = os.path.join(BF16, "model.safetensors.index.json")

# ------------------------------------------------------------------ helpers
def load_bf16(name):
    """Load a bf16 tensor from the sharded safetensors as fp32 (numpy has no bf16)."""
    wm = json.load(open(IDX))["weight_map"]
    shard = os.path.join(BF16, wm[name])
    with open(shard, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n)); base = 8 + n
        m = hdr[name]; s, e = m["data_offsets"]
        fh.seek(base + s); raw = fh.read(e - s)
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32).reshape(m["shape"]).astype(np.float32)

def hadamard(n):
    """Normalized 2^k Hadamard (symmetric, orthogonal: H@H = I)."""
    assert n & (n - 1) == 0, "block size must be power of 2"
    H = np.array([[1.0]], dtype=np.float64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / math.sqrt(n)).astype(np.float32)

def block_had(X, H):
    """Apply block-diagonal Hadamard along the last (K) axis. K must be a
    multiple of the block size (the online-Hadamard-per-block that a real W4A4
    kernel runs; QuaRot's R3/R4). Returns X rotated."""
    B = H.shape[0]; *lead, K = X.shape
    assert K % B == 0, f"K={K} not divisible by Hadamard block {B}"
    Xr = X.reshape(*lead, K // B, B)
    Xr = Xr @ H                       # rotate within each block
    return Xr.reshape(*lead, K)

def quant_sym(x, bits, axis):
    """Symmetric per-<axis> fake quant-dequant. axis=-1 -> per-row scale."""
    qmax = (1 << (bits - 1)) - 1                       # 7 (int4) / 127 (int8)
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.maximum(amax / qmax, 1e-12)
    q = np.clip(np.round(x / scale), -qmax - 1, qmax)  # signed range
    return q * scale

def metrics(ref, approx):
    err = approx - ref
    rms = math.sqrt(float(np.mean(err**2)))
    ref_rms = math.sqrt(float(np.mean(ref**2)))
    relerr = rms / (ref_rms + 1e-12)
    a, b = ref.ravel(), approx.ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    snr = 20 * math.log10((ref_rms + 1e-12) / (rms + 1e-12))
    return relerr, cos, snr, float(np.max(np.abs(err)))

def outlier_ratio(x):
    a = np.abs(x)
    return float(np.max(a) / (np.median(a) + 1e-12))

# ------------------------------------------------------------------ setup
np.random.seed(0)
HB   = int(os.environ.get("W4A4_HAD", "256"))
LYR  = os.environ.get("W4A4_LAYER", "gate_proj")
M    = int(os.environ.get("W4A4_M", "512"))          # prefill tokens
NROW = int(os.environ.get("W4A4_N", "1024"))         # output channels to test

if os.environ.get("W4A4_RANDOM") or not os.path.exists(IDX):
    K = int(os.environ.get("W4A4_K", "5120"))
    W = (0.02 * np.random.randn(NROW, K)).astype(np.float32)
    src = f"RANDOM weights [{NROW},{K}]"
else:
    name = f"model.language_model.layers.2.mlp.{LYR}.weight"
    Wfull = load_bf16(name)
    W = np.ascontiguousarray(Wfull[:NROW, :]).astype(np.float32)
    K = W.shape[1]
    src = f"{name} slice [{NROW},{K}]"

# realistic activations: Gaussian body + 1% heavy channel outliers (the A4 killer)
X = np.random.randn(M, K).astype(np.float32)
outc = np.random.rand(M, K) < 0.01
X[outc] *= 12.0

H = hadamard(HB)
Y_ref = X @ W.T                                       # fp32 ground truth

print(f"== W4A4 accuracy probe ==  {src}")
print(f"   activations: synthetic Gaussian + 1% x12 outliers  M={M} K={K}  Hadamard block={HB}")
print(f"   activation outlier ratio (max/median |x|):  raw={outlier_ratio(X):8.1f}"
      f"   after block-Hadamard={outlier_ratio(block_had(X, H)):8.1f}")
print()
hdr = f"{'config':<14}{'relerr':>10}{'cosine':>10}{'SNR(dB)':>10}{'maxAbsErr':>12}"
print(hdr); print("-" * len(hdr))

def run(tag, wq, xq):
    Y = xq @ wq.T
    r, c, s, mx = metrics(Y_ref, Y)
    print(f"{tag:<14}{r:>10.4f}{c:>10.5f}{s:>10.2f}{mx:>12.4f}")
    return r, s

# W16A16 sanity
run("W16A16",  W, X)
# W8A8 (int8 both): the ~lossless baseline
run("W8A8",    quant_sym(W, 8, -1), quant_sym(X, 8, -1))
# W4A16: int4 weight only (activations stay fp) -- isolates weight-quant error
run("W4A16",   quant_sym(W, 4, -1), X)
# W4A4 NO rotation: int4 weight + int4 activation
r_noR, s_noR = run("W4A4",  quant_sym(W, 4, -1), quant_sym(X, 4, -1))
# W4A4 + block Hadamard (parameter-free QuaRot). Rotate, quantize in rotated
# basis, matmul in rotated basis (== original in fp because H H^T = I).
Xr, Wr = block_had(X, H), block_had(W, H)
Yh = quant_sym(Xr, 4, -1) @ quant_sym(Wr, 4, -1).T
r_H, c_H, s_H, mx_H = metrics(Y_ref, Yh)
print(f"{'W4A4+Had':<14}{r_H:>10.4f}{c_H:>10.5f}{s_H:>10.2f}{mx_H:>12.4f}")
# W4A4 + Hadamard + per-group(128) weight scale: finer weight scale on top
gsize = 128
def quant_group(x, bits, g):
    *lead, k = x.shape; qmax = (1 << (bits-1)) - 1
    xr = x.reshape(*lead, k // g, g)
    amax = np.max(np.abs(xr), axis=-1, keepdims=True)
    sc = np.maximum(amax / qmax, 1e-12)
    q = np.clip(np.round(xr / sc), -qmax-1, qmax) * sc
    return q.reshape(*lead, k)
Yhg = quant_group(Xr, 4, gsize) @ quant_group(Wr, 4, gsize).T
r_Hg, c_Hg, s_Hg, mx_Hg = metrics(Y_ref, Yhg)
print(f"{'W4A4+Had+g128':<14}{r_Hg:>10.4f}{c_Hg:>10.5f}{s_Hg:>10.2f}{mx_Hg:>12.4f}")

# ------------------------------------------------------------------ verdict
print()
print("== reading ==")
print(f"  W4A4 no-rotation SNR = {s_noR:5.1f} dB (relerr {r_noR:.3f});"
      f"  W4A4+Hadamard SNR = {s_H:5.1f} dB (relerr {r_H:.3f})")
gain = s_H - s_noR
print(f"  Hadamard recovers {gain:+.1f} dB of output SNR.")
# heuristic: per-layer SNR < ~20 dB (relerr > ~0.1) reliably breaks codegen evals.
def tag(s): return "USABLE" if s >= 20 else ("MARGINAL" if s >= 14 else "BROKEN")
print(f"  verdict:  W4A4(no-rot)={tag(s_noR)}   W4A4+Hadamard={tag(s_H)}   "
      f"(heuristic: >=20 dB usable, <14 dB breaks code evals)")
print("  NOTE: activations here are SYNTHETIC. For a HumanEval+ delta the "
      "coordinator should re-run with a real calibration-trace X (model at "
      "models/files/); this probe fixes the DESIGN (rotation is mandatory).")
