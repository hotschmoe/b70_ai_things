#!/usr/bin/env python
"""Offline int8 (RTN, W8A8-weight) requant of the Gated-DeltaNet (GDN) projection
weights for the zml qwen3.6-27b W8A8 serve -- "lever 1" prototype.

Scope of THIS script (prototype):
  - Read the GDN projection weights (in_proj_qkv, in_proj_z, out_proj) for one or
    more linear-attn layers from the READ-ONLY w8a8-sqgptq checkpoint.
  - RTN-symmetric quantize each to int8 [dout, d] + per-output-channel scale
    [dout, 1], matching the scheme the zml loader (common_quant.zig QuantizedLinear)
    expects: symmetric, no zero-point, dequant = weight_i8 * weight_scale.
  - VALIDATE on one layer: rel-L2 and max-abs-error of dequant vs original bf16
    (computed in f32), saturation rate, NaN/inf check.

It DOES NOT write the multi-GB checkpoint variant. With --dump it writes a small
per-layer .safetensors (int8 weight + bf16 scale) to the OUTPUT dir only, as a
proven building block for the later full merge -- it never touches the source.

Scheme (must match ZML_GDN_OPT.md lever 1 and the existing sqgptq mlp/attn tensors):
  weight on disk is [dout, d]; d (axis 1) is the CONTRACTING / input dim.
  scale = max(|W|, over axis d=1) / 127            -> per-output-channel [dout, 1]
  W_i8  = round(W / scale).clip(-127, 127)          -> int8 [dout, d]
  dequant = W_i8 * scale  (scale broadcast over the contracting axis)

The existing sqgptq tensors confirm the on-disk convention:
  mlp.gate_proj.weight  I8 [17408, 5120]  weight_scale BF16 [17408, 1]
  mlp.down_proj.weight  I8 [5120, 17408]  weight_scale BF16 [5120, 1]
i.e. int8 weight [dout, d], bf16 scale [dout, 1], reduce over axis 1 -- for BOTH
column-parallel (gate) and row-parallel (down) layers. We match that exactly.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys

import numpy as np
import ml_dtypes  # provides the bfloat16 numpy dtype

BF16 = ml_dtypes.bfloat16

# The 3 big GDN projections to quantize (per linear-attn layer). in_proj_b/a,
# conv1d, A_log, dt_bias, norm stay bf16 (tiny / stateful -- left untouched).
GDN_PROJ = ("in_proj_qkv", "in_proj_z", "out_proj")
KEY = "model.language_model.layers.{L}.linear_attn.{proj}.weight"

# Full-attention layers (self_attn, already int8 in sqgptq). Every OTHER decoder
# layer is linear-attn (GDN). Used only to sanity-check a requested layer index.
FULL_ATTN_LAYERS = set(range(3, 1000, 4))  # 3, 7, 11, ...


# ----------------------------------------------------------------------------
# safetensors reading (manual header parse -> robust bf16 handling via ml_dtypes)
# ----------------------------------------------------------------------------
_ST_DTYPE = {
    "F32": np.float32, "F16": np.float16, "BF16": BF16,
    "I8": np.int8, "U8": np.uint8, "I32": np.int32, "I64": np.int64,
}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n  # header dict, absolute byte offset where tensor data starts


def load_tensor(path, hdr, data_start, key):
    """Load one tensor as a numpy array in its native dtype (bf16 -> ml_dtypes.bfloat16)."""
    meta = hdr[key]
    dt = _ST_DTYPE[meta["dtype"]]
    o0, o1 = meta["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + o0)
        raw = f.read(o1 - o0)
    return np.frombuffer(raw, dtype=dt).reshape(meta["shape"])


# ----------------------------------------------------------------------------
# the requant itself
# ----------------------------------------------------------------------------
def requant_rtn_symmetric(w_f32):
    """RTN symmetric per-output-channel int8 quant of a [dout, d] weight.

    axis 1 (d) is the contracting/input dim -> one scale per output row.
    Returns (w_i8 [dout, d] int8, scale_f32 [dout, 1] float32).
    """
    assert w_f32.ndim == 2, w_f32.shape
    amax = np.max(np.abs(w_f32), axis=1, keepdims=True)  # [dout, 1]
    # guard all-zero rows: keep scale finite & nonzero, the row quantizes to 0 anyway.
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.rint(w_f32 / scale)                    # round-half-to-even
    q = np.clip(q, -127.0, 127.0).astype(np.int8)  # symmetric, drop -128
    return q, scale


def dequant(w_i8, scale):
    """dequant = w_i8 * scale (exactly what common_quant.zig dequantWeight does, in f32)."""
    return w_i8.astype(np.float32) * scale.astype(np.float32)


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(name, w_orig_f32, w_i8, scale_f32):
    """rel-L2 + max-abs-error of dequant vs original, saturation rate, NaN/inf."""
    # Primary: bf16 scale (what is actually stored on disk & seen by the loader).
    scale_bf16_f32 = scale_f32.astype(BF16).astype(np.float32)
    deq_bf16 = dequant(w_i8, scale_bf16_f32)
    # Secondary: f32 scale (upper bound on quality if scale were stored f32).
    deq_f32 = dequant(w_i8, scale_f32)

    def rel_l2(deq):
        num = np.linalg.norm((deq - w_orig_f32).ravel())
        den = np.linalg.norm(w_orig_f32.ravel())
        return float(num / den)

    def max_abs(deq):
        return float(np.max(np.abs(deq - w_orig_f32)))

    q = w_i8.astype(np.int32)
    sat = float(np.mean((q == 127) | (q == -127)))
    at_clip = float(np.mean(np.abs(q) == 127))  # would-saturate (== hit +-127)
    usage = float(np.mean(np.abs(q) >= 64))     # fraction using the top half of range
    nan_inf = bool(np.any(~np.isfinite(deq_bf16)) or np.any(~np.isfinite(scale_f32)))

    print(f"  [{name}] shape={w_orig_f32.shape} dtype_out=int8+bf16_scale")
    print(f"    rel-L2 (bf16 scale, as served) = {rel_l2(deq_bf16):.4e}")
    print(f"    rel-L2 (f32  scale, ideal)     = {rel_l2(deq_f32):.4e}")
    print(f"    max-abs-err (bf16 scale)       = {max_abs(deq_bf16):.4e}")
    print(f"    orig |w| range                 = [{float(np.min(np.abs(w_orig_f32))):.3e}, "
          f"{float(np.max(np.abs(w_orig_f32))):.3e}]")
    print(f"    i8 saturation (|q|==127)       = {at_clip*100:.4f}%  "
          f"(top-half |q|>=64: {usage*100:.2f}%)")
    print(f"    NaN/inf in dequant or scale    = {nan_inf}")
    return {"rel_l2_bf16": rel_l2(deq_bf16), "rel_l2_f32": rel_l2(deq_f32),
            "max_abs_bf16": max_abs(deq_bf16), "sat": at_clip, "nan_inf": nan_inf}


# ----------------------------------------------------------------------------
# FULL MERGE -- rewrite the shard, replacing the 144 GDN projection weights with
# int8 + bf16 scale, stream-copying everything else verbatim. One-shot ~29 GB.
# ----------------------------------------------------------------------------
_GDN_WEIGHT_RE = re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$")

# Files copied verbatim (hardlinked) into the variant dir, unchanged.
_HARDLINK_FILES = (
    "model-visual.safetensors", "model-mtp.safetensors",
    "config.json", "generation_config.json", "recipe.yaml",
    "preprocessor_config.json", "processor_config.json", "chat_template.jinja",
    "tokenizer.json", "tokenizer_config.json",
)


def _pack_header(hdr_dict):
    """Serialize a safetensors header: 8-byte LE length + JSON, space-padded to 8-byte align."""
    hb = json.dumps(hdr_dict, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hb) % 8)) % 8
    hb += b" " * pad
    return struct.pack("<Q", len(hb)) + hb


def merge(src_ckpt, out_dir, shard="model.safetensors"):
    src_shard = os.path.join(src_ckpt, shard)
    hdr, data_start = read_header(src_shard)
    meta = hdr.get("__metadata__", {"format": "pt"})
    entries = sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                     key=lambda kv: kv[1]["data_offsets"][0])

    os.makedirs(out_dir, exist_ok=True)
    out_shard = os.path.join(out_dir, shard)

    # --- Pass 1: build new header + a data-write plan (offsets are data-relative). ---
    new_hdr = {}
    plan = []          # ('copy', o0, o1) verbatim from src, or ('bytes', b) generated
    off = 0
    n_quant = 0
    for name, v in entries:
        m = _GDN_WEIGHT_RE.search(name)
        if m and v["dtype"] == "BF16":
            w = load_tensor(src_shard, hdr, data_start, name).astype(np.float32)
            w_i8, scale = requant_rtn_symmetric(w)          # int8 [dout,d], f32 [dout,1]
            wb = np.ascontiguousarray(w_i8).tobytes()       # row-major, matches bf16 layout
            sb = np.ascontiguousarray(scale.astype(BF16)).tobytes()
            dout, d = w_i8.shape
            new_hdr[name] = {"dtype": "I8", "shape": [dout, d], "data_offsets": [off, off + len(wb)]}
            off += len(wb); plan.append(("bytes", wb))
            sname = name[:-len(".weight")] + ".weight_scale"
            new_hdr[sname] = {"dtype": "BF16", "shape": [dout, 1], "data_offsets": [off, off + len(sb)]}
            off += len(sb); plan.append(("bytes", sb))
            n_quant += 1
        else:
            o0, o1 = v["data_offsets"]; nb = o1 - o0
            new_hdr[name] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + nb]}
            off += nb; plan.append(("copy", o0, o1))
    new_hdr["__metadata__"] = meta
    total_data = off
    print(f"  quantized {n_quant} GDN projection weights (expect 144 = 48 layers x 3)")
    print(f"  new shard header: {len(new_hdr) - 1} tensors; data section {total_data/1e9:.3f} GB")

    # --- Pass 2: write 8-byte len + header + data (verbatim copies streamed, chunked). ---
    CHUNK = 64 << 20
    with open(src_shard, "rb") as fsrc, open(out_shard, "wb") as fout:
        fout.write(_pack_header(new_hdr))
        written = 0
        for op in plan:
            if op[0] == "bytes":
                fout.write(op[1]); written += len(op[1])
            else:
                _, o0, o1 = op
                fsrc.seek(data_start + o0)
                remaining = o1 - o0
                while remaining:
                    buf = fsrc.read(min(CHUNK, remaining))
                    if not buf:
                        raise IOError(f"short read copying verbatim tensor at {o0}")
                    fout.write(buf); remaining -= len(buf); written += len(buf)
        assert written == total_data, (written, total_data)
    print(f"  wrote {out_shard} ({os.path.getsize(out_shard)/1e9:.3f} GB)")

    # --- index.json: add the 144 weight_scale entries, adjust total_size. ---
    idx = json.load(open(os.path.join(src_ckpt, shard + ".index.json")))
    saved = 35_007_249_408  # placeholder recomputed below
    old_total = idx["metadata"]["total_size"]
    # bytes removed from the model shard = old shard data - new shard data
    old_shard_data = hdr[entries[-1][0]]["data_offsets"][1]  # last end == contiguous total
    saved = old_shard_data - total_data
    for name, v in list(new_hdr.items()):
        if name == "__metadata__":
            continue
        if name.endswith(".weight_scale") and name not in idx["weight_map"]:
            idx["weight_map"][name] = shard
    idx["metadata"]["total_size"] = old_total - saved
    with open(os.path.join(out_dir, shard + ".index.json"), "w") as f:
        json.dump(idx, f)
    print(f"  index.json: weight_map {len(idx['weight_map'])} entries; "
          f"total_size {old_total} -> {old_total - saved} (removed {saved/1e9:.3f} GB)")

    # --- hardlink the unchanged files (fall back to copy if cross-device). ---
    for fn in _HARDLINK_FILES:
        s = os.path.join(src_ckpt, fn)
        if not os.path.exists(s):
            print(f"  WARN: source file missing, skipped: {fn}")
            continue
        d = os.path.join(out_dir, fn)
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    # --- NOTE file: GDN is now int8 despite config.json's compressed-tensors ignore list. ---
    with open(os.path.join(out_dir, "GDN_INT8_NOTE.txt"), "w") as f:
        f.write(
            "This variant = qwen3.6-27b w8a8-sqgptq with the 3 big Gated-DeltaNet (GDN)\n"
            "projections (in_proj_qkv, in_proj_z, out_proj) requantized to int8 (RTN\n"
            "symmetric, per-output-channel bf16 weight_scale [dout,1]) for all 48 linear-\n"
            "attn layers -- lever 1 of ZML_GDN_OPT.md. Only model.safetensors +\n"
            "model.safetensors.index.json were rewritten; all other files are hardlinks\n"
            "of the w8a8-sqgptq source.\n\n"
            "NOTE: config.json / recipe.yaml still list `re:.*linear_attn.*` in the\n"
            "compressed-tensors quantization ignore list. That is now STALE for the GDN\n"
            "weights: they ARE int8 on disk here. This is harmless for the zml loader\n"
            "(common_quant.zig QuantizedLinear auto-selects int8 purely on weight_scale\n"
            "PRESENCE, not on the config metadata), but a compressed-tensors-metadata-\n"
            "driven backend (vLLM/sglang) would mis-read these as bf16. Do not serve this\n"
            "variant through a metadata-driven loader without updating the ignore list.\n"
            "GDN in_proj_b/a, conv1d, A_log, dt_bias, norm stay bf16.\n"
            "GPU-gated acceptance still required (Paris probe + degeneracy check, TP=2).\n")
    print(f"  hardlinked {len(_HARDLINK_FILES)} unchanged files + wrote GDN_INT8_NOTE.txt")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="/mnt/vm_8tb/github/b70_ai_things/models/files/"
                    "qwen3.6-27b/w8a8-sqgptq",
                    help="READ-ONLY source checkpoint dir (model.safetensors inside).")
    ap.add_argument("--shard", default="model.safetensors",
                    help="safetensors shard holding the GDN projections.")
    ap.add_argument("--layer", type=int, default=0,
                    help="linear-attn layer index to requant+validate (0 is linear-attn).")
    ap.add_argument("--projs", nargs="+", default=list(GDN_PROJ),
                    help="which GDN projections (default: all 3 big ones).")
    ap.add_argument("--dump", metavar="OUT.safetensors", default=None,
                    help="optional: write int8 weight + bf16 scale for this layer to "
                         "OUT (a small per-layer file, NOT the served checkpoint).")
    ap.add_argument("--merge", metavar="OUT_DIR", default=None,
                    help="FULL MERGE: write a new checkpoint variant to OUT_DIR (rewrites "
                         "the shard with all 144 GDN projections int8; hardlinks the rest). "
                         "The source ckpt is opened READ-ONLY.")
    args = ap.parse_args()

    if args.merge:
        src_shard = os.path.join(args.ckpt, args.shard)
        if not os.path.isfile(src_shard):
            sys.exit(f"ERR: shard not found: {src_shard}")
        if os.path.abspath(args.merge) == os.path.abspath(args.ckpt):
            sys.exit("ERR: refusing to merge onto the source checkpoint dir.")
        print(f"source (read-only): {args.ckpt}")
        print(f"variant out dir:    {args.merge}\n")
        merge(args.ckpt, args.merge, args.shard)
        print("\nMERGE DONE.")
        return

    shard = os.path.join(args.ckpt, args.shard)
    if not os.path.isfile(shard):
        sys.exit(f"ERR: shard not found: {shard}")
    if args.layer in FULL_ATTN_LAYERS:
        sys.exit(f"ERR: layer {args.layer} is a FULL-ATTN layer (self_attn), not GDN. "
                 f"linear-attn layers are all indices NOT in {{3,7,11,...}}.")

    print(f"source (read-only): {shard}")
    print(f"layer {args.layer} (linear-attn), projections: {args.projs}\n")
    hdr, data_start = read_header(shard)

    out_tensors = {}
    for proj in args.projs:
        key = KEY.format(L=args.layer, proj=proj)
        if key not in hdr:
            sys.exit(f"ERR: key not in shard header: {key}")
        meta = hdr[key]
        w = load_tensor(shard, hdr, data_start, key)
        if meta["dtype"] != "BF16":
            print(f"  NOTE: {proj} is already {meta['dtype']} (expected BF16) -- skipping.")
            continue
        w_f32 = w.astype(np.float32)
        w_i8, scale = requant_rtn_symmetric(w_f32)
        validate(proj, w_f32, w_i8, scale)
        print()
        base = KEY.format(L=args.layer, proj=proj)
        out_tensors[base] = w_i8                                   # I8  [dout, d]
        out_tensors[base[:-7] + ".weight_scale"] = scale.astype(BF16)  # BF16 [dout, 1]

    if args.dump and out_tensors:
        from safetensors.numpy import save_file
        os.makedirs(os.path.dirname(os.path.abspath(args.dump)), exist_ok=True)
        save_file(out_tensors, args.dump)
        sz = os.path.getsize(args.dump)
        print(f"wrote per-layer int8 tensors -> {args.dump} ({sz/1e6:.2f} MB, "
              f"{len(out_tensors)} tensors) [prototype artifact, NOT the served ckpt]")


if __name__ == "__main__":
    main()
