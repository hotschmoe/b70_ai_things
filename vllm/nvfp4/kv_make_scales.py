#!/usr/bin/env python3
"""Convert per-layer K/V amax (from sitecustomize block 10 calibration) into vLLM
per-tensor DEQUANT scales for fp8 e4m3 KV cache.

  scale = amax / FP8_MAX * SAFETY      (store: fp8 = real/scale ; read: real = fp8*scale)
  FP8_MAX (e4m3, OCP) = 448.0
  SAFETY (env KV_SAFETY, default 1.0) adds headroom so slightly-larger future values
         do not clip; for a float format extra headroom costs ~no precision.

Input : kv_amax.json  {layer_name: {k_amax, v_amax, n}}
Output: kv_scales.json {layer_name: {k_scale, v_scale}}  (consumed by NVFP4_KV_SCALES_FILE)
"""
import json, os, sys

FP8_MAX = 448.0
SAFETY = float(os.environ.get("KV_SAFETY", "1.0"))
IN = sys.argv[1] if len(sys.argv) > 1 else "/mnt/vm_8tb/b70/tmp_ssd/kv_amax.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/vm_8tb/github/b70_ai_things/vllm/nvfp4/kv_scales_nvfp4_27b.json"

amax = json.load(open(IN))
scales = {}
print(f"{'layer':52s} {'k_amax':>9s} {'v_amax':>9s} {'k_scale':>9s} {'v_scale':>9s}  n")
kmax = vmax = 0.0
for ln in sorted(amax, key=lambda s: (len(s), s)):
    r = amax[ln]
    ka, va, n = r["k_amax"], r["v_amax"], r["n"]
    ks = ka / FP8_MAX * SAFETY
    vs = va / FP8_MAX * SAFETY
    # floor to avoid degenerate zero scale
    ks = max(ks, 1e-6)
    vs = max(vs, 1e-6)
    scales[ln] = {"k_scale": ks, "v_scale": vs}
    kmax = max(kmax, ka); vmax = max(vmax, va)
    print(f"{ln:52s} {ka:9.3f} {va:9.3f} {ks:9.5f} {vs:9.5f}  {n}")
json.dump(scales, open(OUT, "w"), indent=1, sort_keys=True)
print(f"\nlayers={len(scales)} SAFETY={SAFETY}  global k_amax={kmax:.3f} v_amax={vmax:.3f}  FP8_MAX={FP8_MAX}")
print(f"clip@scale=1.0 (values>448)?  K:{'YES' if kmax>FP8_MAX else 'no'}  V:{'YES' if vmax>FP8_MAX else 'no'}")
print("wrote", OUT)
