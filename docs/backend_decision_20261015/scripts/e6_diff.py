#!/usr/bin/env python3
"""E6: per-decode-token device time per kernel family = (n80 - n16) / 64 from unitrace -d summaries."""
import csv, re, sys
from collections import defaultdict

def load(p):
    rows = {}
    for line in open(p):
        line = line.strip()
        if not line.startswith('"'):
            continue
        f = next(csv.reader([line]))
        if len(f) < 3 or not f[1].strip().isdigit(): continue
        name, calls, t = f[:3]
        rows[name] = (int(calls), int(t))
    return rows

def family(n):
    if "MemoryCopy" in n: return n.split('"')[0]
    if "mul_mat_vec_q_moe" in n or "topk_moe" in n: return "MoE routed experts (mmvq_moe + topk)"
    if "convert_mul_mat_vec_bf16" in n or "convert_unary" in n: return "bf16 GEMV + f32->f16 convert"
    if re.search(r"mul_mat_vec|dequantize_mul_mat_vec", n): return "dense quant GEMV (attn/shared/lm_head)"
    if "flash_attn" in n: return "flash attention"
    if "gated_delta" in n or "ssm_conv" in n: return "GDN / conv"
    if "get_rows" in n: return "get_rows"
    if "quantize_row_q8_1" in n: return "q8_1 activation quant"
    return "elementwise/norm/other"

a, b = load(sys.argv[1]), load(sys.argv[2])
ntok = int(sys.argv[3]) if len(sys.argv) > 3 else 64
fam = defaultdict(lambda: [0, 0.0]); top = []
for k in set(a) | set(b):
    dc = b.get(k, (0, 0))[0] - a.get(k, (0, 0))[0]
    dt = (b.get(k, (0, 0))[1] - a.get(k, (0, 0))[1]) / ntok / 1e3
    fam[family(k)][0] += dc / ntok; fam[family(k)][1] += dt
    top.append((dt, dc / ntok, re.sub(r"\(.*", "", k)[:90]))
tot = sum(v[1] for v in fam.values())
print(f"device time per decode token (both cards summed, layer split): {tot:.0f} us")
for f, (c, t) in sorted(fam.items(), key=lambda x: -x[1][1]):
    print(f"{t:9.0f} us {100*t/tot:5.1f}%  {c:7.1f} calls/token  {f}")
print("--- top kernels")
for dt, dc, k in sorted(top, reverse=True)[:14]:
    print(f"{dt:9.0f} us {dc:7.1f} calls  {k}")
