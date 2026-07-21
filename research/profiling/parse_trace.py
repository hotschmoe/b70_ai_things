#!/usr/bin/env python3
"""parse_trace.py -- decompose a vLLM/torch XPU profiler trace into op categories.

Reads a Kineto/Chrome trace (.json or .json.gz or .pt.trace.json) and aggregates DEVICE
(XPU kernel) event durations by a categorized op name, so we can see where prefill/decode
time actually goes (linear GEMM vs GDN/mamba scan vs attention vs all-reduce vs norm/rope
vs sampling). Also prints the top-N raw kernels so ambiguous names can be hand-categorized.

Usage: python3 parse_trace.py <trace.json[.gz]> [top_n=35]
"""
import sys, json, gzip, re, collections

PATH = sys.argv[1]
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 35

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return json.load(f)

# category -> list of regexes (matched against lowercased kernel name), first match wins.
CATS = [
    ("allreduce/collective", [r"all.?reduce", r"reduce.?scatter", r"all.?gather", r"\bccl\b",
                              r"oneccl", r"push.?ar", r"collective", r"\bsched\b", r"nccl", r"xccl"]),
    ("gdn/mamba-scan",       [r"gdn", r"mamba", r"\bssm\b", r"chunk", r"recurrent", r"delta",
                              r"causal.?conv", r"gated.?delta", r"segsum", r"state.?pass"]),
    ("attention",            [r"attention", r"flash", r"\battn\b", r"\bsdpa\b", r"paged",
                              r"reshape.?and.?cache", r"rotary.?emb.*attn"]),
    ("linear-gemm",          [r"int8_gemm", r"nvfp4_gemm", r"fp8_gemm", r"w8a16", r"w8a8", r"w4a",
                              r"gemm", r"matmul", r"\bmm\b", r"cijk", r"\bdpas\b", r"xetla",
                              r"linear", r"cutlass", r"brgemm", r"onednn"]),
    ("quant/dequant",        [r"quant", r"dequant", r"scaled.?mm", r"per.?token", r"e2m1", r"fp4",
                              r"convert.*int8", r"cast.*int8"]),
    ("norm/rope/act",        [r"rms.?norm", r"layer.?norm", r"\bnorm\b", r"rope", r"rotary",
                              r"silu", r"gelu", r"swiglu", r"activation"]),
    ("sampling/logits",      [r"sample", r"top.?k", r"top.?p", r"softmax", r"argmax", r"logit",
                              r"gather", r"embedding", r"penalt"]),
    ("elementwise/copy",     [r"elementwise", r"\bcopy\b", r"\bcat\b", r"\badd\b", r"\bmul\b",
                              r"fill", r"index", r"slice", r"contiguous", r"memcpy", r"memset",
                              r"vectorized", r"reduce_kernel", r"transpose", r"permute"]),
]
COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in CATS]

def categorize(name):
    n = name.lower()
    for cname, pats in COMPILED:
        if any(p.search(n) for p in pats):
            return cname
    return "other/uncategorized"

data = load(PATH)
events = data.get("traceEvents", data) if isinstance(data, dict) else data

# Kineto: device kernels are cat in {"kernel","gpu_op","xpu_op","gpu_memcpy","gpu_memset"}.
# Some XPU builds tag cat="kernel" or "gpu_user_annotation". We take events with a device cat
# OR events on a stream/device track that have a 'dur'. Be permissive but exclude host "cpu_op".
DEV_CATS = {"kernel", "gpu_op", "xpu_op", "gpu_memcpy", "gpu_memset", "Kernel"}
by_kernel = collections.Counter()
cnt_kernel = collections.Counter()
total = 0.0
ndev = 0
for e in events:
    if not isinstance(e, dict):
        continue
    if e.get("ph") != "X":
        continue
    cat = str(e.get("cat", ""))
    dur = e.get("dur", 0) or 0
    name = e.get("name", "")
    # device kernel heuristic
    is_dev = cat in DEV_CATS or "kernel" in cat.lower() or "gpu" in cat.lower()
    if not is_dev:
        continue
    if dur <= 0:
        continue
    by_kernel[name] += dur
    cnt_kernel[name] += 1
    total += dur
    ndev += 1

if ndev == 0:
    print("NO device-kernel events found. cats present:",
          sorted({str(e.get('cat','')) for e in events if isinstance(e,dict)})[:30])
    sys.exit(2)

cat_tot = collections.Counter()
for name, dur in by_kernel.items():
    cat_tot[categorize(name)] += dur

print(f"=== TRACE: {PATH}")
print(f"device-kernel events: {ndev}   total device time: {total/1000:.2f} ms\n")
print("=== BY CATEGORY (share of device-kernel time) ===")
print(f"{'category':<22} {'ms':>10} {'share':>8}")
for cname, dur in cat_tot.most_common():
    print(f"{cname:<22} {dur/1000:>10.2f} {100*dur/total:>7.1f}%")

print(f"\n=== TOP {TOPN} KERNELS (name | ms | share | count | category) ===")
for name, dur in by_kernel.most_common(TOPN):
    short = name if len(name) <= 68 else name[:65] + "..."
    print(f"{dur/1000:>9.2f} {100*dur/total:>6.1f}%  x{cnt_kernel[name]:<6} [{categorize(name)}]  {short}")
