#!/usr/bin/env python3
# 112_parse_trace.py -- decompose vLLM XPU Kineto traces into per-op DEVICE time buckets.
# Joins device-kernel events to their launching CPU op by correlation id, then buckets by op/kernel name.
# Usage: 112_parse_trace.py <prof_dir> [window_meta.txt]
import json, gzip, glob, os, sys, re
from collections import defaultdict

prof = sys.argv[1] if len(sys.argv) > 1 else '/tmp_ssd/prof112'
meta = sys.argv[2] if len(sys.argv) > 2 else None

files = []
for pat in ('**/*.json', '**/*.json.gz', '*.pt.trace.json', '*.pt.trace.json.gz'):
    files += glob.glob(os.path.join(prof, pat), recursive=True)
files = sorted(set(files), key=lambda f: os.path.getmtime(f))

def load(f):
    op = gzip.open if f.endswith('.gz') else open
    with op(f, 'rt') as fh:
        return json.load(fh)

# device/kernel categories seen across torch builds; XPU Kineto uses 'kernel' and 'gpu_op'
KCAT = {'kernel', 'gpu_op', 'xpu_op', 'gpu_user_annotation', 'Kernel', 'gpu_memcpy', 'gpu_memset'}
OPCAT = {'cpu_op', 'operator', 'user_annotation', 'python_function'}

def bucket(name):
    n = name.lower()
    if 'all_reduce' in n or 'allreduce' in n or 'reduce_scatter' in n or 'all_gather' in n \
       or 'allgather' in n or 'ccl' in n or 'oneccl' in n or 'xpu_comm' in n or 'collective' in n:
        return 'COLLECTIVE'
    if 'int8' in n and ('gemm' in n or 'mm' in n or 'matmul' in n or 'scaled' in n):
        return 'INT8_GEMM'
    if 'per_token' in n and 'quant' in n: return 'ACT_QUANT'
    if 'quant' in n and 'dequant' not in n: return 'ACT_QUANT'
    if 'gdn' in n or 'delta' in n or 'recurr' in n or 'chunk_scan' in n or 'fused_recurrent' in n:
        return 'GDN_RECUR'
    if 'conv' in n or 'causal_conv' in n or 'short_conv' in n: return 'CONV'
    if 'attention' in n or 'attn' in n or 'flash' in n or 'sdpa' in n or 'paged' in n or 'fmha' in n:
        return 'ATTENTION'
    if 'rms' in n or 'layernorm' in n or 'layer_norm' in n or 'l2norm' in n or 'norm' in n:
        return 'NORM'
    if 'rope' in n or 'rotary' in n: return 'ROPE'
    if 'silu' in n or 'gelu' in n or 'swish' in n or 'act_and_mul' in n or 'sigmoid' in n: return 'ACT_FN'
    if 'gemm' in n or 'matmul' in n or 'linear' in n or 'addmm' in n or '::mm' in n or n.startswith('mm') \
       or 'brgemm' in n or 'gemm_kernel' in n or 'xetla' in n or 'cutlass' in n or 'gemv' in n:
        return 'BF16_GEMM'      # bf16 GEMMs = GDN in/out proj, lm_head, mtp
    if 'copy' in n or 'cat' in n or 'index' in n or 'slice' in n or 'reshape' in n or 'contiguous' in n \
       or 'memcpy' in n or 'memset' in n or 'pad' in n or 'narrow' in n or 'view' in n or 'clone' in n:
        return 'COPY_RESHAPE'
    if 'embed' in n: return 'EMBED'
    if 'sampl' in n or 'logits' in n or 'argmax' in n or 'topk' in n or 'softmax' in n or 'gather' in n:
        return 'SAMPLE_LOGITS'
    if 'add' in n or 'mul' in n or 'div' in n or 'elementwise' in n or 'eltwise' in n or 'fill' in n: return 'ELTWISE'
    return 'OTHER'

print(f"found {len(files)} trace file(s) in {prof}")
for f in files:
    print('\n' + '=' * 90)
    print(f"TRACE {os.path.basename(f)}  mtime={os.path.getmtime(f):.0f}  size={os.path.getsize(f)//1024}KB")
    try:
        tr = load(f)
    except Exception as e:
        print("  load error:", e); continue
    evs = tr.get('traceEvents', tr) if isinstance(tr, dict) else tr
    if not isinstance(evs, list):
        print("  no traceEvents"); continue

    # category histogram
    cat_dur = defaultdict(float); cat_cnt = defaultdict(int)
    for e in evs:
        if e.get('ph') == 'X':
            c = e.get('cat', '?'); cat_dur[c] += e.get('dur', 0); cat_cnt[c] += 1
    print("  -- ph=X categories by total dur --")
    for c, d in sorted(cat_dur.items(), key=lambda x: -x[1]):
        print(f"     {c:22s} cnt={cat_cnt[c]:8d}  dur={d/1000:11.2f}ms")

    # correlation -> cpu op name (prefer innermost/most-specific = last writer wins by start time)
    corr2op = {}
    for e in evs:
        if e.get('ph') != 'X':
            continue
        if e.get('cat', '') in OPCAT:
            a = e.get('args', {}) or {}
            cid = a.get('External id', a.get('correlation'))
            if cid is not None:
                corr2op[cid] = e.get('name', '?')

    # aggregate DEVICE kernel dur by correlated op + by raw kernel name
    op_dev = defaultdict(float); op_cnt = defaultdict(int)
    kname_dev = defaultdict(float); kname_cnt = defaultdict(int)
    buck_dev = defaultdict(float)
    tot_dev = 0.0; matched = 0.0
    for e in evs:
        if e.get('ph') != 'X' or e.get('cat', '') not in KCAT:
            continue
        d = e.get('dur', 0); tot_dev += d
        kn = e.get('name', '?'); kname_dev[kn[:80]] += d; kname_cnt[kn[:80]] += 1
        a = e.get('args', {}) or {}
        cid = a.get('External id', a.get('correlation'))
        opn = corr2op.get(cid)
        if opn is not None:
            op_dev[opn] += d; op_cnt[opn] += 1; matched += d
            buck_dev[bucket(opn)] += d
        else:
            buck_dev[bucket(kn)] += d   # fall back to kernel-name bucketing when unmatched
    print(f"  -- total DEVICE-kernel time {tot_dev/1000:.2f}ms ; correlated-to-op {100*matched/max(tot_dev,1):.0f}% --")

    print("  -- DEVICE time by BUCKET (op-name when matched, else kernel-name) --")
    for b, d in sorted(buck_dev.items(), key=lambda x: -x[1]):
        print(f"     {b:14s} {d/1000:10.3f}ms  {100*d/max(tot_dev,1):5.1f}%")

    print("  -- top 25 ops by DEVICE time --")
    for o, d in sorted(op_dev.items(), key=lambda x: -x[1])[:25]:
        print(f"     {d/1000:9.3f}ms  cnt={op_cnt[o]:7d}  [{bucket(o):12s}] {o[:58]}")

    print("  -- top 25 raw KERNEL names by DEVICE time --")
    for k, d in sorted(kname_dev.items(), key=lambda x: -x[1])[:25]:
        print(f"     {d/1000:9.3f}ms  cnt={kname_cnt[k]:7d}  [{bucket(k):12s}] {k[:58]}")

if meta and os.path.exists(meta):
    print('\n' + '=' * 90)
    print("WINDOW META (match files to windows by mtime order):")
    print(open(meta).read())
