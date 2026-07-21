#!/usr/bin/env python3
# parse_optrace.py -- parse an ONEDNN_VERBOSE=dispatch,profile_exec capture (from
# decode_optrace.sh) into a per-op decode table and FLAG every matmul that is not
# on the int8 / nvfp4 XMX fast path. RESEARCH_TODO Track 1e.
#
# Usage: parse_optrace.py <raw.log>
#
# What it reads: oneDNN verbose lines (comma-separated). Format varies across oneDNN
# versions (src_s8 vs src:s8, field order), so parsing is defensive: we pull the
# primitive, the implementation string, the src/wei/dst dtypes, the problem shape, and
# the exec time, wherever they sit. Banner lines "##### OP=.. M=.. scheme=.. op=.. #####"
# emitted by the tracer tag each following oneDNN line with its logical op.
#
# VERDICT logic (per matmul), how to read a decode leak:
#   XMX-int      wei dtype in {s8,u8,s4,u4} AND impl is a jit/ocl xe gemm  -> int8/int4 XMX. GOOD.
#   f4-decomp    wei dtype f4/e2m1/e3m0                                    -> nvfp4 fused (B70 has no
#                                                                            f4 DPAS; internal compute
#                                                                            is bf16 BY DESIGN -- this is
#                                                                            the intended nvfp4 path, NOT
#                                                                            a leak). Judge it by SPEEDUP.
#   bf16-compute wei dtype bf16/f16/f32                                    -> a plain float matmul. For a
#                                                                            'scheme=bf16ref' op this is
#                                                                            the reference; for a QUANTIZED
#                                                                            op label it is a LEAK unless it
#                                                                            is the nvfp4 op's own internal
#                                                                            gemm (then f4 reorder precedes).
#   REF-slow     impl contains 'ref'                                       -> oneDNN reference GEMM (~100-500x
#                                                                            slow). Always a leak.
#   REORDER      primitive == reorder, in the TIMED pass                   -> a per-decode-step weight
#                                                                            re-layout/dequant = leak (weight
#                                                                            reorders belong at load only).
# Plus a SPEEDUP column = bf16ref_time(sameM) / this_time. A 'quantized' op with speedup < ~1.1x is
# effectively bf16 => flag SUSPECT even if the dtype looked right.
import re, sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("usage: parse_optrace.py <raw.log>", file=sys.stderr); sys.exit(2)

BANNER = re.compile(r"#####\s*(.*?)\s*#####")
FAST_WEI = {"s8", "u8", "s4", "u4"}
F4_WEI = {"f4", "e2m1", "e3m0", "f4_e2m1"}
FLOAT_WEI = {"bf16", "f16", "f32", "f64"}


def kv_from_banner(text):
    d = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1); d[k] = v
    return d


def parse_dtype(fields_joined, tag):
    # match src_s8 / src:s8 / src:s8::blocked etc.
    m = re.search(r"\b" + tag + r"[:_]([a-z0-9]+)", fields_joined)
    return m.group(1) if m else "?"


def parse_line(line):
    if "onednn_verbose" not in line:
        return None
    parts = [p.strip() for p in line.split(",")]
    # operation phase: exec / create:... / dispatch
    phase = None
    for p in parts:
        if p == "exec" or p.startswith("create") or p == "dispatch":
            phase = p; break
    prim = None
    for p in parts:
        if p in ("matmul", "reorder", "convolution", "inner_product", "gemm",
                 "sdpa", "softmax", "reduction"):
            prim = p; break
    if prim is None:
        return None
    joined = ",".join(parts)
    impl = "?"
    for p in parts:
        # impl strings look like jit:gemm:xe, ocl:gemm:xe, brg:..., ref, jit:ir, etc.
        if re.match(r"^[a-z]+:[a-z0-9:_]+$", p) or p in ("ref", "simple"):
            impl = p; break
    src = parse_dtype(joined, "src")
    wei = parse_dtype(joined, "wei")
    dst = parse_dtype(joined, "dst")
    # exec time = last field that parses as float
    t = None
    for p in reversed(parts):
        try:
            t = float(p); break
        except ValueError:
            continue
    shape = "?"
    for p in parts:
        if re.match(r"^(mb\d+)?m?\d+.*[xk]\d+", p) and "x" in p.lower():
            shape = p; break
    return dict(phase=phase, prim=prim, impl=impl, src=src, wei=wei, dst=dst, t=t, shape=shape)


rows = []
cur = {}
with open(sys.argv[1], errors="replace") as f:
    timed = False
    for line in f:
        if "TIMED PASS" in line:
            timed = True
            continue
        b = BANNER.search(line)
        if b and "OP=" in line:
            cur = kv_from_banner(b.group(1))
            continue
        r = parse_line(line)
        if r is None:
            continue
        # only score steady-state exec lines from the timed pass
        if r["phase"] != "exec":
            continue
        if not timed:
            continue
        r.update(cur)
        rows.append(r)

# bf16 reference time per M (min across shapes is noisy; keep per (label,M))
ref_t = {}
for r in rows:
    if r.get("scheme") == "bf16ref" and r["prim"] == "matmul" and r["t"] is not None:
        ref_t[(r.get("OP"), r.get("M"))] = r["t"]


def verdict(r):
    if r["prim"] == "reorder":
        return "REORDER-leak(per-step)"
    wei = r["wei"]
    impl = r["impl"]
    scheme = r.get("scheme", "?")
    base = None
    if "ref" in impl:
        base = "REF-slow-LEAK"
    elif wei in FAST_WEI:
        base = "XMX-int"
    elif wei in F4_WEI:
        base = "f4-decomp(nvfp4-by-design)"
    elif wei in FLOAT_WEI:
        base = "bf16-compute" + ("" if scheme == "bf16ref" else "-LEAK?")
    else:
        base = "wei=%s?" % wei
    # speedup check for quantized ops
    if scheme not in ("bf16ref", "?"):
        rt = ref_t.get((r.get("OP"), r.get("M")))
        if rt and r["t"]:
            sp = rt / r["t"]
            if sp < 1.1 and base.startswith(("XMX", "f4")):
                base += " SUSPECT(sp<1.1)"
    return base


def speedup(r):
    if r.get("scheme") in ("bf16ref", "?"):
        return ""
    rt = ref_t.get((r.get("OP"), r.get("M")))
    if rt and r["t"]:
        return "%.2fx" % (rt / r["t"])
    return ""


hdr = ("OP", "M", "scheme", "op/prim", "impl", "src->wei->dst", "shape", "t_ms", "sp", "VERDICT")
w = (14, 3, 8, 22, 14, 16, 22, 9, 6, 30)
def fmt(cols):
    return "  ".join(str(c)[:wi].ljust(wi) for c, wi in zip(cols, w))

print(fmt(hdr))
print(fmt(["-" * x for x in w]))
flags = []
for r in sorted(rows, key=lambda r: (str(r.get("OP")), int(r.get("M", 0) or 0), r.get("scheme", ""))):
    v = verdict(r)
    opname = r.get("op", r["prim"])
    line = fmt([r.get("OP", "?"), r.get("M", "?"), r.get("scheme", "?"),
                opname + "/" + r["prim"], r["impl"],
                "%s->%s->%s" % (r["src"], r["wei"], r["dst"]),
                r["shape"], "%.4f" % r["t"] if r["t"] is not None else "?",
                speedup(r), v])
    print(line)
    if "LEAK" in v.upper() or "SUSPECT" in v.upper():
        flags.append((r, v))

print()
if not rows:
    print("NO exec lines parsed. Check: (1) serve/exec ran with ONEDNN_VERBOSE=dispatch,profile_exec;")
    print("  (2) MODE=live requires --enforce-eager (captured graphs do not re-emit oneDNN verbose);")
    print("  (3) the raw log actually contains 'onednn_verbose' lines.")
else:
    if flags:
        print("FLAGGED (%d) -- decode ops off the int8/nvfp4 XMX fast path:" % len(flags))
        for r, v in flags:
            print("  [%s] OP=%s M=%s op=%s impl=%s wei=%s t=%.4fms -> %s"
                  % (v.split()[0], r.get("OP"), r.get("M"), r.get("op", r["prim"]),
                     r["impl"], r["wei"], r["t"] or 0, v))
        print("\nInterpretation: XMX-int = on the int8 path (good). f4-decomp = the nvfp4 fused path")
        print("(bf16 compute by design; only a leak if slower than bf16 -> SUSPECT). A '-LEAK' on a")
        print("QUANTIZED op label, a REF-slow, or a per-step REORDER = a real dequant-to-bf16 leak = free win.")
    else:
        print("NO leaks flagged: every quantized decode op dispatched to an int8/f4 XMX gemm with speedup.")
