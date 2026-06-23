#!/usr/bin/env python3
"""
113_w8a8_perstep_microbench.py -- per-OP device-time decomposition for Qwen3.6-27B W8A8 TP=2.
Times every distinct op at its EXACT per-card (TP=2) shape, for M=1 (decode) and M=2048 (prefill).
Two timers per op:
  dev_us  = device throughput time: N kernels enqueued back-to-back between two xpu.Events / N
            (launch overlaps -> ~= the per-op time you get INSIDE a captured graph).
  call_us = perf_counter + synchronize per call (launch-inclusive -> the EAGER per-op cost).
The gap (call_us - dev_us) ~= the per-op launch/dispatch overhead that graph capture removes.

Ops (per card, TP=2):
  MLP(int8):     act_quant K5120 | gate_up GEMM[5120->17408] | silu*mul | act_quant K8704 | down GEMM[8704->5120]
  FullAttn(int8):act_quant K5120 | qkvg GEMM[5120->7168] | o_proj GEMM[3072->5120]
  GDN(bf16):     in_proj_qkvz GEMM[5120->8192] | conv1d(k4,ch5120) | out_proj GEMM[3072->5120]   (recurrence: custom op, not isolated)
  lm_head(bf16): GEMM[5120->124160]  (M=1 only)
Run: gpu-run --card 0 docker run ... python3 113_w8a8_perstep_microbench.py
"""
import sys, time, csv, os, datetime, ctypes
import torch

assert torch.xpu.is_available(), "XPU not available"
DEV = torch.device("xpu:0")
BW = 581.0        # measured B70 read roofline GB/s
INT8_TOPS = 250.0 # measured effective int8 GEMM peak
torch.manual_seed(0)

p = torch.xpu.get_device_properties(0)
print(f"[dev] {p.name}  EU={getattr(p,'gpu_eu_count','?')}  subslices={getattr(p,'gpu_subslice_count','?')}  "
      f"mem={p.total_memory/2**30:.1f}GiB  subgroups={getattr(p,'sub_group_sizes','?')}", flush=True)
# best-effort core clock via Level-Zero
clk_mhz = None
try:
    ze = ctypes.CDLL("libze_loader.so.1")
    ze.zeInit(0)
    # minimal: skip full struct walk; rely on sysfs read instead (printed by wrapper)
except Exception as e:
    pass

try:
    import vllm._xpu_ops  # load the custom .so
except Exception as e:
    print("[warn] vllm._xpu_ops load:", e, flush=True)
HAS_I8 = hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "int8_gemm_w8a8")
HAS_Q  = hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "dynamic_per_token_int8_quant")
print(f"[ops] int8_gemm_w8a8={HAS_I8}  dynamic_per_token_int8_quant={HAS_Q}", flush=True)

NITER = 100
WARM = 20

def dev_time_us(fn):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize()
    s = torch.xpu.Event(enable_timing=True); e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(NITER):
        fn()
    e.record(); torch.xpu.synchronize()
    return s.elapsed_time(e) * 1000.0 / NITER  # ms->us per call

def call_time_us(fn):
    for _ in range(WARM):
        fn(); torch.xpu.synchronize()
    ts = []
    for _ in range(40):
        t0 = time.perf_counter(); fn(); torch.xpu.synchronize(); ts.append((time.perf_counter()-t0)*1e6)
    ts.sort(); return ts[len(ts)//2]

def i8_gemm(M, K, N):
    A = torch.randint(-127,127,(M,K),dtype=torch.int8,device=DEV)
    As = torch.ones(M,1,dtype=torch.float32,device=DEV)
    B = torch.randint(-127,127,(K,N),dtype=torch.int8,device=DEV)
    Ws = torch.ones(1,N,dtype=torch.float32,device=DEV)
    fn = lambda: torch.ops._xpu_C.int8_gemm_w8a8(A,As,None,B,Ws,None,None,torch.bfloat16)
    return fn, (K*N + M*K + M*N*2)  # int8 w + int8 a + bf16 out bytes

def bf16_gemm(M, K, N):
    A = torch.randn(M,K,dtype=torch.bfloat16,device=DEV)
    B = torch.randn(K,N,dtype=torch.bfloat16,device=DEV)
    fn = lambda: torch.matmul(A,B)
    return fn, (K*N*2 + M*K*2 + M*N*2)

def quant(M, K):
    x = torch.randn(M,K,dtype=torch.bfloat16,device=DEV)
    fn = lambda: torch.ops._xpu_C.dynamic_per_token_int8_quant(x, True, 8)
    return fn, (M*K*2 + M*K)  # read bf16, write int8

def silu_mul(M, I):
    g = torch.randn(M,I,dtype=torch.bfloat16,device=DEV); u = torch.randn(M,I,dtype=torch.bfloat16,device=DEV)
    fn = lambda: torch.nn.functional.silu(g)*u
    return fn, (M*I*2*3)

def conv1d(M, C, k=4):
    # depthwise causal conv over seq; decode M=1 so seq dim tiny -> use M as seq for prefill, 1 for decode
    x = torch.randn(1, C, M+k, dtype=torch.bfloat16, device=DEV)
    w = torch.randn(C, 1, k, dtype=torch.bfloat16, device=DEV)
    fn = lambda: torch.nn.functional.conv1d(x, w, groups=C)
    return fn, (C*(M+k)*2 + C*k*2)

# (label, kind, builder-args)  builder returns (fn, bytes)
def build(label, kind, M, *a):
    if kind=="i8":   fn,b = i8_gemm(M,a[0],a[1])
    elif kind=="bf16": fn,b = bf16_gemm(M,a[0],a[1])
    elif kind=="q":  fn,b = quant(M,a[0])
    elif kind=="silu": fn,b = silu_mul(M,a[0])
    elif kind=="conv": fn,b = conv1d(M,a[0])
    return fn,b

# per-card TP=2 op list: (label, kind, K_or_C, N)   FLOP_NK used for GEMM TOPS
OPS = [
 # MLP int8
 ("MLP.act_quant_in",  "q",   5120, 0),
 ("MLP.gate_up_i8",    "i8",  5120, 17408),
 ("MLP.silu_mul",      "silu",17408, 0),
 ("MLP.act_quant_dn",  "q",   8704, 0),
 ("MLP.down_i8",       "i8",  8704, 5120),
 # Full-attn int8
 ("ATTN.act_quant_in", "q",   5120, 0),
 ("ATTN.qkvg_i8",      "i8",  5120, 7168),
 ("ATTN.o_proj_i8",    "i8",  3072, 5120),
 # GDN bf16
 ("GDN.in_qkvz_bf16",  "bf16",5120, 8192),
 ("GDN.conv1d",        "conv",10240, 0),
 ("GDN.out_proj_bf16", "bf16",3072, 5120),
 # head
 ("LMHEAD.bf16",       "bf16",5120, 124160),
]

stamp = sys.argv[1] if len(sys.argv)>1 else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rows=[]
for M in (1, 2048):
    print(f"\n================ M={M} ({'DECODE' if M==1 else 'PREFILL'}) ================", flush=True)
    print(f"{'op':22s} {'K':>6} {'N':>7} {'dev_us':>9} {'call_us':>9} {'GB/s':>8} {'TOPS':>8} {'launch_us':>9}", flush=True)
    for label,kind,K,N in OPS:
        if label.startswith("LMHEAD") and M!=1:   # lm_head only computes the last token
            continue
        try:
            fn,b = build(label,kind,M,K,N)
            du = dev_time_us(fn)
            cu = call_time_us(fn)
            gbps = b/ (du*1e-6) /1e9
            tops = (2.0*M*K*N)/(du*1e-6)/1e12 if (kind in ("i8","bf16") and N>0) else 0.0
            print(f"{label:22s} {K:6d} {N:7d} {du:9.2f} {cu:9.2f} {gbps:8.1f} {tops:8.1f} {cu-du:9.2f}", flush=True)
            rows.append(dict(M=M,op=label,kind=kind,K=K,N=N,dev_us=round(du,2),call_us=round(cu,2),
                             gbps=round(gbps,1),tops=round(tops,1),bytes=b))
        except Exception as ex:
            print(f"{label:22s}  FAIL {type(ex).__name__}: {str(ex)[:80]}", flush=True)

# ---- assemble per-token DECODE estimate (device time) using layer counts ----
def dev(M,op):
    for r in rows:
        if r["M"]==M and r["op"]==op: return r["dev_us"]
    return 0.0
def call(M,op):
    for r in rows:
        if r["M"]==M and r["op"]==op: return r["call_us"]
    return 0.0

for M,tag in ((1,"DECODE per-token"),(2048,"PREFILL M=2048 per-pass")):
    mlp = dev(M,"MLP.act_quant_in")+dev(M,"MLP.gate_up_i8")+dev(M,"MLP.silu_mul")+dev(M,"MLP.act_quant_dn")+dev(M,"MLP.down_i8")
    attn= dev(M,"ATTN.act_quant_in")+dev(M,"ATTN.qkvg_i8")+dev(M,"ATTN.o_proj_i8")
    gdn = dev(M,"GDN.in_qkvz_bf16")+dev(M,"GDN.conv1d")+dev(M,"GDN.out_proj_bf16")
    head= dev(M,"LMHEAD.bf16") if M==1 else 0.0
    # 64 MLP, 16 full-attn, 48 GDN
    total = mlp*64 + attn*16 + gdn*48 + head
    print(f"\n--- {tag}: per-card DEVICE-time GEMM/quant budget (excludes attention math, GDN recurrence, collectives) ---")
    print(f"  MLP  x64 : {mlp:8.1f}us/layer -> {mlp*64/1000:8.2f}ms")
    print(f"  ATTN x16 : {attn:8.1f}us/layer -> {attn*16/1000:8.2f}ms")
    print(f"  GDN  x48 : {gdn:8.1f}us/layer -> {gdn*48/1000:8.2f}ms")
    print(f"  lm_head  : {head:8.1f}us       -> {head/1000:8.2f}ms")
    print(f"  TOTAL device GEMM+quant: {total/1000:.2f}ms  -> {1000.0/total*1000 if total else 0:.1f} tok/s ceiling (this bucket only)")

results_dir = os.environ.get("B70_RESULTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)),"results")
os.makedirs(results_dir,exist_ok=True)
csvp=os.path.join(results_dir,f"perstep_microbench_{stamp}.csv")
with open(csvp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["M","op","kind","K","N","dev_us","call_us","gbps","tops","bytes"])
    w.writeheader(); w.writerows(rows)
print(f"\n[done] CSV -> {csvp}", flush=True)
