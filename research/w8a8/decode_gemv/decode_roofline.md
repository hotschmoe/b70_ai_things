# W8A8 int8 DECODE fast path -- roofline analysis + GO/NO-GO

(Deliverable analysis doc; named decode_roofline.md because the harness blocks
FINDINGS.md-style report names.)

Target: close the W8A8 27B decode gap vs FP8 (RESEARCH_TODO Track 1a/1b/1c).
Method: read the built ops + the measured roofline (docs/kernel/23, research/w8a8
probe logs), compute the weight-BW floor per shape, compare to what the current
kernel achieves, then decide whether the lever is a NEW small-M GEMV kernel or
something cheaper.

Authoritative prior data reused (NOT re-derived on GPU by me -- coordinator
re-confirms with bench_decode_gemv.py):
- docs/kernel/23_b70_gemv_gemm_roofline.md  (581 GB/s read ceiling; kernels near roofline)
- research/w8a8/w8a8_fused_probe.log  (the built int8_gemm_w8a16 / _w8a8 ops, M=1 + M=2048)
- research/w8a8/FUSEDQ_NOTES.md  (the act-quant is the capture-persistent hotspot)

--------------------------------------------------------------------------------
## 1. The roofline (weight-bandwidth floor)

Decode is weight-BW-bound: per-token time >= (weight bytes read) / BW. B70 measured
read ceiling = **581 GB/s** (spec ~608). int8 weight = 1 byte/element; fp8 = 1
byte/element (SAME bytes -> SAME decode floor). bf16 = 2 bytes.

qwen3.6-27b linear GEMM shapes (hidden 5120, inter 17408, 24 q-heads x256 + 4
kv-heads x256 -> qkv N=8192, o K=6144), s8 weight, M=1 floor at 581 GB/s:

    shape       N       K       s8 MB    floor ms (@581)   bf16 floor ms
    qkv_proj    8192    5120     41.9      0.0722            0.1444
    o_proj      5120    6144     31.5      0.0541            0.1083
    gate_up    34816    5120    178.3      0.3069            0.6138
    down_proj   5120   17408     89.1      0.1534            0.3069

--------------------------------------------------------------------------------
## 2. What the CURRENT kernel achieves (measured, w8a8_fused_probe.log, M=1)

int8_gemm_w8a16 (f16 act x s8 weight, per-channel scale, ONE fused launch):

    shape       w8a16 ms   eff GB/s   % of 581    fp8 bar ms   vs bf16
    gate_up     0.3142      567.3      97.6%       0.3080       1.91x
    down_proj   0.1672      533.1      91.8%       0.1673       1.90x
    qkv(14B*)   0.1365      537.7      92.5%       0.1352       1.86x
    (* probe used the 14B qkv N=14336; bench_decode_gemv.py uses the 27B N=8192.)

**The int8 W8A16 decode GEMM is at 92-98% of the read roofline and is bit-for-bit
as fast as the fp8 bar** (down 0.1672 vs fp8 0.1673). There is NO material
W8A8-vs-FP8 decode gap at the GEMM level -- both read 1 byte/weight and both hit
roofline. The "int8 decode is kernel-bound" gap is NOT in this GEMM.

--------------------------------------------------------------------------------
## 3. Where the real cost is: the per-token INT8 ACTIVATION QUANT

The s8s8 path (int8_gemm_w8a8, used by the shims for M>1; used by the vllm path at
EVERY M via _fusedq) must first int8-quantize the activation. Measured (M=1):

    shape       w8a8 gemm   + act-quant   = s8s8 total   vs w8a16(no quant)
    gate_up     0.3193       0.0373         0.3566        1.13x slower
    down_proj   0.1742       0.1268         0.3010        1.80x slower
    qkv         0.1371       0.0427         0.1798        1.32x slower

The act-quant is a pure tax at decode: at M<~157 the GEMM is BW-bound, so the s8
XMX 2x compute buys nothing, while the quant adds up to +0.127 ms (down_proj).
FUSEDQ_NOTES + doc 23 confirm this quant is **capture-persistent COMPUTE** (the
per-row K reduction), not just a launch -- the B1 parallel-launch fix cut its
latency but it is still ~0.13 ms on the K=17408 down_proj at M=1.

**FP8 avoids this entirely** (fp8_gemm_w8a16 has no act-quant) -- which is exactly
why fp8 looked faster than the W8A8 serve. The int8 equivalent (int8_gemm_w8a16)
ALSO avoids it, but the shims only use it at M==1 (sglang) or never (vllm).

--------------------------------------------------------------------------------
## 4. MTP makes this matter more, not less

With MTP the decode step runs at M = spec_tokens+1 (~6), and the weight is read
ONCE for all M (BW-bound, amortized). The current shims route M>1 to the s8s8 +
act-quant path -- so every MTP step pays the down_proj quant. int8_gemm_w8a16 at
M=6 is still BW-bound (f16-compute crossover ~M=157) and reads the weight once for
all 6 verify tokens -- same weight-BW floor, ZERO quant. So the quant tax is pure
loss across the whole decode+MTP regime.

--------------------------------------------------------------------------------
## 5. Decision

### NO-GO: a new / VNNI16-reordered small-M int8 GEMV kernel
oneDNN int8_gemm_w8a16 is already at 92-98% of the 581 GB/s roofline at M=1. A
reordered-layout GEMV can only chase the residual ~8% on down_proj/qkv (gate_up is
already 97.6%), i.e. <=1.05-1.08x, at high kernel-tuning cost, and oneDNN likely
already selects a near-optimal internal layout. The llama.cpp #21527 3.1x came
from a SUB-roofline baseline; our baseline is already at roofline, so that lever
does not transfer. (An experimental VNNI16 header, int8_gemm_w8a16_reorder.h, is
included compile-only for anyone who wants to test the 8% -- but the math says skip.)

### GO: route small-M decode through the quant-free int8_gemm_w8a16 op
The real, capture-persistent lever (doc 23 said the act-quant was the only int8
decode headroom -- this ELIMINATES it rather than fusing it). Send M<=64 (decode +
MTP verify + light concurrency) through int8_gemm_w8a16; keep s8s8/_fusedq for
M>64 prefill. Additive, env-gated (B70_W8A16_M_MAX), more accurate (f16 act).
Patches in route_w8a16_decode.md (sglang = one-line; vllm = add NT weight view +
route + register_fake).

Expected uplift (eager microbench, M=1): per-op 1.13-1.80x (down_proj the win),
MLP layer 0.657->0.481 ms = 1.37x. HONEST CAVEAT: _fusedq already hides the quant
LAUNCH under capture, so the captured end-to-end win = the quant COMPUTE only. The
microbench's `graph ms` columns are the gate: if W8A16(graph) ~= W8A8(graph) at
M=6, the captured win is small and this is a MODEST decode bump, not a headline.
If W8A16(graph) < W8A8(graph) by ~the quant (likely on down_proj), ship it.
This is distinct from doc 23's failed 2026-06-23 A/B, which SWAPPED the quant op /
toggled norm-fusion (inductor re-fused the decomposed quant) -- here there is no
quant to re-fuse.

--------------------------------------------------------------------------------
## 6. Exact commands for the coordinator (GPU)

### 6a. Microbench -- prove the A/B on real 27B shapes, M in {1,2,4,6,8}, captured
(sglang W8A8 .so has int8_gemm_w8a16/_w8a8/fp8/quant; the vllm v0240_fusedq .so
adds _fusedq. Bench auto-detects whichever ops are present -- run BOTH .so's for
the full picture.)

    ROOT=/mnt/vm_8tb/b70
    ./bin/gpu-run --card 0 docker run --rm --device /dev/dri \
      -v /dev/dri/by-path:/dev/dri/by-path --ipc=host --shm-size 16g \
      -e ZE_AFFINITY_MASK=0 \
      -v $ROOT/w8a8_kernel:/work/kernel:ro \
      -v /mnt/vm_8tb/github/b70_ai_things/research/w8a8/decode_gemv:/work/bench:ro \
      -e B70_XPU_C_SO=/work/kernel/_xpu_C.abi3.so sglang-xpu:woq bash -c \
      'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
       export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; \
       python3 /work/bench/bench_decode_gemv.py'

    # fusedq variant (vllm build): swap the .so + image
    #   -v $ROOT/w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so:/work/kernel/_xpu_C.abi3.so:ro
    #   image vllm-xpu-env:int8g-v0240

Read: for each shape/M the `graph ms` and `GB/s(g)` of W8A16 vs W8A8/fusedq/FP8,
and the whole-model `W8A16/W8A8` speedup column.

### 6b. (optional) compile-check the experimental reorder header -- COMPILE ONLY
    docker run --rm -v /mnt/vm_8tb/github/b70_ai_things/research/w8a8/decode_gemv:/w:ro \
      sglang-xpu:woq bash -c \
      'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; \
       echo "#include \"/w/int8_gemm_w8a16_reorder.h\"" > /tmp/t.cpp; \
       icpx -fsycl -fsyntax-only -std=c++17 /tmp/t.cpp && echo SYNTAX_OK'

### 6c. Serve A/B (only if 6a shows W8A16 faster-or-equal) -- sweep-gated
Apply Patch A (sglang) or Patch B (vllm) from route_w8a16_decode.md, serve with
B70_W8A16_M_MAX=64 vs =1, gate coherence (gate_concurrent_coherence.py /
serve-sweep --smoke), measure captured decode t/s (perf_probe.py). Land only if
faster-or-equal AND coherent (AGENTS.md).

--------------------------------------------------------------------------------
## 7. Files
- bench_decode_gemv.py           GPU microbench (coordinator runs; 6a)
- route_w8a16_decode.md          the GO lever: sglang + vllm routing patches
- int8_gemm_w8a16_reorder.h      experimental VNNI16 GEMV (NO-GO; compile-only, 6b)
- decode_roofline.md             this analysis
