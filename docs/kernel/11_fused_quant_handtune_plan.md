# 11 - Fused quant hand-tune plan: the ops AROUND the int8 W8A8 GEMM

> **[!] RE-PRIORITIZED + MEASURED 2026-06-23 (card 0, :int8g).** The earlier "deprioritized -- capture absorbs the
> dispatch" call was INCOMPLETE: the standalone `dynamic_per_token_int8_quant` is "1 sub-group (32) per row" so at
> M=1 it SERIALIZES the per-row absmax over K -- ~101us for K=17408 (a `[1,17408]` quant!), and that is REDUCTION
> WORK that survives graph capture (kernel/23). The fusions fix BOTH the launch AND the serial reduction (the
> producer already does a parallel K-reduction). MEASURED, both CORRECT (max int8 diff 1):
> - **RMSNorm + int8 quant (qkv, gate_up, K=5120): the native op `_C.rms_norm_dynamic_per_token_quant` EXISTS and
>   RUNS on XPU (via `vllm._custom_ops`) -> 2.06x @M=1 (32.7 vs 67.4us separate).** Just needs WIRING.
> - **silu_and_mul + int8 quant (down_proj, K=17408): NO native op (gap confirmed). Wrote a Triton draft
>   (`contrib/vllm_int8_xpu/silu_mul_quant_triton.py`) -> 1.51x @M=1 (70 vs 105us), 1.73x @M=64.** Triton has a
>   ~60-70us XPU dispatch floor -> a NATIVE SYCL version would beat it (and the win survives capture).
> PRODUCTION: (1) wire `rms_norm_dynamic_per_token_quant` into the int8 linear's RMSNorm-fed inputs (~80 linears,
> 2x); (2) build the native silu+quant for down_proj (~64 linears, >=1.5x). This is real W8A8 decode headroom that
> graph capture does NOT cover. (Below: the original plan -- still accurate; the RMSNorm op is now confirmed XPU-live.)
>
> **[!!] LIVE-CAPTURE CAUTION (2026-06-23, both cards) -- the eager microbenches over-state the captured win.**
> Two live A/B runs in the real GRAPH=1 PIECEWISE serve (JOURNAL 2026-06-23; results/actquant_fusion_ab_14b_*.csv):
> (1) The W4A8 cheapest variant -- just SWAP the standalone pure-torch `_ref` quant for the native
>     `_C.dynamic_per_token_int8_quant` op -- REGRESSED 19% (45.1 -> 36.6 decode t/s). Reason: under PIECEWISE +
>     inductor graph-partition the decomposable `_ref` gets FUSED into the captured graph, while the native custom op
>     is an OPAQUE captured node whose serial K-reduction persists. So the EAGER baseline these microbenches beat
>     (separate, launch-bound ops) is NOT the captured baseline -- the captured baseline is the inductor-FUSED
>     decomposable path, which is much stronger. (2) The easy inductor route, `fuse_norm_quant=true`, served clean
>     (no XPU NameError) but gave only +1.9% with no "fused N patterns" log -> unattributed.
> IMPLICATION for the wiring plan below: the 2.06x / 1.51x are EAGER-vs-eager. Before investing in the fragile
> ~80-linear `rms_norm_dynamic_per_token_quant` wiring, MEASURE whether the native fused op lands as an opaque
> captured node and whether that still beats the current inductor-fused (RMSNorm + decomposed `_ref` quant) captured
> baseline -- the W4A8 result says an opaque op can LOSE to inductor fusion under capture. The genuinely safe win is
> still option (1) in kernel/23: fuse quant INTO the GEMM prologue (one opaque node that ALSO does the GEMM).

Owner: the "...and others" agent (everything on the w8a8 forward path EXCEPT the GEMM itself).
Mission (doc 04 lever B3): make the int8 activation quant DISAPPEAR into the op that produces the
activation -- fuse RMSNorm+quant and SiLU-and-mul+quant so the GEMM's int8 src is emitted in one pass,
with NO extra global round-trip of the f16 activation. ASCII only. Date: 2026-06-20.

VERIFIED vs PROPOSED is marked per item. Code was read on the host
(`ssh root@192.168.10.5:/mnt/vm_8tb/b70/vllm-xpu-kernels`); SYCL skeletons are DRAFT (codex-assisted,
not compiled). Model dims used throughout: Qwen3-14B = hidden H=5120, intermediate I=17408, L=40 layers.

---

## 0. TL;DR

- **The big find [VERIFIED]:** the fused **rmsnorm + per-token int8 quant** kernel ALREADY EXISTS and is
  registered -- `csrc/layernorm_quant.cpp::rms_norm_dynamic_per_token_quant` (op `torch.ops._C.
  rms_norm_dynamic_per_token_quant`, int8 path at line 1034, handles the fused-add-residual case). It is
  NOT WIRED into the model: the int8 linear (`contrib/vllm_int8_xpu/xpu_int8.py::apply_weights`) still
  calls the **standalone** `dynamic_per_token_int8_quant` on an already-materialized f16 `x`. So the kernel
  exists; the **wiring** (a vLLM model-patch) is the missing piece for RMSNorm.
- **The gap to WRITE [VERIFIED gap]:** there is NO fused **silu_and_mul + per-token int8 quant** kernel.
  The existing `silu_and_mul_quant` (`csrc/activation.cpp`) is **FP8-only + static per-tensor scale**
  (TORCH_CHECK out.dtype in {e4m3,e5m2}; `*scale_` is a `[1]` tensor). `silu_and_mul_per_block_quant`
  has an int8 path but it is **per-column-group** (group_size 64/128, DeepSeek block scheme), NOT the
  per-token-symmetric scale our s8s8s32 oneDNN GEMM consumes. **down_proj input has no fused int8 path.**
- **Round-trips wasted [VERIFIED by accounting]:** the standalone path pays **5 f16-sized global passes**
  over the activation per fusable input (producer reads x twice + writes f16 y; quant reads y twice +
  writes int8). Fusion drops it to **3 passes** (one op reads x 3x, writes int8). Net **~6.3 MiB/token**
  of eliminated activation traffic on Qwen3-14B = **~10.9 us/token** at 608 GB/s, PLUS one fewer kernel
  dispatch per fusable input (3/layer x 40 = 120 fewer dispatches/token -- decode is dispatch-bound, doc 04).
- **First kernel to write:** `silu_and_mul_quant_int8` (the only genuinely missing kernel). For RMSNorm,
  no kernel is needed -- just wire the existing fused op + a model-patch. Do both; the wiring is the gate.

---

## a. Analysis of the CURRENT quant + epilogue path

### a.1 The producer -> quant -> GEMM chain today [VERIFIED]

For each int8 Linear, `XPUInt8ScaledMMLinearKernel.apply_weights(layer, x, bias)` receives `x` as the
**fully materialized f16 output** of the preceding op, then:

```
x_q, x_s, _ = torch.ops._xpu_C.dynamic_per_token_int8_quant(x_2d, True, 8)   # standalone, re-reads x
out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, x.dtype)
```

The standalone quant (`csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp`) is a clean 2-pass kernel
(1 work-group == 1 sub-group of size 32 per row; pass1 absmax via `reduce_over_group`, pass2 quantize),
but it operates on an activation that some OTHER kernel already wrote to DRAM. The producers:

```
input_layernorm (RMSNorm)        -> qkv_proj   (int8 linear)   K = H = 5120
post_attention_layernorm (RMSNorm)-> gate_up_proj (int8 linear) K = H = 5120
silu_and_mul                     -> down_proj  (int8 linear)   K = I = 17408
```

(o_proj's input comes from attention output -- no norm/act producer to fuse into; skip for now.)

### a.2 Global-memory pass accounting (the waste) [VERIFIED by byte math]

Per fusable input of width K, on the activation tensor (f16 = 2 B/elt, int8 = 1 B/elt):

```
              | producer writes | quant reads | quant writes | total f16 passes | int8 write
 STANDALONE   |  y_f16 (1 x 2K) | y_f16 (2x 2K)|  q_i8 (1x K) |   3 f16 touches  |   yes
 FUSED        |   (none)        |  x_f16 (3x 2K, recompute)    |   the f16 y      |   yes
              |                 |              |  q_i8 (1x K) | round-trip GONE  |
```

The fused kernel still reads its *input* x (the residual-stream / gate||up) 3x because it recomputes
norm_x/act twice (variance, absmax, quantize) -- K=5120/17408 does not fit per-lane registers. But the
**intermediate f16 `y` (the normed/activated tensor) is never written or re-read** -- that whole
round-trip is the win.

Byte savings per token (Qwen3-14B), eliminating the f16 `y` materialization + re-read:

```
 input                         K       standalone   fused     saved     ratio
 qkv  (RMSNorm->qkv)          5120       35840 B    5120 B    30720 B    7.0x
 gate_up (RMSNorm->gate_up)   5120       35840 B    5120 B    30720 B    7.0x
 down (silu_mul->down)       17408      121856 B   17408 B   104448 B    7.0x
 ------------------------------------------------------------------------------
 per layer (3 inputs)                                        165888 B
 per token (x40 layers)                                     ~6.3 MiB  = ~10.9 us/token @ 608 GB/s
```

(The "standalone" column counts producer-write + 2 quant-reads + int8-write; "fused" counts only the
int8-write, since the fused op's input read is the same residual-stream read the producer already did.)

Plus dispatch: standalone = (producer kernel) + (quant kernel) = 2 launches per input; fused = 1. That
is **120 fewer kernel launches per token** across 40 layers. Doc 04 measured decode is **dispatch-bound**
(graph capture alone gave +187% on w4a8); cutting launches compounds with the PIECEWISE graph win.

### a.3 The dequant EPILOGUE (s32 -> f16) -- is it optimal? [VERIFIED: yes, already correct]

`int8_gemm_w8a8.h` does exactly what doc 04 prescribes:
- per-token (per-M) src scale via **runtime `DNNL_ARG_SRC` scales with a per-row mask**
  (`set_scales(DNNL_ARG_SRC, mask=(1<<0)|(1<<1), {1,k})`), NOT a per-N binary post-op. CORRECT -- a
  binary post-op would broadcast per-N and mis-scale per-token.
- per-channel (per-N) weight scale via `set_scales(DNNL_ARG_WEIGHTS, mask=1<<1)`.
- symmetric -> **no src/weight zero-points** (the w4a8 zp waste, doc 04 B1, is already absent here).
- oneDNN fuses the int32->f16 dequant (src_scale x wei_scale) into the matmul epilogue internally; the
  result is written once as f16. There is **no separate dequant kernel** -- nothing to fuse away.

**Verdict: the epilogue is already optimal. Do not touch it.** The only epilogue-adjacent nuance is the
src-scale dtype (a.4). The fusion levers are entirely on the PROLOGUE (quant) side.

### a.4 Numerics flags found while reading [VERIFIED -- two real discrepancies]

1. **int8 clamp range mismatch.** The fused `rms_norm_dynamic_per_token_quant` int8 path clamps to
   **[-128, 127]** (layernorm_quant.cpp:123) and its reference test agrees (test_fused_norm_quant.py:68).
   But the standalone `dynamic_per_token_int8_quant` clamps to **[-127, 127]** symmetric (the convention
   the s8s8s32 symmetric per-token GEMM dequant assumes: scale = absmax/127). Emitting -128 with a
   127-level symmetric scale dequantizes the most-negative saturated values ~128/127 (~0.8%) too large.
   **Fix: clamp the fused int8 path to [-127, 127]** to match the standalone quant and the GEMM contract.
   (Low magnitude, but it is a real contract break; trivial one-line fix.)
2. **scale floor mismatch.** Standalone uses `scale = max(absmax/127, 1e-5)`. The fused kernel uses
   `scale = (absmax>0) ? absmax/127 : 1.0` (no 1e-5 floor). Align to `max(absmax/127, 1e-5)` so an
   all-zero/denormal row cannot produce a div-by-tiny inv_scale. Same one-line region as fix 1.

### a.5 Scale DTYPE: f32 vs f16 [VERIFIED -- f32 works, no cast needed]

The fused kernels write `scales` as **torch.float32** (hardcoded; TORCH_CHECK at layernorm_quant.cpp:1002).
The standalone quant writes scale in the *input* dtype (f16/bf16). Does the f32 scale reach the GEMM?
`int8_gemm_w8a8.h` sets the src-scale dtype dynamically via `get_onednn_dtype(m1_sc)` (line 85/92), and
`get_onednn_dtype` maps torch.float32 -> `memory::data_type::f32` (onednn_ext.h:1664). oneDNN
`set_scales(DNNL_ARG_SRC, ..., f32)` is supported. **So a f32 per-token scale from the fused kernel feeds
the GEMM directly -- no extra cast op.** Caveat: the primitive-cache key `sc_group_size = (m1_numel<<8) |
m2_numel` is dtype-agnostic, so if a process ever mixed f16-scale and f32-scale calls at the same M/N/K it
could reuse a stale primitive -- not a problem in practice (one path per process), but pin the scale dtype
to f32 everywhere to be safe.

---

## b. Ranked fusion levers (expected savings + WHY)

| # | Lever | Status | Saves / token (14B) | Risk | Why |
|---|---|---|---|---|---|
| L1 | Wire existing `rms_norm_dynamic_per_token_quant` (int8) into qkv + gate_up inputs | kernel VERIFIED-exists; WIRING to write | 2 inputs x (30720 B + 1 dispatch) x 40 = ~2.4 MiB + 80 launches | LOW | kernel + test already exist; pure Python model-patch; biggest bang/effort |
| L2 | WRITE `silu_and_mul_quant_int8` (per-token) + wire into down_proj | kernel + wiring to write | ~104448 B + 1 dispatch, x40 = ~4.1 MiB + 40 launches | MED | down_proj K=17408 is the FATTEST input -> largest single saving; no kernel exists |
| L3 | Fix int8 clamp [-128,127]->[-127,127] + scale floor in fused kernel | one-line C++ | correctness, 0 perf | LOW | matches GEMM symmetric contract (a.4) |
| L4 | Pin fused-kernel work-group to `reqd_sub_group_size(16)` (Xe2 native) | one-line C++ | small BW (avoid SIMD32 split) | LOW | doc 06: Battlemage native SG=16; current kernel lets compiler pick |
| L5 | Vectorize the absmax/quant K-loop in the fused kernels (vec8 of f16) | C++ | BW toward peak on the 3 reads | LOW | aligned vec loads already used in act_and_mul_vec_kernel; mirror it |

Ranked order to execute: **L1 (wire RMSNorm) -> L2 (write+wire silu int8) -> L3/L4 (numerics+SG, fold
into L1/L2 commits) -> L5 (BW polish)**. L1+L2 capture ~6.3 MiB/token + 120 launches; L3-L5 are cheap
correctness/BW riders on the same edits.

WHY this is the right lever set (from doc 04 / doc 06):
- oneDNN **cannot** fuse f16->int8 into the GEMM prologue (src must be pre-quantized s8). So the ONLY
  place to kill the round-trip is the producing op. VERIFIED against int8_gemm_w8a8.h (it TORCH_CHECKs
  s8 src).
- The dequant epilogue is already the doc-04-correct per-M `DNNL_ARG_DST`/`DNNL_ARG_SRC` runtime-scale
  form -- no epilogue lever remains.
- Decode is dispatch-bound (doc 04); each fusion removes a launch AND a DRAM round-trip -- both axes.

---

## c. DRAFT SYCL skeletons (codex-assisted, NOT compiled)

### c.1 rms_norm_quant -- ALREADY EXISTS, do not rewrite [VERIFIED]

The kernel in `csrc/layernorm_quant.cpp::rms_norm_dynamic_per_token_quant_kernel<scalar_t, int8_t,
has_residual>` is exactly the fused design (3 in-register passes, no f16 `y` round-trip):

```
 pass1: (optional residual add, write residual back in scalar_t) + accumulate variance;
        reduce_over_group(plus); inv_rms = rsqrt(var/H + eps) stored in SLM.
 pass2: recompute norm_x = x*inv_rms*weight[i] in regs; absmax; reduce_over_group(max);
        scale = absmax/127 (int8); store scales[token] (f32).
 pass3: recompute norm_x; q = clamp(rint(norm_x/scale)) -> int8.
```

Only edits needed (L3/L4): clamp `[-128,127]` -> `[-127,127]`, add `max(.,1e-5)` scale floor, and pin
`[[sycl::reqd_sub_group_size(16)]]` on the operator (it currently launches `block_size=min(H,1024)`
work-items/token = multi-subgroup full-group reduce, which is the right shape; just pin SG=16 for Xe2).
**No new kernel.**

### c.2 silu_and_mul_quant_int8 -- WRITE THIS (DRAFT) [PROPOSED]

New file `csrc/quantization/fused_kernels/fused_silu_mul_int8_quant.cpp` (or extend activation.cpp).
Input `[num_tokens, 2*d]` (gate||up contiguous halves), output int8 `[num_tokens, d]` + f32
`scales[num_tokens]`. One work-group per token, WG=256 (= 16 native sub-groups on Xe2), vec8 f16 loads,
two passes over gate/up (d=17408 too big to cache in SLM). DRAFT body:

```cpp
// members: const sycl::half* x;  // [num_tokens, 2*d] gate||up
//          int8_t* q;            // [num_tokens, d]
//          float*  scales;       // [num_tokens]
//          int d;
// launch:  nd_range<1>(num_tokens * WG, WG), WG = 256
[[sycl::reqd_sub_group_size(16)]]
void operator()(sycl::nd_item<1> it) const {
  constexpr int V = 8;
  constexpr float kInv127 = 1.0f / 127.0f;
  constexpr float kMinScale = 1.0e-5f;
  auto grp = it.get_group();
  const size_t token = grp.get_group_id(0);
  const size_t lid = it.get_local_linear_id();
  const size_t lsz = it.get_local_linear_range();
  const size_t D = (size_t)d;

  const sycl::half* gate = x + token * (2 * D);
  const sycl::half* up   = gate + D;
  int8_t* qrow = q + token * D;

  auto silu_mul = [](float g, float u) {
    float sig = 1.0f / (1.0f + sycl::exp(-g));
    return (g * sig) * u;
  };

  // pass1: silu(gate)*up in regs -> per-row absmax (vec8, tail-safe)
  float local_abs = 0.0f;
  for (size_t i = lid * V; i < D; i += lsz * V) {
    if (i + V <= D) {
      sycl::vec<sycl::half, V> gv, uv;
      gv.load(0, gate + i); uv.load(0, up + i);
      #pragma unroll
      for (int j = 0; j < V; ++j) {
        float a = silu_mul((float)gv[j], (float)uv[j]);
        local_abs = sycl::fmax(local_abs, sycl::fabs(a));
      }
    } else {
      for (size_t j = i; j < D; ++j) {
        float a = silu_mul((float)gate[j], (float)up[j]);
        local_abs = sycl::fmax(local_abs, sycl::fabs(a));
      }
    }
  }
  float absmax = sycl::reduce_over_group(grp, local_abs, sycl::maximum<float>());
  float scale = sycl::fmax(absmax * kInv127, kMinScale);   // matches standalone quant
  float inv_scale = 1.0f / scale;
  if (lid == 0) scales[token] = scale;                      // f32, feeds GEMM directly (a.5)

  // pass2: recompute act, quantize to [-127,127] symmetric
  for (size_t i = lid * V; i < D; i += lsz * V) {
    if (i + V <= D) {
      sycl::vec<sycl::half, V> gv, uv;
      gv.load(0, gate + i); uv.load(0, up + i);
      #pragma unroll
      for (int j = 0; j < V; ++j) {
        float a = silu_mul((float)gv[j], (float)uv[j]);
        int qi = (int)sycl::rint(a * inv_scale);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qrow[i + j] = (int8_t)qi;
      }
    } else {
      for (size_t j = i; j < D; ++j) {
        float a = silu_mul((float)gate[j], (float)up[j]);
        int qi = (int)sycl::rint(a * inv_scale);
        qi = qi < -127 ? -127 : (qi > 127 ? 127 : qi);
        qrow[j] = (int8_t)qi;
      }
    }
  }
}
```

Register + Python wrapper (mirror silu_and_mul_quant):
```
// torch_bindings.cpp (_C):
ops.def("silu_and_mul_quant_int8(Tensor! out, Tensor input, Tensor! scales) -> ()");
ops.impl("silu_and_mul_quant_int8", torch::kXPU, &silu_and_mul_quant_int8);
```
```python
# register_ops.py
def silu_and_mul_quant_int8(out, input, scales):  # out int8 [T,d], scales f32 [T]
    torch.ops._C.silu_and_mul_quant_int8(out, input, scales)
```

DRAFT iteration notes (codex): WG=256 -> bench 128 vs 256, 512 only if token-count is tiny/latency-bound;
do NOT cache act in SLM for d=17408; reduce_over_group over the full multi-subgroup WG is fine on Xe2.

---

## d. Wiring the fused output into the int8 linear input (the model-patch points)

CUDA does this with torch.compile Inductor passes `fuse_norm_quant`
(`vllm/compilation/passes/fusion/rms_quant_fusion.py`) and `fuse_act_quant`
(`.../act_quant_fusion.py`) -- pattern-match `rms_norm(x)->quant(y)` / `silu_and_mul(x)->quant(y)` and
rewrite to the fused op. **These are CUDA/HIP-gated and DO NOT fire on XPU eager** [VERIFIED via vLLM
docs/design/fusions]. So we wire explicitly with a model-patch (applied at image-bake via
`contrib/vllm_int8_xpu/apply_patches.py`).

**Strategy (recommended): fuse the PRECEDING op into the linear's input, model-side.** Two sub-options:

- (a) Push quant up into a patched RMSNorm/Act that emits int8+scale. REJECTED: RMSNorm does not know its
  consumer is int8 (could be a non-quantized head/router), needs per-call-site knowledge, fragile.
- (b) Patch the decoder layer / MLP forward to call a fused helper that runs the fused op and hands
  (x_q, x_s) straight to `int8_gemm_w8a8`, bypassing the standalone quant. RECOMMENDED: localized to the
  model's `forward`, the int8-ness is known there (the linear is a `CompressedTensorsW8A8Int8`), residual
  handling is explicit, and a clean fallback exists (if the next linear is not our int8 kernel, keep the
  plain RMSNorm/silu path).

### d.1 RMSNorm -> qkv/gate_up [PROPOSED]

Patch `Qwen3DecoderLayer.forward` (and a shared helper) so the layernorm that feeds an int8 linear calls
the fused op. Skeleton:

```python
def _rmsnorm_int8(x, weight, eps, residual):
    # returns (x_q int8 [T,H], x_s f32 [T,1], residual_updated)
    T = x.numel() // x.shape[-1]
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(T, dtype=torch.float32, device=x.device)
    res = residual.contiguous() if residual is not None else None
    torch.ops._C.rms_norm_dynamic_per_token_quant(out, x, weight, scales, eps, None, res)
    return out.view(-1, x.shape[-1]), scales.view(-1, 1), res

# in Qwen3DecoderLayer.forward, when self_attn.qkv_proj uses our int8 kernel:
x_q, x_s, residual = _rmsnorm_int8(hidden_states, self.input_layernorm.weight,
                                   self.input_layernorm.variance_epsilon, residual)
hidden_states = self.self_attn(x_q=x_q, x_s=x_s, ...)   # qkv_proj consumes pre-quantized input
```

The int8 linear gets a new fast entry that ACCEPTS pre-quantized input:
```python
# XPUInt8ScaledMMLinearKernel
def apply_weights_prequant(self, layer, x_q, x_s, bias=None):
    w_q, w_s, *_ = self._get_layer_params(layer)
    out = torch.ops._xpu_C.int8_gemm_w8a8(x_q, x_s, None, w_q, w_s, None, bias, self._out_dtype)
    return out
```
Keep the original `apply_weights(layer, x, bias)` (standalone quant) as the fallback for call sites that
were not patched (o_proj, lm_head, non-Qwen models).

### d.2 silu_and_mul -> down_proj [PROPOSED]

Patch `Qwen3MLP.forward` so the activation that feeds down_proj uses the new int8 fused op:
```python
def forward(self, x):
    gate_up, _ = self.gate_up_proj(x)            # itself an int8 linear (d.1 wires its INPUT)
    T, d = gate_up.shape[0], gate_up.shape[-1] // 2
    x_q = torch.empty((T, d), dtype=torch.int8, device=gate_up.device)
    x_s = torch.empty(T, dtype=torch.float32, device=gate_up.device)
    torch.ops._C.silu_and_mul_quant_int8(x_q, gate_up, x_s)        # the NEW kernel (c.2)
    return self.down_proj.apply_weights_prequant(x_q, x_s.view(-1, 1))
```

### d.3 Per-token scale shape to the GEMM [VERIFIED]

`int8_gemm_w8a8.h` selects per-token vs per-tensor by `m1_sc.numel()` (==1 -> per-tensor, else per-token
with mask `(1<<0)|(1<<1)`, dims `{1,k}`). The fused ops give `scales` of shape `[T]`; reshape to `[T,1]`
(numel = T = M) so the GEMM takes the per-token branch. Dtype f32 is accepted as-is (a.5).

### d.4 BUILD / packaging gotcha [VERIFIED]

The fused ops live in the **`_C` extension** (`TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` ->
`torch.ops._C.*`), built under `BASIC_KERNELS_ENABLED` (CMakeLists.txt:509-522, includes
`layernorm_quant.cpp` + `activation.cpp` + `fused_kernels/`). The minimal build `scripts/44` sets
`BASIC_KERNELS_ENABLED=OFF` and only builds `_xpu_C` (the GEMM/quant namespace). **To ship the fused
path, flip `BASIC_KERNELS_ENABLED=ON` in scripts/44** (it builds the `_C` extension; adds minutes). The
new `silu_and_mul_quant_int8` source goes under the BASIC list too. Verify with
`hasattr(torch.ops._C, "rms_norm_dynamic_per_token_quant")` and `..."silu_and_mul_quant_int8"`.

### d.5 Validation plan (CPU-buildable; GPU run gated via scripts/gpu-run)

1. Numerics: extend `tests/test_fused_norm_quant.py` int8 ref to clamp [-127,127] (matching L3); add a
   `test_fused_silu_mul_int8` mirroring `test_fused_silu_mul_block_quant.py` but per-token symmetric.
   Compare fused-(rmsnorm|silu)+quant against (plain op -> standalone quant) -- expect bit-near int8.
2. End-to-end: serve Qwen3-14B-W8A8-INT8-gptq on `:int8` with the patch, confirm
   `Selected XPUInt8ScaledMMLinearKernel`, run a HumanEval+ smoke (verify served id is the *gptq* dup per
   CLAUDE.md), and a decode-t/s probe vs the standalone-quant baseline (doc 02: 22.6 t/s). Expect the
   ~10.9 us/token + 120-launch saving to lift decode toward FP8's ~29; prefill should be neutral-to-up.
   All GPU touches via `scripts/gpu-run`.

---

## e. Verified source-pointer table (VERIFIED = read/confirmed; PROPOSED = to write)

### Our code on the host (`/mnt/vm_8tb/b70/vllm-xpu-kernels`)

| Path | What | Status |
|---|---|---|
| `csrc/layernorm_quant.cpp:27` (kernel), `:988` (entry), `:1034` (int8) | fused rmsnorm + per-token int8/fp8 quant; handles residual | VERIFIED exists+registered |
| `csrc/torch_bindings.cpp:40` | `rms_norm_dynamic_per_token_quant` registered into `_C` | VERIFIED |
| `csrc/activation.cpp:189,552` | `silu_and_mul_quant` = FP8-only + STATIC per-tensor scale | VERIFIED (not usable for int8) |
| `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cpp` | silu+mul int8 but PER-COLUMN-GROUP (g=64/128), not per-token | VERIFIED (wrong granularity) |
| `csrc/xpu/sycl/dynamic_per_token_int8_quant.cpp` | standalone per-token int8 quant (2-pass, clamp -127..127) | VERIFIED (the op to displace) |
| `csrc/xpu/onednn/int8_gemm_w8a8.h:80-111` | per-token `DNNL_ARG_SRC` + per-channel `DNNL_ARG_WEIGHTS` runtime scales, symmetric, dtype via `get_onednn_dtype` | VERIFIED epilogue-correct |
| `csrc/xpu/onednn/onednn_ext.h:246,1664` | `get_onednn_dtype` maps f32 -> `memory::data_type::f32` | VERIFIED (f32 scale OK) |
| `CMakeLists.txt:509-522` | `BASIC_KERNELS_ENABLED` builds `layernorm_quant.cpp`+`activation.cpp`+`fused_kernels/` (the `_C` ext) | VERIFIED |
| `tests/test_fused_norm_quant.py:141`, `tests/test_fused_silu_mul_block_quant.py` | reference tests (int8 clamps -128..127) | VERIFIED |
| `csrc/quantization/fused_kernels/fused_silu_mul_int8_quant.cpp` | NEW per-token int8 silu+mul kernel (c.2) | PROPOSED |

### Our repo (dev box)

| Path | What |
|---|---|
| `contrib/vllm_int8_xpu/xpu_int8.py:112-116` | int8 linear apply_weights -- calls STANDALONE quant (wiring target d.1) |
| `scripts/44_build_int8_kernel.sh:17-19` | flips `BASIC_KERNELS_ENABLED=OFF` -> must be ON for fused ops (d.4) |
| `docs/kernel/04_decode_optimization.md` (Lever B3) | the mission spec for this fusion |
| `docs/literature/06_xpu_kernel_fastpaths.md` | Xe2 SG=16, native s8s8s32, oneDNN fusion limits |

### Upstream (URLs)

| URL | What |
|---|---|
| https://docs.vllm.ai/en/latest/design/fusions/ | `fuse_norm_quant` + `fuse_act_quant` passes; eliminate intermediate f16; CUDA/HIP-gated (NOT XPU eager) |
| https://github.com/vllm-project/vllm/pull/10906 | original "Dynamic fp8 + rms_norm fusion" pass + `layernorm_utils.cuh` (algorithm template) |
| https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_quant_kernels.cu | CUDA fused rms_norm+quant kernels (the SYCL port's origin) |
| https://github.com/vllm-project/vllm/issues/33026 | "fused silu_mul + block-wise quant Triton" -- confirms upstream act+quant int8/per-token still open |
| https://pytorch.org/blog/portable-vllm-model-inference-kernels-in-helion/ | `silu_and_mul_dynamic_per_token_quant` named as a fused single-launch kernel (template) |

### Honesty flags

- The SYCL `silu_and_mul_quant_int8` skeleton (c.2) and all model-patch code (d) are DRAFT -- not compiled,
  not run on the B70. Numerics + WG sizing must be validated (d.5) before trusting.
- Byte-savings (a.2) and us/token (10.9) are accounting at peak BW (608 GB/s); real decode gain is bounded
  by how dispatch-bound the path is post-graph-capture -- measure, do not quote the us as a speedup.
- The fused rmsnorm kernel EXISTS but has never been wired/run in our serving path -- "exists+registered"
  is VERIFIED from source; "works end-to-end in our model" is UNVERIFIED until d.5 runs.
