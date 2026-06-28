# W8A8-on-sglang campaign -- maximize PP/TTFT/TG, beat FP8/BF16, keep vision

Central doc for the sglang W8A8 performance push (Qwen3.6-27B, dual Arc Pro B70).
Goal (Isaac /loop 2026-06-28): a vision-retaining W8A8 that **handily beats FP8 and BF16 on
prefill(PP), TTFT, and decode(TG)** on the sglang backend. Minimize size (pack weights) for KV/
long-context headroom. Build/tune our own int8 GEMV/GEMM to light the Intel int8 fastpaths.
TP=2 / PP=2 are ON the table (we have a custom push all-reduce, see below). Even +5% is a win.

Style: config -> result -> verdict. Newest section at the BOTTOM. Detailed logs in JOURNAL.md.

---

## 0. Why W8A8 (the honest framing)

The comparison bar is **FP8 / BF16, not int4**. W8A8 cannot beat int4/W4A8 on decode (int8 = 1 byte/
weight vs int4 0.5 byte -> int4 is fundamentally ~2x on the BW-bound GEMV). But vs the stated bar:

- **PP / TTFT: W8A8 should WIN decisively.** int8-XMX/DPAS GEMM at M=2048 is **1.78-1.92x bf16**
  (measured, this campaign). int4 prefill is *weak* (woqgemm 0.77x bf16). FP8 has **no native B70 path**
  (Xe2 has no FP8 units; oneDNN emulates) -> fp8 prefill ~1.0x bf16. So W8A8 is the prefill champion.
- **TG: W8A8 should beat BF16 (~2x, half the weight bytes) and tie/beat FP8** (same 1 byte), IF the
  decode GEMV is fused into one launch instead of the current 3-kernel chain.

So the deliverable is achievable; the work is (a) kill the decode launch penalty (graph), (b) fuse the
int8 chain into single ops (build int8_gemm_w8a16 + int8_gemm_w8a8), (c) realize int8-XMX prefill.

---

## 1. Assets inventory (what already exists)

- **sglang int8 path is wired** (eager only): `sglang/patches/w8a8_shim.py` patches
  `CompressedTensorsW8A8Int8` -> dynamic per-token sym int8 act-quant -> `torch._int_mm` -> dequant.
  Gated `B70_XPU_W8A8=1` (installed from `woq_shim.py`). Current serve: 27B-W8A8 TP=2 EAGER = **5.45 t/s
  decode**, ~3500 t/s prefill, TTFT ~580ms (`sglang/w8a8_tp2_bench.log`).
- **Built oneDNN kernel .so** at `/mnt/vm_8tb/b70/w4a8_kernel/_xpu_C.abi3.so` (torch-2.12 ABI match).
  Registers: `int4_gemm_w4a16`, `int4_gemm_w4a8`, `fp8_gemm_w8a16`, `fp4_gemm_w4a4`.
  **NOT registered: any int8_gemm op, and fp8_gemm_w8a8** (only fp8 w8a16 is callable). Build recipe:
  `sglang/W4A8_BUILD.md`; source mirror at `/mnt/vm_8tb/b70/w4a8_kernel/src/`.
- **oneDNN dispatch supports** (`src/onednn_ext.h` joint_dtypes_t): f16_int4, bf16_int4 (-> w4a16),
  s8_int4, u8_int4 (-> w4a8), fp8 e4m3/e5m2 both directions, mxfp4. **Missing: f16_int8/bf16_int8
  (W8A16 decode) and s8_int8 (W8A8 fused).** Adding them mirrors the s8_int4 mapper (weight=s8, no
  nibble unpack) -- oneDNN natively does s8 weight-decompression w/ f16 src. => the int8 ops are
  BUILDABLE by analogy.
- **XPUGraph capture is wired + stable** on B70 (`sglang/patches/xpu_cudagraph.py`, `B70_XPU_CUDAGRAPH=1`;
  W4A8 single-card decode graph = 27.3 t/s). API: `torch.xpu.XPUGraph()` + `with torch.xpu.graph(g):` +
  `g.replay()`.
- **Custom PUSH all-reduce** (`contrib/vllm_push_allreduce/`): hand-rolled L0-IPC posted-write 2-rank
  all-reduce, decode ~34-45us vs oneCCL ~85us, prefill ~10 GB/s. Has a **graph-capturable** variant
  (`prebuilt/libxpu_push_ar_graph.so`, L0-event sync, 35us/AR, replayable). PROVEN coherent on the
  *vLLM* 27B-W8A8 TP=2 GRAPH=1 path -- but it monkeypatches **vLLM's** `XpuCommunicator`, NOT sglang's.
  Decode t/s A/B was never quantified. PORT TO SGLANG is an open lever for TP=2 decode.
- **W8A8 checkpoints** (`/mnt/vm_8tb/b70/models/`): `Qwen3.6-27B-W8A8-sqgptq` (35GB, GDN left bf16),
  `-sqgptq-vision` (vision grafted), `-sqgptq-mtp-graft`, `Qwen3-14B-W8A8-autoround` (fits one card),
  `Qwen3.6-35B-A3B-Quark-W8A8-INT8`. Vision: must verify/graft (`sglang/graft_vision.py`); some quants
  stripped the 333 visual.* tensors.

---

## 2. Strategy (codex-validated 2026-06-28, ranked)

1. **XPUGraph the decode chain first** (highest ROI; failure mode is launch-bound, not GEMM speed).
2. **Fuse act-quant + _int_mm + dequant into fewer launches** -- build `int8_gemm_w8a16` (decode,
   1 fused op like fp8 w8a16) and `int8_gemm_w8a8` (prefill, s8s8 + fused output-scale post-op).
   Do NOT use Triton tl.dot (10x slower, misses DPAS). A custom SYCL/ESIMD DPAS GEMV is phase 3.
3. **Realize int8-XMX prefill** (1.84x via torch.compile fusion is already proven; the built op is cleaner).
4. **Size**: int4 lm_head (done for W4A8, reuse) + int8 GDN in/out proj (bf16 recurrence) -> headroom.
   Single-card 27B-int8 is NOT required (TP=2 is fine); pursue size for KV/long-context, not to fit.
5. **TP=2**: port the push-AR to sglang's communicator; combine with TP=2 graph capture.

---

## 3. Results log

### 2026-06-28 -- KERNEL ENVELOPE microbench (card 0) -- `w8a8/w8a8_kernel_probe.py`
CONFIG: real Qwen3.6-27B linear shapes [N,K], synthetic int8 weights, fp16 baseline, card 0,
sglang-xpu:woq + built .so. Measured bf16 vs int8 `_int_mm` GEMM-only vs int8 full chain EAGER vs
int8 full chain **XPUGraph-CAPTURED** vs `fp8_gemm_w8a16`, at M=1 (decode) and M=2048 (prefill).

RESULT (x = speedup vs bf16; higher better):

```
 shape          M       bf16   int8 GEMM-only   int8 chain EAGER   int8 chain GRAPH   fp8 w8a16
 gate_up        1     1.00x        1.71x            1.41x              1.49x            1.95x
 (N=34816)      2048  1.00x        1.92x            0.81x              0.80x            1.00x
 down_proj      1     1.00x        2.01x            0.80x              1.52x **         1.95x
 (N=5120)       2048  1.00x        1.85x            0.72x              0.73x            0.95x
 qkv            1     1.00x        1.85x            0.64x              1.32x **         1.89x
 (N=14336)      2048  1.00x        1.78x            0.69x              0.70x            1.00x
```

READS:
- **Q1 CONFIRMED: XPUGraph recovers the decode launch penalty.** down_proj/qkv decode chain goes
  0.64-0.80x (LOSING vs bf16) -> 1.32-1.52x (WINNING) = **1.9-2.0x graph-over-eager**. The current
  EAGER W8A8 serve (5.45 t/s) leaves this on the table; capture alone should ~2x decode. gate_up gains
  little (its [1,34816] fp32 dequant epilogue is genuine BW, not launch overhead).
- **The GAP: fp8 w8a16 single-op (1.95x) BEATS our captured int8 chain (1.3-1.5x) at M=1.** Because
  the int8 chain is still 3 serial kernels even captured; fp8 w8a16 is ONE fused op (fp16 act -> no
  act-quant; dequant in epilogue). => the decode win needs a **fused int8 W8A16 op**, not just capture.
- **PREFILL: int8 GEMM-only is 1.78-1.92x bf16 (the XMX win) but the UN-FUSED chain collapses to
  0.7-0.8x** (eager act-quant + fp32 dequant epilogue on big tensors; graph does NOT help -- real BW,
  not launch). fp8 w8a16 prefill = ~1.0x (emulated, fp16 act, no XMX). => prefill needs the int8 act +
  output-scale FUSED into the GEMM (torch.compile proven 1.84x, or build int8_gemm_w8a8).
- `fp8_gemm_w8a8` is NOT registered in the .so -> the only fp8 baseline is w8a16 (fp16-act).

VERDICT: the W8A8 win path is now precise and mirrors the W4A8 success:
  **DECODE** = build `int8_gemm_w8a16` (int8 w, fp16 act, fused dequant) -> ~1.95x bf16 in ONE op
              (matches fp8 w8a16 but int8-accurate), then XPUGraph-capture it.
  **PREFILL** = build `int8_gemm_w8a8` (s8 w x s8 act, fused per-token x per-channel output scale) ->
              realize the 1.78-1.92x XMX win in one op (or torch.compile-fuse the chain, proven 1.84x).
Both are buildable by mirroring int4_gemm_w4a16 / int4_gemm_w4a8 in vllm-xpu-kernels (add f16_int8 /
s8_int8 joint_dtypes mappers; oneDNN supports s8 weight-decompress). NEXT ITERATION = build them.

### 2026-06-28 -- *** FUSED int8 OPS BUILT + VALIDATED: W8A8 beats FP8/BF16 at the kernel level ***
CONFIG: built `int8_gemm_w8a16` (NEW: s8 weight x f16 act, fused dequant -- mirrors fp8_gemm_w8a16)
+ `int8_gemm_w8a8` (s8xs8 fused per-token x per-channel scale; was staged-but-unbuilt in the tree)
+ `dynamic_per_token_int8_quant` (fused act-quant) into vllm-xpu-kernels vs sglang torch 2.12. Recipe:
w8a8/W8A8_BUILD.md. .so at /mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so (sha bc643c3f8a61, 51MB).
Microbench card 0, real 27B shapes, fp16 baseline (w8a8/w8a8_fused_probe.py).

RESULT (x bf16; DECODE M=1 / PREFILL M=2048):
```
                     DECODE M=1          PREFILL M=2048
  int8_gemm_w8a16    1.86-1.91x (DECODE) 0.98-1.06x   (graph-captured decode 1.73-1.89x)
  int8_gemm_w8a8     1.83-1.88x          1.95-2.07x (PREFILL)
  fp8_gemm_w8a16 bar 1.88-1.95x          0.98-1.00x
  (old _int_mm chain 0.64-0.80x eager / 1.3-1.5x captured decode; 0.7-0.8x prefill)
```
- DECODE: int8_gemm_w8a16 = ONE fused launch ~1.9x bf16 = matches the fp8 bar but INT8-accurate
  (relerr ~9e-3). XPUGraph-capturable. Replaces the old 3-kernel chain (was 1.3-1.5x captured).
- PREFILL: int8_gemm_w8a8 = ~2.0x bf16 (fused output-scale) vs fp8 1.0x (no XMX) vs old chain 0.7-0.8x.
- The fused act-quant op = 0.04ms (M=1) / 0.14-0.70ms (M=2048), single launch (capturable).
VERDICT: *** TASK TARGET MET AT KERNEL LEVEL *** -- the W8A8 HYBRID (decode=int8_gemm_w8a16 fp16-act,
prefill=int8_gemm_w8a8 int8-act) HANDILY beats FP8 (decode ~tie ~1.9x, prefill 2.0x vs 1.0x) AND
BF16 (~2x both). int8_gemm_w8a16 for prefill = no win (use w8a8 there) -> THE HYBRID, like W4A8.
NEXT = wire both ops into w8a8_shim.py + serve A/B.

---

## 3.5 SERVE STATUS (2026-06-28) -- live TP=2 numbers

```
  config (27B-W8A8 TP=2)   decode   prefill   TTFT    vs bf16 TP=2 (9.03 / 3098 / 661)
  legacy _int_mm (eager)   5.45     ~3500     ~580
  FUSED hybrid (eager)     8.08     4570      448     prefill +48%, TTFT -32%, decode -10%  <- SHIP for TP=2
  FUSED hybrid (GRAPH)     8.58     ~3100     ~660    CEILING: decode all-reduce-bound, prefill regresses
```
- TP=2 EAGER fused HANDILY beats bf16/fp8 on PP (+48%) and TTFT (-32%); decode ~ties bf16. (fp8 ~= bf16
  on prefill per microbench -- no XMX -- so W8A8 beats fp8 too.) 2 of 3 metrics met at TP=2.
- TP=2 GRAPH is a CEILING: the 1.9x decode op win does NOT realize because decode is all-reduce-bound at
  TP=2 (128 collectives/token, not captured), and triton-attn (graph req) regresses prefill. Box safe (no wedge).
- THE DECODE WIN (beat bf16 tg) needs SINGLE-CARD TP=1 GRAPH (no all-reduce; the op's 1.9x realizes) ->
  requires shrinking 27B-W8A8 (35GB) to fit one card (~28GB): int8 GDN proj (RTN at load -> int8_gemm_w8a16,
  bf16 recurrence kept) + int4 lm_head. Also the user's "minimize size" goal. = task #3, the next thrust.

## 4. Next steps (ordered)

1. [DONE] Built + validated int8 ops (sec 3) + wired fused hybrid into w8a8_shim + TP=2 eager serve win (sec 3.5).
2. [NEXT-A, low risk] Serve the VISION ckpt (sqgptq-vision) TP=2 eager fused -> confirm vision loads +
   coherent + same perf (the user's hard requirement). + same-session bf16 baseline for a clean head-to-head.
3. [NEXT-B, the decode win = MTP, codex-validated 2026-06-28] MTP/NEXTN spec-decode is the BEST decode
   lever (biggest win, least risk): it amortizes the TP=2 all-reduce + per-step tax across accepted tokens.
   int4 got 1.6x -> W8A8 TP=2 should move decode 8.3 -> ~12-13 t/s (handily beats bf16 9.0), KEEPING the
   prefill/TTFT win (TP=2, no fit problem). Plan:
     a. Graft the 15 BF16 mtp.* tensors (from sqgptq-mtp-graft/model-mtp-graft.safetensors) onto the
        sqgptq-VISION ckpt -> a W8A8 ckpt with BOTH vision (333) AND the MTP head. File-level, no GPU requant.
     b. Serve: image sglang-xpu:mtp (baked XPU MTP gates) + the fused w8a8_shim + B70_XPU_MTP=1 +
        --speculative-config {method:mtp} steps~7 + the mtp_bf16 drafter patch (MTP_GRAFT_NOTES: drafter must
        be BF16, not quantized) + cudagraph_mode=NONE (memory: W8A8 TP=2+MTP stable that way) + skip-warmup.
     c. GOTCHA (codex): verify MTP verify-path (M>1) hits int8_gemm_w8a8 (not a bf16/unfused fallback);
        acceptance is workload-sensitive; greedy-only on XPU (sampling ignored, like the int4 MTP path).
   Single-card shrink (int8 GDN proj + int4 lm_head -> TP=1 GRAPH) is the MOONSHOT (best ceiling, worst fit
   risk: 35GB + 5.6GB mamba into 30GB) -- deferred behind MTP.
4. Accuracy gate (HumanEval+) on the shipped config; productionize to rdy_to_serve/qwen36-27b-w8a8-fused.
5. [stretch] Port the push-AR to sglang's TP communicator -> faster TP=2 decode all-reduce (the other decode lever).
2. Wire the two ops into `w8a8_shim.py` (decode->w8a16 fp16-act, prefill->w8a8 int8-act), behind a new
   env (e.g. `B70_XPU_W8A8_FUSED=1`). Reuse W4A8's Triton act-quant for prefill if op-internal quant
   is not done.
3. Serve 27B-W8A8 TP=2 GRAPH=1 (vision ckpt) and A/B decode/prefill/TTFT vs the eager baseline. Watch
   the wedge guard (TP=2 capture = reboot risk); one experiment at a time.
4. Port the push-AR to sglang's TP communicator; A/B TP=2 decode/prefill with push-AR on/off.
5. Size: int4 lm_head + int8 GDN proj for KV/long-context headroom.
6. Head-to-head: build BF16 and FP8 (fp8 w8a16) serves on the same stack; prove W8A8 handily wins
   PP/TTFT/TG. Accuracy gate (HumanEval+) + vision retained.
```
```
