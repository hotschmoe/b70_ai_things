# research/LESSONS.md -- the cross-cutting optimization ledger

The shared-learnings layer. `research/<theme>/` is the yolo scratchpad (probe hard, discard,
move on); when something REAL falls out, distill it here as a one-row lesson so it propagates
to similar quants and the other backend instead of getting siloed.

## How to use this

- One row per durable optimization/insight. Keep it short: what + where it generalizes + proof.
- `generalizes to` is the point: tag the bit-width / scheme / backend it should transfer to.
  A lesson found on w4a8 that is really about int4 *weights* also applies to w4a16.
- `bench status` is the propagation tracker: list the `(model, quant, backend)` configs the
  lesson has been measured on (uplift/regression), and which still need a re-bench.
- Promoting a lesson into a shelf config -> run `bin/serve-sweep --bench` over the affected
  entries, record deltas, and only update `rdy_to_serve/<backend>/...` if faster AND coherent.

## Ledger

| # | lesson | applies to (origin) | generalizes to | bench status |
|---|--------|---------------------|----------------|--------------|
| 1 | Fused int8 oneDNN gemm: `int8_gemm_w8a16` (decode, fp16-act) + `int8_gemm_w8a8` (prefill, s8-act) beats woqgemm on PP/TTFT/TG. | w8a8 / sglang (27b) | any int8-weight path; both backends (vLLM via `:int8g`) | sglang 27b w8a8: decode ~25.2 t/s, HumanEval+ 0.97/0.93 (PASS). vLLM int8g: measured slower (~9 eager). 35b-a3b w8a8 MoE now serves on sglang too (Route A, eager ~8 t/s; see row below). |
| 2 | Hybrid W4A8 kernel: decode `int4_gemm_w4a16` (fp16 act) / prefill `int4_gemm_w4a8` (s8 act) beats int4-woqgemm on decode (1.83x) and prefill (1.9x). | w4a8 / sglang (27b) | int4-weight paths incl. w4a16; vLLM w4a8 | sglang 27b w4a8: decode ~27.3 t/s (PASS). TODO: re-bench vLLM w4a8 + w4a16 with the hybrid idea. |
| 3 | int4 lm_head (LMHEAD=1, group 32) holds accuracy and adds ~+8% decode vs bf16 lm_head. | w4a8 / sglang (27b) | any int4-weight model (w4a16, 35b int4); check both backends | sglang 27b w4a8: +8% (PASS, accuracy held). TODO: try on w4a16 + 35b-a3b int4. |
| 4 | Vision tower + MTP head must be grafted back into stripped compressed-tensors quants (333 `visual.*` + `mtp.*`), GPU-free. | w8a8/w4a8/w4a16 (model prep) | every compressed-tensors quant of a VLM, both backends | done for 27b w8a8 (shelf) + w4a16/w4a8 (models/graft_w4_complete.sh). UNVERIFIED on-GPU for w4a16/w4a8. |
| 5 | MTP chain-depth (`--speculative-num-steps`) peak is PER-QUANT: w8a8=10, int4=7. Cheaper int8-XMX verify lets deeper drafts net positive; int4's costlier verify peaks shallower. Do NOT copy a steps value across quants. | w8a8 + int4 / sglang | any NEXTN spec-decode entry -- tune steps per quant, do not transfer | w8a8 steps=10 (25.25), int4 steps=7 (15.31). Both shelf entries already at their own peak. |
| 6 | Custom XPUGraph decode capture is single-stream-ONLY: great at bs1/maxreq1 (int4 23.5), COLLAPSES under concurrency (maxreq4 -> 3.55, c4 -> 0.75). For a SERVING (concurrent) shelf entry, prefer MTP-eager over graph-capture. | int4/w8a8 graph sprint / sglang | any graph-capture entry served concurrently; both backends' capture paths | w8a8 chose MTP-eager (correct). FLAG: sglang w4a8 entry is graph/maxreq=1 -- expect it to degrade under the sweep's concurrent (c4) probe; candidate to re-home to an MTP config. |
| 7 | W8A8 int8 decode GEMM (`int8_gemm_w8a16`, fp16-act, 1 launch) is AT the BW read roofline and bit-identical in time to FP8 -- there is NO W8A8-vs-FP8 decode gap at the GEMM level. A reordered/VNNI16 small-M GEMV (llama.cpp #21527 3.1x) does NOT transfer: that win beat a sub-roofline baseline; ours is already 92-98% of 581 GB/s. | w8a8 / vLLM 0.25.1 (27b, TP=2) | any int8-weight decode path, both backends; sets the ceiling for FP8 too | vLLM 27b w8a8: 92-98% of 581 GB/s == FP8 bar, real 27B shapes, eager + XPUGraph (research/w8a8/decode_gemv/bench_decode_gemv.py). RESEARCH_TODO Track 1a M=1-GEMV fast path = proven NO-GO, closed (JOURNAL 2026-07-21). |
| 8 | Route small M (<=64: decode + MTP verify batch) through the quant-free `int8_gemm_w8a16` op instead of the s8s8 two-step act-quant path -- same weight bytes, skips the per-token int8 activation quant. GEMM 1.47-1.49x over the current path, MATCHES FP8, and MORE ACCURATE (f16 act relerr 8.8e-3 vs s8 1.3e-2). | w8a8 / vLLM 0.25.1 (27b, TP=2) | any W8A8 apply path that act-quants at every M; check the sglang w8a8 int8 apply path too | vLLM 27b w8a8 TP=2 captured+MTP3: code decode c1 38.9->40.3 (+3.6%), c4 22.4->23.5 / agg 89.7->94.1 (+4.9%); coherent, gate 18/18 PASS. SHIPPED env-gated (B70_W8A16_M_MAX, default 0=OFF) in vllm/contrib/vllm_int8_xpu/xpu_int8.py + register_fake for PIECEWISE capture. GEMM 1.47x compresses to ~4% e2e (GDN scan / MTP drafter / push-AR dominate the step). JOURNAL 2026-07-21. |
| 9 | TP=2 decode on the dense hybrid is ALL-REDUCE-BOUND (43% of device time; GEMM 39% is at roofline), and that all-reduce is the MTP spec-decode `vllm::all_gather` realized as an eager oneCCL SUM over PCIe (no Battlemage P2P) that BYPASSES the push-AR (which patches only XpuCommunicator.all_reduce). Moving it to the eager host-barrier push-AR is 2.4x SLOWER (CPU-spin barrier per call x ~631 gathers/step >> device saving); only the GRAPH-recorded push-AR (do_ar) is fast, so the real fix is CAPTURING the gather. | nvfp4/w8a8 / vLLM 0.25.1 (27b, TP=2) | any TP>1 spec-decode on XPU without P2P; the push-AR all_reduce override does NOT cover gather-internal all-reduces | traced (research/profiling/parse_trace.py); PP=2 NO-GO (no SupportsPP); PUSH_AR_ALLGATHER redirect NO-GO (48.9->20.7 t/s, kept default-off). Real fix = capture the MTP all_gather / upstream torch-xpu-ops#2992 (needs torch>=2.13). JOURNAL 2026-07-21 s2. |

## Propagation status -- 2026-06-29 pre-sweep audit

Audited the w8a8 sprint wins against every shelf entry before the bench:
- **General sglang serving flags** (`--page-size 64`, `--mamba-ssm-dtype float32`,
  `--disable-overlap-schedule`, `--disable-radix-cache`, `--skip-server-warmup`,
  `--attention-backend intel_xpu`): ALREADY on all sglang entries (int4 + w8a8). No-op.
- **Fused int8 oneDNN kernel + steps=10**: int8-weight + per-quant specific. Correctly NOT
  propagated to int4 (peak=7, already set). The shelf w8a8 already is the best config (no change).
- **CANDIDATES (new work -- must be bench/coherence-gated per the shelf rule, NOT jammed in pre-sweep):**
  1. int4 lm_head g32 (row 3) -> try on sglang `qwen36-27b-int4` (int4-autoround / woqgemm). Different
     kernel path than w4a8's `int4_gemm_w4a16`; needs its own +decode/accuracy measurement.
  2. Fused int8 SOURCE (`kernels/`) -> the vLLM `:int8g` image still ships the older
     `XPUInt8ScaledMMLinearKernel`, NOT today's fused hybrid. Rebuilding `:int8g` from the shared
     `kernels/` source would propagate the int8 win to vLLM w8a8. Image rebuild + vLLM is paused.
  3. sglang w4a8 (row 6) -> graph/maxreq=1 will likely tank the concurrent (c4) sweep probe;
     evaluate an MTP-based w4a8 (the w4a8 ckpt now has a grafted MTP head).
- **sglang W8A8 MoE (35b-a3b): DONE -- a LOADER port + a 1-line cuda.libdevice shim, NOT a kernel build**
  (PROVEN 2026-06-29, `research/w8a8/SGLANG_MOE_PLAN.md`, JOURNAL 2026-06-29; shelf
  `rdy_to_serve/sglang/qwen36-35b-a3b-w8a8`). Go/no-go ANSWERED: the in-tree Triton `use_int8_w8a8`
  fused_moe codegens + runs correctly on B70 (probe cosine 0.9998, int8 PREFILL 1.43x bf16). The ONE
  real kernel fix: stock `per_token_quant_int8` (int8_kernel.py) uses `tl.extra.cuda.libdevice.round`,
  which does NOT link on triton-xpu (`ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED`) -> replace with
  floor/ceil round-half-away (`sglang/patches/int8_actquant_xpu.py`, mirrors w4a8_actquant_triton.py).
  Plus loader plumbing: `quark_moe_int8.py` (Int8MoEMethod + dense dequant + QuarkConfig monkeypatch)
  installed in every process via the woq_shim `.pth` hook; and Quark's 1-D `[N]` weight scales need
  unsqueeze to `[N,1]` for sglang's MoE loader + a 1-D dense scale param (the GDN merged in_proj_*
  loaders strip ChannelQuantScaleParameter's reshape). Serves coherent + stable; decode eager-slow (~8
  t/s) -- graph/MTP/fused-dense are the decode follow-ups. GENERALIZES: any Quark int8 ckpt on sglang-XPU
  hits the same cuda.libdevice round + 1-D-scale issues; this is the recipe. (Per-expert dense
  `int8_gemm_w8a8` loop stays the correctness oracle; fused SYCL `is_w8a8` is the prefill endgame if
  Triton perf ever caps.) The llm-scaler image remains a DEAD END (no `_moe_C` int8 op).

<!-- Add rows as lessons land. Keep origin probes under research/<theme>/; winners -> rdy_to_serve/<backend>/. -->
