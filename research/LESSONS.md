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
| 1 | Fused int8 oneDNN gemm: `int8_gemm_w8a16` (decode, fp16-act) + `int8_gemm_w8a8` (prefill, s8-act) beats woqgemm on PP/TTFT/TG. | w8a8 / sglang (27b) | any int8-weight path; both backends (vLLM via `:int8g`) | sglang 27b w8a8: decode ~25.2 t/s, HumanEval+ 0.97/0.93 (PASS). vLLM int8g: measured slower (~9 eager). TODO: 35b-a3b w8a8 (no entry yet). |
| 2 | Hybrid W4A8 kernel: decode `int4_gemm_w4a16` (fp16 act) / prefill `int4_gemm_w4a8` (s8 act) beats int4-woqgemm on decode (1.83x) and prefill (1.9x). | w4a8 / sglang (27b) | int4-weight paths incl. w4a16; vLLM w4a8 | sglang 27b w4a8: decode ~27.3 t/s (PASS). TODO: re-bench vLLM w4a8 + w4a16 with the hybrid idea. |
| 3 | int4 lm_head (LMHEAD=1, group 32) holds accuracy and adds ~+8% decode vs bf16 lm_head. | w4a8 / sglang (27b) | any int4-weight model (w4a16, 35b int4); check both backends | sglang 27b w4a8: +8% (PASS, accuracy held). TODO: try on w4a16 + 35b-a3b int4. |
| 4 | Vision tower + MTP head must be grafted back into stripped compressed-tensors quants (333 `visual.*` + `mtp.*`), GPU-free. | w8a8/w4a8/w4a16 (model prep) | every compressed-tensors quant of a VLM, both backends | done for 27b w8a8 (shelf) + w4a16/w4a8 (models/graft_w4_complete.sh). UNVERIFIED on-GPU for w4a16/w4a8. |
| 5 | MTP chain-depth (`--speculative-num-steps`) peak is PER-QUANT: w8a8=10, int4=7. Cheaper int8-XMX verify lets deeper drafts net positive; int4's costlier verify peaks shallower. Do NOT copy a steps value across quants. | w8a8 + int4 / sglang | any NEXTN spec-decode entry -- tune steps per quant, do not transfer | w8a8 steps=10 (25.25), int4 steps=7 (15.31). Both shelf entries already at their own peak. |
| 6 | Custom XPUGraph decode capture is single-stream-ONLY: great at bs1/maxreq1 (int4 23.5), COLLAPSES under concurrency (maxreq4 -> 3.55, c4 -> 0.75). For a SERVING (concurrent) shelf entry, prefer MTP-eager over graph-capture. | int4/w8a8 graph sprint / sglang | any graph-capture entry served concurrently; both backends' capture paths | w8a8 chose MTP-eager (correct). FLAG: sglang w4a8 entry is graph/maxreq=1 -- expect it to degrade under the sweep's concurrent (c4) probe; candidate to re-home to an MTP config. |

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
- **sglang W8A8 MoE (35b-a3b): BLOCKED, not a serve-script edit.** Only a Quark-format W8A8 35b
  exists (sglang has no Quark loader) AND sglang/XPU has no int8 fused-MoE kernel (256 experts have
  no int8 path; `_get_recipe` lacks an int8 branch). The benchable int8 MoE is the EXISTING vLLM
  Quark entry (`rdy_to_serve/vllm/qwen36-35b-a3b-w8a8`, true int8 experts via Triton fused-MoE).
  Unblocking sglang = porting the vLLM QuarkMoEMethod + building the fused grouped-int8 kernel
  (the "single missing W8A8 flag" in sycl-tla grouped_gemm). Real kernel/loader project.

<!-- Add rows as lessons land. Keep origin probes under research/<theme>/; winners -> rdy_to_serve/<backend>/. -->
