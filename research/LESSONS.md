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

<!-- Add rows as lessons land. Keep origin probes under research/<theme>/; winners -> rdy_to_serve/<backend>/. -->
