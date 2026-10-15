# S95 retained coverage for first-C2 -> dependent-j1 question

CONFIG -> Existing derived records only; exact paths/bytes/SHA in pins.json.
No269MB raw trace read/parse, GPU/build/model/counter or production edit.
This is historical S95 coverage, not S106 device timing.

RESULT -> PARTIAL: enough anchors to bound the region, insufficient events to
measure/decompose the complete dependency chain without a later bounded raw
extraction. No new GPU profile is needed merely to recover existing events.

cycles.json (5632449B,23cycle records) retains head_events with exact name,
trace_ts_us/duration_us/pid/tid/id, and pair_tail_motifs. Its writer explicitly
keeps only names containing mtp_select, argmax or quantize_q8 (analyze_trace.py
line39). Thus this is a selected event list, not every event between heads.
all_primary_GPU_operations and the859-row operations.csv are name/count/service
aggregates; their min/median/p95 values cannot reconstruct invocation order.
coverage-gaps.json keeps only largest50 entries; none touches WideQ5K or
mtp_select. It is not a substitute for the omitted chain events.

Concrete cycle5 example, SAME primary GPU/raw trace timestamp domain:
C2head id133131 starts1792096674544763.8us, lasts1201.875us and ends
1792096674545965.5us. Select133136 starts1792096674546220.0us/lasts16.875us;
select133137 starts1792096674546239.0us/lasts16.875us. Next retained Q8 quantize
133141 starts1792096674546288.0us/lasts2.604us. The254.5us head-end-to-first-select
span contains omitted post-head work; it is NOT demonstrated idle/wait.
The32.125us second-select-end-to-quantize span also contains omitted work.
These anchors are examples, not12independent trials or attainable savings.

Actual source chain, mtp.cpp:
- capture_tail2145-2164: shared C2 head -> record_post_head(ownerA),
  record_post_head(ownerB) -> two mtp_select calls.
- record_post_head1138-1158: argmax_rows OR sample_tokens, then row_top_prob
  OR row_top_prob_split; optional sub-vocab map_ids. Actual CSV has
  sampler_greedy_kernel_5b2c3f and row_top_prob_split_kernel_9495b0;
  no argmax rows are observed. BOTH actual post-head names are omitted from
  pair_tail_motifs. Their individual timestamps/owner identity are missing.
- capture_pre_step2119-2132: copy_from_mapped_multi(two step/pos copies),
  record_forward(1,row,...false). record_front805-824 then gathers tok_,
  weighted RMS, Q8 quantization and embedding projection.
- NativeEmbed::gather_dev(native_head.cpp285-287) calls iq_embed_rows.
  CSV confirms copy_from_mapped_multi_kernel_66c71e, embed_rows_kernel_284a43,
  native_qsa_rms_norm_weighted::norm_127936 and the Q8 quantizer. Their first
  three invocation timestamps are absent from retained motifs. The mapped
  step/pos copy begins j1's graph; token embedding is the first selected-token
  consumer. A later retained quantizer is not the actual graph/data-ready start.

ONE later extraction, only if root chooses it: reuse reviewed cycles5..16 and
existing first-C2/two-select/first-following-Q8 anchors. For each, retain ALL
primary-GPU complete events overlapping [first-C2 start, first-following-Q8 end],
including boundary crossers and unexpected names, not just a name whitelist.
Preserve raw pid/tid/event id/name/start/duration and metadata. This bounded
output should recover both sampler/probability pairs, selectors, mapped-copy,
embedding and RMS; HOLD if counts/order/roles remain ambiguous. Do not guess
missing events from numerical ID gaps. No CPU-steady/GPU or peer-clock join,
physical idle, bandwidth, queue-fed-ahead, or S106 speed attribution follows.
This note does NOT perform or authorize that raw extraction.
