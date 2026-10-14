# FlashNext request reset observation, 2026-10-09

CONFIG -> Pinned de7fa0 llama.cpp, selected Unsloth UD-Q4_K_XL, runtime39992d70,
static two-card placement, OPT1/fusion1, original checkpoints, ubatch256,
warmup on, no MTP, FP16KV C1/8192. Default-off reset source patches0005+0006.
Same-build36/36 primitives per card pass and full model identity is recorded.
No optimization, performance comparison or serving promotion.

COMMAND -> Three independently launched owned controls: reset-off-history-v1,
reset-metadata-history-v1 and reset-numeric-history-v1. Each preserves strict
screening and native P/P/JSON/P captures, followed by normal stop/removal and
strict per-card plus compiled P2P0 post-health. Arm file is exclusively created
after screening. Offline history analysis plus independent F32 raw-bit audit.

RESULT -> Off arm strict screen fails; native immediate repeat shares35 IDs,
after-JSON differs at22. Metadata and numeric strict screens pass within their
individual starts and all three native prose outputs share35 IDs, but all arms
retain probability drift beginning at token0. This does not establish a fix or
broad same-build determinism. Instrumentation can alter timing; the comparison
is diagnostic, not a clean latency measurement.

Metadata establishes actual consumed prompt ubatches31+4, identical internal
little-endian token FNV hashes on repeats, seq0, rollback0/src0=rs_z=0 for fresh
requests, rs_z=-1 for continuation. Layer0 reset views are R30720 floats and
S786432 floats; continuation views are zero-sized and explicitly unobserved.
It does not record seq_rm success or attention/QSA reset completeness.

Numeric arm captures six paired source/destination views: R and S at the first
ubatch of the first three requests (prose, immediate prose, JSON). Every prior
row is finite and nonzero; every post row is positive zero, with no NaN,
infinity or negative zero. An independent host bit-pattern scan confirms the
12 raw files and their hashes. Both full-size rows are observed, not samples.
The observer uses the same in-order queue, pre-copy/wait, original SCALE once,
and post-copy/wait. Passing these observations cannot prove production ordering
or absence of a race because the waits may hide it.

VERDICT -> Nonfinite contamination or failed lazy clearing is not observed in
these selected layer0 R/S rows. Metadata mapping and numerical reset evidence
narrow this particular hypothesis while probability drift remains unresolved.
Layer1 PLE, later recurrent layers, attention/QSA, head logits and activations
remain unobserved. Fourth after-JSON prose falls outside the three-start capture
bound. Next reference diagnostic is bounded activation/full-logit capture to
find the first downstream divergence, plus explicit PLE/later state coverage.
No state-clear implementation was changed and no backend is promoted.

Raw evidence: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f04-20261009/.
Independent bit audit: reset-numeric-history-v1/offline-raw-reset-analysis.json.
