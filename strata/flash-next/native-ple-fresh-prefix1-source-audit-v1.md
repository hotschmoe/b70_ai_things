# Native PLE fresh prefix1 source audit

CONFIG -> actual NUM9 source29 engine d5q32zc7, prefix[248045], prior warmups,
request fresh1, source-nativeHC/PLE and batch0. Frozen source and selected GGUF
inventory metadata only; original weight/capture binary payloads unread.

COMMAND -> readonly source inspection of generate.cpp fresh admission/reset,
session.cpp session_zero, verify.cpp layer1 pre/staging, ple.dp.cpp exact-source
projection, native_ple_postops.dp.cpp and shared ngram.cpp/ngram.hpp.

RESULT -> fresh parsing sets lookup ceiling0; the fresh selection invariant
rejects any nonzero resume/live/slot/incoming restoration. For serial resume0,
session_zero clears all owned GDN/QSA state and the ENTIRE PLE history, then
resets ple_prev[-1,-1] and ple_token=-1. Main and each stage wait before replay.
The installed ss.ple.hist aliases the carved ss.ple_hist. Source history is
row-fastest[channel,9], with nine normalized prior-token rows; advance shifts
one row/channel and appends the current normalized row.

PLE applies at START of layer1 after immediate native layer0 FFN HC write and
before layer1 HC attention read. Original keyQ8_0[2560,10240] and
valueQ8_0[2560,2560] multiply the original F32 ngram embedding. Strict source
branches bypass legacy embedding Q8_1 quantization, BF16/F16/preprojected paths
and F16 convolution. Norms are ordinary F32 WeightTable views. The original
F32 conv reads weight[channel*4+tap], history[channel*9+3*tap] for taps0/1/2,
and current normalized channel for tap3. Output is hidden+gated+silu(convsum).
At source-initial prefix1, historical terms should all be zero; only tap3 and
the current direct-gated term remain. This is source expectation, not a readback
of actual native history or intermediate tensors.

The host stages PLE row IDs using current accepted token and a local copy of
ple_prev, gathers into h_ple_, and fences before raising the mapped flag. The
verifier resets h_flag_ to0 before each window. Layer1 copies mapped PLE input
only after the corresponding host-ready flag. No source-only demonstration of
actual mapped-input visibility or correct gathered bytes is available here.

Correction to the earlier quick metadata observation: inventory files[0] was
MTP sidecar metadata, so reading only that entry missed the dedicated PLE key.
The SELECTED UD-Q4_K_XL metadata has qwen4exp.ple.eos_token_id=248044,
ngram_size3 and conv_kernel4. Generic tokenizer EOS248046 and BOS248044 are
separate. Native PLE_EOS_TOKEN_ID248044 is correct for this model-specific pad.
Pinned original GGML qwen4exp uses that dedicated PLE key, as independently
confirmed by root. Both native and frozen owned reference pad missing context
as[248045,248044,248044]. No EOS source fix is justified or proposed.

VERDICT -> source reset/strict type/history/first-token boundary paths have no
identified omission. Native warm-history survival is neither shown nor ruled
out numerically. Root's ordinary PLE norm and layer1 SSM norm CPU pack/original
F32 bitwise check is separate evidence; GPU ordinary uploads remain unobserved.
Observed first BIG discrepancy at native-vs-owned layer1 postFFN cannot be
attributed to native PLE or reference math from these source facts alone.
Actual PLE embedding row IDs/bytes, incoming history, projections, gated/norm/
conv result and layer1 boundary states remain unobserved. P30 whole-row captures
can next separate pre-PLE input, attention and FFN boundaries after root runtime
permissions and C19 proof exist. No source patch, weight read, GPU job or runtime
write was performed by this audit.

Follow-up: confirmed current dispatch route gap (source evidence, additive).
Root and HC separately combined the actual mirror profile conditions:
device_plan is enabled, all_resident is false and STRATA_VERIFY_NO_HOST=1.
Thus ar_on() is false and the !no_host per-layer host loop is skipped. In
Verifier::run the PLE row IDs are computed at2056, graph launches at2090, and
the only current-row gather calls are inside ar_on() at2122 or the skipped
!no_host loop at2215. NEITHER gather executes on the current route. Layer1's
copy_from_mapped at900 does not wait on the ar-ready flag when ar_on is false.
The generic gather/fence description above is a source path census; it is NOT
what this current mirror/no-host route executes. Resetting persistent PLE
history cannot repair an absent current PLE embedding gather.

Source dispatch omission is now certain. Actual h_ple_ bytes, whether zero or
stale, remain unobserved; numerical attribution/completeness needs the new
source32 build and matched runtime/original checks. Batch stage_batch at2993
already gathers before launch and is a separate correct source path. HC owns
new source32 prelaunch gather for do_ple&&!ar_on()&&(device_plan||no_host), with
legacy duplicate gathers excluded. No frozen source was modified by this audit.
