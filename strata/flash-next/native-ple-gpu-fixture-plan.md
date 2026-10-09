# Original-source PLE numerical preregistration

CONFIG -> Patches0001..0006, selected original Q8_0 key[2560,10240],
Q8_0 value[2560,2560], originalF32 conv[4,10240], and originalF32 embeddings,
norms and residuals. No compatibility weight or activation representation is
used as an accepted input. GPU runtime remains image39992d70 and a parent-owned
bin/gpu-run one-card lease/device pin with health and teardown checks.

COMMAND -> Build native_ple_source_gpu_fixture.cpp with current overlay headers
before pinned source headers, hc_native_projection.cpp,
cuda/native_ple_postops.dp.cpp and cuda/native_gr_norm.dp.cpp. The sole optional
argument is a JSONL receipt path. CPU_REFERENCE_ONLY mode uses
-DHC_FIXTURE_CPU_ONLY and performs no SYCL discovery or execution.

RESULT -> Preregistered coverage is7 cases at row counts1/2/4/8/9/16/17, totaling
57 sequential normalized-history appends and70 numerical stage checks. Every
case uses full production matrix shapes and k-fastest originalF32 convolution
coefficients. Generated Q8 codes include-128 and127, positive/negative scales
and originalF32 embeddings with values changed by reduced activation precision.
Aligned, anti-aligned and mixed hidden/key rows exercise both gate saturation
tails and finite interior gates. Norm gammas differ across channels and streams.
Initial history has9 row-fastest entries per10240 channels.

The CPU oracle independently decodes exact FP16 scales times signedQ8 codes and
accumulates original projections inFP64. It computes per-stream weighted RMS,
signed-square-root attention scores, sigmoid gate, broadcast value, per-stream
normalized conv input, taps9/6/3/0 at originalF32 k+4*channel indices, SiLU conv,
hidden+gated+activation residual, and every shifted/appended history snapshot.
The oracle retains continuousFP64 intermediates from originalF32 inputs; it does
not mirror the GPU reduction tree or use compatibility images.

Require every stage finite, NMSE<=1e-6 and normalized maximum error<=1e-4, using
max_abs_error / max(1e-6,max_abs_reference). These retain the prior projection and
HC composition gates, preregistered for PLE before any GPU fixture execution.
Each of70 stage oracles must reject a1-percent-scale output perturbation.
Separate CPU negative controls must produce nonzero output differences when
value weights round toBF16, embedding activations round toF16 or blockwiseQ8,
and originalF32 conv weights narrow toF16. Nonzero conversion controls prove
fixture sensitivity; they do not establish the exact runtime fallback error.

Require exact bytes for repeated runs after independent buffers intervene and
for <=8-row projection chunks versus concatenated single-row projections.
Both grouped and single projection variants feed identical sequential scalar
postops calls; compare every intermediate and every history snapshot byte.
Every allocation has128-byte guards; the guard checker must reject an injected
host guard corruption. Original key/value/conv/norm/embedding/hidden bytes must
remain unchanged. Only fixture-owned history is mutable and restored before
repeats. Peak buffers are bounded to approximately100MiB device allocations;
no expert bank, model session or serve is loaded.

The fixture owns a simple per-channel9-row historyshift kernel between actual
native_ple_postops calls. It qualifies kernels against the documented caller
protocol and independent history oracle. It does not execute ple_block,
ple_history_advance, prefill, verifier, ownership intake, graph capture or request
reset. Those actual engine routes remain required model qualification gates.
The fixture explicitly uses patch0006 source_exact postops with originalF32
convolution and directly calls the existing originalQ8/F32 projection primitive.

VERDICT -> A GPU pass requires7 unique case rows,70 unique stage rows,
57 history rows,70 numeric negative controls and a passing guard negative control.
Summary mode must beGPU_PLE_SOURCE_NUMERICAL, cases7, failed0, stages70,
history_rows57, numeric_negative_controls70, pass=true. Case rows identify rows,
exact_repeat, exact_chunk_vs_single, guards, unchanged_weights_input andpass,
plus positive bf16_value_reference_difference, f16_activation_reference_difference,
q8_activation_reference_difference and f16_conv_reference_difference. Stage rows
identify rows andstage, with finite, nmse, normalized_linf andpass.

Named stages arekey_raw, value_raw, key_norm, query_norm, gate, gated, norm_conv,
conv_silu, result andhistory_snapshots. CPU mode reportsCPU_REFERENCE_ONLY and
supplies reference-rounded outputs, so it cannot establish GPU equality/guards.
No serving speed, stability, model coherence or two-device fidelity is claimed.
