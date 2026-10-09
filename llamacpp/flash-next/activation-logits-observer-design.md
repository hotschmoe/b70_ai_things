# Bounded activation/full-logit observer after finite reset evidence

CONFIG -> Pinned de7fa0 llama.cpp source and qualified diagnostic patch set
001+005+006. Existing layer0 reset observation found selected R/S pre-state
finite and post-state positive zero; native repeated-request probability drift
remains unresolved. No new backend switches, weights or corrective kernels.

COMMAND -> Inspect existing graph/eval callback, actual qwen4exp names, ubatch
output mappings and original sampler host access. Prepare default-off backend
patch007 and independent build plan. Run CPU mocked-backend ASan/UBSan contracts.

RESULT -> Patch SHA256:
8bc31416a8254e2c31a4b7de060c7995bf744bfa04aadcbf0a0d6d3e2aeda89e.
Sequential apply passes. Actual observer header compiles under CPU g++17 and
fatal ASan/UBSan. Mock tests pass default-off zero copies/callback replacements,
shape/type/parameter/overflow rejection, original callback chaining/restoration,
output/token correlation, canonical LE signed-zero/NaN payload preservation and
byte quota checked before backend copy. Mock files are40960-byte HC vectors and
993280-byte full vocabulary rows. No real backend or GPU arithmetic ran.

VERDICT -> Concrete bounded observer source/build recipe exists. Actual full
server/SYCL compilation, selected tensor visibility, raw captures, original
output equivalence and healthy teardown remain parent-controlled gates. This
is observation only, not a reset fix or inference/performance qualification.

## Execution and correlation

FLASHNEXT_ACTIVATION_DIAG is default off. ARM must be a new regular file created
only after readiness and screening; DIR is a fresh writable absolute directory.
Read configuration once. Disabled mode adds no diagnostic buffers, tensor names,
graph nodes, device copies/waits or logs. It preserves existing serving flags
and binaries. The new source plan clones the pinned tree independently.

Scope requires one context, one sequence, one-dimensional text token positions,
normal decoder, no speculation/decisions and cache_prompt=false. Position-zero
starts number requests. Bounds:3 requests, first2 ubatches for activations,
8 candidate-logit vectors per request,4096 output mappings per decode,
16 MiB per vector and128 MiB reserved capture bytes per process. Request/ubatch,
logical decode/graph ordinal, actual token IDs/positions/original batch indices,
sequence and tensor dimensions/strides/type are logged. Reused graphs receive
fresh scope metadata; graph construction count is not the decode count.

Existing names selected without renaming or adding nodes:

| Existing tensor | Observation |
| --- | --- |
| hc_init | Complete last-token layer0 input:4x2560 F32 |
| l_last-0 | Complete last-token layer0 output:4x2560 F32 |
| ple_gated_value-1 | Complete last-token layer1 PLE gated activation |
| ple_conv_out-1 | Complete last-token layer1 PLE convolution activation |
| conv_state_at-1 | Complete one-sequence gathered R/P state, resolved by cache_r_l1/cache_ple_r_l1 ancestry |
| state_predelta-1 | Complete one-sequence gathered S state |
| result_output | Complete248320-value raw logit candidate at its last output row |

Gathered R/P/S state is not direct SCALE pre/post coverage. For position-zero
starts it follows the graph's reset/gather; continuation ubatches may contain
valid nonzero history. Missing/unsupported names, types, layouts, sequences or
bounds emit observed-only skips; absent data never means zero. Physical cache
allocations, unrelated rollback rows and model weights are not dumped. PARAM
flags reject. Canonical raw LE F32 preserves NaN payload and signed-zero bits.

The observer chains any original eval callback and restores it with RAII on
return/failure. Only armed selected ubatches install it. Existing callback data
phase follows selected-backend synchronization; the callback therefore splits
execution and can prevent fusions or hide timing races. Capture vectors remain
owned through synchronous backend reads and checked exclusive file writes.
Host copy timestamps include transfer/wait work, and are not device execution
intervals or cross-clock measurements. Diagnostic latency is not clean latency.

## First-generated full vocabulary semantics

The server marks its actual first sampler call using narrow exported libllama
diagnostic functions. Correlation state resides in libllama, avoiding separate
TLS state across the executable and shared library. An existing result_output
candidate is labeled first-generated only when its original batch index matches
that actual sampler call. Otherwise the original get_logits_ith access inside
sampling can supply the complete row, after its existing synchronization.
No new sampler, output flags, forced vocabulary copy or backend-sampling switch
is introduced. Sparse/no-host sampling without a matched graph candidate is
explicitly unobserved. Sampler-result token/task/slot metadata remains recorded.

## Build and required equivalence lanes

Parent command, under bin/gpu-run and existing lifecycle gates:
python3 llamacpp/flash-next/build_cache_diagnostic.py --plan
llamacpp/flash-next/activation-logits-observer-build-plan.json.
The JSON contains exact original source/patch hashes and required overlay files.
Do not apply to the live checkout or overwrite qualified baseline binaries.

Keep model, image, OPT, fusion, checkpoints, cache placement, batch sizes,
threads, sampling and SYCL_CACHE_PERSISTENT identical. Run clean observer-off,
then logit-only and activation+logit observations in paired/interleaved order.
VECTOR=0 disables activation callbacks; LOGIT_NODE=1 may remain in both
instrumented arms when backend sampling needs a full original raw row. A pure
host-accessor arm can set LOGIT_NODE=0 if the original sampler supplies all
logits; unavailable rows stay unobserved. These are diagnostic coverage choices,
not a stack of model fixes. Existing reset-numeric waits are a separate arm.

Require exact output token IDs and full first-vocabulary bytes where both arms
observe them; retain native probability drift. Passing only under callback
fences is a timing-sensitive observation, not evidence the original graph is
correct. Compare same token/ubatch geometry before comparing raw vectors. Use
collect_activation_observer.py to validate raw lengths, hashes, classifications,
correlations, explicit skips and matched-key differences; its first difference
is first OBSERVED, not proof of the earliest unobserved operation.

Before attribution, qualify collector negative controls (one altered captured
byte must fail equality), callback-off/on trace coverage, health and teardown.
If layer0 input matches and output diverges, narrow that layer's arithmetic or
reads next. If layer0 and selected layer1 points match but logits diverge,
expand observed layers/QSA/indexer/expert paths deliberately. This source
increment does not rule out later-layer or attention-state faults.
