# Flash-Next backend source audit, 2026-10-09

CONFIG -> CPU/network-only audit of exact selected Unsloth UD-Q4_K_XL
Flash-Next, two B70s, host overflow experts, complete prefix state and 1/2/4
streams (six is stress only). No GPU operations, ABI builds, model conversion,
source checkout changes, or speed qualification. Existing dirty files retained.

COMMAND -> Read AGENTS.md, campaign, goal and Strata next-steps evidence;
query git ls-remote HEAD; download complete recursive GitHub source trees and
selected files at resolved commits; inspect actual dispatch, loader, scheduler
and state capture code. Raw trees, source files, pins and llama.cpp upstream
comparison are under
/mnt/vm_8tb/b70/results/flashnext_backend_audit_20261009/.
The complete tree responses reported truncated=false. These file reads prove
source behavior, not execution or full tensor inventory compatibility.

RESULT -> Current remote pins differ from some October 8 release pins:

| Backend | Audited source pin | Exact-artifact Intel path and remaining cost |
| --- | --- | --- |
| llama.cpp | local de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b; remote 3d65c90d04d337e88f2b1f7f0061f40a5324e662 | Exact GGUF already loads locally; SYCL expert cache and server slots exist. First-divergence localization remains the primary blocker. |
| Strata | fb58e0dbc8399662c0e47c76578c6e878b14f6cf (remote unchanged) | Existing SYCL split, quantized experts, slots and per-stage in-memory snapshots; Q8/F32 hyper-connection integration and per-stage host mirror ownership need qualification/development. |
| vLLM | 51ef2397685b49ebd90b39bb4b4806a509df19b5 | Qwen4Exp explicitly rejects XPU. Requires model/kernel port, GGUF plugin XPU path and mixed expert tier integration before scheduler advantages apply. |
| SGLang | 6e7beace143964386ce2e4418401b7369f0dabd0 | Qwen4Exp and hybrid prefix machinery exist, but GGUF imports kernels only for CUDA/MUSA/NPU; exact mixed GGUF XPU path requires loader/kernel/tier work. |
| ik_llama.cpp | 04b4ebc2b32240c7495c33cb23ee35ac8108bb2a | Qwen4Exp graph exists; SYCL operation coverage is older and incomplete for that graph. CPU reference candidate, not a turnkey Intel replacement. |
| KoboldCPP | 6b23d96706048e9542a02d809125cb789c6cfeff | Qwen4Exp, Intel Vulkan, CPU MoE placement, recurrent smartcache and real batching exist, but smartcache disqualifies requests from continuous batching. No bounded GPU expert refill cache was found in the inspected placement/adapter code. |
| mistral.rs | 3f2515e9b5adc2ac44949c128c50a13b15721294 | Qwen4Exp GGUF mapping and recurrent prefix snapshots exist; accelerator feature definitions expose CUDA/Metal, not Intel SYCL/Vulkan/XPU. New device backend is the dominant cost. |

The current llama.cpp tip is one commit beyond the locally built pin: Q5_K
reorder-layout MMVQ and fused GLU, touching four SYCL files. It selects an Xe2
BMG path and avoids fused Q5_K beyond five columns. This is a potentially
relevant optimization for this mixed checkpoint, not evidence of a fix for
history drift. Preserve the active diagnostic generation; adopt the commit in
a separately pinned build after localization/primitive gates, without claiming
its advertised speed transfers to this model.
[Exact delta](https://github.com/ggml-org/llama.cpp/commit/3d65c90d04d337e88f2b1f7f0061f40a5324e662).

## Weight tiers and prefix tiers are different mechanisms

llama.cpp's llama-moe-cache.cpp stores model expert weight bytes in bounded
GPU slots, with host expert tensors as backing. Layer split assigns caches to
the layer's device; tensor-parallel split is rejected. The cache does not save
conversation history. Separately, server-context.cpp uses prompt checkpoints
and sequence copy. llama-memory-hybrid.cpp forwards sequence copy and state
save/read to attention and recurrent memory; llama-memory-recurrent.cpp has a
separate PLE convolution row, included in saved state. These are concrete
mechanisms, but unresolved repeated-request drift prevents declaring complete
prefix correctness or a trusted full-model reference.
[Expert cache](https://github.com/ggml-org/llama.cpp/blob/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b/src/llama-moe-cache.cpp),
[hybrid memory](https://github.com/ggml-org/llama.cpp/blob/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b/src/llama-memory-hybrid.cpp),
[recurrent memory](https://github.com/ggml-org/llama.cpp/blob/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b/src/llama-memory-recurrent.cpp).

Strata's VRAM expert cache, pinned expert mirror, PLE row cache, per-request
KV/state, prompt checkpoints and parked conversation images have separate
budgets and lifetimes. Its existing stage-0 expert mirror cannot establish
stage-1 RAM expert coverage. A conversation cache in RAM similarly cannot
establish model-weight offload. Account all of them against the same physical
RAM/VRAM budgets before admitting additional streams.

## Strata two-stage prefix support: actual routes and limits

The !multi_gpu guard at generate.cpp:10187 disables the optional extra
message-boundary checkpoint. It does not disable all split-stage prefix reuse.
The actual pinned SYCL code has the following routes:

- Live continuation at approximately 9465 selects the longest valid live,
  checkpoint, inactive-slot or parked prefix, with token/image/control-vector
  checks. Reusing live state needs no snapshot transfer.
- checkpoint_at at 7497 saves GDN, PLE and indexer running state for the main
  session and every later stage into stage_parts. Root and turn checkpoints
  call it at 10234. Mid-prompt stage-part collection at 7612 records each
  stage at the same position before combining the parts.
- park_current_body at 7333 estimates/captures an image for each stage under
  its device, shares one conversation-cache RAM budget, and splits checkpoint
  parts. Later stage snapshots omit the draft except on the actual draft
  owner. With MTP disabled every stage uses a null draft.
- Incoming snapshots are checked for exact stage count and prevalidated on
  each stage before writes. Restore at 9574/9584 loads every stage; failure
  after prevalidation is fatal. Checkpoint rewind at 9687 restores each stage
  part. STRATA_CKPT_REREAD instead zeros every stage and rereads from token 0,
  providing a useful matched negative control.
- copy_from_slot at 8629 also iterates stages, restoring the appropriate
  running checkpoint and copying the positional KV region per stage.

The shared conversation_state.cpp:154 capture saves gdn_state, ple_hist,
idx_tail, idx_dead and idx_block_pos. Restore reconstructs ple_prev from token
IDs and restores the moving pooled row. The compiled SYCL
conversation_snapshot.cpp captures/restores K, V, scales and pooled indexer
rows, including the partial page and moving spare row. Thus source coverage
extends beyond KV alone. All these routes still require actual two-B70
cached/uncached logit and state validation, cancellation, eviction and
concurrent isolation; source presence is not a passed gate.
[SYCL driver](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/program/generate.cpp),
[running state](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/src/core/conversation_state.cpp),
[SYCL snapshot](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/sycl/src/core/conversation_snapshot.cpp).

Two limits must remain explicit: disk session commands refuse --layer-split at
9054; in-memory parking does not. The optional extra message checkpoint stays
disabled for multi_gpu. Neither restriction requires a new generic prefix
cache to satisfy the requested in-memory serving use case. The implementation
cost is qualification plus any localized state defects, identity hardening,
and RAM/VRAM admission changes, rather than assuming no split snapshots exist.
The existing long-prefill piped path bypasses explicit between-chunk decode
sharing at 10046, so fairness remains a separate scheduling gate.

Disk session identity currently samples file head/tail rather than full hashes,
and this SYCL driver's config backend string falls through to cuda. These
facts require hardening before portable persisted-cache promotion. In-memory
images are process-local, but final qualification should still bind receipts
to the full model lock, tokenizer/template, engine/build and resolved arithmetic.

## Alternative-specific source blockers

vLLM's current qwen4_exp/__init__.py raises NotImplementedError for XPU before
model instantiation; this survives the newer pin. GGUF is now an out-of-tree
plugin. General CPU-offload and prefix-cache settings cannot bypass model and
quantized-kernel absence. Starting a full model after deleting the guard would
not be a justified feasibility test.
[Dispatch](https://github.com/vllm-project/vllm/blob/51ef2397685b49ebd90b39bb4b4806a509df19b5/vllm/models/qwen4_exp/__init__.py),
[GGUF plugin boundary](https://github.com/vllm-project/vllm/blob/51ef2397685b49ebd90b39bb4b4806a509df19b5/docs/features/quantization/gguf.md).

SGLang's gguf.py imports is_xpu but has no XPU import branch for ggml kernels.
Its Qwen4 PLE offload creates torch.cuda.Stream and calls CUDA stream APIs.
The model also carries PLE running-state updates at hybrid radix boundaries,
so generic radix maturity is useful source material; it does not prove exact
GGUF QSA/PLE state and tiered experts work on Intel. Port both math and memory
ownership deliberately if later profiles justify that larger integration.
[GGUF dispatch](https://github.com/sgl-project/sglang/blob/6e7beace143964386ce2e4418401b7369f0dabd0/python/sglang/srt/layers/quantization/gguf.py),
[Qwen4Exp](https://github.com/sgl-project/sglang/blob/6e7beace143964386ce2e4418401b7369f0dabd0/python/sglang/srt/models/qwen4_exp.py).

ik_llama's build_qwen4exp.cpp uses SET_ROWS, MULTI_ADD and indexer operations
that are absent from its SYCL supports_op switch, whose default is false.
IQ4_NL GET_ROWS is also absent. CPU fallback may permit execution, but would
add transfer/host work and needs a real allocation graph census. Its expert
prefetch uses MADV_POPULATE_READ on mmap weights: this is RAM page-cache
warming, not a VRAM expert-slot refill cache. Existing ncmoe static placement
and CPU implementations make it a possible independent CPU graph comparison
if first-divergence work needs one; do not call it qualified yet.
[Graph](https://github.com/ikawrakow/ik_llama.cpp/blob/04b4ebc2b32240c7495c33cb23ee35ac8108bb2a/src/graphs/build_qwen4exp.cpp),
[SYCL coverage](https://github.com/ikawrakow/ik_llama.cpp/blob/04b4ebc2b32240c7495c33cb23ee35ac8108bb2a/ggml/src/ggml-sycl.cpp),
[prefetch](https://github.com/ikawrakow/ik_llama.cpp/blob/04b4ebc2b32240c7495c33cb23ee35ac8108bb2a/ggml/src/ggml-moe-prefetch.cpp).

KoboldCPP has more relevant support than its older reputation suggests:
gpttype_adapter.cpp implements batch_worker_loop and n_seq_max slots;
Vulkan implements gated delta net, SSM convolution, SET_ROWS and IQ4_NL rows;
--moecpu places selected expert layers on CPU. However batch_inputs_eligible
at 4456 returns false when smartcache/smartcontext/contextshift is enabled.
Batch admission removes prior slot state and completion clears it. Its
recurrent smartcache therefore does not currently compose with this batching
route to provide the requested cached concurrent conversations. Closing that
boundary, adding/qualifying weight tiers and measuring the full Vulkan model
is a larger change than using existing llama-server routes. It remains a
possible same-GGUF Vulkan diagnostic lane, with shared upstream math caveats.
[Adapter](https://github.com/LostRuins/koboldcpp/blob/6b23d96706048e9542a02d809125cb789c6cfeff/gpttype_adapter.cpp),
[Vulkan](https://github.com/LostRuins/koboldcpp/blob/6b23d96706048e9542a02d809125cb789c6cfeff/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

mistral.rs includes Qwen4 GGUF tensor bindings and recurrent prefix checkpoint
capacity/bytes/eviction accounting. Its Cargo accelerator features expose
CUDA/Metal plus CPU MKL/Accelerate, not the requested Intel device path. An
Intel port crosses the tensor engine, quant kernels, paged state and transfer
ownership; maturity on other devices is not a low-cost exact-artifact route.
[GGUF bindings](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/mistralrs-core/src/gguf/qwen4exp.rs),
[device features](https://github.com/EricLBuehler/mistral.rs/blob/3f2515e9b5adc2ac44949c128c50a13b15721294/mistralrs-core/Cargo.toml).

VERDICT -> Keep llama.cpp as primary first-divergence development lane because
it already executes the exact artifact and exposes the required weight tiers
and serving mechanisms. Develop Strata as the bounded alternative, with
native hyper-connection fidelity first and stage-specific RAM mirror coverage
next. Its complete in-memory split prefix source routes reduce implementation
cost, but have not earned correctness or latency claims. Prefer these two
existing paths over a broad vLLM/SGLang/other engine port unless measured
critical-path evidence changes that choice.

Retain independent CPU GGUF block dequantization/F32 primitive references;
freeze full-model teacher-forced reference fixtures only after their own state
and repeat gates pass. Neither current engine is yet a trustworthy qualified
full-model authority. A CPU ik_llama graph or same-GGUF Vulkan lane may help
triangulate a localized discrepancy, but readable generation alone does not
establish that reference. No backend is promoted by this audit.
