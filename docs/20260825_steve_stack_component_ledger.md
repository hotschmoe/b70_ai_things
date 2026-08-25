# Steve Qwen3.6 W8A8 Stack Component Ledger

Date: 2026-08-25

## Scope And Inventory Method

This ledger tracks every component relevant to Steve Seguin's Qwen3.6 35B-A3B
Quark W8A8 B70 result. It distinguishes executable source and launch recipes
from bulk benchmark evidence. Steve's work is prior art and evidence; the final
runtime must be rebuilt from attributed source owned by this repository.

Inspected repositories:

| Repository | Frozen revision | Tracked files | Treatment |
| --- | --- | ---: | --- |
| `b70-optimization-lab` | `c1cc2bf68ced2fb82192fd0d1dcb9d266225af04` | 20,816 | Read all Qwen result packets, accepted launcher/config, relevant notes, scripts, and patches; bulk JSON/log evidence is path/hash inventoried. |
| `vllm` | `44fc8fde09fc311d3099dab10366b672d9142ea4` | 5,285 | Current source inspected; closest surviving post-record checkpoint is `e190923b32e1b87fe33d08264bff9215fb7770fc` from 2026-06-16. |
| `vllm-xpu-kernels` | `2dd55f380df753a10a88fcd9e96192561066e713` | 312 | Operator bindings, dense INT8, grouped MoE, GDN, build, and Python interface are in scope. |

The lab contains 84 Qwen36 notes, 120 Qwen36 scripts, 411 Qwen36-named
patches, and thousands of result files. Reading every JSON repetition would not
add source understanding. The completeness rule is therefore: inspect every
relevant executable recipe/source/patch family and the accepted/rejected result
summary for each mechanism; inventory repetitive raw evidence without treating
it as executable software.

## Exact Accepted Identity

The TP2 reference used model revision
`cced56592e8c8935f8220836b4baa04dfd389118`, Quark W8A8, TP2/PP1, no MTP,
PIECEWISE graph, async scheduling, no prefix cache, and direct oneCCL P2P. Its
natural-chat request tokenized to 498 input tokens and produced 512 tokens in
5.96267 seconds of corrected decode, or 85.869114 tok/s.

The launcher overlaid both source trees:

```text
PYTHONPATH=/home/steve/src/vllm:/home/steve/src/vllm-xpu-kernels
LD_LIBRARY_PATH=/home/steve/src/vllm-xpu-kernels/vllm_xpu_kernels:...
```

Active graph/collective settings included:

```text
COMPILATION_CONFIG={"cudagraph_mode":"PIECEWISE"}
VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1
VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1
VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1
VLLM_XPU_CUSTOM_ALLREDUCE_GRAPH_CLONE_INPUT=1
VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1
CCL_ATL_TRANSPORT=ofi
CCL_TOPO_P2P_ACCESS=1
```

Steve did not pass a manual splitting-op list or force inductor graph
partitioning. The later launcher default made scheduling asynchronous because
`VLLM_EXTRA_ARGS='--uvicorn-log-level warning'` replaced the default
`--no-async-scheduling` argument.

## Component Status

| Component | Steve mechanism | Active in 85.87 run | Local status | Verdict |
| --- | --- | --- | --- | --- |
| Model and scales | Exact HF Quark W8A8 revision | Yes | Exact local revision and Quark loader verified | Matched. |
| Dense activation quant | `_xpu_C::per_token_quant_int8_xpu` | Yes | Operator present and selected | Matched. |
| Dense W8A8 GEMM | oneDNN `_xpu_C::int8_gemm_w8a8` | Yes | June registry restored; runtime logs select `XPUInt8ScaledMMLinearKernel` | Matched reachability; binary/source ownership pending. |
| Routed MoE | Xe2 XMX INT8 grouped GEMM with per-row activation and per-channel weight scales | Yes | Native Quark XPU INT8 MoE selected | Matched route; exact binary/source comparison pending. |
| Mixed MoE workspace | BF16 and INT32 persistent scratch interface | Yes in safe TP2 label | Local env enabled | Steve measured a small full-model regression in an earlier arm; not the 5x explanation. |
| Shared expert | Native dense W8A8 linears plus shared/routed combination | Yes | Later-image ABI mismatch bridged narrowly | Coherent; source snapshot comparison pending. |
| GDN decode | Native XPU GDN decode; recurrent fallback limited to prefill | Yes | Native decode and prefill-safe settings active | Coherent; graph ownership and exact SO remain to prove. |
| GDN quant reuse | Clone-safe QKVZ/BA quant reuse | Yes | `clone` setting active | Small lever; view/partial-clone variants were rejected. |
| Fresh GDN state | Zero newly allocated recurrent state | Yes, launcher default | Added to exact local env | Correctness identity; not a 5x speed lever. |
| Graph runtime | Forced-communication PIECEWISE replay | Yes | Exact minimal config now compiles; P2P-off monolithic oneCCL capture stalls | Primary unresolved mechanism. |
| Uniform no-spec descriptor | June Qwen decode capture descriptor | Yes | Restored by narrow adapter | Matched and required for capture. |
| Custom collective wrapper | `vllm::all_reduce` custom op with two clone guards | Yes | August source silently removed the inner clone; local attributed op now restores it | Must retest after reboot; prior exact run was not source-equivalent. |
| Collective transport | oneCCL/OFI direct P2P in graph | Yes | Deliberately off in safe controls | Exact guarded transaction pending. |
| Sampler | XPU greedy top-k fallback | Yes | Active | Not a 5x candidate; exact implementation comparison pending. |
| Scheduler | V1 async scheduling | Yes | Live log confirms async | Matched. |
| Prefix cache | Disabled | Yes | Disabled | Matched. |
| MTP | Disabled | Yes | Disabled | Matched. |

## Native Operator Chain

Steve's 2026-06-09 kernel patch added the complete Quark W8A8 execution path:

```text
BF16 hidden state
  -> native per-token INT8 quantization
  -> oneDNN dense INT8 x INT8 GEMM on XMX
  -> native Xe2 grouped INT8 MoE GEMM1
  -> SiLU/multiply plus second per-token INT8 quantization
  -> native Xe2 grouped INT8 MoE GEMM2
  -> expert gather and shared-expert combination
```

The patch families are:

- `csrc/xpu/quantization/int8_quant.cpp`: dynamic per-row absmax, scale,
  round, clamp, and INT8 output;
- `csrc/xpu/onednn/int8_gemm_w8a8.h`: dense oneDNN matmul with per-token and
  per-output-channel scales plus scratch cache;
- `csrc/xpu/grouped_gemm/xe_2/*`: Xe2 XMX INT8 accumulation, activation and
  weight scale epilogue, and W8A8 policy;
- `vllm_xpu_kernels/fused_moe_interface.py`: two-stage quantized routed-expert
  orchestration;
- vLLM linear, Quark MoE, and XPU expert registries: make the native operators
  reachable from the checkpoint.

The current inspected native hashes are:

```text
e043cfe218588b0440ebc1b0d208646b7ff73e4802bf2393e3d5299d5a3d4fa3  local build _xpu_C
cf482fd898ef965eeac70682027fe5578d5005b5eb6c51a85664a68e151a4a02  local build GDN library
ae330affe0315a5be4ac50478cc15c7874ae6e8fa9fa71cf64d5e5dff158968b  installed/pinned-image _xpu_C
```

The GDN build matches the inspected runtime hash. `_xpu_C` does not, so an
operator-level source/build manifest is still required before claiming binary
identity.

## Measured Lever Attribution

Steve's own controlled results provide the current ordering:

| Change | Decode class | Interpretation |
| --- | ---: | --- |
| Conservative/eager or unusable graph | about 12.83-17 tok/s | Launch and host scheduling dominate. |
| Usable PIECEWISE graph | about 76.48-92 tok/s | Largest lever by far. |
| Custom collectives disabled | about 70.79 tok/s | Collectives add a real but secondary graph-integrated gain. |
| Clone-safe custom collectives | about 95 tok/s TP4 class | Roughly 3 tok/s over Steve's graph baseline in the initial accepted study. |
| Remove inner clone | 99.01 tok/s but corrupted | Speed signal is invalid; clone is required for correctness. |

The accepted compiled graph census was approximately 220 dense activation
quant ops, 220 dense INT8 GEMMs, 101 RMSNorm ops, 81 all-reduces, and 40 shared
MoE-forward ops per captured model graph. Steve's layer timing put MoE at the
largest visible family and collectives behind it once graph replay worked.

## Local Transactions

### P2P-off split control

Config -> exact model, native INT8 path, TP2, PIECEWISE pieces split at
collectives, P2P off.

Result -> coherent 498/512 request at 17.055906 corrected tok/s. This is 5.0x
slower than Steve.

Verdict -> proves model identity and INT8 selection, but not Steve's graph
ownership.

### Manual legacy partition attempt

Config -> P2P off, no push collective, legacy partition mode, repository split
list.

Result -> vLLM code generation failed at
`vllm/compilation/codegen.py:96` because a split index was not an integer.

Verdict -> the local manual split policy is not part of Steve's recipe and can
actively break this source. Do not use it in the exact-control arm.

### Exact minimal graph plus local push all-reduce

Config -> only `{"cudagraph_mode":"PIECEWISE"}`, no manual split list,
P2P off, local Level Zero IPC push all-reduce.

Result -> the model and graph compiled. Rank 0 imported rank 1's scratch, but
rank 1 repeatedly failed to import rank 0's handle with `0x78000004`. The
hardened implementation exchanged status so both ranks fell back together.
Monolithic graph capture then stalled on oneCCL. Both cards passed post-teardown
health.

Verdict -> graph policy is repaired. The local push blocker is asymmetric
loaded-process Level Zero IPC visibility, not its arithmetic or standalone
capture path. The standalone exact-image harness still passes 50/50 replay and
multi-all-reduce at about 35.45 us for 10 KiB.

### Guarded direct-P2P transaction on kernel 7.1

Config -> exact minimal PIECEWISE graph, oneCCL/OFI direct P2P, later pinned
image, and the then-current adapter.

Result -> process-group initialization and model load succeeded, which is
farther than the old kernel failure. The first compiled custom-op all-reduce in
the profile run failed on rank 1 with `UR_RESULT_ERROR_DEVICE_LOST`. Both cards
passed the post-teardown single-card health probe.

Source finding -> the run emitted PyTorch's output-alias warning. Comparing
Steve's June source with the later image proved that the August
`parallel_state.all_reduce` removed the implementation of
`VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT`; the setting was present but inert.
The June source cloned once before the custom op and again inside it. A local
attributed custom op now restores that two-clone contract without overriding a
registered torch operator.

Verdict -> kernel 7.1 did not fix the direct-P2P vLLM failure. The transaction
does not yet isolate whether the device loss comes from oneCCL itself or from
the missing inner-clone/source contract. Reboot before another direct-P2P arm.

## Accepted And Rejected Mechanism Registry

Accepted or required:

- native XPU dense W8A8 quant/GEMM;
- native Xe2 grouped W8A8 MoE;
- forced-communication PIECEWISE graph;
- both collective clone guards;
- async scheduling;
- no prefix cache for the record;
- prefill-safe GDN fallback and native decode;
- greedy XPU sampler fallback.

Rejected or diagnostic-only in Steve's Qwen lane:

- removing either required collective clone when output aliasing corrupts
  tokens;
- fused SiLU plus quant with changed rounding/scaling semantics;
- mixed workspace based only on microbench gains;
- RMSNorm plus INT8 fusion before matching the live FP32-weight semantics;
- GDN view or partial-clone reuse;
- small-N dense GEMM variants that did not improve endpoint speed;
- local argmax and unsafe sampling shortcuts;
- speculative/MTP results that fail state and canary gates;
- experimental full MoE layerlets not endpoint-promoted by the accepted run.

## Next Decisive Transactions

1. After reboot, retest one guarded exact oneCCL P2P transaction with the
   restored June two-clone contract and a fresh compilation cache.
2. If the narrow current-source repair does not reproduce, run the closest surviving 2026-06-16
   vLLM snapshot as a forensic overlay with the same runtime binaries.
3. Convert the required delta into attributed local patches and a pinned image;
   do not retain a Steve checkout mount.
4. Rebuild `_xpu_C` and GDN from local source, then prove op schemas, numeric
   equivalence, graph replay, and hashes.
5. Only after a healthy 80+ tok/s graph baseline, profile MoE, dense GEMM,
   GDN, sampler, and collectives under the actual endpoint step.
