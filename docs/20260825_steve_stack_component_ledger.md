# Steve Qwen3.6 W8A8 Stack Component Ledger

Date: 2026-08-25

## Latest Closure: Graph Timing And Native Checkpoint Recovery

The closest surviving June vLLM source, commit
`e190923b32e1b87fe33d08264bff9215fb7770fc`, now runs as a genuine full-source
overlay with the rebuilt June native package. A 12-component off-device
contract pins graph, collective, GDN, MoE, scheduler, sampler, runner, and
kernel-interface origins and hashes. The public June source and June-9 Python
kernel interface are not ABI-complete together: `xpu_moe.py` passes a
`scratch` argument that the June-9 interface does not accept. The recovered
scratch-aware interface at kernel commit `2dd55f380df753a10a88fcd9e96192561066e713`
closes that seam.

The guarded true-June transaction selected native dense and routed-MoE INT8,
completed 81/81 profile clone fences per rank, captured 9/9 PIECEWISE graphs,
passed the exact p498/o512 metric plus both 16/16 canaries, and passed per-card
and compiled-collective health after graceful teardown. It measured 48.531479
tok/s, 10.548628 seconds server decode, and 311.856 ms client TTFT. That is
6.98 percent above the 45.364920 tok/s August-adapter native-MoE control, but
only 56.52 percent of Steve's 85.869114 tok/s. The remaining gap is 37.337635
tok/s, or 1.7693x.

Built-in graph replay tracing now closes the topology question: the endpoint
observed every piecewise index from 0 through 40 and reported 41 total pieces,
matching Steve's packet exactly. Synchronized pure-decode timing instead puts
the rank-0 local model-forward at 22.674753 ms versus Steve's 5.694625 ms, a
3.9818x execution gap. The local synchronized diagnostic served at 35.469940
tok/s versus Steve's synchronized 84.3075 tok/s. This is not an absent-graph
explanation.

The native provenance also requires a correction. The 55.5 MB package built on
2026-08-25 is an exact reconstruction of the June-9 minimal patch over
`28e1f5e`; it is not the full native source present when Steve recorded the
accepted timing. Steve's object database contains exact checkpoint
`122b698bc245d31668a7fe5f2ad5ce1d07ba08ca` from 2026-06-16 and its child
`3ed399adc384385fed5663a27623f09ecf44e085` from 2026-06-19. The former adds
5,054 lines across 24 files, including output-buffer variants for per-token
quantization and fused SiLU/multiply/quantization. The current
scratch-aware dispatcher conditionally calls these operators, but the June-9
binary lacks them and falls back to allocation plus copy. An exact `122b698b`
native-binary-only A/B is therefore the next decisive test. Dense 27B remains
the transfer control: measure its graph/timing topology and test the dense
quant/output-buffer primitives through a dense-specific adapter without
importing MoE-only layerlet or sidecar conclusions.

Reachability is specific, not hypothetical. Mixed workspace supplies
`gemm1_a`, `gemm1_a_scales`, `gemm2_a`, and `gemm2_a_scales` persistent
buffers. The scratch-aware dispatcher invokes `_per_token_quant_int8_out` for
GEMM1 and GEMM2 in each of 40 MoE layers, or 80 calls per decode step. Without
the native schema it calls the allocating operator, then `copy_` twice into
the persistent outputs. With `122b698b`, the same Python branch dispatches
directly to `_xpu_C::per_token_quant_int8_xpu_out`. Steve's accepted launcher
explicitly unsets `VLLM_XPU_FUSED_MOE_FUSE_SILU_QUANT`, so the sibling fused
SiLU/multiply/quant operator is not attributed to the 84.3075 timing control.

### June-9 To `122b698b` Reachability Map

| Native delta | Source comparison | Accepted-lane reachability | Treatment |
| --- | --- | --- | --- |
| Per-token INT8 quant `_out` | New schema and implementation in `int8_quant.cpp`, `ops.h`, and `torch_bindings.cpp` | Active through mixed-workspace GEMM1/GEMM2 quant, 80 calls/step | Primary binary-only A/B. |
| Fused SiLU/multiply/quant and `_out` | New schema and implementation | Inactive; accepted launcher explicitly unsets its switch | Do not attribute to accepted speed. Test later as its own arm. |
| Grouped-GEMM base Xe2 tile/policy | `gemm_xe2.hpp` and `gemm_xe2_policy.hpp` are byte-identical to the June-9 reconstruction | Active, but unchanged | Not the checkpoint lever. |
| Grouped offsets, sidecar, and layerlets | Large new interfaces and default-off experimental entry points | Inactive in the 84.3075 control | Keep out of the first A/B. |
| GDN sibling source | Three apparent differences are final-newline-only | Active, but executable source unchanged | A rebuilt hash may differ; no GDN code lever exists in this checkpoint. |
| Dense oneDNN INT8 GEMM | Core matmul implementation is byte-identical; scratch cache becomes a configurable ring with default size 1 | Active, default behavior remains one scratch buffer | Not a claimed speed lever unless a later ring-size arm proves it. |
| RMSNorm plus quant | Adds Qwen/Gemma BF16-input/FP32-weight handling and rounding corrections | Inactive because `pass_config.fuse_norm_quant=False` in the accepted/local exact lane | Correctness primitive for a future dense adapter, not this endpoint delta. |

This reachability map is why the first `122b698b` transaction swaps native
siblings only while preserving the `2dd55f38` scratch-aware Python dispatcher.
It is an operator-presence A/B, not a wholesale experimental-layerlet test.

### Exact `122b698b` Native A/B Result

The clean June-16 checkpoint was rebuilt with oneAPI DPC++ 2025.3.3, Release,
Xe2, `bmg-g21-a0` AOT, MoE and GDN enabled, and the pinned torch 2.11 image.
Only `_xpu_C.abi3.so`, `libgrouped_gemm_xe_2.so`, and
`libgdn_attn_kernels_xe_2.so` were replaced in a copy of the June-9 runtime.
The installed `_xpu_C` RUNPATH is `$ORIGIN`. The strict preflight proved both
native quantization output schemas and XPU dispatch before any GPU touch.

The matched unsynchronized endpoint measured 50.370643 tok/s, 307.853 ms
client TTFT, and 10.163436 seconds server decode. The June-9 endpoint under the
same June source was 48.531479 tok/s, so the exact native checkpoint gains
1.839164 tok/s, or 3.79 percent. Both 16/16 repeat canaries passed, graph capture
completed 9/9, and per-card plus compiled-collective health remained green
after graceful teardown. This comparison must not use the 35.469940 tok/s
synchronized diagnostic as its baseline. The result proves native scratch
output matters but does not explain Steve's remaining endpoint gap. Next:
repeat synchronized timing with `NATIVE_STACK=june122-checkpoint`, then measure
integrated graph collective/runtime cost.

The synchronized arm is complete. Across 62 pure-decode samples, rank-0
model-forward is 21.994441 ms versus 22.674753 ms on the June-9 binary, a
0.680311 ms or 3.00 percent reduction. The synchronized endpoint is 36.429308
tok/s versus 35.469940 tok/s (+2.70 percent). GDN is 3.948703 versus 3.927777
ms, logits 1.585771 versus 1.585079 ms, local argmax 1.151977 versus 1.149508
ms, and sampler 0.677830 versus 0.663268 ms: these families are flat within
run noise. The same graph indices 0..40 and total 41 pieces replayed. Steve's
model-forward remains 5.694625 ms, leaving a 16.299816 ms integrated gap.
Current nested Python timing cannot observe the 81 compiled TP collectives
inside graph replay, so a device/runtime timeline is the next measurement.

That bounded two-rank timeline is now complete. Each profiled token contains
41 `zeFenceReset`, 41 `zeEventHostSynchronize`, and 82
`zeCommandQueueExecuteCommandLists` calls, matching the 41 PIECEWISE graph
pieces. Kineto exposes only 1.671066 ms on rank 0 and 2.170872 ms on rank 1;
captured routed MoE and the 81 compiled all-reduces remain opaque. The visible
device ledger is therefore incomplete. Split-die worker affinity is neutral:
50.406626 tok/s versus 50.370643 unbound (+0.07 percent), with all coherence
and post-health gates green. The next intervention must alter a graph/runtime
boundary rather than revisit CPU pinning. Full details and the dense 27B
transfer contract are in `docs/20260826_qwen36_graph_runtime_profile.md`.

## Scope And Inventory Method

This ledger tracks every component relevant to Steve Seguin's Qwen3.6 35B-A3B
Quark W8A8 B70 result. It distinguishes executable source and launch recipes
from bulk benchmark evidence. Steve's work is prior art and evidence; the final
runtime must be rebuilt from attributed source owned by this repository.

Inspected repositories:

| Repository | Frozen revision | Tracked files | Treatment |
| --- | --- | ---: | --- |
| `b70-optimization-lab` | `523ca95b925308391707624530c29359edd05b6a` | 20,816+ | Refreshed through 2026-08-25. Read all Qwen result packets, accepted launcher/config, relevant notes, scripts, and patches; bulk JSON/log evidence is path/hash inventoried. The post-`c1cc2bf` delta is Qwen3.8 TP1/MTP work and does not alter the Qwen3.6 W8A8 control. |
| `vllm` | `44fc8fde09fc311d3099dab10366b672d9142ea4` | 5,285 | Current source inspected; closest surviving post-record checkpoint is `e190923b32e1b87fe33d08264bff9215fb7770fc` from 2026-06-16. |
| `vllm-xpu-kernels` | `2dd55f380df753a10a88fcd9e96192561066e713` | 312 | Operator bindings, dense INT8, grouped MoE, GDN, build, and Python interface are in scope. |

The lab contains 84 Qwen36 notes, 120 Qwen36 scripts, 411 Qwen36-named
patches, and thousands of result files. Reading every JSON repetition would not
add source understanding. The completeness rule is therefore: inspect every
relevant executable recipe/source/patch family and the accepted/rejected result
summary for each mechanism; inventory repetitive raw evidence without treating
it as executable software.

## Exact Accepted Identity

The safe TP2 reference used model revision
`cced56592e8c8935f8220836b4baa04dfd389118`, Quark W8A8, TP2/PP1, no MTP,
PIECEWISE graph, async scheduling, no prefix cache, and direct oneCCL P2P. Its
natural-chat request tokenized to 498 input tokens and produced 512 tokens in
5.96267 seconds of corrected decode, or 85.869114 tok/s.

Steve also retained older TP2 p512/o256 measurements at 91.592312 tok/s for
one run and 91.351052 tok/s across three runs. Those used a repetitive prompt
and have weaker validity gates. The reproduction target is therefore 85.87
tok/s minimum with natural-chat coherence, with about 91.5 tok/s as the older
same-hardware ceiling to explain.

The LocalMaxxing run `cmq9ifq0500b0r8012f27j1xl` supplied by the operator is
a different topology: TP4, four B70s, p512/o512, 32K configured context, no
MTP, PIECEWISE graph, and 99.769699 corrected output tok/s. Steve's stricter
later TP4 program reached 93.550542 tok/s. The current-program TP2 smoke and
TP4 strict values are 85.869114 and 93.550542 tok/s, so TP4 added about 8.9 percent. Older
weaker-gate values show the same scale: about 91.35 TP2 versus 99.77 TP4, or
about 9.2 percent. TP4 is a real but modest final lever; it does not explain
the local 17.06 tok/s TP2 control. The proper two-card target remains at least
85.87 tok/s.

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
FI_TCP_IFACE=eth1
CCL_KVS_IFACE=eth1
```

The accepted launcher also explicitly unset `CCL_ZE_IPC_EXCHANGE` and
`CCL_WORKER_COUNT`. Steve's `eth1` was his bare-metal interface. The equivalent
interface inside this repository's Docker network is `eth0`; matching the
semantic interface is correct, while copying the literal host name is not.

Steve did not pass a manual splitting-op list or force inductor graph
partitioning. The later launcher default made scheduling asynchronous because
`VLLM_EXTRA_ARGS='--uvicorn-log-level warning'` replaced the default
`--no-async-scheduling` argument.

## Component Status

| Component | Steve mechanism | Active in 85.87 run | Local status | Verdict |
| --- | --- | --- | --- | --- |
| Model and scales | Exact HF Quark W8A8 revision | Yes | Exact local revision and Quark loader verified | Matched. |
| Dense activation quant | `_xpu_C::per_token_quant_int8_xpu` | Yes | Operator present and selected | Matched. |
| Dense W8A8 GEMM | oneDNN `_xpu_C::int8_gemm_w8a8` | Yes | June-9 base op selects `XPUInt8ScaledMMLinearKernel`; unchanged in `122b698b` | Dense GEMM is not the direct checkpoint delta. The quantization output primitive may transfer to 27B only through a separately tested dense adapter. |
| Routed MoE | Xe2 XMX INT8 grouped GEMM with per-row activation and per-channel weight scales | Yes | August adapter and June source select `Using XPU Int8 MoE backend`; scratch-aware Python uses the native quant `_out` variant when present, otherwise allocates and copies 80 times/step. | Endpoint dispatch, coherence, and graph capture proven; native scratch execution remains unmatched. Fused SiLU+quant is unset in the accepted lane. |
| Mixed MoE workspace | BF16 and INT32 persistent scratch interface | Yes in safe TP2 label | Local env enabled | Steve measured a small full-model regression in an earlier arm; not the 5x explanation. |
| Shared expert | Native dense W8A8 linears plus shared/routed combination | Yes | Later-image ABI mismatch bridged narrowly | Coherent; source snapshot comparison pending. |
| GDN decode | Native XPU GDN decode; recurrent fallback limited to prefill | Yes | Native decode active; checkpoint source is effectively unchanged from the June-9 reconstruction | Local synchronized GDN is 3.9278 ms versus Steve's 1.5846 ms; likely includes queue/backlog effects and needs native A/B retiming. |
| GDN quant reuse | Clone-safe QKVZ/BA quant reuse | Yes | `clone` setting active | Small lever; view/partial-clone variants were rejected. |
| Fresh GDN state | Zero newly allocated recurrent state | Yes, launcher default | Added to exact local env | Correctness identity; not a 5x speed lever. |
| Graph runtime | Forced-communication PIECEWISE replay | Yes | All 9/9 full-model captures pass; replay trace observes pieces 0..40 with reported total 41; bounded profiling sees 41 fence resets, 41 host waits, and 82 queue submissions/token | Topology exactly matched. Captured MoE and 81 all-reduces remain profiler-opaque; execution inside/around replay is active. |
| Ordinary no-spec PIECEWISE key | Reuse relaxed general non-uniform key | Yes | Erroneous local uniform key removed; off-device default-size contract passes | June behavior matched; no special ordinary-decode key is required. |
| Custom collective wrapper | functional `vllm::all_reduce` custom op with one active required inner clone; nominal graph-clone flag was inert on the accepted outer-op route | Yes | Large profile clones require completion before oneCCL; clone-only profile fence passes all 81 calls while graph recording and decode remain unfenced | Cleared through exact model interleaving and post-health. |
| Collective binary | public oneCCL `4ceafd15`, ARCB, oneAPI 2025.3 | Yes | Pinned-image `542142ac...` library plus exact `0d549c35...` SPIR-V passed the local direct/graph oracle | Graph correctness is locally proven despite a non-semantic build-hash difference from Steve's later artifact. |
| Collective transport | oneCCL/OFI direct P2P in graph | Yes | Unset/default IPC resolved to `pidfd`; 256 direct and 512 XPUGraph iterations passed on both ranks | Raw transport and graph replay are cleared; the remaining failure is in the full vLLM integration. |
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

The June source also exposes two important differences from this repository's
newer shared W8A8 kernel. The June dense oneDNN op keeps a thread-local
scratchpad cache keyed by device and scratchpad size; the current shared
`kernels/int8_gemm_w8a8.h` allocates a scratch tensor on every invocation.
That is a concrete portability A/B after the exact control works, not yet a
measured endpoint bottleneck. The quantizers are not numerically identical:
June uses a fixed 256-lane row reduction, FP32 scales, `sycl::round`, and a
`1e-10` absmax floor, while the newer shared kernel uses adaptive 32-512-lane
groups, activation-dtype scales, `sycl::rint`, and a `1e-5` scale floor. Do not
silently substitute one for the other in a reproduction or quality result.

The exact independently stored reconstruction input is
`kernels/steve_qwen36_quark_w8a8_20260609.patch`, SHA256
`14c2e801da02a7b46e63940dbe41f5c0c45fabb98b3ee4c5bd03d7dc7d0b1266`.
It applies cleanly to official `vllm-project/vllm-xpu-kernels` commit
`28e1f5e74c15744b69cf3b760f6160ceabd15de0`; no Steve checkout is required.

### June source presence versus August installed dispatch

A function-level comparison against the pinned August kernel source
`2dd55f380df753a10a88fcd9e96192561066e713` found the June activation
quantizer, six-argument dense W8A8 oneDNN operator, and base Xe2 grouped W8A8
MoE source. That source result did not establish installed dispatch. A direct
off-device census of the digest-pinned package proved
`per_token_quant_int8_xpu` and `int8_gemm_w8a8` are registered, while
`cutlass_grouped_gemm_w8a8_int8_interface` is absent. Importing the later
Python MoE interface does not add it. August's extra validation, output,
scratch, barrier, offset, policy, and layerlet source therefore cannot be
treated as active until a separate complete August package is rebuilt.

There are regressions at three boundaries. Later vLLM removed the XPU INT8
scaled-mm registry candidate and the required inner functional all-reduce
clone. Its Quark method also unconditionally calls generic Triton
`fused_experts` even though the XPU INT8 MoE oracle remains in the tree. The
installed August native package separately omitted the grouped W8A8
registration. The complete June package fixed registration, but the coherent
47.54 tok/s exact-package endpoint proved that registration alone did not
change Quark dispatch. The local adapter now also restores backend selection,
weight/scale layout conversion, modular-kernel construction, and native apply.
Its wall-time leverage remains unmeasured until the next guarded endpoint run.

Two source-ownership gaps remain. The repository has no independent shared
source for the Xe2 grouped W8A8 MoE implementation or GDN implementation;
today only the attributed June patch or external runtime trees carry those
sources. The shared quantizer is also not June-equivalent: it uses adaptive
32-512-lane workgroups, activation-dtype scales, `sycl::rint`, and a `1e-5`
floor. It must be treated as a separate numeric arm, not silently substituted.

The smallest kernel campaign is therefore: census June/August/shared schemas
and XPU registrations; prove June-to-August quant and dense parity; test shared
quant separately on tiny values and rounding ties; compare dense scratch-ring,
barrier, and output variants one factor at a time; then build a complete August
package before testing grouped MoE base parity, offsets, active-expert, policy,
and reuse variants. GDN accepts only cloned quant reuse; raw sharing and views
already failed repeatability.

The current inspected native hashes are:

```text
2d931484ee0aadd4c9fb6abf494e147a5275210a216426a1eb56add0158bef0d  rebuilt June _xpu_C
f5ddc2ee3c11dcede3a7190b69d6e0dd354bb0727be7519600abaebe9fc4cd2c  rebuilt June grouped library
366935b172b5c9c3cb75bee5d7bfe0434f377a6317314a9a43c853b5d02fe83b  rebuilt June GDN library
ae330affe0315a5be4ac50478cc15c7874ae6e8fa9fa71cf64d5e5dff158968b  installed/pinned-image _xpu_C
7692db81b65be5fdb9d4509f2397d300276451ee9551d43cedb7963eaad70e4a  installed/pinned-image grouped library
cf482fd898ef965eeac70682027fe5578d5005b5eb6c51a85664a68e151a4a02  installed/pinned-image GDN library
542142aca8f3d318616eae0f300aaa47dc62b217831599cb1461212f8aa4dc76  Steve/pinned-image libccl.so.1.0
```

Steve's currently preserved `_xpu_C` and GDN files match the pinned image byte
for byte.
The image label records
`4ceafd1+2dd55f38+44fc8fde0`: oneCCL source, XPU-kernel source, and vLLM source
respectively. The installed grouped sibling exists, but the installed
`_xpu_C` does not register its W8A8 interface. Steve's June notes
state that the accepted controls used a restored 67 MB `_xpu_C`; the surviving
and pinned-image extension is 116706992 bytes. The accepted dirty binary and
its hash are not present in the refreshed lab. The exact June-16 native source
checkpoint is present, however, so source reconstruction is no longer limited
to the June-9 patch. Rebuilt artifacts establish source ownership but cannot
claim byte identity with the unavailable accepted 67 MB file.

The reconstruction audit found that source recovery is substantially stronger
than binary recovery. The public base is `28e1f5e74c15744b69cf3b760f6160ceabd15de0`.
Steve's live Git object database directly resolves checkpoint
`122b698bc245d31668a7fe5f2ad5ce1d07ba08ca`, parent `28e1f5e`, dated June 16,
and child `3ed399adc384385fed5663a27623f09ecf44e085`, dated June 19. The first is the
clean checkpoint before the accepted synchronized timing packet and contains
the complete layerlet/output-buffer work; the second adds the routed-GEMM1
B-layout fix and default-off experiments after that timing. June 9, June 14,
and exact SiLU patches also survive separately. No accepted 54 MB or 67 MB
binary, hash, or cache copy survives, so byte identity remains unavailable,
but exact source identity for the next A/B is recovered.

The build matrix must hold Python 3.12, torch 2.11 XPU, oneAPI 2025.3,
Release/Ninja, XPU/SYCL TLA/Xe2, MoE and GDN enabled, and archive `_xpu_C` plus
both grouped-GEMM and GDN sibling libraries as one ABI set. Build the June 9
minimal patch over `28e1f5e`, then the pre-exact-SiLU approximation from
`122b698`, each with `bmg-g21-a0`, the old multi-target default, and a separate
local-card AOT arm. The hypothesis that 54 versus 67 MB reflects AOT target
coverage remains an inference until the reconstructed ELF/operator census is
measured.

The first independent reconstruction completed on 2026-08-25 using the exact
minimal June 9 patch, `bmg-g21-a0` AOT, torch 2.11.0+xpu, IntelLLVM 2025.3.3,
Release/Ninja, and no GPU device. Its installed `_xpu_C` is 55,523,648 bytes
with SHA256 `2d931484...`; GDN is 2,724,136 bytes (`366935b1...`) and grouped
GEMM is 2,936,608 bytes (`f5ddc2ee...`). The extension's install RUNPATH is
`$ORIGIN`, and every dependency resolved in the pinned image. This reproduces
Steve's recorded 54 MB fresh-build class, not his unavailable accepted 67 MB
binary. It does not represent the later `122b698b` native implementation and
must not be used to dismiss that source delta as an AOT/file-size effect.

The complete materialized runtime package inherits `_C`, `_moe_C`, attention,
and support files from the digest-pinned Intel image, then replaces `_xpu_C`,
both Xe2 siblings, and the patched MoE Python dispatcher. Off-device import
proved every module originated in that package, `FUSEDMOE_AVAILABLE=True`, and
all required dense, quant, grouped-MoE, `_C`, and `_moe_C` schemas had XPU
dispatch. GPU numerical parity, capture/replay, and model serving remain open.
The machine-readable record is
`vllm/w8a8/manifests/qwen36_june_xpu_c_bmg_g21_a0_20260825.json`; the owned
fresh-build recipe is `vllm/w8a8/build_qwen36_june_xpu_c.sh`.

The preserved oneCCL cache used `CCL_ENABLE_ARCB=ON`, release mode, oneAPI
2025.3 `icx`/`icpx`, and two local compile-compatibility edits that qualify the
ESIMD barrier namespace in small all-gather and reduce-scatter. The Qwen TP
graph mainly uses all-reduce, so those two dirty edits are build fixes rather
than a decode lever. A later Steve result provides a stronger artifact gate:
the same public parent/libccl commits produced a graph-validated library hash
`43d94d43506e30096dd099b9d53b54f932be964751e92ff0cbb8d3a37fad6700`
and `kernels.spv` hash
`0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9`.
The pinned image has the exact SPIR-V hash but library hash `542142ac...`.
Compiler paths and embedded build metadata can change a library hash, so this
does not prove bad code; it does mean source-commit equality is not binary or
graph-correctness proof.

## Runtime And Image Bill Of Materials

The pinned S2B image has digest
`sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94`
and label `4ceafd1+2dd55f38+44fc8fde0`. Its relevant runtime inventory is:

| Layer | Pinned image identity |
| --- | --- |
| Base | Ubuntu 24.04, Python 3.12.3, glibc 2.39 |
| Torch | 2.11.0+xpu |
| vLLM package | 0.21.1.dev18+g8df6feb7d.xpu; source label `44fc8fde0` |
| XPU kernels | package 0.1.8.2; source label `2dd55f38` |
| Transformers | 5.8.0 |
| compressed-tensors | 0.15.0.1 |
| Triton XPU | 3.7.0 |
| oneAPI compiler/runtime | compiler packages 2025.3.3; SYCL runtime 2025.3.2; oneDNN 2025.3.0 |
| Container UMD | Intel Compute Runtime 26.14.37833.4; Level Zero loader 1.28.2 |
| oneCCL | source label `4ceafd1`; `libccl.so.1.0` 240177816 bytes, hash `542142ac...` |
| oneCCL device kernels | 5257304 bytes, exact validated hash `0d549c35...` |
| Native ops | `_xpu_C` 116706992 bytes, hash `ae330aff...`; GDN hash `cf482fd...` |

The container does not mount the host UMD libraries. It therefore runs its
26.14 user-mode driver above this host's kernel 7.1 KMD, while the host package
set is Compute Runtime 26.22.38646.4 and Level Zero loader 1.28.2. This mixed
KMD/UMD identity is intentional in the exact image control and must remain
explicit in any reproduced image manifest.

## Hardware And Topology Boundary

The systems share Arc Pro B70 GPUs, but not the same host platform:

| System/evidence | CPU and PCIe topology | Runtime evidence |
| --- | --- | --- |
| Steve Qwen35 June host | AMD EPYC 9015; four B70s on separate PCIe 5.0 x16 root ports; pairs reported `NODE` | Ubuntu 24.04.4, kernel 6.17, UMD 26.14; direct P2P enabled |
| Steve July two-card oneCCL oracle | Threadripper PRO 5955WX; two-card test | Public oneCCL direct 256/256 and XPUGraph 512/512 passed with `pidfd` |
| This host | Threadripper 1950X; the B70s sit under distinct `0000:00` and `0000:40` root complexes through separate switches; host is PCIe Gen3 era | kernel 7.1, host UMD 26.22; measured H2D 12.82 GB/s, host-staged D2D 1.68 GB/s, Torch peer-access false; exact oneCCL direct plus XPUGraph oracle passed |

This makes the direct-P2P device loss a possible platform, mapping, or
KMD/UMD interaction, even though the GPU models match. It is not evidence that
PCIe bandwidth explains the 5x decode gap: the Qwen all-reduces are small and
Steve's own graph/no-graph delta is much larger than his TP2/TP4 scaling.

## Other Steve Repositories And Published Material

The public account inventory was checked rather than assuming the optimization
lab was the only source. The relevant repositories are:

| Repository/revision | Relevant content | Transfer verdict |
| --- | --- | --- |
| `ml-bottleneck` `b5f7edbc` | Models weight/KV reads, fixed per-layer runtime, and collective latency separately; contains the 93.55 tok/s Qwen35 TP4 evidence anchor | Explanatory/calibration model only, not runtime code. Its own formulation reinforces fixed launch/coordination cost over payload bandwidth for small-active MoE decode. |
| `Unofficial-Intel-XPU-Community` `4f5b2146` | Driver/container/topology checklists and the warning that host PCIe generation changed a MiniMax four-card result | Deployment discipline, no hidden W8A8 implementation. Its PCIe4 13.79 GB/s versus PCIe5-class 27.88 GB/s evidence makes host bandwidth a plausible residual after graph works. |
| `vllm-xpu-kernels` `0fd18a7c` current versus S2B `2dd55f38` | Current fork tree differs from the S2B snapshot only in the new FP8 GEMM out-variant files; the June W8A8 kernel work is preserved in lab patches/dirty chronology | No newer hidden Quark W8A8 speed switch. Keep exact S2B control, then evaluate newer kernels separately. |
| `vllm` `5df9999f` current versus S2B `44fc8fde`/June `e190923b` | Current fork tracks later upstream and has massive API/runtime drift; no private ahead commit carries the June accepted overlay | Do not copy current main for the control. The accepted implementation is the June source/patch family already under forensic comparison. |
| `llama.cpp` Intel branch `4302fb59` | B70/Xe SYCL MMVQ routing, q8 activation reuse, GDN cross-op fusion, zero-copy/reorder experiments, and correctness poison gates | Valuable patterns for the UD-Q4_K_XL lane and generic launch/dataflow research, but not ABI-compatible with Quark W8A8 vLLM. |
| `neural.download` `9411bfc2` | Published result catalog generated from the lab evidence | Result index, not an additional runtime implementation. |

Steve's `ml-bottleneck` default B70 model assumes PCIe5/64 GB/s-class links
and about 20 us exposed PCIe-collective latency. This host's measured oneCCL
small-message floor is much higher, but even 81 Qwen graph collectives at
roughly 80-111 us account for about 6.5-9.0 ms, not the local 58.6 ms/token.
That estimate is directional because captured collectives and host-staged
microbenchmarks are not identical. It still rules out payload bandwidth as the
sole 5x cause and keeps graph/host boundary ownership first.

## Source Closure Audit

The closest surviving June vLLM snapshot changes 84 files relative to its
upstream parent (`c51df43005726a09c6eb7348e8c1b00501c70a8e`). The active
accepted path closes through these source families:

- graph admission and replay: `platforms/xpu.py`, `compilation/cuda_graph.py`,
  and `compilation/piecewise_backend.py`;
- collective graph route: `distributed/parallel_state.py` and
  `distributed/device_communicators/xpu_communicator.py`;
- dense W8A8: `model_executor/kernels/linear/` and `_xpu_ops.py`;
- routed/shared MoE: `fused_moe/experts/xpu_moe.py`, runner shared-expert
  code, and `quantization/quark/quark_moe.py`;
- Qwen/GDN: `models/qwen3_5.py`, `layers/mamba/gdn_linear_attn.py`, and the
  XPU GDN backend;
- decode boundary and sampler: `v1/worker/gpu_model_runner.py` and
  `v1/sample/sampler.py`.

The August `piecewise_backend.py` is byte-identical to the June snapshot and
the August graph wrapper is a compatible superset. `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE`
was removed as a flag, but its behavior is now unconditional for XPU because
`XpuCommunicator.ca_comm` is explicitly `None`. This is not a missing action.
The material accepted-path regressions found so far are the removed inner
all-reduce clone, the missing installed grouped W8A8 registration, and the
later Quark method's unconditional generic Triton apply. The Quark launcher
variables `VLLM_XPU_QUARK_W8A8_MOE` and
`VLLM_XPU_FORCE_QUARK_REPACK` have no implementation in either surviving
vLLM snapshot; the checkpoint scheme and Quark registry select the INT8 path.
They are preserved for launch identity but are not performance switches.

The separate August `fa-graphsafe` build specializes Qwen head-dimension-256
FlashAttention kernels and does not match the June accepted binary. It is a
later forensic artifact and must not be overlaid on the exact control.

The same chronology makes both the June-9 package and the recovered June-16
checkpoint required path controls. The 54 versus 67 MB binary difference is a
separate lower-priority residual variable. Steve
measured 87.2888 tok/s with a newly rebuilt 54 MB extension,
restored the 67 MB extension, then measured 89.9613 tok/s in a short clean
control and 92.5220 tok/s in decisive timing. The exact model control must
first establish the full June-package endpoint; only then can A/B work
attribute the smaller accepted-binary residual.

The exact model now establishes the June-9 endpoint at 48.5315 tok/s. The
directly comparable 87.2888 versus 89.9613 historical pair is only a 3.06
percent file-class difference, but that does not compare June-9 source with
checkpoint `122b698b`. Neither the current 116706992-byte build nor file size
alone identifies Steve's accepted 67 MB binary. Test the recovered source and
operator contract; do not substitute a binary-size hypothesis.

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
image, the then-current adapter, and the repository helper's forced
`CCL_ZE_IPC_EXCHANGE=pidfd`.

Result -> process-group initialization and model load succeeded, which is
farther than the old kernel failure. The first compiled custom-op all-reduce in
the profile run failed on rank 1 with `UR_RESULT_ERROR_DEVICE_LOST`. Both cards
passed the post-teardown single-card health probe.

Source finding -> the run emitted PyTorch's output-alias warning. Comparing
Steve's June source with the later image proved that the August
`parallel_state.all_reduce` removed the implementation of
`VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT`; the setting was present but inert.
The accepted source had one active required clone inside the registered op;
its separately set graph-clone flag was inert on the outer custom-op route.
The first local attributed op still did not restore that contract because it
patched `XpuCommunicator.all_reduce`, while GroupCoordinator emitted stock
`vllm::all_reduce` directly. Custom-op implementations execute with
`torch.compiler.is_compiling()` false, so that patch fell through. The
corrected adapter routes GroupCoordinator to the attributed local op without
overriding a registered torch operator. The exact launcher also removes the helper's
forced IPC-exchange setting and uses the container's active `eth0` interface,
matching Steve's unset/default IPC and active-NIC semantics.

Verdict -> kernel 7.1 did not fix the direct-P2P vLLM failure. Later source and
cache inspection placed this failure in compiled profile-run with graph mode
NONE, before any XPUGraph capture. It used August's clone-less functional op;
the first local adapter was not on that control-flow path.

### Raw oneCCL direct-plus-XPUGraph oracle

Config -> post-reboot healthy cards, Steve's later `[4,5120]` BF16 oracle
shape, two XCCL ranks, pinned-image `libccl.so.1.0` hash `542142ac...`, exact
`kernels.spv` hash `0d549c35...`, direct P2P enabled, and Steve's unset/default
IPC exchange and worker-count semantics. Docker bridge networking supplied the
required container `eth0`.

Command -> `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 IPCX=default bash
vllm/w8a8/run_qwen36_oneccl_graph_oracle.sh`.

Result -> both ranks passed 256 direct all-reduces and 512 XPUGraph replays
with zero mismatches and zero maximum absolute difference. Average iteration
time including synchronization and validation was 1.446 ms direct and 0.349
ms under graph on both ranks. The environment left `CCL_ZE_IPC_EXCHANGE`
absent; this oneCCL build reported its effective default as `pidfd`. The first
launcher attempt used host networking, had no `eth0`, and failed in OFI KVS
before any collective; bridge networking corrected the harness. Both cards
passed health after the valid run.

Verdict -> the B70 hardware, Threadripper 1950X topology, kernel 7.1, current
oneCCL library, direct Level Zero P2P, and oneCCL XPUGraph replay satisfy this
pre-model contract. The shape is not Qwen35's 2048-wide model shape, so the
next integration oracle must test that separately. Raw oneCCL does not explain
the 17.06 tok/s result or prior full-model `DEVICE_LOST`. The fault boundary is
now the vLLM custom-op wrapper, required inner-clone alias contract, compiled
execution, or worker/model lifecycle. Do not rebuild oneCCL before testing it.

### Clone-correct vLLM custom-op integration oracle

Config -> first GPU transaction after reboot, exact pinned runtime hashes,
unset/default IPC exchange and worker count, active container `eth0`, direct
P2P, corrected GroupCoordinator routing through
`vllm::s2b_all_reduce_clone`, stock dynamic Dynamo/Inductor, Qwen hidden size
2048, exact decode/profile shapes, and an attempted 81-collective stress.

Command -> `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_vllm_allreduce_graph_oracle.sh`.

Result -> both ranks loaded the expected oneCCL, `_xpu_C`, and SPIR-V hashes;
exported only the local clone op; and passed eager and compiled `[1,2048]`,
compiled `[4,2048]` and `[8192,2048]`, and 256 single-op XPUGraph replays
with no mismatch, input mutation, or output alias. The overall gate remained
red because the synthetic 81-profile-collective arm caused Inductor to retain
81 independent 32 MiB outputs and emit five 16-output Triton fan-out kernels.
Rank 1 DEVICE_LOST while autotuning the second fan-out kernel, before the
81-collective graph stage; this immediate failure was outside the custom op.
Both cards passed post-teardown health.

Verdict -> the isolated clone-correct custom-op/compiler route is cleared at
the real shapes, but the 81-op stress was model-unrepresentative and did not
pass. Replace its independent fan-out with a sequential low-live-buffer chain
before reuse. After a real reboot, use the exact interleaved model as the next
VllmBackend/PIECEWISE gate.

### Exact June-package model transaction

Config -> post-reboot healthy cards, exact model and complete June runtime
hashes, TP2/PP1, minimal `{"cudagraph_mode":"PIECEWISE"}`, vLLM's default
split operations and capture sizes through 48, maxseqs 24, direct P2P,
unset/default IPC and worker count, active container `eth0`, and a fresh cache.

Command -> `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`.

Result -> both ranks initialized, loaded the model, selected dense native W8A8,
and exposed the rebuilt June grouped W8A8/GDN package. Graph capture then hit
the dummy scheduler token-sum assertion. No UR/device error occurred and both
cards passed post-teardown health. Source comparison showed that the local
adapter, not June, added ordinary uniform PIECEWISE descriptors. Sizes
32/40/48 cannot be represented as one-token uniform schedules when maxseqs is
24. June reused the relaxed general key, whose schedules remain valid.

After removing the key and rebooting, the next exact run compiled/profiled the
model, finished graph setup, and reached endpoint health. August had filtered
all nine relaxed general captures because June's eager-prefill runtime variable
was set. June never applied that variable at capture time, and ordinary decode
reused those general graphs. First inference therefore selected an uncaptured
graph and failed with the capture monitor disabled. The client received HTTP
500; teardown was graceful, both cards passed health, and no UR/device error
occurred.

After restoring that capture policy and rebooting, the third exact transaction
captured all 9/9 graphs, reached the endpoint, passed semantic probes and both
16/16 canaries, and tore down with both cards healthy. Its exact p498/o512
metric was 47.5448 corrected tok/s, 336.710 ms TTFT, and 10.7657 seconds of
server decode. The evidence gate rejected the run because request-time Triton
`fused_moe_kernel` JIT proved the registered June grouped operator was still
not selected. Digest-pinned image Quark source SHA256 `7e4c13d2...`
unconditionally calls generic `fused_experts`.

Verdict -> graph replay is now proven in the full coherent endpoint. The
remaining immediate mismatch is the August Quark dispatcher, not operator
registration. The native adapter and no-device ABI contract pass; measure only
that source repair after another reboot, with the graph configuration unchanged.

## Accepted And Rejected Mechanism Registry

Accepted or required:

- native XPU dense W8A8 quant/GEMM;
- native Xe2 grouped W8A8 MoE;
- forced-communication PIECEWISE graph;
- the required inner custom-op clone; the separately set graph-clone flag was
  inert on Steve's accepted outer-custom-op route;
- async scheduling;
- no prefix cache for the record;
- prefill-safe GDN fallback and native decode;
- greedy XPU sampler fallback.

Rejected or diagnostic-only in Steve's Qwen lane:

- removing the required inner clone when output aliasing corrupts tokens;
- fused SiLU plus quant with changed rounding/scaling semantics;
- mixed workspace based only on microbench gains;
- RMSNorm plus INT8 fusion before matching the live FP32-weight semantics;
- GDN view or partial-clone reuse;
- small-N dense GEMM variants that did not improve endpoint speed;
- local argmax and unsafe sampling shortcuts;
- speculative/MTP results that fail state and canary gates;
- experimental full MoE layerlets not endpoint-promoted by the accepted run.

## Next Decisive Transactions

1. DONE: the locally owned exact `[4,5120]` BF16 oneCCL oracle passed 256/256
   direct collectives and 512/512 XPUGraph replays with exact loaded hashes and
   zero mismatch under Steve's unset/default IPC identity.
2. PARTIAL: the no-model vLLM integration oracle cleared exact identities,
   export, eager/compiled real shapes, mutation/alias checks, and single-op
   XPUGraph replay. Its artificial 81-way profile fan-out DEVICE_LOST in a
   generated 16-output Triton pointwise autotune, so the overall oracle and
   81-op graph stage did not pass. Replace that arm with a sequential
   low-live-buffer chain before reusing it.
3. DONE: the August-adapter native-MoE endpoint captured 9/9 graphs, passed the
   exact metric and both canaries, and tore down healthy at 45.3649 tok/s.
4. DONE: the closest surviving June vLLM source plus recovered scratch-aware
   kernel interface passed its 12-component source contract, all endpoint
   gates, and both health layers at 48.5315 tok/s. This is only +6.98 percent;
   broad source drift is not the missing 1.77x.
5. Do not rebuild oneCCL yet: the installed binary passed its mechanism gate.
   Preserve rebuilding the public artifact as a later provenance task only.
6. DONE: built-in replay tracing matched Steve's 41-piece topology. Under the
   same synchronized protocol, rank-0 model-forward is 22.6748 ms versus
   5.6946 ms, a 3.9818x execution gap.
7. DONE: exact checkpoint `122b698b` native-binary-only A/B passed at 50.3706
   tok/s versus 48.5315 tok/s for June-9 (+3.79 percent), with all coherence and
   teardown-health gates green.
8. DONE: synchronized `122b698b` timing reduces model-forward by 3.00 percent
   to 21.9944 ms; broad non-MoE labels are flat and Steve remains at 5.6946 ms.
9. PARTIAL: bounded XPU profiling exposes the exact 41-fence/41-host-wait/
   82-submit structure but not the routed MoE or 81 replay-internal TP
   collectives. Split-die worker affinity is neutral. Change one graph/runtime
   boundary mechanism at a time and use clean endpoint timing for attribution.
10. Convert the required delta into attributed local patches and a pinned image;
   do not retain a Steve checkout mount.
11. Promote recovered native changes into owned source only after schema,
   numeric, graph replay, endpoint, and timing evidence.
12. Transfer each proven graph/runtime mechanism to dense 27B one factor at a
   time. Census its own profile collective shapes and derive its clone-fence
   threshold; do not copy Qwen35's 8192-row threshold.
