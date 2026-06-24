# Why CCL_TOPO_P2P_ACCESS=1 wedges TP>1 vLLM on dual Arc Pro B70 (DEVICE_LOST)

Status: literature/web synthesis, 2026-06-24. Scope: explain WHY setting
`CCL_TOPO_P2P_ACCESS=1` for Intel oneCCL inside a vLLM tensor-parallel (TP>1)
serve on two Intel Arc Pro B70 (Battlemage / Xe2 / G31, `xe` kernel driver,
Linux 7.0) causes a hard `UR_RESULT_ERROR_DEVICE_LOST` at worker-init warmup
all_reduce, AND why the failure persists / corrupts cross-GPU oneCCL +
Level-Zero state until the `xe` driver is reloaded or the box is rebooted.
Cross-ref: `docs/P2P_GPU.md` Sections A and J (our own kernel-level analysis
and H.13/J.15/J.16 wedge observations). This file is the literature backing for
the AGENTS.md "DANGER: P2P in vLLM serve wedges the multi-GPU state" rule.

## Executive summary

The best-supported explanation is a layered failure, not one bug.
`CCL_TOPO_P2P_ACCESS=1` tells oneCCL's Level-Zero ("topo") transport to do
direct GPU-to-GPU peer copies (a `zeDeviceCanAccessPeer`-style fast path)
instead of host-staged USM copies. On our box the two B70s sit on DIFFERENT
PCIe root complexes (cross-die on a 1950X), the worst case for PCIe P2P: the
PCIe spec does not define peer forwarding across host bridges, and the
B70/`xe`/Battlemage P2P path is brand-new and only partially validated. When
oneCCL issues a peer copy across that boundary, the transfer either is not
routable or hits signal-integrity / unsupported-route errors on the wire; the
`xe` copy engine (BCS) faults and resets, the Level-Zero command queue
underneath the collective wedges, and the warmup all_reduce surfaces this to UR
as `UR_RESULT_ERROR_DEVICE_LOST` (err 20). Because the fault happens at the
hardware copy-engine / kernel-driver layer (engine reset, possible page fault,
shared cross-GPU oneCCL/Level-Zero context state) and Level-Zero has no
recovery path for a lost device, the damage is not confined to the crashing
process: the next TP>1 serve re-attaches to a `xe`/oneCCL state that is still in
a reset/faulted condition and fails identically until the driver is reloaded
(`modprobe -r xe && modprobe xe`) or the box is rebooted. Single-process /
single-GPU work survives in the common case because it never crosses the peer
boundary, though a teardown that faults the engine can still degrade a card
(our J.16). This matches the first-party Puget Systems lab result (P2P on -> PCIe
RxErr + copy-engine reset + container deadlock; they ship P2P OFF), vLLM issue
#41663 (our own report: GP fault + xe BCS engine reset at TP=2), and the
multi-process-only Battlemage Level-Zero regressions in intel/compute-runtime
issues #916 and #922.

---

## Q1. What does CCL_TOPO_P2P_ACCESS actually control inside oneCCL?

Short answer: it gates oneCCL's *Level-Zero direct-peer-copy* transport for the
GPU "topo" (topology-aware scale-up) collective path. With it ON, oneCCL tries
to move buffers GPU0<->GPU1 over PCIe peer access (no host bounce); with it OFF,
oneCCL falls back to host-staged USM copies (GPU -> host RAM -> GPU).

What is solidly established:

- For GPU buffers oneCCL's default collective algorithm is `topo` (topology
  aware), which runs an optimized intra-node "scale-up" phase. If you pick a
  non-`topo` algorithm, oneCCL "will copy the GPU buffers to the Host (CPU) and
  will run the specified algorithm." So the `topo` path is exactly the path that
  wants direct device-to-device transport.
  Source: Intel oneCCL Developer Guide, Environment Variables.
  https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-14/environment-variables.html
- Intel's own multi-GPU framing confirms two transport modes: oneCCL has
  "automatic fallback between P2P and USM memory exchange modes." That is the
  knob `CCL_TOPO_P2P_ACCESS` selects between.
  Source: Intel llm-scaler multi-GPU docs (DeepWiki mirror).
  https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism
- The Project Battlematrix / LLM-Scaler container is explicitly built around
  "multi-GPU scaling and PCIe P2P data transfers" using oneCCL, confirming the
  P2P path is a first-class oneCCL transport for B-series.
  Source: Intel Project Battlematrix.
  https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-project-battlematrix.html
- The first-party Puget Systems B70 lab states plainly that
  `CCL_TOPO_P2P_ACCESS=0` "routes inter-GPU communication through Unified Shared
  Memory via host RAM instead of direct peer-to-peer transfers," i.e. the
  variable is the P2P-vs-host-staged switch.
  Source: Puget Systems, Intel Arc Pro B70 Multi-GPU AI Inference.
  https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/

What is NOT confirmed by primary sources (flagged):

- The exact internal name and mechanism. `CCL_TOPO_P2P_ACCESS` does NOT appear
  in oneCCL's published Environment Variables reference, nor (as of the master
  checkout I grepped) in the open-source oneCCL tree under `src/`
  (https://github.com/uxlfoundation/oneCCL). The only topo-prefixed knob that
  surfaces in real configs is `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK` (used to
  disable a fabric-vertex connectivity check; appears in working B70 configs and
  in vLLM #41663). This strongly implies `CCL_TOPO_P2P_ACCESS` is an
  internal / Battlematrix-build / lightly-documented knob whose semantics we
  know empirically (Puget, Seguin, our runs) rather than from a spec page. Treat
  the "it calls `zeDeviceCanAccessPeer` and enables Level-Zero peer copies"
  description as the most plausible mechanism (it matches how oneCCL's Level-Zero
  backend selects topo transport and how Level-Zero exposes peer access), but as
  inference, not a quoted source. Level-Zero does expose peer access and
  cross-device bandwidth properties as standard extensions
  (https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_BandwidthProperties.html),
  which is the API surface such a knob would drive.

Net: CCL_TOPO_P2P_ACCESS = "use direct PCIe peer copies for the GPU topo
collective path (ON) vs host-staged USM (OFF)." Confirmed behavior; exact source
line not public.

---

## Q2. Why does enabling P2P cause DEVICE_LOST specifically on cross-root-complex Battlemage GPUs?

This is the crux, and it stacks three independent liabilities.

### 2a. Cross-root-complex PCIe P2P is undefined / default-blocked

The Linux PCI P2P DMA documentation is explicit: peer transfers are clean only
when they stay inside one PCIe hierarchy (e.g. behind one switch / one root
port). "If the P2P transaction reaches the host bridge then it might have to
hairpin back out the same root port, be routed inside the CPU SOC to another
PCIe root port, or routed internally to the SOC," and because "the PCIe
specification doesn't define the forwarding of transactions between hierarchy
domains," the kernel defaults to BLOCKING such routing; only an allow-list of
known-good CPUs permits cross-host-bridge P2P.
Source: Linux Kernel, PCI Peer-to-Peer DMA Support.
https://docs.kernel.org/driver-api/pci/p2pdma.html

Our two B70s are on DIFFERENT root complexes (GPU0 under RC `0000:00`, GPU1
under `0000:40` -- different 1950X dies). That is precisely the
cross-host-bridge / cross-hierarchy case the kernel calls undefined. AMD Zen
(family >= 0x17, which the 1950X is) IS on the kernel allow-list
(`cpu_supports_p2pdma()` since 5.9 commit `dea286bb71ba`), so policy may PERMIT
it -- but "permitted by policy" is not "physically clean across two dies over
Infinity Fabric." See `docs/P2P_GPU.md` A.2 for the whitelist-vs-IOMMU asterisk
(`6dbbd053e6` voids the whitelist "when an IOMMU is present"; we run
`iommu=pt`, unresolved whether that counts).

### 2b. The B70 / xe / Battlemage P2P path is new and only partially the right path

Linux 7.0 added `[PATCH 13/15] drm/xe: Support pcie p2p dma as a fast
interconnect` (Thomas Hellstrom), part of a multi-device SVM series aimed at
Project Battlematrix / Arc Pro B-series. But (per our `docs/P2P_GPU.md` A.1)
that patch is the HMM/SVM device-private-page migration primitive, NOT the
dma-buf collective-copy path that vLLM/oneCCL TP actually uses. So even on 7.0
the turnkey oneCCL P2P transport is leaning on a peer-copy route that is not the
freshly-landed, validated SVM path. The hardware/driver P2P support for
cross-die B70 is, charitably, immature.

### 2c. The empirical failure signature: copy-engine fault, not a clean -EINVAL

When oneCCL does attempt the peer copy under these conditions, the observed
failure is a HARDWARE/engine fault, which is why it surfaces as DEVICE_LOST
rather than a polite "P2P not supported":

- Puget (first-party, 4x B70): "Direct GPU-to-GPU peer-to-peer memory copies can
  trigger physical PCIe bus transmission errors" -> "the GPU's copy engine to
  reset and container deadlock." They traced one instance to a PCIe riser
  signal-integrity problem and ship `CCL_TOPO_P2P_ACCESS=0` as the absolute
  stability setting.
  https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- vLLM issue #41663 (OUR report, dual B70, `p2p_access:0`): TP=2 worker init
  produces `general protection fault` in the worker plus
  `xe ...: [drm] GT0: Engine reset: engine_class=bcs` on BOTH cards. The crash
  lives in the `ProcessGroupXCCL::allreduce_impl` / Level-Zero warmup path, not
  in single-GPU work. `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` suppresses the BCS
  reset but does not fix the underlying XCCL crash, which directly implicates the
  Level-Zero V2 copy-offload (the engine that would carry a peer copy).
  https://github.com/vllm-project/vllm/issues/41663
- Same class of failure on dual A770 across PCIe hierarchies: ipex-llm #13131
  (one A770 on an M.2 eGPU slot, i.e. a different PCIe branch): TP=2 gives
  `[drm] GPU HANG` then `UR_RESULT_ERROR_DEVICE_LOST`, while each GPU alone is
  fine. The reporter explicitly asks whether the cross-PCIe-hierarchy placement
  is the unsupported factor.
  https://github.com/intel/ipex-llm/issues/13131

The chain is: cross-die peer copy -> BCS copy engine faults / PCIe RxErr ->
`xe` engine reset and/or GP fault -> the Level-Zero command queue carrying the
collective is now invalid -> UR reports `UR_RESULT_ERROR_DEVICE_LOST` (err 20)
to the all_reduce warmup. DEVICE_LOST is the generic "the device handle this
context held is no longer alive" signal, which is exactly what an engine reset
under an in-flight queue produces.

### 2d. Why the raw mp.spawn allreduce microbench survives P2P=1

Our observation that a hand-rolled `mp.spawn` allreduce works with P2P=1 while
the vLLM path dies is consistent with the sources: the failures are specific to
the multi-process, concurrently-initialized, oneCCL/XCCL <-> vLLM-multiproc
worker context. compute-runtime #922 nails this distinction: on a B70 (BMG-G31,
xe), single-process SYCL runs at 94% of peak, but "all multi-rank MPI SYCL
workloads crash during Level Zero initialization" -- "strictly multi-process
regression -- indicates collective state management issue." A microbench that
uses a simpler launch / collective setup can dodge the broken concurrent-init +
topo-transport combination that vLLM's ProcessGroupXCCL hits.
https://github.com/intel/compute-runtime/issues/922

---

## Q3. Why does the failure PERSIST and corrupt state (require driver reload)?

Three reinforcing reasons, in increasing order of how directly the sources
support them:

1. Level-Zero has NO recovery path for a lost device. PyTorch issue #177714
   documents that on XPU, `UR_RESULT_ERROR_DEVICE_LOST`,
   `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, and `UR_RESULT_ERROR_OUT_OF_RESOURCES`
   are FATAL Level-Zero errors that terminate the whole process rather than
   raising a recoverable exception -- the device state is treated as
   irrecoverable within that execution. So nothing in the user-space stack
   "heals" a lost device; it can only die.
   https://github.com/pytorch/pytorch/issues/177714
   (Note err 40 = OUT_OF_RESOURCES and err 20 = DEVICE_LOST are both in this
   fatal set -- consistent with our J.16 where the wedge sometimes presents as
   OUT_OF_RESOURCES / OOM-class rather than DEVICE_LOST.)

2. The damage is at the kernel-driver / hardware engine level, which outlives
   the process. The faults the sources show are `xe GT0: Engine reset:
   engine_class=bcs` and GP faults / page faults -- driver-side engine resets,
   not user-space exceptions. An engine reset (and any associated GPU page fault)
   leaves the `xe` GT in a reset/degraded state and can leave queue/fence state
   that the NEXT process re-attaching to the same device inherits. Battlemage
   `xe` engine-reset storms that escalate to reset loops needing intervention are
   independently reported (Arch BBS, darktable #20257), confirming `xe` engine
   resets on Battlemage are not always self-clearing.
   https://github.com/vllm-project/vllm/issues/41663
   https://bbs.archlinux.org/viewtopic.php?pid=2298888
   https://github.com/darktable-org/darktable/issues/20257

3. The corrupted state is SHARED cross-GPU oneCCL / Level-Zero context state, so
   it poisons later TP>1 runs specifically. compute-runtime #916 (two A770 under
   one Level-Zero context) shows a multi-DEVICE Level-Zero context is a real
   shared object with its own bugs: USM device alloc fails
   (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) ONLY when the context spans two GPUs;
   restricting to one GPU via `ONEAPI_DEVICE_SELECTOR=level_zero:0` fixes it.
   compute-runtime #922 shows concurrent multi-rank `zeInit` on B70 hitting a
   shared-state abort (`resource_info.cpp:15`). Together these establish that the
   cross-GPU Level-Zero/oneCCL collective context is a fragile shared resource on
   Arc/Battlemage, and corrupting it (via the engine reset in 2c) is exactly the
   kind of state that a fresh process cannot rebuild cleanly -- hence "every
   subsequent TP>1 serve also fails."
   https://github.com/intel/compute-runtime/issues/916
   https://github.com/intel/compute-runtime/issues/922

Why a driver reload (`modprobe -r xe && modprobe xe`) or reboot fixes it: those
are the only operations that re-initialize the `xe` GT / engines and tear down
the leaked Level-Zero/oneCCL fabric+context state from scratch. This matches our
own confirmed recovery (AGENTS.md: reboot clears it; lighter option is reloading
`xe` with no `/dev/dri` in use).

Caveat / honest flag: NO single source spells out "oneCCL caches a broken
topology in shared memory across processes." That specific mechanism
(topology-cache poisoning) is plausible and consistent, but the directly
evidenced mechanisms are (a) engine reset / GP fault at the `xe` layer and (b)
multi-GPU Level-Zero context fragility. Our own J.15 note that even
`P2PACCESS=0` TP=2 serves then fail after a string of TP>1 worker-init crashes is
best explained by (a)+(b), i.e. driver/engine-level corruption, not necessarily a
oneCCL env-cache. See "Confirmed vs Speculative."

---

## Q4. Known bug reports / threads

Directly on point (Battlemage / Arc, P2P, DEVICE_LOST, multi-GPU):

- vLLM #41663 -- dual Arc Pro B70 TP=2: GP fault + xe BCS engine reset; the
  canonical reproduction (ours).
  https://github.com/vllm-project/vllm/issues/41663
- intel/ipex-llm #13131 -- dual A770 TP=2 across PCIe hierarchies: GPU HANG +
  UR_RESULT_ERROR_DEVICE_LOST; single-GPU fine.
  https://github.com/intel/ipex-llm/issues/13131
- intel/compute-runtime #922 -- Xe2/BMG-G31 (Arc Pro B70, xe, Ubuntu 26.04,
  kernel 7.0): multi-rank MPI + Level-Zero abort at `resource_info.cpp:15`;
  strictly a multi-process regression (CR 26.05 -> 26.14). Same hardware/kernel
  family as us.
  https://github.com/intel/compute-runtime/issues/922
- intel/compute-runtime #916 -- dual A770 single Level-Zero context: USM device
  alloc fails only when the context spans 2 GPUs (CR 25.40 -> 26.09 regression);
  device-selector to one GPU is the workaround.
  https://github.com/intel/compute-runtime/issues/916
- intel/llvm #15841 -- Level-Zero P2P (`p2p_access.cpp` e2e test) returning
  -995 "device does not support the called function" on a Data Center GPU Max;
  shows Level-Zero peer-access support is patchy even on data-center parts.
  https://github.com/intel/llvm/issues/15841
- pytorch/pytorch #177714 -- XPU Level-Zero DEVICE_LOST / OUT_OF_RESOURCES are
  fatal, non-recoverable RuntimeErrors (explains the no-clean-error behavior).
  https://github.com/pytorch/pytorch/issues/177714

First-party / vendor:

- Puget Systems B70 multi-GPU lab -- recommends `CCL_TOPO_P2P_ACCESS=0`;
  documents P2P-on -> PCIe RxErr + copy-engine reset + deadlock; also gives the
  companion env vars (`VLLM_WORKER_MULTIPROC_METHOD=spawn`,
  `NEO_ReadDeviceBinaryBuiltins=0`, `ZES_ENABLE_SYSMAN=0`).
  https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- Intel Project Battlematrix / LLM-Scaler -- the supported path that DOES use
  oneCCL PCIe P2P (so Intel intends P2P to work on validated B-series configs).
  https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-project-battlematrix.html
- intel/llm-scaler multi-GPU docs -- documents oneCCL "automatic fallback
  between P2P and USM memory exchange modes."
  https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism

Contradicting / nuance: Steve Seguin's b70-optimization-lab runs
`CCL_TOPO_P2P_ACCESS=1` successfully and faster on his B70 box (see
`docs/P2P_GPU.md` B.1), so P2P is not universally broken on B70 -- it correlates
with PCIe topology / signal integrity (his fabric vs our cross-die 1950X vs
Puget's riser). https://github.com/steveseguin/b70-optimization-lab

Note on the docs: I could NOT find `CCL_TOPO_P2P_ACCESS` in the official oneCCL
Environment Variables reference or in the open oneCCL `src/` tree -- it is
effectively undocumented publicly. The Intel oneCCL env-var doc page also
returned no entry for it.
https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-14/environment-variables.html

---

## Q5. Is there a correct / safe way to enable P2P on these cards today?

For OUR box (cross-die 1950X, PCIe Gen3, ACS override, IOMMU off/pt): the
evidenced-stable answer is NO for turnkey vLLM TP -- host-staged (`=0`) is the
only path the sources show as reliable on this topology. Reasons and options:

1. Keep `CCL_TOPO_P2P_ACCESS=0` (host-staged USM). Puget calls the stability
   gain "absolute" with only microsecond host-RAM round-trip cost, and shows TP
   still scales near-2x (host-staging is not the throughput bottleneck; graph
   breaks around collectives are -- see `docs/P2P_GPU.md` B). This is the
   default and what AGENTS.md mandates.
   https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/

2. If you must A/B P2P=1, pair it with the partial-mitigation env vars that the
   sources show reduce (not eliminate) the engine fault: `CCL_ENABLE_SYCL_KERNELS=0`,
   `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`,
   `CCL_ALLREDUCE=ring`, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0` -- the stable
   fallback combo from vLLM #41663. Note `CCL_ALLREDUCE=ring` forces the
   host-copy algorithm, so it partly defeats the point of P2P.
   https://github.com/vllm-project/vllm/issues/41663

3. The genuinely correct topology fix (per the kernel P2P doc and DGX practice)
   is to put both GPUs behind ONE PCIe switch / one root port so peer TLPs never
   cross a host bridge -- a Gen5 PCIe switch (Broadcom/PLX Atlas, Microchip
   Switchtec) with clean signal integrity. That moves us into the
   kernel "behind the same root port" permitted+clean branch and is how
   Battlematrix-validated boxes are wired. See `docs/P2P_GPU.md` Section C/J for
   the composable-fabric plan. This is hardware, not an env var.
   https://docs.kernel.org/driver-api/pci/p2pdma.html

4. Software-stack hygiene that helps multi-GPU init regardless of P2P:
   `VLLM_WORKER_MULTIPROC_METHOD=spawn` (avoid fork-unsafe SYCL/L0 context),
   `NEO_ReadDeviceBinaryBuiltins=0`, `ZES_ENABLE_SYSMAN=0` (Puget). And track CR
   versions: #916 and #922 are version-sensitive regressions, so the "safe" CR
   build matters as much as the env var.

5. Mandatory discipline given the persistence bug: never chain two P2P=1 (or two
   crash-prone TP>1) serve attempts without an `xe` reset between them, and
   probe a single-card matmul for health after any TP>1 teardown that threw
   DEVICE_LOST in shutdown (our J.16). This is the operational rule, not a fix.

Bottom line: on validated Battlematrix-class hardware (single switch, retimed
Gen5, Intel's BOM) P2P is the intended fast path; on our cross-die 1950X it is
not safely enableable today -- host-staged is the stable production path.

---

## Confirmed vs Speculative

Confirmed by primary sources:
- `CCL_TOPO_P2P_ACCESS=0` routes inter-GPU traffic through host RAM (USM);
  `=1` enables direct GPU-to-GPU PCIe peer transfers. (Puget; llm-scaler)
- oneCCL GPU `topo` algorithm is the direct/scale-up path; non-topo copies via
  host. (Intel oneCCL docs)
- Cross-root-complex / cross-host-bridge PCIe P2P is undefined by spec and
  default-blocked by Linux except for an allow-list. (kernel p2pdma doc)
- On dual B70/A770 TP=2, the failure is a `xe` BCS engine reset + GP fault / GPU
  HANG surfacing as `UR_RESULT_ERROR_DEVICE_LOST`, in the XCCL/Level-Zero warmup
  path; single-GPU is unaffected in the common case. (vLLM #41663, ipex-llm
  #13131)
- DEVICE_LOST / OUT_OF_RESOURCES on XPU are fatal, non-recoverable Level-Zero
  errors with no user-space recovery. (pytorch #177714)
- Multi-GPU Level-Zero contexts on Arc/Battlemage are fragile shared resources
  with multi-process-only regressions. (compute-runtime #916, #922)
- P2P=1 is NOT universally broken on B70; it correlates with PCIe
  topology/signal integrity. (Seguin lab vs Puget)
- `CCL_TOPO_P2P_ACCESS` is undocumented in oneCCL's public env-var reference and
  absent from the open oneCCL master `src/` tree. (Intel oneCCL doc; my grep)
- Engine reset on Battlemage `xe` is not always self-clearing. (Arch BBS,
  darktable #20257)

Speculative / inferred (flagged, not directly quoted):
- That `CCL_TOPO_P2P_ACCESS=1` specifically calls `zeDeviceCanAccessPeer` /
  enables Level-Zero IPC peer access. Plausible and mechanism-consistent, but no
  source line shows it; the var is undocumented.
- That oneCCL CACHES a broken topology in shared/persistent state across
  processes (a topology-cache-poison mechanism). The persistence is real and
  confirmed empirically (ours), but the evidenced cause is `xe` engine/driver
  corruption + Level-Zero context fragility, not specifically a oneCCL env-cache.
- The exact internal route the peer copy takes on Linux 7.0 (SVM device-private
  migration vs a dma-buf path). Our `docs/P2P_GPU.md` A.1 argues the landed
  `drm/xe` P2P patch is the SVM primitive, not the oneCCL collective path; the
  collective path's exact backing is not fully pinned by public sources.
- Whether `iommu=pt` voids the AMD-Zen P2P whitelist on our box
  (`pci_p2pdma_whitelist_valid`). Unresolved in the public record; testable.

---

## References

- Linux Kernel, PCI Peer-to-Peer DMA Support: https://docs.kernel.org/driver-api/pci/p2pdma.html
- Intel oneCCL Developer Guide, Environment Variables: https://www.intel.com/content/www/us/en/docs/oneccl/developer-guide-reference/2021-14/environment-variables.html
- oneCCL source (master): https://github.com/uxlfoundation/oneCCL
- Level-Zero Bandwidth Extension Properties (peer/bandwidth API surface): https://oneapi-src.github.io/level-zero-spec/level-zero/1.11/core/EXT_Exp_BandwidthProperties.html
- Puget Systems, Intel Arc Pro B70 Multi-GPU AI Inference Performance: https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- Intel Project Battlematrix: https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-project-battlematrix.html
- intel/llm-scaler Multi-GPU and Parallelism (DeepWiki): https://deepwiki.com/intel/llm-scaler/2.5-multi-gpu-and-parallelism
- vLLM issue #41663 (dual B70 TP=2 GP fault + xe BCS reset): https://github.com/vllm-project/vllm/issues/41663
- intel/ipex-llm issue #13131 (dual A770 TP=2 DEVICE_LOST): https://github.com/intel/ipex-llm/issues/13131
- intel/compute-runtime issue #922 (BMG-G31 multi-rank Level-Zero abort): https://github.com/intel/compute-runtime/issues/922
- intel/compute-runtime issue #916 (dual A770 USM alloc fails on 2-GPU L0 context): https://github.com/intel/compute-runtime/issues/916
- intel/llvm issue #15841 (Level-Zero P2P unsupported, -995): https://github.com/intel/llvm/issues/15841
- pytorch/pytorch issue #177714 (XPU L0 DEVICE_LOST fatal/non-recoverable): https://github.com/pytorch/pytorch/issues/177714
- Arch BBS, Intel Arc engine-reset loop: https://bbs.archlinux.org/viewtopic.php?pid=2298888
- darktable issue #20257 (Battlemage xe engine reset): https://github.com/darktable-org/darktable/issues/20257
- steveseguin/b70-optimization-lab (P2P=1 works on his fabric): https://github.com/steveseguin/b70-optimization-lab
- Internal: docs/P2P_GPU.md Sections A, B, J; AGENTS.md "DANGER: P2P in vLLM serve" rule.
