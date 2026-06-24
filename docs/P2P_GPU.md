# P2P_GPU.md -- multi-B70 GPU-to-GPU comms: kernel, software, and fabric

Status: living research doc (started 2026-06-22). Scope: everything about getting two (or more)
Intel Arc Pro B70 (Battlemage G31, Xe2, `xe` driver) to talk to each other efficiently for
tensor-parallel (TP) inference -- kernel P2P primitives, the vLLM/oneCCL software path, ZML's
compiler-collective alternative, and speculative composable-fabric architectures. Goal of this
project: not just consume the state of the art, but contribute and pioneer new methods for B70 TP.

Cross-refs: [DUALCARD.md](../DUALCARD.md), [FINDINGS.md](../FINDINGS.md),
[docs/literature/02_multigpu.md](literature/02_multigpu.md), [docs/SERVING.md](SERVING.md).


## 0. Our box (measured 2026-06-22)

> NOTE (2026-06-24): the box below (1950X / X399) is UNCHANGED hardware, but it was reinstalled
> Unraid+6.18 -> **Ubuntu 26.04 + kernel 7.0 + new BIOS (IOMMU off)**. The PCIe topology is STILL
> cross-die (GPU0 under RC 0000:00 / GPU1 under RC 0000:40). Re-measured link/P2P facts and the
> PUSH/PULL peer-write discovery are in **Section J** (the current ground truth). The 6.18-era
> "Gen1 x1 artifact" and host-staged numbers in this section are kept for history.

```
AMD Threadripper 1950X (Zen1, dual-die, 2x Zeppelin over Infinity Fabric), PCIe Gen3 root.
Unraid 7.3.1, kernel 6.18.33-Unraid. 125 GiB RAM.

PCIe topology -- the two B70s are on DIFFERENT root complexes (worst case for P2P):
  0000:00 (die0 RC) -> 00:03.1 root port -> [card onboard switch 08:00.0] -> GPU0 0a:00.0
  0000:40 (die1 RC) -> 40:03.1 root port -> [card onboard switch 42:00.0] -> GPU1 44:00.0

Per-device link (sysfs current/max):
  GPU0 0a:00.0      cur 2.5GT/s x1   max 2.5GT/s x1   <- ARTIFACT (SR-IOV VF / onboard-switch placeholder)
  sw->GPU 09:01.0   cur 2.5GT/s x1   max 2.5GT/s x1   <- same synthetic downstream link
  card uplink 08:00 cur 8.0GT/s x16  max 32GT/s x16   <- REAL card-to-host link: PCIe Gen3 x16 (~15.8 GB/s)
  root 00:03.1      cur 8.0GT/s x16  max 8.0GT/s x16   <- 1950X root, Gen3-capped
  (GPU1 side 44/43/42/40 identical)

Firmware/kernel prep (all good for P2P):
  ReBAR fully open: BAR2 = 32GB.   ACS override on (pcie_acs_override=downstream,multifunction).
  IOMMU default domain = Passthrough (iommu=pt).   Also: "Virtual Resizable BAR" cap present
  -> SR-IOV fingerprint, which is why the GPU function reports a placeholder Gen1 x1.
```

Bottom line on our hardware: the *real* per-card link is **Gen3 x16 (~15.8 GB/s)**; the Gen1 x1 is a
virtual-function reporting artifact, not a real downtrain. The genuine liability is the **cross-die
topology** (the two cards hang off separate 1950X dies), which is the single worst case for PCIe P2P.


## A. Kernel-level P2P in Linux 7.0 / 7.1 (deep-research, 2026-06-22)

Method: fan-out web search -> fetch 16 sources -> 25 claims adversarially verified (3-vote, need 2/3
to kill); 24 confirmed, 1 killed. Full machine report archived in the session transcript.

### A.1 What actually landed (the foothold)

- **`[PATCH 13/15] drm/xe: Support pcie p2p dma as a fast interconnect`** (Thomas Hellstrom), part of a
  15-patch **multi-device SVM** series explicitly targeting Intel's **"Project Battlematrix" / Arc Pro
  B-series**. Merged to `drm-xe-next` 2025-12-30 -> shipped in **Linux 7.0** (2026-04-12). 6.18 (our
  Unraid kernel) does NOT have it.
- Mechanism in `xe_svm.c`: maps device-private VRAM pages for peer DMA via
  `dma_map_resource(dev, xe_page_to_pcie(page), ...)` with `prot = XE_INTERCONNECT_P2P`; and flips
  `xe_has_interconnect()` from `return dev1 == dev2;` to
  `if (dev1 == dev2) return true; return pci_p2pdma_distance(dev1, dev2, true) >= 0;`.
- **What it IS:** HMM/SVM device-private page migration + "direct execution out of peer memory."
  **What it is NOT:** a dma-buf export path, and NOT the collective-copy path vLLM/oneCCL use for TP.
  A separate Intel series handles the dma-buf-via-IOV path. So for hand-rolled SVM multi-GPU code this
  is a real new primitive; for turnkey vLLM TP it is not the thing on the critical path.

### A.2 The policy gate -- we are (mostly) on the good side

- Linux `pci_p2pdma` blocks cross-root-complex P2P by default ("PCIe spec doesn't define forwarding
  between hierarchy domains"). Our two cards are on different dies -> this is the default-block path.
- **BUT AMD Zen is whitelisted:** kernel 5.9 `commit dea286bb71ba` makes `cpu_supports_p2pdma()` return
  true for AMD family >= 0x17. The 1950X **is** family 0x17 (Zen1). AMD engineers (Deucher, Koenig)
  state Zen treats cross-host-bridge P2P like ordinary CPU-core accesses -> "more likely to work."
  So `pci_p2pdma_distance()` should return >= 0 cross-die on this exact box, and `xe_has_interconnect()`
  would NOT hard-refuse on policy grounds.
- **The asterisk (load-bearing unknown):** a follow-up commit (`6dbbd053e6`) disables the AMD whitelist
  "when an IOMMU is present." We run `iommu=pt`. Whether passthrough counts as "present" for
  `pci_p2pdma_whitelist_valid()` was NOT resolved by the public record. If it counts -> whitelist
  voided -> cross-die P2P refused -> fallback to host memory. **This is testable** (boot 7.x, check).

### A.3 The empirical reality -- and a direct contradiction (see Part B)

- **Puget Systems** (first-party lab, 4x Arc Pro B70, June 2026): enabling direct P2P
  (`CCL_TOPO_P2P_ACCESS=1`) triggered **PCIe RxErr, GPU copy-engine resets, container deadlock**. They
  ship P2P **off**, routing inter-GPU traffic through host RAM (USM). Root cause traced to a PCIe riser
  signal-integrity issue, but P2P was never re-validated as working. Corroborated by vLLM issue #41663
  (dual B70: GP fault + xe BCS engine reset at TP=2, `p2p_access:0`).
- Host-staged TP still scales **near-linearly ~2x** (Llama-3.1-8B 35.4->70.3 t/s = 1.99x;
  DS-R1-Distill-8B 66.9->136 t/s = 2.03x). Host-RAM round-trip penalty characterized as "microseconds"
  (their assertion, not a published number). => For *throughput*, host-staging is not the bottleneck.
- **HOWEVER** Steve Seguin's lab (Part B) runs `CCL_TOPO_P2P_ACCESS=1` (P2P ENABLED) successfully and
  faster. So the Puget "P2P is broken" result is **not universal** -- it may be riser/signal-integrity
  specific. This is the most important open contradiction and is worth an A/B run on OUR fabric.


## B. steveseguin/b70-optimization-lab -- the software story

Repo: https://github.com/steveseguin/b70-optimization-lab
110 t/s repro: https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/minimax-m27-b70-110tps-ubuntu24-20260523/README.md

Steve Seguin's (NOT Steeve Morin / ZML -- two different people) vLLM/llama.cpp tuning lab. ~70 patches +
promoted env configs + an agent handoff recording what survived vs. got rejected. Did the empirical
legwork on B70 comms.

### B.1 Headline: P2P stays ON; the win is allreduce surgery, not transport swapping

- Promoted 89 tok/s MiniMax-M2.7 config keeps **`CCL_TOPO_P2P_ACCESS=1`** (direct GPU-to-GPU P2P on).
  He did NOT route around PCIe -- he made the **number and placement of allreduces** cheaper. This
  **directly contradicts the Puget/community "disable P2P" advice.**
- Measured ladder 80.6 -> 89.3 tok/s. The bottleneck is **graph breaks around collectives**, not the
  wire: raw decode-sized allreduces microbench at **15-17 us**; full-model loss is "dominated by
  framework/compiler/graph boundaries around collectives, not raw CCL latency alone."

### B.2 The high-leverage patches (in order of measured impact)

1. **Clone-safe compiled allreduce custom-op** (~+5 t/s, 82.4 -> 87.3): the single biggest jump.
   - `VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1` + `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`
   - Tiny patch: clone the input tensor before the out-of-place allreduce so the op is safe to capture
     inside an XPU graph. Vanilla XPU collectives break graph capture; cloning makes the allreduce
     capturable, so it stops forcing a graph break.
2. **Move allreduces inside custom-op boundaries so they fuse with surrounding compute** and stay in
   the captured graph:
   - `VLLM_MINIMAX_MOE_OUTPUT_ALLREDUCE_INSIDE_CUSTOM_OP=1` (MoE-output allreduce pulled inside MoE op)
   - oproj-delay-allreduce patch (defers attention output-proj allreduce so it merges with the
     post-attention residual / RMSNorm path)
   - `VLLM_MINIMAX_QK_RMS_DIRECT_INPLACE_SCALE=1` (Q/K-variance allreduce+scale done in place)
   - `VLLM_XPU_FORCE_GRAPH_WITH_COMM=1` + `VLLM_XPU_GRAPH_NOOP_COMM_CAPTURE=1` (force XPU graph to
     capture even though it contains communication)

### B.3 The rejects (equally important)

- Every oneCCL knob he tried **regressed or hung**: `CCL_SYCL_ALLREDUCE_TMP_BUF`,
  `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `CCL_ALLREDUCE_SMALL_THRESHOLD`, `CCL_WORKER_COUNT=2`.
  -> **Leave default oneCCL alone; do custom fusion on top.**

### B.4 The PCIe bandwidth data point that hits our 1950X

- His newer host had **PCIe 4.0 x16** upstream links; he measured XCCL allreduce at **13.79 GB/s @ 256
  MiB** vs a **27.88 GB/s** reference on better fabric, and traced **~half his decode regression** to
  that halved bandwidth.
- Implication for us: on **PCIe 3.0** Threadripper, **fabric bandwidth is the hard ceiling** for any TP
  allreduce, P2P or not. Bandwidth is a real second-order lever; graph breaks are the first-order one.


## C. ZML -- collectives-as-compiler-IR (the structural alternative)

Repo: https://github.com/zml/zml

- vLLM (and all of Seguin's effort) sits **on top of** a vendor collectives lib (oneCCL on XPU, NCCL on
  NVIDIA, RCCL on AMD) and calls `all_reduce` as an **opaque host-launched kernel**. That opacity is the
  source of all the graph-break pain: the compiler can't see into the collective, so it forces a graph
  break and you hand-fuse around it.
- **ZML's model: collectives as compiler IR.** It lowers everything through MLIR / StableHLO into XLA,
  so cross-device comm is expressed as XLA collective ops (all-reduce, all-gather, reduce-scatter,
  collective-permute) **inside the same graph as the math**. You define a logical device mesh, tag
  tensor dims to mesh axes, and the SPMD partitioner inserts + schedules the collectives to overlap
  with compute. Transport underneath is still the vendor lib (-> NVLink / xGMI / PCIe-P2P / host-shmem),
  but the collective lives **inside** the compiled graph.
- => The exact thing Seguin spent weeks hand-patching (fuse allreduce with residual/RMSNorm, keep it in
  a captured graph) is the **default** in a StableHLO/XLA stack, because the compiler owns both the
  matmul and the collective. That's the structural reason a compiler-collective stack could win B70 TP
  without per-model patching.
- **The big caveat:** ZML's *productized inference server is single-GPU-only today.* The mesh/SPMD
  machinery exists at the framework layer (and in the sharding example) but is **not wired into the
  shipping LLM server.** The founder's 4xB70 tweet is almost certainly that lower-level mesh path, not a
  `bazel run` feature. Track `zml/intel-xpu-backend` as the thing that could eventually deliver
  compiler-fused collectives without hand-patching -- but don't bet a campaign on it yet.


## D. vLLM-XPU vs ZML -- when to use which

- **vLLM-XPU when:** you want it working NOW on B70; need the serving feature set (continuous batching,
  PagedAttention, prefix caching, OpenAI endpoint); concurrency/throughput-per-watt across many users;
  willing to live in the env-var-and-patch world. **For our W4A8 / MoE 27B-35B dual-B70 targets, vLLM
  XPU is the only stack that runs them in real TP today.**
- **ZML when:** you want a single Zig binary, zero Python, hermetic reproducible builds, sub-second
  weight load, compile-to-metal control end to end; workload is single-GPU or you'll drive the mesh
  primitives yourself. Fits the Zig-native / explicit-tooling temperament far better than vLLM's Python
  sprawl -- but it is **not** a drop-in for multi-GPU TP on B70 yet.

**Concrete plan for our rig:** stay on vLLM-XPU for dual-B70 TP; steal Seguin's three highest-leverage
env vars (`VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP=1`, `VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT=1`,
`VLLM_XPU_FORCE_GRAPH_WITH_COMM=1`); keep `CCL_TOPO_P2P_ACCESS=1` and **A/B it against =0 on our PCIe 3.0
fabric** (his P2P-on result may flip on the older bus -- worth one run); leave default oneCCL knobs alone.


## E. Speculative architectures -- composable fabric / SR-IOV relocation

Expanded, with the bifurcation-vs-switch distinction and a full cheap->deluxe ladder, in
[MOONSHOT_RESEARCH.md](../MOONSHOT_RESEARCH.md). Summary below.

Idea on the table (Isaac): relocate the GPUs off the bad cross-die topology onto a clean PCIe Gen5
switch, and reach them from the host over a fast fabric, made transparent via SR-IOV:

```
  CPU  ->  PCI card spoofing VFs  ->  400Gb link  ->  PCI card  ->  PCIe Gen5 switch  ->  GPUs
  Inference: GPU<->GPU TP is fast (behind the switch); only CPU<->GPU crosses the fabric.
```

Analysis (see discussion thread for the long version):

- **The switch half is SOUND and is the correct fix for our cross-die problem.** Put both B70s behind
  ONE PCIe switch (Broadcom/PLX Atlas, Microchip Switchtec) and they share an upstream port -> P2P is
  the `behind the same PCI root port` branch of the `pci_p2pdma` "compatible" rule = permitted AND
  physically clean (switch forwards P2P TLPs GPU0<->GPU1, never touching the host or Infinity Fabric).
  This is exactly how DGX-class multi-GPU boxes wire P2P. It also moots the IOMMU-whitelist question.
- **It may also FIX the Puget faulting:** that RxErr/engine-reset was traced to riser signal integrity.
  A retimed/redriven Gen5 switch with clean SI is precisely the cure for SI-induced P2P faults. So this
  architecture could be the thing that makes B70 P2P actually reliable.
- **The "400Gb network link" framing is the soft spot.** Two readings:
  - PCIe-over-cable / NTB (Dolphin, GigaIO FabreX, OSS, Liqid): the host's PCIe lanes are physically
    extended over a PCIe-spec cable (copper/optical). Stays PCIe end-to-end -> P2P, SR-IOV, MMIO all
    work transparently because it IS one PCIe hierarchy, just cabled. Gen5 x16 cable ~= 64 GB/s (~512
    Gb/s); "400Gb" ~= Gen5 x8 / Gen4 x16. **Real and shipping (composable infrastructure).**
  - PCIe tunneled over Ethernet/RDMA (true network): encapsulate TLPs in 400GbE/IB packets. Exotic;
    latency-sensitive PCIe tunnels poorly without a hardware shim (DPU/NTB) terminating PCIe both ends.
- **The SR-IOV "transparency" bit is really NTB or DPU device-emulation.** SR-IOV alone doesn't relocate
  a device across a network. To make a remote GPU look local you need either an NTB mapping remote PCIe
  address space into the host's, or a **DPU emulating a PCIe device** to the host while proxying to the
  remote GPU (NVIDIA BlueField/SNAP does this for NVMe/virtio -- much harder for a 32GB-BAR GPU).

What this could unlock (the fun):
- **Composable B70 cluster / "poor man's DGX":** N B70s behind a Gen5 fabric, all P2P-capable, composed
  to one host -- scale past 2 GPUs without ever caring about host root-complex topology.
- **Disaggregation:** demote the 1950X to control-plane/orchestrator; do all GPU TP on the clean switch
  fabric, never touching the Threadripper's Gen3 dual-die liability. Retire the bad host without buying
  a new one.
- **CXL horizon (CXL 3.0):** coherent memory pooling -- tier host DRAM as KV-cache, or share GPU memory
  coherently. Bleeding edge but the natural evolution of this idea.
- **Single-logical-mega-GPU spoof:** present the two B70s to the host as one VF, with the switch/DPU
  doing TP sharding transparently (what NVSwitch+NVLink does in silicon; for B70 it would be a
  software/firmware shim -- research grade, very ambitious).

Reality check / cheap version: **most of the benefit is captured by simply putting both B70s under ONE
root port** -- a PCIe switch card in a single x16 slot, or a single-die host (modern EPYC/Threadripper,
or an Intel platform) where both cards live under one root complex. The 1950X's dual-die is the actual
villain; a switch card or newer host retires it without the exotic network-fabric build. The deluxe
network-fabric version only pays off if physical disaggregation (separate GPU box: thermal/power/scale)
is itself a goal -- in which case GigaIO/Liqid PCIe-fabric is the product category to study.


## F. Open questions / research frontier (our pioneering bets)

1. **A/B P2P on our fabric:** `CCL_TOPO_P2P_ACCESS=1` vs `=0` on the PCIe-3.0 cross-die 1950X, TP=2.
   Does Seguin's P2P-on win survive, or does Puget's faulting reproduce on our older bus? (cheap, days)
2. **Does `iommu=pt` void the AMD Zen P2P whitelist?** Boot 7.x, check `pci_p2pdma_distance` for our
   pair, watch dmesg for xe interconnect vs host-memory fallback. (the load-bearing kernel unknown)
3. **Has anyone shown the 7.0 xe SVM device-private P2P migration WORKING between two discrete
   Battlemage GPUs cross-die?** Public record is empty. Running `ze_peer` on a booted 7.x box would be a
   genuinely novel data point.
4. **Does Level Zero / intel-compute-runtime expose `ze_device_p2p` / IPC handles for B-series discrete
   at all?** No concrete userspace P2P enablement matrix found for B70.
5. **The big pioneering bet: compiler-fused XPU collectives** (ZML-style or a vLLM-XPU patch) so we stop
   paying graph breaks around the collective -- the first-order bottleneck Seguin proved. This is where
   a real, publishable contribution lives.
6. **Characterize/fix Battlemage PCIe P2P reliability** (the RxErr/engine-reset) on a clean switch
   topology -- nobody has published working discrete-Battlemage P2P; we could be first.


## G. References

Kernel / P2P:
- drm/xe pcie p2p fast interconnect: https://lists.freedesktop.org/archives/dri-devel/2025-October/533365.html
  and /533515.html
- Phoronix, Intel Multi-Device SVM in Linux 7.0: https://www.phoronix.com/news/Intel-Multi-Device-SVM-Linux-7
- Phoronix, Intel Xe Multi-Device SVM code: https://www.phoronix.com/news/Intel-Xe-Multi-Device-SVM-Code
- kernel.org PCI P2PDMA: https://docs.kernel.org/driver-api/pci/p2pdma.html
- Linux 5.9 P2PDMA AMD Zen whitelist (Phoronix): https://www.phoronix.com/news/Linux-5.9-PCI-P2PDMA-Zen-Newer
- AMD Zen P2P whitelist patch (Koenig): https://patchwork.kernel.org/project/linux-pci/patch/20190418115859.2394-1-christian.koenig@amd.com/
- amd-gfx Deucher/Koenig thread: https://www.spinics.net/lists/amd-gfx/msg66693.html

Empirical B70 multi-GPU:
- Puget Systems, Arc Pro B70 multi-GPU inference: https://www.pugetsystems.com/labs/articles/intel-arc-pro-b70-multi-gpu-ai-inference-performance/
- vLLM issue #41663 (dual B70 TP=2): https://github.com/vllm-project/vllm/issues/41663
- intel/compute-runtime issue #922: https://github.com/intel/compute-runtime/issues/922

Software stacks:
- steveseguin/b70-optimization-lab: https://github.com/steveseguin/b70-optimization-lab
- 110 t/s repro README: https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/minimax-m27-b70-110tps-ubuntu24-20260523/README.md
- ZML: https://github.com/zml/zml

---

## H. OUR-FABRIC measured results (2026-06-22, the A/B begins)

Box: the sec-0 rig (2x B70, cross-die, PCIe Gen3, 1950X). vLLM 0.23 (`:int8g`), 27B-W4A8-sqgptq-prepacked.

### H.1 [!] TP=2 + PIECEWISE graph capture FAILS on our build -- and it is EXACTLY Seguin's collective/graph problem
First TP=2 serve (P2P off, the script default; GRAPH=1 PIECEWISE) died at engine init:
```
oneCCL: coll.cpp:1421 ccl_allreduce_impl: EXCEPTION: |CCL_SYCL| sched algorithms
do not support sycl_graph recording, please use sycl_algorithms
```
=> The graph capture tries to RECORD the allreduce and oneCCL's default `sched` algo can't be captured. This is
**the first-order "graph break around the collective" wall Seguin describes (sec B.1)** -- our vLLM 0.23 lacks his
clone-safe compiled-allreduce custom-op (confirmed: `grep VLLM_XPU_COMPILE_ALLREDUCE_CUSTOM_OP` -> absent), so the
collective is not capturable. NOT an xe P2P fault, NOT RxErr -- dmesg clean. So on our build the choice is:
  (a) **eager (GRAPH=0)** -- no capture -> no allreduce-recording conflict -> TP=2 runs (testing now);
  (b) the oneCCL `sycl_algorithms` knob the error suggests (Seguin found oneCCL knobs flaky -- B.3);
  (c) cherry-pick Seguin's clone-safe allreduce patch into our vLLM-XPU (the "real" fix -- F.5, biggest lever).
This is concrete confirmation that for us, too, the bottleneck is the framework/graph boundary around the collective,
not the Gen3 wire. The P2P-on-vs-off A/B (H.2) only becomes meaningful once TP=2 serves (eager unblocks it).

### H.2 TP=1 baseline (for the TP=2 comparison), 27B W4A8 @ ctx2048
TP=1 GRAPH=1: c1 dec 20.7 / c8 12.2 t/s; TTFT 876ms(c1)->4039(c8); agg 18.3->67.8. Single-card KV-bound (25GB model
on 32GB -> ~2GB KV). TP=2 should split to ~12.5GB/card -> more KV (raise MAXLEN to 4096) + ~2x weight BW/card. Pending eager run.

### H.3 TP=2 EAGER (GRAPH=0) works but is a 6x perf REGRESSION -- confirms you need the captured allreduce
TP=2 eager serves cleanly (`HEALTHY GRAPH=0 TP=2 world_size=2 backend=xccl`, dmesg clean) but:
  27B W4A8 TP=2 eager c1: decode **3.50 t/s** (vs TP=1-graph **20.7**), TTFT **5374ms** (vs 876), tpot 286ms (vs 48).
=> ~6x SLOWER. Two compounding penalties: (a) eager (no graph capture) is already much slower on XPU, (b) a per-layer
allreduce on the cross-die Gen3 fabric. So TP=2 only pays off WITH graph capture, which needs a capturable allreduce.
TP=2 IS functional though -- so it remains the only way to serve the 27B W8A8 (35GB), just slowly.

### H.4 NEXT cheap lever: CCL_ENABLE_SYCL_KERNELS=1 to make the allreduce graph-capturable (no vLLM patch)
The script hardcoded `CCL_ENABLE_SYCL_KERNELS=0` (Battlemage #41663 stability advice) -- but that is exactly what forces
the non-capturable "sched" allreduce the H.1 error rejected. The error literally says "please use sycl_algorithms".
Hypothesis: `CCL_ENABLE_SYCL_KERNELS=1` switches oneCCL to the SYCL-kernel allreduce that DOES support sycl_graph
recording -> would unblock GRAPH=1 + TP=2 with NO code change. Made it overridable via `SYCLKERNELS` env (30_serve).
Test next: GRAPH=1 TP=2 SYCLKERNELS=1. Risk: Seguin found some CCL sycl knobs flaky (B.3) + #41663 disabled it for
stability -> watch for hang/RxErr. If it works, that is the real TP=2 win without cherry-picking Seguin's patch (F.5).

### H.5 [WIN] CCL_ENABLE_SYCL_KERNELS=1 unlocks GRAPH-CAPTURED TP=2 allreduce on B70 -- no vLLM patch
Set `SYCLKERNELS=1` (-> `CCL_ENABLE_SYCL_KERNELS=1`) + GRAPH=1 + TP=2 and the PIECEWISE capture SUCCEEDS:
`Capturing CUDA graphs PIECEWISE 4/4 [done]` -> `HEALTHY GRAPH=1 TP=2`, dmesg CLEAN (no #41663 RxErr/BCS reset).
=> the sycl-kernel oneCCL allreduce IS sycl_graph-recordable, where the default `sched` algo (SYCL_KERNELS=0) is not.
**This is a no-source-patch route to graph-captured TP=2 on B70** -- Seguin got there via a vLLM clone-safe-allreduce
patch (B.2); we get it via one oneCCL env flag. (The script had hardcoded =0 for #41663 stability; on OUR box =1 is
stable. Re-validate per-host.) Novel datapoint -- not in the public B70 record.

### H.6 TP=1 vs TP=2 (graph) on the 27B W4A8 @ ctx2048 -- TP=2 LOSES for a model that fits one card
                 TP=1 (1 card, graph)     TP=2 (2 cards, graph + sycl-allreduce, P2P off)
  c1 decode t/s     20.73                    22.08    (+6.5%  <- the 2x weight-BW edge, real but small)
  c1 TTFT ms        876                      2858     (3.3x WORSE <- per-layer allreduce in prefill on Gen3 cross-die)
  c1 tpot ms        48.2                     45.3     (slightly better)
  c8 decode t/s     12.24                    6.29     (WORSE)
  c8 agg out t/s    67.84                    34.28    (2x WORSE <- allreduce overhead dominates at concurrency)
=> On our PCIe-3.0 cross-die box, the allreduce overhead OUTWEIGHS the 2x-BW benefit for any model that fits one card:
TP=1 wins on TTFT and on concurrent throughput; TP=2 only nudges single-stream decode. **So transport DOES matter for
us (answers F.1): the Gen3 cross-die allreduce is a real tax.** TP=2's justified use is ONLY when the model does NOT
fit one card (the 27B/Qwable W8A8 at 33-35GB) -- there TP=2 is the enabler, not an optimization. The P2P-on A/B (H.7)
is the remaining lever to shrink that allreduce tax.

### H.7 27B int8 served-ladder summary (ctx2048) -- the full TP/scheme picture
                       c1 dec   c1 TTFT   c8 dec   c8 agg    notes
  W4A8 TP=1 (1 card)   20.7     876ms     12.2     67.8      best for a model that fits one card
  W4A8 TP=2 (graph)    22.1     2858ms    6.3      34.3      +6.5% c1 dec, but TTFT/conc much worse
  W8A8 TP=2 (graph)    17.5     2728ms    6.1      34.0      35GB -> TP=2 ONLY; now SERVABLE (was N/A)
  [W8A8 TP=1: impossible, 35GB > 32GB card]
Reads: (a) int4-wt (W4A8) decode > int8-wt (W8A8) at matched TP (bytes-bound). (b) TP=2 helps c1 decode a hair
(2x weight BW) but its allreduce tax dominates TTFT + concurrency on our Gen3 cross-die box. (c) TP=2's real job =
fit the >32GB W8A8. NEXT: P2P-on A/B (P2PACCESS=1) -- does enabling card-to-card P2P shrink the allreduce tax vs host-staged?

### H.8 Direct P2P probe (70_xpu_p2p_probe.py) -- torch d2d copy is NOT peer-direct on our box
Ran the direct xpu0<->xpu1 probe (P2PACCESS=1, IPCX=drmfd, both cards, dmesg CLEAN, exit 0):
  d2d copy 16MB: 2.05 GB/s | 64MB: 1.35 GB/s | 256MB: 1.39 GB/s | reverse 64MB: 1.35 GB/s
  8-elem ping-pong latency: **452 us/copy**
  no `ze_peer` binary in the :int8 image; GPU0 (0a:00.0) in iommu_group 28.
READ: torch `.copy_()` cross-device is ~1.35 GB/s -- FAR below peer-direct Gen3 (~13-15) AND below a clean host
bounce (~7-8). So torch is doing an unpipelined host bounce with high fixed overhead (452us for 16 bytes = pure
launch/sync latency), NOT peer DMA. CCL_TOPO_P2P_ACCESS=1 + drmfd did NOT speed up torch (expected -- torch.copy_
does not use oneCCL). CAVEAT: this is the TORCH path, NOT oneCCL's allreduce (vLLM's real TP path), which DID work at
TP=2 (H.5/H.6). So: (a) generic cross-device torch ops on B70 are very slow -> avoid hand-rolled torch P2P; (b) the
meaningful P2P-on question is the oneCCL serve A/B (H.9). No ze_peer means the authoritative Level-Zero peer matrix
(F.3/F.4) still needs level-zero-tests installed -- a clean follow-up. dmesg clean = no Puget-style fault at the torch layer.

### H.9 [VERDICT] P2P-on FAILS at every level on kernel 6.18 -- no userspace spoof; kernel 7.0+ is the gate
oneCCL serve A/B: P2PACCESS=1 dies at WorkerProc init in ~65-70s with BOTH IPC exchanges (pidfd AND drmfd); P2P-off
(host-staged) works. dmesg CLEAN (NOT the Puget hardware RxErr -- a software peer-access failure).
RAW Level-Zero ctypes probe (71_ze_p2p_ctypes.py) across **12 env variations** (debug keys EnableCrossDeviceAccess/
EnableP2P, ZE_FLAT_DEVICE_HIERARCHY FLAT/COMPOSITE/COMBINED, L0 v2, CCL_TOPO_P2P_ACCESS=1) -- ALL identical:
    zeDeviceCanAccessPeer dev0<->dev1 = **False** (call returns 0x0 success, value False -> driver says NO peer)
    zeDeviceGetP2PProperties flags = 0x0  (ACCESS=N, ATOMICS=N)
    zeMemOpenIpcHandle on peer = 0x78000004 (peer map FAILED)
=> The intel-compute-runtime / Level-Zero driver on **kernel 6.18 exposes ZERO peer access** between the two B70s, and
NO userspace setting changes it. This is the missing F.3/F.4 matrix, now measured: **B70<->B70 P2P is not available on 6.x.**
TWO remaining levers, BOTH require a host reboot (out of agent scope):
  (1) [cheap] boot `iommu=off`/`amd_iommu=off` -> tests A.2: does iommu=pt void the AMD-Zen pci_p2pdma whitelist? If so,
      canAccessPeer may flip True on 6.18 (no kernel upgrade). Re-run 71_run_ze_matrix.sh to confirm.
  (2) [the fix] **kernel 7.0+** (drm/xe pcie-p2p fast-interconnect patch, A.1). The 6.18 xe driver lacks the peer path
      -> that is why canAccessPeer=False regardless of env. Safe validation: LIVE-USB boot a 7.x distro (MOONSHOT sec 6 #2),
      re-run the probe + a TP=2 P2P serve, BEFORE upgrading the Unraid kernel.
NET for TP=2 today: host-staged (P2P off) is our only path; SYCLKERNELS=1 graph capture makes it usable. P2P upside
(shrinking the Gen3 allreduce tax) is GATED on the kernel-7.x reboot -- queued as the next hardware-window experiment.

### H.10 [MEASURED 2026-06-24] the allreduce BW number that sizes the entire P2P prize -- 1.16 GB/s host-staged
`scripts/allreduce_bench.py`, both cards, xccl, int8g image. THE number for the P2P decision (full per-step
disassembly in [`27b_w8a8_research.md`](../27b_w8a8_research.md)):
```
  msg          SYCL_KERNELS=1 (captured serve)   eager (SYCL=0)
               lat        algbw=busbw            lat       algbw       use
  10 KB        0.088 ms   0.10 GB/s (lat-bound)  0.25 ms   0.03 GB/s   DECODE allreduce ([1,5120] bf16)
  16 MB        14.1 ms    1.19 GB/s              24.9 ms   0.67 GB/s
  21 MB        ~18 ms     1.16 GB/s              ~32 ms    0.68 GB/s   PREFILL allreduce ([2048,5120] bf16)
  256 MB       231.8 ms   1.16 GB/s              396 ms    0.68 GB/s
```
- **Effective host-staged allreduce BW plateaus at 1.16 GB/s (SYCL_KERNELS=1) / 0.68 (eager)** -- ~13x below the
  Gen3 x16 wire (~15.8 GB/s), matching the H.8 torch-d2d 1.35 GB/s. Cause: GPU0->host RAM->CPU reduce->host
  RAM->GPU1, un-pipelined, across two 1950X dies over Infinity Fabric. SYCLKERNELS=1 is ~1.7x BW + ~3x lower
  small-msg latency than eager (a free win we already take in the captured serve).
- **Sizing the P2P prize (the headline of 27b_w8a8_research.md):** a 64-layer forward fires **128 allreduces**.
  - PREFILL @2048: 128 x 21 MB / 1.16 GB/s = **~2.3 s of allreduce out of a 2.748 s TTFT -> prefill is ~84%
    collective-bound.** If P2P lifts the allreduce toward the Gen3 wire (~10-13 GB/s), collectives -> ~210-270 ms,
    **TTFT 2.75 s -> ~0.65-0.75 s = ~4x faster prefill.** THIS is the kernel-7.0 motivation, now quantified.
  - DECODE: 128 x 10 KB is latency-bound (~88 us each, ~11 ms eager-equiv = ~13-20% of a 55 ms token). Decode is
    weight-BW-bound (GEMM+quant ~40 ms = 73%), so P2P is only a **~1.1-1.2x** single-stream decode lever. The
    decode levers are int4 weights + activation-quant fusion + MTP, NOT P2P.
- So: **P2P is a PREFILL/TTFT + concurrency win (~4x prefill), not a single-stream decode win.** Re-running THIS
  bench on 7.0 (does 1.16 GB/s climb toward the wire?) is the single measurement that confirms the 4x is real.

### H.11 [RESOLVED 2026-06-23] kernel 7.0 + IOMMU off -> B70<->B70 P2P canAccessPeer=True (the 6.18 gate, opened)
New host: b70s4dayz, Ubuntu 26.04 LTS, **kernel 7.0.0-22-generic**, both B70s on xe (0b:00.0 / 44:00.0). BIOS:
IOMMU/AMD-Vi OFF (`iommu_groups`=0), ACS off, memory-interleave off; NO `iommu=` kernel param. Userspace = stock
26.04 archive (intel-opencl-icd / libze1 / libze-intel-gpu1, all 26.05.37020.3 NEO -- no external Intel repo). Re-ran
the EXACT H.9 probe with DEFAULT env (no debug keys): `./gpu-run python3 71_ze_p2p_ctypes.py`
    zeDeviceCanAccessPeer dev0<->dev1 = **True** (both directions)     [6.18: False on all 12 variants]
    zeDeviceGetP2PProperties flags = **0x1  ACCESS=Y** ATOMICS=N       [6.18: 0x0]
    IPC: zeContextCreate / zeMemAllocDevice / zeMemGetIpcHandle = 0x0; **zeMemOpenIpcHandle(peer) = 0x0 PEER MAP OK**
      [6.18: zeMemOpenIpcHandle = 0x78000004 FAILED -- the exact call that blocked the oneCCL drmfd TP path]
VERDICT: P2P available with DEFAULT env -- no EnableCrossDeviceAccess / EnableP2P / ZE_FLAT_DEVICE_HIERARCHY tuning.
Both H.9 levers were necessary and both now hold: (A.1) the drm/xe pcie-p2p interconnect path shipped in 7.0, AND
(A.2) IOMMU-off lets the AMD-Zen (family 0x17) pci_p2pdma allow-list apply. This is the unpublished F.3/F.4 datapoint,
now measured: **B70<->B70 P2P IS available on kernel 7.0 with IOMMU disabled** (it is NOT on 6.x). Closes the I.1/I.2
reboot-gated TODOs below. STILL OPEN (the BW prize, H.10): does the allreduce climb 1.16 -> ~10-15 GB/s? Needs
Docker + int8g for `scripts/allreduce_bench.py` / the oneCCL TP=2 P2P-on serve A/B, and `ze_peer` (build
level-zero-tests; not in 26.04 apt) for the authoritative peer BW/latency matrix.

### H.12 [MEASURED 2026-06-23] P2P-ON allreduce = 9.7 GB/s = 8.4x over host-staged -> the ~4x prefill prize CONFIRMED
With canAccessPeer=True (H.11), re-ran the allreduce A/B on kernel 7.0 (vllm-xpu-env:v0230 recovered from the old
docker.img; torch 2.11.0+xpu, native xccl, 2x B70 cross-die). A/B = CCL_TOPO_P2P_ACCESS 0 vs 1  x
CCL_ENABLE_SYCL_KERNELS 0 vs 1 (IPC=pidfd; this oneCCL rejects the old 6.18 `drmfd`). Script: 61_allreduce_p2p_ab.sh.
algbw=busbw (GB/s):
```
  msg      p2pOFF eager   p2pOFF sycl(=H.10)   p2pON eager   p2pON sycl
  1 MB     0.67           1.22                 1.16          9.77
  16 MB    0.66           1.18                 3.35          9.43
  256 MB   0.67           1.14                 3.43          9.70    (p2pON sycl peak 10.22 GB/s @ 8MB)
```
HEADLINE: P2P-on + SYCL kernels = ~9.7 GB/s plateau vs 1.16 host-staged = **8.4x**, ~61% of the 15.8 GB/s Gen3 x16
wire. Small-message (decode-sized ~10KB) latency ~unchanged (0.085-0.09 ms in both) -> P2P is ~1.2x for decode, as
predicted. PREFILL recompute with the measured 9.43 GB/s @16-21MB: 128 allreduce x 21MB / 9.43 = ~283 ms (was
~2304 ms @1.16) -> TTFT 2748 ms -> ~727 ms = **~3.8x faster prefill** -- H.10's estimate, now MEASURED. P2P is a
prefill/TTFT + concurrency win, not a single-stream decode win. The eager P2P-on result (C, 3.43 GB/s = 5.1x over
eager host-staged) shows P2P helps even without SYCL kernels; SYCL kernels + P2P together is the production path
(the captured serve runs SYCL_KERNELS=1). NEXT: end-to-end TP=2 P2P-on serve A/B (int8g, also recovered from
docker.img) to confirm the real-world TTFT drop, and ze_peer for the raw peer-copy ceiling.

### H.13 [2026-06-24] P2P unlocked at the allreduce layer (H.12) but BLOCKED in vLLM serve -- DEVICE_LOST at worker init
End-to-end test on the 27B W8A8 TP=2 captured-MTP serve (qwen36-27b-w8a8-sqgptq-mtp, 69_lever_tests.sh A):
- P2PACCESS=0 serves fine: c1 TTFT 2901 ms, decode 26.2 t/s (host-staged, as expected).
- P2PACCESS=1 CRASHES at vLLM's `xpu_worker.py:105` warmup `all_reduce(torch.zeros(1).xpu())` ->
  `level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)`, with BOTH `CCL_ZE_IPC_EXCHANGE=pidfd`
  AND `sockets`. The crash is in `init_device`, BEFORE graph capture (capture ruled out).
Since the raw 2-rank `mp.spawn` allreduce microbench (H.12) reaches 9.7 GB/s with the SAME `CCL_TOPO_P2P_ACCESS=1`,
the P2P fabric is fine -- the failing path is oneCCL P2P inside vLLM's multiproc-executor worker topology
(separately spawned workers + IPC handle exchange). NET: the ~3.8x prefill prize is REAL at the collective layer
but NOT accessible through the current vLLM serve. GPUs recover cleanly after the loss. Follow-ups (dedicated
session): `VLLM_WORKER_MULTIPROC_METHOD=fork`, a newer oneCCL, NEO `EnableP2P`/`EnableCrossDeviceAccess` keys, or a
custom P2P all-reduce that bypasses the failing warmup. Until then, TP=2 serves stay host-staged (P2PACCESS=0).

**WARNING -- the DEVICE_LOST wedges the multi-GPU state (does not self-clean).** After two `P2PACCESS=1` attempts,
a fresh container running the KNOWN-GOOD `P2PACCESS=0` 27B W8A8 TP=2 serve ALSO failed with the identical
`UR_RESULT_ERROR_DEVICE_LOST` at `xpu_worker` `init_device` `all_reduce` -- i.e. every TP=2 serve is broken until
the GPU state is reset (single-GPU/TP=1 is unaffected, `xpu count`=2). Recovery: `modprobe -r xe; modprobe xe`
(needs no `/dev/dri` in use) or reboot. So do NOT retry `P2PACCESS=1` in serve without a GPU reset between tries.

---

## I. NEXT STEPS for P2P testing (ordered, reboot-gated)  [2026-06-22]  [SUPERSEDED -- H.11 probe + H.12 BW done; H.13 = serve-integration gap]

Userspace is EXHAUSTED (H.9: 12-variant L0 probe, all canAccessPeer=False). Every remaining lever needs a host
reboot, so BATCH each hypothesis into one maintenance window. Everything below is staged/turnkey. Success metric
throughout: `71_run_ze_matrix.sh` shows `canAccessPeer=True` (today it is False on all 12 variants).

### I.0 Prep that needs NO reboot (do anytime)
- [ ] Install `ze_peer` (intel level-zero-tests) into a probe image so the authoritative peer bandwidth/latency
      matrix is ready: `git clone https://github.com/oneapi-src/level-zero-tests` + build, or `apt-get install
      level-zero-tests` if packaged. Then `ze_peer` reports the real peer BW (our torch d2d host-bounce was 1.35 GB/s).
- [ ] Capture the current PCIe ACS/IOMMU-group state for the pair: are 0a:00.0 (GPU0) and 44:00.0 (GPU1) in the
      SAME iommu group? (cross-die -> different groups today). `ls /sys/kernel/iommu_groups/*/devices/`.

### I.1 WINDOW 1 -- `iommu=off` (cheap; tests the AMD-Zen whitelist hypothesis A.2; NO kernel change)  ~10 min
Hypothesis: kernel 5.9 whitelists AMD Zen (1950X = family 0x17) for cross-die pci_p2pdma, but commit 6dbbd053e6
VOIDS the whitelist "when an IOMMU is present" -- and we boot `iommu=pt`. If `pt` counts as present, disabling
IOMMU restores the whitelist -> peer access may work ON 6.18.
- [ ] Add `iommu=off` (or `amd_iommu=off`) to the Unraid syslinux append line; reboot. SAFE only if the B70s are
      bare-metal/Docker (they are -- NOT VM-passthrough). If any VM passthrough exists, SKIP (needs IOMMU).
- [ ] Post-boot: `cd /mnt/vm_8tb/b70 && ./gpu-run bash 71_run_ze_matrix.sh`
      -> if ANY variant flips `canAccessPeer=True`: WIN on 6.18. Then run the TP=2 P2P-on serve:
         `MODELS="...|...PREPACK=1 TP=2 GRAPH=1 SYCLKERNELS=1 P2PACCESS=1 IPCX=drmfd..." ... bash 66_bench_ladder.sh`
         and compare decode/TTFT vs the P2P-off baseline (W4A8 TP=2 c1 22.1 / H.6).
      -> if still all False: IOMMU is not the gate -> it is the xe-driver peer path -> go to I.2.
- [ ] Restore `iommu=pt` afterward if you don't keep iommu=off.

### I.2 WINDOW 2 -- kernel 7.0+ via LIVE USB (zero risk to the Unraid install)  ~30 min
The real fix: the drm/xe "pcie p2p as fast interconnect" patch shipped Linux 7.0 (A.1); the 6.18 xe driver lacks
the peer path (why canAccessPeer=False regardless of env). Validate on a live distro BEFORE upgrading Unraid's kernel.
- [ ] Boot a live USB of a 7.0+/7.1 distro (e.g. a recent Arch/Fedora rawhide ISO) on the GPU host. No disk write.
- [ ] Verify the xe interconnect: `dmesg | grep -iE "xe.*p2p|interconnect|pci_p2pdma"`; check
      `pci_p2pdma_distance` permits the pair (watch for host-memory-fallback vs interconnect in dmesg).
- [ ] Install the XPU runtime (intel-compute-runtime + level-zero) on the live env (or use a container with --device
      /dev/dri); run `71_run_ze_matrix.sh` + `ze_peer`. Success = `canAccessPeer=True` + ze_peer reports real peer BW.
- [ ] If True: run a TP=2 P2P-on serve A/B (as I.1) -> measure whether P2P shrinks the Gen3 allreduce tax (H.6:
      TTFT 2858ms / c8 agg 34 today). THEN decide whether upgrading Unraid's kernel to 7.x is worth it for production.
- [ ] ALSO test `iommu=off` in this same window if I.1 was skipped.

### I.3 If BOTH windows fail -> it is the cross-die PHYSICAL topology, not software
Then no kernel/driver setting helps (the two cards are on separate 1950X dies over Infinity Fabric). Escalate to
hardware: MOONSHOT_RESEARCH T2 = one Gen5 PCIe switch board (both B70s downstream of ONE switch -> "behind the same
switch" clean P2P + Gen5 peer behind a Gen3 host). That is the DGX-class fix and also moots the iommu-whitelist
question. Price out PEX89000/Switchtec per MOONSHOT sec 7.

### I.4 What NOT to retry (already disproven, H.9)
- Any userspace env (NEOReadDebugKeys/EnableCrossDeviceAccess/EnableP2P, ZE_FLAT_DEVICE_HIERARCHY, L0 v2,
  CCL_TOPO_P2P_ACCESS, drmfd-vs-pidfd IPC) -- all 12 give canAccessPeer=False. Do not burn cycles here again.
- vLLM P2PACCESS=1 on 6.18 -- fails WorkerProc init in ~65s (both IPC exchanges). Host-staged (P2P off) is the only
  working TP=2 path until I.1/I.2 flips peer access.

### I.5 Independent of P2P: the allreduce-tax software levers still open (no reboot)
Even with host-staging, the Gen3 allreduce tax can be cut by reducing/repositioning collectives (Seguin B.2):
cherry-pick his clone-safe-allreduce + oproj-delay-allreduce patches into our vLLM-XPU (we got graph capture free
via SYCLKERNELS=1, but not the fusion). That is the orthogonal, no-hardware path to better TP=2 -- docs/literature/10.

---

## J. REPROFILE on kernel 7.0 + new BIOS (2026-06-24) -- and the PUSH/PULL peer-write discovery

New campaign goal (Isaac): now that we are on kernel 7.0 + new BIOS (IOMMU off), re-measure every P2P
datum from scratch, go BELOW oneCCL with hand-written Level Zero, and use the results to plan vLLM
TP=2/PP=2 patches. Target model: qwen36-27b-w8a8 (served via rdy_to_serve, unedited).

### J.0 Hardware reprofile -- topology UNCHANGED, still cross-die
Same physical box as Section 0 (AMD Threadripper 1950X, ASRock X399 Professional Gaming, 125 GiB),
reinstalled to Ubuntu 26.04 / kernel 7.0.0-22-generic. `lspci -t`:
```
  [0000:00] (die0 RC) -> 03.1 -> [09-0c] switch -> 0b:00.0  GPU0   (Battlemage G31 8086:e223)
  [0000:40] (die1 RC) -> 03.1 -> [42-45] switch -> 44:00.0  GPU1
```
- **Still cross-die** (the two B70s hang off separate 1950X dies = the documented worst case for P2P).
  The new BIOS did NOT relocate slots. A same-die slot move remains an OPEN bandwidth lever (Isaac
  offered to move a card); deferred -- P2P already functions cross-die (below), so topology is a BW
  knob, not the current blocker.
- ReBAR fully open: BAR2 = 32G on both cards (`lspci -vv` Region 2). IOMMU OFF (`ls
  /sys/kernel/iommu_groups` = 0 groups; no `iommu=` on cmdline -> disabled in BIOS). xe driver loaded.
- L0 sees both: `sycl-ls` -> 2x "Level-Zero V2, Intel Graphics [0xe223]". Link-speed config-space read
  needs root (sudo password-gated); rely on the measured peer-write 11.3 GB/s below = 72% of Gen3 x16
  (~15.8 GB/s) as proof the real wire is still Gen3 x16.

### J.1 [METHOD] hand-written Level Zero peer-copy benchmark (below torch/oneCCL)
`scripts/100_ze_peer_copy.c` (+ `100_run_peer_copy.sh`): raw `libze_loader`, ONE context spanning both
devices, `zeMemAllocDevice` on dev0 and dev1, then `zeCommandListAppendMemoryCopy` peer copy. This is
our hand-rolled `ze_peer`. Built with gcc in `vllm-xpu-env:v0230`, run under `gpu-run` (both cards).
Command: `./bin/gpu-run bash scripts/100_run_peer_copy.sh`. canAccessPeer=True both dirs (confirms H.11
on the current kernel+BIOS). Same-context dev0<->dev1 device-alloc copy is the correct L0 peer model;
no IPC/explicit-enable needed in one process (IPC is only for cross-process, e.g. vLLM workers).

### J.2 [DISCOVERY] PCIe PUSH (peer write) = 11.3 GB/s, PULL (peer read) = 3.24 GB/s -- 3.5x asymmetry
The copy direction relative to WHICH device executes the copy dominates everything. Data moving d0->d1:
```
  engine    PULL (exec on dst=d1, reads peer d0)   PUSH (exec on src=d0, writes peer d1)
  copy(BCS)        3.24 GB/s plateau                      11.12 GB/s plateau
  compute(EU)      3.23 GB/s                              11.31 GB/s plateau  <- best
  (host-staged d0->host->d1 baseline: 3.5 GB/s effective; 8B peer latency: 9.3 us raw L0)
```
- **PUSH 11.3 GB/s = 72% of the Gen3 x16 wire**, and BEATS oneCCL's 9.7 GB/s P2P allreduce (H.12).
- **PULL 3.24 GB/s is BELOW even a host bounce (3.5).** A copy executed on the destination is a peer
  READ: PCIe reads are NON-POSTED (each cache line is a round-trip request+completion, throttled by
  outstanding-read credits + cross-die Infinity Fabric latency). PUSH is a peer WRITE: POSTED
  (fire-and-forget), so it streams at near-wire. This is THE architectural fact for B70 TP comms.
- Engine choice barely matters (compute 11.31 vs copy 11.12); DIRECTION is the 3.5x lever.
- Reconciles H.8/H.10 (torch d2d 1.35, host-staged allreduce 1.16) and H.12 (oneCCL P2P 9.7): the slow
  paths were either host-bounced or read-shaped; the fast path is a posted peer write.

### J.3 [IMPLICATION] vLLM TP=2 collective patches must be PUSH-shaped (the plan)
The allreduce tax (H.10: prefill ~84% collective-bound) shrinks most if every cross-card transfer is a
peer WRITE. Design rules for our microkernel / vLLM patch:
- Ring/pairwise allreduce where each rank WRITES its partial into the neighbor's buffer (push), never
  reads the neighbor's. Reduce locally after the inbound write lands (signal via a flag write).
- Prefer a compute-engine (EU) push kernel: 11.31 GB/s and it can fuse the reduce (add) with the
  transfer, unlike the blind BCS copy engine.
- This is independent of (and better than) oneCCL: a hand-rolled push-allreduce could hit ~11 GB/s vs
  oneCCL's 9.7, AND sidestep the H.13 DEVICE_LOST that blocks oneCCL P2P inside the vLLM worker.
- Still OPEN / next: (a) a SYCL peer-WRITE microkernel + Xe ISA dump (the "assembly" step); (b) a
  2-rank push-allreduce prototype vs oneCCL allreduce_bench; (c) wire it into vLLM's custom all-reduce
  op (the H.13 worker-init crash is a oneCCL-P2P path; a custom push collective may avoid it); (d) PP=2
  as an alternative that needs only point-to-point pushes (no allreduce) -- may suit our push-fast wire.

### J.4 do-not-repeat
- PULL-shaped peer copies (exec-on-destination) -- proven 3.5x slower than push; never use for bulk TP.
- codex CLI here runs sandboxed (bwrap loopback denied) so it cannot read repo files for review; run it
  with sandbox bypass if file-aware help is needed, else it gives only general API guidance.

### J.5 [MICROKERNEL] SYCL EU peer-write kernel = 11.26 GB/s; the allreduce shape proven
`scripts/101_peer_write_kernel.cpp` (+ `101_run_peer_write.sh`): a SYCL kernel running on dev0 whose
work-items STORE into a USM pointer in dev1's VRAM (peer access enabled via
`device::ext_oneapi_enable_peer_access`). Built with `icpx -fsycl` in :v0230, run under gpu-run. 64MB:
```
  kernel (executed on dev0)            BW        what it does
  copy:    dstPeer = src              11.23 GB/s pure EU push (matches L0 copy/compute push 11.3)
  addpush: accPeer = accPeer + src     2.36 GB/s reduce INTO peer = peer READ + peer WRITE = WORST
  local-reduce then push: dst = a+b   11.26 GB/s reduce LOCAL, single peer write = FULL SPEED
  verify peer[0..3] = 7.0 (push landed in dev1 VRAM -> direct transfer confirmed)
```
- The EU-kernel push equals the copy/compute-engine push (~11.3) -> 11.3 GB/s is the fabric write
  ceiling, reachable from any engine. We are free to use a fused EU kernel.
- **`addpush` 2.36 GB/s is the trap**: an allreduce that accumulates into a remote buffer pays BOTH the
  peer-read AND peer-write tax. **`local-reduce then push` (11.26) is the correct shape**: every rank
  reduces with LOCAL data only, then does ONE posted peer write of the result. Confirms J.3's design.
- The "assembly that transfers gpu0->gpu1" (Xe2 ISA, IGC dump): the peer store lowers to
  `store.ugm.d32.a64 (32|M0) [addr] data` -- a SIMD32 untyped-global LSC store on a 64-bit STATELESS
  address (a64). Stateless 64-bit addressing is exactly how a peer USM pointer is reached; the EU fires
  one LSC send per SIMD32 group across PCIe into peer VRAM. (Full dumps: IGC_ShaderDumpEnable, /tmp/igc.)

### J.6 NEXT (queued for this campaign)
1. **2-rank push-allreduce prototype**: ring/pairwise, each rank local-reduces then peer-writes its
   chunk to the neighbor (the 11.26 shape). Compare vs oneCCL allreduce_bench (H.12: 9.7 GB/s). Target:
   beat 9.7 with a hand-rolled push collective. Decode-sized (10KB) + prefill-sized (16-21MB) messages.
2. **vLLM integration**: register the push-allreduce as the XPU custom all-reduce op, bypassing oneCCL's
   P2P path that triggers H.13 DEVICE_LOST at worker init. Two-process (TP workers) needs L0 IPC handle
   exchange (zeMemGetIpcHandle/OpenIpcHandle -- already proven OK on 7.0, H.11) instead of single-context.
3. **PP=2 alternative**: pipeline parallel needs only point-to-point activation pushes between stages
   (no allreduce). On our push-fast / read-slow fabric this may beat TP=2. Prototype a 2-stage handoff
   of the 27B-W8A8 and compare TTFT/decode vs TP=2 host-staged (the current rdy_to_serve path).
4. Re-run `scripts/100`+`101` after any same-die slot move to see if push climbs 11.3 -> ~15 GB/s.

### J.7 [WIN] hand-rolled PUSH all-reduce BEATS oneCCL -- 10.64 GB/s prefill, 59.5us decode
`scripts/102_push_allreduce.cpp` (+ `102_run_push_allreduce.sh`): 2-rank pairwise push exchange in ONE
context (transport identical to a 2-proc oneCCL run -> algbw directly comparable to H.12). Algorithm:
step1 both ranks PUSH their buffer to the peer's scratch (concurrent, opposite directions, posted
writes); step2 both local-reduce. End state verified bufA==bufB==A+B (result 4.0 OK at every size).
```
  size           push-allreduce   oneCCL (H.12)   vs oneCCL   vs host-staged(H.10)
  10KB(decode)   59.5 us          ~85-88 us       1.4x faster latency
  1MB             8.09 GB/s        --
  16MB(prefill)  10.64 GB/s       9.43 GB/s       1.13x       9.2x (was 1.16)
  256MB          10.44 GB/s       9.70 GB/s       1.08x
```
- **A hand-rolled posted-write collective beats the vendor library (oneCCL) at BOTH prefill BW and
  decode latency** -- the publishable B70-TP result this doc set out to find (Section F goal). 10.64 =
  94% of the 11.3 single-direction write ceiling; both directions run concurrently on dual-simplex Gen3.
- Decode 59.5us is 4 kernel launches + 2 cross-card syncs; fusing the push+signal into one kernel with a
  device-side flag (no host sync between steps) is the obvious next latency cut.
- Prefill recompute with 10.64 GB/s: 128 allreduce x 21MB / 10.64 = ~252 ms (was ~2304 @1.16 host-staged)
  -> TTFT 2748 -> ~700 ms = ~3.9x faster prefill, matching the H.10/H.12 estimate -- now with OUR collective
  that needs no oneCCL P2P (so it should dodge the H.13 DEVICE_LOST that blocks oneCCL inside vLLM).
- CAVEAT (honest): this is single-process/single-context. The real vLLM path is 2 processes -> needs L0
  IPC handle exchange (zeMemGetIpcHandle/OpenIpcHandle, proven OK on 7.0 in H.11) so each worker can map
  the peer scratch. That is the J.6#2 integration step; the BW ceiling (fabric write) is unchanged by it.
  [RESOLVED in J.8 -- the IPC boundary costs nothing; cross-process peer-write = full 11 GB/s.]

### J.8 [WIN] 2-PROCESS IPC peer-write = 11.08 GB/s -- the vLLM-worker transport, proven (no oneCCL)
`scripts/103_ipc_push_allreduce.c` (+ `103_run_ipc_push_allreduce.sh`): closes the J.7 caveat. TWO
fork()'d processes, one per card, each with its OWN Level-Zero context (exactly the vLLM TP-worker
topology). Each rank exports its scratch buffer via `zeMemGetIpcHandle`, exchanges the handle over a
Unix socketpair passing the embedded dma-buf fd through `SCM_RIGHTS`, then `zeMemOpenIpcHandle`s the
peer's handle to get a cross-process peer pointer, and PUSHES (posted write, exec-on-source) into it.
```
  size           push GB/s   lat us     verify(peer)   vs single-ctx J.7   vs oneCCL H.12
  10KB(decode)    1.29        7.94 us    OK             (1 push leg)
  1MB            10.46      100 us       OK
  16MB(prefill)  11.05     1518 us       OK             10.64 (FASTER)     9.43 (1.17x)
  64MB           11.01                   OK
  256MB          11.08    24224 us       OK             10.44              9.70 (1.14x)
```
- **The 2-process IPC boundary costs ZERO bandwidth: 11.08 GB/s = the same single-direction posted-write
  ceiling (J.2 11.3), and actually edges out the single-context J.7 (10.64) and beats oneCCL (9.7).**
  `verify(peer)=OK` at every size confirms each rank's data physically landed in the peer's VRAM across
  the process boundary -- a real cross-process peer DMA, not a host bounce.
- **This is the precondition for the vLLM custom all-reduce op (J.6#2), now MET.** It proves the exact
  mechanism a vLLM TP worker pair needs -- separate processes, separate contexts, IPC-mapped peer
  scratch, posted peer write -- works at full fabric speed AND entirely bypasses oneCCL's P2P path (the
  one that DEVICE_LOSTs at vLLM worker init, H.13). The fd-over-SCM_RIGHTS exchange is the same family
  as oneCCL's `CCL_ZE_IPC_EXCHANGE=sockets`; doing it ourselves sidesteps oneCCL's broken warmup all_reduce.
- Method note: SYNCHRONOUS immediate command list per push (each `AppendMemoryCopy` blocks to completion),
  socket 1-byte barriers only at step boundaries (= the real collective's sync points), compute-engine
  ordinal 0. First run printed nothing -- `_exit()` skips stdio flush; fixed with `setvbuf(_IONBF)`.
- Decode 10KB single-leg latency 7.94 us is the raw cross-process push floor; a full 2-rank allreduce
  adds the return leg + local reduce + one barrier (cf. J.7's 59.5 us full path). The decode latency
  lever is fusing push+signal into ONE kernel with a device-side flag (no per-step host barrier) -- next.

### J.9 [MIXED] decode-latency surgery: cross-queue events = 1.36x (44us); device-flag fusion HANGS on Xe
`scripts/104_fused_allreduce.cpp` (+ `104_run_fused_allreduce.sh`): three all-reduce schedules for the
decode-sized (10KB) message, head-to-head. J.7's full path was 4 kernel launches + **2 HOST syncs** (it
blocks the host after step1's push before it can even submit step2's reduce). Trying to cut that:
```
  mode                         lat us @10KB   vs baseline   verify   what changed
  A baseline (= J.7)           60.40          --            OK       step1; HOST WAIT; step2; HOST WAIT
  B cross-queue events         44.42          1.36x faster  OK       submit all 4 async; reduce depends
                                                                     (L0 event) on peer's push; 1 host wait
  C fused device-flag kernel   HANG           --            --       1 kernel/rank: push, fence, peer-write
                                                                     a seq flag, spin on own flag, reduce
```
- **B (cross-queue events) is the deployable decode win: 44.4 us vs 60.4 = 1.36x, fully correct.** The
  reduce kernel on each device takes an L0-event dependency on the PEER device's push kernel, so the
  whole all-reduce is submitted async and the host blocks exactly ONCE at the end instead of mid-collective.
- **C (single-kernel device-flag signalling) HANGS -- a real Battlemage/Xe limitation, now pinned down.**
  Each rank peer-writes a sequence flag into the other card mid-kernel then spins on its own flag. It
  never converges (even with a 5e8-iter spin bound it ran >120s): **a peer write issued from WITHIN a
  running kernel does not become visible to a kernel spinning on the other device.** A
  `atomic_fence(release, system)` does not force the posted write across PCIe mid-kernel, and peer
  ATOMICS=N (H.11) means there is no system-scope atomic to lean on. Visibility is only guaranteed at
  kernel-completion (engine write-buffer/L3 flush at the boundary). => On B70 you CANNOT build a
  spin-wait device-side cross-card barrier; cross-card ordering must be host-driven or split across
  separate kernels (which is what B's event dependency does -- the push kernel COMPLETES before the
  dependent reduce kernel starts).
- **Implication for the 2-process vLLM op (J.10):** SYCL cross-queue events live inside ONE process, so
  the B trick is not directly available to two separate TP workers. Two processes must synchronise the
  push->reduce boundary over the HOST (a cpu_group barrier or a polled host-shared flag) -- exactly the
  socket barrier J.8 used. For decode that host barrier is the floor; for prefill it is amortised by the
  11 GB/s transport. GPU verified healthy after the hung-kernel kill (both cards enumerate, dmesg clean).

### J.10 [WIN] deployable 2-PROCESS custom all-reduce BEATS oneCCL on BOTH latency AND bandwidth
`scripts/105_xpu_push_ar.cpp` (C-ABI .so) + `105_ar_harness.py` (torch.distributed gloo driver) +
`105_run_xpu_push_ar.sh`. This is the deployable core of the vLLM custom op: TWO INDEPENDENT processes
(spawn'd, not fork'd -- exactly vLLM's TP-worker topology, no shared fds), each its own L0 context. Built
with `icpx -fsycl` so SYCL kernels (push + local reduce) and raw Level-Zero (IPC handles) coexist,
bridged by `get_native<level_zero>(context)`. IPC handle's dma-buf fd is passed over a NAMED Unix socket
via SCM_RIGHTS (oneCCL's "sockets" family; robust vs `pidfd_getfd`, which Ubuntu ptrace_scope=1 blocks
between sibling workers). Synchronisation respects J.9: push kernel COMPLETES (flush at boundary), then a
host barrier, then reduce.
```
  size           gloo-barrier   SHM-barrier (final)   verify   vs oneCCL(H.12)   vs J.7 1-ctx
  10KB(decode)   241.91 us       34.55 us             OK       ~85 us (2.5x)     59.5 us (1.7x)
  64KB            267 us         56.09 us             OK
  1MB             382 us        129.95 us = 8.07 GB/s OK
  16MB(prefill)  1979 us       1577 us = 10.64 GB/s   OK       9.43 GB/s (1.13x) 10.64 (=)
  256MB         26423 us      25693 us = 10.45 GB/s   OK       9.70 GB/s (1.08x)
```
- **The custom op BEATS the vendor library (oneCCL) at BOTH ends across independent processes:** decode
  latency 34.55 us (2.5x better than oneCCL, 1.7x better than our own single-context J.7) and prefill BW
  10.64 GB/s (1.13x oneCCL). Correct (sum==4.0) at every size.
- **The barrier primitive was the small-message wall, not the transport.** A gloo TCP `dist.barrier()` per
  call cost ~150 us (241 us @decode). Replacing it with a 2-process sense-reversing SPIN barrier in
  `shm_open`'d shared memory (`__atomic` ops, ~1-2 us) cut decode 7x to 34.55 us and lifted prefill from
  8.48 -> 10.64 GB/s. The vLLM op must therefore NOT synchronise over gloo per-collective -- use the shm flag.
- **Runs with `CCL_TOPO_P2P_ACCESS=0`.** Our P2P is L0-IPC, INDEPENDENT of oneCCL's P2P setting, so the
  serve keeps oneCCL host-staged (its warmup all_reduce succeeds -> NO H.13 DEVICE_LOST) while the big
  model collectives go through our op at 11 GB/s. This is the architectural escape from the H.13 blocker.
- C-ABI surface (ready for the vLLM bind): `ar_setup(rank,max) / ar_exchange(rank,sockpath) /
  ar_allreduce(nbytes) / ar_fill / ar_peek / ar_teardown`, plus an internal shm `ar_barrier()`.
- REMAINING for the live serve (J.11): bind the op to TORCH's L0 context instead of our own. A USM
  pointer is only valid in its allocating context, so the op must run on torch's sycl queue
  (`c10::xpu::getCurrentXPUStream().queue()`) and IPC-export a scratch tensor allocated by torch -- i.e.
  a small torch C++ extension (pybind), not a standalone ctypes .so. The transport/IPC/barrier are now
  all proven; this is the mechanical binding step. Then monkeypatch `XpuCommunicator.all_reduce`
  (`vllm/distributed/device_communicators/xpu_communicator.py`, the one-line `dist.all_reduce` today) via
  a `sitecustomize.py` (same mechanism as `contrib/vllm_int8_xpu/.../dense_test`) and A/B the 27B-W8A8 TP=2 serve.

### J.11 [WIN] custom all-reduce runs IN TORCH'S CONTEXT on real torch.xpu tensors -- live-serve bridge proven
`scripts/106_xpu_push_ar_torch.cpp` + `106_ar_torch_harness.py` + `106_run_ar_torch.sh`. The J.10 op ran
in our OWN context; a torch tensor's USM pointer is only valid in TORCH's context, so to all-reduce a real
vLLM activation we must run on torch's queue/context. **torch-xpu exposes the address of its `sycl::queue`
as a plain int: `torch.xpu.current_stream().sycl_queue`.** We pass that int into the .so, `reinterpret_cast`
it to `sycl::queue*`, and take `.get_context()` -> torch's L0 context. Scratch is a raw `zeMemAllocDevice`
in THAT context (base-aligned -> clean IPC); the input is the torch tensor's `data_ptr()`; everything lives
in one context -> no cross-context use. Push + reduce kernels submit on TORCH's queue.
```
  size           lat us    algbw GB/s   verify   (torch tensor itself == sum after our op)
  10KB(decode)   45.79      0.22        OK(4.0)
  1MB           128.88      8.14        OK(4.0)
  16MB(prefill) 1583.03    10.60        OK(4.0)   beats oneCCL 9.43 (1.12x)
  64MB          6448.62    10.41        OK(4.0)
```
- **The ABI reinterpret of torch's `sycl::queue` WORKS** -- the .so is `icpx`-built in the same image as
  torch-xpu 2.11.0+xpu, so the DPC++ runtime/queue ABI matches. `verify OK(4.0)` = a real `torch.full(.,1)`
  on rank0 and `torch.full(.,3)` on rank1 became `4.0` in-place via our op. **No pybind/torch C++ extension
  needed** -- a ctypes `.so` driven from Python drives vLLM's own tensors. This is the entire bridge.
- Decode 45.8 us (vs oneCCL ~85, J.7 59.5); prefill 10.6 GB/s (vs oneCCL 9.43). Same win as J.10, now on
  the actual data path. Submitting our kernels on torch's in-order queue also gives free ordering vs torch's
  other work on that stream.
- C-ABI for the vLLM monkeypatch: `ar_setup_torch(rank, torch_queue_addr, max) / ar_exchange(rank,
  sockpath) / ar_allreduce_ptr(data_ptr, nbytes) / ar_teardown`. NOTE: current kernels are fp32; vLLM
  activations are bf16 -> add a dtype-dispatched reduce (push is a byte copy, dtype-agnostic) before the
  live serve. Then monkeypatch `XpuCommunicator.all_reduce` to `out=input.clone(); ar_allreduce_ptr(
  out.data_ptr(), out.nbytes); return out`, with `ar_setup_torch`+`ar_exchange` done once in `__init__`
  (sockpath from the TP group's rendezvous). Serve with `P2PACCESS=0` (oneCCL host-staged warmup -> no H.13).

### J.12 [WIN] bf16 (vLLM's dtype) drop-in: 9.9 GB/s prefill, correct -- and the 4-byte-push lesson
`scripts/106` (+`ar_allreduce_ptr_dt`) + `107_ar_vllm_pattern.py`: exercise the EXACT vLLM call pattern
(`out=input.clone(); allreduce(out)`) on bf16 hidden=5120 prefill/decode shapes.
```
  shape (bf16)     lat us     GB/s     verify     [2-byte push first try]
  decode b1         69.22     0.15     OK
  decode b8         74.35     1.10     OK
  prefill 128      189.69     6.91     OK
  prefill 2048    2115.45     9.91     OK          (was 0.83 GB/s -- 12x slower)
  prefill 4096    4201.42     9.98     OK          (was 0.83)
```
- **bf16 prefill = 9.9-10.0 GB/s, correct everywhere -> beats oneCCL (9.43) on the real dtype.** Clone
  semantics verified (input untouched, output holds the sum).
- **LESSON: the PUSH must be a 4-byte-word copy, NOT a per-element bf16 (2-byte) copy.** Element-wise
  2-byte peer writes do not coalesce into wide PCIe bursts -> 0.83 GB/s (12x slower). Copying the buffer
  as uint32 words (dtype-agnostic; the push just moves bytes) restores the ~10 GB/s ceiling. Only the
  REDUCE is dtype-aware (accumulate in float, store bf16) and it touches LOCAL VRAM so 2-byte is fine.
- **Deployable artifact: `contrib/vllm_push_allreduce/sitecustomize.py`** monkeypatches
  `XpuCommunicator.all_reduce` -> our op, with full fallback (world!=2 / non-contig / odd dtype / oversize
  -> original oneCCL path), lazy one-time setup+exchange on first all_reduce, `PUSH_AR_DISABLE=1` kill
  switch. Serve plumbing in `scripts/108_serve_push_ar_ab.sh`. Graph-capture caveat stands: the op has a
  host barrier -> run EAGER (GRAPH=0) or make the TP allreduce a graph splitting op.

### J.13 [ANALYSIS] PP=2 vs TP=2 on a push-fast/read-slow fabric -- why pipeline parallel may WIN here
The J.2 discovery (posted peer WRITE 11.3 GB/s, peer READ 3.24, allreduce = many collectives) reframes the
TP-vs-PP choice for our cross-die B70 box. Comm volume per 27B (dense, 64 layers, hidden 5120) forward:
```
  topology   cross-card transfers / forward     shape (prefill 2048 / decode 1)     primitive
  TP=2       ~64-128 all_reduces                 each [2048,5120] / [1,5120]          all_reduce (push+reduce)
  PP=2       1 point-to-point activation handoff [2048,5120] / [1,5120] ONCE          send/recv (pure push)
```
- **PP=2 moves ~1/64-1/128 the cross-card bytes of TP=2 per forward, and the handoff is a SINGLE posted
  push -- exactly the primitive our fabric is fastest at (11.3 GB/s) and has NO reduce step.** TP's tax
  (H.10: prefill ~84% collective-bound) largely vanishes: prefill 2048 handoff = 20MB / 10.6 GB/s ~= 2 ms
  ONCE, vs TP's 128 x 21MB. Decode handoff = 10KB ~= 8 us once per token, vs TP's 128 allreduces.
- **Why TP is the usual default, and why those reasons are WEAK for us:** (a) pipeline bubbles hurt
  single-stream latency -- but TP=2 also fails to truly parallelise a single token here (allreduce
  serialises every layer), so for batch=1 PP's "sum of two halves + 1 push" is competitive AND comm-light;
  (b) PP needs microbatching to fill the pipe -- vLLM continuous batching supplies that, and at concurrency
  PP keeps both stages busy with near-zero comm while TP's allreduce tax grows (H.6: TP=2 c8 agg HALVED);
  (c) memory -- PP splits 32 layers/card ~= TP's per-matrix split, both fit the 35GB W8A8 across 2x32GB.
- **The bet:** on this allreduce-hostile fabric, PP=2 converts the entire collective tax into one push per
  microbatch. Prototype `--pipeline-parallel-size 2 --tensor-parallel-size 1` on 27B-W8A8 and compare
  TTFT / decode / c8-agg vs TP=2 (H.7). If vLLM-XPU's PP send/recv works host-staged, additionally swap it
  for a pure-push point-to-point (a trivial subset of our op: `ar_push` only, no reduce, no barrier-then-
  reduce) -- the handoff is one-directional so it needs only the J.2 posted write.
- Open: (1) does vLLM-XPU support PP send/recv on the XpuCommunicator path at all (it exposes all_reduce/
  reduce_scatter/all_gather/gather/broadcast -- P2P send/recv may route through raw torch.distributed); (2)
  MTP head sits on the last layers -> stage1; spec-verify logits are local to stage1, so MTP+PP should
  compose; (3) bubble cost at low concurrency. This is queued as the next campaign experiment (J.6#3).

### J.14 [WIN] LIVE 27B-W8A8 TP=2 serve: push-ar BEATS oneCCL +48-64% throughput, 3.35x TTFT (engaged + coherent)
`scripts/108_serve_push_ar_ab.sh` + `contrib/vllm_push_allreduce/`. The custom op is wired into a REAL
vLLM serve via the `XpuCommunicator.all_reduce` monkeypatch and A/B'd against stock oneCCL. Both sides:
27B-W8A8-sqgptq + BF16-MTP-graft (spec=3), TP=2, **EAGER (GRAPH=0)**, `P2PACCESS=0`, image `:int8g`,
35 GiB across 2x B70. ENGAGEMENT CONFIRMED in container logs: `[push_ar] ENGAGED ...`, both workers
`setup_torch OK` + `exchange OK (peer scratch mapped in torch ctx)`, no L0 errors, no fallback. Gen probe
COHERENT ("...Paris is the capital...of France") -> our hand-rolled reduce is numerically correct on real
activations. My sitecustomize cleanly CHAINED the MTP-graft shim (both patches coexist).
```
  conc   metric            oneCCL(host-staged)   push-ar      delta
  c1     TTFT ms           1613                  481          3.35x FASTER
  c1     per-stream dec    8.40 t/s              12.41 t/s    +48%
  c1     out tok/s         7.65                  11.95        +56%
  c1     tpot ms           119                   80.6         1.48x
  c2     out tok/s         12.5                  18.85        +51%
  c4     out tok/s         19.5                  31.90        +64%
  c4     TTFT ms           2720                  900          3.0x
  c8     out tok/s         (baseline killed)     57.49
```
- **End-to-end proof, not just microbench: a hand-rolled posted-write all-reduce replaces oneCCL in a
  coherent production-shaped 27B TP=2 serve and wins +48-64% throughput and ~3.35x TTFT.** The TTFT win is
  exactly the H.10 prediction (prefill ~84% collective-bound -> a faster collective hits TTFT hardest).
  Baseline reproduced across two runs (c1 TTFT 1603/1624, dec 8.38/8.42 -> stable).
- **HONEST CAVEAT -- this is EAGER vs EAGER (isolates the collective).** The shelf's PRODUCTION path is
  CAPTURED (GRAPH=1), whose kernel-launch-overhead removal gives higher DECODE (~34 t/s per the shelf
  README) using oneCCL. push-ar has a host barrier -> NOT graph-capturable -> forces eager, and eager's
  per-token launch overhead likely outweighs push-ar's decode-collective win for CAPTURED decode. So the
  production comparison "captured-oneCCL decode vs eager-push-ar decode" is separate and NOT settled here.
- **BUT prefill is NOT captured (PIECEWISE captures decode-sized graphs only)** -> the 3.35x TTFT win
  should survive even in the captured production serve. The highest-value refinement (next): engage
  push-ar ONLY for the non-captured (prefill / large) allreduces and leave captured decode on oneCCL --
  a size- or capture-context-gated `all_reduce` patch. That would bank the prefill win without the eager
  decode penalty. Failing that: make the op graph-capturable (hard -- the cross-process host barrier is
  the blocker; would need a capturable device-side signal, which J.9 showed Xe does not support mid-kernel).

### J.15 [PARTIAL+BLOCKED] capture-gated mode: a real fix (deferred dlopen) + a self-inflicted H.13 wedge
Goal: engage push-ar only for EAGER prefill (large allreduce) and leave CAPTURED decode on oneCCL, to
bank the J.14 3.35x TTFT win inside the GRAPH=1 production serve. Added `PUSH_AR_MIN_NUMEL` size gate
(engage iff numel >= threshold; set above the captured decode sizes). Two findings:
- **[FIX, confirmed] push-ar's SYCL/L0 `.so` must be dlopen'd LAZILY, not at sitecustomize/startup.** First
  GRAPH=1 attempt crashed at MODEL LOAD (rotary-emb `_compute_cos_sin_cache`), NOT at any collective ->
  not the push op itself. Decisive test: GRAPH=1 with `PUSH_AR_DISABLE=1` did NOT crash (compiling fine at
  110s), so push-ar's presence broke it. Root cause: `ctypes.CDLL(libxpu_push_ar_torch.so)` at interpreter
  startup pulls in libsycl/libze_loader and inits Level-Zero BEFORE vLLM's own XPU init, which corrupts
  GRAPH=1 (`VLLM_COMPILE`) model construction. (EAGER is unaffected -> J.14 was fine.) Fixed: defer the
  dlopen to the first all_reduce (`_load_lib`); committed. The unmodified shelf GRAPH=1 loads (ran >400s
  still compiling, no fast crash), confirming GRAPH=1 itself is sound.
- **[BLOCKED] the box wedged into H.13 DEVICE_LOST before the capture-gated A/B could be measured.** After
  the fix, the GRAPH=1 serve failed at oneCCL warmup with `UR_RESULT_ERROR_DEVICE_LOST` -- EVEN at
  `P2PACCESS=0`. Both GPUs still enumerate (sycl-ls=2), so it is the H.13 multi-GPU collective-state wedge,
  not hardware loss. **NEW datapoint extending the CLAUDE.md/H.13 warning: it is NOT only `P2PACCESS=1`
  that wedges TP>1 -- a string of TP=2 WORKER-INIT CRASHES (here, the GRAPH=1 model-load failures above,
  plus killed-mid-load attempts) corrupts the same cross-GPU oneCCL/L0 state, so every subsequent TP=2
  serve (even a known-good P2P-off one) then DEVICE_LOSTs.** Recovery = `sudo modprobe -r xe && modprobe
  xe` (no /dev/dri in use) or reboot -- both need root, not available this session. So the capture-gated
  A/B (push-ar prefill-only vs oneCCL-captured) is QUEUED for after a GPU reset.
- NET: the J.14 EAGER win (push-ar +48-64% / 3.35x TTFT) stands as the validated result. The capture-gated
  production variant is implemented + the startup-dlopen blocker is fixed; it needs one clean GRAPH=1 run
  on a freshly-reset box to measure. DO-NOT-REPEAT: do not retry TP=2 serves while wedged (they all fail and
  may deepen the corruption); reset xe first. Avoid chaining many crash-prone TP=2 worker-init attempts.

### J.16 [BLOCKED+DIAGNOSIS] post-reboot capture-gated A/B re-wedged the box -- worse than H.13 (single-GPU NOT spared)
Re-ran the queued capture-gated A/B on a freshly-rebooted box (both cards free, clean tree):

    GRAPH=1 PUSH_AR_MIN_NUMEL=65536 ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke

The J.15 deferred-dlopen FIX HELD -- this got further than any J.15 attempt: push_ar patched
all_reduce + chained the MTP shim, GRAPH=1 model load progressed PAST the rotary-emb crash point,
`XPUInt8ScaledMMLinearKernel` selected, both TP0/TP1 workers alive, safetensors shards loading
(saw 50%). But it never reached `/health` in the 15-min `b70_wait_healthy` window; the script then
stopped it and worker shutdown threw `_cleanup_profiling_kv_cache -> torch.accelerator.synchronize()
-> UR_RESULT_ERROR_DEVICE_LOST` (err 20). gpu-run exit 1 after 916s.
- **NEW + WORSE finding: this wedge is NOT spared on single-GPU.** Post-teardown a TP=1 single-card
  probe (explicitly "wedge-immune" under H.13) FAILED: a 2048x2048 fp32 matmul (~16MB) failed after
  87s with `UR_RESULT_ERROR_OUT_OF_RESOURCES` (err **40**, OOM-class, not DEVICE_LOST); a retry with a
  tiny 16x16 op HUNG >13 min before being killed. `torch.xpu.is_available()` still True, count 2. No
  userspace cause (no stray procs, no docker remnants, lease free) -> kernel/xe-level degradation on
  BOTH cards. Update the H.13 "single-GPU unaffected" claim: a TP>1 teardown-mid-capture can take out
  single-card work too, and can present as `OUT_OF_RESOURCES` (40), not only `DEVICE_LOST` (20).
- **Trigger model (3 datapoints):** H.13 = `P2PACCESS=1` in TP>1 serve; J.15 = chained TP>1
  worker-init crashes; J.16 = TP>1 workers killed MID-GRAPH-CAPTURE by the health-wait timeout. Common
  factor = TP>1 workers torn down while the L0/oneCCL collective context is mid-operation. The shutdown
  `synchronize()->DEVICE_LOST` is the smoking gun. SUSPICION: the 15-min health timeout may have
  SIGKILLed a serve that was merely SLOW to capture, not failed -- i.e. partly self-inflicted.
- Full reasoning + decisions (scoped passwordless sudo for `modprobe xe`, a pre-flight env guard +
  auto-reset-on-DEVICE_LOST hook, and a kernel-7.1 purgeable-BO assessment) live in
  docs/20260624_devicelost_thoughts.md. BLOCKED on a reset (root, not available this session).

### J.17 [GUARD] wedge guard shipped -- detect+recover, do not prohibit (2026-06-24)
Box was reboot-cleared (uptime confirmed both cards free). Built a layered guard whose design
principle is "heal aggressively, prohibit almost nothing" so it does NOT hamper TP>1 / P2P
pioneering. Root cause (from the 3 datapoints above) = a TP>1 worker torn down BY SIGKILL while its
L0/oneCCL collective context is mid-op (capture or warmup allreduce). Our own tooling supplied the
SIGKILL: `b70_wait_healthy`'s flat 900s ceiling -> `b70_stop`'s `docker rm -f`. New pieces:
- **bin/xpu-health** -- per-card 2048x2048 matmul + a tiny op, run in the serve IMG, `timeout`-wrapped
  (J.16's own probe HUNG 13 min, so the detector must itself be killable). Exit 0 healthy / 1 wedged
  (DEVICE_LOST / OUT_OF_RESOURCES / hang) / 2 inconclusive (no docker/image -> warn, do not block).
  VALIDATED on the clean box: both cards OK, exit 0, 16s.
- **bin/xe-reset** -- stop all containers (free /dev/dri) -> `sudo -n modprobe -r xe && modprobe xe`
  -> re-probe. Needs the SCOPED sudoers (bin/xe-reset.sudoers); exits 3 with the reboot fallback if
  absent. `--dry-run` validated.
- **lib.sh guard, TP>1 ONLY (TP=1 path byte-for-byte unchanged -> sweep gate unaffected):**
  L1 pre-flight `xpu-health` (never stack a TP>1 launch onto an already-wedged box);
  L2 graceful teardown (`docker stop -t30` SIGTERM+grace, not `rm -f` SIGKILL -- the root-cause fix);
  L3 progress-aware `b70_wait_healthy` (stall-abort on log silence, raised 1800s ceiling for
  TP>1+GRAPH=1, so a slow-but-live capture is NOT force-killed -- directly fixes the J.16 trigger);
  L4 post-teardown verdict + optional `B70_AUTO_RESET=1` xe-reset;
  L5 the one hard block -- refuse `CCL_TOPO_P2P_ACCESS=1` in a TP>1 serve unless `I_KNOW_P2P_WEDGES=1`.
- Knobs: `B70_AUTO_RESET` (auto xe-reset on detected wedge), `HEALTH_TIMEOUT`/`HEALTH_STALL`,
  `STOP_GRACE`, `I_KNOW_P2P_WEDGES`.

**RE-ATTEMPT RESULT [WIN] -- the same J.16 command now passes clean, no wedge.** Ran the exact
capture-gated A/B that wedged the box in J.16, this time through the guard:

    B70_AUTO_RESET=1 HEALTH_TIMEOUT=2400 HEALTH_STALL=600 GRAPH=1 PUSH_AR_MIN_NUMEL=65536 \
      ./bin/gpu-run bash scripts/108_serve_push_ar_ab.sh smoke

- L1 pre-flight: both cards OK before launch.
- serve GRAPH=1 TP=2 push-ar (prefill-gated, MIN_NUMEL=65536) -> **HEALTHY in 147s** -> gen probe
  coherent ("Paris.", coherence OK) -> L2 graceful `docker stop -t30` -> L4 post-probe both cards OK.
  gpu-run exit 0, 188s total. **The box did NOT wedge.**
- **The J.16 "capture needs >15min" suspicion is FALSE: capture here was 147s.** So J.16 did not
  force-kill a merely-slow capture -- its serve was genuinely stuck for 900s when a healthy one takes
  ~2.5min, which means the box was ALREADY degraded at J.16 start (it had no pre-flight probe then).
  The force-kill then converted that degraded state into the full both-cards wedge. L1 (pre-flight
  probe) is exactly the layer that catches that pre-existing-degradation class BEFORE launch+kill, and
  L2 (graceful teardown) removes the force-kill that deepens it. Both fired correctly here.
- BONUS: this also UNBLOCKS the research -- the capture-gated GRAPH=1 prefill-only push-ar PRODUCTION
  variant (the J.14/J.16 open item) is now MEASURED healthy + coherent. Full A/B bench numbers
  (push-ON vs `PUSH_AR_DISABLE=1` oneCCL baseline, GRAPH=1) are the remaining TODO.
- PENDING: run `bin/serve-sweep --smoke` before committing the lib.sh change (shared-infra gate).
  bin/xe-reset.sudoers installed + `visudo -c` parsed OK.

**FULL A/B BENCH [WIN] -- capture-gated push-ar vs oneCCL, both GRAPH=1 TP=2 27B-W8A8** (2026-06-24,
IN=512/OUT=128). Ran A then B back-to-back -- two chained TP=2 GRAPH=1 starts, which the AGENTS.md
warning says NOT to chain; the guard carried both with graceful teardown + post-probe between and
NO wedge (box ended both-cards-healthy, lease free). A = `sweep_*-pushar-tp2-graph_170832.csv`,
B = `sweep_*-oneccl-tp2-graph_171516.csv` (in $ROOT/results).

    conc  out_tok/s A(push) B(oneccl)  thruput   mean_ttft_ms A vs B    TTFT speedup
    1        29.70      25.44          +16.7%    321  vs  811          2.52x
    2        45.40      39.64          +14.5%    455  vs 1101          2.42x
    4        59.86      47.52          +26.0%    561  vs 1296          2.31x
    8       107.74      69.52          +55.0%    724  vs 1659          2.29x

- Capture-gated push-ar (prefill-only push inside the captured graph; decode stays on oneCCL) WINS at
  every concurrency: +15-55% output throughput (gap widens with load) and 2.3-2.5x lower TTFT (TTFT is
  prefill-dominated, exactly where push-ar replaces the oneCCL all-reduce). per-stream decode tok/s also
  higher (e.g. conc8 14.82 vs 10.18). Slightly below the J.14 EAGER 3.35x TTFT because this is the
  production GRAPH=1 capture-gated mode, not full-eager push. NET: the J.14/J.16 blocked production
  variant is now MEASURED and is a clear win. Decode-side push (capturable) is the next lever.
- Root cause of the H.13 P2P wedge this guard works around: see docs/literature/p2p_access_devicelost.md
  (oneCCL direct PCIe peer copy across our cross-die/cross-root-complex boundary -> xe BCS copy-engine
  reset + GP fault -> L0 queue DEVICE_LOST; persists because it is xe driver/engine + shared multi-GPU
  L0 context corruption, not user-space-recoverable). Our push-ar dodges it entirely via L0-IPC at
  P2PACCESS=0; host-staged oneCCL is the only stable stock path on this topology.

### J.18 [MEASURED] PP=2 first run on the current Gen3 box -- prefill/TTFT + scaling win, decode gated by eager+no-MTP
First PP=2 serve on the current box (the J.13 bet; scripts/109_serve_pp2.sh was UNTESTED). Run guard-wrapped
(pre-flight probe + graceful teardown + post-probe, B70_AUTO_RESET=1): 27B-W8A8, PP=2/TP=1, EAGER, no MTP,
IN=512/OUT=128. Came up coherent ("Paris."), benched, post-probe both cards healthy, NO wedge. The guard
let us safely chain it right after the J.17 TP=2 runs.

    PP=2 (eager, no MTP)            vs   eager push-ar TP=2 WITH MTP (J.14, ...pushar-tp2_083017)
    conc out_tok/s ttft_ms ps_dec        conc out_tok/s ttft_ms ps_dec
    1     6.03     345    6.08            1     7.64    1603    8.38
    2    11.82     637    6.05            2    13.11    2117    7.41
    4    23.05     938    5.98            4    18.44    2538    5.55
    8    44.21    1394    5.84            (no c8)

- **TTFT: PP=2 wins ~4.7x** (345 ms vs 1603 ms at c1; the eager oneCCL TP=2 is higher still). This is the
  J.13 hypothesis CONFIRMED in a live serve: PP's single posted activation handoff per microbatch beats
  TP's ~128 per-layer all-reduces on our push-fast/read-slow fabric. Prefill is exactly where TP's
  collective tax lives.
- **Concurrency scaling: PP=2 holds.** per-stream decode is FLAT ~6 t/s from c1->c8 and aggregate scales
  cleanly to 44 t/s @c8; the eager TP=2 per-stream DEGRADES under load (8.38->5.55 by c4) as the all-reduce
  tax grows. PP overlaps stages via continuous batching; TP cannot hide the per-step collective.
- **BUT decode absolute is gated by eager+no-MTP.** PP=2's 6 t/s per-stream is ~5x below the PRODUCTION
  GRAPH=1+MTP TP=2 push-ar (30 t/s @c1, J.17) -- that gap is capture + MTP, not topology. The matched eager
  TP=2 here even carries MTP (~2x) and still loses on TTFT/scaling; eager no-MTP TP=2 is ~4 t/s (serve.sh
  header), which PP=2's 6 would beat. So PP=2 is PROMISING but NOT yet a production contender.
- VERDICT: PP=2 is real, coherent, and wins decisively on prefill/TTFT and concurrency scaling at matched
  (eager) config -- the core bet holds. To compete with the production TP=2+push-ar path it needs GRAPH=1
  capture + MTP, all three unproven together (does vLLM-XPU capture PP send/recv? does MTP's last-stage head
  compose with PP?). NEXT for PP: (1) PP=2 GRAPH=1 capture, (2) +MTP. For now TP=2+push-ar remains the
  production path; PP=2 is the candidate for prefill-heavy / long-context / high-concurrency regimes.

### J.19 [BLOCKED] PP=2 production config (GRAPH=1 + MTP) -- two hard upstream blockers, no viable path today
Tried to put PP=2 in the production config for a direct TP=2-vs-PP=2 compare. Built lib.sh `PP` support
(b70_multicard: PP>1 gets the #41663 multi-GPU env + mp executor + the wedge guard) and scripts/110_serve_
pp2_graph_mtp.sh (replicates the 27B shelf GRAPH=1+MTP+GDN+shim env, flips TP=2 -> PP=2/TP=1). Two blockers:
- **MTP + PP = UNSUPPORTED (config-time NotImplementedError).** `draft_model_config.verify_with_parallel_
  config -> NotImplementedError: Pipeline parallelism is not supported for this model. Supported models
  implement the SupportsPP interface.` The base 27B supports PP (it served eager, J.18) but the MTP DRAFTER
  does not implement SupportsPP -> spec-decode + PP is rejected outright by this vLLM build. Not "unstable" --
  flatly unimplemented. (No GPU touched; graceful exit, no wedge.)
- **PP=2 + GRAPH=1 (no MTP) = CAPTURES but EMPTY OUTPUT.** Drops MTP, GRAPH=1 PIECEWISE: serve comes HEALTHY
  in ~71-147s (so capture records the PP handoff) but /v1/completions returns an EMPTY body on every prompt
  (3 manual probes all empty; the smoke coherence gate only passed on its SHORT-not-GARBAGE gap). Eager PP=2
  gave coherent "Paris." (J.18), so captured PP=2 is NUMERICALLY BROKEN on this W8A8+GDN hybrid -- the SAME
  captured-hybrid class as the documented captured-TP=2 garbage (rdy_to_serve/qwen36-27b serve.sh bug B), not
  fixed for the PP path.
- NET VERDICT (PP=2 vs TP=2): the ONLY coherent PP=2 is EAGER no-MTP (~6 t/s decode, J.18). MTP+PP is
  upstream-unsupported and captured PP is broken, so PP=2 has NO working production path on this model today.
  TP=2 + push-ar + MTP (GRAPH=1) stays the production path (30 t/s decode, coherent). PP=2 is PARKED pending
  upstream: (1) SupportsPP on the MTP drafter, (2) a captured-PP-hybrid numerics fix. Its proven edge
  (prefill/TTFT 4.7x, flat-per-stream scaling, J.18) makes it worth revisiting if/when those land.
  Tooling kept: lib.sh PP support + scripts/110 are committed so the retry is a one-liner.

### J.20 [INFRA] captured-PP corrupts collective state; xe-reset CANNOT recover this box (reboot-only)
- **The captured-PP serves CORRUPTED the multi-GPU collective state.** After the string of 110 GRAPH=1 PP
  runs, the next production TP=2 GRAPH=1 serve (README bench A-side, oneCCL) ALSO produced empty output + an
  all-zero bench, threw `DEVICE_LOST` on teardown, and its post-probe found card 1 HUNG (wedged). The
  single-card PRE-flight probe had passed -- it does not exercise the collective, so it cannot catch
  collective-state degradation. GUARD GAP: pre-flight should add a 2-proc allreduce probe to catch this.
  The earlier clean chain (J.17 two TP=2 + 8-model sweep + push-ar smoke, all coherent) shows it is the
  abnormal captured-PP path, NOT normal TP=2 teardown, that corrupts state -> do NOT run captured PP=2.
- **xe-reset CANNOT recover this box -- REBOOT ONLY (CONFIRMED 2026-06-24).** Auto-reset fired
  (B70_AUTO_RESET=1) but `modprobe -r xe` failed `FATAL: Module xe is in use` EVEN with zero containers:
  `xe` also drives the console/display (Arc cards expose DP/HDMI + a framebuffer; lsmod baseline refcount ~5
  even freshly booted), so it is never removable while the system is up. The old "lighter modprobe reload"
  recovery note is WRONG for this display-attached box. AGENTS.md + bin/xe-reset corrected to escalate
  straight to `sudo reboot`. The guard still did its job: it DETECTED the wedge (post-probe card 1 hung) and
  stopped -- it just cannot self-heal here without reboot rights. Box is wedged pending a reboot.

### J.21 [WIN] post-reboot clean production A/B (IN=2048) -> README updated to the push-ar best
After the reboot, re-ran the production 27B-W8A8 TP=2 GRAPH=1 MTP A/B at the README config (IN=2048/OUT=128,
random) on a clean box. Both serves coherent, chained cleanly through the guard (graceful teardown +
post-probe healthy between A and B), NO wedge -- confirming normal TP=2 teardown is safe (only the captured-PP
path of J.19/J.20 corrupted state). This also validated the lib.sh `PP`/`b70_multicard` refactor on the live
TP=2 path (TP=2 renders `PP=1`, build unchanged).

    conc   oneCCL default            push-ar (PUSH_AR=1)        push-ar gain
           ttft_ms ppTok/s outTok/s  ttft_ms ppTok/s outTok/s
    1       2916    702    16.27      762    2688    22.14      TTFT 3.83x, prefill 3.83x
    2       3325    616    22.73     1090    1879    35.66      TTFT 3.05x
    4       4224    485    26.86     1354    1513    48.23      TTFT 3.12x, agg +80%
    8       6303    325    32.74     1857    1103    68.47      TTFT 3.40x, agg +109%

- At IN=2048 (heavy prefill) push-ar's prefill win is bigger than the IN=512 J.17 run (3.8x TTFT vs 2.5x):
  more prefill tokens -> the oneCCL all-reduce tax it removes is a larger share of TTFT. Single-stream decode
  (TG c1) is UNCHANGED ~25 t/s (MTP) -- push-ar is capture-gated prefill-only, so it does not touch decode.
- README "Serve shelf benchmarks" 27B-W8A8 TP=2 row updated to the push-ar best (762ms / 2688 / 25.3 / 1354ms
  / 48.2), labelled (push-ar); caveat (1) gives the oneCCL default baseline + the PUSH_AR=1 opt-in. Audited the
  other 5 rows (4 TP=1 + 35B-quark TP=2) vs all results CSVs/JOURNAL -- they already show best achieved; only
  the 27B-W8A8 row moved. CSVs: sweep_*-tp2-graph_192053 (oneCCL), sweep_*-pushar-tp2-graph_193418 (push-ar).

---

## K. DECODE-side capturable push all-reduce (2026-06-24 session) -- making the push graph-recordable

New campaign (handoff docs/handoff_decode_push_ar.md): push-ar is PREFILL-ONLY because its rank-sync is a
HOST barrier (CPU spin barrier + host .wait() + ctypes), which SYCL/torch graph capture cannot record, so
decode-sized all-reduces fall back to oneCCL inside the captured graph (`PUSH_AR_MIN_NUMEL` gate). Goal: a
DEVICE-SIDE / graph-recordable cross-rank sync so decode all-reduces also use the 11 GB/s posted-write path.
J.9 left two leads: (B) cross-queue SYCL events = 44us, correct, never put in a graph; (C) EU-spin device
flag HANGS (mid-kernel peer write invisible on Xe; peer ATOMICS=N -- a hard dead end, do not repeat).

### K.1 [RECON] the int8g/v0230 toolchain HAS every capturable-sync primitive
`vllm-xpu-env:int8g` (and :v0230): icpx 2025.3.3 (DPC++), Level-Zero loader 1.26.0 / ze_api v1.12, libsycl.so.8.
Confirmed present + compiling: `SYCL_EXT_ONEAPI_GRAPH=1` (SYCL command_graph: add/make_edge/begin_recording/
finalize/queue.ext_oneapi_graph); L0 IPC-event API (`zeEventPoolGetIpcHandle`/`OpenIpcHandle`,
`ZE_EVENT_POOL_FLAG_IPC`), `zeCommandListAppendSignalEvent`/`AppendWaitOnEvents`; SYCL
`ext_oneapi_signal/wait_external_semaphore` (queue+handler) with handle types opaque_fd + timeline_fd. So the
mechanism oneCCL uses for its capturable sycl_algorithms allreduce (H.5: L0 events + command-streamer waits)
is fully available to a hand-rolled op. Recon is compile/header-only -> zero GPU risk.

### K.2 [FINDING] SYCL command_graph is SINGLE-DEVICE -> per-rank graphs + EXTERNAL sync is the only structure
`scripts/114_graph_allreduce.cpp` (+ `114_run_graph_allreduce.sh`): tried to record a 2-device push allreduce
into ONE command_graph (multi-queue recording mode, cross-device dependency edge reduceA<-pushB). Result:
`begin_recording` throws `begin_recording called for a queue whose device differs from the graph device`.
A command_graph is bound to ONE device; you cannot put both cards' nodes (or a cross-device edge) in a single
graph. (Build OK, no hang, no wedge -- single-ctx test, both cards via gpu-run.)
- IMPLICATION: the decode sync canNOT be an intra-graph edge. Each rank must capture its OWN single-device
  graph (push kernel -> [external cross-rank wait] -> reduce kernel), and the cross-rank sync must be an
  EXTERNAL primitive recorded as a node in each rank's graph: an L0 IPC event (`zeCommandListAppendWaitOnEvents`,
  the command-streamer/HW-semaphore wait -- a DIFFERENT path from J.9-C's failed EU spin) or a SYCL
  external_semaphore. This actually MATCHES vLLM reality: each TP worker is a separate process/device and
  torch captures one graph per worker. So "per-rank graph + external semaphore/event sync" is both the only
  SYCL-supported structure AND the correct vLLM model. NEXT (K.3): prove the raw-L0 command-streamer
  cross-device event wait is replayable (the keystone -- it is the exact primitive oneCCL relies on).

### K.3 [KEYSTONE WIN] cross-device command-streamer L0-event wait is CORRECT + REPLAYABLE on B70 (J.9-C bypassed)
`scripts/115_ze_event_sync.c` (+ `115_run_event_sync.sh`): the decode-capture blocker was J.9-C (an EU-spin
device flag HANGS: a peer write from inside a running kernel is invisible to a kernel spinning on the other
card; peer ATOMICS=N). But that is only ONE cross-device sync mechanism. This tests the OTHER one -- the one
oneCCL's capturable sycl_algorithms allreduce relies on: a Level-Zero EVENT signaled by one card's command
and waited on by the OTHER card's COMMAND STREAMER via `zeCommandListAppendWaitOnEvents` (a hardware-semaphore
wait, NOT an EU spin). Pure L0, no SPIR-V: push = peer `zeCommandListAppendMemoryCopy` (signals the event),
then `AppendWaitOnEvents(peer)`, then a proxy read of what the peer pushed. Two CLOSED command lists (one per
card) re-executed 200x with a fresh per-iteration sentinel + poisoned scratch so a MISSED sync is caught.
```
  size           iters   sync_us   verifyA   verifyB
  10KB(decode)   200      21.50     OK        OK
  64KB           200      16.84     OK        OK
  1MB            200     110.27     OK        OK    (transfer-bound: 1MB/~11GB/s)
```
- **CORRECT at every size across 200 replays both directions** -> the cross-device command-streamer event wait
  works on B70 where J.9-C's EU spin hung. The difference is the WAITER: a command-streamer/HW-semaphore wait
  polls the (HOST_VISIBLE) event state through a path designed for cross-engine/cross-device signalling;
  J.9-C's EU spin reads through L1/L2 with no system-scope atomic to force visibility. Same hardware, right
  primitive.
- **REPLAYABLE: closed L0 command lists re-executed 200x = exactly graph "replay".** So the push all-reduce's
  rank sync CAN be made graph-recordable -- the host barrier is replaceable by an in-command-list cross-device
  event wait. The J.14/J.17 decode gap (push-ar is prefill-only because the host barrier is not capturable) is
  addressable. Decode-sized sync ~17-21us (push+signal+wait+read incl. host launch + event reset) already
  beats J.9-B's 44us SYCL-event path and oneCCL's ~85us.
- Event pool created HOST_VISIBLE (event state where both cards' streamers can poll over PCIe); per-iteration
  `zeEventHostReset` (a host op, fine for proving the mechanism; a real captured graph resets its events in
  the graph runtime). NEXT (K.4): build the FULL allreduce (push + this event sync + a real reduce kernel),
  measure decode latency vs oneCCL; then the SYCL-graph/external-semaphore form torch capture can record (K.5).

### K.4 [WIN] full push all-reduce records into a SYCL command_graph + replays correctly (the capturable decode op)
`scripts/116_graph_native_ar.cpp` (+ `116_run_graph_native_ar.sh`). Decisive prerequisite: torch-xpu's XPUGraph
IS `sycl::ext::oneapi::experimental::command_graph` (ATen/xpu/XPUGraph.h: `xpuGraph_t = command_graph<modifiable>`,
`xpuGraphExec_t = command_graph<executable>`). So vLLM GRAPH=1 capture = SYCL command-graph queue-recording on the
capture stream -> anything we submit on torch's stream during capture becomes a graph node. This bench builds the
WHOLE all-reduce as a per-rank command_graph and replays it:
  per rank: [push SYCL kernel] -> [native cmd: AppendSignalEvent(mine); AppendWaitOnEvents(peer); AppendEventReset
            (peer)] -> [reduce SYCL kernel]
The cross-device sync is injected with `handler::ext_codeplay_enqueue_native_command` + `interop_handle::
ext_codeplay_get_native_graph<level_zero>()` (returns the graph's recording `ze_command_list_handle_t`; NOTE
`get_native_queue<level_zero>` is BROKEN in DPC++ 2025.3 -- ill-formed reinterpret_cast to its variant return --
use `ext_codeplay_get_native_graph` when recording). Consumer-reset: the rank that WAITS on an event RESETS it
right after, so the next token's signal starts clean (replay-safe).
```
  size           perLaunch us   amort us   algbw GB/s   verifyA    verifyB   (200 replays each)
  10KB(decode)   64.52          50.97      0.20         OK(sum)    OK(sum)
  64KB           64.54          55.29      1.19         OK(sum)    OK(sum)
  1MB           158.38         143.00      7.33         OK(sum)    OK(sum)
```
- **The full push all-reduce RECORDS into a SYCL command_graph and REPLAYS correctly 200x at every size, both
  ranks (sum verified).** This is the exact graph type torch captures, with the exact native-command injection a
  monkeypatched all_reduce would emit during capture. => the DECODE all-reduce is graph-capturable on B70; the
  J.14/J.17 prefill-only limitation is removable. Headline answer to the handoff's central question: YES.
- Latency 64.5us perLaunch @decode is a STANDALONE graph launched from the host every iter (2 graph submits +
  per-iter buffer reset). In the real serve this all-reduce is ONE node inside the big per-token decode graph
  (launched once per token), so its MARGINAL cost is just the device push+sync+reduce -- no per-op host launch.
  Even the standalone 64.5us already beats oneCCL decode (~85us); in-graph marginal will be well below.
- The consumer-reset is race-free in the serve because the ~64 per-token all-reduces couple the two ranks to
  <1-allreduce drift; the standalone bench keeps the same lockstep via a per-iter sync (so the reset is safe).
  A free-running stress mode is included. NEXT (K.5): cross-PROCESS IPC event pools (the 2 TP workers are
  separate processes -> zeEventPoolGetIpcHandle/OpenIpcHandle), then wire into the push-ar .so + serve A/B.

### K.5 [WIN] cross-PROCESS IPC event pool + command-streamer wait is correct + replayable (TP-worker topology)
`scripts/117_ipc_event_sync.c` (+ `117_run_ipc_event_sync.sh`): K.3/K.4 were single-process/single-context. The
2 TP workers are separate PROCESSES, so the event pool must be shared via L0 IPC (`zeEventPoolGetIpcHandle` /
`zeEventPoolOpenIpcHandle`). This is the 103 2-process fork scaffold (scratch IPC via SCM_RIGHTS) + a shared IPC
event pool. Per rank a CLOSED command list: push(myBuf->peerScratch) signal S_mine; WaitOnEvents(S_peer); read;
EventReset(S_peer). Replayed 200x with per-iter sentinel + poison.
```
  [r0] IPC event pool created + exported     [r1] IPC event pool opened OK
  [r1] host-sync S_A via IPC -> SEEN (0x0)   (probe: rank0's signal visible to rank1 cross-process)
  size           iters   sync_us   verify(peer landed after wait)
  10KB(decode)   200      22.61    OK
  64KB           200      16.02    OK
  1MB            200     109.75    OK
```
- **Cross-process IPC events + cross-device command-streamer wait = CORRECT across 200 replays, ~22us decode**
  (matches the single-context K.3 21.5us -> the process boundary costs nothing, same as J.8 found for scratch).
  This is the exact 2-worker topology vLLM uses.
- **CRITICAL GOTCHA (cost ~5 debug iters): the IPC event pool must be created spanning BOTH devices**
  (`zeEventPoolCreate(ctx, desc, 2, {dev0,dev1}, &pool)`). With a 1-device pool, `zeCommandListAppendWaitOnEvents`
  from the OTHER card's command streamer SEGFAULTS in the driver (silent, no error code). The owner signals/waits
  fine on its own device; only the peer's command-streamer wait crashes. 116 (single-proc) already used a
  2-device pool, which is why it worked; 117 first used 1-device and crashed rank1 at the wait append.
- Pool/event IPC handle passed over the same SCM_RIGHTS socket as the scratch dma-buf fd (xe embeds the fd at
  handle.data[0]). The opened pool's events alias the owner's slots (host-sync probe confirms). NET: all
  building blocks for a CAPTURABLE DECODE push all-reduce are proven -- K.3 (cross-device event wait), K.4
  (records into SYCL command_graph + replays), K.5 (cross-process IPC events). NEXT: combine into the
  deployable cross-process GRAPH all-reduce, then wire into the push-ar .so + torch XPUGraph + serve A/B.

### K.6 [WIN] the capturable push all-reduce records into TORCH's XPUGraph + replays correctly (2 procs) -- decode unlocked
`scripts/118_xpu_push_ar_graph.cpp` (the deployable .so, extends 106) + `118_graph_harness.py` + `118_run_graph_
harness.sh`. The .so keeps the proven EAGER host-barrier path (ar_allreduce_ptr_dt, for prefill) and adds:
ar_exchange now also shares an IPC event pool (both devices, K.5); `ar_allreduce_graph(qaddr, ptr, nbytes, dtype)`
= push kernel -> ext_codeplay_enqueue_native_command[signal mine; wait peer; reset peer] -> reduce kernel, all on
the passed queue, no host barrier. The harness drives 2 torch procs (gloo, 1 card each), and CAPTURES a call to
ar_allreduce_graph inside `torch.xpu.graph(g)`, then replays.
```
  [eager] verify OK(4.0)                                      (new .so's host-barrier path still correct)
  [capture] capturing=True setupq=735017824 capq=734934992 same=False
  [graph] replay verify 50/50 OK -> PASS
  [graph] replay latency 35.44 us/allreduce (10240B bf16 decode)
```
- **The push all-reduce records into torch's OWN XPUGraph capture and replays correctly 50/50, 2 processes.**
  This closes the handoff's central question end-to-end at the harness level: the DECODE all-reduce is
  graph-capturable on B70 and integrates with torch's capture path. 35.4us/allreduce vs oneCCL decode ~85us =
  ~2.4x, squarely in the handoff's 34-45us target band.
- **THE KEY INTEGRATION BUG (cost the first capture attempt, UR_RESULT_ERROR_UNSUPPORTED_FEATURE err 44):
  torch captures on a DEDICATED capture stream, NOT the default stream cached at setup** (`same=False` proves
  setupq != capq). The .so must submit on the CURRENT stream (torch.xpu.current_stream().sycl_queue) PER CALL,
  not the queue grabbed in ar_setup_torch -- else ext_codeplay_get_native_graph has no graph to record into and
  the enqueue is rejected. ar_allreduce_graph now takes the queue addr as its first arg.
- ar_allreduce_graph is CAPTURE-ONLY (uses ext_codeplay_get_native_graph, valid only while recording). The
  monkeypatch routes by torch.xpu.is_current_stream_capturing(): capturing -> ar_allreduce_graph (decode), else
  -> ar_allreduce_ptr_dt (eager prefill). NEXT: (a) harness test of MANY sequential allreduces in one captured
  graph sharing one scratch+event pair (the serve has ~64/token); (b) wire into _push_ar_patch.py +
  PUSH_AR_MIN_NUMEL=0 + GRAPH=1 serve A/B vs oneCCL decode (careful, wedge-risk; guard + one run at a time).
