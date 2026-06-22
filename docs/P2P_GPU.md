# P2P_GPU.md -- multi-B70 GPU-to-GPU comms: kernel, software, and fabric

Status: living research doc (started 2026-06-22). Scope: everything about getting two (or more)
Intel Arc Pro B70 (Battlemage G31, Xe2, `xe` driver) to talk to each other efficiently for
tensor-parallel (TP) inference -- kernel P2P primitives, the vLLM/oneCCL software path, ZML's
compiler-collective alternative, and speculative composable-fabric architectures. Goal of this
project: not just consume the state of the art, but contribute and pioneer new methods for B70 TP.

Cross-refs: [DUALCARD.md](../DUALCARD.md), [FINDINGS.md](../FINDINGS.md),
[docs/literature/02_multigpu.md](literature/02_multigpu.md), [docs/SERVING.md](SERVING.md).


## 0. Our box (measured 2026-06-22)

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
