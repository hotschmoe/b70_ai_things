# FINDINGS.md -- current evidence ledger

Updated 2026-08-26. This file intentionally contains only findings that should
survive the clean-stack refresh. Detailed commands and raw evidence are in the
linked docs and the active JOURNAL.md window.

## Host and topology

CONFIG -> Linux 7.1.0-070100, two Intel Arc Pro B70 cards on separate root
complexes, host Intel Compute Runtime 26.22.38646.4.

RESULT -> Kernel 7.1 aligned the GuC firmware requirement and cured the former
BCS copy-engine/device-lost hardware wedge. The cards are headless, no process
holds /dev/dri for display, and both PCI functions plus xe auxiliary children
can be unbound for non-reboot recovery.

VERDICT -> Keep kernel 7.1 as the host baseline. Use bin/xe-reset before
considering a reboot. The retired display-hold and GuC-pin theories must not be
reintroduced.

Evidence: docs/P2P_GPU.md and
docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md.

## TP=2 software boundary

CONFIG -> vLLM multiprocess TP=2, oneCCL, graph capture, direct P2P experiments.

RESULT -> Raw peer DMA and standalone oneCCL P2P work. The dangerous failure is
the vLLM process/queue handoff: repeated worker-init or graph-capture failures
can poison later collectives and sometimes both cards. A clone-safe custom-op
contract alone was insufficient; the cloned profile input also needed
completion before another queue consumed it. The proven 8192-row fence was
shape-specific.

VERDICT -> Do not enable arbitrary P2P in vLLM serving. Measure each dense 27B
profile shape and collective count, require matched rank evidence, and run
per-card plus compiled collective health around risky attempts.

## Graph replay boundary

CONFIG -> Exact June Qwen3.6 W8A8 stack on the pinned vLLM image, matched
PIECEWISE and no-MTP FULL-decode arms.

RESULT -> PIECEWISE has 41 graph pieces, 41 fence resets, 41 host event waits,
and 82 command-list submissions per token. FULL decode with Triton attention
has one fence, two waits, and two submissions. It measured 61.5536 tok/s versus
50.3706 for the exact local PIECEWISE control, a 22.20 percent gain.

VERDICT -> Boundary collapse is a proven B70 lever. A refreshed dense 27B stack
must test its own no-MTP FULL arm before rejecting FULL capture.

Evidence: docs/20260826_qwen36_graph_runtime_profile.md.

## Steve transfer

CONFIG -> Exact source checkpoint e190923b, recovered June122 native source,
pinned process UMD 26.14, synchronized and unsynchronized controls.

RESULT -> June122 scratch-targeted quant output measured 50.3706 tok/s versus
48.5315 for matched June9, a 3.79 percent gain. Synchronized model-forward
improved 3.00 percent but remained 21.9944 ms versus Steve's 5.6946 ms. Native
grouped MoE was not the missing lever: Triton W8A8 MoE under FULL reached
64.9843 tok/s. Source-default c10d collectives replicated at a 66.3438 tok/s
mean versus 64.9944 for the custom route.

RESULT -> Steve's accepted 85.8691 arm used default FlashAttention with
PIECEWISE capture. Default FlashAttention cannot enter FULL on this runtime
because its SYCL work-group scratch is unavailable to the SYCL Graph
extension. The exact pinned image already contains Steve-generation UMD 26.14;
host UMD 26.22 is not the missing process-layer match.

VERDICT -> Preserve Steve's trees and exact controls. After the stack refresh,
localize integrated collective cost and other in-graph operators. Do not copy
MoE-only layerlets into dense 27B work.

Evidence: docs/20260825_steve_qwen36_w8a8_forensics.md,
docs/20260825_steve_stack_reproduction_program.md, and
docs/20260825_steve_stack_component_ledger.md.

## Push all-reduce

CONFIG -> TP push communicator initialized before model allocation on the exact
June runtime.

RESULT -> Both ranks imported scratch and IPC events, entered graph capture,
then stalled at the first native push graph submission.

VERDICT -> Preinit fixes IPC import, not loaded graph submission. Do not launch
a full model push arm until the loaded-context oracle captures and replays.

## Ornith-1.5 W8A8

CONFIG -> Local per-output-channel symmetric INT8 RTN, retained Shisa trained
MTP sidecar, sglang TP=2 qualification.

RESULT -> The retained artifact contains INT8 routed experts and eligible text
linears while vision, routers, GDN, lm_head, and the trained MTP sidecar remain
BF16. The first profile was launch-bound and established a current integration
baseline, not a final kernel endpoint.

VERDICT -> Ornith W8A8 remains a headline target on refreshed sglang. Rebuild
backend/kernel binaries before requalification.

Evidence: docs/20260824_ornith15_w8a8_profile.md.

## Model and artifact policy

RESULT -> Live weights are limited to eight manifest entries across Qwen3.8,
NVIDIA Qwen3.6 NVFP4, and Ornith-1.5. Old ABI-specific builds and Docker images
are quarantined.

VERDICT -> Do not restore an old binary into updated PyTorch/backend ABIs.
Rebuild from kernels/ and retained backend source, then requalify.
