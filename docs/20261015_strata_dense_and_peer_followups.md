# Strata dense and peer followups, 2026-10-15

S63 supports preparing one private IQ3_S C6 decode-helper oracle. S64 holds
peer head/dense offload: the current EP path has no bounded existing mechanism
that supplies the required ownership and dependencies. Neither stage measured
a new GPU implementation or changed the serving candidate.

Native audit authors: `/root/ple_upstream` (S63) and `/root/strata_ep_s27`
(S64). These are native Codex audits, not Claude model reviews. Root accepted
GO for S65 source preparation and HOLD for S64. Later
[S66 machine-code evidence](20261015_strata_iq3s_existing_isa.md) narrows S65 to
SWAR-only sign decoding; the aligned-load part of the original S63 proposal
is dropped before build. Frozen S63 findings remain unchanged.

## S63: exact integer decode reuse, GO for preparation

CONFIG -> Current ISTA GSQ-RCO model, main layers 0..47, native IQ3_S/type 21
C6 dense path. Pinned IQ source SHA-256:
`acaa7da63733af9f3831b3a74b99e6176bf46a06e498e0a33c0591f7f4a29851`.
No model payload, compiler, GPU, source/archive change or Claude review.

COMMAND -> Inspect current dispatch and existing expert decode helpers; read
GGUF descriptors only; join format/shape/count to retained S31 cycles 5..9;
execute `python3 audit.py` for finite integer and load-extent contracts.

RESULT -> All 55 IQ3_S header tensors account for 55 calls/cycle and historical
1.9335818 ms/cycle primary GPU service. The largest groups are 2560 -> 6144
(10 calls, 0.8394336 ms), 2560 -> 640 (40, 0.5916298 ms), and 6144 -> 2560
(3, 0.3299362 ms). Header roles do not identify individual trace events.
These are GPU service summaries, not additive request latency or recoverable
savings.

Observed C6 dispatch reaches `row_dot_multi -> Split<21>::load/apply`; weight
decode already reuses each part across six Q8 columns. One distinct candidate
reuses existing expert `load8_a2/load4_a2` and SWAR sign conversion only at
`TY == 21 && NC == 6 && EXACT_N && !STAGE_GRID`. Keep the global Split helper,
all floating-point expressions, K assignment and reductions unchanged.

The host command exited 0: 512 sign-mask cases, 8192 codebook/sign cases and
32 alignment/load-extent cases matched. All aligned helper reads remain within
the immutable 110-byte block, including the final block. The helpers request
12 or 20 bytes per part versus the old six halfword loads requesting 12 bytes;
there is no memory-traffic claim. Compiler folding of the original byte/load
expressions remains unresolved. Prior expert SWAR parity and slower grid
staging are contextual evidence, not a result for this dense C6 substitution.

VERDICT -> GO for one private original/candidate/public-anchor source oracle,
with actual Claude reviews before build/GPU. First representative:
`blk.0.attn_gate.weight`, type 21, N2560/M6144/C6, 6758400 bytes; optional
same-probe N6144/M2560 covers the distinct loop balance. Require strict full
output bytes, finite/guards, immutable weights/Q8 and changed-input direct/
graph checks before whole-consumer timing. No tolerance change, serving claim,
DPAS reroute or staging sweep. Root assigned S65 preparation separately.

Frozen evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/iq3s-dense-next-s63-prep/`.
Manifest `manifest.json` SHA-256:
`2a4c13115c83153a202f7d11ada7fb45e0465ce5b11f3884f2ee1f8bc35f43b4`.
Its seven files retain the exact executed audit, descriptors, S31 join,
integer/load contract, execution record and source pins.

## S64: peer head/dense overlap under native EP, HOLD

CONFIG -> Current all-24576-expert resident EP, C2/depth2 paired draft heads,
Q5_K vocabulary head N2560/M248320, unchanged 200k slots and RAM-cache goals.

COMMAND -> Inspect command guards, head ownership, verifier/drafter graph order,
EP queues/waits/IPC and retained S25/S31/S55/S58 evidence. Read source only;
no payload, review, build, allocation, GPU or configuration change.

RESULT -> EP explicitly rejects layer-split, peer-device and pipeline-windows.
The existing last-stage head/drafter placement is whole-stage ownership, not
an EP-compatible head-only facility. Primary owns the shared target/draft head.
The peer graph's in-order per-layer wait/copy/expert/return chain is submitted
before primary. Target C6 head input exists only after the final expert return,
combine/residual and final HC; it cannot run during those earlier peer waits.
The C2 draft head also depends on both final sample rows and feeds the next
step's token/probability. Whole-head relocation alone supplies no existing
same-chain primary work to hide it under.

Appending dense work behind an EP wait can create a dependency cycle. A second
queue is not proven impossible, but would need explicit readiness, imported
memory, scratch ownership, capture joins and teardown across both users.
NativeHead's ordinary primary allocations are not existing peer IPC mappings;
its device getter is not a ready-made SYCL ownership guard. Full-vocabulary
probability/sampling semantics prevent treating argmax-only sharding as an
equivalent solution. S28 publication correctness does not validate a new IPC
protocol or lifetime.

Saved peer FINAL 2010 MiB is not initial free memory or a current capacity
measurement. A 437043200-byte head replica is 416.796875 MiB, leaving logical
1593.203125 MiB before new buffers/runtime margins; arithmetic fit does not
qualify residency. Preserve all experts, quotas, context and RAM recovery.
S31 peer wait service is not physical idle, spare execution capacity or an
additive savings pool. Earlier empty-card head parity does not answer loaded
EP overlap; S55's small serving signal supplies no universal savings ceiling.

VERDICT -> HOLD a new oracle or integration. No existing bounded placement/
queue knob answers the actual dependency question. Revisit only with a concrete
small EP-compatible ownership/dependency design or independent-work/readiness
evidence. Do not repeat an empty-peer head replay or infer multi-queue failure
from source alone.

Frozen evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/peer-head-overlap-next-s64-prep/`.
Manifest `manifest.json` SHA-256:
`4e008686c8e225e503fab672944e7284762efc4e1295bd9f28781830a12e8348`.
It binds `findings.md` and `source-evidence.json`, which retain exact source
lines/hashes and the historical evidence references.
