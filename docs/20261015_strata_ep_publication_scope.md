# Strata SYCL EP publication: explicit barrier scope

Date: 2026-10-15. Stage S28. Bounded correctness evidence, not a speed or
long-run stability qualification. Flash-Next ISTA GSQ-RCO IQ3_XXS, two B70s.

Submitted: [recutita/Strata PR2](https://github.com/recutita/Strata/pull/2),
head69dee3c30e881b6d5afb70c197e813811423be40, one file/+1/-1. Target is
the author branch sycl-ep-prefill-main behind upstream PR2205; current
main296b59d2 has no EP kernel file. This fix is independent of PUSH PR1.

## Finding and change

A standalone real-shape EP probe observes the expected ready epoch, but
its first payload copy sometimes consumes current inbox poison. Immediate
second and third reads are clean. The single-kernel producer with an explicit
system memory scope on its workgroup barrier passes the bounded oracle.

The proposed production change is one functional line in
`sycl/src/kernels/cuda/ep_kernels.dp.cpp`, inside `ep_push`:

```diff
 sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
-sycl::group_barrier(it.get_group());
+sycl::group_barrier(it.get_group(), sycl::memory_scope::system);
 if (lid == 0) sys_store(peer_flag, *ctr);
```

Payload stores, the release fence, flag helpers, workgroup size, queue
structure and kernel count remain as before. This is the existing opt-in
SYCL expert-parallel path. The return kernel is unchanged: the evidence
does not justify broadening this patch to its separate publication scheme.

## CONFIG -> COMMAND -> RESULT -> VERDICT

CONFIG: Threadripper 1950X, PCIe3, two B70s at BDFs0b/44; fixed Linux7.1
and Compute Runtime26.22. DPC++2026.1, precise FP32, AOT bmg-g31. In-order
queues and Level Zero IPC mappings. Six rows, N2560, K10: 17520-byte
ids/Q8_1 push and 614400-byte return upper bound, 4096-byte guards.

COMMAND: pair-leased `ep_ready_probe`, `--mode flag --poison producer
--host-sync 0 --reps 64 --max-fail 2 --spin 1048576 --timeout-ms 5000`.
Each sample regenerates payload, inbox poison and distinct private-stage
poison. Passing arms include first use plus64 repeats. Both queues drain
before inspection, mapping closure or allocation release. Debug waits
record the actual wanted/first/last epochs and spin count on the receiver.

| Producer | Wait / extra reads | Result |
|---|---|---|
| Frozen native | Debug / 2 | First use passes, next two first reads fail:1316/1374 poison words; later reads clean |
| Payload and flag in separate same-queue kernels | Debug / 2 | 65/65 pass |
| Same split-kernel diagnostic | Native / 0 | 65/65 pass |
| Single-kernel mirror, default-scope barrier | Debug / 2 | First two samples fail:112/648 poison words; later reads clean |
| Same-binary explicit-scope mirror | Debug / 2 | 65/65 pass |
| Actual native EP translation unit with scoped barrier | Native / 0 | 65/65 pass |

The split diagnostic changes only the publication kernel boundary, not
payload cache hints; it is not the proposed runtime fix. Default/scoped
mirrors differ only in the barrier argument. The failing native/debug arm
records wanted2/3 and actual transitions1->2/2->3; mirror wanted1/2 and
actual0->1/1->2. Wrong wanted epochs do not explain THOSE failures. All
return, guard and spin checks pass. Numerical failures exit normally after
draining; no timeout, GPU crash or reset occurred. Failed-arm timing is
excluded; CPU reviewers were active, so no speed comparison is claimed.

For the final standalone arm, reuse the original S27 probe object and
compile the actual patched EP translation unit before the frozen archives.
The link map confirms that the original EP archive member is not pulled.
This avoids inferring production behavior only from a probe-local clone.

RESULT, engine canary: one incremental CPU-only engine build passes in
87.888s. The frozen new engine, head-pair option OFF and batch depth2,
passes stable/research identity and registry checks. All24576 experts remain
resident:7152 primary and17424 peer. Both concurrent64-token coding replies
match the retained S16 prefixes; each offers46 and accepts40 drafts.
Concurrent arithmetic323/667, separate reasoning391 and the complete tool
round trip pass. Normal stop and lease-wrapper exit0. Final both-card health
and ten compiled two-rank4x5120 BF16/P2P0 collectives pass.

VERDICT: retain the small explicit-scope fix and submit it to the EP
contribution branch. This clears the bounded freshness discriminator before
head-pair serving tests. It does not demonstrate production corruption in
the old engine, a speed gain, complete compiler causality, general IPC
correctness, new full-cache qualification or daily-shelf promotion.

## Native ISA and correction to S27

S27 incorrectly said native BMG ISA was unavailable. `ocloc disasm` had
already emitted hidden `.text.*.asm` files; a plain listing missed them.
Actual Opus5.5 found and independently reviewed them; root reviewed both
the original EP and the mirror/scoped disassembly. Historical evidence is
preserved, with this explicit correction.

| Before ready-flag store | Default mirror | Explicit-scope mirror |
|---|---|---|
| Payload store | store.ugm.d32.a64 | same |
| Pre-flag UGM fence | fence.ugm.none.group,0x0210001F | fence.ugm.invalidate.gpu,0x0210261F |
| Workgroup synchronization | sync.bar | same |
| Ready-flag store | store.ugm.d32x1t.a64.uc.uc | same |
| Trailing UGM fence | fence.ugm.none.gpu | same |

The source system scope survives as SPIR-V CrossDevice memory scope on
the barrier. The decoded hardware mnemonic is GPU scope, not a hardware
system-scope proof. The original separate pre-barrier release is absent
as a distinct CrossDevice barrier in the observed default SPIR-V. A
compiler scope-folding explanation is plausible, but the responsible pass,
legality and full failure mechanism have not been proved. Normal permitted
optimization is not ruled out. The1536-byte mirror/scoped kernel sections
differ in10 raw bytes: one live fence instruction and two null moves after
EOT, not a claim that only one raw byte changed.

## Identities and review provenance

All SHA256 values:

| Artifact | SHA256 |
|---|---|
| Original native EP source | 164ea229cce162fd2de09c894774ed89462d0632110f29958919f604caeb6b85 |
| Mirror/scoped probe source, retained verbatim | a73dc02eb520ca6611f7e04dd36e5e572b94bcfa93b7b9ea5e8c8d7849c9d25f |
| Same-binary mirror/scoped probe | 9a0c3fdcd5a567b20e28cfd380f8e458235175628bab9838de25424a928888bf |
| Actual EP-TU replacement probe | 7ea2a745dc4cc02f545b2b6cb02cca46c2f9dd16d67434eda8c41a854a81fbf3 |
| Frozen engine strata-ep-scope-s28 | 152055a6204a3911e96b59aab8d6788a1a434891e1e1ebd9098c9e21b8d11f13 |

Actual `claude-fable-5-1` prepared the barrier discriminator; actual
`claude-opus-5-5` reviewed original native ISA and lowering. CLI response
metadata verifies the models. Initial bounded sessions hit turn limits;
useful artifacts were retained and Opus's focused final follow-up completed.
Root owns GPU dispatch, native-TU validation, source review and serving.
Root does not adopt Fable's broader return-path change or a proven compiler
bug claim. A native Codex worker prepared minimal upstream applicability
and contribution materials, without GPU execution or remote writes.

The tracked probe adds `--push native|mirror|scoped|split`. Native means
the archive linked at build time; a fresh build against the new archive
contains the fix. Frozen old binaries retain the failing native control.
Raw logs, command arrays, mappings/lifetime audit, ISA and source hashes:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ep-ordering-s28/`.
The summary is `s28-report.json`; the tracked one-line patch is
`strata/flash-next/patches/ep-push-system-barrier.patch`.

Loop reflection: reuse probe objects and hidden ISA before rebuilding.
Batch fully drained numerical arms before one health closure where safe.
Use a small model canary for the one-line transport change; no repeated
four-by200k run, soak, daily-server restoration or fresh framework setup.
The next performance experiment is same-depth head pairing against its
OFF control; the S26 head-pair path itself remains GPU-unqualified.
