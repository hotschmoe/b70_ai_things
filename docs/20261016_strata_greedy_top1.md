# S112: private greedy top1 on B70 (2026-10-16)

The isolated candidate passed all 48 strict cases. Its completed two-owner
host chain averaged 165.55 us for the reference and 116.41 us for the
candidate across two nested ABBA rounds, about 49.14 us lower (29.7%). This
measures synthetic-logit greedy selection plus the unchanged probability
and selector work on one B70. Serving gain and production qualification
remain unmeasured; S112 changed no production source or default.

CONFIG -> Historical S108 identified two single-WG greedy samplers taking
179.1135 us combined service per examined first-head interval. The existing
multi-WG argmax was excluded by an inherited NVIDIA major8 selector;
SYCL's cluster-greedy route was disabled. Local introducing/port history,
source contracts and actual barrier-header evidence are retained in the
[small-record bundle](../strata/flash-next/lowlevel_probe/greedy_top1_s112/README.md).
Historical trace service and this private host measurement have different
scopes; their difference does not estimate current critical-path savings.

The private candidate clones the existing argmax helpers/kernel into its
own namespace and registered kernel name. It adds one device-scope acquire
on the RMW leader before publishing the last-WG flag through a work-group
barrier. The prior release and post-barrier fences remain unchanged.
Root independently reversed the changed statement and recovered the exact
original source span. The original greedy/probability/select implementations
come from the unchanged S107 archive. A previous B70 runtime failure of the
unselected original argmax was not observed.

For 248320 vocabulary elements, the candidate uses 61 WG256 chunks instead
of one WG1024 scan. The private oracle keeps two owners with separate
reference/candidate buffers and scratch. It tests 4097 elements first,
then 248320, with eager and captured changing A/B/A inputs. Fixtures cover
unique and tied maxima across actual WG boundaries, signed zero, negative
finite values, NaN/-Inf mixtures, all-NaN/-Inf fallback, +Inf ties and a
positive minimum subnormal competing with lower-index +0. The oracle
requires exact IDs against an independent host reference and exact
probability, selected residual, token and mapped-output bytes between
routes. It also checks read-only inputs, guards, unused rows and zero
scratch counters after every completed replay. Scratch is initialized once;
failed capture/replay drains, reports and stops.

COMMAND -> One CPU-only private TU compile/link in the pinned e41 image,
DPC++2026.1, O3 precise, SG32, per-kernel BMG-G31 AOT, four jobs, UID1000,
networknone and noDRI. Original kernels were reused from the pinned archive.
Compile/link durations were 9.355/6.104 s; the wrapper completed in 28 s.
The resulting private executable SHA256 is
`5c8281323022a15f6ff8c89b988cf6fd95d1c4db528f40f6f05c678f407667ec`.

An offline audit bound actual original and candidate host metadata/name
pointers to the selected private-executable images. One candidate ISA
decode was reused for review. Root observed the returning atomic and GPU
invalidate/wait before local flag publication/barrier, and the final ID
store/counter reset. Reference images belong to this private executable;
unchanged archive members alone do not prove production-main device-image
identity. Denormal-mode interpretation outside the tested fixture remains
unknown; no missing metadata field is treated as zero.

Actual source reviews were `claude-opus-5-5` (70.713 s) and
`claude-fable-5-1` (75.236 s). Actual compiled reviews were the same IDs
(37.142/66.121 s). Root read both full replies at each gate and authorized
one finite card0 process after source, selected-image, ISA and runner review.
The runtime used `bin/cpu-run --quiet` outside `bin/gpu-run --card 0`, with
matching ZE/device pins. Explicitly revalidated same-boot S107 health was
reused before the run; card0 post-health was fresh.

RESULT -> Both shapes and all 48 strict cases passed before timing. Only
248320 was timed, with four observations per route in two ABBA rounds:

| Within-route observation | Reference host chain, us | Candidate host chain, us |
| --- | ---: | ---: |
| 1 | 166.116 | 116.482 |
| 2 | 164.744 | 116.261 |
| 3 | 166.136 | 116.342 |
| 4 | 165.205 | 116.543 |

The nested round means were 165.430 -> 116.3715 us and
165.6705 -> 116.4425 us, reductions of 29.655% and 29.714%. These are
completed host chains including both owners' greedy/probability/select
work and queue completion. Staging, output poisoning and readback were
outside each measured chain. No isolated argmax GPU timing, physical
traffic, early mapped-publication timing, population stability or model
throughput was established. Timing sample count is four per route in one
process; the 48 fixtures do not establish arbitrary-input correctness.

Native, identity, analyzer, Docker-start, normal removal, post-health,
driver and tee exits were zero. Final container state was stopped with
OOMKilled=false; the full owned CID and name were absent before fresh
card0 health passed. Before/after identity matched. There was one process,
zero retries, no forced stop and no reset. The full leased wrapper closed
in 26 s. A full collective was not rerun for this standalone one-card
primitive. Exact closure/native/analysis/health records are indexed in the
bundle; binaries, objects, device images and noisy inventories remain on
disk with recorded identities.

VERDICT -> Accept the bounded private PASS. Actual post-result reviewers
`claude-opus-5-5` (29.882 s) and `claude-fable-5-1` (23.814 s) agreed with
that scope. Root prioritizes S113's no-build fixed-policy prefill discriminator,
then separately reviewed defaultOFF MTP integration of this candidate.
Probability omission remains deferred; full-head/HC work stays source-only
until a concrete mechanism warrants another probe.

Root rejected three extensions of the evidence: another S113 mismatch
would not identify the cause of S94 or prove independence from suffix
policy; zeroing scratch after a failed replay would mask failure and is
not allowed recovery; multiplying historical cycle frequencies cannot
produce a current removable serving budget. S94 remains unresolved.
The retained [root post-result decision](../strata/flash-next/lowlevel_probe/greedy_top1_s112/post-review/root-post-decision.json)
records these corrections. Production integration, matched C2 serving and
any eventual winning candidate's context/cache qualification remain due.
