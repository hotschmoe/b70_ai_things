# S54: generic PLE helper submitted upstream (2026-10-15)

CONFIG -> [Draft PR #2623](https://github.com/Niko1221/Strata/pull/2623)
contains only the already tested generic prefetch key-match/drain-all fix.
[Bug #2622](https://github.com/Niko1221/Strata/issues/2622) records reproduction.
This is not ready for review: full CUDA/HIP backend builds are outstanding.
No reviewer was requested and no cross-backend or serving-speed claim is made.

The fix matches requested immutable row-vectors to any available prefetched
slot, preserving requested order and allowing duplicates to reuse a slot.
It drains all outstanding tickets after a collect failure while retaining the
first error. The existing cap, fallback, mapped path and logical-byte semantics
remain unchanged. No S49 metadata/counters or HC/kernel change is submitted.

COMMAND -> Refresh main and overlapping contributions; apply the exact existing
S47 patch in a clean linked worktree; verify byte equality with the tested
candidate; commit one file, push to the existing fork and read remote bytes.
Open bug first, add its URL to a final PR body, create DRAFT, then read back
bodies/state/head/file/commit counts. No new tests/build/GPU or runtime mutation.
The CPU checkout lease lasted less than one reported second.

RESULT -> Base296b59d2, source commit
`627db6223f9870a4f0d5598e8042b353dfaeafde`, branch
`hotschmoe/Strata:ple-prefetch-keymatch-s54`. Remote candidate SHA256
`f76bb2b9d8a082ad9489252ff71a7dd9e13cedfc01c92c27fffbf2c95a29bfd3`
equals the actual S47-tested source. Patch SHA256
`217550f5994f9159c9a443a81bc71429f481fd823493b44f189f11fc241199b5`.
API readback: OPEN, isDraft=true, reviewRequests=[], one commit/one file
src/kernels/ngram.cpp. Both published bodies equal the prepared bytes.
No automatic publication review rejection occurred.

Existing validation is scoped: S47 actual CPU88 cases/byte oracle and later
SYCL serving build, with helper common in both S49 arms. Error draining was
source reviewed, not fault injected. No extra CPU rerun is useful for identical
bytes. [S47](20261015_strata_ple_prefetch_readiness.md) and
[S49](20261015_strata_batch_ple_prefetch.md) retain evidence and limitations.

Upstream [AGENTS.md](https://github.com/Niko1221/Strata/blob/296b59d2f73c8f04a805df29c2fa97655adace78/AGENTS.md)
says: "Build every backend a file touches (CUDA, HIP, SYCL) before asking for
review." The draft explicitly discloses missing CUDA/HIP builds, requests no
review and remains unready. This boundary is respected without adding a new
cross-backend build campaign to current B70 optimization work.

VERDICT -> Useful narrow fix published as draft; readiness remains incomplete.
Targeted all-state keyword searches found no duplicate, not an exhaustive
review of every diff. PR1516 adds nearby context requiring rebase if merged
first; its small-gather path remains positional. PR2073/2326 accept this patch.
Main was unchanged at the submission refresh. No long-lived backend fork.

Raw prep and submission folders:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/ple-upstream-s54-{prep,submit}/`.
Frozen evidence: 37 files/584415 bytes; manifest SHA256
`df4d7472956bf08a946807ef27cc9812c2b6dd0d73bde1423dff6be8267f3a59`.
