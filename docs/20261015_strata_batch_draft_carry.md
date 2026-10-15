# Strata batch-to-solo draft accounting

Date: 2026-10-15. S30; telemetry correction, no inference-kernel change.

## Problem and change

In S29, a 1500-token request returned only 28 offered and 21 accepted
drafts. Those figures belonged to its final 31-token solo tail. Earlier
batched draft counts were lost when the final native DONE replaced the
slot leg's counters. Wall-clock throughput and output hashes remain valid;
complete-request acceptance comparisons from that experiment do not.

`serve/server.py` already carries generated tokens, decode time and prompt
figures across solo/slot handoffs. Seven additional helper lines carry
optional `drafts_offered` and `drafts_accepted` through the same path.
Absent or None fields stay unknown; an explicitly reported zero remains
known. Mixed reporting preserves the sum of known legs, which can still
be partial when an older engine omits fields.

The separate patch `strata/flash-next/patches/batch-leg-draft-carry.patch`
applies after the existing batch-MTP restoration/accounting patch. The
original S16 patch and historical evidence are preserved. The new patch
extends the existing CPU test module with one method covering six cases;
it does not add a new test framework or rebuild the native engine.

## Evidence

CONFIG -> Upstream main296b59d2 and own open accounting PR2511 were
refreshed before changing source. The seven helper lines and regression
were applied to both the runtime and the owned contribution worktree.
Frozen native engine152055a6, paired heads ON, batch depth2, spec4,
two200k slots, 32GiB/4-entry parking and all24576 experts remain the S29
candidate. Runtime Python SHA256 after the fix:
`a18cabeeecd151925c9835f53a8a296c16a8b8f31adb1b8372b75190532ecfc7`.

The actual native argv contains `--spec 4`. INFO advertises `spec=6`,
`mtp_max=4`, `lookup=3`: existing suffix drafting expands the verifier
window while preserving the requested MTP maximum. The startup log
separately confirms batch depth2, so each batch slot emits at most3 rows.
This distinction matters when identifying head shapes in later traces.

COMMAND -> Run the actual new test method against AST-extracted original
and patched helpers. Run the existing slot-draft module and the existing
server handoff/metrics tests in the serving image, CPU-only with no DRI
device or network. Start one leased observed server and send concurrent
64/512-token versions of retained S29 coding requests.

RESULT -> Original helpers fail the multi-leg regression, reporting
28/21 instead of the constructed1134/935 total. Patched helpers pass all
six cases, including absent/None and explicit zero. Six project CPU tests
pass, including both existing full server tests using their fake native
protocol engine. No host dependency installation was necessary.

The live long request has an actual BDONE(cancel) with40 accepted/46
offered, followed by a final solo DONE with304/371. The sum344/417 equals
both final server state and API timings. The short request's39/46 also
equals its native completion sum. Both output texts are prefixes of the
retained S29 outputs. Completion counts are64/512; SSE events are never
treated as individual tokens. Arithmetic323/667, separate reasoning391
and the tool round trip also pass.

The same server subsequently passes the full winning-candidate cache
oracle: four199501-token histories, two active256-token replies, and
four RAM returns each reusing199839 held tokens/reading31 new/replaying
zero retained history. Draft KV byte-readback passes. Normal shutdown,
lease exit0, both-card health and ten compiled two-rank collectives pass.
See S30 in `20261014_flashnext_strata_optimization.md` for memory,
restoration and mixed-prefill diagnostics.

The follow-up was pushed and API-verified on existing
[Strata PR2511](https://github.com/Niko1221/Strata/pull/2511),
head `1fce5591a0ede4d5451316f462ad8acd528a21e6`. It remains open;
this is a contribution, not an upstream merge claim.

VERDICT -> The observed batch-to-solo loss is corrected. This is not a
new throughput comparison, an acceptance-quality comparison, or proof of
all cancellation/multiple-handoff schedules. Native generated counts are
not summed across admission and BDONE because those can overlap.

## Observation and review limits

The diagnostic entry installs the ordinary Intel adapter hooks and leaves
native Popen, the executable, and the ordinary pump unchanged. It buffers
completion records per request and writes once after the original
generator's finally/add_carry. The records contain normalized DONE/BDONE
tokens, final fields and source hashes. They are parser observations,
not a raw stdout trace. Small observer overhead remains, so timings from
this server are diagnostic rather than clean performance evidence.

Actual responding models were `claude-opus-5-5` and `claude-fable-5-1`.
Their initial three-turn sessions hit the turn cap; Opus initially left
no verdict. A resumed, tool-free Opus response completed successfully,
reviewed per-leg replacement/carry paths and approved this bounded live
oracle for the telemetry-only fix. It did not prove every cancellation
schedule. Fable retained a preliminary GR-down split/direct lead; source
selection and numerical/performance evidence remain separate work.

A resumed tool-free Fable cache review also completed. Its requested
draft readback and phase-aligned RAM evidence were already enabled. Root
rejects its claim that alternating single-slot windows could average
above3: native `bt_windows` increments once per verifier submission and
`bt_rows` sums that window's kept rows. If each window had only one slot,
every count and hence their mean would be at most3. A mean above3 proves
at least one two-slot window; an additional arbitrary above4 threshold
is unnecessary. Source pointers and the full dissent are retained in
`fable-cache-review-disposition.json` and `model-review-closure.json`.

The cache harness now reads the native startup batch depth and requires
average emitted rows above depth+1, plus overlapping content intervals,
to prove two active decode slots. At depth2 a single slot can emit3 rows;
the old above2 threshold was insufficient. A native warning disabling
batch MTP fails the gate. Fresh startup-log attribution remains required.

Raw evidence is under:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/draft-carry-s30/`.
See `cpu-regression.json`, `cpu-project-tests.json`, `source-change.json`,
`launch-input-hashes.json`, `live-fixed-api/native-counter-verdict.json`,
the frozen first-pair protocol there, and `opus-carry-verdict/`.


## PR2511 review follow-up (2026-10-15)

The04:12 review comment predates carry fix1fce5591 at05:05. A fresh actual
Opus5.5 consultation (45.081s) confirms the fix is present; root corrects
its suggested legacy wording to preserve any existing admission counts.
Follow-up91b02a5c documents optional BDONE accepted/offered fields and
removes independent restore-payload telemetry from the PR. Non-MTP slot
counts are0/0; accepted means verification-accepted, which can include a
draft not emitted when the request stops. Server helpers/tests and draft
emitter changes are unchanged; cache header/test return exactly to base.
Six existing targeted CPU tests pass in1.937s (CPU-only container11.236s).
No CUDA/SYCL rebuild or GPU rerun; CUDA remains inspected, not built.

The existing fork branch was fast-forwarded, PR title/body updated and
[the reply posted](https://github.com/Niko1221/Strata/pull/2511#issuecomment-6199010242).
Final public six-file PR scope and reply body are verified. The measured
local runtime/combined overlay is unchanged; telemetry is parked as a
separate patch, with no new PR. [Exact small follow-up records](../strata/flash-next/patches/pr2511-followup-20261015/README.md).
