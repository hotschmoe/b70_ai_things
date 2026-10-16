# S142 matched CPU-ring diagnostic, without unitrace

See [dated findings](../../../../docs/20261016_strata_unprofiled_cpu_ring.md).

- source/: exact executed controllers, direct engine wrapper, config, original
  frozen contents manifest and S136 deltas. The wrapper's executed mode0775 is
  preserved; content-only manifest predates the chmod. No model/binary copies.
- first-startup-failure/: original PermissionError before native EXEC, exit1,
  wrapper1 and root's explicit mode-only corrected setup decision. Preserved
  separately; this was not an unrecorded native benchmark retry.
- runtime/: corrected root phase commands/exits, outside-before-start prehealth
  receipt and inside-after-start exactcopy, native identity, ONE matched pair,
  full820-record ringlog, complete parsed ring/statistics, normal stop and health.
  Pair JSONs retain full request/text/SSE/usage/nativework; no redundant streams.
- independent/: source brackets, precise old/new CPU interpretation and pins.
- source-copy-index.json: absolute external source paths, exactcopy byte/SHA/mode
  identities, two deterministic Docker identity projections and external refs.

S136 full CPU/GPU evidence stays in its existing bundle/external originals.
CPU-comparison.json includes complete old/new rollups; full new parsed analysis
and raw820-ringlog are retained here. No giant rawtrace or duplicate engine logs
are copied. Two Docker identity projections preserve named fields/values; the
index records exact original identities and transformation recipe. They do not
claim complete original JSON identity.

Historical source README names run/reused-prehealth.json before start; actual
root receipt was outside before start and copied inside after run/ creation.
No source was changed for this lifecycle detail. Generic stop fields recording
an unrun functional campaign or trace_JSON_validated=false do not turn this
CPU-only diagnostic into a failed functional or new rawtrace attempt.

No new actual Claude call is claimed by this curation. Independent review is
from the source/CPU evidence agent; root owns decisions and GPU execution.
