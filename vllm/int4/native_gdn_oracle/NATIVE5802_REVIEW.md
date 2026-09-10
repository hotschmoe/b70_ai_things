# Independent bounded review of native5802 port

CONFIG -> Runtime agent's frozen r276-upstream5802-conv-contract.patch against
reconstructed shipped1e90+R35/R50 source. Reviewer made no edits to candidate
source, build files or backend. Exact reviewed hashes and diff are retained
in native5802_review.json. Original installedbinary failure remains unchanged.

COMMAND -> Compared candidate causal_conv1d.hpp to upstream git5802a414;
read active-width checks and launch arguments; evaluated40 chronological
window selections over activewidth1..4, previousaccepted1..4 and every legal
nextaccepted count. All work CPU-only.

RESULT -> Header differs from upstream5802 only by existing R35 dynamic
active-width calculation from num_virtual_tokens/num_spec_decodes, rather
than the allocated state-table column count. This retained difference is
intentional and relevant: state-table columns remain at configuredmaximum
when a call emits fewer tokens. The newly ported rolling-conv kernel changes
match upstream5802. The original physical state-table rowstride is preserved.

Interface guard computes Width-1+(activewidth-1), equal to the kernel's
newstate write extent. It does not derive state length from padded physical
stride or maximum tablewidth. For width4/currentallocatedhistory6:

- Any previousaccepted1..4 selects a valid three-row oldwindow.
- Activewidthq writes2+q rows: oldwindowlast2 followed byq newinputs.
- Every legal nextaccepted1..q selects exactly the chronological three-token
  history after that many newinputs. All40 selections pass.
- When q<4, the unused allocatedtail remains irrelevant: the nextaccepted
  count cannot exceedq. A worker migration may copy a larger physicaltail,
  but resetsaccepted1 and only the valid firstthree rows are consumed next.

The guard covers the newwrite extent. It does not independently validate
arbitrary invalid device acceptedcounts; the reviewed production metadata
contract bounds those counts by the prior valid speculativewidth and the
allocatedsix-row capacity. This review does not generalize to arbitrary
caller-supplied corrupt metadata.

VERDICT -> No source/guard blocker to the deliberate CPU native rebuild.
Preserving R35 while applying the rolling fix is consistent for the production
MTP3 state representation. This is not build success or GPU numerical/model
qualification. Candidate immutableidentity, native dependency checks, health,
unchanged rolling numerical oracle and model boundary regressions remain.
