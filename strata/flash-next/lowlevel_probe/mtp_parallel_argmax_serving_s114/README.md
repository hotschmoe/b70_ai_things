# S114: default-OFF MTP parallel argmax in main

CONFIG -> One new opt-in MTP-only greedy top-1 kernel in the same new main
binary for both arms; STRATA_MTP_PARALLEL_ARGMAX=0/1. All other candidate
features OFF. Existing probability and select stay enabled and unchanged.

COMMAND -> Actual source review, one incremental CPU-only build, selected
compiled audit plus two-image offline EOT addendum, independent host audit,
then one finite OFF/ON C2 comparison under the v2 controllers. One 107/256
warm pair and three fresh 107/1500 pairs per arm; functionals and normal
QUIT/removal/full per-card and collective post-health close both arms.

RESULT -> PASS_MATCHED_DISCRIMINATOR. Aggregate three-pair median
103.073716 -> 103.673778 tok/s (+0.582168%). Per-stream decode medians
53.485314/55.836291 -> 53.858032/56.240725 tok/s. Every reported matching,
marker, functional, normal teardown and health gate passed. Process-cumulative
PLE work differs: gathers129/128, prefetch_tokens429/428, rows199200/199184.
Those totals include prefill/admission/functionals and are not matched
coding-window work. Both raw final aggregates are retained.

VERDICT -> CLOSED, retain default OFF; no immediate repeat or cache campaign.
Actual post-serving Opus68.061816s/Fable86.535391s, terminal0, and root agree.
This remains a bounded descriptive discriminator, fixed OFF then ON with three
nested pairs per process. No statistical, stability, new cache, default or
shelf promotion claim.

The [dated document](../../../../docs/20261016_strata_mtp_parallel_argmax.md)
explains the mechanism, compiled evidence and limits. The
[S112 private evidence](../greedy_top1_s112/README.md) is separate from this
first main integration and does not predict serving gain.

Evidence layout:

- `integration/`: exact ASCII patch, inverse, private header, source pins,
  inverse/default proof, contract and historical design/source packet. Full
  original/candidate UTF8 MTP source remains external.
- `source-reviews/`: actual source replies, filtered responding IDs/local
  elapsed/exit receipts and root source decision.
- `build-prep/`, `build-actual/`: exact recipes, expected source/unchanged
  input pins, actual three-action Ninja plan, compiler/image/ownership,
  build log/exit, changed and unchanged artifact receipts. No objects or main.
- `audit-prep/`, `compiled-first-audit/`: actual selected Info/cache/name/image
  and payload/host forwarding records. The first raw-image HOLD remains
  intact. Source ownership proof and compiled field forwarding are distinct.
- `default-isa-addendum/`: two selected decodes and bounded EOT/control-flow
  proof, including the first analyzer's direct-jump predicate failure and
  correction without repeating either decode. Full device ASM stays external.
- `host-independent-audit/`: bounded actual helper/bind/component-pair tail
  disassembly, exact-string flag gate, launch geometry and owner forwarding.
- `compiled-reviews/`: actual replies, filtered receipts and full bounded
  packet. Root's execution decision is `serving-v2/root-gpu-go.json`.
- `postreviews/`: actual closed-serving replies, prompts/commands, filtered
  receipts, combined S114/S116 source packet and root final decision. S116 has
  ONE CPU-only private build/selected-audit GO, not GPU or production GO. The
  anchor provenance addendum resolves an inspected unbuilt S39 v1 checker to
  the actual built v2 source/747776-byte layout; no old-evidence inconsistency.
- `serving-v1-historical/`: historical six-controller source and validation.
  `serving-v2/`: actual source with owned-CID abnormal-exit checks and a final
  diagnostic gate before post-health; finite root driver included. Historical
  SOURCE-ONLY status records are preserved verbatim.
- `serving-actual/`: configs, live process/loader identity, small exact
  requests/functionals/native/capture/comparison/closure records, plus compact
  derived coding HTTP records. Their full request/text/concatenated SSE
  content/usage/work/metrics are retained. Raw chunk metadata/timestamps and
  response IDs remain external; no raw chunk-boundary or raw ID equality
  claim. `curation/compact_http_records.py` records the deterministic transform.
- `record-copies.json`: source-target exact hashes, derived record identities
  and external raw HTTP identities. `external-artifacts.json` indexes selected
  binary identities from retained receipts without a new binary read/hash.

The host marker proves first eligible HOST selection once, possibly during
startup capture. It is not GPU dispatch, call-count, all-shape or all-owner
coverage. Absent spill/scratch metadata is UNKNOWN. Raw full sampler,
probability and candidate images differ; accepted native computational-prefix
reuse is narrower than whole-image/SPIR-V/program equivalence. Whole-pair
graph is OFF and its caller bodies were not disassembled.

Root rejects universal FP stored-forwarded equivalence and division/mul0.25
claims in Fable's S116 reply; actual materialized operands, instruction order,
control mode and lowering remain audit gates. S111 probability omission is
SOURCE ONLY, so Fable's declaration that the entire probability/selector/
submission class is experimentally closed is not accepted.
