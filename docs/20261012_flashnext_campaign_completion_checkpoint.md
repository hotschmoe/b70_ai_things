# Flash-Next campaign completion checkpoint: 2026-10-12

The full objective is docs/20261009_flashnext_goal_prompt.md. This checkpoint
supersedes the old audit's historical permission/build status, without replacing
its unmet requirements. The selected artifact is Unsloth UD-Q4_K_XL on two B70s
plus host RAM. Requested performance targets remain 2000 tok/s prefill and
50 tok/s decode; neither is achieved by qualified representative results.

## Current evidence

- Original MTP paired natural diagnostic: normal prefill cuts produce identical
  full512 histories with MTP OFF/ON and the accepted serial reference. See
  docs/20261012_flashnext_original_mtp_normal_prefix_cuts_result.md. This is
  deterministic greedy history evidence, not a full original/native math proof.
- HC T2 reuse: actual paired HC0/HC1 run passed. Full512 histories and EOS match;
  root reconstructed both raw reports and verified all52 artifact hashes.
  Results live under f17-source37-20261010/hc-t2-marked-paired-natural-v3.
  Decode elapsed37596.477026->28670.911860ms, unchanged prefill
  43768.756374->40341.620893ms. One ordered instrumented pair cannot establish
  a repeat-qualified speed benefit.
- Teacher prefill head omission preserves full512 history and reduces own-head
  calls2048->1, with2047 skipped. See
  docs/20261012_flashnext_original_mtp_teacher_head_result.md. Unchanged decode
  timing varied substantially; the prepared ABBA must run before a clean claim.
- Source58 cooperative scheduling: full512/64 histories match OFF/ON and serial.
  Largest long-stream pause13427.507->942.970ms, with worse median/P95 gaps.
  See docs/20261012_flashnext_source58_cooperative_scheduler_result.md.
  Actual151 service events show real stage work; no device-only time or actual
  runtime budget ledger was inferred from host elapsed.
- Loaded shared original MTP: one original34 weight pack and four private
  sessions on logicalGPU1 passed full248320-head, proposal and complete-bank
  interleaved/serial comparison, stale-owner refusal and lifetime retirement.
  Results: f17-source37-20261010/original-mtp53-shared-loaded-v1-run-v1.
  Report SHA256 b86da0aac41fb817f280b8b3424bfeeb347663dade8f875e8c36a355d27a7a32.
  Synthetic teacher rows do not qualify MAIN math or API/cache behavior.
- Strict card and compiled collective health, original identity and normal
  owned retirement passed for the completed experiments above. The optional
  real-model host timing build completed; its runtime is pending, not evidence.

## Required work still open

| Requirement | Remaining qualification |
| --- | --- |
| Exact model fidelity | Original/native arithmetic contracts and remaining layer2 discrepancy; broader intermediate/full-logit checks |
| Original MTP serving | Connect shared weights/private sessions to paired slots and actual committed residual rows; terminal catchup and state snapshots |
| Complete prefix caching | Matching target and draft recurrent/QSA/indexer/PLE state; reuse counts, cache logits, shared/divergent histories, eviction, rewind, cancellation, stale identity and restart |
| Interactive concurrency | Real API one/two/four requests and bounded six; isolation, cancellation and state migration; latency/fairness distributions |
| Profiling | Admit current host spans, correlate host/queue/device/transfer/cache/sampling/stream intervals with calibrated ownership and trace coverage |
| Performance | Clean matched interleaved/ABBA results, repeated launches and meaningful negative controls; both requested tok/s targets unmet |
| Memory hierarchy | Cache memory/expert residency tradeoffs and measured RAM/NVMe PLE comparisons |
| Reproducible serving | hotschmoe-dd first, live /v1/models cross-check, manifest, qualified shelf, launch/stop instructions |

No shelf promotion or full-goal completion is claimed. Preserve failed controls
and frozen source plans. Root remains the sole GPU experiment coordinator.
