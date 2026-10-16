# Strata night handoff, 2026-10-16

Stopped at the user's request after finishing S148. No new experiments or model
consultations should start tonight. The longer optimization goal remains
incomplete and is paused, not achieved.

Both experiment servers shut down normally; their full container IDs and names
are absent. Both cards and compiled collectives passed post-health. GPU and CPU
leases are free. The daily server was not routinely restored. Existing user
changes were preserved. Kernel/driver/backend stack was not upgraded.

Last completed work:

- S144 grouped QSA commit had no observed matched serving gain; keep OFF/parked.
- [S145 private GDN](20261016_strata_gdn_commit_layer_oracle.md) passed complete
  235339776-byte state and continuation checks; its isolated bundle median
  improved26.04%, which does not forecast serving.
- [S146 native tags](20261016_strata_native_graph_tag_method.md) work on card0,
  but substantially perturb this small graph. Routine integration is parked.
- [S147 source/build](20261016_strata_gdn_commit_layer_main.md) passed; new main
  SHA98f0357dd01d176476edcb101ec489280deb2925399a64d31b44ef3abfe17200.
  Default-OFF GDN batching uses existing helpers and a narrow layout guard.
- [S148 same-new-binary serving](20261016_strata_gdn_commit_serving_hold.md) is HOLD. Warm and repeats1/2 matched exact work.
  Repeat3 retained identical request/text/SSE/usage, but owner1 drafts/accepted
  changed1140/934->1142/933; windows556->557, solo2->3, roundedrows5.33->5.32.
  No cause is assigned. ON functionals were not run after this gate failed;
  the final comparator was not run. All measured rows are retained, with no
  selecting the first two pairs or claiming aggregate speed. No retry tonight.

The all-candidate-OFF S148 configuration is the current comparison baseline,
not a newly promoted shelf. Both arms used analytic RoPE0 and explicit
GDN_SPLIT0/GDN_COMMIT_SPLIT1/DBG_GDN0, fixed suffix0/lookup0/spec4/MTPmax4/
min-p0, no CPU ring/unitrace, QSAgroup/C2/C6 and other candidates OFF apart
from the sole GDN flag under test. Keep the measured controls and failed gate.

When work resumes, first decide how to handle S148's unmatched terminal work
without relaxing parity or asserting a numerical fault. Source-only review also
identified a possible later small combined configuration: Q5 C2+C6 runtime
words01010101 and attention-value pipeline1; that combination is unmeasured
and percentages are not additive. Consult actual Opus/Fable at that decision.
No further commit restructuring or profiling campaign is scheduled here.

After a combined configuration earns selection, reuse the S30 full-context
sequence: two active near200k histories, then four distinct conversations
with two parked under32768MiB/four-entry RAM budget. Require nonzero restore
payloads, held-history reuse, new prompt work<=128, prior-tail replay0,
needles/no cross-chat leakage and real overlapping decode. The2s cached TTFT
is diagnostic. Retained S30 correctness is bounded to its own configuration.

Do not run historical S30 launchers unchanged: they hardcode an old executable,
image, observer and container. bench_context_four.py also expects
strata-ab-serve; current serving controllers use arm-specific names. The tracked
serve-iq3xxs-200k.json is an older16GiB/split27 control, not the S30/current
32GiB EP template. Adapt the existing harness narrowly rather than add a new
framework. Historical full run was about13minutes, not a new speed estimate.

Current runtime evidence remains under the retained result root in
gdn-commit-layers-s148-serving-prep/root-HOLD-summary.json. Build and serving
patches, strict gates and source identities are preserved in the dated records.

[Retained source-only proposal](../strata/flash-next/lowlevel_probe/gdn_commit_layers_serving_s148/handoff-source-findings.md) was frozen before S148 closed; its pending status is historical.
