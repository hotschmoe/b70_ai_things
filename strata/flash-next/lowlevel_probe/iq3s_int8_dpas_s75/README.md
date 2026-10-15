# S75 IQ3_S C6 INT8 DPAS record copies

Historical source/controller/evidence copies for review. These retain exact
absolute paths to external preparation, build and run locations. They are not
an in-place runnable recipe; do not execute copied controllers or overwrite
completed build/run paths. Any authorized future run needs separately reviewed
new outputs and root identity/cleanup decisions.

CONFIG -> Two real type21 IQ3_S C6 matrices, canonical public Q8_1 input,
unchanged private original and one local-memory SG16 INT8 DPAS mapping.
COMMAND -> V2 adapter-only supported AOT prelink and ordinary host link; one
quiet card0 window; eager/repeat/changed/single+20-call graphs, full byte gates,
then eight nested OCCO/COOC timing rounds for each shape.
RESULT -> 118 snapshots, 92 binaries, 10 DOTs/86 nodes/76 edges, immutable
inputs/weights/Q8, guards, tails, pre/post health and owned-container removal
PASS. Candidate completed-time medians were about 7.7-7.9 times the original.
VERDICT -> Keep production unchanged. No serving arm, retry or knob sweep.
This rejects the tested conservative mapping, not INT8 DPAS generally.

See docs/20261015_strata_iq3s_int8_dpas_probe.md. record-copies.json binds all
exact copies to original paths/hashes. No model payload, compiled object,
executable, full snapshot set or full ISA is copied. Full candidate assembly
and canonical image identities remain external under iq3s-int8-dpas-s75-isa.

Copied source preparation and static-audit status strings retain their historical
pending wording; root-runtime-decision.json records the subsequent completed
primitive. V1 remains preserved source-only. V2 changes only the FP statement
split plus paths/pins, without a v1 compiled comparison. Actual source review
completion records identify responding Opus5.5/Fable5.1. Root read both full
post-result responses and both accept the bounded rejection/no GPU followup.
Completed external manifest: iq3s-int8-dpas-s75-completed-manifest.json, SHA256
ff730eec68f5f4bc74220c58c29e330f5f4aac0fa75ebbb08a22a5b8b4ab4287.
The dated report binds source/post review and ISA manifests separately.
