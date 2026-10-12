# Fresh-reset paired original MTP controls

CONFIG -> Exact original 34-tensor MTP on GPU1; target 24/24 split on both
B70s, capacity4096. V5 resets both target sessions before binding; precision,
weights, target math and forced control shapes unchanged. This is a control
prerequisite, not a natural-output or serving qualification.

COMMAND -> Root-owned bin/gpu-run invocation of
run_original_mtp53_two_stage_v4.py --phase controls, compile root
/mnt/vm_8tb/b70/build/original-mtp53-two-stage-compile-v5.
Actual session28844 completed normally in368s. Root independently replayed
control_rows, residency, the unique fresh-reset marker, all29 artifact hashes,
allfive pre/post identity rows and normal owned terminal/EOF/removal.

RESULT -> PASS: six restored stage frames and three forced steps, including
rejection, acceptance and own-proposal acceptance. Producer errors empty;
strict per-card and compiled paired collective health, kernel check and
fullfive identity scans passed. The unchanged compile receipt is
830e1bd2cb7bcbbc1465426d47d0e0fdad7b460b0fa5fc35c1d582b7ed5d2415;
ELF1316eceb3a2d901d6811855779307a5cd0d7907530e2dde0815e439c24220963.
Control report is external at
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/original-mtp53-paired-controls-v4/report.json
with SHA25622a0927264130c8ce13862795153c9373855f09de8b8a6dbbf333dbf9817b393.

VERDICT -> The corrected fresh-state prerequisite is admitted. Root launched
session10108 for the same V5 first-case P2048/GEN512 OFF/ON comparison, with
this actual control report digest. Its natural output, reference fidelity and
latency remain pending. No claim of full three-role/repeated qualification,
API serving, cache completion or production speed follows from controls.
