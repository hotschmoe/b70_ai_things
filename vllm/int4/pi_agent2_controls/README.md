# Matched TP1 agent2 fresh-scale controls

CONFIG -> Phase0328900c oncard0/port18151 and phase+MRV1 7b107d0e oncard1/18152.
Both freshbe02 FP8KV, MTP3, prefix ON, FULL graphs,100K, c4, batch32768,
memory0.96, P2P0. Stable primary hotschmoe-dd plus registered control aliases.
No startup TP collective hook in TP1. Strict selected-card lifecycle retained.

COMMAND -> prepare.py freezes independent config/source/scales/manifest/output
paths. Launch each plan later via vllm/cache800k/run_arm.py; its server acquires
the selected lease and pins the device, runs strict selected pre/post health,
and verifies teardown. No GPU launch by preparation.

RESULT -> CPU plan test verifies matched config and scales, selected device/port,
TP1, no TP2-only hook, and exact agent2 warm+target manifest. Startup gate
requires observed block1600 before inference. Replay sends warm once then exact
target twice; strict final gate preserves all heuristic flags and requires
positive cached tokens for both targets. Original failed outputs are never
inserted into request history. All generated tools remain unexecuted.

VERDICT -> Prepared causal controls only. The prior fixed TP2/fresh-scale arm
emitted new garbled invalid JSON absent from input, while older scale/precision
variants produced valid arguments on identical payloads. This pair narrows
MRV1 versus TP/runtime/calibration effects but does not independently isolate
fresh-scale accuracy. Cards differ; cross over if outcomes differ. A stable
serial row0 does not exercise the known concurrent double-permutation defect.
No failure relabel, full Pi semantic pass or production qualification implied.
