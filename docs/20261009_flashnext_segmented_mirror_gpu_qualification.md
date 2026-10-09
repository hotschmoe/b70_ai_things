# Bounded original expert mirror qualification

CONFIG -> Exact selected UD-Q4_K_XL, Strata generation6, unchanged pinned
oneAPI/UMD image39992d70, default grouped expert kernels. Ten original experts
on layers0/2/11/32/47, expert0/17, three tokens each. Actual Q4_K/Q5_K gate/up
and Q5_1/Q8_0 down layouts. No GROUPED_V1 override. Partial fixture, not serving.

COMMAND -> Build stage-mirror oracle generations1/3/4, then parent-owned
run_stage_mirror_gpu_oracle.py and separate schema2 controller_v2.py.

RESULT -> Initial generation1 compilation fails: pinned toolchain uses
experimental/graph.hpp. Generation2 was prepared but never executed;
generation3 additionally uses actual DPCT void recording APIs and builds.
First GPU run fails closed on hidden packet equality. Source inspection proves
the default fused kernel never writes its reserved raw-hidden scratch region;
the oracle read its own0xA5 placeholder. No model kernel defect is established.
That failed run, traces, partial report and normal removal/post-health remain.

Generation4 preserves default model kernels, inputs and graph mode. A separate
fixture GPU observer reconstructs the same native-exp/F32 expression from actual
gate/up outputs. It does not claim to observe the fused kernel's original raw
hidden values. Independent CPU Q8_1 encoding of this reconstruction must equal
the actual fused hidden packets byte for byte. Input packets also match their
independent CPU representation. Gate/up and actual-packet down arithmetic use
independent original-weight dequantization and FP64 references; reconstruction
is separately labelled. Strict gates remain NMSE1e-6/normalized-Linf1e-4.

Card0/card1/pair all pass ten source experts, three mixed layout pairs and40
stage metrics per case. Worst NMSE1.03152073e-13/normalized-Linf4.85313131e-7.
GPU versus original-F32-activation expert output differs by at most0.01019814
relativeL1, within the separately declared upstream0.03 quantization gate.
This quantization difference is not implementation error or a widened HC gate.
Mirrored/resident outputs and repeated captured graphs are byte-identical.
Actual GPU-built plan pointers supply both byte-read and expert math kernels;
their source payloads match independently frozen original-GGUF offset hashes.

Each case has33,587,200 mirrored bytes,6+4 bounded segments,10 storage source
reads and20 mirror-host reads. Missing entries, foreign stage/binding, global
budget shortage, undersized segment and allocated partial-read failure reject.
Every graph still replays exactly after source close while its wrapper retains
the real mirror owner. Graph retirement precedes owner release. UR trace has
126 successful allocations/frees,12 registered segment/table owners, two native
context bridges and zero live logical allocations. Missing/duplicate/failed-free
negative controls reject. Normal process exit/removal and strict per-card plus
compiled P2P0 collective pre/post-health pass, with no reported fault signatures.
CPU report gates additionally reject18 malformed synthetic reports; those tests
are validator evidence, not GPU evidence.

VERDICT -> Bounded original expert source/tier/default-kernel arithmetic and
graph/owner lifetime pass on both cards. Full roughly48GiB mirror residency,
all expert coverage, full-model state/logits/coherence, two-card serving,
prefix reuse, concurrency, latency and shelf qualification remain required.

Raw evidence:

- /mnt/vm_8tb/b70/build/strata-stage-mirror-oracle-hchl9kz8/receipt.json
- /mnt/vm_8tb/b70/build/strata-stage-mirror-oracle-44i9k7zi/receipt.json
- /mnt/vm_8tb/b70/build/strata-stage-mirror-oracle-2u6_8_xg/receipt.json
- F06/stage-mirror-gpu-v1/receipt.json (failed, unchanged)
- F06/stage-mirror-gpu-v2/receipt.json, original source roster, per-card/pair
  reports, raw UR logs, independently parsed ledgers and health/fault logs

F06 is under /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/.
