# S74: BMG INT8 matrix compilation and static ISA

The pinned BMG-G31 compiler path accepts the signed INT8 matrix expression
and emits one signed INT8-to-INT32 DPAS instruction. No binary was executed;
runtime capability, integer outputs and performance remain unmeasured.

CONFIG -> Intel2026.1.0, immutable image
`sha256:e41d59c0940e1bd2f6f64c9d5d2d7452b77812b0a8407fac8d67cacec5d7fdd2`.
One SG16 matrix operation: A8x32 row-major signed INT8, B32x16 column-major
signed INT8, C8x16 INT32. Six token rows are active and two zero, with all128
cells stored. There is no IQ decode, Q8 quantizer, local scale or FP32 epilogue.

COMMAND -> Root read complete source/build scripts. Actual fresh Opus5.5 and
Fable5.1 source reviews both allowed the bounded offline build. One cpu-run
build, no DRI/network, UID1000 and disk temporary files, normal SYCL BMG AOT.
Source/script and six installed matrix-header pins pass before and after.
The build never executes the binary, including its no-argument path.

RESULT -> Compile exit0 in an18-second CPU lease. The root wrapper wall
47.786s includes waiting for S73's CPU lease; it is not compile cost.
Executable82648 bytes, SHA256
`37728fb9a58097420c43f2b383574a209c154ce4d1420a90c4182326795eb9ff`.
Source SHA256
`fcb7e5f8158b59795aa2c74907257435dcf28005b0d1c20b9040d241e4d68600`.

Native audit author `/root/mtp_q4_share` extracted the single existing image
and ran offline ocloc under a10-second CPU lease, no DRI/network or compiler.
Root read the full findings/report. Kernel `_ZTS17Int8K32Capability` has:

| Static artifact | Observed |
| --- | --- |
| Main text | 896 bytes |
| Requested subgroup / SIMD | 16 /16 |
| GRFs | 128 |
| has_dpas | true |
| Main DPAS instructions | 1 |
| Main UGM sends | payload load, two data loads, one data store |

The actual instruction is
`dpas.8x8 (16|M0) r18:d null:d r9:b r5.0:b`.
Both inputs are signed bytes and the destination is INT32. Data loads use
block2d forms and the output uses a block2d store. These are static instructions,
not executed counts or physical-traffic measurements. Post-EOT text is excluded
from counts. Missing explicit scratch/SLM fields do not prove runtime zero;
resources do not establish occupancy or a speed bottleneck.

Selected zebin5016 bytes, SHA256
`503399a741cdbe20f2b873b192198f5bb4e8b75bd68c892df56ae4fe2dc2fea0`;
main text SHA256
`3531f2f0c4d22391ad8e5857e72c0e44253b4a73d1c937710916f1e8103ce644`.

VERDICT -> Compiled/static prerequisite PASS, runtime untested. S71's IQ3_S
per32 scale algebra and this AOT result justify source preparation for a new
canonical-Q8 projection. They do not establish the local staging, coordinate
ownership, native FP32 partial/XOR tree, public output equality or speed of
that future kernel. Existing slower FP16 grouped-XMX and packed-Q6 paths remain
separate. No full-model fork, serving change or performance forecast follows.

Actual post-result Opus30.268s/Fable35.469s accept this narrow claim and S75
source preparation, with no runtime endorsement. Root read both full responses.
The shared S73 reviews retain exact actual model/session IDs in their records.
The optional source `--run` contains an INT64 reference and input-readonly
check, but neither the128 outputs nor that runtime path has been tested here.

Raw root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/`.
Preparation/build: `int8-xmx-capability-s74-prep/`;
root command/completion/log: `int8-xmx-capability-s74-root/`;
selected image, full ISA, command and report: `int8-xmx-capability-s74-isa/`.
Exact [recorded source/recipes](../strata/flash-next/lowlevel_probe/int8_xmx_capability_s74/README.md).

Original source-preparation manifest SHA256
`af5b5ea74ab2329bb9ae102e7cb7671e482e4d4f5d8599560b2cebc1b110fd68`.
Separate offline ISA manifest23 files/28321 bytes, SHA256
`a2a0d6479e96f78b710d11264a36b78923b1173bb469c3bf6e93f01431e69e97`.
Completed build manifest: `int8-xmx-capability-s74-completed-manifest.json`,
13 files/100341 bytes, SHA256
`ae422e7bb0cb171c38dd1fb51b5e51715913f0572244e6e3442743f3a5da43a0`.
The original external `int8-xmx-capability-s74-manifest.json` is the frozen
source-only subset and remains unchanged; it is not a completed-build manifest.

Shared actual source consultation manifest:
`hc-more-groups-source-reviews-s73-manifest.json`, SHA256
`0d710aa9e2b7cf029f5a47bb4ca7f9094be6eef693b618768a3f0ef60521229c`.
Shared post consultation manifest:
`hc-more-groups-postreviews-s73-manifest.json`, SHA256
`55547052d4e4979690d3eff1d721359191937e88bfb34fa024b130fd09870129`.
Exact actual model/session IDs also appear in the S73 dated report.

Dev-loop decision: stop at the compiled/static prerequisite. A separate
runtime-only capability lease would duplicate setup before the full projection
prototype, whose strict numerical gates must pass regardless. No standalone
S74 runtime job is queued.
