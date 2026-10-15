# Strata attention comparison binding audit (S91, 2026-10-15)

CONFIG -> The current S90 trace identifies substantial full32-query INT8
attention work. S91 prepared a private score-only 12-head reduce-scatter
against the byte-exact current public attention object. The serving source,
archive and frozen S89 executable were unchanged. This was a CPU build and
offline audit, with no GPU execution or new performance measurement.

COMMAND -> One CPU-leased, no-DRI build in the pinned e41 image took32s:
public-attention prelink2.279s, private compile8.573s, fixture compile7.045s,
private prelink2.432s and ordinary host link0.194s. The public object was
extracted from the current archive and compared before/after. Only two
selected M1/NC1 device images were decoded offline, taking10s. Host symbols,
relocations and linked kernel-info state were inspected separately.

RESULT -> The executable SHA256 is
`51e2fd7b58a41493893facd947c1c08c368eb3ddb80145eac1ab41e895d6b41a`.
Both complete device bundles appear once in it. Their selected device IDs
differ in `qsa_decode_attn_batch` versus `reduce12_batch`, and their image
and instruction bytes differ. That alone does not prove separate dispatch.

Both host objects define the SAME weak
`CompileTimeKernelInfo<dpct_kernel_name<attn_chunk_kernel_2ad0a4,scalar1>>`
and the same `getDeviceKernelInfo` cached object and guard. Both host call
paths refer to those symbols. The linked executable retains one instance,
although the two object initializers name different device kernels. This
conflation defeats the intended independent public-versus-private comparison.
Renaming exported functions was insufficient; host metadata must also be
separate. No GPU test was run on this executable.

The selected images also show an unfavorable resource change:

| Static resource | Public | Private reduce12 |
| --- | ---: | ---: |
| SIMD width | 32 | 32 |
| GRFs | 128 | 128 |
| SLM bytes | 15872 | 15872 |
| Barrier count | 1 | 1 |
| Private size | Field absent | 3584 |
| Declared single-space scratch buffer | Field absent | 4096 bytes |
| Static SS/a32/d32 store/load sites | 0/0 of this pattern | 28/28 |

Missing public metadata is not proof of zero runtime scratch or spills.
Each listed private SIMD32 message has128 payload bytes if every lane is
active. Summing each static site once gives3584bytes in each direction;
this is not an execution count, physical traffic, occupancy or measured
latency. Fewer source shuffle expressions did not guarantee cheaper code.

A separate offline check found the same host-binding problem in the retained
S13 value-pipeline probe. Its executable SHA256
`7411d17bc61b49dcb516605cab4f7cde1aaaeff34892c2396d4364fad45bc4b2`
matches the hash recorded in the historical S13 document. The two final
host initializers both use kernel-info address0x42d598, cached object0x5d4810
and guard0x5d4818. The kernel-info name pointer selects the PUBLIC batch
kernel. This directly audits the retained historical binary, rather than
inferring its behavior from S91.

The old S13 logs, sources and journal entry are preserved. Their exactness
and near-equal timings do not establish that the distinct value candidate
was exercised. The current conclusion is candidate attribution unsupported,
not a new correctness result or a measured speed regression. Baseline and
health observations are not invalidated by this finding.

VERDICT -> Park the S91 reduce12 mapping. Do not rebuild the same mapping
merely to fix its names, and do not execute its prepared GPU runner. No
runtime, serving, stability, target or promotion claim follows from this
stage. Both GPUs remain available; no health run is needed for CPU-only work.

The next preparation revisits S13's value pipeline with all11 candidate
attention chunk/merge class tags made private, as well as private host
function names. This new binding evidence justifies revisiting its earlier
neutral classification. Reuse the current public object and its completed
AOT wrapper; compile only the private candidate, fixture and wrapper. Before
any GPU test, inspect actual separate host kernel-info state, its registered
names and the corresponding device images. Source names alone are not proof.
The strict36-case full-output/scratch, changed-input, repeat, memory-guard and
readonly-input checks remain required; the prepared fixture has no new
tolerance or case sweep. S92 preparation is not a corrected runtime result.

The development-loop improvement is concrete: reject an invalid comparison
before model loading or a GPU measurement, and reuse completed public AOT
work. The current serving candidate and S30 RAM-recovery evidence remain
available; the2000prefill/150per-stream targets remain open.

Fresh independent consultations responded as `claude-opus-5-5` in56.221s
(session737f6022-198e-4992-bcfc-7e5646641e06) and `claude-fable-5-1`
in138.816s (session644a2f98-f7b1-4002-8c26-fb43f5abf751). Both accept
the S91 hold and approve the S92 CPU build, with a separate GPU decision
pending actual host/device binding and resources. Root read both full
answers and independently reconstructed the complete candidate from its
baseline and patch. S92 remains unbuilt at this checkpoint.

Root resolves overbroad review conditions: unmodified common helper IDs may
legitimately overlap, while the attention metadata must be separate. Check
all materialized attention tags; do not force unused emissions. Link warnings
alone are insufficient. The fixture has sentinel regions, not protected
guard pages; its merge is exercised as part of the full pipeline. Source
allocation identity does not require identical compiler SLM metadata. The
existing FP64 baseline sanity threshold stays separate from strict cross-arm
byte equality. No extra grep-only source edit or full public rebuild is
needed before the already required actual binding audit.

Evidence root:
`/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014`.
S91 build/audit: `attn-score-reduce12-s91-prep/build` and `build-audit`.
Historical audit: `attn-score-reduce12-s91-historical-binding-audit`, including
the documentary binary-hash supplement. Large binaries, disassemblies and
historical raw outputs remain external. Small exact records are retained in
`strata/flash-next/lowlevel_probe/attention_binding_s91/`.
