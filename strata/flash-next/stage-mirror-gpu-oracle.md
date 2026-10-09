# Bounded actual segmented mirror and expert arithmetic oracle

CONFIG -> Frozen v6 engine libraries, original selected four GGUF shards,
compatibility pack's original native_experts metadata, patch0012 actual
GgufExpertSource/StageExpertMirror, and parent-owned card0/card1/pair execution.
The fixture selects experts0/17 in layers0/2/11/32/47, covering actual Q4_K/Q5_K
 gate/up with Q5_1/Q8_0 down (integer types12/13 and7/8). It uses the source-default
grouped mode, not a changed kernel mode. No expert tensor conversion is added.

COMMAND -> Build with build_stage_mirror_gpu_oracle.py --engine-receipt V6_RECEIPT.
The builder snapshots oracle.cpp, stage_mirror_q8_reference.hpp and
plan.snapshot.json, links actual complete v6 libraries, and records library,
source/reference/plan/binary hashes. Compilation exposes no GPU devices.
The parent controller independently freezes original expert SHA/offset/type
identities from the pinned inventory before any GPU run. It owns the lease,
health, bounded launch, removal and post-health.

RESULT -> The fixture mirrors10 actual expert blobs (33587200 bytes) into two
actual disjoint owners with4MiB segment limits:6 segments for layers0..31 and4
for32..47. Each source owner also has its device pointer table. The unchanged
actual source constructor allocates its512 bounded staging buffers (about1.91GiB).
The parent uses an8GiB container cap for those buffers, mirror images and bounded
dequant/reference/GPU/readback/source-cache buffers; no full expert bank/model is
loaded.

Original blobs are independently gathered from all four GGUF tensor directories,
checking names, shapes, dtypes and expert offsets. Their bytes/SHA and source FNV
must match actual mirror gathering. The actual resident_plan_bound kernel reads
the device address table; a byte consumer dereferences the pointer produced by
that real planner and compares every original expert byte/SHA. Its destination
and every input/workspace allocation have128-byte guards. Source blobs are
256-byte aligned already, so this subset has no source padding bytes to invent;
all segment requested bytes equal the exact original padded extents.

The same actual GPU-produced pointer feeds native_expert_grouped for3 token
entries. That output must be byte-identical to the same kernel reading an
unchanged resident device copy. Captured graph replays must repeat those bytes,
and every graph must replay successfully after GgufExpertSource closes and drops
its references while the graph wrapper retains the actual stage owner. This
proves the bounded table/owner path, not complete model graph integration.

Arithmetic checks separate declared quantization from implementation error:

- Original-weight ggml dequantizers plus FP64 accumulation over originalF32
  activations produce the source float SwiGLU reference. This follows upstream
  native_expert_parity's float contract, including its F32 hidden materialization.
- The real CPU native quantizer/vec-dot path supplies a separate native CPU result.
  Its vec-dot activation format is declared by NativeFmt and may differ from the
  GPU Q8_1 format; this is reported as a quantization difference.
- An independent host Q8_1 encoder replicates block32 F32amax/127,
  round-away/clamped signed codes, FP16 nearest-even scale and XOR-reduced original
  F32sum. Actual GPU input and hidden packets must match every encoded byte.
- Gate/up reference dots use original dequantized weights and that independently
  encoded input. SiLU follows those independent gate/up references. Down is
  checked against an independent FP64 dot over verified actual hidden packets,
  isolating the down implementation from upstream hidden rounding. The selected
  Q5_1 path uses the quantized-code min sum; Q5_0/type6's original-sum correction
  is implemented only as an optional helper control, not misidentified as a
  selected tensor format.

Each of40 gate/up/SiLU/down implementation stage metrics must be finite with
NMSE<=1e-6 and normalized maximum error<=1e-4, normalized by
max(1e-6,max_abs_reference). This gate catches shared resident/mirror scale or
index bugs that their byte equality alone cannot detect. Separate relativeL1
metrics GPU-versus-float, nativeCPU-versus-float and directGPU-versus-nativeCPU
must be<=3e-2, preregistered from upstream native_expert_parity before execution.
The wider quantization comparisons do not replace the strict implementation
gates and do not change any HC/PLE tolerance.

Negative controls exercise actual missing-entry resident plans (empty/error,
no fallback), foreign-stage lookup and wrong residency/context bind, complete
budget shortage and too-small segments. A private isolated source directory with
one empty role file forces an actual read failure after bounded host segments
allocate; no owner, aliases or byte counters may publish. No invalid source or
foreign pointer is submitted to a GPU math kernel.

Counters must record10 original storage gathers and20 CPU mirror-host API reads,
with33587200 bytes in the single global mirror budget. These counters explicitly
refer to source API calls; GPU pointer-table arithmetic consumes those images
separately. All selected pointers must exist and every foreign/unselected device
 table entry must be zero. Legacy device_alias presence hints cannot substitute
for the explicit per-stage table's selected pointer.

All positive host segments and device tables have owner/context/extent markers.
Chronological UR traces must retire every registered allocation within its
matching destruction interval and globally balance every allocation/free,
including partial-read cleanup. The parent rechecks missing/double/failed-free
negative controls, terminal exit, container removal and strict pre/post-health.
No postfree pointer-type observation is an acceptance criterion and no physical
backing release is inferred from logical frees alone.

VERDICT -> Independent Q8 reference CPU tests pass all63488 finite FP16 values,
known source packets, finite sum/clamp, selected Q5_1/Q8_0 dot behavior, optional
Q5_0 correction and numeric/extent rejection controls. The frozen fixture/plan
and builder are ready for parent compilation and leased GPU qualification.
This remains a bounded partial tier/kernel fixture. Full390 HC/PLE source
qualification, every model expert's VRAM-or-owning-stage-mirror placement,
required48GiB backing, full one-/two-card model/history/state/concurrency and
healthy teardown remain separate mandatory serving gates.

Immutable compile generations: v1 source compilation failed on the unavailable
experimental/command_graph.hpp include and its receipt is preserved at
/mnt/vm_8tb/b70/build/strata-stage-mirror-oracle-hchl9kz8/receipt.json. Prepared v2
corrected the include but was not executed. v3 additionally matches the frozen
DPCT graph wrapper's void begin/end APIs; it checks the returned graph pointer
before finalize. Both earlier sources/plans remain unchanged. Use
build_stage_mirror_gpu_oracle_v2.py with --plan stage-mirror-gpu-oracle-plan-v3.json.
Actual v3 source compilation/linking passed at
/mnt/vm_8tb/b70/build/strata-stage-mirror-oracle-44i9k7zi/receipt.json.
Parent GPU execution is in progress; no numerical/lifecycle result is asserted
until its complete terminal receipt and post-health establish it.

Default-fused observability correction: the actual v3 GPU run failed its hidden
packet check because the oracle read the unwritten hidden scratch at2fa. The
source-default swiglu_q8_1_entries_kernel writes HQ directly and never stores raw
hiddenF32 there. This was an oracle error; the failed GPU receipt/report/trace
remain preserved. Parent per-card and compiled P2P0 post-health passed without a
fault signature. No numerical pass is inferred from that run.

V4 keeps the production default fused path and original inputs/kernel/mode. A
separate fixture GPU observer evaluates the audited contractoff expression
(g / (1 + sycl::native::exp(-g))) * up from the actually observed gate/up, writing
an explicitly derived buffer. Independent CPU Q8_1 encoding of that buffer must
match the actual fused HQ bytes. Original raw fused hiddenF32 remains UNOBSERVED;
the report declares raw_fused_hidden_observed=false and
hidden_reconstruction_observed=true. The stage is silu_reconstruction, not a
claimed raw-hidden measurement. Gate/up, reconstruction and down-from-actual-HQ
retain NMSE1e-6 and normalized maximum error1e-4. CPU std::exp is not substituted
for the hardware native-exp intrinsic in a byte equality claim. On packet or
numeric mismatch, bounded diagnostics save observed gate/up, reconstructedH,
actual/expected input/HQ packets and a metadata scope statement. Schema2 report
and v4 plan distinguish these observations; all earlier generations are unchanged.
