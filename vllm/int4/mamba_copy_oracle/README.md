# Padded Mamba state-copy oracle, prepared only

CONFIG -> Exact R276 phase worker/model Mamba sources are SHA-pinned. The MRV1
count-fix image keeps these same source bytes. Qwen TP1 geometry: V heads48,
K heads16, dimensions128, conv_dim10240; convFP16 and temporalFP32.
MTP3 conv history6 uses122880 bytes; temporal48*128*128*4 uses3145728 bytes.
The natural3268608 bytes occupy padded3407872-byte pages (resolved832*4096).
MTP0 history3 uses61440 conv bytes and the same temporal state.

Allocator verification -> GPUModelRunner allocates raw pages; MambaBase slices
conv at offset0 and temporal at offsetconvbytes WITHIN EACH page. Both views
retain full-page stride0. Qwen preserves these aliases with set_. The oracle
constructs exactly that interleaving, including sentinel bytes after each state.

COMMAND -> oracle.py defaults to a CPU-only printed plan. No GPU flag was run.
A future owned lifecycle must hold bin/gpu-run --card N with matching device
pin, strict selected-card pre/post health, an8GiB container and bounded timeout.
Do not reset both GPUs under a single-card lease. Source/ABI/image identities
must be retained. `--run-xpu` is only the inner diagnostic workload, not a
standalone approved serving/lifecycle command.

Actual call paths -> MRV1 align+MTP creates the postprocess context and calls
MambaSpecDecodeGPUContext.run_fused_precopy, launching
precopy_mamba_align_fused_kernel with(reqs,states,16), COPY_BLOCK_SIZE1024 and
HAS_IDX_MAPPING=False. It calls _copy_mamba_state_block -> _memcpy_u64_tiled.
MTP0 has no speculative context: actual copy specs feed batch_memcpy, which
launches batch_memcpy_kernel. These are deployed Triton paths; no native SYCL
copy replacement appears in this exact chain. Default conv layout is SD; the
oracle rejects DS. Both kernel callable source files are verified at execution.
The metadata initializer is the actual installed method, including pointer,
element-size, stride0 and natural temporal element-count extraction.

RESULT -> CPU container without devices imported only Torch for CPU arrays,
then AST-executed exact metadata/copy-spec functions. All12 fixture geometries
and byte regions pass, including noncontiguous padded views and no source/dest
overlap. The independently specified byte ranges agree with actual copy specs.
The GPU kernels were NOT RUN. Cache agent independently reviewed the fixture
geometry, source/destination permutation, biases and source preservation.

The prepared GPU cases cover padded forward/backward copies, accepted-token
bias0/1/3 for MTP3, fresh/same-block no-ops, MTP0 copies and packed controls.
Each result compares the entire backing allocation byte-for-byte against an
independent CPU reference. This checks destination bytes, untouched conv tails,
padding and all unrelated/source blocks. Kernel failures stop execution;
completed numerical failures remain visible in per-case output.

VERDICT -> Concrete numeric copy diagnostic ready for a future owned GPU plan.
No copy bug has been demonstrated. Copy source already widens block IDs to
int64 and uses natural temporal bytes rather than padded stride as copy length.
Native GDN arithmetic accessing the same padded views remains separate.
This single-layer/group fixture does not verify multiple-group pointer mapping,
publication timing, accepted-row correctness, graph replay or model quality.

Raw: /mnt/vm_8tb/b70/results/bang_isolation_20260910/mamba-copy-oracle

## Owned per-card lifecycle prepared

lifecycle_plans.json records separate card0/card1 choices using the exact
7b107d phase+MRV1 image. No launch occurred. lifecycle.py defaults to printing
the chosen frozen plan. Its explicit --run requires the selected inherited
lease descriptor and the existing image pair-preflight PASS, verifies every
snapshot/helper hash and image ID, then runs strict selected-card pre-health,
the twelve-case oracle and strict selected-card post-health.

Each oracle uses8GiB/4CPU, a420s bound, explicit physical device pin, separate
fresh cache/output and read-only fixture/source snapshots. The common reviewed
preflight helper owns process-group termination and labeled Docker cleanup;
uncertain cleanup retains the lease until absence is verified. No reset occurs
under a single-card lease. Failed health defers recovery to the parent.
Full JSON is retained on numerical exit1; malformed/crashed runs still reach
post-health after cleanup. Health success cannot mask an oracle failure.

CPU lifecycle tests cover numeric failure retention, crash post-health and
failed post-health blocking success without reset. Both prepared unleased
commands fail before GPU work, and frozen identities/device/resource pins pass.
The exact launch argv are in lifecycle_plans.json. Select a card only after
parent scheduling; these plans have not qualified the kernels on hardware.
