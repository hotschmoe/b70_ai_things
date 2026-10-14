# Current52/56 expert placement: source-only findings

CONFIG -> Shipped full24576 STRP ranking8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf,
static adapt-swaps0/current native device-plan/two24-layer stages/original UD weights.
No assumption that this ranking is poor. make_profile.py's94.9 versus69.2 example
is historical IQ3_S/four-way/5805-pair evidence, not current B70 UD coverage.

Source findings:

- tools/make_profile.py appends base before traces by default. The shipped base
  already has every24576 pair, so adding a trace without --reorder/--no-base
  leaves the ranking unchanged. Prefer --reorder to retain shipped fallback.
- Native paired decoder drive_pool_split forwards global layer/IDs to
  drive_pool_multi, whose dump format is layer,k,IDs,zero weight placeholders
  (8+8*k bytes). Count frequencies only; these are not actual router weights.
- Current normal GEN has STRATA_VERIFY_NO_HOST=1. Verifier::run skips its CPU
  pool-service loop even for a device-plan/mirror route; actual fairness GEN
  logs show zero host/cache counters. --dump-routing is written only by Drive
  pool callbacks. It therefore cannot promise complete production GEN routes.
  Current BGEN private-slot run_slot_rows services pool for non-all-resident
  stages; its mapped doorbell copies all IDs even when resident, so the dump can
  observe all selections of that callback path. Whole all-resident stages can
  still bypass host service; require observed record completeness, not inference.
- The serving path exits with std::_Exit(0), flushing stdout/stderr only. Its
  routing FILE is not closed/flushed there; the fclose at12978 is after serving's
  early exit and belongs to one-shot mode. A normal serve dump can lose the final
  stdio buffer. Do not launch a known-incomplete dump as authoritative capture.
- One-shot mode closes routing but its no-MTP window currently initializes
  T=S_mtp (reserved spec2) whereas serving GEN uses T1. Its extra proposed rows
  are not clean production decode frequencies. Avoid substituting that trace.
- Existing saved route-named arrays found are layer0/short numerical/CPU MTP
  diagnostic artifacts. No complete admitted current52/56 representative routing
  trace was found in currentF17. These are not calibration frequency samples.

Small direct capture plan (requires a reviewed new source generation;57 frozen
boundary candidate must not be changed): use one active BGEN at a time on native2,
mirror0/default existing owned52-compatible config. Add checked fflush/fclose of
routing before normal serving _Exit. Preregister an uncaptured GEN1 admission
followed by ordinary private-slot decode; capture only its N-1 later positions.
For each request require exact48*(generated-1) records, layers0..47 per position,
canonical k10/unique IDs0..511, complete88-byte records and exact file census/SHA.
No MTP/pipeline/concurrent owners. This avoids forcing the production GEN no-host
path to a different host-service graph just to collect data. Full route capture
for production GEN requires a separate small owned device-side observation port;
--dump-routing alone does not supply it.

Calibration must use new preregistered representative prompts, not existing A/C/D
speed cases. Freeze disjoint calibration/heldout text+token hashes and grouping
before capture. Example minimum:6 calibration prompts across code/operations/
document work;3 distinct heldouts, one per family. Root selects authentic inputs
and executes tokenizer/capture. Build NEW full profile only from calibration:

  python /ACTUAL_SDK/source/tools/make_profile.py CALIBRATION_TRACE_FILES --base /ACTUAL_SDK/source/data/expert-profile.bin --reorder --out NEW_PROFILE.bin

Verify complete24576 permutation/reverse table and unchanged original weights.
Do not use the tool's default output, which overwrites the shipped source file.
Heldout ranking inputs stay absent from calibration. Replay heldouts against
shipped/new profiles in matched fresh owned processes, identical byte-budget/
profile-selection flags, source/math/context/input/output/EOS and full guards.
Demand coherent complete histories; report heldout TTFT/token gaps/decode and
actual per-stage resident allocation. Only profile is varied; warm-up/temperature
limitations stay visible. No claim from a rank-derived frequency curve.

Miss/byte accounting:

- multi_misses is distinct CPU jobs, not all VRAM misses. At pcie-frac1 it can be0
  while the GPU streams many misses. cache_hits/(hits+admitted+refused) excludes
  offload_entries; a reported near100% value may omit PCIe work.
- pcie_experts counts distinct fetched/offloaded experts per layer-window, not
  total selected occurrences. Repeated selections must be deduplicated within a
  window. Actual static resident membership can be taken from owned FC38 expert
  inventory/start-end maps; adapt0/no-borrow/unchanged map must be proved.
- A complete selected-ID trace plus actual map yields actual selected misses,
  per-stage frequency coverage and distinct miss groups. Bind each group's
  actual native lay.blob_bytes(layer) from a small source-owned metadata receipt
  or exact original tensor-header layout, not a generic bytes/expert constant.
  Sum distinct misses*blob bytes as logical miss payload. It is not measured
  bus traffic or bandwidth: direct/mapped/kernel-read versus DMA mode matters.
- Actual DMA bytes require the existing fetches/bytes callback route where
  pcie_mode0, or admitted same-device PTI MemoryCopy byte events. In auto/mode2,
  fetch_blobs kernels read mapped blobs; counting copy events misses this traffic.
  Do not call logical payload measured PCIe bytes.

RESULT -> Source/metadata audit only. No GPU, compiler, payload, model process,
profile mutation or runtime capture. Findings identify prerequisites and honest
measurement definitions. No profile quality or speed claim.

VERDICT -> Ranking is a viable weight-preserving control only after complete
source-compatible calibration capture and disjoint heldout measurements. First
small source fix is serving routing close; complete ordinary GEN capture remains
an explicit separate observation task. No blind profile replacement.
