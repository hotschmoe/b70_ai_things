# Current PLE host-input observer33 CPU source checkpoint

CONFIG -> exact corrected32 source plan01884ae3, NEW source33, source63/header27,
all eight fresh ABI targets, unchanged six Python runtime files. Existing
semantic32 gather/fences and inference/quantized bytes are unchanged.

COMMAND -> python3 -m unittest discover -s strata/flash-next -p
 'test_ple_input33_cpu_v1.py'

RESULT ->21 CPU tests PASS. Fresh pristine source paths plus all33 tracked
patches reconstruct with GNU patch, without git/Docker/runtime writes; all63
final hashes/27 headers match. Actual source-body order checks establish real
successful gather/fences before publish, publish before submit_begin, actual
ext_oneapi_graph before submit_returned, and actual GEN prefix_diag.finish before
terminal using still-current SFD.active/ordinal. Synthetic raw/metadata/log
fixtures reject wrong/stale/duplicate events, reordered SFD request/resume,
missing/early/wrong/cancelled terminal, graph submit rc failure, last truncation,
wrong prev/window/stage/epoch, changed raw SHA, symlinks and row byte/metadata
mismatch. Original checker preflights all four frames/raw fields before any
original row lookup; no-payload-on-metadata-failure controls pass.

VERDICT -> READY frozen bounded CPU source prototype, NOT runtime readiness.
No host C++ compiler is installed and Docker/SDK compilation is prohibited in
this session; actual C++ compilation/ASAN/execution of the new helper remains
UNOBSERVED. OFF/unarmed zero-read/effect behavior is source-body/branch evidence,
not a claimed native execution test. No GPU, original weight or actual runtime
capture was performed. Remaining native compilation and real observation gates
cannot be substituted by synthetic proofs.

STRATA_PLE_INPUT33 defaults OFF/unset/0. Enabled1 only publishes from the actual
source32 ple_precollected path AFTER gather_batch succeeds and existing fences,
BEFORE actual graph submission. EAGER is rejected for enabled observation.
The source requires armed SFD, activations, exact native PLE, reused0, ordinal1..4,
accepted prefix lengths1/2/4/8, model tokens<248320, stage0 coveringlayer1 and
current windowT1..8. The runtime pilot collector is deliberately T1/last-prompt
only. Unarmed warmups publish nothing. OFF adds no GPU operation or raw-input
read/hash/filesystem output. No borrowed embedding/row pointer is retained;
only small numeric epoch/terminal bookkeeping persists.

The publisher writes raw LE_F32[T,2560] h_ple and LE_U32[T,16] ple_rows using
openat/exclusive/O_NOFOLLOW files in a nonsymlink opened directory. Metadata
binds PID/stage/device/range/SFD ordinal/epoch/pos/T/full accepted GEN IDs,
actual window tokens/prev/row IDs/source binding/no_host/device_plan/AR and raw
SHA/size/encoding. Four frames max; each at most81920 embedding+512 row bytes.
Host publication delays launch and is not a clean latency measurement.

Producer lines establish SFD request <resume0 <published <submit_begin
<submit_returned(rc0) <actual matching GEN terminal. Terminal occurs after
existing prefix_diag.finish, not at submit return, and preserves SFD PID/ordinal.
Submission return does NOT prove GPU completion or input consumption. Actual
GPU copy/graph-handle association, PLE key/value/norm/history/conv math and full
model/operator equivalence remain UNOBSERVED. Parent native clean quit, owning
frees, health, source4/two-page identity and actual C1 proof remain separate gates.

The strict collector requires raw byte/hash metadata equality and original
accepted-prefix/previous-token alignment. The original comparator CLI always
RECOLLECTS directory+producer log+request roster/source binding, compares to the
supplied proof and preflights every frame before constructing the original
provider. Independent ngram hash computes expected rows with dedicated PLE
EOS248044, then original IQ4_NL row decode must match F32 embedding BITWISE.
That comparison reads only bounded original rows when root executes it later.
It does not claim the GPU consumed those exact host bytes or qualify PLE math.

Future actual workflow requires fresh33 SDK/oracles/full390/current-source and
C1 generation11 proof, not old32/C110 transfer. Baseline observer flags stayOFF.
The P30 V2 driver must recollect source33 proofs and run original row/embedding
comparison for BOTH matched P30 OFF/ON arms, retaining P30 whole-prefix coverage,
lastFFN/SFD192 bitwise pairs, full head/48 controls and128MiB owning-free gates.
C111/P30V2 orchestration remains the next source checkpoint, not implemented by
this bounded observer freeze. Example future comparator command:

```sh
python3 strata/flash-next/verify_ple_input33_original_v1.py --proof <collected-source33.json> --directory <actual-capture-directory> --requests <exact-accepted-roster.json> --producer-log <same-process-engine.combined.log> --binding-sha256 <actual-bound-plan-sha> --le <32-or-48> --model-identity <current-complete-source4-receipt> --output <NEW-original-input-proof.json>
```
