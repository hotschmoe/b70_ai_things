# Own-input source37 QSA arithmetic control

CONFIG -> This successor preserves the frozen V3 output tap and all HC V3
arithmetic. It adds a fresh original projection producer and a separate
standalone SYCL control. Native captures are comparison targets only. Inputs
are original-weight Q/K/V/indexer projections, own gamma and own zero history;
no captured normalized values, selected IDs, states or ULP adjustments enter.

COMMAND -> Targeted source/CPU checks:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_owned_layer3_qsa_control_cpu_v4 test_owned_layer3_qsa_projection_cpu_v3 test_owned_layer3_qsa_decode_candidate_cpu_v2 test_owned_layer3_qsa_contract_cpu_v1 -q
```

The complete inherited and new suite adds these four modules to the frozen
HC V3 command:

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest test_owned_layer3_qsa_control_cpu_v4 test_owned_layer3_qsa_projection_cpu_v3 test_owned_layer3_qsa_decode_candidate_cpu_v2 test_owned_layer3_qsa_contract_cpu_v1 test_owned_hc_device_rs_cpu_v3 test_owned_hc_device_rs_cpu_v2 test_owned_hc_device_rs_cpu_v1 test_native_rms_owned_device_ops_cpu_v8 test_native_rms_device_ops_cpu_v1 test_native_rms_entry_cpu_v7 test_native_rms_setvars_cpu_v6 test_native_rms_lifecycle_cpu_v5 test_native_rms_kernel_journal_cpu_v5 test_native_rms_receipt_binding_cpu_v4 test_native_rms_owned_timeout_cpu_v3 test_native_rms_publisher_cpu_v2 test_native_rms_rsqrt37_owned_cpu_v1 test_native_rms_rsqrt37_cpu_v1 -q
```

RESULT -> Source/CPU controls only. No new original payload, compile, device,
Docker, or model execution has been performed by the source author.

VERDICT -> Root and peer source review plus an isolated compile must precede
actual execution. A later passed helper means scoped function execution and
direct/replay coherence; full model arithmetic and quality remain unqualified.

## Deliberate two-phase execution

The root first runs `qualify_owned_layer3_qsa_producer_v4.py` with the same
current fixture, source37/C137 card0 baseline, closed V8, original identity,
NUM10 first targets, P30 prefix4 targets and qualified bulk helper used by the
closed HC V3 experiment. It also requires `--hc-prerequisite-root` naming that
exact closed HC V3 report. The qualifier retains all inherited health, journal,
source/publisher/page, fresh build, exact command/image/resource/mount, actual
EOF, timeout cleanup, receipt/inspection/library and model terminal gates.

```
python3 strata/flash-next/qualify_owned_layer3_qsa_producer_v4.py --fixture RMS_INPUTS --prepared C137_CARD0 --baseline-adjudication C137_ADJ --prior-rms-root CLOSED_V8 --hc-prerequisite-root CLOSED_HC_V3 --first-hc-native-root NUM10 --prefix4-native-root P30 --model-identity ORIGINAL_IDENTITY --bulk-build-root QUALIFIED_BULK --output NEW_PRODUCER
```

Phase0 computes only the first HC gate and has no QSA projection rows. Only its
exact full normalized/Q81 gate permits phase1, which computes fresh zero-state
prefix4. The inherited token guard rejects wrong IDs and stale producer reuse
before expensive math. Phase1 publishes the complete own projection records and
the inherited own QSA intermediate outputs. Those outputs are later comparison
targets, never helper operands. Root metadata confirms epsilon and resolved
unscaled RoPE; unsupported original scaling fails before helper preparation.

Once the producer is fully closed and its current public reader passes, root
prepares the binary own input fixture. This preparation recollects the genuine
producer before and after the exact raw transfer; it cannot borrow old captures.

```
python3 strata/flash-next/prepare_owned_layer3_qsa_inputs_v4.py --own-producer-root NEW_PRODUCER --output NEW_INPUTS
python3 strata/flash-next/qualify_owned_layer3_qsa_control_v4.py --fixture NEW_INPUTS --own-producer-root NEW_PRODUCER --prepared C137_CARD0 --baseline-adjudication C137_ADJ --output NEW_CONTROL
```

Both actual qualifiers enter through `bin/gpu-run`, retaining both leases for
compiled per-card/P2P0 health while the leaf is pinned to physical card0. Compile
uses the admitted compiler image; execution uses the normal C388 runtime image,
SYCL_CACHE_PERSISTENT=0 and UR tracing off. Neither helper has model/pack mounts.
The QSA standalone helper has no interactive stdin; the producer's unchanged
RS service retains its strict interactive protocol.

## Actual linked functions and state scope

`owned_layer3_qsa_gpu_v4.cpp` calls unchanged linked source37 production norm,
fused norm/RoPE, KV append, native indexer, resident decode and gate functions.
Five actual object compile blocks and defines must match the admitted precise
HC object profile; production device-link flags are separately identified.
The new leaf argv substitutes only its source/output paths. Build/source/header,
ELF and actual mapped runtime library identities are recollected.

The helper checks actual resolved RopeScaling fields, epsilon, native enabled
flags and fused capability before printing its config. Q reads its own Q|gate
rows with stride512; K uses stride256 and indexer query stride128. Windows are
the exact two-row/one-row/one-row source schedule. Selection is the own identity
roster for 1..4 cells; the unused future entries are masked by the real step
width. Zero KV/indexer state is restored outside each complete direct/replay
route. Actual step buffers, positions, inputs and allocations remain live
through graph retirement.

The compiled leaf also echoes the exact ten loaded own input files before any
operation. The reader joins their consumed bytes/SHA to the current genuine
fixture, rather than trusting only pre/post file hashes. The marker roster
includes actual device name/vendor/driver and FP capability flags; capabilities
are explicitly not compiler-lowering or intrinsic-accuracy observations.

Fourteen fields per route include separate weighted norm, fused Q/K/indexer
RoPE, resident attention and gating, and real KV/indexer snapshots at all three
window boundaries. Separate norm outputs are not internal fused witnesses.
Internal norm arguments and score/softmax remain explicitly unobserved. The
existing uncontracted CPU score is one declared candidate, not a compiler
lowering claim. No alias/cache or whole-model graph qualification transfers.

## Read-only admission

The current input fixture must equal the full strictly recollected own producer
roster. Every output is confined, regular, correctly sized and finite. Consumed
bytes are directly SHA-bound and reread/stat checked. All three full routes must
be bitwise equal. Full own inherited Q/K/indexer RoPE, gate and final indexer
outputs are compared without tolerance or a pass cutoff; differences remain
descriptive. The owned semantic marker roster must be unique, ordered and newline
complete, with actual graph/free terminal and no native error marker.

Both public readers bind original typed receipts and inspections, exact complete
artifact membership before and after comparison, current source/input/prior
proofs, actual EOF and stop-before-drain, exact health/journal commands plus raw
fault text, compiled/container/CLI chronology, and ordered publisher-four/page
brackets. The producer's final model computation terminal follows all own
computation and precedes its full-four scan. No failed old proof is rewritten.
