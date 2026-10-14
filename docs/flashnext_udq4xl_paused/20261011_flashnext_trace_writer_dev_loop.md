# Trace writer CPU development-loop candidate

CONFIG -> Source/CPU-only audit while root owns the live full-cache V9 GPU run.
Frozen producer `buffered_native_trace_epoch_v5.py` inherits two ASCII text
handles from `buffered_api_trace_sink_v3.py`. No live source/file changes,
GPU touches, model payload reads, deadline changes or event filtering.

COMMAND -> Read both sources; bounded live sample reads total5MiB (trace first
2MiB, combined first1MiB, trace tail2MiB). Copy only the tail sample to a CPU
temporary file. Run an isolated TextIO buffering probe, four ABBA sink replays
with10833 records each, a separate3611-record cProfile replay, and eight CPU
equivalence/failure tests. Benchmark script and source plan are below.

RESULT -> Frozen V5 calls TextIOWrapper.tell about five times per received
record. TextIO tell flushes its underlying write buffer. A synthetic raw writer
observed zero writes after text.write, then one write immediately after tell.
Thus the periodic0.1s flush does not effectively batch these UR records.
The fixed-point end-offset loop also dumps the complete JSON record twice.

Live sample locations:
`/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/shared-v9-whole-suite-postboot-prepared-v1/actors/shared-capture0/child/`

Initial stat observed trace4,911,827,762 bytes and combined1,548,769,051 bytes.
Tail read observed trace4,944,043,510 bytes;3611 complete records, sequence
6905732 through6909342. Native_receive3603, native_send2, plus six semantic
engine/waiter records. File growth is observation, not a terminal verdict.

| CPU replay | V5 old | V6 candidate |
| --- | ---: | ---: |
| First ABBA wall seconds | 0.36711 | 0.23870 |
| Second ABBA wall seconds | 0.36634 | 0.23924 |
| Mean wall seconds | 0.36672 | 0.23897 |
| Event rate range per second | 29509-29571 | 45280-45383 |

Candidate component wall reduction34.84 percent; event-rate ratio1.535.
This is copied-sample CPU sink replay, not inference or real request speed.
Python3.14 host execution is distinct from the pinned live Python runtime.

Separate profile3611 records:18,039 text tell calls used29.38ms;7227 JSON dumps
used77.11ms; SHA update plus hexdigest used5.41ms; periodic/semantic explicit
flush methods about0.04ms. These diagnostic timings include profiler effects
and should not be added to the clean ABBA timings. Serialization and unintended
flushes are stronger candidates than removing identity-prefix SHA checks.

Candidate `strata/flash-next/buffered_native_trace_epoch_v6.py` uses binary
owned buffers and exact trace/combined byte counters. It encodes the two sides
of the sorted top-level JSON end-offset field once, then solves only decimal
offset width. Nested fields with the same name remain untouched. All event
fields, exact combined lines, sequence/time/process/file identities, prefix
SHA, combined offsets, phase marker records and ACK anchors remain. Semantic
flushes and periodic flush interval remain; total text quota and exact first
rejected record/source-line preservation remain. Short writes fail closed.

Eight CPU tests pass: complete V5/V6 byte equality including Unicode, nested
end-offset names and decimal boundaries; prefix/marker/anchor seals; identical
quota-overflow accepted bytes and rejected evidence; partial trace/marker short-write failures with retained rejected bytes;
flush failure and bounded periodic visibility; and
the text-tell flush probe. None establishes an actual producer epoch or GPU
request qualification. The candidate is not imported by the live V9 observer.

Artifacts:

- `strata/flash-next/test_buffered_native_trace_epoch_v6_cpu.py`
- `strata/flash-next/benchmark_buffered_native_trace_epoch_v6_cpu.py`
- `strata/flash-next/buffered-native-trace-epoch-v6-source-plan.json`
- CPU report `/tmp/flashnext-trace-v6-CPU-benchmark-20261011-v2/report.json`
- Copied sample `/tmp/flashnext-trace-CPU-sample-5s806dx6/sample.json`

VERDICT -> Keep V9 live/frozen evidence intact. Independently review the new
writer and repeat the copied-sample benchmark in the pinned CPU image first.
Any future integration needs a new observer/source plan and complete epoch,
namespace, phase ACK/anchor, prefix-seal, visibility, quota, raw observation,
output equivalence and healthy teardown qualification. No filtering, silent
dropping, longer deadline or inference speed claim is justified by this test.

Frozen source-plan SHA256: `a6467fad07ae019cb9c56a75c90c18d5a89311dec525ba14c80300ae55a45468`.
Final CPU report SHA256: `387faf96e2d5ae6a822ea4d767cc0b6e573c22c504a449f9edc1807283232d90`.
Earlier draft CPU report remains preserved at its original temporary path.
