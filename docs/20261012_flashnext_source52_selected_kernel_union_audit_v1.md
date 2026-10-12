# Source52 selected armA named kernel unions (UNADMITTED)

CONFIG -> Actual source52 bounded P2048 warmup/selected GEN8 recorder, copy OFF, both cards, H36 OFF. Existing observed receipt f63275fc covers selected sequence(20,583306] and581094 exact Clock/Chrome instance joins. No per-arm/experiment public admission is borrowed. Original traces remain immutable.

COMMAND -> CPU streaming original inspector with one output-only aggregate over its exact selected instance dictionary. No parser/bounds/raw hash/device/query predicate is changed. New helper analyze_source52_selected_kernel_unions_v1.py; saved F17/source52-v5-armA-kernel-unions-unadmitted-v1.json SHA961d90ce573c2f133abff3b0561f3548c065f01dcbe2c4dbd1ee777174dee0ed (30177 bytes). Original reduction equals prior receipt exactly; raw before/after hashes retained. Two small synthetic union/family controls PASS; no GPU/build/model/provider reads.

RESULT -> Same-device raw domains separately:

| Named family | Device0 union ms / count | Device1 union ms / count |
|---|---:|---:|
| HC native symbols |4555.151 /92814|4271.693 /88744|
| Dequantization symbols |1434.361 /57670|1533.389 /61678|
| Vendor/named GEMM |436.574 /58054|472.473 /62062|
| GDN symbols |71.517 /864|70.716 /864|
| QSA/attention/indexer |145.940 /1422|145.116 /1422|
| Explicit DMA copy |1613.432 /9150|1324.293 /5829|

Family unions are subsets, not additive exclusive latency accounting. HcProjectionKernel<true> dominates named kernels:3601.200ms/29454 operations on device0 and3320.964ms/25360 on device1. Generic GEMM names do not prove whether a call serves an expert, dense projection, or particular layer. Launch counts do not prove token count or workgroup data reuse.

All observed operation unions8.442s/7.974s within raw hulls33.084s/33.118s. Time uncovered by these observations24.641s/25.144s is NOT physical idle; other-card serialized stages, unobserved operations and profiler effects can occupy it. Per-device gap median4.844/6.979us, P9546.354/48.229us, P9968.750/86.042us. Largest device0 gap4.658s follows D2H20930560B and precedes copy_i32_from_mapped; device1 gap4.401s follows a memory fill and precedes H2D20971520B. Context names/raw bounds are retained, not causally labeled host delay.

VERDICT -> HC scalar projection and dequantization are tangible optimization leads in this instrumented selected slice. No physical utilization/idle/bandwidth or mapped-RAM traffic inference; explicit H2D payload retains its original narrow meaning. No temporal prefill/decode split is justified: selected source frame brackets the whole request, H36 is disabled and no calibrated phase boundaries are saved. Selected CPU-command gap metrics are null: cpu_op rows lack selected instance-ID joins, and guessed host offsets are refused. Large flow-category dictionaries and per-instance terminal output are avoided. Native/API/speed/fullmath qualification remains absent.
