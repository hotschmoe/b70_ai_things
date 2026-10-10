# Source35 synthetic GPU HC projection arithmetic gate

CONFIG -> original public hc_q8_0_project_f32 / hc_f32_project_f32 from exact
source35 kernels/current archives; independently validated host FMA32lane+XOR
helper V2 plus supplemental finalized host runtime/library/source admission.
No full HC,exp,rsqrt,quantizer change or original model input is included.

COMMAND -> PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s
strata/flash-next -p 'test_hc_projection_arithmetic35_fixture_cpu_v1.py'.

RESULT -> 12 CPU tests PASS using synthetic analytic bytes and mocked root host
helper/admission. Actual tiny raw input/output/hash/manifest collection runs;
no C++ compile,actual helper,GPU,Docker,runtime/model payload/git work by agent.
Known answers distinguish XORorder1 versus0,lane cancellation,FMA -2^-46,
midpoint ties,signedQ8 codes,half-subnormal scale and F32subnormal behavior.
Four preregistered LCGseed3535 held-out cases useF32/Q8_0 K320/10240,M3,T2.
13 cases yield38 output floats; two wrong-order/weight negative cases differ
from their controls. Device subnormal mismatch is FAIL, never auto-relaxed.

VERDICT -> frozen source/CPU gate only. Root owns actual expected-data export,
fresh SYCL leaf compile and owned runtime parent. Source33review owns distinct
prepare_hc_projection35_leaf_v1.py,compile_hc_projection35_v1.py and
qualify_hc_projection35_v1.py plus its immutable compile plan. Frozen GDN parent
is a lifecycle template, not a launcher admitting HC receipts.

ROOT-only expected-data export after genuine supplemental host proof:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 strata/flash-next/hc_projection_arithmetic35_fixture_v1.py prepare --output <NEW-synthetic-package> --host-run-root <final-host-runtime-dir> --host-helper <fresh-pinned-host-ELF> --host-build-root <fresh-host-build-dir>
```

Public APIs: prepare(output,host_run,helper,build_root)->manifest;
input_binding(package)->strict current manifest; collect(package,raw,log)->raw
comparison. prepare calls the actual qualified host helper for each row after
finalized_binding admission of compile/analytic/finalV2 adapter/tinyraw/source/
ELF/libm/otherhostlibs/CPU/kernel/env. It checks that admission again afterward.
input_binding requires complete canonical corpus bytes and per-row helper/input/
output hashes and independently recollects the validated PythonF32 schedule.
Updating expected bytes plus a fabricated success hash does not establish parity.

Package/inputs has only cases.tsv and weight/input bytes; package/expected and
manifest remain outside the GPU mount. Leaf command is --inputs /inputs --output
/results/raw-new. The leaf calls only public HC projections, records actual
in-order queue/backend/device/vendor/driver/nativecontext/device handles and
selector+affinity0, emits complete T1/T2 raw output bytes and explicitly frees
owned buffers. A matching B70 name is not physical-card identity; parent must
bind card0/PCI mapping and inherited lease descriptors independently.

Collector compares raw complete bytes against bound host rows,not printedPASS.
It requires unique actual case/terminal/config markers and refuses missing,
duplicate,foreign,truncated/nonfinite outputs or synthetic/host source drift.
Subnormal or other mismatches stay numeric_bitwise_passed=false; tolerance isNone,
device_intrinsics/modelmath remainfalse. Raw collector alone always reports
actual_GPU_execution_qualified=false and parent lifecycle qualificationfalse.

The real parent must join fresh compile receipt/controller/ELF/source/current
archive/eightSDK binary identities, immutable package manifest, pair leases plus
physicalcard0 pin, strictpercard+compiledP2P0 pre/posthealth,journalfault gate,
owned named terminalexit/removal,actualUR2 allocation/free ledger and negative
controls,new postterminal/posthealth full4 with both known page guards/failure
preservation,then recheck sources/package/host proof. Record current compiler
and observable runtime library identities in fresh receipts. Current archive
hashes do not retroactively certify original SDK build provenance. No numeric
PASS string or fabricated receipt may substitute for those joins. Even full
component PASS is separate from fulloriginal math,cache/concurrency or speed.
