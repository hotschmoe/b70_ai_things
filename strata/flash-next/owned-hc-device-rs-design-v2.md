CONFIG -> NEW V2 strict reader/chronology successor to frozen V1/7cc5.
All V1 source and original references remain unchanged. Own argument generation,
actual sycl::rsqrt expression, metadata shim, source/flags/libraries, first exact
normalized/Q81 gate and optional zero-state prefix4 computation are unchanged.

COMMAND -> Root-only qualify_owned_hc_device_rs_v2.py with the exact V1 arguments.
Actual compilation/GPU/model payload execution is unobserved by this agent.
After peer review, root can generate the isolated compile recipe with:

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 - <<'PY'
import os,shlex
from pathlib import Path
import qualify_owned_hc_device_rs_v2 as q
out=Path('/mnt/vm_8tb/b70/build/owned-hc-device-rs-service-v1-isolated-20261010')
fixture=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/native-rms-rsqrt37-preregistered-inputs-v1')
argv,compile,runtime=q.expected_recipes(out,fixture,os.getpid())
print(shlex.join(compile))
PY

Root creates out/build and runs the printed owned Docker compile command under
the applicable GPU lease. It selects /leaf/owned_hc_rsqrt37_service_v1.cpp only;
all SDK37 compiler/object/device-link flags and source/archives stay identical.
Save command/stdout/zero exit/inspection/removal before full owned qualification.
Compiler image39992, runtime imageC388, conditional setvars and exact bounds
remain unchanged. Isolated compile does not grant runtime/reference authority.

RESULT -> Source/CPU only. Every consumed own array directly hashes the exact
raw bytes used, checks stat before/after, and requires an identical second read
and unchanged stat. A later correct path hash cannot bless earlier wrong bytes.
New producer/public chronology orders original first and optional prefix4 CPU
terminals, final work CPU terminal, phase/model terminal, actual stdout completed
and retired/EOF finish, GPU terminal, posthealth/journal and new publisher4.
Unattempted prefix4 cannot borrow an epoch. Finite typed monotonic epochs and
original typed work report/command receipts remain mandatory.

Exact prior 88-control command (11 original experiment +77 dependencies):

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest \
 test_owned_hc_device_rs_cpu_v1 test_native_rms_owned_device_ops_cpu_v8 \
 test_native_rms_device_ops_cpu_v1 test_native_rms_entry_cpu_v7 \
 test_native_rms_setvars_cpu_v6 test_native_rms_lifecycle_cpu_v5 \
 test_native_rms_kernel_journal_cpu_v5 test_native_rms_receipt_binding_cpu_v4 \
 test_native_rms_owned_timeout_cpu_v3 test_native_rms_publisher_cpu_v2 \
 test_native_rms_rsqrt37_owned_cpu_v1 test_native_rms_rsqrt37_cpu_v1 -q

Exact full V2 95-control command adds the7 successor controls:

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=strata/flash-next python3 -m unittest \
 test_owned_hc_device_rs_cpu_v2 test_owned_hc_device_rs_cpu_v1 \
 test_native_rms_owned_device_ops_cpu_v8 test_native_rms_device_ops_cpu_v1 \
 test_native_rms_entry_cpu_v7 test_native_rms_setvars_cpu_v6 \
 test_native_rms_lifecycle_cpu_v5 test_native_rms_kernel_journal_cpu_v5 \
 test_native_rms_receipt_binding_cpu_v4 test_native_rms_owned_timeout_cpu_v3 \
 test_native_rms_publisher_cpu_v2 test_native_rms_rsqrt37_owned_cpu_v1 \
 test_native_rms_rsqrt37_cpu_v1 -q

All tests are source/synthetic CPU controls. No real model/device/compile proof
is transferred from them. New controls cover exact first read, changed second
read and first/prefix/final CPU/phase/EOF/GPU/source chronology contradictions.

VERDICT -> No actual V2 compile/GPU or numerical result. V1 was not runtime-ready
under the peer findings. V2 remains an explicitly scoped device arithmetic
reference; no internal native argument or universal correct rounding claim.
