# D1 native capture -- CPU validated; GPU not run

The previous EngineCore exit is SIGSEGV11 from saved proc/stat field52.
The PC is unknown. Its second timed C traceback dump is truncated, making
the monitor a possible contributor. No GPU retry is authorized here.

Prefer GDB over a new signal shim. The target initially has SIGSEGV caught;
the existing handler's owner/intent is unknown. A replacement shim that
does not preserve it can manufacture a different runtime fault. Calling
dladdr/backtrace from a SIGSEGV handler also has signal-safety risks.
An async-write-only PC shim could work, but provides no native backtrace
and still requires carefully chaining the unknown handler.

GDB setup is local to this one research container. No host core_pattern,
Yama, Apport or system setting changes. Core0 plus the host Apport pipe
does not yield a reliable private core just by raising ulimit. GDB can
write a core directly if needed later, but do not request a full process
core initially: mapped model/host-USM state may make it very large.

Prepared files: Dockerfile.gdb, check_packages.py, make_gdb_commands.py.
Image build adds only GDB/dependencies and rejects any changed existing
package. CPU setup completed in268s (estimate30-90s was low), with25 new
packages and zero existing package-version changes. All parent image
layers remain. Base ID ea234fbf7123cba58b387f4046a0c32192dd5b6febead3684dbd8690882bf572;
debug ID b6a97730abd8bcb76b73f3d336c6e5b2a4bbdd096622c366c0b7f9d1f7a49564.
No host settings changed. Reuse the cached image; no repeat package setup.

Proposed CPU setup, only after parent approves cost:

```
MAX_JOBS=4 bin/cpu-run docker build \
  -t vllm-flashnext-xpu:d1-gdb-r1 \
  -f /mnt/vm_8tb/b70/build/vllm-flashnext/d1/native_capture/Dockerfile.gdb \
  /mnt/vm_8tb/b70/build/vllm-flashnext/d1/native_capture
```

CPU-only synthetic target in a container without /dev/dri passed twice:
GDB attached under per-container SYS_PTRACE/default seccomp, recorded two
SIGSEGV PCs/backtraces, delivered one recoverable fault to its existing
handler, and preserved the second fatal waitstatus11. Debugger portion
was0.315s; exact early-options/resume-handshake wrapper took9s. Artifacts
remain under build/vllm-flashnext/d1/native_capture/setup-r1, not in repo.

Future reviewed GPU window requires private copies of xpu-run and merged
lifecycle to select the derived image and add only --cap-add SYS_PTRACE.
Try default Docker seccomp first; do not disable it broadly unless the
CPU preflight demonstrates a denied required debugger syscall. Root GDB
inside this container attaches to the EngineCore namespace PID after
READY; parent waits the gdb.events.cont gdb-resumed.json handshake before
any diagnostic client. Early -iex options disable auto-loading and
debuginfod before attachment; no external symbol download.
This avoids following multiprocessing's resource-tracker child by mistake.

Generate commands with make_gdb_commands.py --out SHARED_OUTPUT, then:

```
docker exec --user 0 CONTAINER /usr/bin/gdb --nx --batch -q \
  --pid ENGINE_NAMESPACE_PID -x SHARED_OUTPUT/capture.gdb \
  > SHARED_OUTPUT/gdb.log 2>&1
```

SIGSEGV logging captures PC/registers/shared-library mapping and bounded
native backtraces, then delivers the signal to the original handler.
Some SIGSEGV may be recoverable; label one fatal only if the process
actually terminates. No calls into target Python/C++ and no GPU events,
queue waits or tensor representations are requested by GDB. Attaching
and taking backtraces pause the process; no latency qualification.

The private diagnostic overlay disables both D1 C dump_traceback_later
and Python dump_traceback paths; retain host breadcrumbs/proc sampler.
make_native_snapshot.py/native_window.sh prepare the reviewed private
runner/lifecycle and preserve shared harness/plugin source. Keep the
same N1-original recipe and exact client sequence through first math,
120s postREADY body, lifecycle/post-health outside. A native SIGSEGV PC
in CPython dumping code versus SYCL/native MoE is a concrete discriminator;
no source similarity or T6 shape alone establishes cause. A clean finite
run would not prove removing the C watchdog fixed serving stability.
Debugger stops can perturb EP device flag timing; this is fault
localization only, not an unperturbed serving/speed comparison.
