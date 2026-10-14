# Source42 H36 terminal correction and separate PTI closing gap

CONFIG -> actual source41 normal serve ends at generate.cpp12025 with
std::_Exit(0). Actual host73892 has four H36 CLOSED records but no SESSION:
registered atexit(exit_export) is bypassed. The old collector refusal is
correct and remains unchanged. Existing SDKs/results/plans are immutable.

COMMAND -> new patch0042 is built from exact source41 SDK41 and parent plan
2f40973f. It changes only normal serve terminal code and H36 header. After the
original mirror/graph/source shutdown, enabled owner explicitly calls
exit_export before existing fflush/std::_Exit(0). Header sets atomic on=false
before deleting owned event storage, so repeat explicit/atexit calls cannot
emit again or double free. DefaultOFF emits nothing and has no heap, GPU
query/wait or model arithmetic work. Normal fast_exit remains unchanged;
error _Exit(1) paths remain failures without fabricated session closure.

RESULT -> five source CPU controls pass: exact full-byte undo/two-file patch,
normal/error exit preservation, unchanged GPU wait/query call roster, owner/
idempotent header order and source42 fresh-build roster. The standalone CPU
header fixture is source only and has not been compiled/executed by this
author. Root can compile it without SYCL/device/model dependencies:

    g++ -std=c++17 -pthread -I strata/flash-next strata/flash-next/host_terminal42_cpu_fixture_v1.cpp -o NEW_CPU_FIXTURE
    NEW_CPU_FIXTURE on NEW_EXISTING_OUTPUT_DIRECTORY fast
    NEW_CPU_FIXTURE on ANOTHER_NEW_EXISTING_OUTPUT_DIRECTORY natural
    NEW_CPU_FIXTURE off THIRD_NEW_EXISTING_OUTPUT_DIRECTORY fast

Each ON case must emit exactly one SESSION and one CLOSED/file despite two
explicit closes; natural return additionally runs the old atexit callback.
OFF must emit neither and own no ring allocation. Root records actual compiler,
source/binary hashes and terminal proof. This is header lifetime evidence only.

VERDICT -> source-only source42 proposal; fresh complete8ABI build and new
source/upload/model control remain required. Final header hashes bind the
new expected source closure; old added-header payload/ordered patch proof is
retained as historical generation bytes, not a final-payload equality waiver.
Source41 runtime/aliases/proofs are not relabeled42. No registry mutation.

Separate PTI gap, NOT fixed by patch42:

- tracer.cc359 Fini is an ELF destructor invoking Teardown; native _Exit skips
  it. Teardown deletes UniTracer; tracer.h214 disables tracing, flushes data,
  finalizes/deletes collectors and deletes ChromeLogger. Those terminal paths
  are not reached by the model fast exit.
- ChromeLogger destructor/Flush in chromelogger.h closes JSON and buffered
  events. Partial capacity flushes do not prove final closure.
- Clock Writer destructor in pti_unitrace_clock_export_v1.h emits final bridge
  site2 and closes FD. _Exit skips it too. OS descriptor closure is not a
  replacement for that explicit source-bound terminal record.

Proposed diagnostic-only closing route for separate review/build:

1. Fresh tool exports a versioned noexcept C finalizer under a dedicated
   diagnostic enable flag. It validates current owner/role, runs existing
   Teardown exactly once, then explicit idempotent clock Writer.CloseOnce.
   The close retains the final bridge and error latch; a registered destructor
   later becomes harmless. Symbol/code/library identity must be checked.
2. A separate defaultOFF native diagnostic hook, after original model shutdown
   and H36 close but before _Exit, resolves that exact loaded symbol only when
   the explicit flag is enabled. Missing symbol/failed close is an actual
   diagnostic failure. Normal fast_exit and all ordinary model paths remain.
3. Retain actual mapped fresh injection census, final clock bridge/FD status,
   valid complete Chrome JSON, raw frame/instance/flow joins, model outputs,
   owned EOF/full4/health. Teardown can perform stock profiler waits/queries;
   it is not a zero-added-wait observer. No atexit-global replay, changed
   normal exit semantics or invented tool-complete marker is permitted.

This PTI route is DESIGN ONLY, not implemented or runtime-qualified. Fresh
budget build19516 does not close the native exit gap. No model/GPU/compiler
or tool query action was performed by this author.
