# Opt-in owned PTI explicit finalizer prototype

CONFIG -> native normal fast _Exit bypasses both ELF Fini/Chrome teardown and
clock Writer destructor. Patch42 fixes H36 only. This is a separate diagnostic
prototype: fresh tool overlay plus native patch43 over exact source42 bytes.
All accepted source/install/SDKs are unchanged; fresh complete builds required.

COMMAND -> future ON explicitly sets STRATA_DIAGNOSTIC_PTI_FINALIZE=1 and
UNITRACE_MODEL_FINALIZE_V1=1, with existing clock export/bridge flags. OFF omits
both new flags. Future native hook runs after original shutdown/H36 close,
before existing fflush/_Exit0. It only resolves already-loaded versioned
b70_pti_model_finalize_v1, verifies canonical /pti/lib/libunitrace_tool.so
origin, passes actual version/PID/TID and fails diagnostic close if missing/
wrong/failed. DefaultOFF returns before dlsym; normal exit semantics remain.

The tool initializes immutable PID/TID role in its existing constructor only
when opted in. Its noexcept C export runs existing Teardown once then explicit
Clock CloseOnce. Owner/version errors and partial callback failures latch;
retry may finish necessary cleanup but never erase failure. Clock close emits
existing final bridge site2, closes descriptor once, forbids later emission
and makes the later registered destructor harmless. Teardown/Chrome existing
flush/finalize code remains; no global atexit replay or invented completeness.
Existing Teardown may perform profiler waits/queries and source logging; this
is explicitly not zero-added-waits and is outside model request latency.

RESULT -> six source CPU controls pass; standalone owned/idempotent/partial
failure guard and actual clock-close fixture sources are supplied for root
CPU compilation. No compilation/model/tool/GPU run by this author. Guard CPU
fixture invokes real callbacks and expects original failure retained after
cleanup retry; clock fixture tests fast/natural close and late emit86.

Root CPU commands, with source/compiler/binary/terminal receipts:

    g++ -std=c++17 -pthread -I strata/flash-next strata/flash-next/pti_model_finalize_cpu_fixture_v1.cpp -o NEW_GUARD_FIXTURE
    NEW_GUARD_FIXTURE
    g++ -std=c++17 -pthread -I strata/flash-next strata/flash-next/pti_clock_close_cpu_fixture_v1.cpp -o NEW_CLOCK_FIXTURE

Run each clock case in an independent no-device container/output with writable
/results/unitrace: fast/natural require one initial+one final bridge, late
must preserve prior complete rows and exit86. CPU-owned provenance only; no
full native/tool ABI or actual model qualification.

VERDICT -> SOURCE PROTOTYPE, not launch-ready. Required independent follow-up:
CPU fixture/build/exported-symbol checks, fresh exact tool/native source43
build, actual before/after mapped fresh library census, strict complete
Chrome/final clock close parsing, raw budgets, model outputs/health/full4/
owned EOF. Explicit success return/marker alone does not prove all output
writes succeeded: original Chrome/logger/raw predicates remain mandatory.
Actual model operation/copy-byte/H36-to-L0 mapping/calibration flags stayfalse.
Old root clock-budget19516 ELF has no new finalizer and cannot be substituted.
No source42/41 result is relabeled source43, and registry stays unchanged.
