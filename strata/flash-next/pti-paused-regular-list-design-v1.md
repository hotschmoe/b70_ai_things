# Opt-in paused regular-list retention and loaded graph fixture

CONFIG -> Fresh PTI source/tool overlay over accepted sampled source. Actual
warm Chrome56cbb021 contained57 ImmediateAppendCommandLists submissions and
zero ZEGraph API names;89 normalized operations were only24 big_fill_zero,
39 fills and26 barriers. H36 observed complete model graph reuse, so this
is missing tracer child coverage, not complete model device profiling.

SOURCE -> Current ShouldCaptureAppend admits paused appends only for explicit
ZE graph capture/fork tracking. SYCL ordinary closed-list appends fail this
predicate while paused; the resumed immediate-list submission then cannot
clone absent list.commands_. NEW exact defaultOFF flag
UNITRACE_RETAIN_PAUSED_REGULAR_LISTS=1 admits only tracked non-immediate lists.
Kernel descriptors/events remain stored in the ORIGINAL ordinary-list code.
Active submission/normalization/clock math and event ownership/retirement code
are unchanged. Metric modes are refused for this opt-in.

This deliberately retains original instrumentation even while paused, which
includes original PrepareToAppend global timestamp queries and timestamp
events. No new query callsite is added, BUT more original query calls may
execute than stock paused mode. No no-extra-query/no-extra-wait/timing-overhead
claim. Warmup emits no normalized submitted operation because collection is
still paused; original resumed submission obtains actual submit timestamps.
Any timestamp event/reset/replay loss remains unqualified until actual fixture.

BUILD COMMAND -> root only, fresh source/ELFs/receipt:

    bin/gpu-run python3 strata/flash-next/build_pti_paused_regular_lists_v1.py

CPU-only C388/2CPU4GB recipe, original pinned icpx/compiler/header revisions,
no devices/model mounts. New output/source roots are fixed distinct directories.
Builder closes previous failure-scope omissions with privilege/device-domain
checks, original client settlement and ownership retained through terminal/ID
absence even on cleanup failures. Failure remains latched. No build by author.

FIXTURE -> pti_paused_regular_list_fixture_v1.cpp uses the SAME SYCL
command_graph/UR regular-list path, in-order GPU queue, two named dependent
kernels, owned shared2-int state, warmup graph replay ->values1,2 ->stdout
WARM_DONE. Owner performs actual named IPC resume using accepted CLI and sends
R only after actual state1->0. Exactly one selected replay ->values2,4; graph
retired before owned allocation free/normal terminal. Run matched same-tool
OFF retention0 versus ON retention1, both initially paused, same source/inputs,
card0 and full strict health/lease/currenttool/runtime/EOF/cleanup boundaries.
No model payload. Root must freshcompile before any GPU run.

COVERAGE -> both runs must have no pre-resume normalization records and exact
normal callback/clock/Chrome ownership/properties. OFF expected no child names;
ON requires exactly one PausedRegularFixtureA and B normalized instance,
unique IDs matching exact Chrome GPU rows and actual raw device timestamps.
Do not require graph_command1: these are ordinary closed lists. Only fixture
child scope is tested; full model operation coverage remainsFalse. Any missing,
duplicate, malformed time or lifetime/error/partial output fails. No model
retry before loaded fixture/current public admission passes.

VERDICT -> Source-only. No old ELF reuse, SDK/model math changes, registry
mutation, current model capture, clock alignment or performance promotion.
Fresh install hashes/runtime proofs and actual selected child/lifetime witness
must be added by root after build. Patch apply/undo/query-callsite and tiny
roster/device-domain tests are CPU source checks, not native execution.
