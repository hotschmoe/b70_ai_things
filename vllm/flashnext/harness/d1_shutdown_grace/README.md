# D1 private shutdown grace discriminator, CPU preparation only

CONFIG: repeat SAME retained-default control/config/observer/request body
from d1-existing-control-r1. Only supported CLIshutdown timeout30 and
Dockerstop45 replace actual5 and Docker20. Passivehost/proc observer starts
immediately before normalDockerstop, after ALLclient timings finish.
No kernel/graph/FA/fork/shared lifecycle change, no newGPU hook, ptrace,
GDB, faulthandler, extra signal, reset or build. InitialCONTROL treatmentflags remainOFF. Wrapper can also accept explicit
routeq8f for a LATERseparate rootgrant, with same30/45/sources/observer.
No auto-chain; allowingCLIchoice is not authorization.

COMMAND: CPU reads confirm actualsource of5, not an assumed default:
arg_utils.py1928 supports --shutdown-timeout, int/default0. Actualfork
platforms/xpu.py445 conditionally maps0->5 soXPU gracefully releases
oneCCL/LevelZero. Actualcontrol serve.logline63 at19:04:56 explicitly logs
[xpu.py:447] XPUplatform:set server shutdown_timeout=5; actuallive/lmx
server-config.json records5. Positive30 bypasses that0conditional.
Originalsource xpu.pySHA097b700117ae487e9242cb59c89a85aa957cd14caa13427308f7328c65b1f982.

APIlauncher269 passes config.shutdown_timeout toMPClient.shutdown;
CoreEngineProcManager336 joins viautils.shutdown611 withthat deadline.
Observedworker.done19:09:52 followedAPIforcekill19:09:56 at5s deadline.
Worker.done isNOTEngineCore/process.done: core.py807 subsequently calls
scheduler.shutdown andcleanup_dist_env_and_memory. Distributed cleanup
parallel_state2521 destroysgroups, gc.collect, empty_cache/empty_host_cache;
thenPython/C++finalization can remain. EPState has static graphmap and
optionalSYCLqueue; PState/Region rawpointers have no teardownhook. These
are source-backed remaining surfaces, NOT a blocked-phase diagnosis.
No new instrumentation or EP cleanup patch is justified by the5s log.

RESULT: private make_snapshot.py reuses EXACT productiontrimmedclient
builder and observer; prepared/grace-vs-control.diff shows ONLYteardown
watcher/Docker45, while window.sh addsCLI30 and metadata/privatepins.
Production defaultcontrol and its forcekillreject remain unchanged.
Same onepromo/shortbench/minimalF1/canonicalONLY, postREADY240s body.
No startup timeout change. Original canonical61.5/61.8 requestcomplete
and healthyGPUs remain separate from unsuccessful naturalprocess exit.

teardown_watch.py usesdocker top ONCE to map actualEngineCorehostPID,
thenreadsleader/threadstat,wchan,status every0.5s for<=45s. It writes an
armedfile after first sample; Dockerstop starts after atmost7s handshake (docker.top lookup timeout5s).
No deviceioctl orsignals. Sampling ends onZ/disappearance/45s. Rawstat52
waitstatus is recorded when observable; missingPID never impliesexit0.
Wchan0/readerrors are limited visibility, not proof of an idleorblocked
thread. Processreaping can prevent observing Z/exitstatus; lifecycle logs
must independently show noforcekill andnormalclose. No PC/backtrace claim.

Handshake requires/reports first_sample_saved true. If lookup/handshake/
sampler fails, normalDockerstop STILLruns, monitorfailure stays inoutcome,
and no server is stranded.

CPU test verifies proc field52 on naturally exiting child(code7), exact
CLI/Docker/privateclient deltas, ASCII/AST/bash. No Torchimport/GPU touched.
CPUcheck requires --out to a fresh scratchdirectory AND --prepared to
the generated private snapshot directory, safe aftersourceport.
Source outputs/scratch areB/d1/shutdown_grace_r1, production files are
read-only dependencies; this preparation has not been ported or launched.

VERDICT: CPU_READY only. If parent grants after review/newsourcefreeze:

    ARM=control TAG=d1-shutdown30-control-r1 bash vllm/flashnext/harness/d1_shutdown_grace/window.sh

Bothcards ~7-9min inclload/samebody/nativeprepost plus<=45sDockergrace.
OUTB/d1/shutdown_grace_r1/d1-shutdown30-control-r1;
liveoutputsB/f1/merged-d1-shutdown30-control-r1. Source/private/SOpins,
actualconfig/branch/canonical/strict/fatalhealth checks retained. Private
outcome MUST record actuallive shutdown_timeout30; missing/wrongvalue
fails. No weakeningengineforcekill detector andno retrospective success.
Naturalexit within30 wouldsupport timeout-limitedcleanup only for this
finite recipe/body; continuedforcekill adds boundedprocwitness, notproof
of deadlock or EP fault. NoGPU job queued andno reset of healthyGPUs.
