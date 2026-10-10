# Batch observer earlier-stage stamp37 source audit

CONFIG

Exact consumed source35 SDK: /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T233718Z-i6s74tbl/source
Parent build plan current-ple-prompt35-engine-build-plan-v1.json:
82004f6cee0f975c433d245b304aff10d67dd33e028cf926a08ef5eead892ef7
Source verify.cpp SHA cdd837fd3dd31873094587e549623bae887765c48d8c4de7f542185f52c4d84f
Source batch_fidelity_observer.hpp SHA ddd2475f38f42e731a99ff52678cda9e742b20c75ead38383a735ca0add9db39
Failed actual paired native2 child report SHA 48c91e9b8569d79a018151b175fa2f9d1767cfbb66b8789f48ae208a9444713b
Failed actual engine.combined.log SHA c84da2ca140957b0857d449cb9d797471fdbbbb69820eac9d5ea06eddf700afc

COMMAND

Read-only source/log audit and source reconstruction. CPU controls:
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s strata/flash-next -p test_batch_stage_stamp37_cpu_v1.py -v
No compiler, GPU, Docker, model payload read or consumed SDK mutation.

RESULT

Actual log: warm RID1001/RID1002 each BDONE length32; ARM then RID2001
admission_target_verify spans complete T2 [0,58) across stage0 [0,32)
and stage1 [32,48). Final output reports invalid_argument:
"batch observer graph seal/stamp mismatch". No observed successful selected
admission residual/logit publication; no inference correctness verdict follows.

Source verify.cpp1628 records each stage's residual layers. Lines1637-1642
copy all three handoff components then return true for le_<n_layers.
The sole batch_snapshot_->stamp(*cs) is at1741, after the head/logits path.
Thus non-head stage0 cannot set its graph roster stamped bit. Its diagnostic
T1 admission capture begins roster at1798, executes record_window, then seals
at1856. observer.hpp82 rejects !graphs_[i].stamped. This exact host exception
requires no numerical, device/context, graph-key collision or stamp-clock
hypothesis. Warm unarmed graphs do not build diagnostic rosters; ARM activates
separate batch_admission_exec_ / batch_observe_exec_ caches. Snapshot is
per-Verifier, built with owning queue/device/lb/le/head at3523. Stage1 already
has the head stamp; the same missing earlier-stage stamp also affects an
activated non-head multirow capture_batch seal2835 and solo migration T1.
The saved terminal exception alone does not encode the host backtrace; the
source route proves the earliest selected stage0 T1 graph cannot seal.

Patch0037 adds exactly one existing guarded observer stamp after all three
handoff graph copies and before the non-head early return. Existing final-head
stamp, graph cache keys/selection, roster layout, queue/device/context guards,
request identities, layer/head producer sets, epoch/frame publication and
model arithmetic/state/handoff calls remain byte-identical. Observer OFF
(batch_observe_ false or no snapshot) adds no metadata/device work. Activated
observer adds its intended two device memcpy marker nodes to the earlier
stage graph; it adds no wait and does not claim no diagnostic overhead.

10 CPU controls PASS: source35 stage0 missing-stamp failure/head success;
new split-stage seals; exact source single addition and placement; OFF branch;
missing/duplicate/foreign producer and head rejection; wrong-layout/duplicate
stamp rejection; real queue/device/context/roster guard presence; exact all63
source/27 added-header/36 patch/eight ABI/six Python closure; optional trace36
composition preserving every other byte. CPU roster witness is source-indexed
semantic simulation, not compiled SYCL or actual GPU graph execution.

Optional combined36+37 plan applies immutable H36 then this exact patch to its
unchanged handoff anchor:64 source/28 added-header/37 patch/eight ABI/six Python.
H36 header/patch/plan/evidence remain frozen; H36 must stay OFF for paired
observer tests because H36 ON admits bounded onecard serial only.

VERDICT

Deterministic observer construction defect, supported minimal source repair;
not evidence of model math failure or restored twoGPU correctness. Native ABI
build/runtime remains UNEXECUTED. Root must fresh-build all eight ABI targets
and transitive libraries, verify source/header/runtimePython/archive/binary
closure and actual code presence, use new generation admission/upload390,
original four-shard/page checks, pair lease/card maps/strict per-card plus
compiled P2P0 pre/posthealth/journal/owned terminal/logical frees.

First matched OFF/ON experiment should retain exact paired native2 configuration,
complete real functional outputs/head/accepted token IDs and both-stage
admission plus multirow residual/logit rosters, immutable raw arrays and request
identities. No old source35 proof transfers. Subsequent4/6 concurrency,
full model arithmetic/broad quality/latency remain independent open gates.
No tolerance adjustment, fitted constants or speed/stability claim.
