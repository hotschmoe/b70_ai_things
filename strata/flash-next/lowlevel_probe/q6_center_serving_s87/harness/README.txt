S87 finite same-binary Q6-center serving comparison (source-only handoff)
2026-10-15 /root/ple_upstream

CONFIG -> COMMAND -> RESULT -> VERDICT
CONFIG: one new frozen binary, STRATA_Q6_CENTER=0/1; native two-card EP,
149/363 experts per layer all resident, NVMe/direct PLE, C2/depth2/head-pair
ON, common HC, S49 OFF, head RPW2 OFF. Explicit Q6 WIDE1/MIN0/MAXC8,
A2=1/RPW1/PF0/Q6_PACKED0. TRACE1 matched both arms; CPU rings/profile absent.
One native process owns both GPUs; parallel2 creates two batch slots.

COMMAND (root executes only after artifact/source audit):
cd /mnt/vm_8tb/github/b70_ai_things
s87=/mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q6-center-serving-s87
bin/cpu-run python3 "$s87/harness-prep/make_configs.py" --exe /mnt/vm_8tb/b70/results/flashnext_ista_gsq_rco_iq3xxs_strata_cachefix_20261014/q6-center-serving-s87-build/build/strata-q6-center-s87 --sha256 a1e35c4ddecff6cfcca8cd61945d0243274f4867e8eff4d134c4ad535b920900

Root announces two cards and expected finite duration to other agents, then
records full per-card plus compiled-collective prehealth before arm0. A prior
full posthealth can be reused only with an explicit no-intervening-GPU/stack
change audit. Do not restore the daily server or repeat full-cache tests.
For arm0, then arm1, substitute arm=0 or arm=1 below, stop on any failure:
arm=0
bin/cpu-run python3 "$s87/harness-prep/start_arm.py" "$arm"
bin/cpu-run --quiet python3 "$s87/harness-prep/client.py" "$arm" warm
bin/cpu-run --quiet python3 "$s87/harness-prep/client.py" "$arm" timed
bin/cpu-run --quiet python3 "$s87/harness-prep/client.py" "$arm" functional
bin/cpu-run --quiet python3 "$s87/harness-prep/stop_and_health.py" "$arm"

After both arms complete:
python3 "$s87/harness-prep/compare.py"

Detached serve holds gpu-run pair lease, never a long CPU lease. Client quiet
lease is outside commands. stop_and_health has NO outer gpu-run: it joins
normal QUIT/removal before separately acquiring each health lease. Cleanup
remains required on client failure; invoke owned cleanup, preserve all failed
artifacts, no restart/retiming/force kill or foreign-name mutation. Arm1 may
reuse arm0's immediately prior full posthealth under root coordination.

RESULT: pending; no harness execution, GPU/build/model call by this task.
Five Python files parse and shell syntax checked only. Source freeze binds
all six controllers in harness-source-manifest.json and expected-source.sha256.

VERDICT gates: one warm and three fresh107-token paired1500-token requests;
request/text/SSE/usage and five present finite nonnegative native work fields;
windows/paired/solo/printed rows/new-capture logs exactly matched; functionals;
model/binary/config/effective env; residency; exactly one correct first-eligible
host marker after warm; normal QUIT/full CID+NAME absence and posthealth.
Any unmatched gate suppresses aggregate speed summary. Marker proves host
selection and successful captured-path use, not independent GPU dispatch.
Client wall is primary, with per-stream TTFT/content-event gaps/completion/
decode and whole-request output throughput plus pair aggregate. SSE events
can contain multiple tokens; printed row averages are rounded; logged new
captures omit cache-hit layouts. No device-service/zero-overhead/promotion claim.
