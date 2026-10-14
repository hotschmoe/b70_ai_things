# strata/flash-next

Live: model-lock-ista-iq3.json (ISTA GSQ-RCO IQ3_S and IQ3_XXS artifact hashes).
The serving build and configs are outside the repo under
/mnt/vm_8tb/b70/build/strata-ab; launch notes are in JOURNAL.md (2026-10-13).

Current measured config: serve-iq3xxs-200k.json, stock upstream
296b59d2/v0.1.44.1. Split27, two200k, MTP4, int8 KV, all main experts in
VRAM, parking16384 MiB/4 entries. Strata spec4 includes the seed and offers
at most three MTP drafts. Bounded capacity, forced RAM restoration,
concurrent coherence, reasoning/tool checks and teardown health passed.
Evidence: docs/20261014_flashnext_strata_optimization.md. The soak observer
is stopped at user request; multi-day stability remains unqualified.

Start persistently after the previous endpoint has stopped and GPU health
has passed:

```sh
systemd-run --user --unit=strata-optimized --collect \
  --property=WorkingDirectory=/mnt/vm_8tb/github/b70_ai_things \
  /bin/bash /mnt/vm_8tb/github/b70_ai_things/strata/flash-next/launch.sh
```

Stop: `docker stop -t 30 strata-ab-serve`. Endpoint: :18085/v1,
`hotschmoe-dd` first, detailed IQ3_XXS alias retained.
Switch to 27B: `bash strata/flash-next/fallback-27b.sh` (:18080/v1).
The wrapper passes the existing prerequisite check and shell syntax check;
its complete model switch has not been exercised in this stage.

Probe scripts operate through the already-leased server:

- bench_serving.py: raw SSE, live identity, matched fresh prompts/timings;
  --spec-min-p varies the per-request draft-confidence cutoff.
- bench_concurrency.py --template <saved-coding-response.json>: matched two
  overlapping requests, aggregate output, per-stream timings and hashes.
- bench_context.py --parking: near-full concurrent retrievals, overwrite both
  GPU slots with short chats, then verify concurrent cached-prefix returns.
- soak.py: HTTP metadata, container RAM, per-card DRM fdinfo, and error lines;
  no test prompts, GPU queries or automatic restarts. Stop its persistent
  observer with `systemctl --user stop strata-soak.service`.

Experimental EP: serve-iq3xxs-ep-experiment.json uses the temporary upstream
PR2205/1798 build. Decode improves, but prefill regresses to about580 tok/s;
this is not the daily-driver config. EP/spec3 also failed a new tool
continuation that completes on layer/spec4. Two-stream EP total improves
about10%, but both measured paths offer no concurrent MTP drafts.
launch.sh adds SYS_PTRACE only for EP
to permit Level Zero IPC with Docker's default seccomp profile. Upstream:
https://github.com/Niko1221/Strata/pull/2468. ep_ipc_probe.cpp records import
status; ep_transfer_probe.cpp failed before large-transfer measurement and
was revised to retain full-sized native mappings across modes; see S5.
The fixed-lifetime S5 transfer probe passes24 returned-byte checks, but DMA
is slower than the uncached kernel (1.65 vs3.32 GB/s on the large buffer).
The S3 failure remains recorded; its cause is not isolated. Build/run and
recovery evidence are in the S3/S5 document/raw logs.

The paused UD-Q4_K_XL RAM-spill campaign code that used to live here was
moved to the gitignored archive/flashnext-udq4xl-paused-20261014/ on
2026-10-14 and remains in git history before that commit. Its docs are in
docs/flashnext_udq4xl_paused/.

Opt-in FP32 peer reduction: patches/ep-prefill-reduce.patch applies after
upstream PR2205 on296b59d2; serve-iq3xxs-ep-reduce-experiment.json points
to the frozen tested binary.26-case two-card numerical probe source is
ep_prefill_reduce_probe.cpp. Matched32K prefill576 ->1106,8192 chunk1265;
text can differ due to FP32 regrouping. Functional and teardown checks pass,
but no full-context candidate qualification or daily-driver promotion.
Upstream contribution targets PR2205's author branch:
https://github.com/recutita/Strata/pull/1. S6 records full evidence.


Optional EP PUSH return experiment: `serve-iq3xxs-ep-push-experiment.json`
uses frozen `strata-push-s9`, exact workspace12288 and both
STRATA_EP_PREFILL_REDUCE=1 / STRATA_EP_PREFILL_PUSH=1. Same-binary pull/PUSH
32K medians1308.4/1530.9 tok/s and25.077/21.444s client TTFT; nine matched
responses exactly equal. Reduction changes FP32 addition order relative
to full-row combine. This configuration has short-prompt/functional and
teardown-health evidence, not full200k or shelf qualification. See S9.
Build only the `strata` target when editing this patch; the optional
standalone numerical probe also accepts `--push`.


For operation latency use Intel unitrace -d -h -s with Chrome kernel/API
logging around the leased native engine. Stop/flush the session and run a
canary before native shutdown (_Exit bypasses profiler destructors). Capture
SYCL graphs with collection ON before pausing, or replayed kernels are
missing. Match the actual serving config, not just the stable model alias.
`summarize_trace.py trace.json --output <directory>` exports counts, GPU and
host operation median/p95/range plus per-device busy interval unions. Use
--start-us/--end-us for retained request windows and an explicitly calibrated
--timestamp-shift-us when the profiler clock differs from the client clock.
Retain raw trace, submission report and calibration. Kernel copies require
source geometry to annotate transfer bytes; profiled host copy calls include
queue waits. Measure speed separately without tracing. See S10 for complete
prefill/decode census, the clock issue and matched timings.
