# Qwen3.8 W8A8 + DSpark -- loop ledger

Standing feedback file for agents looping on
`docs/20260818_qwen38_w8a8_dspark_campaign.md`.

Read this **after** the campaign living header and **before** JOURNAL.
Newest loop at the **bottom**. One `## LOOP N` block per iteration.
Do not rewrite old loops. The campaign's living header points here.

Full evidence lives in `JOURNAL.md` (config -> command -> result ->
verdict). Dead-end closures live in
`docs/20260818_qwen38_w8a8_dspark_deadends.md`. This file is only
the handoff: what was picked, what changed, what the next loop
should do, and what it must not redo.

Block template (copy from campaign section L.3):

```
## LOOP N -- YYYY-MM-DDThhmmZ -- <one-line pick>

Picked:
Why this, not the other open row:
GPU:
Command:
Log:
Result:
Verdict: GO / NO-GO / BLOCKED / DEAD-END / RUNNING
Changed beliefs:
Next pick:
Do not:
Restore:
JOURNAL:
```

---

## NEXT PICK (keep this line true)

E3 -- oneDNN barriers-on A/B vs AGASYNC 29.4.
PRE.15 list is written
(`docs/20260819_qwen38_027_only_features.md`);
Phase 2 stays closed. First: fetch Steve lab
(local clone stale at 03f98aaf), read 9f90e2c /
GDN-scratch note for the env, restart
SPECTOK=4 GRAPH=1 ALLGATHER_ASYNC @122880 with
it, G1, bench_code c1. Do not start P4.2 / S2 /
train / Phase 2 / DD. D8-D4 stand. Wipe
b3f7e9e010 before changing SPECTOK. Scheduler
01a01813e05d stays (c1 29.4 < 41.2).

---

## LOOP 0 -- 2026-08-18 -- campaign + loop protocol, no GPU

Picked: author `docs/20260818_qwen38_w8a8_dspark_campaign.md` and
  the loop/dead-end sidecar files. No on-GPU work.
Why this, not the other open row: campaign did not exist; Phase 0
  GPU work is gated on a lease + stopping DD.
GPU: none. DD `hotschmoe-dd` 3.6 NVFP4 TP=2 left up on :18080.
Command: docs only.
Log: n/a
Result: campaign is LOOPING. Living header Next pick = P0.1.
  Sidecars created empty-of-GPU-results.
Verdict: GO (plan + protocol). Research locked. No number moved.
Changed beliefs: this campaign is a continual loop, not a one-shot
  plan. Subsequent agents must read this ledger first and write a
  LOOP N block even when they only unblock yaml / scripts.
Next pick: P0.1 HE+ on grafted W8A8-gptq MTP3. Optional no-GPU
  first step: evals yaml id `qwen3.8-27b-W8A8-gptq` /
  `qwen3.8-27b-W8A8-gptq-mtp3`. Then stop DD, serve
  `vllm/w8a8/serve_qwen38_27b.sh` MTP3, HE+ 164 + `bench_code`
  c1/c4, restore DD.
Do not: enter Phase 2, start a DSpark train, invent a PSpark
  checkpoint, take DD down for editing, overwrite w8a8-gptq.
Restore: DD was never stopped.
JOURNAL: ### 2026-08-18c

---

## LOOP 1 -- 2026-08-18T0509Z -- no-GPU yaml unblock for P0.1

Picked: unblock -- add `qwen3.8-27b-W8A8-gptq` and
  `qwen3.8-27b-W8A8-gptq-mtp3` to `evals/configs/models.yaml`
Why this, not the other open row: GPU slot held by DD
  `hotschmoe-dd`. L.4 allows section 8 item 1 yaml unblock
  before the GPU slot for P0.1. LOOP 0 named this first step.
GPU: lease HELD both cards by b70_daily_0 pid=460274 since
  2026-08-18 03:02:11. DD not stopped. :18080 id `hotschmoe-dd`.
Command: edited `evals/configs/models.yaml` (two new rows).
Log: n/a
Result: both campaign ids exist. HE+ still unmeasured. No
  number moved. Weights at `models/files/qwen3.8-27b/w8a8-gptq`
  still present (grafted vision+MTP).
Verdict: GO (unblock)
Changed beliefs: P0.1 served id is `qwen3.8-27b-W8A8-gptq-mtp3`
  (B70_NOMTP=0 MTPTOK=3 MAXLEN=131072). Base
  `qwen3.8-27b-W8A8-gptq` is MTP-off. Do not score against
  `hotschmoe-dd`.
Next pick: P0.1 GPU. First command:
  `bash vllm/daily_driver_serve.sh stop`
  then
  `B70_NOMTP=0 MTPTOK=3 GRAPH=1 MAXLEN=131072 UTIL=0.90 TP=2
  PORT=18080 NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq-mtp3
  ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start`
  Query /v1/models, then HE+ 164 thinking-off greedy. Multi-hour:
  start, LOOP-STARTED with log+pid, restore DD after job ends.
Do not: start HE+ against `hotschmoe-dd`; enter Phase 2; train;
  take DD down for more editing; overwrite w8a8-gptq.
Restore: DD was never stopped.
JOURNAL: ### 2026-08-18d

---

## LOOP 2 -- 2026-08-18T0546Z -- P0.1 HE+ started on W8A8-gptq MTP3

Picked: P0.1 -- HE+ 164 thinking-off greedy on grafted W8A8-gptq MTP3
Why this, not the other open row: living-header Next pick; LOOP 1
  yaml unblock landed; last verdict was GO not RUNNING.
GPU: lease HELD both cards by serve wrapper pid=464242 since
  2026-08-18 05:41:43. DD stopped. :18080 id
  `qwen3.8-27b-W8A8-gptq-mtp3` (root /models/qwen3.8-27b/w8a8-gptq,
  max_model_len 131072). NAME=qwen38_w8a8. P2P=0.
Command:
  bash vllm/daily_driver_serve.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  B70_NOMTP=0 MTPTOK=3 GRAPH=1 MAXLEN=131072 UTIL=0.90 TP=2 PORT=18080
    NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq-mtp3
    ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
  evals/.venv/bin/python -u evals/orchestrator/run_evals.py
    --endpoint http://192.168.10.5:18080/v1
    --model qwen3.8-27b-W8A8-gptq-mtp3 --quant W8A8-gptq-mtp3
    --tiers 1 --tier1-dataset humaneval --limit 164
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_heplus.log
  serve: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_serve.log
  he+ pid 467692 (file loop2_heplus.pid)
  serve wrapper pid 464242 (file loop2_serve.pid)
  result dir:
    evals/results/20260818T054613Z__qwen3.8-27b-W8A8-gptq-mtp3__W8A8-gptq-mtp3
Result: G0 id match. G1 Paris exact / 17*23=391 / fib iterative.
  Serve HEALTHY 234s, gen probe "Paris." OK. HE+ generating 164
  (thinking=off, greedy, seed=1234). Plus still unmeasured.
Verdict: RUNNING
Changed beliefs: 3.8 W8A8 MTP3 @131k still loads coherent on
  int8g-v0260 GRAPH=1. Do not score this slot as hotschmoe-dd.
  Next fire must finish this pick, not start P0.2.
Next pick: P0.1 finish. First command: `ps -p 467692`.
  Live -> one status line, STOP. Dead -> read plus from the
  result dir / he+ log, write JOURNAL verdict, leave W8A8
  serve up, STOP. Do not start P0.2/bench in that fire.
  How to tell done: pid 467692 gone AND log has pass@1 / plus
  (or a traceback). Gen ~20-40 min historically.
Do not: start a sibling HE+ or P0.2; wait on this fire; enter
  Phase 2; train; overwrite w8a8-gptq; P2P=1; score DD;
  start daily_driver_serve.sh.
Restore: DD stays PARKED. After HE+ ends leave
  `qwen38_w8a8` / `qwen3.8-27b-W8A8-gptq-mtp3` up for P0.2.
  Do not run `vllm/daily_driver_serve.sh start`.
JOURNAL: ### 2026-08-18e (operator override ### 2026-08-18f)

---

## LOOP 3 -- 2026-08-18T0611Z -- P0.1 HE+ finish: died 90/164

Picked: P0.1 finish (monitor pid 467692 / write plus)
Why this, not the other open row: living-header Next pick;
  last verdict RUNNING; do not start a sibling.
GPU: lease FREE both cards. DD PARKED. :18080 empty.
  qwen38_w8a8 Exited(0) 2026-08-18T06:01:59Z. xpu-health
  card 0 OK after teardown.
Command:
  ps -p 467692; docker inspect qwen38_w8a8
  tail /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_heplus.log
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 \
    --img vllm-xpu-env:int8g-v0260 --timeout 90
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_heplus.log
  crash: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_serve_crash_tail.log
  raw: evals/results/20260818T054613Z__qwen3.8-27b-W8A8-gptq-mtp3__W8A8-gptq-mtp3/tier1_evalplus/humaneval_raw.jsonl
Result: HE+ generated 90/164 (HumanEval/0..89) then
  connection refused. Serve: VllmWorker-0 NEO abort
  linear_stream.h:84 during MTP-verify (GRAPH=1
  CGRECLAIM=0). No plus. Cards healthy. No DEVICE_LOST.
Verdict: NO-GO
Changed beliefs: 3.8 W8A8 GRAPH=1 CGRECLAIM=0 MTP3 is
  unsafe for HE+ 164 (~15 min / 90 sequential gens).
  Same leak as 2026-07-07 / 2026-08-05. Do not retry
  that pair. Do not CGRECLAIM=1000 (instantiate SEGV
  on W8A8 v0.26). HE+ retry = GRAPH=0 MTP3.
Next pick: P0.1 retry GRAPH=0 MTP3 then HE+ 164.
  First command: xpu-health card 0, then
  B70_NOMTP=0 MTPTOK=3 GRAPH=0 MAXLEN=131072 UTIL=0.90
  TP=2 PORT=18080 NAME=qwen38_w8a8
  SERVED=qwen3.8-27b-W8A8-gptq-mtp3
  ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
  then same run_evals HE+ 164 (new result dir).
  Multi-hour: LOOP-STARTED + RUNNING, STOP.
Do not: retry GRAPH=1 CGRECLAIM=0; CGRECLAIM=1000;
  start P0.2; publish a plus from 90 samples; start DD;
  enter Phase 2; overwrite w8a8-gptq; P2P=1.
Restore: DD stays PARKED. Serve left DOWN (wrong flags
  for a 164). Lease released. Cards healthy. Next fire
  starts GRAPH=0 serve. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18g

---

## LOOP 4 -- 2026-08-18T0643Z -- P0.1 HE+ started on W8A8-gptq MTP3 GRAPH=0

Picked: P0.1 retry -- HE+ 164 thinking-off greedy on GRAPH=0 MTP3
Why this, not the other open row: living-header Next pick after
  LOOP 3 NO-GO. GRAPH=1 CGRECLAIM=0 is closed for 164.
GPU: lease HELD both cards by docker-wait wrapper pid=471943
  since 2026-08-18 06:43:13. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-mtp3` (root /models/qwen3.8-27b/w8a8-gptq,
  max_model_len 131072). NAME=qwen38_w8a8. GRAPH=0 --enforce-eager.
  P2P=0.
Command:
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  B70_NOMTP=0 MTPTOK=3 GRAPH=0 MAXLEN=131072 UTIL=0.90 TP=2 PORT=18080
    NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq-mtp3
    ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
  evals/.venv/bin/python -u evals/orchestrator/run_evals.py
    --endpoint http://192.168.10.5:18080/v1
    --model qwen3.8-27b-W8A8-gptq-mtp3 --quant W8A8-gptq-mtp3
    --tiers 1 --tier1-dataset humaneval --limit 164
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop4_heplus.log
  serve holder: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop4_serve.log
  he+ pid 471978 (file loop4_heplus.pid)
  serve wrapper pid 471943 (file loop4_serve.pid)
  result dir:
    evals/results/20260818T064331Z__qwen3.8-27b-W8A8-gptq-mtp3__W8A8-gptq-mtp3
Result: G0 id match. G1 Paris exact / 17*23=391 / fib iterative.
  Serve HEALTHY 173s GRAPH=0. HE+ generating 164
  (thinking=off, greedy, seed=1234). Plus still unmeasured.
Verdict: RUNNING
Changed beliefs: 3.8 W8A8 MTP3 @131k loads coherent on
  int8g-v0260 GRAPH=0. Do not retry GRAPH=1 CGRECLAIM=0
  for this HE+. Next fire must finish this pick, not P0.2.
Next pick: P0.1 finish. First command: `ps -p 471978`.
  Live -> one status line, STOP. Dead -> read plus from the
  result dir / he+ log, write JOURNAL verdict, leave GRAPH=0
  W8A8 serve up, STOP. Do not start P0.2/bench in that fire.
  How to tell done: pid 471978 gone AND log has pass@1 / plus
  (or a traceback). GRAPH=0 gen slower than GRAPH=1.
Do not: start a sibling HE+ or P0.2; wait on this fire; enter
  Phase 2; train; overwrite w8a8-gptq; P2P=1; score DD;
  start daily_driver_serve.sh; retry GRAPH=1 CGRECLAIM=0.
Restore: DD stays PARKED. After HE+ ends leave
  `qwen38_w8a8` / `qwen3.8-27b-W8A8-gptq-mtp3` GRAPH=0 up
  for P0.2 only if plus >= 0.90. Do not run
  `vllm/daily_driver_serve.sh start`.
JOURNAL: ### 2026-08-18h

---

## LOOP 5 -- 2026-08-18T0739Z -- P0.1 HE+ finish: plus 0.927

Picked: P0.1 finish (read plus from LOOP 4 HE+)
Why this, not the other open row: living-header Next pick;
  last verdict RUNNING; pid 471978 dead with a plus.
GPU: lease HELD both cards by docker-wait pid=471943 since
  2026-08-18 06:43:13. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-mtp3` (root /models/qwen3.8-27b/w8a8-gptq,
  max_model_len 131072). NAME=qwen38_w8a8. GRAPH=0.
Command:
  ps -p 471978
  tail /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop4_heplus.log
  cat evals/results/20260818T064331Z__qwen3.8-27b-W8A8-gptq-mtp3__W8A8-gptq-mtp3/summary.json
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop4_heplus.log
  result:
    evals/results/20260818T064331Z__qwen3.8-27b-W8A8-gptq-mtp3__W8A8-gptq-mtp3
Result: pass@1 base **0.957** plus **0.927** (164/164,
  gen 2700s, eval 40s, thinking=off, greedy, seed=1234).
  G0 still matches. Serve left up. Plus == Q4_K_M 0.927;
  base 1.3 pts under 0.970. Gate plus >= 0.90 PASSES.
Verdict: GO
Changed beliefs: grafted W8A8-gptq MTP3 is quality-ok for
  DSpark work. GRAPH=0 completed 164; GRAPH=1 CGRECLAIM=0
  did not. Speed work is allowed. SQ/AutoRound not forced.
Next pick: P0.2 W8A8 @262k MTP-off KV_FP8=0 (bf16 KV) then
  G1/Paris. First: stop current MTP3 serve, then
  B70_NOMTP=1 GRAPH=0 MAXLEN=262144 UTIL=0.90 TP=2 PORT=18080
  NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq
  ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
Do not: start P0.2 in this fire; start DD; retry GRAPH=1
  CGRECLAIM=0; enter Phase 2; train; overwrite w8a8-gptq;
  treat this as a DSpark number (it is MTP3 HE+ only).
Restore: DD stays PARKED. GRAPH=0 MTP3 serve left UP
  (next pick will replace it). Lease still held by 471943.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18i

---

## LOOP 6 -- 2026-08-18T0814Z -- P0.2 native 262k MTP-off + Paris

Picked: P0.2 -- W8A8 @262k MTP-off KV_FP8=0 (bf16 KV), G1
Why this, not the other open row: living-header Next pick
  after LOOP 5 GO. KV_FP8=0 first as specified.
GPU: lease HELD both cards by docker-wait pid=476139 since
  2026-08-18 08:14:12. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq` (root /models/qwen3.8-27b/w8a8-gptq,
  max_model_len 262144). NAME=qwen38_w8a8. GRAPH=0 MTP-off.
  P2P=0.
Command:
  NAME=qwen38_w8a8 bash vllm/w8a8/serve_qwen38_27b.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  B70_NOMTP=1 GRAPH=0 MAXLEN=262144 UTIL=0.90 TP=2 PORT=18080
    NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq KV_FP8=0
    ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop6_serve.log
  serve wrapper pid 476139 (file loop6_serve.pid)
Result: HEALTHY 198s. G0 id match max_model_len 262144.
  G1 Paris exact / 17*23=391 / fib iterative. Native 262k
  fits MTP-off UTIL=0.90 TP=2. KV_FP8 env is a no-op on
  the W8A8 3.6 serve.sh (bf16 KV).
Verdict: GO
Changed beliefs: 3.8 W8A8 native 262144 is real (not only
  229376). GRAPH=0 MTP-off loads coherent. KV_FP8 A/B
  needs a W8A8 serve hook before the fp8-KV half.
Next pick: P0.3 MTP3 @ longest ctx that fits, then
  bench_code c1. First: stop this MTP-off serve, then
  B70_NOMTP=0 MTPTOK=3 GRAPH=0 MAXLEN=131072 UTIL=0.90
  TP=2 PORT=18080 NAME=qwen38_w8a8
  SERVED=qwen3.8-27b-W8A8-gptq-mtp3
  ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
  Query /v1/models, G1, then bench_code c1. Push ctx
  only after c1 lands.
Do not: start P0.3/P0.4 in this fire; start DD; retry
  GRAPH=1 CGRECLAIM=0; enter Phase 2; train; overwrite
  w8a8-gptq; invent a KV_FP8 hook this fire.
Restore: DD stays PARKED. 262k MTP-off serve left UP
  (next pick replaces it). Lease held by 476139.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18j

---

## LOOP 7 -- 2026-08-18T0844Z -- P0.3 MTP3 GRAPH=0 bench_code c1 13.8

Picked: P0.3 -- MTP3 @131k GRAPH=0, G1, bench_code c1
Why this, not the other open row: living-header Next pick
  after LOOP 6 GO. Start 131k; do not push ctx this fire.
GPU: lease HELD both cards by docker-wait pid=479279 since
  2026-08-18 08:43:23. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-mtp3` (root /models/qwen3.8-27b/w8a8-gptq,
  max_model_len 131072). NAME=qwen38_w8a8. GRAPH=0 MTP3.
  P2P=0.
Command:
  NAME=qwen38_w8a8 bash vllm/w8a8/serve_qwen38_27b.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  B70_NOMTP=0 MTPTOK=3 GRAPH=0 MAXLEN=131072 UTIL=0.90 TP=2 PORT=18080
    NAME=qwen38_w8a8 SERVED=qwen3.8-27b-W8A8-gptq-mtp3
    ./bin/gpu-run bash vllm/w8a8/serve_qwen38_27b.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://192.168.10.5:18080/v1 qwen3.8-27b-W8A8-gptq-mtp3 1 256 3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop7_bench_code_c1.log
  serve wrapper pid 479279 (file loop7_serve.pid)
Result: HEALTHY 188s. G0 match. G1 Paris / 391 / fib.
  bench_code c1 avg **13.8** / best 15.3 t/s (out 256).
  MTP accept_len 2.39-3.06, pos0 0.69-0.88. Not < 2.0.
  13.8 < 26.62 (GRAPH=1 sweep TG). GRAPH=0 eager tax.
Verdict: NO-GO
Changed beliefs: GRAPH=0 MTP3 code c1 is ~14 t/s. Graft
  and MTP accept are fine. Beating 26.62 needs capture
  (not this pick). Pre-campaign 26.62 is still the best
  W8A8 speed. Do not treat 13.8 as a shelf number.
Next pick: P0.4 off-shelf DSpark on W8A8. First: write
  vllm/dflash/serve_qwen38_w8a8_dspark.sh (clone M1,
  target W8A8-gptq, method=dspark, THINK_BUDGET=0,
  SERVED=qwen3.8-27b-W8A8-gptq-dspark7). Then stop this
  MTP3 serve and start k=7 GRAPH=0, G1+G4.
Do not: start P0.4/P0.5 in this fire; start DD; retry
  GRAPH=1 CGRECLAIM=0; enter Phase 2; train; overwrite
  w8a8-gptq; method=dflash; publish 13.8 as a win.
Restore: DD stays PARKED. GRAPH=0 MTP3 serve left UP
  (next pick replaces it). Lease held by 479279.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18k

---

## LOOP 8 -- 2026-08-18T0915Z -- P0.4 W8A8 + DSpark k=7 accept table

Picked: P0.4 -- clone serve script, off-shelf DSpark k=7
  GRAPH=0, G1, G4
Why this, not the other open row: living-header Next pick
  after LOOP 7 NO-GO. Default order P0.4.
GPU: lease HELD both cards by docker-wait pid=482217 since
  2026-08-18 09:13:53. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark7` (root
  /models/qwen3.8-27b/w8a8-gptq, max_model_len 131072).
  NAME=qwen38_w8a8_dspark. GRAPH=0 method=dspark k=7.
  P2P=0.
Command:
  wrote vllm/dflash/serve_qwen38_w8a8_dspark.sh
  NAME=qwen38_w8a8 bash vllm/w8a8/serve_qwen38_27b.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  GRAPH=0 SPECTOK=7 MAXLEN=131072 PORT=18080 NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://192.168.10.5:18080/v1 qwen3.8-27b-W8A8-gptq-dspark7 1 256 3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop8_bench_code_c1.log
  serve wrapper pid 482217 (file loop8_serve.pid)
Result: HEALTHY 143s. G0 match. G1 Paris / 391 / fib.
  bench_code c1 avg **11.1** / best 12.1 (GRAPH=0).
  G4 mean accept_len **2.46** pos0 **0.62** (0.52-0.80).
  pos0 not < 30%. Train not forced.
Verdict: GO
Changed beliefs: off-shelf RadixArk/fp8-b70 DSpark accepts
  on W8A8 (pos0 ~62%, similar to NVFP4 58.4%). Ugly-accept
  train path is not this number. Speed gap vs MTP3 is
  GRAPH=0 + verify cost, not pos0. Do not method=dflash.
Next pick: P0.5 sglang 0.5.15 W8A8 3.8 NEXTN smoke
  (loads + Paris). Do not start it this fire.
Do not: start P0.5 / train / DD; method=dflash; retry
  GRAPH=1 CGRECLAIM=0; enter Phase 2; overwrite w8a8-gptq;
  publish 11.1 as a GRAPH=1-class win.
Restore: DD stays PARKED. DSpark serve left UP
  (next pick may replace it). Lease held by 482217.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18l

---

## LOOP 9 -- 2026-08-18T1728Z -- S0 DSpark k=7 GRAPH=1 bench_code c1 26.2

Picked: S0 -- DSpark k=7 GRAPH=1 short bench_code c1 (256x3)
Why this, not the other open row: living-header Next pick
  after operator 2026-08-18m. Speed hole is GRAPH=0.
GPU: lease HELD both cards by docker-wait pid=488659 since
  2026-08-18 17:26:36. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark7` (root
  /models/qwen3.8-27b/w8a8-gptq, max_model_len **122880**).
  NAME=qwen38_w8a8_dspark. GRAPH=1 PIECEWISE method=dspark
  k=7 CGRECLAIM=0 P2P=0.
Command:
  NAME=qwen38_w8a8_dspark bash vllm/dflash/serve_qwen38_w8a8_dspark.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  GRAPH=1 SPECTOK=7 MAXLEN=131072 ... start  -> KV OOM (D1)
  GRAPH=1 SPECTOK=7 MAXLEN=122880 PORT=18080 NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://192.168.10.5:18080/v1 qwen3.8-27b-W8A8-gptq-dspark7 1 256 3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop9_bench_code_c1.log
  131k crash: loop9_graph1_131k_crash.log
  serve wrapper pid 488659 (file loop9_serve.pid)
Result: 131k UTIL=0.90 ValueError KV 6.84 < 6.98 GiB (est
  max 128128). No DEVICE_LOST. Card 0 OK. Retry 122880
  HEALTHY ~150s, capture 1.28 GiB. G0 match. G1 Paris /
  391 / fib iterative. bench_code c1 avg **26.2** / best
  31.2 (out 256, wall ~10.5s). Bench-window accept 2.44
  pos0 0.59. G1 hold; no HE+.
Verdict: GO
Changed beliefs: GRAPH=1 is the DSpark speed hole
  (11.1 -> 26.2). 26.2 ~= MTP3 GRAPH=1 26.62, not a
  beat, not 41.2. 131k GRAPH=1 DSpark at UTIL=0.90
  does not load (D1). Do not HE+ under GRAPH=1.
Next pick: P0.5 sglang 0.5.15 W8A8 3.8 NEXTN smoke
  (loads + Paris). Do not start it this fire.
Do not: start P0.5 / train / DD; method=dflash; retry
  GRAPH=1 @131k UTIL=0.90; retry GRAPH=1 CGRECLAIM=0 as
  a long-eval fix; enter Phase 2; overwrite w8a8-gptq;
  publish 26.2 as beating MTP3 or 41.2.
Restore: DD stays PARKED. GRAPH=1 DSpark serve left UP
  (next pick replaces it). Lease held by 488659.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18n

---

## LOOP 10 -- 2026-08-18T1759Z -- P0.5 sglang 0.5.15 W8A8 3.8 NEXTN smoke

Picked: P0.5 -- sglang 0.5.15 W8A8 3.8 NEXTN (loads + Paris)
Why this, not the other open row: living-header Next pick
  after LOOP 9 S0 GO. Default speed order: P0.5 then leftover
  k-sweep.
GPU: lease HELD both cards by docker-wait pid=492514 since
  2026-08-18 17:56:32. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-nextn` (root same, max_model_len
  **8192**). NAME=qwen38_w8a8_sglang. IMG=sglang-xpu:mtp-0515
  NEXTN steps=10 draft=11 RADIX=1 PUSH_AR=1 P2P=0.
Command:
  NAME=qwen38_w8a8_dspark bash vllm/dflash/serve_qwen38_w8a8_dspark.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  PORT=18080 NAME=qwen38_w8a8_sglang MAXLEN=8192
    ./bin/gpu-run bash sglang/serve_qwen38_w8a8_0515.sh start
  chat G1 thinking-off Paris / 17*23 / fib
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop10_serve.log
  wait pid 492514 (file loop10_serve.pid)
Result: Loads Qwen3_5ForCausalLMMTP compressed-tensors.
  /health 200 after first gen (JIT). G0 match. G1
  thinking-off: Paris exact, 17*23=391, fib iterative.
  No DEVICE_LOST. No c1 this fire (smoke only).
Verdict: GO
Changed beliefs: 3.8 W8A8-gptq ports to sglang 0.5.15 +
  w8a8_shim + NEXTN without a new model class. Shim/ABI/GDN
  is not the P0.5 dead-end. First /health 503 is Triton JIT,
  not a load fail. DSpark on sglang-XPU stays Phase 3.
Next pick: leftover k=3/4/7 x greedy/prob. First command:
  NAME=qwen38_w8a8_sglang bash sglang/serve_qwen38_w8a8_0515.sh stop
  then GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
  NAME=qwen38_w8a8_dspark ./bin/gpu-run bash
  vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  (122880 not 131k -- D1). G1 then bench_code c1.
Do not: start k-sweep / train / DD this fire; method=dflash;
  retry GRAPH=1 @131k UTIL=0.90; enter Phase 2 or 3 DSpark
  port; overwrite w8a8-gptq; score this slot as MTP3/DSpark.
Restore: DD stays PARKED. sglang NEXTN serve left UP (next
  pick replaces it). Lease held by 492514.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18o

---

## LOOP 11 -- 2026-08-18T1826Z -- leftover k=3 GRAPH=1 G1 fail

Picked: leftover k-sweep first cell -- GRAPH=1 SPECTOK=3
  greedy @122880
Why this, not the other open row: living-header Next pick
  after LOOP 10. LOOP 10 named k=3 first.
GPU: lease HELD both cards by docker-wait pid=498559 since
  2026-08-18 18:26:03. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark3` (root
  /models/qwen3.8-27b/w8a8-gptq, max_model_len **122880**).
  NAME=qwen38_w8a8_dspark. GRAPH=0 (reverted) method=dspark
  k=3 CGRECLAIM=0 P2P=0.
Command:
  NAME=qwen38_w8a8_sglang bash sglang/serve_qwen38_w8a8_0515.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080 NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  # G1 fail "duct". revert:
  NAME=qwen38_w8a8_dspark bash vllm/dflash/serve_qwen38_w8a8_dspark.sh stop
  GRAPH=0 SPECTOK=3 MAXLEN=122880 ... start
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop11_serve.log
  fail: loop11_graph1_k3_g1fail.log
  revert: loop11_revert_graph0.log
  wait pid 498559 (file loop11_serve.pid)
Result: GRAPH=1 HEALTHY 147s. G0 match. G1 "duct" on
  chat and completions. accept_len 1.00 pos0 0.000.
  Loaded torch_compile_cache/b3f7e9e010 from k=7.
  No DEVICE_LOST. No c1 published. GRAPH=0 revert G1
  Paris / 391 / fib hold.
Verdict: NO-GO
Changed beliefs: GRAPH=1 k=3 is not safe on a k=7
  compile cache. "duct" + 0% accept is G1 fail, not
  a speed number. GRAPH=0 k=3 is coherent. Train is
  not forced (GRAPH=0 G1 holds). Packet D2.
Next pick: wipe
  /mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010
  then GRAPH=1 SPECTOK=3 MAXLEN=122880 G1 only. If G1
  holds, bench_code c1. If still duct, close k=3 GRAPH=1
  and do k=4.
Do not: publish a GRAPH=1 k=3 c1; retry GRAPH=1 without
  wiping that cache; retry GRAPH=1 @131k; method=dflash;
  train; start DD; enter Phase 2.
Restore: DD stays PARKED. GRAPH=0 k=3 serve left UP.
  Lease held by 498559. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18p

---

## LOOP 12 -- 2026-08-18T1854Z -- D2 retry: GRAPH=1 k=3 cold compile

Picked: D2 retry -- wipe b3f7e9e010, GRAPH=1 SPECTOK=3
  G1 only
Why this, not the other open row: living-header Next pick
  after LOOP 11 NO-GO. D2 Retry-if was now true.
GPU: lease HELD both cards by docker-wait pid=504031 since
  2026-08-18 18:54:36. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark3` GRAPH=0 (reverted)
  max_model_len **122880**. NAME=qwen38_w8a8_dspark.
Command:
  docker --entrypoint /bin/rm ... -rf
    /vllm_cache/torch_compile_cache/b3f7e9e010
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  # G1 still "duct". revert GRAPH=0. wipe hash again.
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop12_serve.log
  fail: loop12_graph1_k3_cold_g1fail.log
  revert: loop12_revert_graph0.log
  wait pid 504031 (file loop12_serve.pid)
Result: Cache wiped (host sudo no tty; docker rm worked).
  Cold compile 1.59s (not Directly load). HEALTHY 147s.
  G1 "duct" ct=32, accept_len 1.00 pos0 0.000. No
  DEVICE_LOST. No c1 published. GRAPH=0 revert G1 Paris
  / 391 / fib. Hash wiped again (shared across k).
Verdict: DEAD-END
Changed beliefs: GRAPH=1 k=3 is broken even cold.
  Compile hash b3f7e9e010 ignores SPECTOK. Do not retry
  GRAPH=1 k=3. Do not leave a k=3 graph on that hash.
  GRAPH=0 k=3 stays coherent. Train not forced.
Next pick: leftover k=4 GRAPH=1 G1 @122880. Confirm
  b3f7e9e010 is gone, then start. If G1 holds, bench_code
  c1. If duct, packet and stay GRAPH=0 / leftover prob.
Do not: retry GRAPH=1 k=3; publish a k=3 GRAPH=1 c1;
  retry GRAPH=1 @131k; method=dflash; train; start DD.
Restore: DD stays PARKED. GRAPH=0 k=3 serve left UP.
  Lease held by 504031. Hash b3f7e9e010 GONE.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18q

---

## LOOP 13 -- 2026-08-18T1921Z -- leftover k=4 GRAPH=1 c1 28.7

Picked: leftover k=4 GRAPH=1 G1 + bench_code c1 @122880
Why this, not the other open row: living-header Next pick
  after LOOP 12 D3. k=3 GRAPH=1 closed.
GPU: lease HELD both cards by docker-wait pid=506899 since
  2026-08-18 19:21:06. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark4` (root
  /models/qwen3.8-27b/w8a8-gptq, max_model_len **122880**).
  NAME=qwen38_w8a8_dspark. GRAPH=1 PIECEWISE method=dspark
  k=4 CGRECLAIM=0 P2P=0. Cold compile (hash was gone).
Command:
  NAME=qwen38_w8a8_dspark bash vllm/dflash/serve_qwen38_w8a8_dspark.sh stop
  ./bin/gpu-run --card 0 ./bin/xpu-health --card 0 --img vllm-xpu-env:int8g-v0260
  GRAPH=1 SPECTOK=4 MAXLEN=122880 PORT=18080 NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://192.168.10.5:18080/v1 qwen3.8-27b-W8A8-gptq-dspark4 1 256 3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop13_serve.log
  bench: loop13_bench_code_c1.log
  wait pid 506899 (file loop13_serve.pid)
Result: Hash was GONE. Cold compile 1.53s (not Directly
  load). HEALTHY 142s. G0 match. G1 Paris / 391 / fib.
  bench_code c1 avg **28.7** / best 31.2 (out 256,
  wall ~8.7s). Bench-window accept_len **2.45**, pos0
  **0.65** (0.58-0.71). Beats MTP3 26.62 and k=7
  GRAPH=1 26.2. Still < 41.2. No DEVICE_LOST.
Verdict: GO
Changed beliefs: GRAPH=1 k=4 is coherent; k=3 GRAPH=1
  fail is k-specific, not all k!=7. Compile hash still
  b3f7e9e010 (ignores SPECTOK) -- now holds k=4 graphs.
  Wipe before next k GRAPH=1. k=4 greedy beats MTP3.
  Train not forced.
Next pick: leftover k=4 GRAPH=1 probabilistic accept
  on this live serve (no restart). Then P1.5 / P1.7 /
  P1.6.
Do not: retry GRAPH=1 k=3; start k=7 GRAPH=1 without
  wiping hash; retry GRAPH=1 @131k; method=dflash;
  train; start DD; enter Phase 2.
Restore: DD stays PARKED. GRAPH=1 k=4 serve left UP.
  Lease held by 506899. Hash b3f7e9e010 is k=4 now.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18r

---

## LOOP 14 -- 2026-08-18T1949Z -- k=4 GRAPH=1 probabilistic accept

Picked: leftover k=4 GRAPH=1 prob accept table on live
  serve (no restart)
Why this, not the other open row: living-header Next pick
  after LOOP 13 GO. Completes P0.4 leftover greedy/prob.
GPU: lease HELD both cards by docker-wait pid=506899 since
  2026-08-18 19:21:06. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark4` GRAPH=1 k=4 @122880.
Command:
  inline chat temp=1.0 top_p=0.95 top_k=20 thinking-off
  3 coding prompts out=256; scrape vllm:spec_decode_*
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop14_prob_accept.log
Result: G0 match. G1 Paris exact. Window accept_len
  **3.16**, pos0 **0.80**, acc_rate 0.54 (drafts 246).
  Per-pos 0.80 / 0.57 / 0.45 / 0.34. Coherent code
  previews. Not the 20% band. No c1 republish (greedy
  28.7 stands). No DEVICE_LOST.
Verdict: GO
Changed beliefs: off-shelf DSpark accept on W8A8 is
  FINE under sampling too (pos0 80% > greedy 65%).
  Train is not forced. P0.4 leftover sweep is done.
  Next is kernels (P1.5), not more k or train.
Next pick: P1.5 small-M w8a16 default-on. First look
  at W8A16_M_MAX / int8_gemm_w8a16 and a serve flag.
  Live k=4 serve can stay until P1.5 needs a restart.
Do not: start P1.5 / train / DD this fire; retry
  GRAPH=1 k=3; wipe k=4 hash while this serve is up;
  method=dflash; enter Phase 2.
Restore: DD stays PARKED. GRAPH=1 k=4 serve left UP.
  Lease held by 506899. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18s

---

## LOOP 15 -- 2026-08-18T2024Z -- P1.5 W8A16_M_MAX=8 KV OOM

Picked: P1.5 small-M w8a16 default-on at long ctx
Why this, not the other open row: living-header Next pick
  after LOOP 14. L.4: kernels not train.
GPU: lease HELD both cards by docker-wait pid=512036 since
  2026-08-18 20:24:04. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark4` GRAPH=1 k=4 @122880
  (reverted W8A16_M_MAX=0).
Command:
  W8A16_M_MAX=8 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-w8a16
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  # OOM. revert W8A16_M_MAX=0 same k=4 GRAPH=1
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop15_w8a16.log
  oom: loop15_w8a16_oom.log
  revert: loop15_revert.log
  wait pid 512036
Result: W8A16 ON: model 27.31 GiB/card, available KV
  **-0.93 GiB**, ValueError no cache blocks. No
  DEVICE_LOST. No c1 published. Revert W8A16=0:
  HEALTHY 147s, G1 Paris exact. c1 28.7 stands.
Verdict: DEAD-END
Changed beliefs: any W8A16_M_MAX>0 clones NT s8
  weights. 122880 GRAPH=1 DSpark cannot pay that.
  Stay W8A16_M_MAX=0 at long ctx. Kernel TODO is
  s8s8-layout w8a16, not a UTIL bump.
Next pick: P1.7 push-AR on DSpark verify gather.
  First command: stay on this k=4 GRAPH=1 serve
  (PUSH_AR already default ON for large AR; check
  whether verify gather is already push or still
  oneCCL).
Do not: retry W8A16_M_MAX>0 @122880; UTIL>0.90 as
  a P1.5 fix; train; start DD; retry GRAPH=1 k=3.
Restore: DD stays PARKED. GRAPH=1 k=4 W8A16=0 serve
  left UP. Lease held by 512036.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18t

---

## LOOP 16 -- 2026-08-18T2053Z -- P1.7 ALLGATHER_ASYNC c1 29.4

Picked: P1.7 PUSH_AR_ALLGATHER_ASYNC=1 on k=4 GRAPH=1
Why this, not the other open row: living-header Next pick.
  Live serve had all_reduce push ON, gather still oneCCL.
  Host-barrier ALLGATHER=1 was 2.4x slower on NVFP4;
  ASYNC is the remaining gather lever.
GPU: lease HELD both cards by docker-wait pid=514764 since
  2026-08-18 20:52:29. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark4-agasync` GRAPH=1 k=4
  @122880. P2PACCESS=0. W8A16_M_MAX=0.
Command:
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1
  W8A16_M_MAX=0 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-agasync
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py ... 1 256 3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop16_agasync.log
  bench: loop16_bench_code_c1.log
  wait pid 514764
Result: ALLGATHER_ASYNC ENGAGED. HEALTHY 142s. G0 match.
  G1 Paris / 391 / fib. bench_code c1 avg **29.4** /
  best 33.2 (out 256, wall ~7.7s). Bench-window accept
  ~2.57 / pos0 ~0.65. No DEVICE_LOST. vs baseline
  28.7 / 31.2 wall 8.7s.
Verdict: GO
Changed beliefs: W8A8 DSpark k=4 verify gather is
  NOT the NVFP4 MTP 631-gather tax. Eager-async push
  on gather is slightly faster here, not 2.5x slower.
  Keep ALLGATHER_ASYNC on this recipe. Host-barrier
  ALLGATHER=1 still not retried (NVFP4 closed).
Next pick: P1.6 fusedq e2e. First look at B70_FUSEDQ /
  int8_gemm_w8a8_fusedq. Leave this serve up until
  P1.6 needs a restart.
Do not: start P1.6 / train / DD this fire; retry
  W8A16_M_MAX>0 @122880; P2P=1; method=dflash;
  retry GRAPH=1 k=3.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC serve
  left UP. Lease held by 514764.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18u

---

## LOOP 17 -- 2026-08-18T2128Z -- P1.6 fusedq e2e c1 28.3

Picked: P1.6 fusedq e2e on k=4 GRAPH=1 AGASYNC
Why this, not the other open row: living-header Next pick.
  Live v0260 overlay had no fusedq op. Mount v0240
  fusedq SO.
GPU: lease HELD both cards by docker-wait pid=522546 since
  2026-08-18 21:28:37. DD PARKED. :18080 id
  `qwen3.8-27b-W8A8-gptq-dspark4-agasync` (reverted)
  GRAPH=1 k=4 @122880. W8A16=0.
Command:
  GDN_SO=w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so
  B70_EXTRA_ENV="PUSH_AR_ALLGATHER_ASYNC=1 B70_FUSEDQ=1"
  GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-fusedq ... start
  bench_code 1 256 3
  # revert needed wipe b3f7e9e010 then AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop17_fusedq.log
  bench: loop17_bench_code_c1.log
  revert2: loop17_revert2.log
  wait pid 522546
Result: has_fusedq True. HEALTHY 142s. G1 Paris / 391
  / fib. c1 avg **28.3** / best 30.7 (wall 8.9s) vs
  AGASYNC **29.4** / 33.2. No DEVICE_LOST. Do not
  publish 28.3 as a win. Revert without wipe failed
  (cached fusedq op). Wipe + AGASYNC G1 Paris holds.
Verdict: NO-GO
Changed beliefs: v0240 fusedq SO loads on v0260 and
  is coherent, but decode c1 does not move up. Compile
  hash b3f7e9e010 stores fusedq graphs -- wipe when
  leaving that SO. Keep AGASYNC 29.4. P1.6 TTFT/PP
  not claimed.
Next pick: S1 SergiioB 3.8 GPTQ-Int4 MTP4 1xB70 smoke.
  AGASYNC serve stays up until S1 needs the cards
  (S1 is 1xB70; can use --card 0 and leave card 1,
  or stop this TP=2 first).
Do not: retry this fusedq SO for c1; retry W8A16>0
  @122880; train; start DD; enter Phase 2.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC serve
  left UP. Lease held by 522546. Hash rebuilt as
  non-fusedq. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18v

---

## LOOP 18 -- 2026-08-18T2153Z -- S1 3.8 GPTQ-Int4 MTP4 1xB70 smoke started

Picked: S1 -- SergiioB 3.8 GPTQ-Int4 MTP4 1xB70 smoke
Why this, not the other open row: living-header Next
  pick after LOOP 17 / P1.6 verdict. Cookbook after
  fusedq. S1 is --card 0.
GPU: lease HELD both cards by docker-wait pid=522546
  (AGASYNC still up during fetch). DD PARKED. :18080
  id qwen3.8-27b-W8A8-gptq-dspark4-agasync until
  fetch+pull finish.
Command:
  nohup bash vllm/cookbook_campaign/s1_qwen38_gptq_int4_smoke.sh
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop18_s1.log
  status: loop18_s1.status
  hf: loop18_hf.log  pull: loop18_pull.log
  s1 pid 523137 (file loop18_s1.pid)
Result: S1_STATUS=FETCH at start. Image was missing.
  Weights were missing (~19.6G rev 9d189a60). No
  G1 / phase_bench this fire.
Verdict: RUNNING
Changed beliefs: 3.8 cookbook pin is not on disk;
  must fetch f01e24f6 + 9d189a60 before the 1-card
  smoke. Do not mix with 2c427ef. launch.sh now has
  dense38-gptq (qwen3_xml, PUBLIC_IMAGE_38).
Next pick: S1 finish. First command: `ps -p 523137`.
  Live -> one status line, STOP. Dead -> write
  verdict from status/G1/bench, leave AGASYNC up,
  STOP. Do not start D/E in that fire.
Do not: enter Phase 2; pip-install 0.27; demote
  W8A8; publish 83.7 as our c1; start DD; train;
  retry fusedq SO; W8A16>0 @122880; mix digests.
Restore: DD stays PARKED. AGASYNC left UP during
  fetch. Script stops it only after artifacts
  exist, then restores AGASYNC after G1/bench.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18w

---

## LOOP 19 -- 2026-08-18T2220Z -- S1 finish post-first 47.58

Picked: S1 finish (pid 523137 dead, status DONE)
Why this, not the other open row: last verdict RUNNING;
  do not start a sibling.
GPU: lease HELD both cards by docker-wait pid=528521
  since 2026-08-18 22:12:54. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880
  (restored). S1 container gone.
Command:
  ps -p 523137; cat loop18_s1.status / loop18_g1.log
  / loop18_phase_bench.json
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop18_s1.log
  bench: loop18_phase_bench.json
  restore: loop18_restore.log
Result: G1 Paris / 391 / fib. phase_bench median
  post_first **47.58** tok/s (n=5/5, p~1150/g128).
  vs cookbook 83.7 and 08-10 3.6 dense 52.1. MTP
  pos0 0.80. AGASYNC restored, Paris exact. No
  DEVICE_LOST. Not a W8A8 c1.
Verdict: GO
Changed beliefs: 0.27.2rc1 digest f01e24f6 + pinned
  9d189a60 loads coherent on 1x B70. This box does
  not reproduce 83.7 at MAXSEQS=8 / entropy p~1150.
  47.58 is a C1 ceiling reference, not a reason to
  drop W8A8. Patches applied. qwen3_xml works.
Next pick: E1 xpu_shard_top1 A/B on this live
  AGASYNC serve (PRE.11 once). First look at the
  flag, then restart+G1+bench_code c1 vs 29.4.
  Do not start E1 this fire.
Do not: publish 47.58 or 83.7 as W8A8 c1; demote
  W8A8; enter Phase 2; pip-install 0.27; mix
  digests; train; start DD; retry fusedq SO;
  W8A16>0 @122880.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 528521.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18x

---

## LOOP 20 -- 2026-08-18T2255Z -- E1 shard_top1 flag is MTP-only

Picked: E1 xpu_shard_top1 A/B (PRE.11 once) -- look
  at the flag first, as LOOP 19 named.
Why this, not the other open row: living-header Next
  pick after LOOP 19 GO.
GPU: lease HELD both cards by docker-wait pid=528521
  since 2026-08-18 22:12:54. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
  No restart.
Command: inspect SpeculativeConfig, DSparkSpeculator,
  live torch.ops._xpu_C, host SOs. curl /v1/models +
  Paris.
Log: n/a (no GPU job). Packet D6.
Result: use_local_argmax_reduction unused by
  DSparkSpeculator (full-vocab compute_draft_logits
  + Markov + argmax). Live SO has no xpu_shard_top1.
  Proto SO at nvfp4_top1_proto has shard+int8.
  Paris exact. No c1 published. Serve left up.
Verdict: DEAD-END
Changed beliefs: PRE.11 NVFP4 path is MTP
  get_top_tokens, not DSpark. Do not flip that SPEC
  field on method=dspark. Real hook is
  _sample_sequential + proto SO.
Next pick: D6 hook -- overlay
  DSparkSpeculator._sample_sequential + GDN_SO=
  nvfp4_top1_proto, G1, bench_code vs 29.4.
  Do not start it this fire. Wipe compile hash on
  SO swap (D5).
Do not: restart just to add an ignored JSON field;
  swap proto SO without a speculator hook; demote
  W8A8; enter Phase 2; train; start DD; retry
  fusedq; W8A16>0 @122880.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 528521.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18y

---

## LOOP 21 -- 2026-08-18T2328Z -- D6 shard-top1 hook c1 28.4

Picked: D6 hook -- DSparkSpeculator shard-top1 +
  nvfp4_top1_proto SO, G1, bench_code vs 29.4
Why this, not the other open row: living-header Next
  pick after LOOP 20 D6. Retry-if was now true.
GPU: lease HELD both cards by docker-wait pid=536779
  since 2026-08-18 23:28:03 (reverted AGASYNC).
  DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
Command:
  GDN_SO=nvfp4_top1_proto ... shardtop1 start
  bench_code 1 256 3
  # revert: wipe b3f7e9e010 then AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop21_serve.log
  bench: loop21_bench_code_c1.log
  revert: loop21_revert.log
  wait pid 536779
Result: Hook ENGAGED. HEALTHY 163s. G1 Paris / 391
  / fib. c1 avg **28.4** / best 29.8 (wall 9.1s) vs
  AGASYNC **29.4** / 33.2. No DEVICE_LOST. Revert
  HEALTHY 137s, Paris exact.
Verdict: NO-GO
Changed beliefs: DSpark shard-top1 hook is
  coherent (G1 hold) but not a decode win. Keep
  AGASYNC 29.4. Overlay file stays off by default.
  Wipe hash when leaving proto SO (D5/D7).
Next pick: E2 host-barrier ALLGATHER=1 A/B on this
  live AGASYNC serve vs 29.4. Do not start E2 this
  fire.
Do not: remount proto SO for c1; retry SPEC flag
  on dspark; train; start DD; enter Phase 2;
  retry fusedq; W8A16>0 @122880.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 536779.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18z

---

## LOOP 22 -- 2026-08-18T2354Z -- E2 host-barrier ALLGATHER c1 26.6

Picked: E2 PUSH_AR_ALLGATHER=1 host-barrier A/B
  vs AGASYNC 29.4
Why this, not the other open row: living-header
  Next pick after LOOP 21. Remaining verify-AR.
GPU: lease HELD both cards by docker-wait pid=541759
  since 2026-08-18 23:54:34 (reverted AGASYNC).
  DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
Command:
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER=1 GRAPH=1 SPECTOK=4
    MAXLEN=122880 SERVED=...-aghost ... start
  bench_code 1 256 3
  # revert ASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop22_serve.log
  bench: loop22_bench_code_c1.log
  revert: loop22_revert.log
  wait pid 541759
Result: host-barrier ENGAGED. HEALTHY 132s. G1
  Paris / 391 / fib. c1 avg **26.6** / best 28.3
  (wall 9.0s) vs AGASYNC **29.4** / 33.2. No
  DEVICE_LOST. Revert HEALTHY 137s, Paris exact.
Verdict: NO-GO
Changed beliefs: host-barrier gather is slower
  on W8A8 DSpark too (26.6 < 29.4). Keep ASYNC.
  Both gather levers measured. Next is prefill
  (P4.1), not another gather flag.
Next pick: P4.1 prefix-cache TTFT baseline on
  this live AGASYNC serve. Do not start P4.1
  this fire.
Do not: retry ALLGATHER=1; remount proto SO;
  retry SPEC flag; train; start DD; enter Phase 2;
  retry fusedq; W8A16>0 @122880.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 541759.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-18aa

---

## LOOP 23 -- 2026-08-19T0342Z -- P4.1 prefix-cache TTFT 1528->449 ms

Picked: P4.1 -- prefix-cache TTFT baseline on
  k=4 GRAPH=1 AGASYNC @122880
Why this, not the other open row: living-header
  Next pick after LOOP 22 / post-reset. Default
  after hard reset. Last GPU serve died exit 255
  on reboot; systemd had brought DD back.
GPU: lease HELD both cards by docker-wait pid=9319
  since 2026-08-19 03:40:48. DD PARKED (official
  stop; systemd still enabled/active-exited).
  :18080 id qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880. P2PACCESS=0. W8A16_M_MAX=0.
Command:
  bash vllm/daily_driver_serve.sh stop
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1
  W8A16_M_MAX=0 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-agasync
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
  G1 thinking-off; fixed-prompt cold then warm TTFT
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop23_serve.log
  g1: loop23_g1.log
  ttft: loop23_ttft.log
  wait pid 9319
Result: HEALTHY 208s. G0 match. ALLGATHER_ASYNC
  ENGAGED. G1 Paris / 391 / fib. Prefix cache
  hits. IN=2040 cold TTFT **1528 ms** PP 1335
  (0 hits) -> warm **449 / 446 ms** PP 4544
  (1664 hits / 4080 queries over 2 warms).
  IN=8085 cold **2875 ms** PP 2813 -> warm
  **573 ms** PP 14122. 262k not measured
  (MAXLEN 122880). No DEVICE_LOST. No c1
  published this fire (already 29.4).
Verdict: GO
Changed beliefs: prefix cache is ON and HITS on
  W8A8 DSpark GRAPH=1 GDN (not the NVFP4 hits=0
  bug). Warm 2048 TTFT 449 ms vs DD NVFP4 347 ms.
  Campaign-table 262k TTFT still needs a
  different serve (P0.2 MTP-off). Prefill is
  not the remaining 29.4 vs 41.2 gap.
Next pick: no-GPU write of the 0.27-only feature
  list (PRE.15) from notes already on disk.
  Do not start it this fire.
Do not: start P4.2 / S2 INT4-AR / train / Phase 2
  / DD; retry D8-D4; W8A16>0 @122880; 262k on
  this GRAPH=1 DSpark recipe.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 9319.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19d

---

## LOOP 24 -- 2026-08-19T0403Z -- PRE.15 0.27-only feature list

Picked: no-GPU PRE.15 write + leftover W8A8 speed
  notes from disk
Why this, not the other open row: living-header
  Next pick after LOOP 23 GO. Last verdict not
  RUNNING. P4.1 already landed.
GPU: lease HELD both cards by docker-wait pid=9319
  since 2026-08-19 03:40:48. DD PARKED
  (b70_daily_0 Exited; systemd enabled
  active/exited). :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
  No restart. No GPU job.
Command: wrote
  docs/20260819_qwen38_027_only_features.md
  from campaign + SergiioB digest f01e24f6 +
  Steve 924b518 + LOOP 6/16-23. No fetch.
Log: n/a
Result: PRE.15 list exists. 0.27/2.13 features
  do not close 29.4 vs 41.2. Phase 2 stays
  closed. Leftover queue: E3 oneDNN barriers,
  compile-key, fusedq TTFT, sycl-tla, KV_FP8
  hook, 262k TTFT, G5. No c1 published.
Verdict: GO
Changed beliefs: PRE.15 is satisfied as a
  document and still does not unlock Phase 2.
  0.26 already serves the draft. Steve
  barriers / compile-cache / GDN scratch are
  0.26 steals, not 2.13 reasons. Local Steve
  clone is stale (03f98aaf).
Next pick: E3 oneDNN barriers-on A/B vs 29.4.
  First: git -C /mnt/vm_8tb/b70/b70-optimization-lab
  fetch, read 9f90e2c / GDN-scratch note, then
  restart this AGASYNC recipe with that env,
  G1, bench_code c1.
Do not: enter Phase 2; start S2 INT4-AR; train;
  start DD; P4.2; retry D8-D4; W8A16>0 @122880;
  method=dflash; overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 9319.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19e
