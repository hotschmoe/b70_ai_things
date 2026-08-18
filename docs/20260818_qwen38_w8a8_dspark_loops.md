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

P1.5 -- small-M w8a16 default-on (verify cheaper).
LOOP 14 GO: k=4 GRAPH=1 prob accept_len 3.16 / pos0 0.80
(not collapsed). Greedy c1 28.7 stands. Train not forced.
Hash b3f7e9e010 holds k=4 graphs -- wipe before any other
k GRAPH=1. Do not retry GRAPH=1 k=3 (D3). Do not start DD.
After P1.5: P1.7 / P1.6. Quality floor HE+ 0.957/0.927.
c1 28.7 < 41.2.

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
