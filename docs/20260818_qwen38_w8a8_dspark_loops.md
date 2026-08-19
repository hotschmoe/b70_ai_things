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

BARRIER=0 A/B on the same combined
GDN SO (isolate getenv vs rebuild).
Do not retry ALLGATHER_GRAPH as an
env-only fire (D17: 0 graph fires,
c1 27.9). Capturing DSpark verify
needs moving the gather into the
piecewise graph, not another flag.
Do not overlay 51MB GDN-OFF SO.
Do not P2P=1 / D16. Live hold k1bar
31.9.

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

---

## LOOP 25 -- 2026-08-19T0436Z -- E3 oneDNN barrier env absent

Picked: E3 -- Steve oneDNN barriers-on A/B vs
  AGASYNC 29.4
Why this, not the other open row: living-header
  Next pick after LOOP 24 GO. Last verdict not
  RUNNING.
GPU: lease HELD both cards by docker-wait pid=9319
  since 2026-08-19 03:40:48. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
  No restart.
Command:
  git -C /mnt/vm_8tb/b70/b70-optimization-lab fetch
  git show 9f90e2c3 + origin/main INT4-AR notes
  strings live vllm_xpu_kernels/_xpu_C.abi3.so
Log: n/a (no GPU job)
Result: Four flags named
  VLLM_XPU_ONEDNN_INT{4,8}_{COMPLETION_BARRIER,INPUT_DEPENDENCY}.
  Live int8g-v0260 _xpu_C has 0 ONEDNN_INT
  strings. Steve patch is xpu-kernels getenv
  around oneDNN execute, not an image env.
  No c1. Serve left up.
Verdict: DEAD-END (D9)
Changed beliefs: E3 is a kernel port, not a
  B70_EXTRA_ENV flip. Do not restart AGASYNC
  just to set those vars. INT4 flags are his
  INT4-AR SO; our path is INT8 completion
  barrier in kernels/int8_gemm_w8a8.h.
Next pick: compile-key SPECTOK+SO (no GPU).
  First: find v0260 compile cache key. Do not
  start the barrier kernel port this fire.
Do not: set unused ONEDNN_INT env; overlay
  Steve INT4-AR SO; enter Phase 2; start S2;
  train; start DD; P4.2; retry D8-D4.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 9319.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19f

---

## LOOP 26 -- 2026-08-19T0512Z -- compile-key SPECTOK+SO

Picked: leftover compile-key -- put SPECTOK +
  mounted _xpu_C SO in the 0.26 cache key
Why this, not the other open row: LOOP 25 Next
  pick named this leftover first. Operator 19h
  then YOLO-deferred it; this fire was already
  on the leftover. One arm.
GPU: lease HELD both cards by docker-wait pid=9319
  since 2026-08-19 03:40:48. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880.
  No restart.
Command:
  docker exec ... SpeculativeConfig.compute_hash
    dummy k=3 vs k=4 (stock same)
  install vllm/dflash/patches/v0260/compile_key_spectok_so.py
  selftest PASS
Log: n/a (no GPU job)
Result: Stock spec hash identical for k=3/k=4
  (c3d9d001b44dc00c). After hook they differ.
  _xpu_C sha256 74faead73d93... now in
  compile_factors. Serve script prepends
  /opt/compile_key_shim. Live workers unchanged.
Verdict: GO
Changed beliefs: b3f7e9e010 ignores SPECTOK
  because SpeculativeConfig.compute_hash omits
  num_speculative_tokens, and ignores GDN_SO
  because compiler_hash is inductor-only. Hook
  does not unstick D3 k=3 GRAPH=1 (cold duct).
  Next GRAPH=1 0.26 DSpark start gets a new
  hash; do not wipe b3f7e9e010 just to land
  this. S2a is on disk (19.02 GB auto-round).
Next pick: S2b 3.8 INT4-AR speed YOLO.
  First: stop NAME=qwen38_w8a8_dspark, xpu-health
  card 0, then
  TP=2 MTPTOK=5 GRAPH=1 PORT=18080 NAME=qwen38_int4ar
    SERVED=qwen3.8-27b-W4A16-autoround-mtp5
    ./bin/gpu-run bash vllm/w4a16/serve_qwen38_27b_int4ar.sh start
  G1 first. Then bench_code c1 + phase_bench
  after-TTFT. Default IMG f01e24f6. Do not
  stay on int8g-v0260.
Do not: start S2 this leftover fire (already
  not started); retry GRAPH=1 k=3; overlay
  Steve INT4 SO on W8A8; start DD; Phase 2;
  wipe b3f7e9e010 while AGASYNC is up.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease held by 9319.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19i

---

## LOOP 27 -- 2026-08-19T0611Z -- S2b INT4-AR 0.27 speed

Picked: S2b SPEED on public 0.27 `f01e24f6`. One arm.
Why this, not the other open row: living-header Next
  pick after LOOP 26; operator YOLO; S2a on disk.
GPU: card 0 HELD pid=28261 docker wait qwen38_int4ar.
  card 1 free. DD PARKED. :18080 id
  qwen3.8-27b-W4A16-autoround-mtp5 @16384
  IMG f01e24f6 (not int8g-v0260). P2PACCESS=0.
Command:
  NAME=qwen38_w8a8_dspark stop; xpu-health card 0
  TP=2 GRAPH=1 MTPTOK=5 ... start  -> D10
  TP=1 GRAPH=1 isolated-triton ... start -> G1 FAIL D11
  TP=1 GRAPH=0 isolated-triton ... start -> G1 PASS
  python3 vllm/nvfp4/bench_code.py ... 1 256 3
  python3 vllm/cookbook_campaign/phase_bench.py
    --prompt-tokens 512 --gen-tokens 128 --n 5
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop27_*
Result: TP=2 stock 2021.15 device_fd; host 2021.17
  overlay ImportError sycl8 vs nightly sycl9.
  GRAPH=1 TP=1 G1 garbage (倒 loops). GRAPH=0 TP=1
  G1 Paris/391/fib. bench_code c1 **12.8** / 12.8.
  after-TTFT median **16.66** tok/s TTFT 1.090s.
  MTP accept_len ~3.06 pos0 ~0.85 (phase_bench
  delta). Not 101.922. No DEVICE_LOST.
Verdict: GO (gated TP=1 GRAPH=0 speed). NO-GO on
  the 101.922 TP=2 GRAPH=1 cell (D10+D11).
Changed beliefs: 0.27 TP=2 needs a SYCL-9 oneCCL,
  not 2021.17-from-0.24. Isolated TRITON_CACHE
  (nightly is libsycl.so.9; shared 0.26 cache is
  .so.8). AutoRound mtp.layers are INT4; cookbook
  B70_MTP_BF16_DRAFT expects .weight and dies.
  GRAPH=1 on this nightly+ckpt is incoherent.
Next pick: S2c HE+ 164 thinking-off greedy
  seed=1234 on this live id. First:
  evals/.venv/bin/python -u evals/orchestrator/run_evals.py
    --endpoint http://192.168.10.5:18080/v1
    --model qwen3.8-27b-W4A16-autoround-mtp5
    --quant W4A16-autoround-mtp5 --tiers 1
    --tier1-dataset humaneval --limit 164
Do not: fake 101.922; retry D10/D11; overlay
  2021.17 on f01e24f6; B70_MTP_BF16_DRAFT on this
  ckpt; int8g-v0260; start DD; overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=0 TP=1 INT4-AR
  left UP for S2c. Lease card 0 pid 28261.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19j

---

## LOOP 28 -- 2026-08-19T0634Z -- S2c HE+ STARTED on INT4-AR GRAPH=0

Picked: S2c -- HE+ 164 thinking-off greedy seed=1234
  on live GRAPH=0 TP=1 id
  qwen3.8-27b-W4A16-autoround-mtp5
Why this, not the other open row: LOOP 27 Next pick;
  S2b speed verdict is on disk; last verdict not
  RUNNING. One arm. Multi-hour: start, STOP.
GPU: card 0 HELD pid=28261 docker wait qwen38_int4ar.
  card 1 free. DD PARKED. :18080 id
  qwen3.8-27b-W4A16-autoround-mtp5 @16384 root
  /models/qwen3.8-27b/int4-autoround. IMG f01e24f6.
Command:
  # G1 Paris PASS on the live serve
  evals/.venv/bin/python -u evals/orchestrator/run_evals.py
    --endpoint http://192.168.10.5:18080/v1
    --model qwen3.8-27b-W4A16-autoround-mtp5
    --quant W4A16-autoround-mtp5
    --tiers 1 --tier1-dataset humaneval --limit 164
    --seed 1234
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop28_heplus.log
  he+ pid 28816 (file loop28_heplus.pid)
  result dir:
    evals/results/20260819T063444Z__qwen3.8-27b-W4A16-autoround-mtp5__W4A16-autoround-mtp5
Result: G0 id match. G1 Paris exact still holds.
  HE+ generating 164 (thinking=off, greedy,
  seed=1234). Plus unmeasured.
Verdict: RUNNING
Changed beliefs: do not start a sibling HE+ or
  restore W8A8 while this pid is live. GRAPH=0
  0.27 INT4-AR still coherent after 31 min idle.
Next pick: S2c finish. First: `ps -p 28816`.
  Live -> one status line, STOP. Dead -> write
  plus vs 0.957/0.927 + fail lists, restore
  W8A8 AGASYNC unless plus < 0.90. How to tell
  done: pid 28816 gone AND log has pass@1 / plus
  (or a traceback).
Do not: start a sibling HE+; retry D10/D11;
  fake 101.922; start DD; overwrite w8a8-gptq;
  wait on this fire.
Restore: DD stays PARKED. INT4 GRAPH=0 serve
  left UP. Lease card 0 pid 28261.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19k

---

## LOOP 29 -- 2026-08-19T0743Z -- S2c HE+ 0.963/0.915 + restore AGASYNC

Picked: S2c finish (pid 28816 dead) + restore
  W8A8 AGASYNC
Why this, not the other open row: LOOP 28
  Verdict RUNNING; plus is now on disk.
GPU: after restore, lease HELD both cards
  pid=33246 docker wait qwen38_w8a8_dspark.
  DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880
  IMG=int8g-v0260. P2PACCESS=0.
Command:
  ps -p 28816  # GONE
  cat .../summary.json  # 0.963 / 0.915
  NAME=qwen38_int4ar ... stop
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1
    W8A16_M_MAX=0 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-agasync
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop28_heplus.log
  restore: loop29_restore.log
  result:
    evals/results/20260819T063444Z__qwen3.8-27b-W4A16-autoround-mtp5__W4A16-autoround-mtp5
Result: HE+ 164/164 thinking-off greedy seed=1234.
  pass@1 base **0.963** plus **0.915** (gen 2403s,
  eval 41s). vs W8A8 0.957/0.927 and Q4_K_M
  0.970/0.927. Base misses: 32, 91, 95, 116,
  140, 145. Plus-only: 39, 76, 97, 125, 141,
  151, 154, 163. plus 0.915 >= 0.90. Restore
  HEALTHY 229s, G1 Paris exact. No DEVICE_LOST.
Verdict: GO (S2c). S2 quality arm closed.
Changed beliefs: AutoRound INT4-AR 3.8 is a
  quality-ok W4A16 (base beats W8A8, plus 1.2
  pts under). Not a 101.922 vehicle on 0.27
  (D10/D11). Do not requant A.2-A.4 from this.
Next pick: leftover G5 18/18 on this live
  AGASYNC. Do not start G5 this fire.
Do not: retry D10/D11; fake 101.922; start DD;
  overwrite w8a8-gptq; enter Phase 2; start
  INT4 again unless operator YOLO.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease pid 33246.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19l

---

## LOOP 30 -- 2026-08-19T0805Z -- leftover G5 18/18 PASS

Picked: leftover G5 -- gate_concurrent_coherence
  3x6=18 on live AGASYNC
Why this, not the other open row: LOOP 29 Next
  pick. Last verdict GO not RUNNING. S2 closed.
GPU: lease HELD both cards pid=33246 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync @122880
  IMG=int8g-v0260. P2PACCESS=0. MAXSEQS=2.
Command:
  python3 -u vllm/gate_concurrent_coherence.py
    http://127.0.0.1:18080/v1
    qwen3.8-27b-W8A8-gptq-dspark4-agasync 3 6 200
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop30_g5.log
Result: G1 Paris exact. G5 **18/18 PASS** (OK=18,
  no GARBAGE/ERROR, no "!!!!"). Serve stayed up.
  No DEVICE_LOST. No new c1 (29.4 already G1-gated;
  now also G5-gated).
Verdict: GO
Changed beliefs: 3.8 W8A8 DSpark k=4 GRAPH=1
  AGASYNC @122880 is concurrent-coherent under
  mixed prefill+decode even at MAXSEQS=2 (queue
  piled). Do not treat "!!!!" as open on this
  recipe.
Next pick: leftover fusedq TTFT on this live
  serve. Do not start it this fire.
Do not: retry D5/D10/D11; fake 101.922; start DD;
  overwrite w8a8-gptq; enter Phase 2; start INT4.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease pid 33246.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19m

---

## LOOP 31 -- 2026-08-19T0834Z -- P1.6b fusedq TTFT skipped (D5)

Picked: leftover fusedq TTFT / P1.6b
Why this, not the other open row: LOOP 30 Next
  pick. Last verdict GO not RUNNING.
GPU: lease HELD both cards pid=33246. DD PARKED.
  :18080 id qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260. No restart.
Command:
  ls /mnt/vm_8tb/b70/w8a8_kernel_v0260_fusedq
  docker exec ... _xpu_C.abi3.so size
  # G1 Paris
Log: n/a (no GPU job)
Result: no v0260 fusedq tree. Live SO 61125200 B
  vs v0240 fusedq 61139960 B. D5 retry-if false.
  Did not remount v0240 SO. G1 Paris. No c1.
  P4.1 1528/449 stands.
Verdict: BLOCKED (D5 retry-if)
Changed beliefs: LOOP 26 compile-key does not
  unlock remounting the same v0240 fusedq SO.
  P1.6b stays closed until a v0260-ABI rebuild.
Next pick: leftover P1.8 sycl-tla C1. First:
  read kernels/SYCLTLA_SCAFFOLD.md, then
  microbench in v0240. Do not start this fire.
Do not: remount v0240 fusedq; retry D10/D11;
  fake 101.922; start DD; enter Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease pid 33246.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19n

---

## LOOP 32 -- 2026-08-19T0918Z -- P1.8 stock sycl-tla C1 NO-GO

Picked: leftover P1.8 sycl-tla C1 microbench
Why this, not the other open row: LOOP 31 Next
  pick. D5 retry-if still false. One arm.
GPU: stopped AGASYNC for card 0. Health OK.
  After restore, lease HELD pid=38888 docker
  wait qwen38_w8a8_dspark. DD PARKED. :18080
  id qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260.
Command:
  NAME=qwen38_w8a8_dspark stop
  ITERS=50 ./bin/gpu-run --card 0 bash
    /mnt/vm_8tb/b70/sycl-tla-bench/run_bench.sh
  # restore AGASYNC SPECTOK=4 GRAPH=1
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop32_sycltla.log
  restore: loop32_restore.log
Result: binaries AOT Jul 3 ran 51s. Stock bf16
  47-81% of 608 GB/s. Mixed bf16_s8 **1.1-1.5%**
  roof (M-flat). vs oneDNN W8A8 M=1 88-100% of
  581. Isolated not 1.2x. Restore HEALTHY 137s,
  G1 Paris. No DEVICE_LOST. No c1 (29.4).
Verdict: NO-GO (D12)
Changed beliefs: stock sycl-tla tiles are not
  the 29.4 vs 41.2 closer. Do not wrap them
  e2e. Rectangular TiledMMA is the retry-if.
Next pick: leftover B1 KV_FP8 hook. Do not
  start it this fire.
Do not: retry D12 stock tiles; remount fusedq;
  retry D10/D11; fake 101.922; start DD;
  overwrite w8a8-gptq; enter Phase 2.
Restore: DD stays PARKED. GRAPH=1 k=4 AGASYNC
  serve left UP. Lease pid 38888.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19o

---

## LOOP 33 -- 2026-08-19T0940Z -- B1 KV_FP8 hook G1 @131k

Picked: leftover B1 -- add KV_FP8 hook on 3.8
  W8A8 path, then KV_FP8=1 A/B G1 (capacity).
Why this, not the other open row: LOOP 32 Next
  pick. LOOP 6 KV_FP8 env was a no-op.
GPU: lease HELD pid=41598 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync-kvfp8
  max_model_len 131072 kv_cache_dtype fp8_e5m2
  IMG=int8g-v0260 GRAPH=1 SPECTOK=4.
Command:
  # hook in vllm/w8a8/serve_qwen38_27b.sh and
  # vllm/dflash/serve_qwen38_w8a8_dspark.sh
  KV_FP8=1 B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1
    GRAPH=1 SPECTOK=4 MAXLEN=131072
    SERVED=...-agasync-kvfp8 ... start
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop33_kvfp8_131k.log
Result: hook maps KV_FP8=1 -> --kv-cache-dtype
  fp8_e5m2. HEALTHY 137s. GPU KV **294,695**
  tok, 2.25x conc @131k, 8.12 GiB KV.
  G1 Paris / 391 / fib. No DEVICE_LOST. No
  c1 (29.4 is bf16 @122880). Uncalibrated
  fp8; HE+ not re-run.
Verdict: GO (capacity). Default KV_FP8 stays 0.
Changed beliefs: GRAPH=1 k=4 DSpark fits 131k
  with fp8_e5m2 (D1 was k=7 bf16 OOM). Hook
  is 3.8-wrapper only; 3.6 shelf untouched.
Next pick: leftover P4.1b 262k TTFT MTP-off.
  Do not start it this fire.
Do not: HE+ under GRAPH=1 KV_FP8; retry D12;
  remount fusedq; retry D10/D11; start DD;
  overwrite w8a8-gptq; enter Phase 2.
Restore: DD stays PARKED. KV_FP8=1 k=4 GRAPH=1
  @131k left UP. Lease pid 41598.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19p

---

## LOOP 34 -- 2026-08-19T1010Z -- P4.1b 262k MTP-off TTFT

Picked: leftover P4.1b -- 262k TTFT MTP-off
  KV_FP8=0, same cold/warm cells as P4.1.
Why this, not the other open row: LOOP 33 Next
  pick. Last startable leftover.
GPU: after restore, lease HELD pid=46901
  docker wait qwen38_w8a8_dspark. DD PARKED.
  During measure: id qwen3.8-27b-W8A8-gptq
  @262144 GRAPH=0 MTP-off. Restored
  qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880.
Command:
  B70_NOMTP=1 GRAPH=0 MAXLEN=262144 KV_FP8=0
    PREFIXCACHE=1 SERVED=qwen3.8-27b-W8A8-gptq
    bash vllm/w8a8/serve_qwen38_27b.sh start
  G1; cold+2x warm TTFT; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop34_serve.log
  ttft: loop34_ttft.log + loop34_ttft_matched.log
  restore: loop34_restore.log
Result: HEALTHY 148s. G1 Paris / 391 / fib.
  Clean-cold IN=1581 TTFT **978 ms** PP 1617
  (0 hits) -> warm **488 / 511 ms** PP 3238
  (1664 hits). vs P4.1 DSpark IN=2040 cold
  1528 / warm 449. IN=6261 cold 2637 (832
  prior hits) warm 417. Matched IN=2037/8085
  warms 389/476 (colds contaminated). Prefix
  hits. Restore HEALTHY 142s, G1 Paris.
  No DEVICE_LOST. No c1 (29.4).
Verdict: GO. Prefill still not the 29.4 vs
  41.2 gap. Leftover startable queue empty.
Changed beliefs: 262k MTP-off GRAPH=0 prefix
  cache hits; clean-cold 2048-class TTFT is
  faster than DSpark GRAPH=1 (978 vs 1528)
  and warm is similar (488 vs 449).
Next pick: no startable leftover. Verify
  cost remains. Retry-if only. Do not start
  Phase 2 this fire.
Do not: retry D10/D11/D12 stock; remount
  fusedq; HE+ under GRAPH=1; start DD;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4
  AGASYNC @122880 left UP. Lease pid 46901.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19q

---

## LOOP 35 -- 2026-08-19T1035Z -- idle; retry-if still false

Picked: leftover queue empty -- recheck
  Retry-if, do not invent work.
Why this, not the other open row: LOOP 34
  Next pick. Last verdict GO not RUNNING.
GPU: lease HELD pid=46901 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260. No restart.
Command:
  ls w8a8_kernel_v0260_fusedq  # missing
  strings live _xpu_C | ONEDNN_INT  # 0
  # G1 Paris
Log: n/a
Result: D5/D9/D10/D12 retry-if still false.
  G1 Paris. No c1. Serve left up.
Verdict: BLOCKED (no startable leftover)
Changed beliefs: do not start Phase 2 or
  remount closed SOs to keep the scheduler
  busy. 29.4 vs 41.2 needs a kernel retry-if.
Next pick: same. Retry-if only. Scheduler
  stays (29.4 < 41.2; S2c HE+ on disk).
Do not: invent a pick; retry D10/D11/D12
  stock; remount fusedq; start DD; enter
  Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4
  AGASYNC @122880 left UP. Lease pid 46901.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19r

---

## LOOP 36 -- 2026-08-19T1130Z -- S2b new nightly c48edf76

Picked: S2b SPEED on today's public nightly
  `c48edf76` (D10/D11 retry-if: new image).
  One arm. Not int8g-v0260.
Why this, not the other open row: operator
  YOLO S2b; LOOP 27 only smoked f01e24f6;
  leftover retry-if named a new 0.27 image.
GPU: lease HELD pid=55697 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260.
Command:
  docker pull vllm/vllm-openai-xpu@sha256:c48edf76...
  NAME=qwen38_w8a8_dspark stop; xpu-health
  TP=2 GRAPH=1 MTPTOK=5 isolated-triton start
  TP=1 GRAPH=1 isolated-triton start; G1
  restore AGASYNC k=4
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop36_*
Result: nightly is v0.26.1rc1.dev942+g5a4c8d992
  SYCL-9 + 2021.15. TP=2 device_fd (D10).
  TP=1 GRAPH=1 G1 garbage, accept 0 / pos0
  0.000 (D11). No 101.922 cell. Restore G1
  Paris. No DEVICE_LOST.
Verdict: NO-GO on 101.922. D10+D11 addenda.
Changed beliefs: today's nightly is not a
  SYCL-9 oneCCL fix. Steve 2025.3 oneCCL
  NEEDED libsycl.so.8 -- wrong ABI for these
  nightlies. 101.922 is his 0.21 / SYCL-8
  stack, not overlay-on-0.27.
Next pick: Steve graph-safe FA / 0.21
  SYCL-8 stack. First: build FA from
  experiments/qwen27_graphsafe_flash_attention
  inside qwen38-b70 (icpx 2025.3.3) or
  stand up his vLLM 0.21 vehicle. Then G1
  GRAPH=1 TP=2 on INT4-AR. Do not start
  that build this leftover writeup.
Do not: retry c48edf76/f01e24f6 TP=2;
  overlay host/qwen38-b70 2021.17 on SYCL-9;
  int8g-v0260 for INT4; fake 101.922;
  start DD; overwrite w8a8-gptq; Phase 2.
Restore: DD stays PARKED. GRAPH=1 k=4
  AGASYNC @122880 left UP. Lease pid 55697.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19s

---

## LOOP 37 -- 2026-08-19T1155Z -- S2b intel/vllm 0.21 TP=2

Picked: S2b on intel/vllm:0.21.0-xpu
  (torch 2.11, SYCL-8, in-image 2021.17).
  One arm. Not int8g-v0260.
Why this, not the other open row: LOOP 36
  Next pick Steve 0.21 / FA; FA binaries
  not on disk; this public 0.21 image is
  the SYCL-8 vehicle.
GPU: lease HELD pid=66009 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260.
Command:
  docker pull intel/vllm:0.21.0-xpu
  stop AGASYNC; xpu-health v0260
  TP=2 GRAPH=1 start_int4ar_intel021.sh
  G1 completions -> 500; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop37_*
Result: TP=2 loaded, no device_fd.
  Graph disabled (comms). First generate
  AssertionError gdn spec_sequence_masks
  is None. No 101.922. Restore G1 Paris.
  No DEVICE_LOST.
Verdict: NO-GO on 101.922. D13.
Changed beliefs: SYCL-8 + 2021.17 unsticks
  D10 load on this image. intel/vllm
  8df6feb7d GDN XPU op does not accept
  MTP spec_sequence_masks. Steve 101.922
  is 44fc8fde0 + FA, not this digest.
  GRAPH=1 TP=2 is disabled here anyway.
Next pick: Steve graph-safe FA / GDN spec
  kernels / vLLM 44fc8fde0. Build in
  qwen38-b70 (icpx 2025.3.3) then G1.
Do not: retry intel/vllm 0.21.0-xpu MTP
  generate; retry SYCL-9 nightlies; overlay
  SYCL-8 2021.17 on .so.9; int8g-v0260 for
  INT4; fake 101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4
  AGASYNC @122880 left UP. Lease pid 66009.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19t

---

## LOOP 38 -- 2026-08-19T1216Z -- D13 GDN spec fallback G1 fib bangs

Picked: D13 retry -- overlay Steve MTP
  GDN spec fallback on intel/vllm 0.21
  TP=2. One arm.
Why this, not the other open row: LOOP 37
  Next pick named GDN spec kernels / FA;
  the fallback is the exact assert fix
  and applies to this image without a
  44fc8fde0 rebuild.
GPU: lease HELD pid=71700 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 id
  qwen3.8-27b-W8A8-gptq-dspark4-agasync
  @122880 IMG=int8g-v0260.
Command:
  stop AGASYNC; xpu-health v0260
  CACHE_NAME=intel021_gdnfb start_int4ar_intel021.sh
  G1; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop38_*
Result: HEALTHY 161s. No GDN assert.
  Paris + 391 hold. Fib chat reasoning
  "!!!!" bangs. GRAPH disabled. No 101.922.
  Restore G1 Paris. No DEVICE_LOST.
Verdict: NO-GO on 101.922. D13 addendum.
Changed beliefs: Python fallback is not
  enough for G1-gated speed. Need Steve
  FA / 44fc8fde0. GRAPH=1 TP=2 still
  refuses comms on this image.
Next pick: build Steve graph-safe FA +
  vLLM 44fc8fde0 in qwen38-b70 (icpx
  2025.3.3). Then G1 GRAPH=1 TP=2.
Do not: retry this overlay as a speed
  cell; retry unpatched 8df6feb7d MTP;
  SYCL-9 nightlies; int8g-v0260 for INT4;
  fake 101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. GRAPH=1 k=4
  AGASYNC @122880 left UP. Lease pid 71700.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19u

---

## LOOP 39 -- 2026-08-19T1252Z -- 44fc8fde0 overlay + kernel build

Picked: S2b Steve vLLM 44fc8fde0 overlay
  on intel/vllm 0.21 + start 2dd55f38
  _xpu_C rebuild. One arm.
Why this, not the other open row: LOOP 38
  Next pick named 44fc8fde0 + FA. Clone
  finished; overlay is the serve attempt.
GPU: lease HELD pid=79276 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
  loop39_kbuild CPU-only.
Command:
  VLLM_SRC=... start_int4ar_intel021.sh
  docker run loop39_kbuild cmake _xpu_C
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop39_*
Result: FORCE_GRAPH PIECEWISE on. int4_gemm
  8 vs 7 ABI. Kernel build ~150/705.
  Restore G1 Paris. No DEVICE_LOST.
Verdict: RUNNING (kernel rebuild)
Changed beliefs: 44fc8fde0 already has
  FORCE_GRAPH_WITH_COMM. Image _xpu_C is
  7-arg int4; Steve kernels add
  input_dependency. Need rebuilt SO.
Next pick: finish kbuild. First:
  docker ps loop39_kbuild. Live -> STOP.
  Dead+SOs -> serve 44fc8fde0 + XPU_C_SO
  G1. Dead without SOs -> log/packet.
Do not: retry 44fc8fde0 without new SO;
  retry D13 overlay as speed; SYCL-9
  nightlies; int8g-v0260 INT4; fake
  101.922; start DD; Phase 2; overwrite
  w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 79276. kbuild running.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19v

---

## LOOP 40 -- 2026-08-19T1305Z -- kbuild live 295/705 STOP

Picked: LOOP 39 RUNNING finish -- status
  only. kbuild live so STOP.
Why this, not the other open row: LOOP 39
  Next pick: live -> one status line STOP.
GPU: lease HELD pid=79276 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080
  AGASYNC @122880 IMG=int8g-v0260.
  loop39_kbuild CPU-only still Up.
Command:
  docker inspect/logs loop39_kbuild
  G1 Paris
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop40_kbuild_snap.log
Result: kbuild 295/705 oneDNN gpu/intel.
  No SOs. G1 Paris. No c1.
Verdict: RUNNING
Changed beliefs: none. oneDNN compile is
  the long pole; do not kill it.
Next pick: same. First: docker ps
  loop39_kbuild. Live -> STOP. Dead+SOs
  -> 44fc8fde0 + XPU_C_SO G1.
Do not: wait this fire; overlay without
  SO; retry D13 overlay as speed; SYCL-9
  nightlies; int8g-v0260 INT4; fake
  101.922; start DD; Phase 2; overwrite
  w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 79276. kbuild left running.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19w

---

## LOOP 41 -- 2026-08-19T1335Z -- kbuild live 687/705 STOP

Picked: LOOP 39/40 RUNNING finish --
  status only. kbuild live so STOP.
Why this, not the other open row: ledger
  Next pick: live -> one status line STOP.
GPU: lease HELD pid=79276 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080
  AGASYNC @122880 IMG=int8g-v0260.
  loop39_kbuild still Up 46 min.
Command:
  docker inspect/logs loop39_kbuild
  G1 Paris
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop41_kbuild_snap.log
Result: kbuild 687/705; GDN compile
  started. No SOs. G1 Paris. No c1.
Verdict: RUNNING
Changed beliefs: none. Do not kill kbuild
  this close to the link.
Next pick: same. First: docker ps
  loop39_kbuild. Live -> STOP. Dead+SOs
  -> 44fc8fde0 + XPU_C_SO G1.
Do not: wait this fire; overlay without
  SO; retry D13 overlay as speed; SYCL-9
  nightlies; int8g-v0260 INT4; fake
  101.922; start DD; Phase 2; overwrite
  w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 79276. kbuild left running.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19x

---

## LOOP 42 -- 2026-08-19T1423Z -- 44fc8fde0+SO GRAPH=1 oneCCL graph fail

Picked: overlay 2dd55f38 _xpu_C+GDN on
  44fc8fde0 TP=2 GRAPH=1 FORCE_GRAPH G1.
Why this, not the other open row: LOOP 41
  Next pick; kbuild DEAD exit 0 with SOs.
GPU: lease HELD pid=90132 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  VLLM_SRC+XPU_C_SO+GDN_LIB start
  restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop42_*
Result: SOs 8-arg int4. First NameError
  MLARoPE pass; retry fuse=false compile
  156s then allgather sycl_graph fail
  (sched vs sycl_algorithms). No 101.922.
  Restore G1 Paris. No DEVICE_LOST.
Verdict: NO-GO (D14)
Changed beliefs: 2dd55f38 SO unsticks
  int4 ABI. In-image 2021.17 cannot
  record allgather in SYCL graphs.
  Steve 4ceafd1 oneCCL is the gap.
Next pick: build oneCCL 4ceafd1 then
  retry FORCE_GRAPH. First: Steve
  oneccl_ll256/build-public-oneccl.sh
  in qwen38-b70/intel/vllm (icpx 2025.3).
Do not: retry FORCE_GRAPH on stock
  2021.17; 44fc on 7-arg kernels; D13
  overlay as speed; SYCL-9 nightlies;
  int8g-v0260 INT4; fake 101.922; start
  DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 90132. SOs kept on disk.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19y

---

## LOOP 43 -- 2026-08-19T1438Z -- oneCCL 4ceafd1 rebuild STARTED

Picked: D14 retry-if -- build Steve
  public oneCCL 4ceafd1 vs SYCL-8 2025.3.
  CPU only. Leave AGASYNC up.
Why this, not the other open row: LOOP 42
  Next pick; FORCE_GRAPH on stock 2021.17
  is closed.
GPU: lease HELD pid=90132 docker wait
  qwen38_w8a8_dspark. DD PARKED. :18080
  AGASYNC @122880. loop43_cclbuild no GPU.
Command:
  clone b52f40c0 + libccl 4ceafd1; patch
  docker run loop43_cclbuild cmake install
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop43_cclbuild.log
Result: configure OK, ARCB ON, build
  ~47/260. G1 Paris. No c1.
Verdict: RUNNING
Changed beliefs: none yet. oneCCL compile
  is the D14 missing artifact.
Next pick: finish cclbuild. First:
  docker ps loop43_cclbuild. Live -> STOP.
  Dead+libccl.so.1.0 -> overlay FORCE_GRAPH
  G1. Dead without so -> log/packet.
Do not: wait this fire; FORCE_GRAPH on
  2021.17; 44fc on 7-arg kernels; D13
  overlay as speed; SYCL-9 nightlies;
  int8g-v0260 INT4; fake 101.922; start
  DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 90132. cclbuild running.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19z

---

## LOOP 44 -- 2026-08-19T1515Z -- 4ceafd1 overlay TP=2 device_fd

Picked: overlay Steve 4ceafd1 oneCCL on
  44fc8fde0+2dd55f38 SO FORCE_GRAPH G1.
Why this, not the other open row: LOOP 43
  Next pick; cclbuild DEAD exit 0 with so.
GPU: lease HELD pid=109648 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  CCL4CE+VLLM_SRC+XPU_C_SO start pidfd
  IPCX=sockets retry; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop44_*
Result: wrapper CCL_ROOT=/opt/ccl4ce.
  device_fd at worker init pidfd and
  sockets. No 101.922. Restore G1 Paris.
  No DEVICE_LOST.
Verdict: NO-GO (D15)
Changed beliefs: 4ceafd1 on this image
  loses the 2021.17 TP=2 init path.
  Graph-replay lib is not a drop-in.
Next pick: GRAPH=0 TP=2 44fc+SO+2021.17
  G1 then bench. Not a 101.922 cell.
Do not: retry 4ceafd1 overlay; FORCE_GRAPH
  on 2021.17; 44fc on 7-arg kernels; D13
  overlay as speed; SYCL-9 nightlies;
  int8g-v0260 INT4; fake 101.922; start
  DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 109648. 4ceafd1 so kept.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19aa

---

## LOOP 45 -- 2026-08-19T1535Z -- GRAPH=0 TP=2 44fc+SO c1 13.4

Picked: GRAPH=0 TP=2 44fc8fde0+2dd55f38
  SO + in-image 2021.17 G1 + bench.
Why this, not the other open row: LOOP 44
  Next pick; D14/D15 close GRAPH=1.
GPU: lease HELD pid=115282 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  GRAPH=0 no CCL4CE VLLM_SRC+SO start
  bench_code + phase_bench; restore
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop45_*
Result: G1 PASS. bench_code c1 **13.4**.
  phase_bench OOR 40 n_ok=2/5. No 101.922.
  Restore G1 Paris. No DEVICE_LOST.
Verdict: GO (gated speed) / NO-GO (101.922)
Changed beliefs: GRAPH=0 TP=2 on 44fc+SO
  +2021.17 is coherent. TP=2 without
  graph is ~TP=1 12.8. phase_bench 1k
  entropy prefill OOR at UTIL=0.88.
Next pick: CPU in-image 4ceafd1+kernels
  then GRAPH=1 G1. Leave AGASYNC up.
Do not: retry 4ceafd1 overlay; FORCE_GRAPH
  on 2021.17; GRAPH=0 as 101.922; D13
  overlay as speed; SYCL-9 nightlies;
  int8g-v0260 INT4; fake 101.922; start
  DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP.
  Lease pid 115282. SOs kept on disk.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ab

---

## LOOP 46 -- 2026-08-19T1643Z -- in-image s2b pid=host GRAPH=1 hang

Picked: bake 4ceafd1+2dd55f38+44fc into
  intel/vllm:0.21.0-xpu-s2b (CPU), then
  GRAPH=1 TP=2 G1. No bind overlay.
Why this, not the other open row: LOOP 45
  Next pick. D15 closed overlay.
GPU: lease pid=128045 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  bash vllm/w4a16/bake_intel021_s2b.sh
  stop AGASYNC; xpu-health
  IMG=intel/vllm:0.21.0-xpu-s2b BAKED=1
  GRAPH=1 TP=2 start; then pid=host retry
  restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop46_*
Result: bake OK (tag s2b). First GRAPH=1
  device_fd (pidfd unsupported -> drmfd).
  pid=host+seccomp unconfined: TP=2 loads,
  compile 154s, then capture hang 18+ min
  (workers 100% CPU, no new logs). No G1.
  No 101.922. Restore G1 Paris.
  No DEVICE_LOST.
Verdict: NO-GO (101.922). D15 retry-if
  consumed (pid=host). D16 capture hang.
Changed beliefs: 4ceafd1 in docker needs
  --pid=host --security-opt
  seccomp=unconfined. Bake != overlay
  does not fix device_fd. GRAPH=1 with
  4ceafd1 still does not reach G1.
Next pick: Steve graph-safe FA on baked
  s2b+pid=host GRAPH=1 G1.
Do not: retry overlay; bake without
  pid=host; wait out same hang; FORCE_GRAPH
  on 2021.17; GRAPH=0 as 101.922; D13
  overlay as speed; SYCL-9 nightlies;
  int8g-v0260 INT4; fake 101.922; start
  DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 128045.
  Image intel/vllm:0.21.0-xpu-s2b kept.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ac

---

## LOOP 47 -- 2026-08-19T1740Z -- graph-safe FA overlay GRAPH=1 hang

Picked: Steve graph-safe FA (local_accessor
  + force-chunk) on baked s2b+pid=host
  GRAPH=1 TP=2 G1.
Why this, not the other open row: LOOP 46
  Next pick; D16 retry-if is FA.
GPU: lease pid=135689 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  bash vllm/w4a16/build_graphsafe_fa.sh
  FA_DIR=.../fa-graphsafe/vllm_xpu_kernels
  IMG=intel/vllm:0.21.0-xpu-s2b BAKED=1
  CACHE_NAME=intel021_s2b_fa GRAPH=1 start
  restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop47_*
Result: FA --full OK (libattn 27.4 MB).
  FA2_FORCE_CHUNK=1. FlashAttention v2.
  Compile 159.64s then same hang 7+ min
  (workers 100% CPU, shm_broadcast, no
  capture log). No G1. No 101.922.
  Restore G1 Paris. No DEVICE_LOST.
Verdict: NO-GO (D16 addendum). FA is not
  the capture unstick.
Changed beliefs: stock vs graph-safe FA
  both hang after compile on this docker
  4ceafd1 GRAPH=1 path. Need a dump.
Next pick: capture-dump D16 hang (strace
  Worker_TP after compile).
Do not: retry FA as unstick; overlay
  4ceafd1; bake without pid=host; wait
  out hang; FORCE_GRAPH on 2021.17;
  GRAPH=0 as 101.922; int8g-v0260 INT4;
  fake 101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 135689.
  FA so kept. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ad

---

## LOOP 48 -- 2026-08-19T1814Z -- D16 strace dump after compile

Picked: capture-dump D16 hang (strace
  Worker_TP after torch.compile).
Why this, not the other open row: LOOP 47
  Next pick; D16 retry-if is dump.
GPU: lease pid=142782 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  IMG=s2b BAKED=1 CACHE=intel021_s2b
  GRAPH=1 pid=host; after compile
  CAP_PTRACE strace 12s; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop48_*
Result: compile cache 6-8s. Hang is
  userspace spin: 298904 sched_yield /
  12s, 18916 poll, 124 ioctl
  DRM_IOCTL_XE_EXEC_QUEUE_GET_PROPERTY
  on renderD128 AND renderD129, 4ceafd1
  mapped. No G1. No 101.922. Restore
  G1 Paris. No DEVICE_LOST.
Verdict: GO (dump). NO-GO 101.922.
Changed beliefs: D16 is XE exec-queue
  poll + yield, not a blocking syscall
  and not FA. Both ranks poll both cards.
Next pick: CCL_LOG_LEVEL=info name the
  collective; stop 60s post-compile.
Do not: wait hang; retry FA; overlay
  4ceafd1; P2P=1; FORCE_GRAPH on 2021.17;
  GRAPH=0 as 101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 142782.
  Strace kept. No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ae

---

## LOOP 49 -- 2026-08-19T1836Z -- CCL_LOG_LEVEL=info names silence

Picked: CCL_LOG_LEVEL=info on s2b+pid=host
  GRAPH=1; stop 60s post-compile.
Why this, not the other open row: LOOP 48
  Next pick; D16 retry-if is name coll.
GPU: lease pid=147490 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  CCL_LOG_LEVEL=info IMG=s2b BAKED=1
  CACHE=intel021_s2b GRAPH=1 start
  wait compile+60s; restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop49_*
Result: info ON. Compile 6.02s. Last CCL
  at compile: "no ports", in_order
  streams family7. Then 60s silence
  (shm_broadcast only). No per-coll
  name. ARC=0; fabric ports 0; ATL
  ofi/tcp. No G1. No 101.922. Restore
  G1 Paris. No DEVICE_LOST.
Verdict: GO (log). NO-GO 101.922.
  info cannot name the spinner.
Changed beliefs: do not set
  CCL_SYCL_ALLREDUCE_ARC=1 (Steve
  deadlock). Hang is after stream
  setup with no further CCL_INFO.
Next pick: COMPILE_ALLGATHER_CUSTOM_OP=1
  GRAPH=1 G1.
Do not: ARC=1; wait hang; retry FA;
  overlay; P2P=1; GRAPH=0 as 101.922;
  start DD; Phase 2; overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 147490.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19af

---

## LOOP 50 -- 2026-08-19T1911Z -- AGCUSTOM GRAPH=1 hang

Picked: VLLM_XPU_COMPILE_ALLGATHER_CUSTOM_OP=1
  on s2b+pid=host GRAPH=1 G1.
Why this, not the other open row: LOOP 49
  Next pick; Steve wait_tensor.
GPU: lease pid=152886 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  AGCUSTOM=1 CACHE=intel021_s2b_agcustom
  IMG=s2b GRAPH=1 start; stop if hang
  restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop50_*
Result: env ON (unknown-VLLM warning).
  New cache compile 156.54s then same
  D16 hang (workers 100% CPU,
  shm_broadcast, no /v1/models). No G1.
  No 101.922. Restore G1 Paris. No
  DEVICE_LOST.
Verdict: NO-GO (D16 addendum). AGCUSTOM
  is not the unstick.
Changed beliefs: docker GRAPH=1 with
  4ceafd1 still hangs after FA and
  AGCUSTOM. Next is host-not-docker.
Next pick: host-not-docker Steve venv
  GRAPH=1 G1.
Do not: retry AGCUSTOM; wait hang; FA;
  overlay; ARC=1; P2P=1; GRAPH=0 as
  101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 152886.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ag

---

## LOOP 51 -- 2026-08-19T1938Z -- HOSTNS privileged GRAPH=1 hang

Picked: privileged + host net/pid/cgroup
  GRAPH=1 G1 (no host Steve venv/oneAPI).
Why this, not the other open row: LOOP 50
  Next pick host-not-docker; no venv.
GPU: lease pid=157611 docker wait
  qwen38_w8a8_dspark after restore. DD
  PARKED. :18080 AGASYNC @122880.
Command:
  stop AGASYNC; xpu-health
  HOSTNS=1 CACHE=intel021_s2b IMG=s2b
  GRAPH=1 start; stop 60s post-compile
  restore AGASYNC
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop51_*
Result: compile 6.04s. Same D16 hang
  (workers 111% CPU, no /v1/models).
  Privileged hostns does not unstick.
  No G1. No 101.922. Restore G1 Paris.
  No DEVICE_LOST. No host venv exists.
Verdict: NO-GO (D16 addendum). Docker
  isolation is not the hang.
Changed beliefs: pid=host+privileged+
  host net/cgroup still hangs. True
  host PID needs extracted rootfs.
Next pick: CPU extract s2b rootfs then
  GRAPH=1 host PID.
Do not: retry HOSTNS; AGCUSTOM; FA;
  wait hang; ARC=1; P2P=1; overlay;
  GRAPH=0 as 101.922; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD stays PARKED. AGASYNC UP
  k=4 GRAPH=1 @122880. Lease pid 157611.
  No daily_driver_serve.sh start.
JOURNAL: ### 2026-08-19ah

---

## LOOP 52 -- 2026-08-19T2005Z -- pivot to INT8 kernel session

Picked: operator pivot. Stop 101.9 / D16
  overlay fires. Dedicated INT8 kernel
  session, then later W8A8 quality.
  HF check + Ornith-1.5 VRAM math.
Why this, not the other open row: leftover
  queue empty after L34; L35-51 were D16
  archaeology. Operator asked for kernels
  then quality, not another s2b hang.
GPU: none this fire. Lease still held by
  pid 157611 docker wait qwen38_w8a8_dspark
  (AGASYNC). DD PARKED.
Command: scheduler_delete 01a01813e05d;
  living-header + NEXT PICK retarget;
  HF API sizes for Freaksterz + Ornith-1.5.
Log: n/a (no GPU)
Result: 30m 101.9 scheduler gone. Next
  pick is K1 D9 barrier port. Freaksterz
  Qwen3.8-27B-SmoothQuant-W8A8-INT8 is
  the later quality A/B (plain
  compressed-tensors, ~30 GB, KLD 0.011).
  Ornith-1.5-35B-A3B-NVFP4 = 23.4 GB
  disk (fits 2x32). Ornith-1.5-397B-NVFP4
  = 238 GB disk (needs ~8x32 resident).
Verdict: GO (retarget). No number moved.
Changed beliefs: 101.922 is parked, not
  the next GPU job. Two more B70s do not
  land 397B NVFP4 GPU-resident.
Next pick: K1 D9 port into kernels/.
Do not: extract s2b; retry D16; start DD;
  Phase 2; overwrite w8a8-gptq; download
  397B this session.
Restore: DD PARKED. AGASYNC still up
  (untouched). Lease pid 157611.
JOURNAL: ### 2026-08-19ai

---

## LOOP 53 -- 2026-08-19T2008Z -- K1 D9 INT8 barrier port + SO

Picked: K1 port D9 completion-barrier
  getenv into kernels/int8_gemm_w8a8.h
  and w8a16.h; rebuild v0260 _xpu_C.
Why this, not the other open row: LOOP 52
  Next pick; 101.9 parked.
GPU: none. Lease pid 157611 AGASYNC.
  DD PARKED. :18080 AGASYNC @122880.
Command:
  edit kernels/int8_gemm_w8a{8,16}.h
  copy into vllm-xpu-kernels-w8a8
  docker loop53_kbuild int8g-v0260
  build_xpu_c.sh
Log: docker logs loop53_kbuild
Result: SO 51113216 B sha 9fa5d9a2
  strings have INT8_COMPLETION_BARRIER.
  Copy
  /mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so.k1barrier
  June baseline kept. AGASYNC untouched.
  No c1 (no GPU A/B this fire).
Verdict: GO (port + SO).
Changed beliefs: getenv is in the v0260
  ABI SO. GPU A/B still needed vs 29.4.
Next pick: overlay k1barrier + BARRIER=1,
  wipe compile hash, G1+bench vs 29.4.
Do not: overlay Steve INT4 SO; extract
  s2b; retry D16; start DD; Phase 2;
  overwrite w8a8-gptq.
Restore: DD PARKED. AGASYNC UP. Lease
  pid 157611. S2 scheduler deleted.
JOURNAL: ### 2026-08-19aj

---

## LOOP 54 -- 2026-08-19T2051Z -- K1 GDN+barrier c1 31.9

Picked: K1 GPU A/B. Loop 53 51MB SO
  lacked gdn_attention. Rebuilt combined
  GDN+int8+barrier vs int8g-v0260, then
  overlay + BARRIER=1 + AGASYNC.
Why this, not the other open row: T0
  of the 2-card TP docket. Overlaying
  the 51MB file would drop GDN.
GPU: lease free after AGASYNC stop;
  then pid 170431 docker wait. DD PARKED.
Command:
  docker w8a8gdn_v0260_k1 int8g-v0260
    GDN=ON + barrier headers
  GDN_SO=w8a8_kernel_v0260_k1barrier/_xpu_C.abi3.so
  GDN_LIB=.../libgdn_attn_kernels_xe_2.so
  B70_EXTRA_ENV="PUSH_AR_ALLGATHER_ASYNC=1
    VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER=1"
  GRAPH=1 SPECTOK=4 MAXLEN=122880 W8A16=0
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-k1bar
  bench_code c1 256x3
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop54_*
Result: SO 61145432 B sha 4ae45315.
  HEALTHY 198s. Barrier warn both ranks.
  AGASYNC ENGAGED. G1 Paris / 391 /
  iterative fib. c1 avg **31.9** / best
  34.0 (wall 7.5s) vs 29.4 / 33.2.
  No DEVICE_LOST.
Verdict: GO
Changed beliefs: D9 retry-if is done.
  Combined v0260 GDN SO + BARRIER=1 is
  the new hold. 51MB GDN-OFF file is
  not a vLLM overlay. Win may mix
  rebuild + getenv; BARRIER=0 isolates.
Next pick: T1 / P1.7 capture verify
  gather (D8 retry-if).
Do not: overlay 51MB k1barrier; Steve
  INT4 SO; D16; P2P=1; start DD; Phase 2.
Restore: DD PARKED. k1bar UP @122880.
  Lease pid 170431.
JOURNAL: ### 2026-08-19ak

---

## LOOP 55 -- 2026-08-19T2104Z -- T1 ALLGATHER_GRAPH c1 27.9

Picked: P1.7 / D8 retry-if -- record
  CSAG gather-internal SUM via
  ar_allreduce_graph when capturing.
Why this, not the other open row: T1
  of the 2-card TP docket after K1 GO.
GPU: lease after restore pid 175724.
  DD PARKED.
Command:
  PUSH_AR_ALLGATHER_GRAPH=1 + AGASYNC
  + BARRIER=1 same GDN SO GRAPH=1 k=4
  G1 + bench_code 256x3
  restore k1bar (no ALLGATHER_GRAPH)
Log: /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop55_*
Result: Wrap +captured-do_ar ON. Graph
  fire count **0**. G1 Paris / 391 / fib.
  c1 **27.9** / 31.0 vs 31.9 / 34.0.
  No DEVICE_LOST. Restore HEALTHY 147s
  Paris OK. id dspark4-k1bar.
Verdict: NO-GO (D17)
Changed beliefs: DSpark verify gather
  is still eager (LOOP 16). Env cannot
  put it in the graph. Keep 31.9 hold.
Next pick: BARRIER=0 control on this SO.
Do not: retry ALLGATHER_GRAPH env-only;
  51MB SO; P2P=1; D16; start DD.
Restore: DD PARKED. k1bar UP. Lease
  pid 175724.
JOURNAL: ### 2026-08-19al
