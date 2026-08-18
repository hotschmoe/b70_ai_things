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

P0.1 finish -- HE+ is RUNNING. First: `ps -p 467692` and
`tail /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop2_heplus.log`.
If live: one status line, STOP, no sibling. If done: write plus,
leave the W8A8 research serve up, STOP. Do not start DD. Do not
start P0.2 in that fire. Operator 2026-08-18f: DD stays PARKED.

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
