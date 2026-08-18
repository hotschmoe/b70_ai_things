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

P0.1 -- HE+ on grafted W8A8-gptq MTP3 (thinking-off, greedy, 164).
Yaml ids exist. First command: `bash vllm/daily_driver_serve.sh stop`
then serve `qwen3.8-27b-W8A8-gptq-mtp3` @131k. Multi-hour: start HE+,
write LOOP-STARTED, restore-DD after the job, Verdict RUNNING.

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
