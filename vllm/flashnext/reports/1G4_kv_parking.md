# G4 -- exact-tail KV parking

Owner of vllm/flashnext/kvoffload/ after G1. Spec is
reports/2O2_kv_parking_spec.md. This file records what was measured.
Copy bandwidth below 7.5 GB/s is the swap_blocks_batch path.

## Verdict

T2 passed on Ornith, card 1, 2026-10-20 10:20. A real second turn
restored 3063 tokens from host RAM. The GPU prefix counter did not
move. Greedy text matched a never-evicted control. Restored TTFT was
0.175 s against a 0.417 s control prefill of the same prompt.

The copy was ops.swap_blocks_batch. The load was 131235840 bytes in
0.063618 s, which is 2.063 GB/s. That is not the copy-engine target.
W8, T3, FULL_DECODE_ONLY, 200K, and Flash-Next T6 are not measured yet.

## T2 configuration

- Card 1, one process, TP=1. bin/gpu-run --card 1 via
  harness/xpu-run.sh (serve_ornith.sh is the only lease holder).
- Model: models/files/ornith-1.5-35b-a3b/gptq-int4-mixedcal-v2.
- Identity from /v1/models: hotschmoe-dev, then
  ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
- Port 18111. dtype float16. language-model-only. max-model-len 32768.
  gpu-memory-utilization 0.90. num-gpu-blocks-override 45.
  max-num-seqs 2. prefix caching on. --park-finished-tail.
  compilation mode 0, PIECEWISE.
- Connector: ParkingOffloadConnector, kv_both, cpu_bytes 34359738368
  (32 GiB), slots 4, offload_prompt_only false. Container --shm-size
  40g, --memory 80g. Host mmap file was 34.36 GB, 1542 chunks.
- Fork /mnt/vm_8tb/b70/build/vllm-flashnext/vllm branch b70/flashnext,
  tip 9761b7b48a at record time. G4 commits, not pushed:
  c38d0a6197 (F-A, running-state block when the flag is set) and
  e442b4b53f (CLI registration of --park-finished-tail). Default off.
- Prompt: P1=3000, greedy R1=64, U2=17 raw ids. E = 3000+64-1 = 3063.
  Turn-2 prompt = 3081. Block size 1088, so the tail is 887 tokens
  past a 2176-token prefix. Gate: computed = 18, cached = 3063,
  host = 3063 minus device, replay 0, greedy match, restored TTFT
  below the same-prompt control.

## T2 command

```
/tmp/g4-launch-t2.sh
```

Attempt 2 ran:

```
CARD=1 vllm/flashnext/kvoffload/run_t2.sh
```

Health before and after: bin/xpu-health --card 1.
Logs: /mnt/vm_8tb/b70/build/vllm-flashnext/logs/g4-t2-driver-c1-r3.txt,
g4-t2-serve-c1-pass.txt, g4-t2-turn-c1.json.

Attempt 1 of the same launcher (09:58) waited 180 s, lost card 1 to
H1, and exited 124 before the server was ready. It did not score T2.

## T2 result

Health before ok. Ready 10:20:05. Engine line:

```
G4 parking: slots=4 chunks=1542 block=1088 exact_tail=True
groups=[(0, gdn, 1088), (1, gdn, 1088), (2, gdn, 1088),
(3, full_attention, 1088)] offload_prompt_only=False
```

Queued tails: turn 1 end=3063 tail=887 full_keys=2; control end=3144
tail=968 full_keys=2; fillers B and C end=30464 tail=0 full_keys=28.
Restore line: `G4 load tail end=3063 local=0 keys=6` (3 GDN state
blocks plus 3 attention blocks). The serve log has no
`G4 state group` warning. The dest index was (3063-1)//1088 = 2, and
that block was not already hashed.

| request | salt | prompt | cached | computed | ttft s | out |
| --- | --- | --- | --- | --- | --- | --- |
| turn 1 | park | 3000 | 0 | 3000 | 0.498 | 64 |
| control | control | 3081 | 0 | 3081 | 0.417 | 64 |
| B | fill-b | 30464 | 0 | 30464 | 4.935 | 1 |
| C | fill-c | 30464 | 0 | 30464 | 4.938 | 1 |
| restored | park | 3081 | 3063 | 18 | 0.175 | 64 |
| cold filler | cold | 30464 | 0 | 30464 | 4.934 | 1 |

JSON pass=True. host_delta=3063, device_delta=0. Greedy token ids and
text matched for 64 tokens. Both replies start
".\n\n<think>\nLet me look at this input carefully."
External prefix-cache hit rate on the engine log after the restore
was 3.0 percent because the fillers dominate the token count. The
per-request counters above are the gate.

Host load counter: 131235840 bytes, 0.063618 s, 2.063 GB/s.
First store window (turn 1 and control): 391249920 bytes, 0.089957 s,
4.35 GB/s. Both are swap_blocks_batch (H2D stages through a CPU
memcpy; D2H waits each pageable fragment).

Health after: xpu-health HEALTHY card 1, gpu-run 26 s, exit 0, at
10:22:06. No xe-reset. T2_DRIVER_EXIT 0.

## What T2 does not show

- 7.5 GB/s. The plugin libb70_kvpark.so is compiled (90904 bytes,
  icpx 2026.1.0) and is not wired into the worker. bench_copy.py has
  not run.
- Two streams decoding while a third is restored (T3), or an ITL stall.
- FULL_DECODE_ONLY capture with a parked tail.
- A 200K context. Ornith here was capped at 32768, and the GPU block
  override was 45 so the host hit could be forced.
- Flash-Next, PLE, or T6. F-D is not started. qwen4_exp was not edited.
- Four host slots evicting each other. A, control, B, and C all fit
  in the 32 GiB pool. A left the GPU because of the 45-block cap
  (device_delta 0), not because the host LRU dropped it.
- MTP. Eagle groups are a zero-size copy slot and were absent on this
  serve (speculative_config was off).

## Fixes already in the plugin

- offload_prompt_only is forced false, so decoded tokens are in the
  tail. E includes the reply except the unfed sampled token.
- The chosen tail lives on ParkingScheduler._tail_choice. An earlier
  serve (09:49, log g4-t2-serve-c1-r1.txt) died because
  RequestOffloadState is a slots dataclass and cannot take
  _g4_tail_key. Health after that crash was HEALTHY.
- GDN state is copied to block (end-1)//block, which is the slot
  allocate_external_computed_blocks fills and the worker reads.
- Whole-conversation policy is ConversationCachePolicy. Store failure
  discards the index record. A pin uses BlockPool.touch and
  free_blocks, under the name _pin_holds.
- An exact resend (end >= prompt length) does not offer the tail, so
  the scheduler's post-load minus-one backoff cannot double-apply GDN.
  T1 has not been run.
- F-B is not a scheduler change. scheduler.py already keeps a
  non-aligned ext_tokens count when the local hit has no partial tail.
  T2 is that case (local 0, E=3063). The post-load minus-one backoff
  was not touched. Commit 765993c197 adds
  test_kv_connector_keeps_non_aligned_external_hit. A CPU run of the
  same numbers (no pytest in the venv) got scheduled=41, connector
  hits=23, on a 64-token prompt with a 23-token external hit.
- F-C is not committed. The deferred tail store completed before the
  restore lookup (host 3063, device 0).

## Still open

1. W8 bench of copy_batch versus swap_blocks_batch, 16 MiB to 1 GiB,
   on a CPU-quiet card, with xpu-health before and after.
2. Wire the op on the copy queue, with a compute-queue order, behind
   a fallback to swap_blocks_batch. Re-check T2 on that path.
3. T3: two decoding streams plus one restore, ITL p50/p99 stall at
   most 10 percent, then 4 parked and 2 active.
4. FULL_DECODE_ONLY.
5. 200K-class context once Flash-Next is serving. Otherwise the
   largest Ornith context, and say so.
6. F-D and T6 only after Flash-Next is serving and the XPU model file
   is free to edit. Coordinate with C7 in this journal. Do not push
   the fork.
