# G4 -- exact-tail KV parking

Owner of vllm/flashnext/kvoffload/ after G1. Spec is
reports/2O2_kv_parking_spec.md. This file records what was measured.
H2D below 7.5 GB/s in the first T2 was swap_blocks_batch. The
direct-copy T2 is the later section.

## Verdict

T2 passed on Ornith, card 1, 2026-10-20 10:20. A real second turn
restored 3063 tokens from host RAM. The GPU prefix counter did not
move. Greedy text matched a never-evicted control. Restored TTFT was
0.175 s against a 0.417 s control prefill of the same prompt.

The first copy was ops.swap_blocks_batch. That load was 131235840
bytes in 0.063618 s, which is 2.063 GB/s. The same gate on
copy_batch, card 1 at 11:38, moved 131235840 bytes in 0.009678 s,
which is 13.560 GB/s. Restored TTFT was 0.124 s against a 0.414 s
control. The pre-shutdown store counter was 1369210880 bytes in
0.298174 s, which is 4.592 GB/s. T3,
FULL_DECODE_ONLY, 200K, and Flash-Next T6 are not measured yet.

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

- 7.5 GB/s in both directions. H2D clears it. D2H does not. See W8.
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

## W8 copy engine

Two runs on card 1. Health before and after both times: xpu-health
HEALTHY, exit 0. No xe-reset. Bandwidth is decimal GB/s. The timed
region is the sync wait on the copy queue after one warmup copy.
Bytes are checked every 4096 bytes. bad=0 on the run that copied.

### First run, no number

Library libb70_kvpark.so, 90904 bytes, 08:41, no `ordered` symbol.
Log g4-copy-driver.txt. COPY_START 10:40:42. malloc_host returned
USM host and the queue group was 1. The bench then raised
`ValueError: Overflow when unpacking long long` because the device
pointer does not fit in a signed int64 tensor
(`dev_ptr` on the next run was `0xffffd556b7400000`). gpu-run exit 1
in 15 s. The pointer pack is now a uint64 tensor viewed as int64.

### Second run, ordered library

Library libb70_kvpark_ordered.so, 95512 bytes, built 10:47.
`ordered()=1`, `engine_ordinal=1`. cpu-run --quiet around the
copies. Log g4-copy-driver-r2.txt. COPY_START 10:51:12. Host
buffers are sycl::malloc_host. One buffer per direction, then the
same bytes split into 1 MiB fragments.

| bytes | h2d | d2h | 1 MiB frags | swap h2d | swap d2h | bad |
| --- | --- | --- | --- | --- | --- | --- |
| 16777216 | 13.344 | 4.636 | 13.372 | 8.853 | 4.661 | 0 |
| 67108864 | 14.031 | 4.583 | 13.469 | 8.110 | 4.677 | 0 |
| 268435456 | 14.223 | 4.659 | 13.488 | 6.042 | 4.680 | 0 |
| 1073741824 | 14.037 | 4.670 | 13.442 | 4.119 | 4.656 | 0 |

worst one-buffer is 4.583 GB/s (D2H). COPY_BENCH printed. Process
exit 1 because the gate requires both directions at or above 7.5.
H2D is above that bar and above the 8.5 GB/s Strata H2D probe.
D2H matches swap_blocks_batch on the same USM-host pointer, so the
copy queue did not raise the device-to-host rate. swap H2D falls
as the buffer grows (the CPU staging copy). Our H2D stays near 14.

Health after 10:52:09, card 1 HEALTHY. The op is not wired into a
serve. B70_KVPARK_COPY stays 0.

### Host bias and compute stream

Library libb70_kvpark_flags.so, 106584 bytes, compile rc=0 at 11:00.
Same card, 256 MiB, four runs after one warmup. bad=0 on every arm.
Health before and after HEALTHY, exit 0, health after at 11:08:52.
FLAGS_DRIVER_EXIT 0 means the probe returned, not that D2H cleared
7.5 GB/s.

| path | h2d | d2h |
| --- | --- | --- |
| compute stream, malloc_host | 13.812 | 4.687 |
| copy queue, cached | 14.205 | 4.690 |
| copy queue, uncached | 14.014 | 4.663 |
| copy queue, write-combined | 14.165 | 4.649 |

D2H stays at the copy-queue malloc_host rate. The cap is not the
copy engine and not the host cache mode.

### mmap pool pointer

Same library and card, 256 MiB /dev/shm mapping, four runs.
`bench_copy: mmap bytes=268435456 h2d=9.435 d2h=4.451 bad=0`.
Health after 11:12:37, card 1 HEALTHY, exit 0. MMAP_DRIVER_EXIT 0
is the probe returning. H2D on the mapping clears 7.5 GB/s and is
below the 14 GB/s USM-host figure. D2H matches the other arms.

## T2 on copy_batch

CONFIG: card 1, port 18115, tag=copy, B70_KVPARK_COPY=1,
B70_KVPARK_SO=libb70_kvpark_flags.so. Same Ornith serve as the
swap-path T2 otherwise (45 GPU blocks, 32 GiB pool, slots 4,
offload_prompt_only false, shm 40g, memory 80g). Identity
hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
Health before HEALTHY. Log g4-t2-serve-c1-copy.txt and
g4-t2-turn-c1-copy.json.

The worker stderr during init is `b70_kvpark: copy queue group 1`.
The pool is still the pinned mmap (34.36 GB). The Python line
"G4 direct copy queue ordinal" is not in the serve log.

RESULT: pass=True. host_delta=3063, device_delta=0, computed=18,
cached=3063, greedy and text match. control TTFT 0.414 s, restored
TTFT 0.124 s, cold filler TTFT 4.906 s. The client scraped the
load counter around that one restore: 131235840 bytes in 0.009678 s
= 13.560 GB/s, and the engine log has load_size_count=1. The
pre-shutdown store counter, at 11:38:32, is 1369210880 bytes in
0.298174 s = 4.592 GB/s. The 11:38:42 scrape is 440729600 bytes.
That number is smaller because SIGTERM had already started and the
counters reset. It is not the run total.

H1c's M3 bench also posted to port 18115. The 11:38:12 engine line
says Running: 0. Our first tail was queued at 11:38:16 and the
restore ran at 11:38:28, so that bench was not in the restore step.
No EngineDead. gpu-run for the serve exited 0 after docker stop.
The engine process was force-killed at shutdown, which is the stop
path.

Health after did not run in that script. At 11:38:53 E2 took card 1
(vllm-fn-lowbit-e2). gpu-run waited 90 s and timed out. The file
g4-t2-health-after-c1-copy.txt is that timeout. The driver exit 124
was that timeout, and the retry was stopped so the pass would not
be run again. The later check, after E2 released the card, is
HEALTHY at 11:56:16 in g4-t2-health-after-c1-copy-late.txt.

## T3 ITL

CONFIG: card 1, port 18112, B70_KVPARK_COPY=1,
libb70_kvpark_flags.so, 45 GPU blocks, PIECEWISE, max_num_seqs 4.
Two 128-token streams decode while one parked conversation is restored.
Identity hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.
Health after both runs HEALTHY.

RESULT, run r2 (12:10, log g4-t3-itl-c1-r2.json): cached=3063,
computed=18, host_delta=3063, device_delta=0. Load 131235840 bytes in
0.009544 s = 13.750 GB/s, one load. Restored TTFT 0.144 s. Greedy
matched the solo control for 50 of 64 tokens, then repeated the
bigram (18, 2195) once. The control text is a short loop, and the
rest of the restored ids are that loop shifted by two. Run r1 had
the same cache counts and greedy_match false, without an index.
The solo T2 on this copy path matched all 64.

Decode gaps until the restored request's first token, milliseconds:
17.6, 17.9, 17.8, 24.1, 83.2, 35.5 on one stream and 17.7, 17.9,
17.8, 24.1, 83.0, 35.8 on the other. Baseline p50 was 17.4 ms.
The first three gaps are within 3 percent of that. The 83 ms gap is
the 18-token prefill step. p50 of the six gaps is 21.0 ms and p99
is 81 ms, so the literal 10 percent bar on that window fails.
The copy, at 9.5 ms, did not move the gaps that precede the prefill.

## T3 cycle

CONFIG: card 1, port 18113, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
32 GPU blocks, max_len 27000, PIECEWISE, max_num_seqs 4, 4 slots,
32 GiB host pool, shm 40g. Parks are 1488 prompt tokens plus a 32-token
reply. Identity hotschmoe-dev then ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.

RESULT r7 (13:03:58-13:08:51, logs g4-t3-cycle-driver-c1-r7.txt and
g4-t3-serve-c1-cycle-r7.txt): health before ok. At the first restore
the client had small_n=522 and big_n=0. The 24000-token prefill had
not emitted a token. Engine from 13:07:09 to 13:08:09: Running 1,
Waiting 2, KV cache 83.9 to 93.5 percent, prompt throughput 0 except
one 2399 tok/s window. At 13:08:12 the long request stored end=26799
(24000+2800-1) and the only load tail fired: end=1519 local=0 keys=5.
Turn 0 passed: host 1519, device 0, computed and cached match, greedy
match. Turns 1-3 were host 0 device 0. Turn 4 was host 1088 device 0.
Turns 5-7 were host 0 device 1088. No later load tail. Health after
rc 0. The client then raised NameError on active_n, after
CYCLE_RESULT was printed. That name is removed.

VERDICT: one parked tail restored from host under a live engine, and
the other seven turns did not. A prompt that fills the 32-block pool
is not scheduled beside the restore. The restore waits until that
request finishes, and the finished request takes one of the four host
slots.

RESULT r8 (13:16:13-13:19:50, logs g4-t3-cycle-driver-c1-r8.txt and
g4-t3-serve-c1-cycle-r8.txt): both actives were in decode before the
restores (17 and 8 tokens) and still running after them (287 and 296
tokens, streams_alive true). Engine during the first restores:
Running 3, KV cache 41.9 percent. Turns 0-3 passed. Each one loaded
tail end=1519 local=1088 keys=4, host 431, device 1088, computed and
cached match, greedy match. 431 + 1088 = 1519. Turns 4-7 sent a
different user span of the same length. The first visit had already
replaced the parked tail with end=1560, which is not a prefix of that
prompt. Those turns cached only the aligned 1088 block (computed 441,
host 0). Health after rc 0.

VERDICT: four parked conversations restored from host while two
requests decoded. The second visit must append the first reply. A
replay of a different same-length prompt cannot see the replaced tail.

RESULT r9 (13:22:45-13:26:22, logs g4-t3-cycle-driver-c1-r9.txt,
g4-t3-serve-c1-cycle-r9.txt, g4-t3-cycle-c1-r9.json): health before
ok, health after rc 0. Actives at the start 17 and 8 tokens; at the
end 289 and 298, streams_alive true. Engine Running 3, KV 41.9
percent. All eight turns loaded the tail (end 1519 then 1560, local
1088, keys 4). Host plus device equals the parked end on every turn:
431+1088=1519, then 472+1088=1560. Computed is 10 and cached matches
on every turn. TTFT 0.126 to 0.139 s. Greedy matched on turns 0-6.
Turn 7 matched 29 of 32 tokens of a 5-token loop, then index 29 was
40822 against the solo gold 14308. No xe-reset.

VERDICT: cache accounting passed for 4 parked conversations and 2
live decodes, including the extended second visit. Exact greedy
failed on the last three tokens of one turn. A bad restore would
diverge at token 0. This is the same late-loop drift seen in the
ITL window. The strict cycle gate is not met.

## FULL_DECODE_ONLY

CONFIG: card 1, port 18116, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
45 GPU blocks, max_len 32768, max_num_seqs 2, compilation mode 0,
cudagraph_mode FULL_DECODE_ONLY, capture sizes 1,2,3,4,8. The engine
resolved CUDAGraphMode.FULL_DECODE_ONLY (2, 0). Capture log:
"Capturing CUDA graphs (FULL): 2/2", finished in 1 s. Copy queue
group 1. Identity hotschmoe-dev then
ornith-1.5-35b-a3b-GPTQ-int4-mixedcal-v2.

COMMAND: bash /tmp/g4-launch-t2-graph.sh 1. Logs
g4-t2-graph-driver-c1.txt, g4-t2-serve-c1-graph.txt,
g4-t2-turn-c1-graph.json.

RESULT: health before ok. Health after HEALTHY, 12 s, rc 0.
pass=True. host=3063, device=0, computed=18, cached=3063, greedy
match, text match. Load tail end=3063 local=0 keys=6. Load
131235840 bytes in 0.009645 s = 13.607 GB/s. Restored TTFT 0.120 s.
Control TTFT 0.412 s. No xe-reset.

VERDICT: the real second turn passes on the full decode graph. The
host tail is restored before the new-turn prefill, and decode replay
uses that block table.

## Ornith 200K stand-in

Flash-Next is not serving. This is Ornith GPTQ int4, text
max_position_embeddings 262144, one card, TP=1. It is not the
Flash-Next 4-slot 200K target.

CONFIG: card 1, port 18117, B70_KVPARK_COPY=1, libb70_kvpark_flags.so,
NUM_GPU_BLOCKS=0 (profiled pool), max_len 204800, PIECEWISE,
max_num_seqs 2, 4 slots, 32 GiB host, shm 40g. P1=200000. Copy queue
group 1. KV cache size 257312 tokens. Maximum concurrency for 204800
tokens: 1.26x.

COMMAND: bash /tmp/g4-launch-t4.sh 1. Logs g4-t4-driver-c1.txt,
g4-t2-serve-c1-t4.txt, g4-t2-turn-c1-t4.json, g4-t4-mem-c1.txt.

RESULT: health before ok. Health after HEALTHY, 12 s, rc 0.
pass=True. end=200063, host=200063, device=0, computed=18,
cached=200063, greedy match, text match. Load tail end=200063
local=0 keys=187 (183 full keys plus a 959-token tail). Load
4164321280 bytes in 0.299164 s = 13.920 GB/s. Restored TTFT 0.833 s,
of which the copy is 0.299 s. Control TTFT 84.788 s for the same
200081-token prompt. Turn 1 prefill TTFT 84.339 s. The four slots
held the 200063 park, the 200144 control, and two 30464 fillers.
The cold fifth conversation evicted one slot at 13:40:26, after the
restore at 13:40:19. MemAvailable stayed at or above 78051432 kB
(74.4 GiB). Container RSS was not sampled. No xe-reset.

VERDICT: one Ornith conversation of 200063 tokens restored from host
RAM in 0.299 s, under the 0.42 s target, with replay 0 and greedy
equality. The copy is 13.920 GB/s. Client TTFT is 0.833 s because it
includes the 18 new tokens. This does not qualify the Flash-Next
200K case.

## Still open

1. D2H remains about 4.6 GB/s in the serve and in every microbench.
   H2D on the pinned mmap pool is 13.560 GB/s. The script default
   for B70_KVPARK_COPY stays 0.
2. T3 strict greedy on all eight turns. r9 missed three tokens at
   the end of turn 7. Cache counts passed on all eight.
3. Ornith 200063-token restore passed (0.299 s, 13.920 GB/s).
   Flash-Next 200K, four parked copies, and T6 stay closed until
   that model is serving.
4. F-D and T6 only after Flash-Next is serving and the XPU model file
   is free to edit. Coordinate with C7 in this journal. Do not push
   the fork.
