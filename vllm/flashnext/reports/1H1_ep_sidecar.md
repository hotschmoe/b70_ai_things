# H1 EP sidecar (b70_ep_xpu)

Owner: H1 / H1b / H1c. Package: `vllm/flashnext/plugins/b70_ep_xpu`.
vLLM stays one rank on card 0. The plugin owns card 1 and its expert half.
Transport is a same-context peer store plus device flags. No oneCCL.
Design: `reports/2O1_ep_design.md` section 6, milestones M0-M3.

Library: `b70_ep_xpu/_C.so`. M0 and M1 used sha256
`bd4f4aa010c718daa6a2c983e8a93670acdd2e4df1f094c0819ed68517c39594`
(476912 bytes, built 2026-10-20 09:38). M2 uses sha256
`7f0981fd20a2375965fd89612bb32969a71ca6906ef94bfa9a6d1d0e4926f010`
(476912 bytes, built 2026-10-20 10:53), which host-waits both queues
at the end of an eager `ep_moe`. Capture still records the pipelined
chain. M3 uses sha256
`e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870`
(481240 bytes, built 2026-10-20 12:07). A recording card-0 stream
records the card-1 graph, skips that host wait, and finalizes the
card-1 graph on the last MoE layer. Slotted GGUF library
`b70_gguf_xpu/_C.ep.so` sha256
`55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0`
(not rebuilt).

## M0 -- transport oracle

Status: pass (2026-10-20 09:50).

CONFIG: both cards, path 2, spin 200000, container `vllm-fn-h1c-m0`,
memory 8g. Init warmup touches every peer page in both directions,
sys-stores the flag pages, runs one dummy push/return, then resets
flag epochs before any scored sample.

COMMAND: `bash /tmp/h1c-m0-window.sh` (health, then
`harness/xpu-run.sh --card both`, then health). Logs
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1c-m0-window.out` and
`h1c-m0-probe.log`.

RESULT:

- IPC open in torch's shared context fails
  (`zeMemOpenIpcHandle` 2013265924). Same-context peer store works.
  `same_store=1 path=2`. Peer atomics are 0. BDFs `0000:0b:00.0` and
  `0000:44:00.0`.
- Preflight stores match from 64 B through 64 KB (card 0 to 1) and
  614400 B (card 1 to 0).
- Warmup: touch inbox/parts/flag1/flag0 all 1, flag readback 1/1,
  exchange attempt 1 with err0=0 err1=0, epochs reset.
- Eager oracle 65/65. Sample s=0 (first scored use) is
  push_bad=0 ret_bad=0 err0=0 err1=0. H1b's s=0 timeout is gone.
- Graph replays 65/65. Capture itself does not execute
  (`capture_executed=0`); the capture check of s=0 still shows the
  poison counts. The replay of s=0 is clean.
- Mirror arm (default-scope barrier, not a gate) still shows a
  partial push: push_bad 1700, 1596, 1566 with err=0. The flag is
  visible before every word. Left as a sensitivity control.
- Flag one-way p50 1.771 us both directions (min 1.667, max 1.980).
- 64 KB push, 64 groups: p50 5.834 us, 11.233 GB/s.
- Concurrent 64 KB: card0 5.834 us, card1 5.833 us, host wall 26.631 us.
- Uncached and plain consumer loads of 10240 B: both bad=0.
- Health before and after: HEALTHY, both cards. No xe-reset.
  PROBE_RC=0. Window 09:46:03 to 09:50:40 once the lease was held
  (queued from 09:39 behind D3).

VERDICT: M0 gate met. Handshake is under 5 us each way. The 64 KB
payload is 5.8 us, which is the data move, not the flag.

## M1 -- one real layer

Status: pass (2026-10-20 10:21). Log `h1c-m1-layer-r2.log`.

CONFIG: blk.0 of the 35B UD-IQ3_XXS GGUF, E=256 split 128/128,
topk=8, T=1,2,4,8,4096. Gate: bitwise `y_ep == y0+y1`, cosine > 0.999,
and max|y_ep - y_full| / max|y_full| < 1e-2.

RESULT: every T is bitwise (`own_max=0`) with cosine 1. Scale-relative
error is 5.9e-8, 6.6e-8, 1.4e-7, 1.2e-7, 8.9e-8. Absolute max at
T=4096 is 3.0e-8. The per-element ratio at T=4096 is 7.5e-2 because a
few near-zero outputs are divided by 1e-8; that ratio is logged as
`elem_rel` and is not the gate. An earlier run (r1) failed only that
ratio and then rebound the devices. r2 does not. Layer time: T=1 EP
500 us vs one-card 410 us; T=8 EP 592 us vs one-card 818 us.
`PASS m1`, M1_RC=0. Health before and after HEALTHY (after rc 0 at
10:40:23). No xe-reset on this run.

VERDICT: the split matches one-card `moe_layer` inside FP32 reorder.

## M2 -- eager serve

Status: pass (2026-10-20 11:12). Binary sha256 `7f0981fd`.

CONFIG: 35B UD-IQ3_XXS, `B70_EP=1`, world size 1, sidecar owns card 1,
eager, port 18114, container `vllm-fn-h1-m2`, max_len 2048, spin
200000, path 2. Greedy gate is the C4/C7 GGUF string, temperature 0,
16 tokens, prompt "The capital of France is":

` Paris.\nThe capital of France is Paris.\nThe capital of France is`

COMMAND: `bash /tmp/h1c-m2-window.sh`. Logs
`h1c-m2-window.out`, `h1c-m2-serve.log`, `h1c-m2-greedy.json`,
`h1c-m2-bench.json`. r1 is kept as `h1c-m2-serve-r1.log` and
`h1c-m2-window-r1.out`.

RESULT:

- r1 (sha `bd4f4aa0`, 10:42-10:48) split all 40 MoE layers and finished
  engine init. The greedy request then died: layer 1 `err1=11` because
  card 1's next wait was queued through card 0's expert compute.
  Health after HEALTHY. No xe-reset.
- r2 rebuild host-waits both queues at the end of eager `ep_moe` and
  throws on that call if a flag timed out. Warmup moe attempt 1
  layer 0 err0=0 err1=0. Split layer 39 E=256 n0=128 types=21,21,23.
  No `EP flag timeout`. READY 11:10:45.
- Both greedy texts equal the C4 loop string. `same=true`, errors
  empty. The JSON `match_c4` flag is false only because the window
  script still compared against the int3/int4 landmarks sentence.
- Identity: first `hotschmoe-dev`, alias
  `qwen3.6-35b-a3b-ud-iq3-xxs`. Coherence ok, same, paris.
- Eager C1 ITL p50 81.84 ms, p90 83.19, p99 83.78, 12.60 tok/s,
  TTFT 217.93 ms, n=32. C2 both streams ITL p50 87.33 ms. C4 eager
  one-card C1 was 87.62 ms. This number is recorded, not the M2 gate.
- Health before HEALTHY (11:07:45, queued behind E2 from 10:53).
  Health after HEALTHY rc 0 at 11:12:23. SERVE_RC=0. No xe-reset.

VERDICT: M2 text and identity gates met. The eager host wait is not
in the captured graph. M3 is the speed gate.

## M3 -- FULL_DECODE_ONLY

Status: pass (2026-10-20 12:57).

CONFIG: both cards, port 18115, container `vllm-fn-h1-m3`, `B70_EP=1`,
`GRAPH=1`, `GRAPH_MODE=FULL_DECODE_ONLY`, `ATTN=FLASH_ATTN`,
`MAX_LEN=33792`, `MAX_SEQS=4`, `GPU_MEM=0.90`, `B70_XPU_CG_RECLAIM=500`,
capture sizes `[1,2,3,4,8]`. `_C.so` sha256
`e783539c4fa49f774a115a816209201a189a78d8d2c051f5b56c30575eb2c870`.
`_C.ep.so` sha256
`55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0`.
Text gate is the C4 loop string. Speed gate: C1 ITL at most 1.05x the
one-card 11.11 ms (11.665 ms). C2 is recorded against 16.51 / 16.65 ms
and against C7 stock DP=2 eager ITL 121.40 ms.

COMMAND: `bash /tmp/h1c-m3-window.sh` PID 3286770. Logs
`h1c-m3-window-r2.out`, `h1c-m3-serve.log`, `h1c-m3-bench.json`,
`h1c-m3-greedy.json`.

Two earlier launches are not this measurement. The 11:38 window
benched G4's Ornith server on port 18115 (`alias_ok=False`). Those
ITL numbers are discarded (`h1c-m3-*-falseport.*`). The 11:44 window
loaded the model, then FULL capture raised Level Zero error 44 inside
`moe_ep` because the eager host wait ran on the recording stream
(`h1c-m3-serve-capture44.log`). No DEVICE_LOST. That window's
health-after finished HEALTHY at 12:53:25 (`H1c M3 done rc=1`).
No xe-reset.

RESULT:

- Queued from 12:12:49. Health-before HEALTHY, rc 0, at 12:55:14.
- EP init path=2, spin=200000, reclaim=500. Warmup exchange and moe
  attempt 1 both err0=0 err1=0. Split layers 0-39.
- Profile capture finalized card-1 graphs for keys 4 and 3. The real
  FULL capture then finalized keys 4, 3, 2, and 1. No key=8 line.
  Capture finished in 1 s, pool 0.03 GiB. Same 4-graph set as the
  one-card C7 run. No error 44. No EP flag timeout.
- KV cache 14.72 GiB, 630487 tokens, 18.66x at 33792.
- READY 12:56:40. Identity `hotschmoe-dev` then
  `qwen3.6-35b-a3b-ud-iq3-xxs`. `first_ok=True`, `alias_ok=True`.
- Greedy temperature 0, 16 tokens, both runs:
  ` Paris.\nThe capital of France is Paris.\nThe capital of France is`.
  `match_c4=true`. Bench coherence OK/OK, same string.
- C1 ITL p50 11.36 ms, p90 11.62, p99 12.35, 89.50 tok/s, TTFT
  142.82 ms, completion_tokens 63 (requested 64). 11.36 / 11.11 =
  1.022, under the 11.665 ms cap.
- C2 stream0 ITL p50 15.31 ms, p90 16.06, p99 16.39, 71.14 tok/s,
  TTFT 243.01 ms, 64 tokens. Stream1 ITL p50 15.74 ms, p90 16.12,
  p99 16.42, 65.97 tok/s, TTFT 242.36 ms, 40 tokens. One-card C7 was
  16.51 ms (46 tokens) and 16.65 ms (40 tokens). C7 stock DP=2 eager
  C1 ITL was 121.40 ms.
- GATE_RC=0. SERVE_RC=0 at 12:56:53. Health-after HEALTHY rc 0 at
  12:57:23. No xe-reset. Container removed.

VERDICT: M3 text, identity, and C1 speed gates met. C2 ITL is below
the one-card graph reference on both streams. M0-M3 are closed.

## Phase A -- Flash-Next adapt (CPU)

Status: CPU checks passed 2026-10-20 14:55. GPU serve not started.

The 35B path stays the contiguous half. `B70_EP_SPLIT=auto` selects
the snake only when the layer has 512 experts. Flash-Next registers
per-layer int32 maps (`ep_register_map`). Local slots are the
ascending global ids of that card's set. `moe_ep`'s torch signature
is unchanged. MTP prefixes still take the plain `B70GGUFMoEMethod`,
so the drafter stays on card 0.

Split source: STRP v1
`/mnt/vm_8tb/b70/build/strata-ab/src/data/expert-profile-sycl.bin`
plus `B70_EP_Q0` (default 138). Port of
`src-ep/sycl/src/program/generate.cpp` traffic snake. `B70_EP_SHARE0`
empty leaves the night share off.

CPU test `plugins/b70_ep_xpu/tests/test_placement.py`:
`PLACEMENT_OK`, `PLE_GUARD_OK`. Every one of 48 layers is 138 on
card 0 and 374 on card 1, disjoint, covering 0..511. Local slots are
`0..n-1`. The first two unique profile experts of a layer go to
card 0 then card 1. Layer 0 card-0 head is `(3, 7, 10, 17)`.
Ranked pairs: 24576. A 4-expert synthetic snake matches the
`0,1,1,0` deal, including the quota redirect and `share0=0.25`.

GGUF types follow each tensor. Routed ids accepted: 8, 10, 11, 12,
13, 14, 16, 17, 18, 20, 21, 22, 23, 42. The IQ3_XXS shard is mixed
(`reports/02_model_support.md`): expert downs are IQ4_NL (18 layers)
or Q2_0 (30); expert gate and up are IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS, or IQ3_S. Row-byte checks: IQ3_XXS K=2560 is 980,
Q2_0 I=640 is 180, IQ4_NL I=640 is 360. The shorthand
IQ3_XXS+IQ3_XXS+Q2_0 pack is 11361484800 bytes on card 0 and
30791270400 bytes on card 1 across 48 layers. Profile loads
`index_select` the chosen rows and replace the CPU parameter with
the card-0 pack. The linear half copy is unchanged.

PLE: `ple_guard.py` wraps `install_host_table`. The owner must be a
numpy memmap (a view's `.base` chain counts). `B70_PLE_NBYTES`
(28800138240 on the serve) checks the byte count. A second data
pointer raises. MemAvailable under 8 GiB raises. `B70_EP_MEM_MIN_KB`
raises that floor. The test used a 64-byte memmap, not a second
26.8 GiB file. MemAvailable at the test was 118655020 kB.

QSA: `qsa_cover.py` runs only when `B70_QSA=1`. It calls
`b70_qsa_xpu.register()` and marks an MTP decoder (`"mtp"` in the
prefix) with `mark_draft`. Unset `B70_QSA` does not import the model.

Shared expert: `apply` still drops the shared-expert arguments. One
log line records that the runner adds that output and the weights
stay on card 0. They are not packed into the routed split.

Library: `b70_ep_xpu/_C.so` sha256
`5a0a047c67fc192588007e93b69c22a8ed9c781d2a891056dff200cceab067ac`
(499200 bytes, 2026-10-20 14:55). `ep_register_map` copies both maps
onto card 0. Load rejects a slot outside the local count, a
duplicate slot, or an expert on both cards or on neither. Dispatch
clamps any other bad slot to -1. An earlier link the same hour
(sha `308cd69d...`, 495104 bytes) stored the caller's pointer and
is not the binary to serve. `_C.ep.so` sha stays
`55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0`.
`_C.graph.so` sha
`b3c15e14ba5811d90dfe8a65e4ccc1353fd67b3f44e4180810fac22b4ed49979`
exports the `b70_moe_layer` and `b70_reserve_decode_queue` symbols
`moe_ep.cpp` dlsyms. The serve points `B70_GGUF_SO` at that file.

Serve script (not launched): `harness/serve_flashnext_ep.sh`,
wrapper `plugins/b70_ep_xpu/tests/m6_serve.sh`. World size 1,
`--card both`, port 18116, container `vllm-fn-h1-fn`, docker memory
100g. No `--data-parallel-size`, no `--enable-expert-parallel`.
`B70_EP=1`, `B70_QSA=1`, `B70_EP_SPLIT=profile`, `B70_EP_Q0=138`,
`B70_PLE_NBYTES=28800138240`. First served name `hotschmoe-dev`,
alias `qwen3.8-flash-next-iq3-xxs`. Start aborts under
`B70_EP_MEM_MIN_KB` (default 8 GiB) and warns under 72 GiB. A
watchdog docker-stops the container if MemAvailable falls under
8 GiB. The script does not exec `xpu-run`.

Phase B is closed below. Greedy matched 323 and 667. C1 and C2
ITL were measured under `FULL_DECODE_ONLY`. The latency bar and
the 2x200K KV budget were missed.

## Phase B -- Flash-Next sidecar (H1e)

Status: measured (2026-10-20 17:07 UTC). Gates passed. Latency
bar missed. KV budget missed. No H1e serve is still running.

Four windows, same recipe, both cards via `bin/gpu-run`. Driver
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1e-fn-window.sh`.
Serve `harness/serve_flashnext_ep.sh`. World size 1, port 18116,
container `vllm-fn-h1-fn`, util 0.98, max_len 2048,
max_num_batched_tokens 2048, max_num_seqs 4, `FULL_DECODE_ONLY`,
`FLASH_ATTN`, capture sizes `1,2,3,4,8`, `B70_XPU_CG_RECLAIM=500`,
`B70_GGUF_REF=0`, `B70_EAGER_QK=0`, `B70_QSA=0`, `B70_EP_SPLIT=profile`,
`B70_EP_Q0=138`. No data parallel, no vLLM expert parallel, no
oneCCL, no MTP. Engine `v0.32.1rc1.dev94+g5fd62688f`. FA overlay
`/mnt/vm_8tb/b70/build/vllm-flashnext/c7b-fa-overlay` on
`PYTHONPATH`. C7b's recipe items used here: no packed-layer bf16
placeholders, PLE host rows staged outside the graph, util 0.98.
MemAvailable under 8388608 kB aborts the serve. No sample on the
success window went under that floor, so the serve was not aborted.

Libraries at the success launch (`h1e-fn-window-ok.out`):

- Live `b70_ep_xpu/_C.so` sha256
  `2e22f57ea997a5c28d7256a79adf9ea00d28efdf923a001a6928ff7fad7e21a0`
  (the binary that produced the numbers below).
- Saved Phase A binary
  `logs/b70_ep_xpu_C.so.5a0a047c`, sha256
  `5a0a047c67fc192588007e93b69c22a8ed9c781d2a891056dff200cceab067ac`.
  It is the owned-map build. It is not the binary to restore.
- Loaded MoE library `b70_gguf_xpu/_C.o3.so` sha256
  `be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb`.
  The serve log prints `B70_GGUF_SO=.../_C.o3.so`.
- On-disk `_C.graph.so` sha256
  `e9e5282d9d4cc3def64d17516e847021d623735fde357fef690208ab291befa1`.
  The success log does not load it.
- This serve does not isolate a library-only speedup. C7b's graph
  serve loaded `_C.c12.so` sha `b3c15e14` (their report).

Attempts:

1. 16:19:21, EP sha `5a0a047c`, MoE `_C.o3.so`. Health before and
   after HEALTHY. Serve 16:19:54 stopped 16:19:59, `SERVE_RC` 1.
   The script printed `_C.o3.so is missing b70_moe_layer` and never
   reached the engine. Cause: `nm -D | grep -q` under `pipefail`
   (grep closes the pipe; SIGPIPE is a false miss). The symbol
   check in `serve_flashnext_ep.sh` now runs `nm -D` into `grep`
   with stdout to `/dev/null`. Logs: `h1e-fn-window-nmfalse.out`,
   `h1e-fn-serve-nmfalse.log`.
2. 16:23:27, same EP sha `5a0a047c`. Health before and after
   HEALTHY (`h1e-health-after-flag2.txt`, 29s, exit 0). Serve
   16:23:56, `SERVE_RC` 1, `NOT_READY`. Log line
   `EP flag timeout layer=2 err0=0 err1=11`. err1=11 is card 1's
   `flag_in` wait. err0 stayed 0 because the wait kernel continues
   and card 1 still posts `flag_ret`. Logs: `h1e-fn-window-flag2.out`,
   `h1e-fn-serve-flag2.log`.
3. 16:42:21, EP sha `1a134e8e` (eager prejit of the real `(T, topk)`,
   spin still 200000; that file was overwritten by the next link).
   Health before HEALTHY. Serve 16:43:37, stop 16:47:03, `SERVE_RC` 1,
   `NOT_READY`. Log line `EP flag timeout layer=2 err0=12 err1=0`.
   err1=0 means the prejit cleared `flag_in`. err0=12 is card 0's
   `flag_ret` wait, which starts after card 0's 138 experts and
   missed card 1's 374-expert return at profile T=2048 topk=10.
   The preserved window ends at the `health after 16:47:03` header,
   so that attempt's post-health lines are not in the copy. The
   next window's health-before at 16:49:51 was HEALTHY. Logs:
   `h1e-fn-window-flag12.out`, `h1e-fn-serve-flag12.log`.
4. 16:49:51, EP sha `2e22f57e`. `moe_ep.cpp` keeps capture spin at
   `s.spin` (init log `spin=200000`) and, when not capturing, raises
   the eager spin to 10000000 after `eager_prejit_layer`. Health
   before HEALTHY (35s, exit 0), queued first behind X2. Serve
   acquired 16:56:03. Model load 14.73 GiB in 131.08 s (log 16:59:57).
   HTTP ready 17:00:34. Stop 17:02:25, `SERVE_RC` 0. Health after
   HEALTHY (33s, exit 0), queued behind X1
   `vllm-fn-x1-flashnext-mtp0`. Preserved window mtime 17:07:08 is
   `WINDOW_DONE`. No xe-reset. Logs: `h1e-fn-window-ok.out`,
   `h1e-fn-serve-ok.log`.

Success log has 0 `EP flag` lines, 0 `error 44` lines, and 0
`kernel not compiled` lines. It has 816 `EP eager prejit` lines.
Card-1 graphs finalized: profile bar keys 4 then 3, then capture
keys 4, 3, 2, 1. Each finalized key is preceded by an eager prejit
of all 48 layers at that T and by `EP capture implicit`. No
`key=8` line. `max_num_seqs` is 4, so size 8 was not captured.
`EP init` prints `path=2`. Every split line on the success tail is
`n0=138 n1=374` of 512.

Identity (`h1e-models.json`): first id `hotschmoe-dev`, alias
`qwen3.8-flash-next-iq3-xxs`, both `max_model_len` 2048. Bench
identity `first_ok` true, `alias_ok` true.

Greedy (`h1e-greedy.json`, `GREEDY_RC` 0), temperature 0: `17*19`
text `323` match true (prompt 25, completion 4); `23*29` text `667`
match true (prompt 25, completion 4).

Bench `h1e-bench-graph.json` (copy `h1e-bench-graph-ok.json`), tag
`ep-graph-2048`, `BENCH_RC` 0. Command was `bin/cpu-run --quiet`
on `harness/bench_latency.py` against `127.0.0.1:18116`, model
`hotschmoe-dev`, decode 64, prefill target 1024. cpu-run waited
on `b70s4dayz` and then finished in 25s, exit 0. `mtp.enabled` is
false. The Strata night-final-v3 bar printed in that file is C1
7.092 ms (141 tok/s) and C2 10.870 ms (92 tok/s per stream), and
that bar includes MTP.

- C1: warmup TTFT 1828.39 ms; measured TTFT 264.67 ms; prompt 24;
  completion 64; ITL p50 60.156 ms, p90 89.402 ms, p99 98.676 ms,
  mean 54.368 ms; decode 18.393 tok/s.
- C2 stream 0: TTFT 3255.48 ms; prompt 29; completion 64; ITL p50
  32.061 ms, p90 93.210 ms, p99 135.471 ms, mean 56.304 ms; decode
  17.761 tok/s.
- C2 stream 1: TTFT 3271.47 ms; ITL p50 32.061 ms, p90 93.209 ms,
  p99 135.477 ms, mean 56.409 ms; decode 17.728 tok/s.
- Prefill: built 1017 of target 1024; TTFT 9551.77 ms; 106.47 tok/s.
  `FULL_DECODE_ONLY` leaves prefill eager. This is a 1k probe under
  max_len 2048, not the Strata 4k/32k prefill point.

C1 p50 is 8.48 times the 7.09 ms bar. C2 p50 is 2.95 times the
10.87 ms bar. C2 p50 sits below its mean, so the stream distribution
is skewed (p90 is about 93 ms). Coherence: `ok` true, `same` true,
`mentions_paris` true, verdicts `OK`/`OK`. The stored texts contain
a non-ASCII byte, so they are not pasted here.

KV, card-0 worker only (`h1e-kv.txt`, 17:00:21): available KV cache
memory 12.06 GiB; XPU KV cache size 84,224 tokens; maximum
concurrency for 2,048 tokens per request 41.12x; kv lcm block size
1600. 84,224 tokens is 0.421 of one 200K stream and 0.211 of the
2x200K target. `max_model_len` 2048 caps each request below that
pool. C7b's DP=2 graph serve, different binary and a split KV pool,
reported 0.64 GiB with DP0 4352 tokens and DP1 4608 tokens
(`reports/1C7b_flashnext_serve.md`). Their C1 ITL p50 was 81.75 ms
(14.88 tok/s) and their C2 p50 values were 33.20 ms and 33.14 ms.
Those figures are a different serving mode from this world-size-1
sidecar.

Host at HTTP ready 17:00:34 (`h1e-rss-ready.txt`): MemAvailable
49996456 kB, AnonPages 7766384 kB, vllm RSS 1769736 kB, EngineCore
RSS 5242584 kB. Window ticks before ready: 118427456 kB (16:56:58),
105087168 (16:57:58), 70621740 (16:58:59), 64328640 (16:59:59).
Serve-start MemAvailable was 118497944 kB. Lowest recorded sample
is the ready snapshot, 47.7 GiB, above the 8 GiB abort.

VERDICT: greedy 323 and 667 passed on the FULL graph sidecar.
C1 ITL p50 60.156 ms and C2 ITL p50 32.061 ms per stream miss the
Strata bar, which includes MTP. Card-0 KV is 84,224 tokens at
context cap 2048, short of 2x200K. Health before and after the
success window was HEALTHY. No xe-reset. The live sidecar library
stays sha `2e22f57e`. q0 stays 138.
