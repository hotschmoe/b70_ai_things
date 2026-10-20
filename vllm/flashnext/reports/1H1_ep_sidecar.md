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

Phase B gate, still open: greedy `17 * 19` -> 323 and `23 * 29` ->
667, then C1/C2 ITL under `FULL_DECODE_ONLY`. Strata C1 reference
is about 141 tok/s (7.1 ms ITL) with MTP. Latency is the top metric.

## Phase B -- Flash-Next sidecar (H1e)

Status: waiting to launch (2026-10-20 15:52 UTC). No GPU serve yet.

C7b's report `reports/1C7b_flashnext_serve.md` is absent. At 15:51
card 0 was held by O3 `run_kq.sh` (pid 3373613, since 15:50:29).
Card 1's owner file was empty. C7b's FlashAttention side build
exited `BUILD_FAIL` at 15:49:10: cmake could not find git to clone
cutlass (`logs/c7b-fa-build.log`). The installed
`libattn_kernels_xe_2.so` is still the wheel that rejected
`16,256,64,false,false,false` at 15:39. Launch waits until that
report exists, or until both owner files have stayed empty for
10 minutes.

Libraries rechecked: `b70_ep_xpu/_C.so`
`5a0a047c67fc192588007e93b69c22a8ed9c781d2a891056dff200cceab067ac`
(499200 bytes). `b70_gguf_xpu/_C.graph.so` and `_C.so`
`be76661bd120aa2f` (10135104 bytes, 15:19:23), which is O3.
`b70_moe_layer` and `b70_reserve_decode_queue` are both exported.
`_C.ep.so` stays `55c154ee`. `_C.c12.so` stays `b3c15e14`.

Serve script `harness/serve_flashnext_ep.sh` now defaults
`gpu_memory_utilization` to 0.98 and accepts `MAX_BATCHED`.
Window driver (not started):
`/mnt/vm_8tb/b70/build/vllm-flashnext/logs/h1e-fn-window.sh`.
It is C7b's graph recipe on the sidecar: world size 1, port
18116, container `vllm-fn-h1-fn`, util 0.98, max_len 2048,
max_num_batched_tokens 2048, max_num_seqs 4, `FULL_DECODE_ONLY`,
`FLASH_ATTN`, capture `1,2,3,4,8`, `B70_XPU_CG_RECLAIM=500`,
`B70_GGUF_REF=0`, `B70_EAGER_QK=0`, `B70_QSA=0`. QSA is off
because the 15:10 eager serve that returned 323 and 667 did not
set it. MTP is not enabled. The fork already drops packed-layer
bf16 placeholders and stages PLE rows in `_graph_emb` outside
the graph. MemAvailable under 8 GiB aborts the serve. The bar
is Strata C1 7.09 ms (141 tok/s, with MTP) and C2 10.9 ms per
stream. Card-0 KV tokens are recorded from the engine log.
