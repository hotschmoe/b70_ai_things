# H1 EP sidecar (b70_ep_xpu)

Owner: H1 / H1b / H1c. Package: `vllm/flashnext/plugins/b70_ep_xpu`.
vLLM stays one rank on card 0. The plugin owns card 1 and its expert half.
Transport is a same-context peer store plus device flags. No oneCCL.
Design: `reports/2O1_ep_design.md` section 6, milestones M0-M3.

Library: `b70_ep_xpu/_C.so` sha256
`bd4f4aa010c718daa6a2c983e8a93670acdd2e4df1f094c0819ed68517c39594`
(476912 bytes, built 2026-10-20 09:38). Slotted GGUF library
`b70_gguf_xpu/_C.ep.so` sha256
`55c154ee90f33bec810d24a2cd6ac0c3b6b1a4e53bdc78aed50a59113e8c9bf0`
(not rebuilt for this binary).

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

Status: not started. Script `/tmp/h1c-m2-window.sh`, port 18114,
container `vllm-fn-h1-m2`. Gate: `/v1/models` lists `hotschmoe-dev`
first, greedy 16-token text equals
` Paris, a city renowned for its rich history, culture, and iconic landmarks.`

## M3 -- FULL_DECODE_ONLY

Status: not started. Script `/tmp/h1c-m3-window.sh`, port 18115,
container `vllm-fn-h1-m3`, `B70_XPU_CG_RECLAIM=500`. Gate: same text,
C1/C2 ITL versus one-card 11.11 / 16.5 ms and versus C7 stock DP=2
eager ITL 121.40 ms.
