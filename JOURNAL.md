# JOURNAL

Active window starts at 2026-08-20bq. The removed history is preserved locally
at `archive/to-delete-20260826/JOURNAL.before-trim.md` until quarantine purge.

### 2026-08-20bq - LOOP 21: attach k1bar-pc1 59m G1 GO

CONTEXT -> Overnight 30m. NEXT PICK
  park. k1bar-pc1 UP 59 min. No second
  serve. No extra bench_code.

CONFIG -> attach live ...-k1bar-pc1
  TP=2 GRAPH BARRIER AGASYNC P2P=0.

COMMAND ->
  ```
  g1_probe http://192.168.10.5:18080/v1
  docker logs | grep DEVICE_LOST
  ```

RESULT -> G1 Paris/391 chat Paris no
  bangs. No DEVICE_LOST. JSONDecode
  at load only (same as boot). Serve
  Up 59 min.

VERDICT -> Attach GO. Leave up. Park.
  Do not demote 31.9. Do not P2P.
  Do not start DD. Do not start Q8.

### 2026-08-20br - LOOP 22: attach k1bar-pc1 ~90m G1 GO

CONTEXT -> Overnight 30m. NEXT PICK
  park. k1bar-pc1 UP ~90 min.

CONFIG -> attach live ...-k1bar-pc1
  TP=2 GRAPH BARRIER AGASYNC P2P=0.

COMMAND ->
  ```
  g1_probe http://192.168.10.5:18080/v1
  ```

RESULT -> G1 Paris/391 chat Paris no
  bangs. No DEVICE_LOST. Serve Up.

VERDICT -> Attach GO. Leave up. Park.
  Do not demote 31.9. Do not P2P.
  Do not start DD. Do not start Q8.

### 2026-08-20bs - LOOP 23: overnight closed

CONTEXT -> Operator: close it all up and
  postmortem. How did Pliny Q8 go?

CONFIG -> Scheduler 01a01dff8593 cancelled.
  docker stop -t 30 qwen38_w8a8_dspark.
  xpu-health both cards.

COMMAND ->
  ```
  scheduler_delete 01a01dff8593
  docker stop -t 30 qwen38_w8a8_dspark
  ./bin/xpu-health
  ```

RESULT ->
  Cards free. HEALTHY. Postmortem
  docs/20260820_lmx_overnight_postmortem.md.
  Pliny Q8_0 2x hold **32.03** (1x 17.93),
  G1 always GO, 0.73x vs Q4_K_M 43.8.
  Biggest overnight move: W1 47.58 -> 65.08.

VERDICT -> CLOSED. Holds 34.9 / 31.9 /
  65.08 / 32.03. DD PARKED. P2P=0.

### 2026-08-20bt - 7.1 P2P fabric re-measure (L0 + oneCCL, no vLLM serve)

CONTEXT -> Operator: why Steve P2P=1 works and
  our vLLM TP=2 hangs; profile actual
  card-to-card copies; board vs kernel.

CONFIG -> kernel 7.1.0-070100, GuC 70.58.0,
  runtime 26.22.38646.4. 1950X / ASRock X399
  Professional Gaming BIOS P4.05. GPU0
  0000:0b:00.0 under RC 0000:00, GPU1
  0000:44:00.0 under RC 0000:40. Uplinks
  09:00.0 / 42:00.0 = 8.0 GT/s x16.
  Stopped Pliny+open-webui. No vLLM P2P serve
  (LOOP 4 hang already gated).

COMMAND ->
  ```
  IMG=vllm-xpu-env:v0260 ./bin/gpu-run bash scripts/100_run_peer_copy.sh
  IMG=vllm-xpu-env:v0260 ./bin/gpu-run bash scripts/102_run_push_allreduce.sh
  IMG=vllm-xpu-env:v0260 ./bin/gpu-run bash scripts/103_run_ipc_push_allreduce.sh
  IMG=int8g-v0260 allreduce_bench.py x4
    P2P=0/1 x SYCL=0/1
  ./bin/xpu-health
  ```

RESULT -> L0 PUSH 11.21 GB/s / PULL 3.24 /
  host bounce 3.52 / 8B 8.60 us. PUSH AR
  10.66 GB/s @16MB, 50.8 us @10KB. IPC PUSH
  11.06 GB/s @16MB, 13.1 us @10KB. oneCCL
  mp.spawn: P2P=0 ~1.1 GB/s; P2P=1 eager
  3.47 (PULL); **P2P=1 SYCL 10.39 GB/s**.
  Health GO after P2P=1 microbench.

VERDICT -> Fabric P2P is fine. vLLM hang is
  worker warmup, not Threadripper DMA.
  Steve is EPYC 9015. Do not enable vLLM
  P2PACCESS. PUSH_AR already matches the
  10.4 GB/s P2P-SYCL number. P2P_GPU K.10.

### 2026-08-20bu - five serial BW campaigns written

CONTEXT -> Operator: campaign each of INT8
  GDN, fused quant, XPUGraph/MRV2, VNNI16,
  DSpark accept~3. Then NVIDIA-shaped
  W4A16/W4A4/XMX, then 4x MoE vs TB 73.

CONFIG -> docs only. No GPU.

COMMAND -> wrote
  docs/20260820_b70_bw_campaigns.md
  RESEARCH_TODO living header.

RESULT -> A-E serial. F = W4A8 is the XMX
  steal (W4A16 kernel is dequant-to-BF16,
  NVFP4 is not INT4 DPAS). G = no 3-17B
  active MoE matches Qwen3.8-27B TB 73;
  Ornith 1.5 is 67.8. 4x128G fits 122B-A10B
  W4A16.

VERDICT -> Plan parked until LOOP 1. Do
  not mix levers. Do not start DD.

### 2026-08-20bv - W4A8 full-send successor campaign written

CONTEXT -> Operator: full-send Qwen3.8-27B W4A8,
  extract every Intel path, accuracy later,
  dual-card (gpu0 quant / gpu1 kernels),
  fresh vLLM/sglang/llamacpp W4A8 container,
  train DSpark INT if it beats FP. New Grok
  session should be able to start from one
  doc. No public 3.8 W4A8 on HF.

CONFIG -> docs only plus a 3.8 copy of the
  149 two-group producer. No GPU. Cards free.

COMMAND -> wrote
  docs/20260820_qwen38_w4a8_campaign.md
  docs/20260820_qwen38_w4a8_loops.md
  docs/20260820_qwen38_w4a8_deadends.md
  vllm/w4a8/README.md
  scripts/151_quantize_qwen38_27b_w4a8.sh
  (copy of 149; SRC=qwen3.8-27b/bf16;
  IMG=int8g-v0260; default DATAFREE=1 ->
  models/files/qwen3.8-27b/w4a8-rtn-gdn;
  DATAFREE=0 -> w4a8-gptq-gdn).
  RESEARCH_TODO headline swapped to this
  campaign. A-E parked.

RESULT -> Successor standing prompt with
  Path H (hybrid W4A16-decode / W4A8-prefill,
  3.6 proven 27.3) vs Path X (native XMX all
  M) vs Path S (s4 DPAS TOPS). K0-K19 loop
  catalog. 13 imported dead-ends (AutoRound,
  grouped _int_mm, joint_matrix, compile hang,
  NVFP4-as-XMX, P2PACCESS, etc). Byte-budget
  EXPECTED only until 151 census.

VERDICT -> New session fires dual-card
  day-1. Do not bake images first. Do not
  start DD. P2PACCESS=0. ASCII. Journal
  each loop.

### 2026-08-21a - LOOP 1: 151 DATAFREE RTN W4A8+GDN GO

CONTEXT -> W4A8 full-send day-1 card 0.
  No public 3.8 W4A8. Produce via 151.

CONFIG -> DATAFREE=1 CARD=0
  IMG=vllm-xpu-env:int8g-v0260
  SRC=models/files/qwen3.8-27b/bf16
  OUT=models/files/qwen3.8-27b/w4a8-rtn-gdn
  two-group W4A8 MLP/attn + W8A8 GDN fat.
  P2PACCESS unset. DD parked.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-151-rtn \
    ./bin/gpu-run --card 0 \
    env DATAFREE=1 CARD=0 \
    bash scripts/151_quantize_qwen38_27b_w4a8.sh
  ```

RESULT -> 443s exit 0. DataFreePipeline
  119s. GDN hit 144/144. Stage B
  packed=256 int4, int8-kept=144, graft
  vis=333 mtp=15. 20.616 GiB. is_prepacked
  w4a8=True. Arch Qwen3_5ForCausalLM.
  CT 0.18.0 in the live container.
  Host rm RAW bounced (root-owned);
  removed via docker + chown 1000.

VERDICT -> GO. Pipeline smoke artifact
  exists. GPTQ fire 2 after load-gate.
  Do not bake. Do not start DD.

### 2026-08-21b - LOOP 2: K1 kernel matrix GO

CONTEXT -> Day-1 card 1. 3.8 shapes, no
  3.8 W4A8 file. 3.6 w4a8-sqgptq stand-in.

CONFIG -> card1 ZE_AFFINITY_MASK=1
  IMG=int8g-v0260
  SO=w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so
  M in {1,2,4,8,16,32,64,256,2048}
  7 shapes. Path S proto_int4 after.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-k1 \
    ./bin/gpu-run --card 1 \
    bash vllm/w4a8/run_k1_matrix.sh
  ```

RESULT -> First start: image Entrypoint
  is leftover `sleep` -> sleep -c. Rerun
  --entrypoint bash, 25s, CSV n=441.
  gate_up M=1 w4a16 0.161ms 552 GB/s 95%
  of 581, 3.72x bf16. down_proj M=1
  w4a16 0.079ms 565 GB/s 97%. w4a8_op
  tied with H at M=1. w4a8_full pays
  ~17 us quant on down_proj (not 101 us).
  M=2048 gate_up: w8a8 260 TOPS, w4a8_op
  197, w4a16 135. gdn_ba N=96 <3% roof.
  Path S s8 208 / s4 558 / s2 560 TOPS.

VERDICT -> GO. Split-M: decode Path H,
  large-M Path X (or W8A8 TOPS). N=96
  keep BF16. Do not mix H/X unnamed.

### 2026-08-21c - LOOP 3: K0 file census GO

CONTEXT -> Fail-closed before any 3.8
  W4A8 speed claim. Dispatch proof is
  the GRAPH=0 serve, not this file scan.

CONFIG -> models/files/qwen3.8-27b/w4a8-rtn-gdn
  CPU safetensors. No GPU.

COMMAND -> python census by category
  (mlp / self_attn / gdn_fat / gdn_other
  / lm_head / embed / mtp / visual)

RESULT -> MLP I32 7.969 + attn I32 0.781
  GiB. GDN fat I8 5.156 GiB (not BF16).
  gdn_other BF16 0.048. lm_head 2.368.
  hot no vis/mtp 18.967. vis 0.858 mtp
  0.791. down_proj [5120,2176] I32,
  q_proj [12288,640] I32, in_proj_qkv
  [10240,5120] I8, in_proj_b [48,5120]
  BF16. No blanket linear_attn ignore.

VERDICT -> File census GO. Next: GRAPH=0
  vLLM smoke card 1, served id
  qwen3.8-27b-W4A8-rtn-gdn. Do not GPTQ
  until load-gate. Do not start DD.
  P2PACCESS=0.

### 2026-08-21d - LOOP 4: W4A8 campaign journal + 30m loop

See `docs/20260820_qwen38_w4a8_journal.md` (this
campaign's journal from here on). Arming /loop 30m.
Next fire = GRAPH=0 smoke, then every 30m. DD PARKED.

### 2026-08-21e - LOOP 5: GRAPH=0 W4A8 load-gate GO

See `docs/20260820_qwen38_w4a8_journal.md`. Paris/391/fib
coherent. Serve Up :18081. GPTQ fire 2 unblocked.

### 2026-08-21f - LOOP 6: 151 GPTQ fire 2 STARTED

See `docs/20260820_qwen38_w4a8_journal.md`. pid=353913
log results/logs/151_qwen38_w4a8_20260821_010557.log

### 2026-08-21g - LOOP 7: GRAPH=0 attach ~6.3 tok/s

See `docs/20260820_qwen38_w4a8_journal.md`. Not bench_code c1.

### 2026-08-21h - LOOP 8: ATTACH GPTQ layer 9/64

See `docs/20260820_qwen38_w4a8_journal.md`. pid=353913 still.
~2.5h left. Do not start a second 151.

### 2026-08-21i - LOOP 9: ATTACH GPTQ layer 30/64

See `docs/20260820_qwen38_w4a8_journal.md`. ~1h left.

### 2026-08-21j - LOOP 10: ATTACH GPTQ layer 49/64

See `docs/20260820_qwen38_w4a8_journal.md`. ~25 min + save left.

### 2026-08-21k - LOOP 11: GPTQ W4A8 artifact GO

See `docs/20260820_qwen38_w4a8_journal.md`. 20.616 GiB census GO.

### 2026-08-21l - LOOP 12: GPTQ GRAPH=0 smoke GO

See `docs/20260820_qwen38_w4a8_journal.md`. :18082 Up. Paris/391/fib.

### 2026-08-21m - LOOP 13: GPTQ GRAPH=1 ~24.5 tok/s GO

See `docs/20260820_qwen38_w4a8_journal.md`. ~3.9x GRAPH=0. Not 31.9.

### 2026-08-21n - LOOP 14: bench_code c1 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. GRAPH=1 GPTQ. Not 31.9.

### 2026-08-21o - LOOP 15: HYBRID=1 e2e 1.00x NO-GO

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 HYBRID=0.

### 2026-08-21p - LOOP 16: GRAPH=1 MTP3 !!!! false 61.7 D14

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 NOMTP.

### 2026-08-21q - LOOP 17: K16 c=2 agg 47.7 G1 OK

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21r - LOOP 18: K16 c=4 agg 91.4 G1 4/4

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21s - LOOP 19: K16 c=8 agg 145.8 G1 8/8

See `docs/20260820_qwen38_w4a8_journal.md`. K16 concurrent row complete. Score stays 25.0 c1.

### 2026-08-21t - LOOP 20: K4 M=4,8 still BW; D15 pad-M

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21u - LOOP 21: K5 stock sycl-tla D16

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21v - LOOP 22: K10 prefill ~2870 / ~2750 tok/s

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21w - LOOP 23: M=2048 Path X 1.4-1.7x H

See `docs/20260820_qwen38_w4a8_journal.md`. Split-M confirmed at prefill M.

### 2026-08-21x - LOOP 24: attach c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21y - LOOP 25: K8 lm_head g32 isolated 1.27 ms

See `docs/20260820_qwen38_w4a8_journal.md`. e2e lm_head INT4 still open.

### 2026-08-21z - LOOP 26: attach c2 agg 48.5

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21za - LOOP 27: K12 N-pad 96->128 D17

See `docs/20260820_qwen38_w4a8_journal.md`. Keep ba BF16.

### 2026-08-21zb - LOOP 28: attach c8 agg 147.8

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21zc - LOOP 29: K13 g32/g64 slower D18

See `docs/20260820_qwen38_w4a8_journal.md`. GROUP=128 stays.

### 2026-08-21zd - LOOP 30: K15 TP=2 GRAPH=0 PUSH_AR c1 3.7

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays TP=1 GRAPH=1 25.0.

### 2026-08-21ze - LOOP 31: K15 GRAPH=1 TP=2 D19 segfault

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays TP=1 GRAPH=1 25.0.

### 2026-08-21zf - LOOP 32: GRAPH=1 TP=1 restore c1 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0 c1.

### 2026-08-21zg - LOOP 33: D04 joint_matrix still gated

See `docs/20260820_qwen38_w4a8_journal.md`. Leave D04 closed.

### 2026-08-21zh - LOOP 34: K17 off-shelf DSpark pos0 66% / c1 14.6

See `docs/20260820_qwen38_w4a8_journal.md`. Score stays 25.0. Do not night-train.

### 2026-08-21zi - LOOP 35: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21zj - LOOP 36: GRAPH=1 DSpark 8192 KV miss

See `docs/20260820_qwen38_w4a8_journal.md`.

### 2026-08-21zk - LOOP 37: GRAPH=1 DSpark c1 34.7 coherent

See `docs/20260820_qwen38_w4a8_journal.md`. Spec row 34.7. NOMTP honesty 25.0.

### 2026-08-21zl - LOOP 38: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21zm - LOOP 39: k=4 c1 33.5 NO-GO vs k=7 34.7

See `docs/20260820_qwen38_w4a8_journal.md`. Keep SPECTOK=7.

### 2026-08-21zn - LOOP 40: restore k=7 c1 35.2

See `docs/20260820_qwen38_w4a8_journal.md`. Spec path restored.

### 2026-08-21zo - LOOP 41: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21zp - LOOP 42: MAXSEQS=2 c1 32.4 / c2 agg 47.2

See `docs/20260820_qwen38_w4a8_journal.md`. Keep MAXSEQS=1 for isolated spec.

### 2026-08-21zq - LOOP 43: restore MAXSEQS=1 c1 33.8

See `docs/20260820_qwen38_w4a8_journal.md`. Isolated spec restored.

### 2026-08-21zr - LOOP 44: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21zs - LOOP 45: k=3 c1 31.0 NO-GO vs k=7 34.7

See `docs/20260820_qwen38_w4a8_journal.md`. k-sweep closed 7>4>3.

### 2026-08-21zt - LOOP 46: restore k=7 c1 34.1

See `docs/20260820_qwen38_w4a8_journal.md`. Isolated spec restored.

### 2026-08-21zu - LOOP 47: attach NOMTP c1 25.0 / prefill 2880

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score and prefill hold.

### 2026-08-21zv - LOOP 48: attach DSpark c1 33.2 / prefill 2615

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold. Prefill tax ~9%.

### 2026-08-21zw - LOOP 49: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21zx - LOOP 50: sglang float16 GDN triton crash

See `docs/20260820_qwen38_w4a8_journal.md`.

### 2026-08-21zy - LOOP 51: sglang bf16 GARBAGE

See `docs/20260820_qwen38_w4a8_journal.md`. Do not GRAPH=1 sglang yet.

### 2026-08-21zz - LOOP 52: restore DSpark k=7 c1 33.4

See `docs/20260820_qwen38_w4a8_journal.md`. Spec path restored.

### 2026-08-21aaa - LOOP 53: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21aab - LOOP 54: attach DSpark c1 35.2

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold.

### 2026-08-21aac - LOOP 55: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21aad - LOOP 56: attach DSpark c1 35.5

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold.

### 2026-08-21aae - LOOP 57: attach NOMTP c1 hold 25.0

See `docs/20260820_qwen38_w4a8_journal.md`. NOMTP score holds 25.0.

### 2026-08-21aaf - LOOP 58: attach DSpark c1 37.9

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold (do not replace 34.7).

### 2026-08-21aag - LOOP 59: K8 e2e int4 lm_head g32 c1 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. 1.08x vs 25.0; leave LMHEAD=1 Up.

### 2026-08-21aah - LOOP 60: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aai - LOOP 61: DSpark+LMHEAD c1 33.8

See `docs/20260820_qwen38_w4a8_journal.md`. Coherent; no 1.10x vs 34.7.

### 2026-08-21aaj - LOOP 62: attach NOMTP lmhead32 c1 27.0 / PP 2897

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP and prefill hold.

### 2026-08-21aak - LOOP 63: attach DSpark+LMHEAD c1 34.7

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold matches 34.7.

### 2026-08-21aal - LOOP 64: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aam - LOOP 65: attach DSpark+LMHEAD c1 35.8 / PP 2635

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold. PP tax ~9%.

### 2026-08-21aan - LOOP 66: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aao - LOOP 67: attach DSpark+LMHEAD c1 33.4

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold (restore-class vs 34.7).

### 2026-08-21aap - LOOP 68: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aaq - LOOP 69: attach DSpark+LMHEAD c1 31.1

See `docs/20260820_qwen38_w4a8_journal.md`. Spec low hold vs 34.7; accept jitter.

### 2026-08-21aar - LOOP 70: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aas - LOOP 71: attach DSpark+LMHEAD c1 33.5

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold; recovered from 31.1.

### 2026-08-21aat - LOOP 72: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aau - LOOP 73: attach DSpark+LMHEAD c1 35.3

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold.

### 2026-08-21aav - LOOP 74: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aaw - LOOP 75: attach DSpark+LMHEAD c1 34.6

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold.

### 2026-08-21aax - LOOP 76: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aay - LOOP 77: attach DSpark+LMHEAD c1 32.3

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21aaz - LOOP 78: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21aba - LOOP 79: attach DSpark+LMHEAD c1 38.8

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold (high of band vs 34.7).

### 2026-08-21abb - LOOP 80: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abc - LOOP 81: attach DSpark+LMHEAD c1 33.4

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abd - LOOP 82: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abe - LOOP 83: attach DSpark+LMHEAD c1 36.9

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abf - LOOP 84: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abg - LOOP 85: attach DSpark+LMHEAD c1 38.1

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abh - LOOP 86: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abi - LOOP 87: attach DSpark+LMHEAD c1 33.0

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abj - LOOP 88: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abk - LOOP 89: attach DSpark+LMHEAD c1 36.4

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abl - LOOP 90: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abm - LOOP 91: attach DSpark+LMHEAD c1 34.6

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abn - LOOP 92: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abo - LOOP 93: attach DSpark+LMHEAD c1 31.3

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abp - LOOP 94: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abq - LOOP 95: attach DSpark+LMHEAD c1 31.8

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abr - LOOP 96: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abs - LOOP 97: attach DSpark+LMHEAD c1 31.5

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abt - LOOP 98: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abu - LOOP 99: attach DSpark+LMHEAD c1 37.0

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abv - LOOP 100: attach NOMTP lmhead32 c1 hold 27.0

See `docs/20260820_qwen38_w4a8_journal.md`. K8 NOMTP holds 27.0.

### 2026-08-21abw - LOOP 101: attach DSpark+LMHEAD c1 34.1

See `docs/20260820_qwen38_w4a8_journal.md`. Spec hold vs 34.7.

### 2026-08-21abx - LOOP 102: DISARM 15m scheduler

See `docs/20260820_qwen38_w4a8_journal.md`. 15m loop DISARMED. Serves left Up.

### 2026-08-21aby - LOOP 103: GRAPH=0 TP=2 262k hotschmoe-dd c1 3.7

See `docs/20260820_qwen38_w4a8_journal.md`. Long-ctx load-gate. Not a speed DD.

### 2026-08-21abz - LOOP 104: dual 1-card Ornith MixedCal-v2 + NVFP4 27B

See `docs/20260820_qwen38_w4a8_journal.md`. :18080 Ornith MTP1 131k c1 66.1; :18081 NVFP4 100k c1 64.5.

### 2026-08-23a - OBLITERATED V3 fixed-merge Q4_K_M acquisition

CONFIG -> HF `OBLITERATUS/Qwen3.8-27B-OBLITERATED` revision
`2648a6231b82328c601ba27b9ffd5029057d0e33`, Q4_K_M file only. The
pre-V3 Q8_0 was bad per operator and was permanently deleted. 0xSero B70
SYCL image retained. External MTP sidecar fetched for fallback inspection.

COMMAND -> `hf download ... Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
--revision 2648a623...`; `sha256sum`; `gguf_dump --no-tensors --json`.

RESULT -> 16810714400 bytes; SHA256
`c5e4fe705883e244a468c9e445c8d6ba37fd310b0113e25d2b8a7f2d6f1243e8`.
GGUF name `Qwen3.8 27b S99 Merged Fixed`, qwen35, native ctx 262144,
65 blocks, 866 tensors, embedded `blk.64.nextn.*` MTP head. Both cards
passed xpu-health before serve work.

VERDICT -> GO. Exact V3 fixed-merge artifact proven; Q8 is not a fallback.

### 2026-08-23b - OBLITERATED Q4_K_M DP=2 wrapper fix

CONFIG -> two one-card `qwen38-b70:latest` replicas, nginx :18080,
Q8 KV, ctx 245760, MTP off, lab Q4K doors on.

COMMAND -> `./bin/gpu-run bash
llamacpp/serve_qwen38_obliterated_q4km_dp2.sh start`.

RESULT -> first start restart-looped before llama.cpp output. `bash -x`
showed exit while sourcing oneAPI setvars under Bash nounset. Source setvars
before `set -u`; syntax/shell checks green. Cards stayed HEALTHY.

VERDICT -> GO after wrapper fix. Not a model, quant, or context failure.

### 2026-08-23c - OBLITERATED Q4_K_M DP=2 no-MTP @245760 GO

CONFIG -> V3 Q4_K_M, DP=2 as independent TP=1 replicas, Q8 KV,
ctx 245760 per replica, batch/ubatch 1024/256, parallel 1, Q4K lab doors,
temp 0, repeat penalty 1.15, thinking off. nginx serves `hotschmoe-dd`.

COMMAND -> wrapper start under `gpu-run`; four Paris gates; `/v1/models`;
phase_bench unique cold p512/g128 n=5 ignore-eos through nginx, then two
phase benches simultaneously against :18181 and :18182.

RESULT -> both replicas and proxy coherent. API reports id hotschmoe-dd,
n_ctx 245760, train ctx 262144, Q4_K_M. Proxy alternated 18182/18181.
Serial nginx median 23.93 tok/s. Simultaneous card medians 23.84 and 23.85,
aggregate **47.69 tok/s** with effectively 2.00x DP scaling. Prefill proxy
567.6/589.1 tok/s; TTFT 2.037/1.964 s. Raw JSON under
`results/logs/qwen38_obliterated_q4km/`.

VERDICT -> GO baseline. Identity, large context, DP=2, and one endpoint are
proven. Embedded-MTP A/B plus sustained concurrent soak remain before shelf.

### 2026-08-23d - embedded MTP3 A/B @245760: +71.7% aggregate

CONFIG -> same V3 Q4_K_M DP=2 config as 2026-08-23c, changing only to
embedded `draft-mtp`, draft max 3, Q8_0 draft KV. No external sidecar.

COMMAND -> restart with `ENABLE_MTP=1 MTP_SIDECAR=0 MTP_DRAFT_MAX=3`;
same p512/g128 n=5 phase bench through nginx and simultaneously direct.

RESULT -> full 245760 context fit on both cards. Serial nginx 41.25 tok/s.
Direct medians 40.35 and 41.51, aggregate **81.86 tok/s** versus 47.69
no-MTP (+71.7%). Draft acceptance varied ~0.42-1.00, mean len 2.26-4.00.

VERDICT -> GO. Embedded MTP3 is a large controlled speed win.

### 2026-08-23e - MTP3 deterministic coherence/equivalence gate

CONFIG -> seven deterministic tests on both direct replicas, save full text;
restart without MTP and compare the same greedy seed-1 prompts.

COMMAND -> `llamacpp/qualify_qwen38_obliterated_q4km.py` against :18181
and :18182 in both modes, with MTP JSON as the no-MTP reference.

RESULT -> both MTP replicas byte-identical on 7/7. MTP vs no-MTP exact on
6/7 per card; the seventh differed only `No.` vs `No;` with the same correct
logic. Both modes made the same modular-arithmetic miss (0 instead of 71).
Paris, 391, Fibonacci, sort, logic, and exact 24-line squares were coherent.

VERDICT -> GO MTP coherence. No MTP-only garbling or correctness regression;
record the shared modular miss as model quality, not speculation corruption.

### 2026-08-23f - MTP3 DP=2 c4 mixed-load soak PASS

CONFIG -> MTP3, ctx 245760, c4 through nginx for 300 seconds; six validated
short/medium/long cases with degeneracy checks and upstream tracking.

COMMAND -> `gpu-run python3
llamacpp/soak_qwen38_obliterated_q4km.py --concurrency 4 --duration 300`.

RESULT -> 338/338 coherent, 9488 output tokens, zero coherence failures,
zero degenerate outputs, zero request errors. Routes card0/card1 = 171/167.
All cases ran 55-58 times, including 57 long exact-marker prefills.

VERDICT -> GO shelf promotion for MTP3. Concurrent behavior is coherent and
the two independent replicas balance evenly.

### 2026-08-23g - MTP3 real 152289-token request PASS

CONFIG -> shelved MTP3 candidate, ctx 245760/Q8_0 per replica, unique cold
entropy prompt through nginx, actual prompt above the requested 150k floor.

COMMAND -> `phase_bench.py --prompt-tokens 75000 --gen-tokens 8 --n 1
--skip-warmup --ignore-eos --timeout 1800`, under `gpu-run`.

RESULT -> API usage prompt=152289, completion=8, coherent text. Server prompt
eval 1184699.71 ms / 152289 = 128.55 tok/s; total 1185582.65 ms. Draft
acceptance 4/7, mean len 2.33. `truncated = 0`; no context shift. The phase
harness's printed TTFT excludes the response-header wait here; server timing
is authoritative.

VERDICT -> GO large context. Real >150k request proven; live slot is 245760.

### 2026-08-23h - final shelf restart and identity gate PASS

CONFIG -> promoted llama.cpp shelf entry with no environment overrides.

COMMAND -> stop/start/status under `gpu-run` via
`rdy_to_serve/llamacpp/qwen38-27b-obliterated-q4km/serve.sh`.

RESULT -> exact model SHA passed; defaults were MTP3, Q8_0 KV, ctx 245760.
Both replicas and four Paris gates passed. `/v1/models`: hotschmoe-dd,
n_ctx 245760, train 262144, n_params 27320697856, Q4_K Medium. Proxy routes
alternated 18182/18181/18182/18181. Three restart-unless-stopped containers
remain UP. Tracked systemd unit validates, but install needs interactive sudo.

VERDICT -> GO live daily-driver shelf. Operator sudo install is the only
persistence handoff; Docker restart policy keeps the present runtime live.

### 2026-08-23i - pinned exact-file reprovision guard

CONFIG -> manifest source revision plus exact Q4_K_M file list; generic HF
entries remain whole-repository downloads when no file list is present.

COMMAND -> `ONLY=qwen3.8-27b/obliterated-q4km bash models/fetch.sh --list`;
Bash syntax and manifest YAML parse.

RESULT -> fetch plan is pinned to revision `2648a623...` and only
`Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf`. `fetch.sh` now honors optional
`source.revision` and `source.files`; unpinned entries still parse normally.

VERDICT -> GO. Fresh reprovisioning cannot accidentally redownload the bad
old Q8 or mix another repository revision into this shelf entry.

### 2026-08-23j - Open WebUI launched against obliterated daily driver

CONFIG -> existing `open-webui` data volume, host port 3000, auth disabled,
Ollama disabled, OpenAI backend `http://192.168.10.5:18080/v1`, live API key
from `/mnt/vm_8tb/b70/secrets/dd_api_key`. Existing container had a stale
backend at port 8010.

COMMAND -> remove and recreate only the disposable `open-webui` container,
preserving the `open-webui` volume; wait for Docker health; query `/models`
from inside the container with its configured credentials.

RESULT -> container healthy at `http://192.168.10.5:3000`; backend returned
exact model id `hotschmoe-dd`. Existing WebUI data volume was preserved. The
three llama.cpp/nginx daily-driver containers were not restarted.

VERDICT -> GO for operator chat testing of the fixed obliterated model.

### 2026-08-23k - updated Steve/Sergio performance comparison

CONFIG -> clean canonical Steve clone
`/mnt/vm_8tb/b70/research/b70-lab-agent` and clean canonical Sergio clone
`/mnt/vm_8tb/b70/community_repos/intel-arc-pro-b70-inference-cookbook`.
Preserve the older divergent Steve checkout untouched.

COMMAND -> fetch and fast-forward both clean clones; inspect current claims,
repro packets, and Qwen3.8 methodology; compare against local MTP3 direct
40.35/41.51 tok/s and 81.86 two-stream sum.

RESULT -> Steve at `0107f278a1486b6177fc5d4e6b7b44e04f14bc52` and Sergio at
`dca0249684769b0a945a8d702352fdeea658852a`. Steve's verified standard
llama.cpp Q4_K_M no-spec TP1 is 27.81 tok/s; local per-card mean is 47.2%
higher, but model, MTP, KV, and context differ. Steve refuted 101.922 because
its greedy margin changed output; honest 101.170 remains non-promotable at
only 21-22/25 inter-arm agreement. Sergio's one-card GPTQ-INT4 MTP4 is 81.20
with BF16 draft, 112.65 with optional draft INT4, and 106.7 on the current
greedy C1 stack. Local per-card decode is materially slower; its advantages
are two isolated lanes, proven coherent MTP, and a real 152289-token request
inside a 245760-token slot. Full caveats and arithmetic are in
`docs/20260823_obliterated_q4km_peer_comparison.md`.

VERDICT -> good daily-driver performance, not a raw single-card record. Treat
all peer percentages as orientation until a matched harness is run. The old
Steve 101.922 headline is corrected in active docs and must not be promoted.

### 2026-08-23l - stock Qwen3.8 4-bit coding selection

CONFIG -> compare stock Qwen3.8-27B Q4_K_M, AutoRound W4A16, and NVFP4 on
the common local HumanEval+ 164 thinking-off greedy sandbox gate. Read exact
`summary.json` results; do not substitute speed or Paris gates for coding.

COMMAND -> inspect the four Qwen3.8 result directories under `evals/results`:
stock Q4_K_M, W4A16 AutoRound, Inferact NVFP4, and RadixArk NVFP4.

RESULT -> Q4_K_M 0.970/0.927; W4A16 0.963/0.915; Inferact NVFP4
0.939/0.915; RadixArk NVFP4 0.933/0.890. Q4_K_M leads W4A16 by one base
and two plus problems, a small observed lead rather than a universal quant law.

VERDICT -> select stock Q4_K_M as the quality-first daily driver. A matched
LiveCodeBench/agentic comparison remains unmeasured.

### 2026-08-23m - stock Q4_K_M TP=2 daily-driver switch PASS

CONFIG -> exact ggml-org stock Q4_K_M revision `0669b986...`, file SHA256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`,
TP=2, F16 KV, 262144 context, MTP off, lab doors off, alias `hotschmoe-dd`,
API key protected on port 18080. Preserve the obliterated shelf but stop it.

COMMAND -> reflink the exact verified model into
`models/files/qwen3.8-27b/q4km-ggml-org`; validate manifest/fetch plan; under
`gpu-run`, stop the obliterated DP=2 shelf and start
`rdy_to_serve/llamacpp/qwen38-27b-q4km/serve.sh`; run identity, Paris, Open
WebUI backend, and c1 128-token coding-smoke gates.

RESULT -> both cards healthy; full model SHA passed. Runtime logs show
`SYCL0,SYCL1 --split-mode tensor --tensor-split 1,1`, LAB_DOORS=0, MTP off.
API reports `hotschmoe-dd`, Q4_K_M, n_ctx/train_ctx 262144, 26.896B params.
Paris exact. Open WebUI healthy and sees `hotschmoe-dd`. Fresh coding smoke
averaged 35.2 tok/s, best 35.4, versus the historical matching 32.8 tok/s.

VERDICT -> GO live stock quality-first TP=2 daily driver. TP=2 preserves the
coding-qualified topology. DP=2 offers two lanes but changes KV/context
numerics and remains unqualified on the stock HumanEval+ gate.

### 2026-08-23n - TP=2 inference profile: synchronization dominates sglang decode

CONFIG -> 2x B70 under one both-card `gpu-run` lease; kernel 7.1 and compute
runtime 26.22.38646.4. Production stock Q4_K_M llama.cpp TP=2 baseline plus
Qwen3.6-27B compressed-tensors W8A8 sglang 0.5.6/0.5.15, TP=2, MTP10, eager,
8K, radix off. `CCL_TOPO_P2P_ACCESS=0` throughout. Compare the 1M shelf push
gate with `PUSH_AR_MIN_NUMEL=0`; no P2P-on serve or unsafe peer-write arm.

COMMAND -> current torch 2.12 oneCCL `scripts/allreduce_bench.py`; Level Zero
IPC push `scripts/106_run_ar_torch.sh`; stage-separated
`sglang/profile_w8a8_0515_vs_0506.sh`; shelf `serve.sh run` at push gates 1M
and 0; five-batch push-all mechanism trace. Logs and exact commands are in
`docs/20260823_tp2_inference_profile.md`.

RESULT -> oneCCL small-message floor 78-111 us at 4-64 KiB and large-message
plateau 1.15-1.24 GB/s. Push: 40 us at 10 KiB and 10.66 GB/s at 16/64 MiB.
Real default decode: 795 AR / five batches, 319/878 ms ranks 0/1 = 41.3%/65.9%
of device time; identical math time exposes a 2.75x collective imbalance.
Push-all mechanism trace: collective device time 8/8 ms, total device time
469/469 ms versus 774/1333, imbalance gone. End-to-end: c1 21.36 -> 22.48
(+5.2%), c4 aggregate 19.76 -> 19.81 (flat), TTFT 584 -> 595 ms, stable 2K
soak 16.36 -> 17.35 (+6.1%); coherence passed. Restored stock Q4_K_M daily
driver: identity/coherence pass, port 18080, ctx 262144; fresh baseline in this
session 37.88 tok/s.

VERDICT -> TP=2 communication synchronization is sglang decode bottleneck #1,
but raw P2P bandwidth is not: tiny payloads pay 128-159 serial boundaries and
host/runtime queue synchronization. Large prefill is bandwidth-bound and push
is already the right fix. After push-all, BF16 GEMM is 46.5%, INT8 GEMM 20.6%,
activation quant 6.8%, so math becomes the device target while eager host sync
limits wall realization. Next: A-B-B-A + serve-sweep before any gate promotion;
then remove/amortize host barriers and reduce boundary count. Do not pursue
P2P-on serve or broad direct peer writes.

### 2026-08-24a - TP=2 campaign C1/C2: push-all GO signal, one-host-sync NO-GO

CONFIG -> 2x B70 under serialized both-card leases, kernel 7.1, P2PACCESS=0.
C1: sglang Qwen3.6-27B W8A8 GPTQ, TP=2, MTP10, eager, 8K, radix off;
A-B-B-A with A `PUSH_AR_MIN_NUMEL=1048576`, B=0, fresh process per arm.
Each arm: identity/env, repeated phase timing, native c1/c4, cold prefill,
real-code c1/c4, 24-stream mixed coherence, 6.4K soak, stats/fatal/health.
C2: 159-call BF16 sequences at [1,11,44]x5120, oneCCL vs current push vs
one-host-sync async-safe, injected rank delay, exact output verification.

COMMAND -> `bin/gpu-run bash sglang/campaign_push_ar_abba.sh`; corrected C2
after a pre-collective `ccl`-backend setup failure by importing oneCCL and using
`xccl`, then `bin/gpu-run bash sglang/run_tp2_push_sync_microbench.sh`.
Artifacts: `results/logs/sglang_push_ar_abba_20260823T230835Z/` and
`results/logs/tp2_push_sync_20260824T010017Z/`.

RESULT -> C1 balanced deltas: native c1 +7.622%, extended soak +4.420%,
real-code c1 +6.364%, code c4 aggregate +3.315%, random c4 aggregate +0.340%,
c1/c4 TTFT +1.989%/+0.869% improvement, cold-prefill TTFT -0.037%/-0.800%.
All 96 mixed streams coherent; four 6.4K soaks coherent/stable; no fatal marker;
all health gates green. Entropy phase timing had 6.9-18.9% CV and restart
spread >5% in both arms, so that diagnostic invalidated its own gate. C2 us/call
at rows 1/11/44: oneCCL 133/170/373; current push 76/82/115; one-host-sync
safe 84/90/120. All exact. Production stock Q4_K_M restored after both blocks:
hotschmoe-dd, Q4_K_M, ctx 262144, health/coherence pass; cards free.

VERDICT -> C1 core serving result is a reproducible GO candidate; keep shelf
default at 1M until a deterministic final review replaces the invalid entropy
phase diagnostic. C2 one-host-sync is NO-GO: 4-10% slower than current push,
so do not risk native-event escalation or a serve port. C3 next: replicated MTP
input embedding removes 10/159 ARs exactly (~1.184 GiB/card), then XPU delayed
MLP-AR plus residual/RMSNorm fusion. Full campaign ledger in
`docs/20260823_tp2_optimization_campaign.md`.

### 2026-08-24b - TP=2 campaign C3a replicated embedding mechanism GO

CONFIG -> sglang 0.5.6 Qwen3.6-27B compressed-tensors GPTQ W8A8, TP=2,
MTP10, eager, radix off, context 131072, push gate 1048576, P2PACCESS=0.
Baseline uses the stock sharded input embedding. Candidate uses SGLang's
native full-table `VocabParallelEmbedding` on the target and shares that exact
module with NEXTN; LM head stays sharded. Additional table storage is exactly
1.184082 GiB/card. Feature is opt-in and defaults off.

COMMAND -> `bin/gpu-run bash
sglang/campaign_mtp_replicated_embedding_mechanism.sh`; after the baseline
profiler wrote both complete traces and segfaulted during teardown, reuse its
preserved trace/corpus and run only the candidate with `SKIP_BASELINE=1`.
Parse raw Kineto `record_param_comms` via
`sglang/parse_tp2_collective_census.py`; compare the eight-prompt canonical
greedy JSON with `cmp`. Artifacts are under
`results/logs/mtp_replicated_embedding_mechanism_20260824T013500Z/`; raw traces
are under `/mnt/vm_8tb/b70/sgl_cache/c3_mtp_replicated_embedding_20260824T013500Z/`.

RESULT -> both baseline ranks measured 159 all-reduces plus 11 all-gathers per
decode iteration. Both candidate ranks measured exactly 148 plus 11. The 11
removed calls were nine BF16 `[1,5120]` and two BF16 `[11,5120]` embedding
all-reduces per iteration; group `[0,1]`, dtype, async mode, and the 11 logits
all-gathers were unchanged. Both ranks logged full shape `(248320,5120)`, BF16,
2.368164 GiB and pointer-identical target/draft sharing. Eight fixed prompts,
2,048 output tokens total, were byte-identical (SHA256 `5638db8f...e54c`).
At context 131072, capacity changed 182208 -> 143360 tokens (78.68% retained),
leaving 12288 tokens of headroom and passing the 139264 hard gate. Candidate
profile trigger completed normally; baseline's post-trace profiler teardown
segfault was isolated from serving evidence. All card-health gates passed and
stock Q4_K_M production was restored at `hotschmoe-dd`, Q4_K_M, context 262144.

VERDICT -> C3a mechanism GO. This corrects the prior 10/159 estimate: native
target plus shared-draft replication removes 11 boundaries, giving 148. Keep
the feature default-off and advance to deterministic A-B-B-A serving
qualification. Do not infer an end-to-end win from profiler device time.

### 2026-08-24c - TP=2 campaign C3a A-B-B-A PASS and shelf promotion

CONFIG -> sglang 0.5.6 Qwen3.6-27B compressed-tensors GPTQ W8A8, TP=2,
MTP10, eager, radix off, context 131072, push gate 1048576, P2PACCESS=0.
A-B-B-A changed only `REPLICATE_MTP_EMBED`: A=0, B=1. Every fresh process
ran the eight-prompt deterministic corpus, native c1/c4, isolated prefill,
real-code c1/c4, 24-stream mixed coherence, 6.4K soak, fatal scan, and health.

COMMAND -> `bin/gpu-run bash
sglang/campaign_mtp_replicated_embedding_abba.sh`; analyze with
`sglang/analyze_mtp_replicated_embedding_abba.py`. Artifacts:
`results/logs/mtp_replicated_embedding_abba_20260824T021002Z/`.

RESULT -> drift-balanced candidate deltas: native c1 +4.690%, extended soak
+2.198%, code c1 +3.678%, code c4 aggregate +3.129%, standard c4 aggregate
+1.425%, and standard c4 per-stream +2.788%. Native c1 TTFT was flat
(-0.025%); native c4 TTFT improved +1.911%. Isolated prefill c1/c4 TTFT was
-1.689%/-0.785%, within the -2% gate. Both mirrored c1 and soak comparisons
favored B. All four deterministic outputs had identical SHA256
`5638db8f...e54c`; 96/96 mixed streams passed; four soaks were coherent and
stable at 1.05x; no fatal marker; every post-stop card probe passed. Candidate
capacity was 143360 versus baseline 182208 and passed both capacity gates.
Stock Q4_K_M production was restored: `hotschmoe-dd`, Q4_K_M, context 262144.

VERDICT -> GO and promote. Set `REPLICATE_MTP_EMBED=1` as the W8A8 shelf
default; rollback remains `REPLICATE_MTP_EMBED=0`. C3b is next: validate the
existing 63-edge delayed-MLP contract on XPU, then fuse push reduction with
residual add and Gemma RMSNorm. Test interaction with push-all before promoting
both communication levers together.

### 2026-08-24d - TP=2 campaign C3b delayed-MLP contract-only PASS

CONFIG -> sglang 0.5.6 Qwen3.6-27B compressed-tensors GPTQ W8A8, TP=2,
MTP10, replicated target/draft embedding on, eager, radix off, context 4096,
max requests 1, push gate 1048576, P2PACCESS=0. Candidate alone enables
`B70_XPU_DELAY_MLP_AR=1`. The fail-closed shim accepts only dense Qwen3.5,
TP=2/PP=1/EP=1, MoE-TP equal to TP, DP and quant communication off, BF16
contiguous `[M,5120]`, rows/request batch 1-128, and no graph capture. It
changes the upstream should-delay decision; the original prepare-attn generic
MoE-TP all-reduce plus Gemma RMSNorm remains the sole arithmetic consumer.

COMMAND -> `bin/gpu-run bash sglang/campaign_c3b_delayed_mlp_contract.sh`.
Run a fresh env-off baseline then env-on candidate, each with the shelf
coherence gate and the same eight-prompt deterministic corpus at 128 output
tokens. Require exact two-rank route counters, byte identity, fatal-log scan,
pre/post health, and exact stock production restoration. Artifacts:
`results/logs/c3b_delayed_mlp_contract_20260824T041922Z/`.

RESULT -> both candidate ranks emitted exactly `eligible=63 consumed=63
generic=63` on the first target forward. All eight baseline and candidate
responses completed at 128 tokens and their canonical JSON was byte-identical.
Both files had SHA256 `a962728a6bd977f1b5856309e4b13eaf58aefa335e2511cda9d51c9dc25a6c6b`.
Baseline and candidate model/env/mount identities passed; no device-lost,
out-of-resources, engine-dead, NaN, or garbage marker appeared. Every card
probe passed. Stock Q4_K_M production restored coherent at `hotschmoe-dd`,
context 262144. Analyzer verdict: PASS, contract-only, no performance claim.

VERDICT -> C3b lifecycle/group contract proven across all 63 non-final target
MLP edges. Do not promote the delay-only switch: it removes no collective,
launch, or host wait. Next build the true BF16 fused primitive in the same push
IPC library: push plus proven host rendezvous, then one asynchronous SYCL
reduce/residual/Gemma kernel with a scratch ring. Preserve `bf16(local+peer)`
before `bf16(ar+old_residual)`; do not reassociate the three-term sum. Gate in
a randomized two-rank numerical/stress microbench before any serve port.

### 2026-08-24e - C3b fused boundary kernel fast, serving integration NO-GO

CONFIG -> sglang W8A8 TP=2, MTP10, replicated input embedding, eager, push AR,
P2PACCESS=0. Candidate fuses peer reduction, BF16 residual add, and Gemma
RMSNorm for BF16 `[M,5120]`. It packs the peer copy as uint32 and uses aligned
16-byte vec8 loads/stores. Dispatch is fail-closed to bit-exact measured rows
M=1-8,10,11; M=9 and larger shapes keep SGLang's original immediate AR path.

COMMAND -> `bin/gpu-run env FAST_MAX_ROWS=11 WORKGROUP_SIZE=512
ROWS=1,2,3,4,5,6,7,8,9,10,11 STRESS_CALLS=1024 bash
sglang/run_fused_ar_rmsnorm_microbench.sh`; real mechanism via `bin/gpu-run
env CTX=131072 bash sglang/run_c3b_fused_mechanism.sh`; serving qualification
via `bin/gpu-run bash sglang/campaign_c3b_fused_boundary_abba.sh`. Artifacts:
`/mnt/vm_8tb/b70/fused_ar_rmsnorm/results/20260824_packed_vec8_m1_through_m11/`,
`results/logs/c3b_fused_mechanism_strict_rows_ctx131k_20260824/`, and partial
ABBA `results/logs/c3b_fused_boundary_abba_20260824T073940Z/`.

RESULT -> microbench candidate/gold speedup was 1.92x at M1, 1.93x M2,
1.92x M3-4, 1.88x M5-6, 1.80-1.81x M7-10, and 1.78x M11. M1-8,10,11 were
bit-exact across four adversarial cases; M9 had one 1-ULP cancellation
mismatch and is excluded. The 1024-call delayed ring stress kept residual and
cross-rank output exact. The strict real mechanism gate reached 40960 eligible
and consumed boundaries with generic=0 on both ranks and coherent 17.98 tok/s.
In the position-balanced serve run, B1 beat A1 on the fixed-content mechanism
soak 17.94 vs 16.01 (+12.1%) and regime soak 18.19 vs 16.37 (+11.1%), but lost
the mandatory paired phase result 13.89 vs 16.43 and perf c1 20.48 vs 21.51.
Coding c1 was flat 22.1 vs 22.2; c4 aggregate was within band at 72.1 vs 72.8;
24/24 mixed streams passed. The campaign was stopped after B1 because the two
predeclared per-pair win checks were already impossible to pass; cleanup and
both-card health passed, endpoint left down.

VERDICT -> kernel primitive GO, current delayed-boundary serving integration
NO-GO and not promoted. Keep the shelf flags default off. The mixed serving
metrics indicate that removing about 50 us from an isolated boundary does not
reliably improve end-to-end decode under all speculative workloads. Move to C4
post-communication math; retain C3b as a proven research primitive for a later
integration that removes more boundaries or host scheduling overhead.

### 2026-08-24f - Unsloth UD-Q4_K_XL matched profile and embedded MTP GO

CONFIG -> llama.cpp SYCL TP=2, native context 262144, F16 KV, P2PACCESS=0,
LAB_DOORS=0. Matched arms: stock Q4_K_M MTP-off, Unsloth UD-Q4_K_XL MTP-off,
and XL embedded NEXTN MTP3. Exact file sizes and SHA256 identities were checked
against the live container. Final restore was disabled for the chained campaign.

COMMAND -> `bin/gpu-run env RUN_MTP=1 RUN_EVIDENCE=1 RUN_HEPLUS=0
FINAL_RESTORE=none bash llamacpp/campaign_qwen38_ud_q4k_xl.sh full`.
Artifacts: `results/logs/qwen38_ud_q4k_xl_campaign_20260824T060230Z/`.

RESULT -> all campaign hard gates passed. Q4_K_M and XL produced the same 6/7
canary result and identical text hashes, including the shared modular miss.
MTP3 was byte-exact to XL MTP-off on all seven responses. Matched median decode
was Q4_K_M 35.57, XL MTP-off 34.44 (-3.18%), and XL MTP3 43.19 tok/s (+25.41%
vs XL MTP-off). Coding was 35.67, 34.04 (-4.56%), and 53.87 tok/s (+58.26%).
XL MTP-off prefill TTFT was 5.553 vs 4.814 s (+15.34%); MTP3 was 5.799 s.
Evidence captured 9 quant-type lines, 139 all-reduce census lines, and one
fusion exit. Cards stayed healthy and the endpoint was left down as requested.

VERDICT -> exact XL artifact and embedded MTP serve path are GO for final
quality qualification. XL changes the weight kernel workload: only 3/65
gate/up pairs are both Q4_K, so the Q4_K-only reordered SwiGLU custom op remains
inapplicable; communication/activation/GDN work still applies. Run full
HumanEval+ with MTP enabled before changing the production shelf default or
installing the tracked systemd unit. Add per-quant MMVQ device-time counters
before optimizing XL's Q5_K/IQ4_XS-heavy weight path.

### 2026-08-24g - C4 shaped math census and INT8 LM-head kernel gate GO

CONFIG -> sglang 0.5.6 Qwen3.6-27B compressed-tensors GPTQ W8A8, TP=2,
MTP10, replicated input embedding, eager push all-reduce, P2PACCESS=0. Parsed
the existing five-step post-C3 shaped decode traces. The first candidate keeps
the sharded BF16 LM head as a fallback and adds a load-time, symmetric
per-output-channel RTN INT8 copy. M=1 uses `int8_gemm_w8a16`; M>1 uses the
single-launch dynamic activation quantizer plus `int8_gemm_w8a8`.

COMMAND -> `python3 sglang/parse_tp2_math_census.py <both replicated DECODE
traces> --steps 5`; then `bin/gpu-run --card 0 docker run ... python3
/work/lmhead_int8_probe.py`; then the TP=2 load/coherence gate with
`bin/gpu-run env PUSH_AR_MIN_NUMEL=0 LMHEAD_INT8=1 ... serve.sh smoke`.
Artifacts: `results/logs/c4_math_census_replicated_20260824.tsv`,
`results/logs/c4_lmhead_int8_probe_20260824.log`, and
`results/logs/c4_lmhead_int8_smoke_20260824.log`.

RESULT -> the exact TP shard LM-head shape `[124160,5120]` dominates the math
trace: 45 M=1 calls cost 95.38 ms and 10 M=11 calls cost 21.46 ms, 116.84 ms
total over five scheduler steps, about 25% of rank-0 device time. GDN BF16
projections are second at about 58 ms. On the real rank-0 head weights, INT8
weight rel-L2 was 0.01062. M=1 improved 2.1328 -> 1.0797 ms (1.975x); M=11,
including activation quantization, improved 2.1546 -> 1.1795 ms (1.827x).
Both shapes were finite with 100% top-1 agreement in the probe. The full TP=2
candidate loaded, emitted coherent MTP output, engaged push AR, stopped
cleanly, and left both cards healthy. Endpoint remained down.

VERDICT -> isolated kernel and load gates GO. This is the highest-value C4
target ahead of GDN INT8. It is not promoted: the output head is
quality-sensitive and the candidate adds 0.592 GiB/card while retaining BF16.
Run position-balanced A-B-B-A serving gates, then HumanEval+ if performance
passes. Keep `LMHEAD_INT8=0` as the shelf default until both gates pass.

### 2026-08-24h - C4 hybrid INT8 LM head serving NO-GO; W8A16 repair gated

CONFIG -> same sglang W8A8 TP=2, MTP10, replicated embedding, push-all,
P2PACCESS=0, context 131072 C4 stack. A1 used the BF16 TP-sharded head. B1
quantized both target and draft heads per rank, retained BF16 fallback storage,
used W8A16 for M=1, and used dynamically quantized W8A8 for M>1. The endpoint
policy was down between arms and after the campaign.

COMMAND -> `bin/gpu-run bash sglang/campaign_c4_lmhead_int8_abba.sh`;
campaign stopped during B1 after the predeclared capacity and speculative
acceptance failures made promotion impossible. Then `bin/gpu-run --card 0
docker run ... python3 /work/lmhead_int8_probe.py` tested both INT8 routes at
M=1 and M=11. Artifacts:
`results/logs/c4_lmhead_int8_abba_20260824T090040Z/` and
`results/logs/c4_lmhead_int8_routes_probe_20260824.log`.

RESULT -> A1 passed coherence, 24/24 mixed streams, and a coherent 6400-token
soak at 16.41 tok/s with 1.08x first/last-window variation. A1 speculative
acceptance was normally about 0.40-0.76. B1 installed four INT8 heads (target
plus draft on both ranks) and reached more than 10000 routed calls/rank, but
all 32 reported acceptance samples were exactly 0.00 with accept length 1.00;
observed decode was about 4.8 tok/s. B1 also reduced token capacity from
143360 to 104576, a loss of 38784 slots or 27.05%, exactly matching its two
persistent 0.592-GiB INT8 copies/card. Cleanup stopped the candidate, both
cards passed health, and the endpoint remained down. The exact-shape repair
probe found W8A16 speedups of 1.965x at M=1 and 1.895x at M=11; W8A16 was
faster than W8A8 at both shapes and had lower M=11 relative L2 error (0.01068
vs 0.01348). Both routes had 100% top-1 agreement on the probe corpus, while
their M=11 outputs differed by relative L2 0.00822.

VERDICT -> this hybrid LM-head serving implementation is a hard NO-GO and
remains default off. Isolated top-1 agreement did not predict MTP behavior.
The candidate left the draft's independently created INT8 bundle attached
after SGLang shared the target BF16 head, and it also used different W8A16 and
W8A8 numerical routes for draft and target-sized projections; this run did not
isolate their individual contributions to the acceptance collapse. Repair both
conditions: use W8A16 for every head shape, quantize target once per rank,
replace BF16 storage before KV sizing, and alias the draft to that same INT8
weight and scale. Gate exact target/draft/rank mechanism markers, full context
capacity, acceptance, coherence, and balanced serving performance before
considering HumanEval+ or promotion.

### 2026-08-24i - Unsloth XL MTP3 per-quant route census PASS

CONFIG -> llama.cpp SYCL TP=2, Unsloth UD-Q4_K_XL, embedded NEXTN MTP3,
native context 262144, F16 KV, LAB_DOORS=0, P2PACCESS=0. A separately tagged
`qwen38-b70:quant-census` image used the exact production source commit and
two pinned optimization patches plus a third default-off counts-only patch.
The instrument adds no event timing, queue barrier, or wait.

COMMAND -> `bash llamacpp/qwen38-b70/build_image.sh`; inspect candidate and
production image IDs; then `bin/gpu-run bash
llamacpp/run_qwen38_ud_q4k_xl_quant_census.sh`. The fixed workload generated
512 coding tokens with embedded MTP3, stopped gracefully, and parsed at-exit
rows with `llamacpp/parse_quant_census.py`. Artifacts:
`results/logs/qwen38_ud_q4k_xl_quant_census_20260824T101342Z/`.

RESULT -> coherent 512/512 completion. Logical and actual callback totals both
equaled 149724. MMVQ accounted for 146716 calls (97.99%); DEQ_GEMM was 3008
(2.01%). Width 4 dominated at 137360 calls (91.74%). By quant type, Q5_K was
53862 calls (35.97%), Q8_0 32644 (21.80%), Q6_K 21482 (14.35%), IQ4_XS 19740
(13.18%), and Q4_K 19176 (12.81%). Using exact packed block sizes and treating
each actual callback as one full packed-weight read gives an attribution
estimate of 2.696 TiB total: Q5_K 37.70%, Q6_K 33.45%, IQ4_XS 14.86%, and Q4_K
11.15%. The largest individual weight-volume shapes were the Q6_K vocab head
`5120x124160` at width 1 (14.56% across both devices), Q5_K `5120x8704` width 4
(14.34%), and IQ4_XS `5120x8704` width 4 (10.00%). Both cards passed final
health and the endpoint remained down.

VERDICT -> counts-only mechanism PASS and exact XL kernel ordering established.
Prioritize width-4 Q5_K MMVQ, then the Q6_K vocab/width-4 routes, then IQ4_XS.
Q4_K-only fusion is not the main XL lever. Add minimally perturbing per-route
event timing before claiming device-time shares; callback counts and packed
byte estimates are attribution, not latency measurements. The XL MTP3
HumanEval+ gate remains required before shelf promotion.

### 2026-08-24j - C4 repaired shared W8A16 LM-head mechanism PASS

CONFIG -> sglang W8A8 TP=2, MTP10, context 131072, max requests 4, replicated
embedding, push-all, C3b off, P2PACCESS=0. The repaired default-off candidate
quantizes the target head once per rank, replaces its BF16 Parameter storage,
aliases the draft to the same INT8 weight and FP16 scale before KV sizing,
asserts SGLang's later official share, and uses W8A16 for every row count.

COMMAND -> exact-shape single-card route probe via `bin/gpu-run --card 0
docker run ... python3 /work/lmhead_int8_probe.py`; then `bin/gpu-run bash
sglang/run_c4_lmhead_int8_mechanism.sh`. Artifacts:
`results/logs/c4_lmhead_int8_routes_probe_20260824.log` and
`results/logs/c4_lmhead_int8_mechanism_20260824T102120Z/`.

RESULT -> W8A16 improved the real TP shard by 1.965x at M=1 and 1.895x at
M=11, faster and lower-error than W8A8 at both shapes. The full serve emitted
the exact four target/draft/rank ready identities and two shared-rank markers;
all four routes exceeded 100 calls. Token capacity was 201600, up 40.62% from
the BF16 baseline 143360 and 92.78% from the broken candidate 104576. The fixed
640-token response was coherent. Fixed-request accept samples had rate
0.16-0.24 and length 2.58-3.45; final internal average accept length was 3.785,
so the previous exact-zero collapse was eliminated. Eight deterministic
prompts were nonempty and the concurrent gate passed 4/4. All hashes/config
checks passed, no fatal marker appeared, both cards stayed healthy, and the
endpoint remained down. Analyzer verdict: PASS.

VERDICT -> repaired memory/lifecycle/acceptance mechanism GO, but not yet a
performance or quality promotion. Acceptance is lower than the usual BF16
range, so run the full position-balanced A-B-B-A now in progress. Require the
predeclared serving gains and nonregressions; run HumanEval+ only if performance
passes. Keep `LMHEAD_INT8=0` as the shelf default.

### 2026-08-24k - C4 repaired shared W8A16 LM-head serving NO-GO

CONFIG -> sglang W8A8 TP=2, MTP10, context 131072, replicated embedding,
push-all, C3b off, P2PACCESS=0. Position-balanced A-B-B-A compared the BF16
TP-sharded head against the repaired candidate that replaces target BF16
storage with one per-rank INT8 copy, aliases draft storage and scale, and uses
W8A16 for every target/draft row count. Artifacts and source hashes were frozen
across all four arms. The endpoint remained down between arms and after exit.

COMMAND -> `bin/gpu-run bash sglang/campaign_c4_lmhead_int8_abba.sh`.
Artifact: `results/logs/c4_lmhead_int8_abba_20260824T103059Z/`.

RESULT -> all four arms passed exact config/model identity, eight deterministic
responses were byte-identical across every comparison, mixed serving passed
24/24 per arm, every soak stayed coherent, no fatal marker appeared, and all
card-health checks passed. Candidate mechanism evidence was exact on both
ranks: target storage replaced, draft storage aliased, shared weight/scale
asserted, and all four role/rank W8A16 routes exercised. Capacity increased
143360 -> 201600 tokens (+40.62%). Balanced geometric-mean deltas were phase
decode +8.42%, warm c1 +5.58%, coding c1 +5.31%, warm c4 aggregate +1.52%,
coding c4 aggregate +2.76%, and 6400-token decode -0.23%. Prefill and TTFT were
inside the nonregression bands. The 2000-token regime soak was repeatably about
20.7 tok/s on both candidates versus 17.68 on both baselines, but the required
long soak did not confirm it: A1/B1/B2/A2 were 17.06/15.93/18.42/17.28 tok/s.
B1 degraded 20.93 -> 12.08 tok/s across its long windows (1.73x), while B2 was
stable at 1.05x. Phase medians also had high within-arm CV and the closing A2
baseline beat B2. Formal analyzer verdict: FAIL on soak stability, both-pair
phase and soak wins, long-soak gain, within-process CV, and restart spreads.

VERDICT -> current W8A16 LM-head serve path is NO-GO and remains default off.
The isolated 1.9x head kernel speed and repeatable roughly 5% c1 gains are real
research signals, but they do not survive the mandatory long-serving gate.
The exact deterministic canary does not show a quality regression, yet the
candidate changes speculative delta counts on longer continuations, so do not
spend a HumanEval+ run or promote the shelf. Retain the default-off prototype
for controlled fixed-token/device-time work; move engineering effort to the
measured XL per-quant routes and the remaining GDN projections.

### 2026-08-24l - llama.cpp main-queue event profiling NO-GO

CONFIG -> llama.cpp SYCL TP=2, Unsloth UD-Q4_K_XL, MTP off, native context
262144, F16 KV, LAB_DOORS=0, P2PACCESS=0. Candidate image
`sha256:5029a9d394eacd46b48686b564fcc93a410c27a6b1064630008eaec83ef748d1`
used the exact production source and optimization patches plus default-off
counts and SYCL-event timing instrumentation. The timing path enabled queue
profiling only when sampling was requested. A follow-up single-start probe set
the timing skip to UINT64_MAX, so no timing barrier, atexit registration, or
event timestamp read could execute, and also set
`UR_L0_USE_DRIVER_INORDER_LISTS=0`. Restart policy was fixed to propagate to
the container and set to `no`. The endpoint stayed down throughout.

COMMAND -> `bin/gpu-run bash
llamacpp/01_qwen38_ud_q4k_xl_quant_timing_campaign.sh full`; after interrupting
the unintended restart loop and fixing restart propagation, `bin/gpu-run bash
llamacpp/02_qwen38_ud_q4k_xl_profile_queue_probe.sh`. Artifacts:
`results/logs/qwen38_ud_q4k_xl_quant_timing_20260824T122012Z/` and
`results/logs/qwen38_ud_q4k_xl_profile_queue_20260824T125710Z/`.

RESULT -> production and candidate timing-off arms were byte-identical on all
seven deterministic canaries. Five 256-token decode medians were 32.4421 and
33.1043 tok/s respectively; the candidate was +2.04%, just outside the strict
absolute 2% inertness band, so the formal gate was not relaxed. Counts-only
completed coherently at 32.2041 tok/s with exact logical/actual total 297206;
98.07% of callbacks were MMVQ and width 1 dominated. Every profiling-enabled
start failed during model load with `UR_RESULT_ERROR_DEVICE_LOST` at MUL_MAT
after exactly 11 logical and 11 actual callbacks. Each route had at most two
calls, below timing skip 4, and no `[QUANT-TIMING]` row appeared. The decisive
restart-disabled probe reproduced the same failure with skip UINT64_MAX and
driver in-order lists disabled. Its failure frontier was device-0 IQ4_XS
`K=5120 rows=8704`; the corresponding device-1 callback was absent. Both cards
passed health after cleanup.

VERDICT -> main-queue `sycl::property::queue::enable_profiling` is a hard
NO-GO on this B70 TP=2 oneAPI 2025.3 stack. The evidence excludes the timing
barriers and timestamp reads because none were reachable. Do not use event
profiling on the production queues and do not retag production. Replace it
with a separate default-off evidence image that uses normal queues and sparse
synchronized host-clock timing around complete logical TP=2 matmuls. Treat
those results only as isolated-operation attribution because the waits remove
normal pipeline overlap. Keep the endpoint down until the research campaign
finishes or the user explicitly requests service.

### 2026-08-24m - llama.cpp full-process VTune launch NO-GO

CONFIG -> candidate-only llama.cpp SYCL TP=2 Unsloth UD-Q4_K_XL mechanism
gate, MTP off, P2PACCESS=0, all queue/source profiling variables forced off.
The first arm ran normally. The second launched the same server as a child of
VTune 2025.10 `gpu-offload`, with collection start-paused, both B70 PCI
adapters selected, call-stack and characterization collection disabled, and
restart policy `no`. A 20-minute health deadline and endpoint-down/no-restore
cleanup were predeclared.

COMMAND -> `bin/gpu-run bash
llamacpp/run_qwen38_ud_q4k_xl_vtune_gate.sh full`. The first invocation exited
before server start because the empty-port guard returned `rg` no-match instead
of explicit success; the guard was fixed and the exact gate rerun. Artifacts:
`results/logs/qwen38_ud_q4k_xl_vtune_20260824T131519Z/` and
`results/logs/qwen38_ud_q4k_xl_vtune_20260824T131636Z/`.

RESULT -> the normal reference became healthy, produced exact 32- and
512-token responses, and measured 31.6456 tok/s post-first-token with 0.2602 s
TTFT. Its fresh teardown and both-card health passed. The VTune child never
became healthy before the 20-minute deadline. It remained alive at about one
full CPU core and 15.2% host memory, `/health` consistently reported `Loading
model`, and logs contained no Level Zero, device-lost, or out-of-resource
error. VTune's Pin launcher was still instrumenting the large SYCL process;
collection was never resumed and no timed request ran. The trap stopped and
removed the only trace container. Port 18080 remained down and a post-cleanup
leased health probe passed both cards.

VERDICT -> launching the complete llama.cpp process under VTune Pin is NO-GO
for this campaign: even paused collection changes startup enough to miss a
20-minute service gate, so it cannot provide minimally perturbing decode
evidence. This is a profiler mechanism failure, not a GPU stability failure or
a model performance result. Keep production tags unchanged. Next isolate an
attach-after-load VTune path with scoped ptrace permission so normal model
initialization is untouched; if attach is also too invasive, fall back to
normal-queue sparse synchronized logical-op timing and label it isolated
service-demand attribution only. Endpoint remains down.

### 2026-08-24n - llama.cpp VTune attach captures decode but detach NO-GO

CONFIG -> same candidate-only XL TP=2 MTP-off config and normal unprofiled SYCL
queues. Both reference and trace servers loaded normally with restart `no`.
Only the trace container received `CAP_SYS_PTRACE`; Docker default seccomp,
container PID namespace, and non-privileged mode were retained. After health
and a 32-token warmup, VTune 2025.10 `gpu-offload` attached to container PID 1,
status proved `PID=1 STATE=RESUME NAME=llama-server`, and exactly one
deterministic 512-token request ran before `vtune -command stop`.

COMMAND -> `bin/gpu-run bash
llamacpp/03_qwen38_ud_q4k_xl_vtune_attach_gate.sh full`. The harness wait was
interrupted only after the hard detach-survival gate was impossible: the target
had already exited. Offline `vtune -finalize` was then attempted without GPU
devices. Artifact:
`results/logs/qwen38_ud_q4k_xl_vtune_attach_20260824T135239Z/`.

RESULT -> normal reference and attached requests both completed exact 512
tokens with identical SHA256
`438c77bec0d18bf7430e4e5c7b3b7c80d91aeab69874768b8626f89a077af203`.
Reference decode/TTFT were 32.1011 tok/s and 0.2635 s; traced were 27.7284
tok/s and 0.3420 s, 86.38% decode and 1.298x TTFT. Decode stayed above the 85%
floor but TTFT missed the 1.25x ceiling. Attach readiness took about 5.1 s and
the collection ran. However, `vtune -command stop` terminated the attached
PID-1 server with exit 255 instead of detaching while leaving it healthy. The
detached exec could not write its completion status because its container died,
so the result was not finalized. Offline finalization failed with VTune
`0x4000001e Cannot load raw collector data`. No timing report is trustworthy.
Fail-closed cleanup removed the stopped container; port 18080 was down and both
cards passed final leased health. No Level Zero or device-lost error appeared.

VERDICT -> attach-after-load VTune is also NO-GO as a campaign profiler on this
stack. It captured coherent decode with tolerable throughput overhead, but its
stop path kills the served process, exceeds the TTFT perturbation band, and
leaves an unusable raw result. Do not broaden container privileges or retry the
same Pin mechanism. External VTune is closed alongside profiled SYCL queues.
Use normal-queue sparse synchronized logical-op timing only as explicitly
isolated service-demand evidence, or advance directly from the exact route and
byte census to measured kernel candidates. The next low-risk serving candidate
is the existing GDN-INT8 artifact through current W8A16/W8A8 kernels. Endpoint
remains down.

### 2026-08-24o - C4 target-GDN INT8 mechanism fails only declared BA scope

CONFIG -> sglang W8A8 TP=2, Qwen3.6-27B SQ-GPTQ target plus MTP10/draft11,
graph and radix cache off, P2P access off, promoted replicated MTP embedding and
push-all enabled, all experimental C3b and LM-head switches off. The candidate
checkpoint contains 144 target GDN INT8 weights and 144 BF16 scales; a generated
read-only config overlay replaced only compressed-tensors metadata and declared
`linear_attn.in_proj_ba`, MTP, vision, and `lm_head` ignored. The mechanism gate
required exact route counts on both ranks, capacity non-regression, fixed and
same-process deterministic generation, mixed concurrency, soak stability,
artifact immutability, health, and endpoint-down cleanup.

COMMAND -> `./bin/gpu-run bash sglang/01_c4_gdn_int8_mechanism.sh`.
Artifact: `results/logs/c4_gdn_int8_mechanism_20260824T141330Z/`.

RESULT -> checkpoint, overlay, mount, container, served identity, fused-kernel,
artifact, and both health audits passed. Capacity increased 143360 -> 226688
tokens (+58.1%) and exact TP=2 target-weight residency saving remained
2,766,962,688 bytes/rank. The fixed 640-token request, two deterministic
eight-prompt corpora, 24/24 mixed streams, and a 1600-token 21.45 tok/s soak
were coherent; the soak first/last ratio was 1.00x. Speculation stayed live
with server average accepted length 4.466. Both rank traces agreed exactly:
240 target qkvz and 320 combined out-projection W8A8 calls over five steps,
zero BF16 target qkvz, five preserved BF16 MTP out and qkv calls, and 320
K3072 activation quantizations. The intended large GDN projection kernels used
19.66-19.67 ms qkvz plus 14.80-14.85 ms out per rank, versus about 35.73 plus
22.25 ms in the baseline census, a 40.5% combined device-time reduction.
However, both ranks also showed 240 W8A8 N48 calls and 880 K5120 activation
quantizations: packed `in_proj_ba` routed W8A8 instead of the declared 240 BF16
calls and expected 640 quantizations. The unexpected tiny GEMM itself cost only
1.31-1.32 ms, but it violated the predeclared candidate boundary.

VERDICT -> formal FAIL on route scope only; no performance GO and no shelf
change. Coherence, stability, capacity, and intended large-projection kernel
evidence passed. Diagnose whether compressed-tensors matching occurs against a
packed loader name rather than `linear_attn.in_proj_ba`, make the smallest
metadata-only correction that demonstrably retains BA in BF16, and rerun the
same narrow mechanism gate before any balanced A-B-B-A. Keep the endpoint down
until the research campaign is finished or the user explicitly requests it.

### 2026-08-24p - C4 packed BA ignore root cause and metadata fix

CONFIG -> CPU-only inspection of the exact `sglang-xpu:mtp` image, its
compressed-tensors `should_ignore_layer` implementation, Qwen3.5
`packed_modules_mapping`, the candidate safetensors metadata, and layer-0 BA
values. No GPU or endpoint was used. The failed config ignored the runtime
fused name with `re:.*linear_attn\.in_proj_ba$`.

COMMAND -> call `should_ignore_layer` for
`model.layers.0.linear_attn.in_proj_ba` with the model mapping
`in_proj_ba: [in_proj_b, in_proj_a]`, first with the fused regex and then with
both checkpoint-leaf regexes. Run `python3 -m unittest
sglang.tests.test_c4_gdn_int8_static`, `py_compile`, the checkpoint audit, ASCII
checks, and `git diff --check` after replacing the rule in config, generator,
analyzer, tests, and plan.

RESULT -> SGLang expands `in_proj_ba` to the two checkpoint leaves before
testing ignores. The old fused regex returned false; the paired `in_proj_b` and
`in_proj_a` regexes returned true. With the old rule, SGLang allocated a packed
INT8 BA parameter without any checkpoint scale and the packed loader used
`copy_` to cast the BF16 leaf weights into INT8 storage. Across all 96 BA leaf
tensors, all 23,592,960 coefficients had absolute value below 1 and became zero
at runtime; no BA scale tensor existed. This was hidden runtime representation mutation, not an on-disk
rewrite; unchanged checkpoint hashes and coherent output did not validate it.
The corrected metadata ignores both leaves, preserves the exact BF16 BA route
contract, and rejects any BA `weight_scale`. Four static tests, syntax, the
144-weight audit, exact 2.577 GiB/rank saving, ASCII, and diff checks passed.

VERDICT -> root cause confirmed and smallest correction is metadata-only. Do
not reinterpret the first mechanism result as an expanded-scope optimization:
its BA path was invalid. Rerun the unchanged GPU mechanism gate and require
exactly 240 BF16 BA calls plus 640 K5120 activation quantizations per rank
before any A-B-B-A. Endpoint remains down by campaign policy.

### 2026-08-24q - corrected C4 target-GDN INT8 mechanism PASS

CONFIG -> identical fail-closed TP=2 mechanism gate from 2026-08-24o, with the
only candidate correction being compressed-tensors ignores for both checkpoint
leaves `linear_attn.in_proj_b` and `linear_attn.in_proj_a`. The BF16 BA route
contract remained fixed at 240 calls and K5120 activation quantization at 640
calls over five steps per rank. Production restoration remained disabled.

COMMAND -> `./bin/gpu-run bash sglang/01_c4_gdn_int8_mechanism.sh`.
Artifact: `results/logs/c4_gdn_int8_mechanism_20260824T143518Z/`.

RESULT -> formal analyzer PASS. Both rank traces matched every exact route:
target qkvz W8A8 240, combined out W8A8 320, target qkvz BF16 0, preserved MTP
out/qkv BF16 5/5, corrected BA BF16 240, and K5120/K3072 quantization 640/320.
Total device time was nearly symmetric at 458.831/458.811 ms. Candidate qkvz
and out kernels used 19.662/14.917 ms on rank 0 and 19.709/14.890 ms on rank 1.
Capacity was 226368 versus the 143360 baseline. A fixed 640-token response was
coherent, two deterministic eight-prompt corpora were byte-identical, mixed
load passed 24/24, and the 1600-token soak was coherent and stable at 17.62
tok/s with a 1.00x first/last ratio. Server accepted-length average was 4.488.
Every checkpoint, overlay, container, served-id, artifact immutability, fatal
marker, pre/post health, and endpoint-down check passed. Both cards passed an
additional leased exit probe, and the command exited 0 after 773 seconds.

VERDICT -> corrected mechanism GO; the candidate is valid for performance
testing, not yet for the shelf. Run the predeclared position-balanced A-B-B-A
against the current W8A8 shelf and require both phase pairs, balanced phase and
soak thresholds, prefill/TTFT bands, restart/CV stability, byte identity, mixed
coherence, and health. Endpoint remains down by campaign policy.

### 2026-08-24r - combined target-GDN INT8 A-B-B-A NO-GO

CONFIG -> one continuous dual-card lease and four fresh sglang TP=2 serve
lifecycles in A-B-B-A order. A used the current W8A8 SQ-GPTQ shelf checkpoint
and native config. B used the corrected target-GDN INT8 checkpoint with the
read-only two-leaf metadata overlay. Both used MTP10/draft11, graph/radix off,
P2P access off, promoted push-all and replicated MTP embedding on, and C3b and
LM-head experiments off. Predeclared gates required both phase pairs to win,
balanced phase >=3%, both 6400-token soak pairs to not regress, balanced soak
>=2%, TTFT/prefill within 5%, B phase CV and restart spreads <=5%, byte-exact
B restart outputs, mixed coherence, exact runtime identity, immutable
artifacts, health, and endpoint-down cleanup.

COMMAND -> `./bin/gpu-run bash sglang/02_c4_gdn_int8_abba.sh`.
Artifact: `results/logs/c4_gdn_int8_abba_20260824T145553Z/`.

RESULT -> formal analyzer FAIL after 5902 seconds. Position-balanced deltas
were phase decode -17.107%, warm c1 -8.528%, c4 stream -0.187%, c4 aggregate
+1.529%, and long soak +2.888%. Phase medians were A1/B1/B2/A2
20.634/14.926/16.647/17.526 tok/s: both matched B pairs lost. B phase CVs were
18.86%/6.64%, and B restart phase spread was about 10.9%, so all three phase
gates failed. Warm c1 reproduced at A1/B1/B2/A2 22.76/21.03/20.90/23.08.
Long soaks were 17.29/17.89/17.73/17.33 tok/s: both B pairs won, the balanced
gain cleared 2%, and B restart spread was about 0.9%. Candidate c4 aggregate
was a smaller repeatable signal at 20.37/20.72 versus baseline 20.28/20.19.
All TTFT and prefill deltas stayed within 3.2%. The strict all-soak stability
check also failed because A1 printed 1.11x against the analyzer's 1.10x ceiling;
B1/B2 were both 1.05x and A2 was 1.10x. This baseline-only miss does not alter
the decisive phase and c1 rejection.

All four fixed responses, deterministic corpora, 24/24 mixed gates, and long
soaks were coherent. B1/B2 fixed outputs and eight-prompt corpora were byte
identical. Every checkpoint audit, model/id, config mount, environment, feature
marker, artifact hash, fatal-marker, per-arm health, and final health check
passed. No Level Zero, oneCCL, P2P, or cross-card stability failure occurred.
The endpoint remained down.

VERDICT -> combined qkvz plus out-projection INT8 is a serving NO-GO and stays
default-off. It produces a genuine +2.89% sustained-decode and +1.53% c4 signal
plus 2.577 GiB/rank capacity saving, but separate small-M activation quant and
dispatch costs erase the 40.5% projection-kernel win at c1. Next build and gate
an out-projection-only candidate, where the kernel economics were strongest and
48 qkvz quant/dispatch sequences per step can be removed. If that split retains
the soak/c4 gain without c1 loss, then pursue fused or reused GDN activation
quantization. Endpoint remains down by campaign policy.

### 2026-08-24s - GDN out-projection-only INT8 mechanism PASS

CONFIG -> new compressed-tensors artifact
`w8a8-sqgptq-gdn-out-proj-int8` with exactly 48 target GDN `out_proj` INT8
weights and 48 BF16 scales copied byte-for-byte from the combined source. All
qkv/z/b/a leaves remain base-checkpoint BF16 and scale-free; unchanged auxiliary
files are hardlinks. The dedicated overlay ignores both packed qkvz leaves and
both packed BA leaves. The candidate-only TP=2 mechanism retained MTP10/draft11,
graph/radix off, P2P access off, promoted push-all and replicated MTP embedding,
and endpoint-down cleanup. Its trace contract required qkvz and BA BF16, only
target out projection INT8, and exact activation-quant counts on both ranks.

COMMAND -> `./bin/gpu-run bash
sglang/03_c4_gdn_out_proj_int8_mechanism.sh`. Artifact:
`results/logs/c4_gdn_out_proj_int8_mechanism_20260824T164538Z/`.

RESULT -> formal analyzer PASS after 758 seconds. Both rank traces matched
exactly over five steps: W8A8 qkvz 0, BF16 qkvz 240, W8A8 out shape 320,
preserved BF16 MTP out 5, BF16 BA 240, preserved BF16 MTP qkv 5, and K5120/
K3072 activation quantization 400/320. Rank device totals were closely matched
at 462.576/461.939 ms. Target out W8A8 kernels used 14.820 ms on each rank;
qkvz remained BF16 at 35.386/35.365 ms. Capacity increased 143360 -> 164992.
The artifact saves 1,509,457,920 checkpoint bytes and 754,483,200 bytes/rank at
TP=2. The fixed 640-token output, two byte-exact deterministic corpora, 24/24
mixed requests, and 1600-token 17.03 tok/s soak were coherent; soak first/last
was 1.00x and server average accepted length was 4.374. All audit, overlay,
container, served-id, hash, fatal-marker, health, and endpoint-down checks
passed. Both cards passed the additional leased exit probe.

VERDICT -> out-projection-only mechanism GO; no shelf or performance claim yet.
It successfully removes the qkvz INT8 activation-quant/dispatch sequence while
preserving the intended out-projection kernel and 0.703 GiB/rank saving. Run
the same strict position-balanced A-B-B-A against the current shelf. Endpoint
remains down by campaign policy.

### 2026-08-24t - GDN out-projection-only INT8 A-B-B-A NO-GO

CONFIG -> one continuous dual-card lease and four fresh sglang TP=2 serve
lifecycles in A-B-B-A order. A used the current W8A8 SQ-GPTQ shelf checkpoint
and native config. B used the audited out-projection-only INT8 checkpoint with
exactly 48 target GDN out-projection INT8 weights, 48 BF16 scales, and the
read-only corrected overlay; qkv/z/BA and MTP remained BF16. Both variants used
MTP10/draft11, graph/radix off, P2P access off, promoted push-all and replicated
MTP embedding on, and C3b and LM-head experiments off. Predeclared gates were
unchanged from the combined-candidate A-B-B-A: both phase pairs win, balanced
phase >=3%, both 6400-token soak pairs nonregress, balanced soak >=2%,
TTFT/prefill within 5%, candidate phase CV and restart spreads <=5%, byte-exact
candidate restart outputs, mixed coherence, exact runtime identity, immutable
artifacts, health, and endpoint-down cleanup.

COMMAND -> `./bin/gpu-run bash sglang/04_c4_gdn_out_proj_int8_abba.sh`.
Artifact: `results/logs/c4_gdn_out_proj_int8_abba_20260824T170145Z/`.

RESULT -> formal analyzer FAIL after 5878 seconds. Position-balanced deltas
were phase decode -9.439%, warm c1 +0.611%, c4 stream +3.812%, c4 aggregate
+1.100%, and long soak -4.956%. Phase medians were A1/B1/B2/A2
18.921/15.806/17.404/17.728 tok/s: both candidate pairs lost by 16.461% and
1.827%. B phase CVs were 18.37%/15.81%, and restart phase spread was 9.620%,
so the phase gain, within-process CV, and restart gates all failed. Long soaks
were 17.43/16.50/16.48/17.27 tok/s: both candidate pairs lost by 5.336% and
4.574%; B restart soak spread was only 0.121%, so the sustained regression was
repeatable. Capacity remained 164992 for B versus 143360 for A.

All four soaks were coherent and stable, all 96 mixed streams passed, and
TTFT/prefill deltas stayed within 3.151%. B1/B2 deterministic eight-prompt
corpora were byte-identical, but their separate fixed outputs were coherent
and not byte-identical. Every checkpoint audit, config mount, model/id, feature
marker, artifact hash, fatal-marker, per-arm health, and final health check
passed. No Level Zero, oneCCL, P2P, or cross-card stability failure occurred.
The endpoint remained down and both cards were healthy at exit.

VERDICT -> out-projection-only INT8 is a serving NO-GO and stays default-off.
The 0.703 GiB/rank capacity saving and small warm-c4 signal do not compensate
for the repeatable phase and sustained-decode losses. Together with the
combined-candidate result, this identifies separate small-M activation
quantization and dispatch boundaries as the next leverage point. Prioritize a
shared/reused GDN activation quantization path; retain qkvz-only as a bounded
K5120-versus-K3072 attribution probe, not as a presumed shelf candidate.
Endpoint remains down by campaign policy.

### 2026-08-24u - llama.cpp TP=2 SYCL queue-profiling root cause isolated

CONFIG -> rebuilt `qwen38-b70:quant-timing` from pinned llama.cpp commit
`4302fb599` plus the pinned TP=2/Q4_K_XL and repository census/timing patches.
The build exposed and repaired a latent bad final hunk in the quant-census
patch; the complete stack then applied and compiled. The image labels the exact
timing-patch SHA. Two fresh UD-Q4_K_XL TP=2 MTP-off arms differed only in
`GGML_SYCL_QUANT_TIMING_QUEUE_PROFILE=0/1`. Both used sample 64, skip
18446744073709551615, and restart policy `no`, making timing barriers and
timestamp reads unreachable. The ordinary-queue arm had to become healthy
before the profiling-queue arm, with card health checked before, between, and
after the arms.

COMMAND -> `./bin/gpu-run bash
llamacpp/04_qwen38_ud_q4k_xl_queue_profile_isolation.sh full 2`. Artifact:
`results/logs/qwen38_ud_q4k_xl_queue_profile_isolation_20260824T185644Z_tp2/`.

RESULT -> analyzer PASS with classification `queue_property_root_cause` after
554 seconds. The profiling-off arm loaded healthy, returned a coherent Paris
response, exposed exact `hotschmoe-dd` identity, emitted zero timing records,
and stopped cleanly. The profiling-on arm exited once with
`UR_RESULT_ERROR_DEVICE_LOST` at `Error OP MUL_MAT` after exactly 11 actual
quant callbacks and before any timing record. Both arms retained restart policy
`no` and RestartCount 0. Identity, environment, code-hash, no-barrier,
endpoint-down, and pre/between/final health gates all passed. Both cards were
healthy at exit.

VERDICT -> merely constructing the SYCL queue with `enable_profiling` is the
TP=2 device-loss trigger on this stack. The cause is not an event-timing
barrier, timestamp read, restart chain, raw P2P failure, or the counts-only
census instrumentation. Do not use profiled queues or event timestamps here;
retain counts-only census and nonprofiled external methods. Next run the
exact-M=11 W8A16 versus current W8A8 versus BF16 kernel ledger. The endpoint
remains down by campaign policy.

### 2026-08-24v - exact-M=11 W8A16 kernel ledger GO

CONFIG -> one leased Arc Pro B70 card, production `sglang-xpu:mtp` image, and
the production W8A8 kernel SO. Each exact Qwen3.6-27B TP=2 per-rank shape used
M=11 and compared BF16 against the complete current BF16-to-FP16,
dynamic-activation-quant, W8A8, BF16-output chain and the candidate
BF16-to-FP16, quant-free W8A16, BF16-output chain. Both INT8 paths shared the
same `[K,N]` stride-0-1 B_nt weight view used by the live Sglang shim. Shapes
and target-step call weights were GDN qkvz 5120x8192 x48, GDN/attention out
3072x5120 x64, MLP gate-up 5120x17408 x64, MLP down 8704x5120 x64, and
attention qkv 5120x7168 x16. Two forward/reverse repeat blocks used 20 warmups
and 100 timed iterations, XPU-event plus synchronized-wall timing, numerical
checks, P2P access off, and pre/post card health. No server or endpoint action
was performed.

COMMAND -> `./bin/gpu-run --card 0 bash
sglang/05_c4_m11_w8a16_microbench.sh`. Artifact:
`results/logs/c4_m11_w8a16_microbench_20260824T191407Z/`.

RESULT -> formal PASS in 127 seconds. W8A16 device-time gains versus the full
current W8A8 chain were GDN qkvz 37.102%, GDN/attention out 35.562%, MLP
gate-up 18.837%, MLP down 38.541%, and attention qkv 36.904%. The qkvz/out
weighted gain was 36.227%, from 16.856 to 10.750 ms per target-step ledger;
the all-route weighted gain was 31.075%, from 43.965 to 30.303 ms. Candidate
device CVs were 0.029-1.313%, and current W8A8 CVs were 0.177-0.726%. All
outputs were finite. W8A16 relative L2 error versus BF16 was 0.00885-0.00939,
lower than W8A8's 0.01235-0.01316 on every shape. All artifacts retained their
hashes, card 0 passed both health probes, both leases were free at exit, and
the endpoint remained down.

VERDICT -> kernel-ledger GO for a default-off `B70_W8A16_M_MAX=11` Sglang
serving mechanism. This is a higher-leverage candidate than unfused GDN INT8:
it removes activation quantization from the dominant speculative M=11 path,
improves the local numerical approximation, and reuses the existing weight
layout without the old vLLM duplicate-residency cost. Do not make a shelf or
end-to-end speed claim until exact runtime routing, deterministic output,
mixed-load coherence, capacity, and balanced serving gates pass. Endpoint
remains down by campaign policy.

### 2026-08-24w - TP=2 M<=11 W8A16 serving mechanism GO

CONFIG -> the native Qwen3.6-27B SQ-GPTQ W8A8 checkpoint at 131072 context,
MTP10/draft11, eager/radix off, push-all, replicated MTP embedding, P2P access
off, and the existing shared B_nt INT8 weight layout. A new strict
`B70_W8A16_M_MAX=11` route sent rows 1 through 11 to quant-free W8A16 and kept
larger rows on the current dynamic-quant plus W8A8 path. Values outside the
validated 1..11 range fail closed; unset preserves the prior M=1-only route.
Mechanism-only route telemetry was enabled, while LM-head INT8, delayed/fused
MLP boundaries, GDN INT8 overlays, and graph capture remained off. The gate
required exact dual-rank five-step routes, no M=11 activation quantization,
unchanged capacity, deterministic replay, concurrent coherence, identity,
immutable artifacts, endpoint-down cleanup, and card health.

COMMAND -> `./bin/gpu-run bash sglang/06_c4_m11_w8a16_mechanism.sh`.
Artifact: `results/logs/c4_m11_w8a16_mechanism_20260824T192541Z/`.

RESULT -> formal PASS after 565 seconds. Each rank recorded exactly 320 MLP
gate-up, 320 MLP down, 80 full-attention qkv, and 80 full-attention out W8A16
calls over five M=11 steps, for 800 calls/rank. The corresponding M=11 W8A8
counts were all zero, and activation-quant counts at K5120, K8704, and K3072
were all zero. Rank total device times were closely matched at 424.213 and
423.591 ms. Capacity stayed exactly 143360 tokens with 4.46 GB available GPU
memory. The repeated eight-prompt corpora were byte-identical, all 24 mixed
streams were coherent, and the initial fixed response was coherent. Exact
served identity, container environment, mounted shim, image, route logs,
artifact hashes, fatal-log scan, and pre/post health checks passed. The
endpoint remained down, and both cards were healthy and leases free at exit.

VERDICT -> M<=11 W8A16 mechanism GO. It removes all measured M=11 activation-
quant boundaries for the 160 target W8A8 linears per decode step without a
weight clone, capacity cost, coherence failure, or TP/P2P instability. Advance
to a strict position-balanced A-B-B-A against the unchanged M=1 baseline with
route telemetry disabled in every performance arm. Do not promote the shelf
threshold until c1, c4, soak, TTFT/prefill, restart stability, deterministic
output, mixed coherence, and acceptance behavior pass. Endpoint remains down
by campaign policy.

### 2026-08-24x - TP=2 M<=11 W8A16 A-B-B-A strict FAIL

CONFIG -> one continuous dual-card lease and four fresh Sglang TP=2 serve
lifecycles in A-B-B-A order. A was the current M=1-only W8A16 threshold; B sent
all M=1..11 rows through the quant-free W8A16 path. Both arms used the same
Qwen3.6-27B SQ-GPTQ W8A8 checkpoint, 131072 context, MTP10/draft11, eager/radix
off, push-all, replicated MTP embedding, P2P access off, and unchanged 143360
token capacity. Route telemetry and unrelated candidate features were off.
The predeclared gate required both matched phase and soak pairs to win, balanced
phase >=3%, balanced soak >=2%, warm TTFT/prefill within 5%, candidate phase CV
and restart spreads <=5%, byte-exact candidate restart outputs, mixed-load
coherence, exact identities and artifacts, health, and endpoint-down cleanup.

COMMAND -> `./bin/gpu-run bash sglang/07_c4_m11_w8a16_abba.sh`.
Artifact: `results/logs/c4_m11_w8a16_abba_20260824T194529Z/`.

RESULT -> formal analyzer FAIL after 5758 seconds. The candidate won both phase
pairs and both 6400-token soak pairs. Position-balanced phase decode was +8.184%
and sustained soak was +5.416%. Warm c1 was -0.720%, c4 aggregate -0.633%,
acceptance -1.555%, c1 TTFT +2.598%, and prefill TTFT improved 1.038%/0.510%
at c1/c4. Candidate phase medians were 18.631/18.289 tok/s versus baseline
17.628/16.516; candidate soaks were 18.06/18.32 versus 17.27/17.24. The sole
formal failure was candidate within-process phase CV: 13.27%/17.39% versus the
5% ceiling. Candidate restart phase and soak spreads passed. All fixed outputs,
deterministic corpora, 96 mixed streams, and soaks were coherent; candidate
restart outputs were byte-identical. Every identity, config, capacity, feature,
artifact, fatal-marker, health, and endpoint-down check passed.

VERDICT -> no shelf promotion. M<=11 W8A16 has a real sustained +5.4% signal,
but it does not improve the warm c1 or c4 serving rows and failed the strict
within-process stability gate. Archive it as a strong mechanism and possible
future revisit; pause this kernel branch while the product-choice campaign
compares Qwen3.6 W8A8, Qwen3.8 UD-Q4_K_XL, and 8-bit Ornith with Pi on local
Terminal-Bench 3.0. Endpoint remains down.

### 2026-08-24y - Ornith-1.5 W8A8 XPU build and TP=2 product qualification

CONFIG -> pinned `shisa-ai/Ornith-1.5-35B-A3B-MTP` revision
`779a91ed5b7597bc6db383d9fffb4343b67892ea`, preserving its trained BF16 MTP
sidecar. XPU RTN used symmetric per-output-channel INT8 weights and dynamic
per-token INT8 activations. Routed experts and eligible text linears were stored
INT8; vision, routers, GDN/linear-attention, lm_head, and MTP stayed BF16. The
serve qualification used Sglang 0.5.15.post1, TP=2, 262144 context, MTP
steps=3/draft=4, extra-buffer radix cache with INT8 Mamba checkpoints, and
`CCL_TOPO_P2P_ACCESS=0`. Experts used the fused INT8 W8A8 MoE path; dense text
linears used the current one-time-dequant BF16 compute fallback.

COMMAND -> `./bin/gpu-run --card 0 bash
sglang/w8a8/quantize_ornith15_quark_w8a8.sh`; then `./bin/gpu-run env
CTX=262144 RADIX=1 MTP=1 PORT=18080 bash
sglang/w8a8/serve_ornith15_w8a8.sh start`; qualification probes ran through
full dual-card `gpu-run` leases.

RESULT -> the real Arc XPU conversion completed in 426 seconds. It quantized
32,610,713,600 elements into 30,880 INT8 tensors with 30,880 matching scales,
relative L2 0.008452, RMSE 8.968e-05, and max absolute error 0.003322. All
62,565 indexed keys resolved across 17 shards; 19 BF16 MTP tensors remained and
the sidecar SHA256 stayed
`73c6e839971fff3c6d78dbcb6a15895bbab340a2898e98aa6943070751de712e`.
TP=2 loaded 18.06 GB target weights plus 1.70 GB MTP per card. MTP recorded mean
accept length 3.275 and 75.83% acceptance on its qualification request. A
4,129-token cache probe improved from 7.743 seconds cold to 0.241 seconds warm.
The 250,042-token near-context retrieval returned the correct early needle in
370.478 seconds cold and 5.450 seconds warm with byte-identical outputs. Native
OpenAI tool parsing returned the exact requested call and arguments. The mixed
prefill/decode gate passed 8/8 coherent streams. Card health remained clean.

VERDICT -> qualified for the Pi + TB3-local-70 product-choice campaign. This is
a real GPU-built and fused-expert W8A8 MoE artifact, with the dense BF16 compute
fallback explicitly disclosed. Keep the research endpoint live at port 18080
while the three-task Pi smoke runs; do not promote a shelf entry before the
model-selection gates finish.

### 2026-08-24z - Ornith W8A8 MTP1 semantic profile: launch-bound first

CONFIG -> refreshed Steve's `b70-optimization-lab` to clean upstream revision
`0cf5b751` without overwriting his preserved local graft, refreshed Sergio's
Arc B70 cookbook to `44e97e1`, and refreshed 0xSero's `qwen38-b70` to
`e873853`. The controlled local serve used Ornith-1.5-35B-A3B W8A8 RTN,
Sglang 0.5.15.post1, TP=2, 8192 context, eager execution, overlap/radix off,
one active request, MTP1/draft2, and `CCL_TOPO_P2P_ACCESS=0`. Both cards were
already at the existing 230 W cap. Default-off Kineto semantic ranges covered
target/MTP, decoder layer family, GDN, full attention, MoE routing, shared
expert, routed W8A8 experts, and quantized dense projections without inserting
XPU synchronizations.

COMMAND -> `./bin/gpu-run bash
sglang/w8a8/profile_ornith15_w8a8.sh`. Runtime artifacts:
`results/logs/ornith15_w8a8_profile_20260824T231106Z/` and
`/mnt/vm_8tb/b70/sgl_cache/ornith15_w8a8_profile_20260824T231106Z/`.

RESULT -> clean PASS in 396 seconds. The p512/g128 cookbook-style median was
11.644 output tok/s. MTP1 mean accept length was 1.975 and draft acceptance
97.5%, proving draft quality was not the limiter. Each verify step launched
about 1247 device kernels/copies, including 84 all-reduces, 238 dense/router
GEMMs, 82 fused-MoE kernels, 41 top-k calls, and 80 expert activation-quant
kernels. Slow-rank device work was 24.97 ms/verify versus about 169.6 ms of
unprofiled verify wall implied by output rate and acceptance. The slow rank's
all-reduce work was 14.06 ms/verify; its five-step all-reduce total was 70.30
ms versus TP0's 24.83 ms. The instrumented trace spans were 93.1%/95.6% idle.
Sglang also reported that both exact B70 `E=256,N=256` INT8 W8A8 MoE tuning
files were missing and used generic Triton configs. The first semantic install
missed only top-k/all-reduce labels due a wrong `TopK` module reference; raw
correlation retained the exact measurements and the import was repaired.
Cards were healthy before/after and the endpoint was stopped.

VERDICT -> current Ornith Sglang is launch/scheduler bound first, collective
bound second, and expert-GEMM bound third. Eliminating measured collective
device time entirely only raises the current-path ceiling to about 12.7 tok/s;
eliminating full CPU collective call time gives about 13.5. The ideal ceiling
from current slow-rank device work is about 79 tok/s, consistent with Sergio's
70.7 no-spec / 96.4 MTP1 and Steve's graph/eager split. Next try narrow Sglang
MTP1 graph capture with P2P off; if it cannot capture coherently, port this
artifact to Steve's current vLLM Quark W8A8 piecewise-graph path. Then tune the
missing MoE configs and attack the 2.8x rank collective asymmetry. Full report:
`docs/20260824_ornith15_w8a8_profile.md`.

### 2026-08-25a - Steve stack forensics and exact Qwen S2B P2P-off control

CONFIG -> pinned S2B image
`intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94`,
exact Qwen3.6-35B-A3B Quark W8A8 checkpoint, TP=2, maxlen 8192, no MTP,
PIECEWISE graph, explicit all-reduce/all-gather split boundaries, and
`CCL_TOPO_P2P_ACCESS=0`. A Qwen-only local adapter restored the June XPU INT8
linear candidate, bridged the later image's partial shared-expert API merge,
and restored a no-spec uniform PIECEWISE descriptor. It imported no Ornith
compatibility code. The metric exactly followed Steve's natural-chat protocol:
requested p512, one o64 warmup, streaming o512 measurement, and ignore EOS.

COMMAND -> `B70_LOGDIR=/mnt/vm_8tb/b70/results/logs ./bin/gpu-run bash
vllm/w8a8/serve_qwen36_s2b_control.sh run`.

RESULT -> the model loaded native Quark W8A8 INT8, compiled, captured, became
healthy in 112 seconds from the warm cache, and passed both semantic canaries.
The prompt tokenized to 498 tokens. The measured 512-token response was
coherent ASCII with 624.292 ms client TTFT, 30.009976 s corrected decode time,
17.055906 corrected output tok/s, and 16.740457 end-to-end output tok/s.
Steve's accepted matched result was 85.869 tok/s and 5.96267 s decode. Artifact:
`/mnt/vm_8tb/b70/results/logs/qwen36_s2b_p2p0_steve_metric_20260825T030225Z.json`.
Both cards were healthy after teardown.

VERDICT -> the clean native-INT8 and graph control is now coherent, but remains
5.0x slower in decode than Steve's matched result. Steve's accepted path kept
direct-P2P oneCCL communication inside the forced graph; the local safe
P2P-off control splits at per-layer collectives. The next highest-information
transaction is the existing capturable Level Zero IPC push all-reduce inside
replay with P2P access still off, not another raw bandwidth microbenchmark.
The full clean-room ownership program, including `_xpu_C`, overlay mechanics,
SGLang transfer, 27B transfer, and TP/PP/DP/single-card coverage, is recorded in
`docs/20260825_steve_stack_reproduction_program.md` and `RESEARCH_TODO.md`.

### 2026-08-25b - Exact graph policy and push-all-reduce loaded-process blocker

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision, pinned S2B image, TP=2,
P2P access off, no MTP, async scheduling, PIECEWISE graph, and the local
capturable Level Zero IPC push all-reduce. The second arm removed push AR and
tested the older local legacy partition path. A final exact arm supplied only
Steve's `{"cudagraph_mode":"PIECEWISE"}` config with no repository split-op
list or forced inductor graph partitioning. Push-AR scratch was raised to 64
MiB. Its chained source adapter was made importable to Dynamo, IPC open gained
bounded retries, and rank-local open status was exchanged so asymmetric setup
could not deadlock.

COMMAND -> `./bin/gpu-run env EXACT_STEVE_CC=1 PUSH_AR=1 PUSH_AR_GRAPH=1
P2PACCESS=0 NAME=qwen36_s2b_exactcc_pushar PORT=18080 bash
vllm/w8a8/serve_qwen36_s2b_control.sh run`; legacy arm used `IGP=false
PUSH_AR=0 NAME=qwen36_s2b_legacy_p2p0`. The standalone graph harness ran in
the exact pinned image before and after rebuilding the push-AR library.

RESULT -> the legacy arm failed in `vllm/compilation/codegen.py:96` because
the injected split policy produced a non-integer split index. The exact minimal
arm compiled successfully in 80.92 seconds, proving that Steve's graph policy
removes that failure. Push-AR rank 0 opened rank 1's scratch, while rank 1
failed rank 0's Level Zero IPC handle 25 times with `0x78000004`. The hardened
status exchange made both ranks fall back to oneCCL, after which monolithic
capture stalled as expected with P2P off. Teardown completed and both cards
passed the single-card health probe. The rebuilt push-AR library hash is
`3ed15e33235d359e3cd696bf844cc8781da475a2d144f3e2b12d215feea3844d`.
The standalone exact-image harness remained correct across 50/50 graph replays
and eight-all-reduce replay sequences at about 35.45 us for a 10 KiB tensor.

VERDICT -> remove the local manual split policy from exact Steve controls. The
remaining safe-path blocker is asymmetric Level Zero IPC import in a loaded
vLLM worker, not push-AR math, capture mechanics, or raw B70 P2P capability.
Steve's own results put the dominant lever in usable whole-decode graph replay:
about 16.7 to 92 tok/s, while clone-safe custom collectives added roughly 3
tok/s. Proceed with one guarded kernel-7.1 exact oneCCL direct-P2P transaction,
then bisect the closest surviving June vLLM source if the later image still
does not reproduce. Endpoint remains down and card health is green.

### 2026-08-25c - Kernel-7.1 exact direct-P2P fail and hidden clone-contract drift

CONFIG -> exact Qwen3.6 Quark W8A8 TP=2 control, pinned S2B image, async
scheduling, no MTP or prefix cache, minimal Steve PIECEWISE compilation config,
oneCCL/OFI, `CCL_TOPO_P2P_ACCESS=1`, and the explicit repository wedge
override. No local push all-reduce was active. This was one guarded transaction
with no chained retry.

COMMAND -> `./bin/gpu-run env EXACT_STEVE_CC=1 PUSH_AR=0 P2PACCESS=1
I_KNOW_P2P_WEDGES=1 NAME=qwen36_s2b_exactcc_p2p1 PORT=18080 bash
vllm/w8a8/serve_qwen36_s2b_control.sh run`; then stop and one dual-card
`bin/xpu-health` lease.

RESULT -> unlike the old kernel path, both XCCL workers initialized and the
34.15 GiB checkpoint loaded normally. The exact graph compiled, then rank 1
failed on the first compiled `vllm::all_reduce` during profile-run with
`UR_RESULT_ERROR_DEVICE_LOST` (error 20). Teardown completed and both cards
passed the post single-card health probe. The run also emitted PyTorch's custom
op output-alias warning despite both clone environment settings. Source diffing
found why: Steve's 2026-06-16 `parallel_state.all_reduce` honors the inner
`VLLM_XPU_CUSTOM_ALLREDUCE_CLONE_INPUT` and clones before dispatch, while the
August image removed that code entirely. The setting was inert locally. A new
attributed `vllm::s2b_all_reduce_clone` adapter op now restores Steve's exact
two-clone contract and passes a no-device schema/import check. The launcher also
restores Steve's Qwen MoE, no-repack, and zero-fresh-GDN defaults and supports
an isolated host compilation-cache mount.

VERDICT -> kernel 7.1 cured the GuC/BCS hardware wedge but did not cure the
oneCCL-vLLM direct-P2P model all-reduce failure. This transaction was not yet
source-equivalent because the later image silently ignored the required inner
clone. Reboot before the next P2P transaction; then retest the restored clone
contract from a fresh cache. If it still fails, use the import-proven closest
surviving June vLLM snapshot as the next one-factor forensic overlay. Endpoint
remains down; immediate post-teardown card health was green.

### 2026-08-25d - Steve native-stack closure and IPC identity correction

CONFIG -> read-only provenance audit of Steve's refreshed optimization lab,
closest June vLLM snapshot, current vLLM/XPU-kernel trees, preserved oneCCL
build/install, pinned S2B image, exact model config, and the local forensic
launcher. No GPU operation was run because the next direct-P2P transaction is
reboot-gated.

COMMAND -> source and result inventories with `git ls-files`, `git diff`,
`rg`, `sha256sum`, `readelf`, and no-device pinned-image Python imports; Docker
network inspection; `bash -n` and `py_compile` on the local adapter/launcher.

RESULT -> Steve's preserved oneCCL is an ARCB release build from source
`4ceafd15`, made with oneAPI 2025.3. Its 240177816-byte `libccl.so.1.0` hash is
`542142aca8f3d318616eae0f300aaa47dc62b217831599cb1461212f8aa4dc76`,
byte-identical to the pinned image. Steve's currently preserved `_xpu_C` and
GDN hashes also match that image exactly (`ae330aff...` and `cf482fd...`). This
proves current-snapshot parity, not June-record binary parity: the June controls
explicitly used a restored 67 MB `_xpu_C`, while the surviving/image extension
is 116706992 bytes. The June extension and its hash were not preserved in the
refreshed lab. The oneCCL tree's only dirty source edits qualify the ESIMD
barrier namespace in small all-gather/reduce-scatter; they do not implement the
decode all-reduce lever. The August piecewise backend is unchanged from June,
and its graph wrapper is a compatible superset. The old no-op
communicator-capture setting is now unconditional through
`XpuCommunicator.ca_comm = None`. The material accepted-path source regression
found remains the removed inner all-reduce clone. Steve's launcher also unset
`CCL_ZE_IPC_EXCHANGE` and `CCL_WORKER_COUNT`, while the failed local transaction
forced `pidfd`; it pinned his active bare-metal `eth1`, whose Docker-equivalent
interface here is `eth0`. The local exact launcher now reproduces those
semantics using trailing name-only Docker env removals and explicit `eth0`.
Steve's older TP2 p512/o256 evidence reached 91.35-91.59 tok/s, establishing a
weaker-gate ceiling above the 85.87 tok/s natural-chat smoke.

VERDICT -> current-snapshot native binaries, model revision, PIECEWISE backend,
and major graph flags are closed, but the June record's 67 MB `_xpu_C` is not.
Reboot, then make one guarded fresh-cache direct-P2P run with both June clone
guards and unset/default IPC exchange. If it fails, bisect with the
import-proven June vLLM source snapshot while holding current native binaries
fixed. Once graph replay works, reconstruct and compare the June kernel build;
it is a plausible residual speed lever, not the first explanation for the 5x
gap. The separate August graph-safe FlashAttention build is later forensic
material and must not enter the exact control.

### 2026-08-25e - TP4 identity correction and oneCCL graph-oracle ownership

CONFIG -> refreshed Steve's public lab from `c1cc2bf` to
`523ca95b925308391707624530c29359edd05b6a`, inspected the supplied
LocalMaxxing run `cmq9ifq0500b0r8012f27j1xl`, the Qwen35 TP2/TP4 family
packets, and Steve's later public oneCCL direct/XPUGraph oracle and build
recipe. Inspected the pinned image's oneCCL install, SPIR-V, package/runtime
versions, and this host's CPU/PCIe topology. No GPU operation was run because
the direct-P2P lane remains reboot-gated.

COMMAND -> LocalMaxxing `/api/leaderboard?run=...&limit=1`; Steve lab source,
result, launcher, and patch reads; pinned-image `find`, `sha256sum`, `readelf`,
and no-device package imports; host `lscpu`, `lspci -tv`, `uname`, and package
inventory. Added the attributed local
`vllm/w8a8/qwen36_oneccl_graph_oracle.py` plus its guarded Docker wrapper.

RESULT -> the supplied public result is TP4, not TP2: four B70s, exact model
revision `cced5659`, PIECEWISE graph, no MTP, p512/o512, 32K context, and
99.769699 tok/s. Steve's current-program values are 85.869114 for the TP2
smoke and 93.550542 for strict TP4; older weak-gate values are about 91.35 TP2
and 99.77 TP4.
TP4 therefore adds about 9 percent, not the missing local 5x. Steve's later
oneCCL artifact proves a stronger pre-model contract: public libccl
`4ceafd15` passed 256/256 direct and 512/512 `[4,5120]` BF16 XPUGraph replays
with `pidfd`. The pinned image has the same exact `kernels.spv` hash
`0d549c35...`, but its 240177816-byte library hash `542142ac...` differs from
Steve's oracle-validated `43d94d43...`. Source equality is therefore not yet
binary or graph-correctness proof. The systems also share B70 GPUs but not the
host: Steve's June Qwen35 host was EPYC 9015/PCIe 5, his later two-card oracle
was Threadripper PRO 5955WX, and this host is Threadripper 1950X with the two
cards below distinct PCI domains on a PCIe Gen3-era platform.

The wider public-repository audit found no hidden second W8A8 implementation.
Steve's current vLLM fork is later upstream drift; the accepted overlay remains
the June lab source/patch chronology. The current XPU-kernel fork adds an FP8
out-variant relative to the S2B tree, not a new Quark W8A8 route.
`ml-bottleneck` is a calibrated explanatory model, and the community repo is
deployment/topology guidance. The Intel llama.cpp branch contains useful B70
MMVQ, activation-reuse, GDN-fusion, and poison-gate patterns, but they target
GGML/SYCL rather than the vLLM Quark ABI.

VERDICT -> retain 85.87 tok/s as the two-card coherent target and about 91.5
tok/s as the older screen ceiling. TP4 is a modest later scaling option, not
the current explanation. After reboot, run exactly one local direct-plus-graph
oracle transaction with Steve's June unset/default IPC identity and record the
loaded hashes. If it passes, reset before the clone-correct full-model arm. If
it fails, rebuild Steve's pinned public oneCCL source and require the oracle to
pass before another model load. This isolates collective graph correctness
from vLLM graph ownership and avoids another blind 34 GiB model transaction.

### 2026-08-25f - Exact oneCCL direct-plus-XPUGraph oracle passes locally

CONFIG -> post-reboot healthy cards, exact `[4,5120]` BF16 Qwen verifier
all-reduce shape, two XCCL ranks, pinned S2B image, direct P2P enabled, Steve's
unset/default IPC exchange and worker-count semantics, pinned-image oneCCL hash
`542142ac...`, and exact device-kernel hash `0d549c35...`. Docker bridge
networking supplied the semantic container interface `eth0`.

COMMAND -> `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 IPCX=default bash
vllm/w8a8/run_qwen36_oneccl_graph_oracle.sh`; then a dual-card
`./bin/gpu-run bash -lc './bin/xpu-health'`. An initial host-network launcher
attempt failed in OFI KVS because this host has no interface named `eth0`; it
never reached a collective, both cards remained healthy, and the launcher was
corrected to bridge networking.

RESULT -> 256/256 direct all-reduces and 512/512 XPUGraph replays passed on
both ranks with zero mismatches and max absolute difference 0.0. Average time
including synchronization and validation was 1.446 ms direct and 0.349 ms
graph on both ranks. Loaded library and SPIR-V identities matched the required
hashes. Although the environment left `CCL_ZE_IPC_EXCHANGE` absent, oneCCL
reported its effective default as `pidfd`. Both cards passed post-run health.
Machine-readable evidence is
`results/oneccl_oracle/qwen36_tp2_oneccl_default_20260825T065907Z.json`.

VERDICT -> raw oneCCL direct P2P and XPUGraph work correctly on this exact B70
pair, Threadripper 1950X host, kernel 7.1, and pinned current binary. Neither
PCIe topology nor oneCCL graph correctness explains the 17.06 tok/s endpoint
or the prior full-model `DEVICE_LOST`. The active fault boundary is above raw
oneCCL: vLLM's custom-op wrapper, restored two-clone alias contract, compiled
graph ownership, or worker/model graph lifecycle. Preserve the reset boundary,
then run one clone-correct exact-model transaction from a fresh cache. A
oneCCL rebuild is no longer the next action.

### 2026-08-25g - Custom-op route correction and no-model integration gate

CONFIG -> read-only audit of Steve's accepted June vLLM source, clone A/B
notes, current pinned-image source, the prior local failure log and compile
cache, Qwen checkpoint config, and preserved XPU-kernel Git bundle/patches. A
no-device custom-op execution probe and Dynamo export were run in the pinned
image. No GPU transaction was run because the direct-P2P lane is reset-gated.

COMMAND -> `git show`, `rg`, `nl`, Python `inspect`, a CPU-dispatch custom-op
`torch.compile` probe, and a no-device export through the local sitecustomize
adapter. Added `vllm/w8a8/qwen36_vllm_allreduce_graph_oracle.py` and its
guarded Docker launcher.

RESULT -> the prior local clone adapter was inert. With
`VLLM_XPU_USE_CUSTOM_OP_COLLECTIVES=1`, GroupCoordinator emits the registered
outer custom op directly. Its Python implementation executes with
`torch.compiler.is_compiling()` false, so patching XpuCommunicator never routed
to the local replacement. Steve set two clone flags, but source control flow
and his neutral graph-clone-off A/B prove only the inner registered-op clone
was active and required. Removing it produced the recorded alias warning and
corrupted token soup. August merge drift removed that clone. The corrected
adapter now routes GroupCoordinator to `vllm::s2b_all_reduce_clone`; no-device
Dynamo export contains that op and no stock `vllm::all_reduce`.

The previous `DEVICE_LOST` occurred during vLLM profile-run, which forces graph
mode NONE, before any XPUGraph capture or replay. Its real profile tensor shape
is `[8192,2048]`; the exact cached Qwen backbone has 81 all-reduce nodes. The
new reset-bounded oracle therefore gates eager, compiled `[1,2048]`,
`[4,2048]`, and `[8192,2048]`, compiled XPUGraph replay, and an unrolled
81-collective graph while checking output identity, input mutation, pointer
aliasing, and the exported op name.

The June 54 MB and accepted 67 MB `_xpu_C` binaries are not recoverable from
Git objects, bundles, images, caches, or manifests. Source reconstruction is:
public base `28e1f5e`, preserved private sequence `122b698` through `3b4effe`,
and the recorded June W8A8/layerlet/exact-SiLU patches. The first build matrix
will compare `bmg-g21-a0` with the old multi-target AOT default under the exact
torch 2.11/oneAPI 2025.3 ABI; the size difference being AOT coverage remains an
inference.

VERDICT -> the first failed model transaction was not clone-correct, and raw
oneCCL has already passed. The next highest-information transaction is the
new no-model compiled custom-op oracle after reboot, not another full model
load or a oneCCL rebuild. If compiled profile-shape execution passes, continue
within that one transaction through graph and 81-collective replay; then reset
again before the corrected exact model control.

### 2026-08-25h - Hardened custom-op integration oracle

CONFIG -> pre-GPU independent review of the locally owned two-rank vLLM
custom-op oracle and launcher. The scope is the real GroupCoordinator custom
op under stock Dynamo/Inductor, not vLLM's VllmBackend/PIECEWISE partitioner or
interleaved Qwen model execution. No GPU operation was run; this boot's guarded
direct-P2P transaction remains consumed.

COMMAND -> `python3 -m py_compile`, `bash -n`, `git diff --check`, no-device
pinned-image Torch API inspection, exact BF16 expected-value review, and source
review of lifecycle cleanup, runtime identity checks, CLI validation, dynamic
shape compilation, and 81-collective mutation/alias coverage.

RESULT -> the oracle now fails closed on exact loaded oneCCL, `_xpu_C`, and
oneCCL SPIR-V hashes before process-group initialization; records arguments,
software, topology, graph, compiler, and cache settings; checks the direct
input on the first unrolled collective so a missing clone cannot hide; compiles
dynamic shapes; reproduces 81 `[8192,2048]` profile collectives; and checks
input mutation and output aliasing during both single and 81-collective graph
replay. Exceptions produce best-effort per-rank checkpoints, while distributed
cleanup cannot replace the primary failure. Syntax and whitespace gates pass.

VERDICT -> the reset-bounded transaction is ready but intentionally not run on
this boot. After reboot, run exactly `./bin/gpu-run env
I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_vllm_allreduce_graph_oracle.sh`. A pass clears the
custom-op plus stock compiler layer only; the following reboot-bounded exact
model control remains the VllmBackend/PIECEWISE gate.

### 2026-08-25i - Independent June W8A8 native reconstruction

CONFIG -> official `vllm-project/vllm-xpu-kernels` base `28e1f5e74c`, the
owned June 9 patch SHA256 `14c2e801...`, patched tree `c882c446...`, pinned
Intel image digest `f2e5a94e...`, torch 2.11.0+xpu, IntelLLVM 2025.3.3,
Release/Ninja `-j2`, `bmg-g21-a0` AOT, Xe2 MoE plus GDN, and no `/dev/dri`.
The materialized runtime package inherited support artifacts from that exact
image and replaced only `_xpu_C`, the grouped/GDN Xe2 siblings, and patched
`fused_moe_interface.py`.

COMMAND -> independent no-hardlink clone of the official-base Git object from
a local mirror, origin rewritten to official GitHub, then patch and exact
CMake/Ninja `_xpu_C` build; component install; SHA256, ELF/RUNPATH, dependency,
module-origin, operator-schema, and XPU-dispatch census in a no-device pinned
container. The committed owned recipe does not use that mirror; it fetches
official GitHub directly with `bash
vllm/w8a8/build_qwen36_june_xpu_c.sh`.

RESULT -> build completed in 55 minutes. Installed `_xpu_C` is 55,523,648
bytes, SHA256 `2d931484...`, with `$ORIGIN` RUNPATH. GDN is 2,724,136 bytes,
SHA256 `366935b1...`; grouped GEMM is 2,936,608 bytes, SHA256 `f5ddc2ee...`.
All dynamic dependencies resolved. The complete package imported `_C`,
`_moe_C`, rebuilt `_xpu_C`, and the patched dispatcher from its own path;
`FUSEDMOE_AVAILABLE=True`. Native activation quant, dense W8A8, grouped W8A8,
SiLU, expert-map/remap, and MoE-gather schemas were present with XPU dispatch.
The manifest is
`vllm/w8a8/manifests/qwen36_june_xpu_c_bmg_g21_a0_20260825.json`.

Source comparison also proved pinned August kernel commit `2dd55f38` already
contains June's base activation quantizer, dense W8A8 GEMM, and grouped W8A8
MoE path. Its additions are optional output, scratch, barrier, offset, policy,
and reuse arms. The later vLLM dispatch and all-reduce clone regressions, not
missing June native math, remain the leading explanation for the endpoint gap.

VERDICT -> source ownership and the off-device dispatch gate are achieved for
the minimal June native replacement set. Its size reproduces Steve's recorded
54 MB fresh-build class, not the unrecoverable accepted 67 MB binary. Do not
claim numerical or performance parity yet: after the required reboot boundary,
the custom-op collective oracle comes first; leased GPU numeric/capture tests
for this kernel package follow as a separate transaction.

### 2026-08-25j - Clone-correct vLLM custom-op oracle partial pass

CONFIG -> first GPU transaction after the requested reboot; pinned S2B image
digest `f2e5a94e...`, torch 2.11.0+xpu, exact loaded oneCCL
`542142ac...`, `_xpu_C` `ae330aff...`, and oneCCL SPIR-V `0d549c35...`;
two XCCL ranks; direct P2P; unset/default `CCL_ZE_IPC_EXCHANGE` and
`CCL_WORKER_COUNT`; active container `eth0`; corrected GroupCoordinator route
through `vllm::s2b_all_reduce_clone`; stock dynamic Dynamo/Inductor; eager and
compiled `[1,2048]`, compiled `[4,2048]` and `[8192,2048]`, single-op
XPUGraph, and an attempted unrolled 81-collective profile/graph stress. The
oracle cache was `/mnt/vm_8tb/b70/vllm_oracle_cache`.

COMMAND -> exactly `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_vllm_allreduce_graph_oracle.sh`; after teardown,
`./bin/gpu-run bash -lc './bin/xpu-health'`, followed by the definitive isolated
card-1 completion `./bin/gpu-run --card 1 ./bin/xpu-health --card 1`. No model
or second TP2/P2P experiment was chained.

RESULT -> runtime identities matched on both ranks, and Dynamo export contained
`torch.ops.vllm.s2b_all_reduce_clone` with no stock `vllm::all_reduce`. Both
ranks passed 64 eager `[1,2048]`, 64 compiled `[1,2048]`, four compiled
`[4,2048]`, four compiled `[8192,2048]`, and 256 compiled XPUGraph replays
with zero output mismatches, zero input mutations, and zero output aliases.
Compiled single-op replay averaged about 0.710 ms and XPUGraph replay about
0.454 ms per iteration including synchronization and validation.

The overall oracle then failed on rank 1 during the synthetic unrolled
81-collective `[8192,2048]` arm. The immediate failing instruction was not an
all-reduce: Inductor had transformed the harness's independent `input + offset`
operands into five artificial Triton pointwise fan-out kernels, each writing
16 separate 32 MiB outputs. Rank 1 threw `UR_RESULT_ERROR_DEVICE_LOST` while
autotuning the second 16-output kernel
`triton_poi_fused_add_s2b_all_reduce_clone_1`; torchrun then terminated rank 0.
The real model interleaves its 81 collectives with layer math and does not
materialize this 81-way fan-out plus final sum, so this last failure is not
evidence that the corrected custom op itself failed. It does mean the oracle's
overall pass gate was not met and the 81-collective graph stage was not run.
Both cards passed post-teardown single-card matmul health. Evidence is
`results/oneccl_oracle/qwen36_tp2_vllm_allreduce_graph_20260825T152954Z.log`
and its `rank0.partial.json` and `rank1.partial.json` checkpoints. The retained
generated program is
`/mnt/vm_8tb/b70/vllm_oracle_cache/torchinductor/24/c24fybkziv5qe2t2vhe4glqadzw332ii7jamsxahq2ndgxsjluwb.py`,
SHA256 `5b5767fb0cdf4aeb37908170bb08f66e4f438deb60fe38f7b012993afc63f996`.

VERDICT -> the corrected GroupCoordinator op, required single inner clone,
exact runtime identities, real decode/profile shapes, stock compiled execution,
and single-op XPUGraph replay are cleared. Correct the oracle's artificial
wide-fan-out stress before reusing that test, but do not spend the next reboot
on it: the higher-information next transaction remains the corrected exact
Qwen model control, whose real VllmBackend graph contains interleaved layer
math. Preserve an actual reboot boundary before that run, use a new isolated
cache with Steve's unset/default IPC exchange and active `eth0`, and do not
describe this partial result as an 81-collective pass. Historical references
to a required two-clone contract are superseded: only the inner registered-op
clone was active and required in Steve's accepted route.

### 2026-08-25k - Installed grouped-MoE mismatch and exact-control closure

CONFIG -> CPU-only audit after the clone-correct oracle transaction; no
`/dev/dri` was mounted and no new GPU operation was run. Compared the complete
locally rebuilt June runtime package against the package installed in image
digest `f2e5a94e...`. Pinned model identity remained revision `cced5659...`,
config hash `b2a92fb7...`, and index hash `c973ada0...`. The corrected exact
model transaction is TP=2, PP=1, PIECEWISE, async, no MTP, no prefix cache,
maxlen 32768, maxseqs 24, utilization 0.90, p512/o512, direct P2P, unset IPC
exchange and worker count, container `eth0`, a fresh persistent Inductor cache,
the local inner-clone adapter, and the complete June package.

COMMAND -> no-device fresh-container schema censuses with
`qwen36_june_august_kernel_arm.py` for June full/grouped identity, August dense
identity, and an August grouped negative control; no-device import of the
June package plus `qwen36_s2b_sitecustomize.py`; and
`PREFLIGHT_ONLY=1 ... run_qwen36_s2b_clone_exact_control.sh`. Inspected the
preserved 17.0559 tok/s server log. Added an exact fixed-ChatML JSON/color
16-repeat canary, a June/August numeric/repeatability/XPUGraph kernel arm, an
A-B-B-A launcher and summary, and corrected the synthetic 81-collective oracle
to a sequential low-live-buffer dependency chain. Python compile, shell
syntax, ASCII, whitespace, module-origin, model-hash, SO-hash, and schema gates
passed.

RESULT -> the installed August package registers native per-token INT8
quantization and dense W8A8 GEMM, but does not register
`_xpu_C::cutlass_grouped_gemm_w8a8_int8_interface`. Its grouped sibling SO is
present, which made source/file presence an invalid reachability proxy. The
preserved endpoint log confirms request-time JIT of Triton's
`fused_moe_kernel`; the 17.0559 tok/s run was dense INT8 plus Triton routed MoE,
not Steve's all-native W8A8 route. The complete June package registers quant,
dense, grouped, SiLU, remap, and gather operators and loads `_xpu_C` from
`/opt/june-runtime`. The exact preflight pins June `_xpu_C` `2d931484...`,
grouped `f5ddc2ee...`, GDN `366935b1...`, inherited `_C` `57174764...`,
inherited `_moe_C` `ea4c20a8...`, all other package SOs, oneCCL
`542142ac...`, and SPIR-V `0d549c35...`. The pinned August package remains a
valid quant/dense A-B-B-A arm but cannot be a grouped arm without a separate
complete August rebuild. No GPU numeric or performance result is claimed.

VERDICT -> routed-MoE dispatch joins clone/graph ownership as a leading
mechanism; the old 17 versus 85.87 comparison did not isolate graph overhead.
After an actual reboot, run exactly one leased
`run_qwen36_s2b_clone_exact_control.sh` transaction. It now fails closed on
model, runtime, import, graph, metric, model-id, semantic-probe, JSON16/16,
color16/16, and fatal-device evidence. Reboot again before any later P2P/TP2
or kernel transaction. Do not promote a shelf entry from this forensic gate.

### 2026-08-25l - Exact June-package model reaches graph capture; local key rejected

CONFIG -> new boot ID `06b81fbb-bdef-456d-a6e9-185811c66792`; both cards
healthy; exact Qwen revision `cced5659...`; complete rebuilt June package;
TP=2, PP=1, PIECEWISE with vLLM's default split operations and default capture
sizes `[1,2,4,8,16,24,32,40,48]`; maxseqs 24; async; no MTP or prefix cache;
direct P2P; unset/default IPC exchange and worker count; container `eth0`; and
fresh cache. The boot-started single-card daily container was stopped before
the lease; it had P2P off and never joined this transaction.

COMMAND -> exactly `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`. After source comparison,
ran the no-device `qwen36_piecewise_capture_contract.py` against the pinned
image and complete June package, then the launcher's full
`PREFLIGHT_ONLY=1` identity gate. No second GPU transaction was run.

RESULT -> all runtime/model hash gates passed. Both ranks initialized the
process group, loaded 34.15 GiB total weights, selected native dense W8A8, and
registered the June grouped W8A8 and GDN package. The engine then failed before
endpoint health while capturing graphs, on both ranks, at
`gpu_model_runner.py:12486`: `assert sum(num_scheduled_tokens_list) ==
num_tokens`. There was no `DEVICE_LOST`, `OUT_OF_RESOURCES`, or other UR error,
and both cards passed post-teardown health.

Source comparison against June `e190923b` proved the failure was local adapter
drift. June ordinary no-spec decode reused the relaxed non-uniform PIECEWISE
key. The adapter instead added uniform keys for all sizes; at 32, 40, and 48
tokens the one-token dummy schedule was capped at maxseqs 24 and could not sum
to the capture size. The adapter key is removed. The off-device contract now
proves zero ordinary-decode-specific descriptors and valid general schedules
for all nine default sizes without narrowing Steve's minimal compilation
config. The exact launcher's false `splitting_ops=[]` evidence check is also
corrected to require the observed vLLM default list. The complete repaired
CPU-only preflight passed. Primary evidence SHA256 values are committed ASCII
server log `304cd943...` (raw pre-sanitization `d8fcdfb2...`; the four-line
non-ASCII vLLM banner was replaced and CR progress formatting normalized), run
log `55de77c5...`, kernel preflight
`86a5c234...`, and
PIECEWISE contract `a54ad767...` under
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T163105Z` and
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T164600Z_repairpreflight`.

VERDICT -> the exact package and native-op path crossed model load and reached
the remaining VllmBackend graph gate. This failure does not measure endpoint
speed and does not implicate a GPU wedge or native math. Preserve the consumed
direct-P2P boot boundary. After another actual reboot, rerun the same exact
transaction with a new cache; do not pin smaller capture sizes or alter Steve's
minimal PIECEWISE configuration.

### 2026-08-25m - Exact control reaches inference; August capture filter conflicts with June replay key

CONFIG -> new boot ID `30f19437-793a-468e-a54a-ce0ded8f55cc`; kernel 7.1;
exact Qwen revision `cced5659...`; complete rebuilt June package; TP=2, PP=1,
PIECEWISE with default split operations and capture sizes
`[1,2,4,8,16,24,32,40,48]`; maxseqs 24; async; no MTP or prefix cache;
direct P2P; unset/default IPC exchange and worker count; container `eth0`; and
fresh cache. The boot-started P2P-off daily TP=2 service was allowed to finish
initialization and pass health, then stopped under the two-card lease with exit
0 before the exact transaction.

COMMAND -> exactly `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`. After teardown, performed
source-only comparison against June vLLM `e190923b` and the pinned August image,
then ran the launcher's no-device identity and PIECEWISE contract gate as
`STAMP=20260825T174500Z_capturecontractpreflight PREFLIGHT_ONLY=1
I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`. No second GPU transaction
was run.

RESULT -> model and runtime identities passed, both card pre-health probes
passed, both XCCL ranks initialized with direct P2P, and the model loaded 16.88
GiB per card. Compilation completed in 100.03 seconds and the initial profile
run completed in 23.26 seconds. The engine then logged that it skipped all nine
non-uniform PIECEWISE captures because prefill replay was disabled; graph setup
finished in one second with zero additional graph memory, and endpoint health
passed. The first semantic request JIT-compiled model kernels and failed on both
ranks with `RuntimeError: CUDA graph capturing detected at an inappropriate
time. This operation is currently disabled.` The client received HTTP 500, so
no speed metric or canary artifact exists. Teardown was graceful and both card
post-health probes passed. There was no `DEVICE_LOST`, `OUT_OF_RESOURCES`, or
other UR error.

June source uses `VLLM_XPU_DISABLE_PREFILL_CUDAGRAPH_REPLAY=1` only to make
non-uniform prefill dispatch eager. It still captures the relaxed general
PIECEWISE descriptors because ordinary decode reuses them. August added a
capture filter under the same variable. Combined with June's no-specific-key
dispatcher, that filter removes every graph ordinary decode can select. The
adapter now temporarily hides only this variable while August builds its
capture list, preserving the independent spec/decode filters and restoring the
variable before runtime dispatch. The v2 no-device contract proves zero
ordinary specific keys, valid schedules at every default size, retention of
all nine general capture descriptors, and preservation of the runtime setting.
The full no-device preflight passes.

Primary evidence is
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T173515Z`: committed ASCII
server log SHA256 `6fde09ed...` (raw `74639f19...`), run log `06e5b83e...`,
kernel preflight `86a5c234...`, and pre-repair PIECEWISE contract
`a54ad767...`. The repaired v2 contract is
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T174500Z_capturecontractpreflight/piecewise_capture_contract.json`,
SHA256 `2f3bd3ac...`.

VERDICT -> the corrected exact stack now clears process-group initialization,
model load, compile/profile, graph setup, and endpoint health. The inference
failure is a reproducible June/August capture-policy mismatch, not a speed
result or hardware wedge. Preserve the consumed direct-P2P reboot boundary.
After another actual reboot, rerun the identical exact transaction with a new
cache; do not disable the June eager-prefill runtime policy or narrow capture
sizes.

### 2026-08-25n - Exact control is coherent at 47.54 tok/s; Quark MoE still routes through Triton

CONFIG -> new boot ID `e2d5777d-f6bb-4d92-a718-0fb07ae17919`; kernel 7.1;
exact Qwen revision `cced5659...`; complete rebuilt June runtime package;
TP=2, PP=1, default-size PIECEWISE graphs, maxseqs 24, async, no MTP or prefix
cache, direct P2P, unset/default IPC exchange and worker count, container
`eth0`, and a fresh compilation cache. The boot-started P2P-off daily service
reached health and was then stopped gracefully under the two-card lease.

COMMAND -> exactly `./bin/gpu-run env I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`. After its one guarded GPU
transaction and teardown, inspected the digest-pinned image source without a
GPU and ran `STAMP=20260825T183000Z_native_moe_preflight PREFLIGHT_ONLY=1
I_KNOW_P2P_WEDGES=1 bash
vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh`. No second GPU transaction
was run on this boot.

RESULT -> both ranks initialized with direct P2P, loaded 16.88 GiB per card,
compiled in 100.46 seconds, profiled in 23.05 seconds, and captured all 9/9
general PIECEWISE graphs in 26 seconds using 1.64 GiB. Endpoint identity and
semantic probes passed. The exact natural-chat p498/o512 request was coherent:
512 output tokens, 336.710 ms client TTFT, 11.0845 seconds end to end,
10.7657 seconds of vLLM decode time, 46.1908 client output tok/s, and 47.5448
tok/s corrected after the first token. JSON and color canaries each passed
16/16 with zero mismatch. Teardown was graceful; both post-health probes
passed, with no `DEVICE_LOST`, `OUT_OF_RESOURCES`, UR, or alias marker.

The launcher exited 1 only because its strict evidence gate found request-time
`fused_moe_kernel` Triton JIT. Source inspection proved the mounted June
package was reachable but not selected for routed experts: digest-pinned image
Quark source SHA256 `7e4c13d2...` unconditionally calls generic
`fused_experts`, despite containing the XPU INT8 MoE oracle and experts class.
The prior log line showing the grouped schema proved registration only. A
narrow adapter now restores backend selection, `E,N,K` to `E,K,N` weight
layout, `E,N,1` to `E,N` scale layout, native kernel construction, and native
apply while retaining the image's RoutedExperts ABI. Its no-device contract
passed with SHA256 `ed9ee40f...`.

The measured control is 2.788x the earlier 17.0559 tok/s split-collective arm,
55.37% of Steve's 85.8691 tok/s, and closes 44.31% of that absolute gap. The
remaining gap is 38.3243 tok/s or 1.806x; decode remains 4.8030 seconds slower
than Steve. Primary committed ASCII evidence under
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T180624Z` is server log
SHA256 `cac6838b...` (raw pre-sanitization `58a55026...`), run log
`d6f26543...`, metric `8b1213cc...`, JSON canary `b501be3e...`, color canary
`2865ea7a...`, kernel preflight `86a5c234...`, and capture contract
`2f3bd3ac...`.

VERDICT -> the graph/capture, dense INT8, clone-safe collective, and direct-P2P
repairs collectively recover a large coherent gain, but this is not Steve's
native routed-MoE path and is not an exact reproduction. The fatal MoE-JIT
gate correctly prevents promotion. Preserve the consumed direct-P2P reboot
boundary. After an actual reboot, run the identical fresh-cache transaction
with the new native Quark MoE adapter; require the XPU backend log, absence of
request-time `fused_moe_kernel`, exact metric/canaries, and healthy teardown.

### 2026-08-25o - Display diagnosis retired; non-reboot xe recovery ladder passes

CONFIG -> kernel 7.1.0-070100; boot ID
`e2d5777d-f6bb-4d92-a718-0fb07ae17919`; B70 display functions
`0000:0b:00.0` and `0000:44:00.0`; no running GPU-capable container; no
`/dev/dri` holder. All 16 connectors reported disconnected and disabled,
`/proc/fb` was empty, and the VT console was the dummy device. Both endpoints
were initially bound to xe with two `mei_gsc` plus two `mtd_intel_dg`
auxiliary children. Scoped sudo installed the root-owned
`/usr/local/sbin/b70-xe-reset-helper`; sudoers allows only helper list,
unbind-all, bind-all, flr-all, and exact xe modprobe add/remove operations.

COMMAND -> added `bin/xpu-collective-health`, which runs two XCCL ranks, one
eager all-reduce, and ten `torch.compile` functional all-reduces at
`[4,5120]` BF16 with P2P=0. Established its green baseline under `gpu-run`,
then ran under the self-acquired two-card lease:

```text
./bin/xe-reset --method rebind
./bin/xe-reset --method reload
./bin/xe-reset --method flr
```

RESULT -> the first collective-probe development attempt omitted the
established `SYS_PTRACE`/unconfined-seccomp container permissions and failed
DRM-FD exchange before any collective. Adding those container permissions
produced `COLLECTIVE_HEALTH_OK` in 25 seconds. Rebind unbound both endpoints,
rebound both, restored both PCI-qualified render paths and all four auxiliary
bindings, passed card 0/card 1 matmuls, and passed compiled collective health.
Reload unbound both and printed `xe_refcount_after_unbind=0`; both
`modprobe -r xe` and `modprobe xe` succeeded, automatic reprobe restored both
cards, and both health layers passed. FLR unbound both, successfully reset
both endpoints using their advertised `flr bus` reset method, rebound both,
and both health layers passed. The boot ID was unchanged after every stage.

VERDICT -> the old `xe` display-held/reboot-only diagnosis was false: the
module was in use because the GPU endpoints had not been unbound first.
`bin/xe-reset` now implements a guarded rebind -> xe reload -> endpoint FLR
ladder and reboot is only the final fallback. The shared multi-card serve
guard now adds the compiled collective probe before launch and after teardown,
closing J.20's single-card-only detection gap. Clean-state mechanics are
proven; the next naturally occurring deep wedge must record which rung clears
corrupted state. Current cards are on different physical Threadripper root
domains (`pci0000:00` and `pci0000:40`). A same-root slot move is an optional
controlled A/B, not an assumed fix; full runbook and four-card caveats are in
`docs/20260825_xe_nonreboot_recovery_and_pcie_topology.md`.

SHARED-INFRA GATE -> a full `bin/serve-sweep --smoke` was attempted because
`bin/` and `_common/lib.sh` changed. It exposed pre-existing shelf/harness
defects rather than a valid all-green gate: all three llama.cpp entries reject
the harness `smoke` action in favor of their separate start/gate API; paused
vLLM entries reference the absent local `vllm-xpu-env:v0230`; and the NVFP4
launcher treats `smoke` as detached start, allowing the harness to advance
while its container still owns the GPUs. The sweep was stopped, both leftover
containers removed under the lease, and per-card plus compiled collective
health both passed. The unrelated single-card sglang int4 shelf also failed
KV-pool allocation with `OUT_OF_RESOURCES`; sglang W4A8 passed health and
coherence.

Targeted qualification then passed both production sglang TP=2 shelves after
adding direct pre/post collective guards: 27B W8A8 passed health, coherence,
push-AR engagement, and both post-health layers; 35B-A3B W8A8 passed health,
coherence, and both post-health layers. A current-image vLLM 27B W8A8 targeted
smoke proved the `_common/lib.sh` pre/post collective hooks but its engine
exited during initialization; both post-health layers passed, proving no
driver or collective degradation. A duplicate failure-cleanup call then
overwrote the captured root-cause log; `b70_teardown` now preserves an existing
log when the container is already absent. The mandatory all-shelf gate remains
RED on those pre-existing artifact/action failures and must not be reported as
green or bypassed by retagging a different image.

### 2026-08-25p - Exact TP=2 collective boundary localized; clone-only profile fence passes

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 control; TP=2; PIECEWISE 9-size
capture; locally rebuilt June kernel package; June-compatible native INT8 MoE
route; clone-safe `s2b_all_reduce_clone`; kernel 7.1; compute runtime
26.22.38646.4; current pinned oneCCL; cards at `0000:0b:00.0` and
`0000:44:00.0`. Added per-rank all-reduce stages with monotonic timestamps and
made the shared health waiter ignore 60-second `shm_broadcast` coordinator
heartbeats when deciding whether workers had stalled. Every guarded direct-P2P
arm began after `./bin/xe-reset --method reload`; each reload kept boot ID
`e2d5777d-f6bb-4d92-a718-0fb07ae17919` and passed both health layers.

COMMAND -> P2P-off localization:

```text
./bin/gpu-run env \
  STAMP=20260825T224000Z_artrace_p2p0 P2P_ACCESS=0 \
  MOE_TRACE=1 ALLREDUCE_TRACE=1 ALLREDUCE_TRACE_SYNC=1 \
  ALLREDUCE_TRACE_MAX_CALLS=256 STALL_TIMEOUT=180 \
  ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p0_20260825T214109Z \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both ranks completed prior work, pre-sync, clone enqueue, and clone
sync. Rank 1 entered the first `[8192,2048]` BF16 `tp:0` collective at
15701451732254 ns; rank 0 entered at 15701696809283 ns, 245.077 ms later.
Neither emitted collective return and zero MoE calls began. The heartbeat-aware
guard aborted after 180 seconds of real worker silence. Graceful teardown,
both card probes, and compiled P2P-off collective health passed. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p0_20260825T224000Z_artrace_p2p0`.

COMMAND -> direct-P2P all-stage diagnostic after a clean xe reload:

```text
./bin/gpu-run env \
  STAMP=20260825T222600Z_artrace_p2p1 P2P_ACCESS=1 \
  MOE_TRACE=1 ALLREDUCE_TRACE=1 ALLREDUCE_TRACE_SYNC=1 \
  ALLREDUCE_TRACE_MAX_CALLS=256 STALL_TIMEOUT=180 \
  ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_20260825T210027Z \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both ranks completed all 81 model-profile collectives and all 40
native MoE calls. KV cache allocation completed. Both ranks then completed
graph warmup through collective call 162. At actual graph recording, call 163
emitted pre-sync start and `torch.xpu.synchronize()` raised `wait cannot be
called for a queue which is recording to a command graph`. This was a
diagnostic incompatibility, not a device failure. Both post-health layers
passed. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T222600Z_artrace_p2p1`.

COMMAND -> first bounded all-stage fence proved the graph could capture when
synchronization stopped after profile call 81. Then implemented a shape-bounded
production mechanism and ran the minimized clone-only arm after another clean
xe reload:

```text
./bin/gpu-run env \
  STAMP=20260825T224800Z_clonefence_p2p1 P2P_ACCESS=1 \
  MOE_TRACE=0 ALLREDUCE_TRACE=1 ALLREDUCE_TRACE_SYNC=0 \
  ALLREDUCE_TRACE_MAX_CALLS=81 PROFILE_FENCE_MIN_ROWS=8192 \
  PROFILE_FENCE_STAGES=clone STALL_TIMEOUT=240 \
  ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_20260825T210027Z \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> clone-only synchronization crossed all 81 profile collectives,
left graph warmup/recording and decode unfenced, captured all 9/9 PIECEWISE
graphs, and reached health in 198 seconds. Semantic probes passed. The exact
p498/o512 request produced 512 coherent tokens at 311.421 ms client TTFT,
11.283383 seconds decode, and 45.364920 corrected output tok/s. JSON and color
canaries each passed 16/16 with zero mismatch. Teardown was graceful; both
card probes and compiled collective health passed. The corrected strict cache
gate found the custom op under `torch_compile_cache` and the launcher exited
0. Evidence and hashes are in
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T224800Z_clonefence_p2p1`.

DEFAULT-PATH PREFLIGHT -> ran the launcher with `P2P_ACCESS=1`,
`PREFLIGHT_ONLY=1`, and no trace or fence override. All three off-device
contracts passed. Its emitted config proved the guarded default is
`profile_fence_min_rows=8192 profile_fence_stages=clone` while tracing remains
off. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_20260825T230000Z_default_clonefence_preflight`.

COMMIT HYGIENE -> the four committed server logs were mechanically converted
to ASCII after capture; model, trace, timing, and error text were retained.
Raw -> committed SHA256 pairs for P2P-off trace, all-stage P2P trace, bounded
profile fence, and clone-only fence are respectively
`b1d6a2dd...` -> `cb9d9fdf...`, `10974c0c...` -> `2a4c2b7e...`,
`d95ada35...` -> `aeedf3ad...`, and `32ad2e6a...` -> `f16b7ef5...`.

VERDICT -> P2P-off deadlocks inside oneCCL after matched rank entry. Direct P2P
works when the asynchronous clone is complete before oneCCL consumes it. A
clone-only fence for profile tensors with at least 8192 rows is sufficient; no
pre-rank or post-collective fence is required, and no synchronization enters
command-graph recording or decode. This closes the compiled TP=2 collective
boundary. The native grouped-MoE control reaches only 52.83% of Steve's
85.8691 tok/s and is 4.58% slower than the prior generic Triton MoE control.
The next frontier is the remaining 1.893x graph/runtime/kernel gap, not another
collective-boundary retry.

### 2026-08-25q - True June source control closes scratch ABI; 48.5315 tok/s

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision
`cced56592e8c8935f8220836b4baa04dfd389118`; TP=2/PP=1; P2P=1;
PIECEWISE 9-size capture; async; no MTP or prefix cache; complete locally
rebuilt June native package; closest surviving June vLLM source
`e190923b32e1b87fe33d08264bff9215fb7770fc`; clone-completion fence for
profile tensors with at least 8192 rows; kernel 7.1 and compute runtime
26.22.38646.4. A new off-device contract pinned 12 source components covering
graph, collective, GDN, routed MoE, scheduler, sampler, runner, and the fused
kernel interface.

COMMAND -> first true-source transaction:

```text
./bin/gpu-run env \
  STAMP=20260825T233000Z_june_source SOURCE_STACK=june-e190 \
  P2P_ACCESS=1 PROFILE_FENCE_MIN_ROWS=8192 STALL_TIMEOUT=300 \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the exact June source loaded both ranks and selected native dense
and routed-MoE INT8, then failed during profile before its first collective:
`xpu_fused_moe() got an unexpected keyword argument 'scratch'`. June
`xpu_moe.py` passes persistent scratch, while the reconstructed June-9
`fused_moe_interface.py` does not accept it. This was a deterministic Python
ABI mismatch, not a device or collective failure. Graceful teardown left both
per-card health and compiled collective health green. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_20260825T233000Z_june_source`.

COMMAND -> retain the same native binaries and mount only the recovered
scratch-aware fused-MoE Python dispatcher from kernel commit
`2dd55f380df753a10a88fcd9e96192561066e713`:

```text
./bin/gpu-run env \
  STAMP=20260825T235000Z_june_scratch SOURCE_STACK=june-e190 \
  P2P_ACCESS=1 PROFILE_FENCE_MIN_ROWS=8192 STALL_TIMEOUT=300 \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the 12-component source contract passed with no missing tokens or
origin/hash failures. Both ranks selected `XPUInt8ScaledMMLinearKernel` and
`Using XPU Int8 MoE backend`, completed 81/81 profile clone fences, allocated
a 955090-token KV pool, and captured 9/9 PIECEWISE graphs in 9 seconds using
1.62 GiB. Endpoint health arrived in 244 seconds. Semantic probes passed. The
exact p498/o512 metric produced 512 coherent tokens at 311.856 ms client TTFT,
10.548628 seconds server decode, and 48.531479 corrected output tok/s. JSON and
color canaries each passed 16/16. Teardown was graceful; both per-card health
and compiled collective health passed; launcher exit was 0. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_20260825T235000Z_june_scratch`.
Key hashes: source contract `9e5b64c0...`, metric `c04b9f98...`, JSON
`7117ac66...`, color `70ea2934...`, and committed run log `d3b043da...`.

COMMIT HYGIENE -> the failed and successful server logs were mechanically
converted to ASCII after capture. Their raw -> committed SHA256 pairs are
`1eadeb70...` -> `493be3c6...` and `9ddb6e0b...` -> `cff23d0e...`.

VERDICT -> true June source is a measured +3.166559 tok/s, or +6.98 percent,
over the 45.364920 tok/s August-adapter native-MoE control. It reaches only
56.52 percent of Steve's 85.869114 tok/s and leaves 37.337635 tok/s, or
1.7693x, unexplained. The scratch-aware Python interface is required; wholesale
June source is not the missing speed lever. Steve's recorded fresh 54 MB versus
restored 67 MB `_xpu_C` control differed by only about 3 percent, so binary
size is also lower priority. Next, instrument actual decode graph-piece replay
and per-family host/device time. Transfer proven mechanisms separately to dense
27B, whose lack of routed MoE removes this scratch/dispatcher confound, and
derive its profile clone fence from its own collective shapes.

### 2026-08-26a - Exact graph topology matches; synchronized decode is 3.982x slower

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision
`cced56592e8c8935f8220836b4baa04dfd389118`; true-June vLLM source
`e190923b`; June-9 minimal native package; recovered `2dd55f38`
scratch-aware MoE dispatcher; TP=2/PP=1; direct P2P; clone-completion fence
for profile tensors with at least 8192 rows; PIECEWISE 9-size capture; built-in
device-synchronized decode timing on rank 0 after 32 skipped steps and every
16th step; built-in graph replay trace capped at 4096 records; fresh compile
cache. Steve comparison uses the committed rank-0 reference derived from
timing-summary SHA256 `7e9d805a...` and run-summary SHA256 `6ab63849...`.

COMMAND:

```text
./bin/gpu-run env \
  STAMP=20260826T002000Z_june_synctiming SOURCE_STACK=june-e190 \
  P2P_ACCESS=1 PROFILE_FENCE_MIN_ROWS=8192 STALL_TIMEOUT=300 \
  DECODE_TIMING=1 DECODE_TIMING_SYNC=1 \
  DECODE_TIMING_SKIP_FIRST=32 DECODE_TIMING_STEP_SKIP_FIRST=32 \
  DECODE_TIMING_STEP_EVERY=16 \
  CUDAGRAPH_REPLAY_TRACE=1 CUDAGRAPH_REPLAY_TRACE_MAX_LINES=4096 \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> fresh compile, 81/81 profile clone fences, native dense and routed
MoE INT8 selection, 9/9 captures, semantic probes, and both 16/16 canaries
passed. The synchronized diagnostic produced 35.469940 corrected output tok/s,
508.363 ms client TTFT, and 14.433521 seconds server decode. Graceful teardown
left both per-card probes and compiled-collective health green; exit was 0.
The replay trace reported `total_piecewise_compiles=41` and observed every
piece index 0..40. There were 369 capture starts/finishes, 492 direct
starts/finishes, and 1187 replay starts/finishes before the configured trace
cap. This exactly matches Steve's recorded 41-piece topology.

RESULT -> 62 pure-decode timing steps put local rank-0 model-forward at
22.674753 ms versus Steve's 5.694625 ms: +16.980128 ms and 3.9818x. Other
matched nonexclusive labels were GDN 3.927777 versus 1.584578 ms (2.4788x),
postprocess 1.903643 versus 0.312854 ms (6.0848x), logits 1.585079 versus
0.229150 ms (6.9172x), local argmax 1.149508 versus 0.070528 ms (16.2986x),
and sampler 0.663268 versus 0.144735 ms (4.5826x). Steve's synchronized
endpoint was 84.307543 tok/s, so the timing gap is real endpoint execution,
not only observer overhead. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_20260826T002000Z_june_synctiming`.
Key SHA256 values: replay `1d031a65...`, timing summary `3c80aa75...`,
comparison `1e89a7fa...`, metric `790e7145...`, JSON `b3787b35...`, color
`2c6bff6e...`, and committed run log `0f70231e...`. The server log was
mechanically made ASCII and trailing-whitespace-clean after capture: raw
`58d42b6b...` -> committed `e527d1e5...`.

SOURCE REVIEW -> the prior rebuilt package is the June-9 minimal patch over
`28e1f5e`, not the native tree present for Steve's June-19 timing. The live
kernel Git object database resolves exact checkpoint `122b698b` (June 16,
+5054/-169 across 24 files) and later child `3ed399a` (June 19 after the
17:02 UTC timing run). Compared with the June-9 reconstruction, GDN executable
source and grouped-GEMM Xe2 base tile/policy are unchanged. RMSNorm fusion is
disabled, dense INT8 uses the same default-one scratch behavior, and the
layerlet/sidecar arms are default-off. The active checkpoint delta is
`_xpu_C::per_token_quant_int8_xpu_out`: mixed workspace performs GEMM1 and
GEMM2 quantization in each of 40 MoE layers, or 80 calls/step. Without that
schema, the scratch-aware dispatcher allocates temporary quant outputs then
copies them into workspace buffers. Steve explicitly unset fused SiLU+quant.

VERDICT -> graph count, piece selection, and broad June vLLM source are closed.
The first controlled native A/B is exact `122b698b` siblings with the same
vLLM source, Python dispatcher, graph, collective, and launch configuration.
It tests native scratch-targeted quant output, not shared-object size or
experimental layerlets. For dense 27B, repeat this replay/timing census and
port only proven reusable quant/output primitives through a dense-specific
adapter; derive its collective fence from its own profile shapes.

### 2026-08-26b - Exact June-16 native scratch output gains 3.79 percent

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision
`cced56592e8c8935f8220836b4baa04dfd389118`; true-June vLLM source
`e190923b`; scratch-aware MoE dispatcher from `2dd55f38`; exact clean native
checkpoint `122b698bc245d31668a7fe5f2ad5ce1d07ba08ca`; pinned torch 2.11 image;
oneAPI DPC++ 2025.3.3; Release; Xe2 `bmg-g21-a0` AOT; MoE and GDN enabled;
TP=2/PP=1; direct P2P; clone-completion fence for profile tensors with at least
8192 rows; PIECEWISE 9-size capture; fresh compile cache. The runtime package
is a copy of the June-9 control with only `_xpu_C.abi3.so`,
`libgrouped_gemm_xe_2.so`, and `libgdn_attn_kernels_xe_2.so` replaced. Installed
`_xpu_C` RUNPATH is `$ORIGIN`. Native hashes are `631f7331...`, `7d38d160...`,
and `ee0481c8...` respectively.

COMMAND:

```text
docker run --rm --user 1000:1000 --entrypoint bash \
  -v /mnt/vm_8tb/b70/steve-repro/june122-xpuc-regular-20260826:/repro \
  intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94 \
  -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null; \
       VLLM_XPU_AOT_DEVICES=bmg-g21-a0 \
       VLLM_XPU_XE2_AOT_DEVICES=bmg-g21-a0 \
       ninja -C /repro/build -j2 -v _xpu_C'

./bin/gpu-run env \
  STAMP=20260826T012300Z_june122_endpoint SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint P2P_ACCESS=1 \
  PROFILE_FENCE_MIN_ROWS=8192 STALL_TIMEOUT=300 \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the first relocated binary import failed closed because the build
RUNPATH still named `/repro/build`. `cmake --install --component _xpu_C`
rewrote it to `$ORIGIN`. The new `native-out` preflight then proved exact
schemas and XPU dispatch for `_xpu_C::per_token_quant_int8_xpu_out` and
`_xpu_C::silu_and_mul_quant_int8_xpu_out`; the original June-9 `full` suite
also remained green. No GPU was touched until both CPU-only lanes passed.

RESULT -> model load, 81/81 profile clone fences, native dense and routed MoE
INT8 selection, 9/9 graph captures, semantic probes, and both 16/16 canaries
passed. The endpoint measured 50.370643 corrected output tok/s, 307.853 ms
client TTFT, and 10.163436 seconds server decode. Graceful teardown left both
per-card probes and compiled-collective health green; exit was 0. Evidence:
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_20260826T012300Z_june122_endpoint`.
Key SHA256 values: metric `02c20e0c...`, JSON `3f3497de...`, color
`ee7eafd9...`, preflight `38788447...`, run log `5d398df5...`, and committed
server log `2713348e...`. The server log was mechanically made ASCII and
trailing-whitespace-clean after capture: raw `cdf6a929...` -> committed
`2713348e...`; the run log was trailing-whitespace-cleaned from raw
`d81c7ae2...` to committed `5d398df5...`.

VERDICT -> compare endpoint with endpoint: 50.370643 versus the matched
June-9 true-source control at 48.531479 is +1.839164 tok/s, or +3.79 percent.
The 35.469940 result is a deliberately synchronized diagnostic and is not an
apples-to-apples baseline. Native scratch-targeted quant output is a real win,
but it explains only a small part of Steve's remaining 85.869114 tok/s gap.
Next: repeat synchronized timing on `NATIVE_STACK=june122-checkpoint`, then
measure integrated graph collective/runtime cost. Dense 27B should inherit the
operator-presence, graph-piece, timing, and correctness methodology, not the
MoE-only workspace/layerlet code or the Qwen35-specific 8192-row fence.

### 2026-08-26c - June-16 synchronized gain is 0.680 ms inside model-forward

CONFIG -> same exact model, image, true-June source, scratch-aware dispatcher,
`122b698b` native runtime, TP=2 direct-P2P collective, 8192-row profile clone
fence, and 9-size PIECEWISE graph as 2026-08-26b. This arm enables rank-0
device-synchronized decode timing after 32 skipped steps and every 16th step,
plus the 4096-record graph replay trace. It uses a fresh compile cache. The
matched June-9 reference is 2026-08-26a under the identical timing protocol.

COMMAND:

```text
./bin/gpu-run env \
  STAMP=20260826T013300Z_june122_synctiming SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint P2P_ACCESS=1 \
  PROFILE_FENCE_MIN_ROWS=8192 STALL_TIMEOUT=300 \
  DECODE_TIMING=1 DECODE_TIMING_SYNC=1 \
  DECODE_TIMING_SKIP_FIRST=32 DECODE_TIMING_STEP_SKIP_FIRST=32 \
  DECODE_TIMING_STEP_EVERY=16 \
  CUDAGRAPH_REPLAY_TRACE=1 CUDAGRAPH_REPLAY_TRACE_MAX_LINES=4096 \
  I_KNOW_P2P_WEDGES=1 \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> 81/81 profile clone fences, model load, 9/9 graph captures,
semantic probes, and both 16/16 canaries passed. The synchronized diagnostic
measured 36.429308 corrected output tok/s, 503.144 ms client TTFT, and
14.053663 seconds server decode. Both per-card probes and compiled-collective
health passed after graceful teardown. The replay trace again observed every
piece index 0..40 and reported 41 pieces.

RESULT -> across 62 pure-decode samples, rank-0 model-forward is 21.994441 ms
versus 22.674753 ms on June-9: -0.680311 ms, or -3.00 percent. The synchronized
endpoint improves from 35.469940 to 36.429308 tok/s (+2.70 percent). GDN is
3.948703 versus 3.927777 ms, postprocess 1.907910 versus 1.903643 ms, logits
1.585771 versus 1.585079 ms, local argmax 1.151977 versus 1.149508 ms, and
sampler 0.677830 versus 0.663268 ms. These broad families are unchanged within
run noise. Steve's model-forward is 5.694625 ms, leaving 16.299816 ms and a
3.8623x ratio.

EVIDENCE ->
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_20260826T013300Z_june122_synctiming`.
Key SHA256 values: timing summary `6f749cb0...`, Steve comparison
`251a70ff...`, June-9 comparison `16eb883e...`, metric `cc8e4928...`, JSON
`052e8470...`, and color `84056f8d...`. The server log was mechanically made
ASCII and trailing-whitespace-clean from raw `5302c504...` to committed
`2406da65...`; the run log was trailing-whitespace-cleaned from raw
`650147b6...` to committed `64b17ed8...`.

VERDICT -> the exact native quant output path is now fully localized: it saves
about 0.68 ms in model-forward and does not change GDN or post-model runtime.
It is not the remaining speed lever. The next target is integrated device and
runtime time for the 81 compiled TP collectives inside each graph-replayed
decode step; current nested Python labels only see direct/capture calls, not
replay-internal collectives. Dense 27B must repeat this collective census on
its own graph because its layer count, shapes, and communication schedule differ.

### 2026-08-26d - Runtime profile exposes 41 host waits; CPU affinity is neutral

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision `cced5659`; pinned
`f2e5a94e` image; true-June vLLM source `e190923b`; exact June-16 native
checkpoint `122b698b`; TP=2 direct P2P; no MTP or prefix cache; 41-piece
PIECEWISE graph; reused the already proven June-16 compile cache. The first arm
enabled the e190 torch profiler for a separate p512/o512 request with two
delay iterations and eight recorded iterations. The second clean arm pinned TP
worker 0 to `0-7,16-23` and worker 1 to `8-15,24-31` while leaving EngineCore
unbound. Both workers used memory node 0 because the 1950X exposes UMA.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint XPU_PROFILE=1 P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_20260826T013300Z_june122_synctiming \
  STAMP=20260826T020000Z_june122_xpu_profile \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh

./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CPU_BIND=split-die P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_20260826T013300Z_june122_synctiming \
  STAMP=20260826T022000Z_cpu_split_die_fixed \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both profile ranks recorded eight decode iterations. Every token on
both ranks had median counts of 41 `zeFenceReset`, 41
`zeEventHostSynchronize`, 82 `zeCommandQueueExecuteCommandLists`, 123
`zeCommandListAppendBarrier`, and 105 kernel launches. The 41/41/82 signature
matches the 41 PIECEWISE graph boundaries. Visible device work averaged
1.671066 ms on rank 0 and 2.170872 ms on rank 1. Rank 0 exposed 0.960325 ms
GEMM, 0.367745 ms GDN, 0.222418 ms full attention, and 0.026757 ms collective
per iteration. Rank 1 exposed a 0.525390 ms final all-gather. Captured routed
MoE and the 81 TP all-reduces did not appear as individual Kineto device
events. The profiler perturbed later runtime: the post-profile endpoint was
38.389977 tok/s and profiled CPU iteration ranges were about 33.4 ms, so these
are diagnostic-only timings.

RESULT -> the first affinity attempt failed before worker initialization.
e190 bound EngineCore to rank 0's CPU mask, then rank 1 could not expand beyond
its parent's allowed CPUs and `numactl` rejected `8-15,24-31`. Both health
layers remained green. The exact-source adapter was narrowed to leave
EngineCore unbound and bind only worker subprocesses. The corrected arm logged
both intended worker masks, loaded and captured in 101 seconds, passed the
semantic probe and both 16/16 canaries, and measured 50.406626 tok/s, 306.627
ms TTFT, and 10.155217 seconds server decode. The clean unbound endpoint was
50.370643 tok/s, 307.853 ms, and 10.163436 seconds. Graceful teardown left both
cards and the compiled two-rank collective healthy.

VERDICT -> the local runtime crosses every graph-piece boundary each token;
the leading residual is integrated XPUGraph, collective, and host coordination,
not missing capture topology. Kineto's visible device total is incomplete and
must not be subtracted from synchronized model-forward. Split-die worker
affinity changes throughput by only +0.07 percent and is not a reason to move a
card. Next, alter one graph/runtime boundary at a time and retain the clean
endpoint plus coherence and health gates. Dense 27B must repeat its own graph
piece, driver-call, collective-shape, and fence-threshold census. Full report:
`docs/20260826_qwen36_graph_runtime_profile.md`.

### 2026-08-26e - Full-decode capture gains 22.20 percent

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision `cced5659`; pinned
`f2e5a94e` image; true-June vLLM source `e190923b`; exact June-16 native
checkpoint `122b698b`; TP=2 direct P2P; no MTP; no prefix cache; fresh compile
cache. This intervention changes the accepted PIECEWISE graph mode to
`FULL_DECODE_ONLY` and selects `TRITON_ATTN`. Mixed prefill stays outside full
capture. The health stall limit was raised to 600 seconds so a slow first
capture would not be killed mid-initialization.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 \
  STALL_TIMEOUT=600 STAMP=20260826T024000Z_full_decode_triton \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the server became healthy in 259 seconds. Six decode FULL graphs
captured in 26 seconds and consumed 0.11 GiB. The semantic probe passed. The
p498/o512 metric measured 61.553562 corrected output tok/s, 332.269 ms client
TTFT, and 8.317629 seconds server decode. The exact June-16 PIECEWISE control
measured 50.370643 tok/s, 307.853 ms, and 10.163436 seconds. Full decode gains
11.182920 tok/s, or 22.20 percent, and cuts server decode by 1.845808 seconds;
TTFT rose 24.416 ms in this one comparison. JSON and color canaries both
passed 16/16. Graceful teardown left both card probes and compiled two-rank
collective health green.

VERDICT -> replay-boundary reduction is the largest local lever measured after
enabling graph capture itself. The arm reaches 71.68 percent of Steve's
85.869114 tok/s and leaves 24.315552 tok/s. It does not reproduce Steve's
accepted PIECEWISE command; PIECEWISE remains the provenance control. It does
prove that FULL capture is viable for the exact no-MTP June stack and retires
the blanket B70 FULL-blocked statement. The stock/MTP GDN speculative-shape
and SYCL-scratch failures remain separate. Next, profile this full-decode arm
to prove the expected fence/host-wait collapse, then isolate the remaining 81
collectives and native MoE work inside its single replay. Dense 27B must test
its own no-MTP full-decode arm rather than inherit the old MTP blocker.

### 2026-08-26f - Full-decode profile collapses replay to one fence

CONFIG -> same exact `FULL_DECODE_ONLY` plus `TRITON_ATTN` no-MTP control as
2026-08-26e, with the proven compile cache reused and a bounded torch XPU
profile: two delayed iterations and eight recorded decode iterations per rank.
The profile request was separate from the ordinary metric and repeat canaries.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN XPU_PROFILE=1 P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=600 ALLOW_EXISTING_CACHE=1 \
  CACHE_DIR=/mnt/vm_8tb/b70/vllm_cache_qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_20260826T024000Z_full_decode_triton \
  STAMP=20260826T025000Z_full_decode_profile \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both ranks recorded eight iterations. Median per-token driver counts
fell from PIECEWISE's 41 `zeFenceReset`, 41 `zeEventHostSynchronize`, and 82
`zeCommandQueueExecuteCommandLists` calls to FULL decode's 1, 2, and 2.
Visible device work was only 1.077620 ms on rank 0 and 1.062816 ms on rank 1;
GDN, full attention, routed MoE, and the 81 all-reduces are inside the opaque
full graph and do not appear as individual device events.

RESULT -> after skipping the first two profiled iterations, six steady-state
rank-0 samples began the longest wait 9.215851 ms into the iteration, waited
2.731938 ms, and had 2.163153 ms of host work after the next graph submission;
mean iteration range was 14.429410 ms. Rank 1 measured 8.938722, 3.300870,
1.962969, and 14.517247 ms. This wait is the exposed tail of the preceding
asynchronous full graph after overlap with host input preparation, not the full
graph duration.

RESULT -> the profiled request measured 61.543223 tok/s and the ordinary
request afterward measured 61.559842 tok/s, within 0.01 percent of the clean
61.553562 result. Semantic output, both 16/16 canaries, graceful teardown, both
card probes, and compiled two-rank collective health passed.

VERDICT -> the 22.20 percent FULL win is exactly replay-boundary reduction,
and that boundary is now one graph per token. The remaining 24.3156 tok/s gap
is dominated by execution inside the opaque graph plus smaller host work. Next
work must expose or optimize the 81 in-graph collectives and native MoE path;
further piece-count reduction is closed. Dense 27B should reuse the same
PIECEWISE-versus-FULL driver census and opacity guard.

### 2026-08-26g - Triton W8A8 MoE beats June122 native by 5.57 percent

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision `cced5659`; pinned
`f2e5a94e` image; true-June vLLM source `e190923b`; June122 native package;
TP=2 direct P2P; no MTP or prefix cache; `FULL_DECODE_ONLY` plus
`TRITON_ATTN`; fresh cache. June e190 exposes `--moe-backend triton` but its
generic `TritonExperts` gate admits INT8 only on CUDA. The opt-in intervention
relaxes only the exact per-channel-weight/dynamic-token Quark W8A8 pair and
prints a marker in every process.

COMMAND -> first propagation-debug attempt, then scoped reset and corrected
transaction:

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN MOE_BACKEND=triton P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T031000Z_full_decode_triton_moe \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh

./bin/xe-reset --method rebind

./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN=TRITON_ATTN MOE_BACKEND=triton P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T031500Z_full_decode_triton_moe_retry \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the first attempt over-escaped a shell comparison and propagated the
intervention value as `0`. Both workers rejected Triton during model
construction, before weight loading, profile execution, or graph capture.
Per-card and compiled-collective post-health passed. The scoped unbind/rebind
reset returned both endpoints, four xe auxiliary bindings, both card probes,
and compiled collective health under the same boot ID.

RESULT -> the corrected run emitted `triton_int8_intervention=1`; every process
logged the intervention marker, and both workers selected `Using TRITON Int8
MoE backend`. Model load completed, both ranks crossed all 81 profile clone
fences, and six FULL decode graphs captured in 38 seconds using 0.11 GiB. The
p498/o512 metric measured 64.984330 corrected output tok/s, 363.490 ms client
TTFT, and 7.878107 seconds server decode. The matched native-MoE FULL control
was 61.553562 tok/s, 332.269 ms, and 8.317629 seconds. Triton gains 3.430768
tok/s (+5.57 percent) and saves 0.439522 seconds of server decode. Semantic
output, JSON 16/16, color 16/16, graceful teardown, both card probes, and
compiled two-rank collective health passed.

EVIDENCE ->
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_moe_triton_intervention_20260826T031500Z_full_decode_triton_moe_retry`.
The constructor-debug attempt is preserved beside it with stamp
`20260826T031000Z_full_decode_triton_moe`.

COMMIT HYGIENE -> successful server/run logs were mechanically converted to
ASCII and trailing-whitespace-clean from raw `445c28e4...`/`1e350d2d...` to
committed `1122f05c...`/`446d7c78...`. The constructor-debug server/run logs
changed from `734683d4...`/`27a2c68c...` to
`5f477cf7...`/`35d5917a...`.

VERDICT -> the recovered June122 native grouped-MoE path is slower, not the
missing Steve mechanism. The best exact local arm now reaches 75.68 percent of
Steve's 85.869114 tok/s and leaves 20.884784 tok/s. Next isolate or replace the
81 TP collectives inside the opaque full graph, then inspect other accepted
runtime/kernel families. Dense 27B must transfer the graph-first method and
omit this MoE-only support gate, expert layout, grouped GEMM, layerlet, and
sidecar code.

### 2026-08-26h - Push preinit closes IPC import; loaded graph submit stalls

CONFIG -> exact June vLLM source `e190923b`; June122 native runtime; TP=2 XCCL
group; local Level Zero push-allreduce SO
`3ed15e33235d359e3cd696bf844cc8781da475a2d144f3e2b12d215feea3844d`;
strict communicator preinitialization; `[5120]` BF16 per rank; XPUGraph native
push capture; 25-second expected-stall timeout. The oracle does not load model
weights. `PUSH_AR_GRAPH_INPLACE=1` removes only the adapter's graph-time clone;
the June outer compiled custom op already owns its required clone.

COMMAND ->

```bash
./bin/gpu-run env STAMP=20260826T040500Z_safe_repro_fixed \
  ORACLE_TIMEOUT=25 EXPECT_LOADED_GRAPH_STALL=1 \
  bash vllm/w8a8/run_qwen36_push_ar_init_oracle.sh

./bin/xe-reset --method rebind
```

RESULT -> both ranks completed scratch, shared barrier, and IPC event-pool
exchange before graph capture. Both logged `PREINIT group=tp:0 ... ready=1`,
then `capturing=True`. Neither returned from the native push graph call before
timeout. The same stall occurred with the extra graph clone retained and with
it removed. The timeout-safe wrapper removed the container and reported the
known blocker. Scoped unbind/rebind restored both endpoints on the same boot;
both single-card probes and the compiled TP=2 collective probe passed.

RESULT -> Steve's exact 85.869114 TP2 command was re-read from the retained
artifact. It explicitly uses async scheduling and prefill-only GDN fallback,
but no attention option. June `e190923b` therefore selects its XPU default
FlashAttention backend. The local 61.553562/64.984330 FULL arms explicitly
force Triton attention and remain labeled graph/runtime interventions.

VERDICT -> early communicator creation concretely fixes the previous
asymmetric Level Zero IPC import. The remaining push blocker is loaded June
vLLM/XCCL native graph submission, not math, handle exchange, rank skew, or
clone count. Do not attempt a full-model push run until this oracle completes
capture and replay. Standalone torch XPUGraph success is insufficient. Dense
27B must inherit this loaded-context gate, while omitting the experimental
adapter and binary until they pass it. The next one-factor exact-stack test is
the default-FlashAttention versus forced-Triton graph boundary.

### 2026-08-26i - Default FlashAttention cannot enter FULL SYCL graph

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision `cced5659`; pinned
`f2e5a94e` image; true-June vLLM source `e190923b`; June122 native package;
TP=2 direct P2P; `FULL_DECODE_ONLY`; default attention; Triton MoE
intervention; no MTP or prefix cache; fresh cache. This changes only attention
from the successful 64.984330 tok/s FULL/Triton-attention/Triton-MoE arm.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint CGMODE=FULL_DECODE_ONLY \
  ATTN= MOE_BACKEND=triton P2P_ACCESS=1 \
  I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T041000Z_full_default_attn_triton_moe \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> both workers selected Flash Attention and Triton MoE. Model load and
compile completed, and both ranks crossed all 81 profile clone fences. The
first of six FULL decode captures then failed in
`vllm_xpu_kernels/flash_attn_interface.py` at `_vllm_fa2_C.varlen_fwd`:
`sycl_ext_oneapi_work_group_scratch_memory` is not available through the SYCL
Graph extension. No endpoint metric was produced. Teardown left both card
probes and the compiled two-rank collective probe green.

EVIDENCE ->
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_moe_triton_intervention_20260826T041000Z_full_default_attn_triton_moe`.
The server log was mechanically converted to ASCII and LF from raw SHA256
`c689c0bf4b485353c09c2a01db344ad6aeb5de9e3a14fcec67371eb8edc2f833`
to committed SHA256
`8a4c17acede89f268c7bf1c43ebc3316c0c1fbcb88a9f9b0579aa92bff6ab7c8`.
Trailing-space cleanup changed the ASCII run log from raw
`64cb238f6b5f545b3648b3aeda0be58ae406cbee7ea864f0ed054daa19fc2443`
to committed
`1c7bf118d8ef6d0faded776d33535a73b4aaf517256f2c820a79bc1e8b0f6452`.

VERDICT -> Steve's no-override default FlashAttention identity works with his
PIECEWISE graph policy, not with this local FULL speed boundary. Triton
attention is a required and labeled current-runtime intervention for the
61.553562/64.984330 FULL results. Do not retry default Flash FULL without a
concrete Flash-kernel or isolated user-mode SYCL Graph change. Linux 7.1 stays
fixed.

### 2026-08-26j - June source-default c10d collectives cross PIECEWISE TP2

CONFIG -> exact June source/native provenance control; Qwen3.6-35B-A3B Quark
W8A8; TP=2 direct P2P; PIECEWISE; default Flash attention; native MoE; async
scheduling; no MTP or prefix cache; fresh cache. All four recovered custom
collective switches were changed together from one to their June source
defaults of zero. A scoped PCI unbind/rebind first restored both endpoints,
both card probes, and compiled TP=2 collective health on unchanged boot ID
`e2d5777d-f6bb-4d92-a718-0fb07ae17919`.

COMMAND ->

```bash
./bin/xe-reset --method rebind

./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint COLLECTIVE_MODE=source-default \
  CGMODE=PIECEWISE ATTN= MOE_BACKEND=auto \
  P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T043000Z_collectives_source_default \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> the endpoint reached health in 320 seconds. The p498/o512 metric
measured 51.091606 corrected output tok/s, 315.694 ms client TTFT, and
10.020679 seconds server decode. Semantic output, JSON 16/16, color 16/16,
graceful teardown, both card probes, and compiled TP=2 collective health all
passed. Both persisted rank graphs contained 243 `_c10d_functional`
all-reduce references and zero `torch.ops.vllm.all_reduce` references. The
initial evidence gate rejected only the custom-route-specific `profile clone
complete` marker after all workload and health gates passed; the corrected
gate now requires mutually exclusive compiled graph identities per route.

EVIDENCE ->
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_collectives_source_default_20260826T043000Z_collectives_source_default`.
`compiled_collective_route_evidence.txt` preserves both rank-graph hashes,
route counts, and a representative c10d source line outside the runtime cache.
The server log was mechanically converted to ASCII and LF from raw SHA256
`6991f9542fc7a8f4b7db51b51af79d9b913e96a6e71f63bc18fd94ade3d0aa76`
to committed SHA256
`412de7d65a84ad816c59f7d98d23d73a040e4774a9150eb8730db61d4f6eb469`.
Trailing-space cleanup changed the ASCII run log from raw
`bec1953c3ade2f5d4694eb84f7c806af9374a8d532713ffc1037c16805684008`
to committed
`4fc85ba2a8e30d58e9e222911bafc1f0bc0be56c1897dc692ff59427ef1ef85b`.

VERDICT -> June source-default c10d and the recovered custom `vllm.all_reduce`
route both cross exact PIECEWISE TP=2 on kernel 7.1/runtime 26.22. The
source-default observation is 0.720963 tok/s (+1.43 percent) above the nearest
50.370643 custom control, but one sample is not a speed claim. The parent
accepted-result summary's null env fields cannot observe child-launcher
exports, while the retained launcher was reconstructed after the June result;
exact June-15 collective identity remains ambiguous. Preserve both labeled
controls. Next compare route-specific FULL/Triton execution to isolate the 81
in-graph collectives without changing the fixed host kernel.

### 2026-08-26k - Source-default c10d sets a 66.2555 tok/s FULL best

CONFIG -> exact Qwen3.6-35B-A3B Quark W8A8 revision `cced5659`; pinned
`f2e5a94e` image; true-June vLLM source `e190923b`; June122 native package;
TP=2 direct P2P; `FULL_DECODE_ONLY`; `TRITON_ATTN`; Triton W8A8 MoE
intervention; no MTP or prefix cache; fresh cache. All four custom-collective
switches remained at zero, so this changes only the collective route from the
matched 64.984330 tok/s custom-op FULL control.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint COLLECTIVE_MODE=source-default \
  CGMODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN MOE_BACKEND=triton \
  P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T045000Z_full_triton_source_default_collectives \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> endpoint health arrived in 326 seconds. All six FULL decode graphs
captured. The p498/o512 metric measured 66.255519 corrected output tok/s,
360.426 ms client TTFT, and 7.726791 seconds server decode. The matched custom
control was 64.984330 tok/s, 363.490 ms, and 7.878107 seconds. Source-default
c10d gains 1.271189 tok/s (+1.96 percent) and saves 0.151316 seconds decode.
Semantic output, JSON 16/16, color 16/16, graceful teardown, both card probes,
and compiled TP=2 collective health passed. Each persisted rank graph contains
243 `_c10d_functional` all-reduce references and zero
`torch.ops.vllm.all_reduce` references.

EVIDENCE ->
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_moe_triton_intervention_collectives_source_default_20260826T045000Z_full_triton_source_default_collectives`.
`compiled_collective_route_evidence.txt` preserves both rank graph hashes and
route counts. Mechanical ASCII/LF/trailing-space cleanup changed the server log
from raw SHA256
`62f077bb64c2561f98da8c004059ee7a4962db67d0081e86497fd4be137214cb`
to committed
`18edc162da160bbf08e13125b1c44e99effeb376814e77f5048ebc4a719b4fcd`,
and the run log from raw
`f2bf673d1391d9f4650a583193e063965b2cc8519948cb3046a6bf2fcc576bbb`
to committed
`762068f02790f276a3970daac832f21485a814b9b1a67a803429fcd8730bf484`.

VERDICT -> source-default c10d is the current campaign best and reaches 77.16
percent of Steve's 85.869114 tok/s. The custom `vllm.all_reduce` wrapper is not
the missing accelerator on this full-graph stack. Because the route delta is
only 1.96 percent and each arm currently has one matched sample, repeat before
promoting the delta as stable. Linux 7.1 remains fixed; the remaining gap is in
another user-mode runtime/kernel behavior, not host-kernel provenance.

### 2026-08-26l - C-S-C-S replicates source-default c10d advantage

CONFIG -> fresh-cache repeats of the exact FULL/Triton control from entry k.
All model, source, native binary, TP=2 P2P, graph, attention, MoE, scheduling,
request, and health settings remained fixed. The third arm restored all four
custom-collective switches to one; the fourth returned all four to zero.

COMMAND ->

```bash
./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint COLLECTIVE_MODE=clone-custom \
  CGMODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN MOE_BACKEND=triton \
  P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T050000Z_full_triton_custom_collectives_repeat \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh

./bin/gpu-run env SOURCE_STACK=june-e190 \
  NATIVE_STACK=june122-checkpoint COLLECTIVE_MODE=source-default \
  CGMODE=FULL_DECODE_ONLY ATTN=TRITON_ATTN MOE_BACKEND=triton \
  P2P_ACCESS=1 I_KNOW_P2P_WEDGES=1 STALL_TIMEOUT=900 \
  STAMP=20260826T051500Z_full_triton_source_default_repeat \
  bash vllm/w8a8/run_qwen36_s2b_clone_exact_control.sh
```

RESULT -> custom repeat measured 65.004555 tok/s, 361.244 ms TTFT, and
7.875629 s server decode. Source-default repeat measured a new best of
66.432037 tok/s, 360.671 ms, and 7.706482 s. Combined C-S-C-S samples are:

```text
custom:        64.984330  65.004555  mean=64.994443
source-default: 66.255519  66.432037  mean=66.343778
mean delta:                              +1.349335 tok/s (+2.08 percent)
```

The custom within-route spread is 0.031 percent; c10d spread is 0.266 percent.
Both repeats passed semantic output, JSON 16/16, color 16/16, graceful
teardown, both card probes, and compiled TP=2 collective health. Custom rank
graphs each contain 162 `torch.ops.vllm.all_reduce` and zero c10d references;
source-default graphs each contain 243 c10d and zero custom references.

EVIDENCE -> the custom repeat is
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_moe_triton_intervention_20260826T050000Z_full_triton_custom_collectives_repeat`;
the source-default repeat is
`results/logs/qwen36_s2b_exactcc_clone_p2p1_june_e190_native_june122_cg_full_decode_only_attn_triton_attn_moe_triton_intervention_collectives_source_default_20260826T051500Z_full_triton_source_default_repeat`.
Each contains compiled route evidence. Mechanical ASCII/LF/trailing-space
cleanup changed custom server/run logs from raw
`eef5ab9cb2dfb3cbd58883c159c93c050d2b7d4b13010385ab504990fda90a00`/
`48f1a6ac20c6ca5283fedd6d80afb4adf413cacbb8c0f6f20b283acc0e063413`
to committed
`dc39fae51a1c984ab033a35b47e54ebf171fbde3a5db15b40a95ecd63dff5c91`/
`bcc36ce0d0ff3de49175b5a6661dbfb49b9f4e5833fd476f0798ac9fddc0ca55`,
and source-default server/run logs from raw
`6cd6ed63088ea1b2de9d1a796d685210441a693f32a5f96ebeddf27f4c7c1cb5`/
`d4bf576f94f6b83ddcb1dc03e297a8aea82db78c78b43d1d21bd2d87cf8eac33`
to committed
`00d5b2fba29e312611911e0100e1dc48c674eb046c35bfeb40feb1d59682308e`/
`17d01ad19678749fc182d3981e23965b5509849843a924d8f40809a3da40767a`.

VERDICT -> the approximately 2 percent source-default c10d advantage is
replicated, not run noise. Use source-default for the fastest no-MTP/no-DFlash
FULL control and retain the custom route only as accepted-provenance evidence.
The new 66.432037 best reaches 77.36 percent of Steve's 85.869114 and leaves
19.437077 tok/s. The next target is another accepted user-mode runtime/kernel
family, with Linux 7.1 unchanged.

### 2026-08-26m - Exact container already matches Steve-era UMD 26.14

CONFIG -> read-only runtime identity audit of pinned exact-control image digest
`f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94`;
no GPU devices, host package changes, or container mounts.

COMMAND ->

```bash
docker run --rm --entrypoint bash \
  intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94 \
  -lc 'dpkg-query -W intel-opencl-icd level-zero; \
       sha256sum /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.15.37833 \
                 /usr/lib/x86_64-linux-gnu/libze_loader.so.1.28.2'

docker image inspect \
  intel/vllm@sha256:f2e5a94eb1dba7ac91f247a69a87a6b3caa4ca24b9bb5e62ceed1a8b9dbe5d94 \
  --format '{{json .Config.Volumes}}'
```

RESULT -> the container package is Intel Compute Runtime
26.14.37833.4-1~24.04~ppa1 and Level Zero loader 1.28.2. SHA256 values are
`98605c30dcf0d6a0636f23898470086c8545494e198a8f375519b60b5daf983a`
for `libze_intel_gpu.so.1.15.37833` and
`0fe232b18985ae078dd546b57bc6d11bacf1030834c0544f7e3feb53ed71c1d0`
for `libze_loader.so.1.28.2`. Image volumes are null. The exact serve launcher
mounts model, cache, source, and native-kernel paths, not host UMD libraries.

VERDICT -> the exact vLLM process already uses Steve-generation UMD 26.14 over
the fixed Linux 7.1 KMD. Host Compute Runtime 26.22 is not the process UMD and
is not the missing reproduction lever. Do not downgrade kernel or host runtime
packages. The remaining 19.437077 tok/s lies in source/native behavior or host
hardware/topology, not an unperformed 26.14 user-mode match.

### 2026-08-26n - Clean-stack identity freeze and upstream branch refresh

CONFIG -> post-cleanup repository at `a045b98`; fixed Linux
`7.1.0-070100-generic`; two B70 `8086:e223` cards; no live server. The
existing dirty Ornith launchers, sitecustomize, and push-allreduce binary were
reviewed and preserved as user work. No quarantined source or binary was
restored.

COMMAND ->

```bash
git status --short --branch
git log --oneline --decorate -20
./bin/gpu-run bash -lc './bin/xpu-health'
./bin/gpu-run ./bin/xpu-collective-health --p2p 0 \
  --img vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f

git -C /mnt/vm_8tb/b70/steve-s2b/vllm fetch --all --prune --tags
git -C /mnt/vm_8tb/b70/steve-s2b/vllm-xpu-kernels fetch --all --prune --tags
git -C /mnt/vm_8tb/b70/steve-s2b/oneccl-src fetch --all --prune --tags
```

RESULT -> host Compute Runtime is `26.22.38646.4`, Level Zero loader is
`1.28.2-2`, IGC is `2.36.3`, DMC is `2.6`, GuC is `70.58.0`, and HuC
is `8.2.10` on both cards. All eight manifest artifacts are present. Both
per-card probes passed. The exact-image compiled P2P-off two-rank probe passed
ten functional all-reduces.

RESULT -> the newer retained official vLLM image is
`f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`:
vLLM `0.27.2rc1.dev77+gac7509e2b`, torch `2.13.0+xpu`, Compute Runtime
`26.27.39122.11`, Level Zero `1.32.0`, and oneCCL library SHA256
`3d6eb6672226592f59948ae82cb0ab961c2fe74e2234c9be1d0f2fdab2fed647`.
The retained Sglang image is `0.5.15.post1` on torch `2.12.0+xpu`; current
official Sglang release/head are `0.5.18`/`bede6bc37c5d`. Current official
vLLM and XPU-kernel heads are `cde7ba92da0e` and `a397c58eb778`.

RESULT -> Steve's fetched branch
`research/qwen36-int4-exactness-20260818` remains exactly
`44fc8fde09fc311d3099dab10366b672d9142ea4`; the June source remains
`e190923b32e1b87fe33d08264bff9215fb7770fc`. Official XPU-kernels was added
as a second remote and fetched without changing detached Steve work. Its
`a397c58e` head includes newer fused Qwen RMSNorm and GDN/MTP fixes. Steve
kernel fork main mixes about 19,000 inserted lines of production candidates
and WIP, so it must not be merged wholesale. The exact Steve control image
`f2e5a94e...` and every ABI-specific clean-stack extension are absent after
cleanup. The automatic collective-health default still names that missing
image and is therefore inconclusive unless `--img` is supplied.

VERDICT -> the clean P0 identity and health boundary is frozen. P1 is not
complete: Sglang, its XPU kernel, and every custom extension still require
pinned refreshed builds. Use official current source as the base, then port
Steve's dense INT8 output-buffer, scratch-ring, dependency, and quant-dedup
changes one factor at a time. Current Sglang supports compressed-tensors FP8
on XPU but still rejects compressed-tensors W8A8 INT8, so the local INT8 route
remains a deliberate port rather than an upstream feature.

### 2026-08-26o - Refreshed vLLM loads Qwen3.8 W8A8; eager TP2 is 3.53 tok/s

CONFIG -> local `qwen3.8-27b/w8a8-gptq` compressed-tensors checkpoint;
official vLLM image digest `f01e24f6...`; target only; text only; BF16 KV;
8K context; prefix cache off; async scheduling off; graph and torch.compile
off; source-default XCCL with P2P off; p512/o512 random benchmark. No custom
source or native extension was mounted.

COMMAND ->

```bash
./bin/gpu-run --card 0 env DEVICE=0 \
  bash vllm/w8a8/serve_qwen38_27b.sh smoke

./bin/gpu-run bash -lc '
  ./bin/xpu-collective-health --p2p 0 --img \
    vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
  B70_COLLECTIVE_HEALTH=0 IN=512 OUT=512 CONC=1 \
    bash vllm/w8a8/serve_qwen38_27b.sh run
  ./bin/xpu-health --img \
    vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
  ./bin/xpu-collective-health --p2p 0 --img \
    vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
'
```

RESULT -> TP=1 selected `TritonInt8ScaledMMLinearKernel` for
`CompressedTensorsW8A8Int8`, then failed the capacity gate while allocating
the 2.37 GiB BF16 LM head: 30.45 GiB was already allocated on a 31.89 GiB
card. Graceful teardown and the card-0 probe passed. This is a capacity result,
not a loader or kernel failure.

RESULT -> changing only the required capacity axis to TP=2 reached health in
204 seconds on the first smoke and 117 seconds on the timed run.
`/v1/models` returned `qwen3.8-27b-W8A8-gptq`; the deterministic probe
completed coherently with `Paris`. Each rank used 16.52 GiB for model load
and 17.13 GiB total consumed memory, leaving a 9.92 GiB KV cache and a
245,760-token aggregate cache census at this 8K configuration.

RESULT -> eight c1 p512/o512 requests measured 0.01 request/s, 3.53 aggregate
output tok/s, 1829.03 ms mean TTFT, 280.15 ms mean TPOT, and 3.57 tok/s
per-stream decode. The CSV is
`/mnt/vm_8tb/b70/results/sweep_qwen3.8-27b-W8A8-gptq-tp2-eager_20260826_082437.csv`
with SHA256
`e0ea44e577c1ad9034e4d648ed0124253079a8c3fb5b4bf51a81a5c033709db6`.
The preserved server log SHA256 is
`9ba9bb110d7b6e39e454927dad8904c6a53e22868aa1ee6a6db8f0da4f651871`.
Graceful teardown, both card probes, and the exact-image compiled two-rank
P2P-off collective probe passed.

VERDICT -> updated upstream vLLM now supplies a coherent true-INT8 loader and
Triton W8A8 dense route for this Qwen3.8 artifact. The 3.53 tok/s result is an
eager TP2 denominator, not an optimization or shelf qualification; concurrent
coherence, long context, graph/compile, MTP, and repeated timing remain open.
The next high-information vLLM arm is compile without graph, followed by a
separately guarded graph policy. The primary backend track still needs the
pinned Sglang 0.5.18/current-source build and deliberate W8A8 INT8 port.

### 2026-08-26p - Current vLLM nightly keeps the Qwen3.8 W8A8 eager control coherent

CONFIG -> official XPU nightly digest `2ac07cf8...`; source commit
`46638857fdbb`; torch `2.13.0+xpu` at `cf30153c...`; Triton XPU `3.7.2`;
vllm-xpu-kernels `0.1.13.2`; Compute Runtime `26.27.39122.11`; Level Zero
`1.32`; target only; TP=2; eager; 8K; P2P off; source-default XCCL. The vLLM
source is five commits behind `cde7ba92d`; those five later commits do not
touch the Qwen3.8, W8A8, XPU, compilation, or collective paths used here.

COMMAND ->

```bash
./bin/gpu-run bash -lc '
  img=vllm/vllm-openai-xpu@sha256:2ac07cf8fde4631de59912f2349729cf130947671b85c087550885cae8e65c46
  ./bin/xpu-collective-health --p2p 0 --img "$img"
  IMG="$img" B70_COLLECTIVE_HEALTH=0 \
    bash vllm/w8a8/serve_qwen38_27b.sh smoke
  ./bin/xpu-health --img "$img"
  ./bin/xpu-collective-health --p2p 0 --img "$img"
'
```

RESULT -> the exact-image compiled P2P-off collective preflight passed. Each
rank loaded 16.52 GiB and selected `TritonInt8ScaledMMLinearKernel` for
`CompressedTensorsW8A8Int8`. `/v1/models` returned the unambiguous
`qwen3.8-27b-W8A8-gptq` ID and the deterministic generation probe completed
coherently with `Paris`. Graceful teardown, both per-card probes, and the
exact-image compiled collective post-check passed. The preserved log is
`/mnt/vm_8tb/b70/b70_qwen38_w8a8_vllm_refresh.log`, SHA256
`415d2b0b37d9d866114581a350b7d2bb67c2ea6eff8ff81ed6ea08a8c2c4e857`.

VERDICT -> advance the refreshed vLLM eager control to digest `2ac07cf8...`.
This is an identity and coherence qualification, not a speed or shelf claim.
A matched performance run remains required before comparing it with the
earlier 3.53 tok/s denominator.

### 2026-08-26q - Torch 2.13 TreeSpec blocks Qwen3.8 compile without graphs

CONFIG -> the exact nightly configuration from 2026-08-26p; change only from
eager to vLLM compile/Inductor with CUDAGraph mode NONE, compile size 1, SYCL
collectives off, and P2P off. Both the legacy vLLM FX splitter
(`use_inductor_graph_partition=false`) and current default partitioner
(`true`) were tested. No graph capture occurred.

COMMAND ->

```bash
./bin/gpu-run bash -lc '
  img=vllm/vllm-openai-xpu@sha256:2ac07cf8fde4631de59912f2349729cf130947671b85c087550885cae8e65c46
  B70_COLLECTIVE_HEALTH=0 GRAPH=1 CGMODE=NONE COMPILESZ=1 \
    SYCLKERNELS=0 P2PACCESS=0 IGP=false \
    bash vllm/w8a8/serve_qwen38_27b.sh smoke
  ./bin/xpu-health --img "$img"
  ./bin/xpu-collective-health --p2p 0 --img "$img"
'
```

RESULT -> both ranks loaded the correct INT8 scheme and weights, then Torch
FX `split_module` failed before health while inspecting a cross-partition
`example_value`: `free_symbols()` rejected the zero-leaf empty-arguments
`TreeSpec`. The raw log is `/tmp/b70_qwen38_w8a8_vllm_compile_nograph.log`,
SHA256 `d7f81c777d6db090a642e11432507a7c9f791015cf0b3e04d11d5d9d93ae3323`.
The same assertion remains in current PyTorch main. The IGP=true arm failed at
the same boundary; its log SHA256 is
`b1e6cc3f35137a5278bea3ec41cd1783dea8f841104edc84736f5c0a96ada3ac`.

RESULT -> a temporary exact-Torch-commit-guarded diagnostic treated only a
zero-leaf `TreeSpec` as symbol-free. It cleared the first assertion, completed
the 27.60-second Dynamo transform, and compiled several regions. It then
proved the deeper incompatibility when AOTAutograd rejected that structural
object as a flat partition input: `all flat_args must be KNOWN_TYPES or opaque
types`. The diagnostic was removed rather than retained as a false fix. Its
log SHA256 is
`6cea642ed3083c1e2aca66184a64a8ce5eafae3b62f5ed9664a7b0d271b1e13b`.

RESULT -> every failed arm shut both workers down gracefully. Both card
probes and the exact-image compiled two-rank P2P-off collective post-check
passed after every arm; the GPUs remained healthy and free.

VERDICT -> reject compile-only Qwen3.8 on this Torch 2.13/vLLM bundle. This is
an upstream graph-partition metadata incompatibility, not an INT8 kernel,
capacity, graph-capture, or collective failure. Do not stack more TP2 retries
on this boundary. Keep the vLLM lane eager and move primary effort to the
exact SGLang/XPU-kernel build and narrow W8A8 port.

### 2026-08-26r - Exact current-source SGLang XPU image is rebuilt cleanly

CONFIG -> fixed Linux `7.1.0-070100-generic`; host-matched Compute Runtime
`26.22.38646.4`, Level Zero `1.28.2`, IGC `2.36.3+21719`, and GMM `22.10.0`;
exact SGLang commit `bede6bc37c5d9638099ebb948d93b9e2a7799f10` and tree
`938cf2b1b71bbc60e5d18d8388f1388ca0eff5a7`; exact sgl-kernel-xpu commit
`2d10888c069350ff20a192338d568dec945c9594` and tree
`3f975153d4d430535c759e57cb176e141a1b25c8`; torch `2.13.0+xpu` at
`cf30153c...`; Triton XPU `3.7.2`. All MoE, FMHA, and MLA kernel features were
built from the exact tracked source with eight jobs. No quarantined binary or
backend source was restored.

COMMAND ->

```bash
BUILD_JOBS=8 bash sglang/refresh/build.sh

docker run --rm --entrypoint bash \
  b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd \
  -lc 'cat /opt/b70-build-manifest/wheel-sha256.txt; python -m pip check'

./bin/gpu-run ./bin/xpu-health --img \
  b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd
./bin/gpu-run ./bin/xpu-collective-health --p2p 0 --img \
  b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd
```

RESULT -> the complete native kernel wheel built in 2 hours 43 minutes. Its
SHA256 is `f2dbd9a223056c530d0c8043d482d684e7dceb2f0778fb37ac351e6ea4736ffd`.
The SGLang wheel is `0.5.19.dev443+gbede6bc37c`, SHA256
`04909cc7d9241d3565f385e5e50f016344f6c9bb10f36005e9f8ec607315bbb4`.
The final image digest is
`8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd`.
`pip check` reports no broken requirements. XGrammar `0.2.1` and TVM FFI
`0.1.13.post3` import; the metadata-only `triton==3.7.2` alias leaves the
actual `triton-xpu==3.7.2` module and files intact. Both per-card probes and
the compiled ten-iteration TP=2 P2P-off collective probe passed.

RESULT -> the environment-gated `B70_XPU_W8A8=1` port catches only the exact
upstream XPU rejection for compressed-tensors W8A8 INT8. It retains the
upstream load/requantization logic, stores the transposed INT8 weight and
channel scales, releases the duplicate checkpoint weight, and applies dynamic
per-token symmetric INT8 through `torch._int_mm`. A card-0 numerical oracle
through the actual patched scheme returned
`W8A8_NUMERICAL_OK shape=(3, 32) max_error=0.0 freed_weight=True`.

VERDICT -> the refreshed SGLang and every ABI-specific native component now
have an exact source and image identity. The W8A8 port is a deliberately
narrow functional baseline, not yet a speed claim; optimize or replace its
generic `torch._int_mm` path only after full-model coherence and teardown.

### 2026-08-26s - Refreshed SGLang Qwen3.8 W8A8 TP2 baseline qualifies

CONFIG -> image digest `8678399d...`; local Qwen3.8-27B compressed-tensors
W8A8 GPTQ checkpoint; served ID `qwen3.8-27b-W8A8-gptq`; TP=2; BF16 residual
and KV state; 8K context; target only; eager attention and linear attention;
no graph, radix cache, overlap scheduler, or MTP; source-default c10d; oneCCL
SYCL kernels and P2P off. The matched transport profile uses OFI, Level Zero
v1, explicit two-card visibility, and pidfd IPC with oneCCL's observed drmfd
fallback.

COMMAND ->

```bash
./bin/gpu-run bash sglang/w8a8/serve_qwen38_w8a8.sh start

./bin/gpu-run bash sglang/perf_regime.sh \
  sglang_qwen38_w8a8_refresh 18080 qwen3.8-27b-W8A8-gptq \
  /models/qwen3.8-27b/w8a8-gptq qwen38-w8a8-refresh-tp2-eager

./bin/gpu-run python3 bin/serve-soak.py \
  --base-url http://localhost:18080/v1 \
  --model qwen3.8-27b-W8A8-gptq --concurrency 4 \
  --duration 300 --max-tokens 128 --timeout 300

./bin/gpu-run bash sglang/w8a8/serve_qwen38_w8a8.sh stop
./bin/gpu-run ./bin/xpu-health --img \
  b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd
./bin/gpu-run ./bin/xpu-collective-health --p2p 0 --img \
  b70-sglang-xpu@sha256:8678399dce536377f67760868b166744eb149ff9146e344476bb124e0c5933cd
```

RESULT -> the first launch with only `CCL_TOPO_P2P_ACCESS=0` loaded both
ranks, then failed the first embedding all-reduce with
`mem_to_ipc_handle: device_fd is invalid value`. The required xe rebind
recovered both cards; exact-image per-card and compiled collective checks
passed. Adding only the explicit retained multi-GPU transport profile cleared
that boundary. oneCCL reported pidfd unavailable and fell back to drmfd.

RESULT -> the successful arm reached health in 213 seconds, including about
90 seconds of first-run Triton KDA compilation. Each rank loaded 16.90 GB and
retained 14.99 GB free immediately after load; the final pool census reported
3.24 GB available and 371,584 total tokens. `/v1/models` returned the exact
served ID. Arithmetic returned exactly `42`; two independent temperature-zero
Rayleigh answers were byte-identical and coherent. Four simultaneous distinct
arithmetic requests all returned exact answers.

RESULT -> the retained matched 2048-input/128-output regime measured warm c1
at 3.76 per-stream decode tok/s, 3.43 aggregate output tok/s, and 2338.70 ms
mean TTFT. Warm c4 measured 2.73 per-stream decode tok/s, 8.74 aggregate
output tok/s, and 4100.56 ms mean TTFT. These figures are not directly matched
to the earlier vLLM p512/o512 result. The regime's deleted historical
`sglang/soak_probe.py` reference failed and is not counted as evidence.
The live mixed-prefill c4 soak then completed 32/32 requests over 300 seconds
with zero degeneracy and zero errors at 13.4 aggregate output tok/s.

RESULT -> graceful TP teardown completed in 18 seconds. Both exact-image
per-card probes and the compiled ten-iteration TP=2 P2P-off collective
post-check passed. The image digest observed by the live container was exactly
`8678399d...`; no scheduler exception occurred in the successful arm.

VERDICT -> the current-source SGLang generic INT8 path is now the qualified
coherent and stable denominator. Its 3.76 tok/s c1 decode is intentionally
unoptimized and must not be promoted to the shelf. Profile the generic dynamic
quantization/`torch._int_mm` path next, then port a fused dense INT8 kernel,
graph capture, and MTP as separate matched arms.

### 2026-08-26t - Native oneDNN W8A8 raises Qwen3.8 decode by 60.6 percent

CONFIG -> exact qualified SGLang parent image `8678399d...`; retained clean
`vllm-xpu-kernels` commit `2dd55f380df753a10a88fcd9e96192561066e713`
and tree `2416da2ad02ff58717edb864fa839442a15ca3d2`; only `_xpu_C` enabled;
TLA, basic, FA2, MoE, GDN, MQA logits, and XPU allocator extension families
disabled. The native route uses the retained SYCL per-token symmetric INT8
quantizer and oneDNN s8xs8 GEMM with FP32 activation and channel scales. The
generic `torch._int_mm` route remains the default. The qualified TP=2 arm sets
both asynchronous device dependencies:
`VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY=1` and
`VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER=1`. Every other serving, transport,
model, context, and eager-scheduler setting matches 2026-08-26s.

COMMAND ->

```bash
bash sglang/refresh/build_int8.sh
bash sglang/refresh/build_int8_runtime.sh

./bin/gpu-run --card 0 docker run --rm --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
  -e ONEAPI_DEVICE_SELECTOR=level_zero:0 -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY=1 \
  -e VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER=1 \
  -v "$PWD:/repo:ro" \
  b70-sglang-xpu-int8-runtime@sha256:fd9c806d517c073336d63a35b03eb552452c4c63d60b16bf55ec322de37bbc7d \
  bash -lc 'source /opt/intel/oneapi/setvars.sh --force >/dev/null && \
    python /repo/sglang/refresh/w8a8_native_oracle.py'

IMG=b70-sglang-xpu-int8-runtime@sha256:fd9c806d517c073336d63a35b03eb552452c4c63d60b16bf55ec322de37bbc7d \
  NATIVE=1 ONEDNN_INPUT_DEP=1 ONEDNN_BARRIER=1 \
  NAME=sglang_qwen38_w8a8_native PORT=18081 \
  ./bin/gpu-run bash sglang/w8a8/serve_qwen38_w8a8.sh start

./bin/gpu-run bash sglang/perf_regime.sh \
  sglang_qwen38_w8a8_native 18081 qwen3.8-27b-W8A8-gptq \
  /models/qwen3.8-27b/w8a8-gptq \
  qwen38-w8a8-native-bothdeps-tp2-eager

./bin/gpu-run python3 bin/serve-soak.py \
  --base-url http://localhost:18081/v1 \
  --model qwen3.8-27b-W8A8-gptq --concurrency 4 \
  --duration 300 --max-tokens 128 --timeout 300
```

RESULT -> the selective wheel built in 20 minutes 33 seconds and has SHA256
`06b949707d186bcba58fbc0f567f8db2cfc2836e0399ba9c01a365238e984cf3`.
The ABI image digest is `aeb939fa...`. A separate tracked-Python overlay keeps
route iteration out of that ABI layer; its final runtime image digest is
`fd9c806d...`, and the installed dispatcher SHA256 is
`4010ad0e011e8d1a13a43ed72fab9239db679e3eefdb828837226e2e8d34ac46`.
Both exact operator schemas registered and `pip check` reported no broken
requirements. No quarantined source or binary was restored.

RESULT -> the card-0 oracle matched its host reference exactly: quantized
bytes, FP32 scales, and BF16 GEMM output all had zero error. Sixteen repeated
quantizer and GEMM calls were bit-identical. With both dependencies enabled,
generic/native mean full-chain milliseconds were 0.4087/0.1867 at
M1-K5120-N17408 (2.19x), 0.4085/0.1068 at M1-K8704-N5120 (3.83x),
0.4233/0.2652 at M128-K5120-N17408 (1.60x), and 0.4114/0.1365 at
M128-K8704-N5120 (3.01x).

RESULT -> an initial extension image accidentally retained the old baseline
Python shim. Its logs explicitly reported `torch._int_mm`, so two attempted
TP arms were correctly rejected as generic controls and no native claim was
made from them. The runtime overlay fixed that boundary; every actual native
rank then logged `native per-token quant plus oneDNN GEMM`, and both dependency
markers fired on both ranks. Each rank loaded 16.91 GB and retained 14.99 GB
after load. `/v1/models` returned only `qwen3.8-27b-W8A8-gptq`.

RESULT -> thinking-mode Rayleigh traces were not byte-identical under either
the matched generic or native route, even with an explicit seed; one generic
response also exhausted its cap. This was a probe-design confound, not native
evidence. With Qwen thinking disabled, two native Rayleigh responses were
byte-identical and exactly matched the generic control SHA256
`a4dd7bbb7997619f22ca42e68d16a4ca4b10f56075d6c80af16b2ad28879966a`.
Four concurrent distinct arithmetic requests returned exactly 45, 78, 93,
and 189.

RESULT -> matched p2048/o128 warm c1 measured 6.04 per-stream decode tok/s,
5.25 aggregate output tok/s, and 2153.64 ms TTFT. Relative to the qualified
generic result, decode improved 60.6 percent, aggregate improved 53.1 percent,
and TTFT fell 7.9 percent. Warm c4 measured 3.60 per-stream decode tok/s,
11.34 aggregate output tok/s, and 3498.44 ms TTFT: improvements of 31.9,
29.7, and 14.7 percent respectively. The deleted historical soak probe again
did not run and is not counted.

RESULT -> the supported 300-second c4 soak completed 48/48 requests with zero
degeneracy and zero errors at 20.4 aggregate output tok/s, 52.2 percent above
the generic soak. Graceful teardown completed in 12 seconds. Exact-runtime
per-card checks and the compiled ten-iteration P2P-off collective passed both
before and after the successful native arm. The preserved startup log is
`/mnt/vm_8tb/b70/sglang_qwen38_w8a8_native_bothdeps.log`, SHA256
`aa64fd49350b0372ceb1357bb5970ede5de5b348262342ddf6c554e16d4e53e7`.

VERDICT -> the refreshed native dense W8A8 route is a coherent, stable, and
material full-model win over the generic denominator. Keep both device-side
dependencies for the qualified control. Test input and completion dependency
removal independently before graph or MTP work; do not promote to the shelf
until those arms and the remaining campaign are complete.

### 2026-08-26u - Native W8A8 dependency removal has no material speed win

CONFIG -> exact runtime image `fd9c806d...`, Qwen3.8 W8A8 TP=2, and every
setting from 2026-08-26t. Four one-variable dependency profiles were compared:
both input and completion enabled, input only, completion only, and neither.
Each arm used a fresh server. Both-off was tested only after each independent
removal cleared correctness and health.

COMMAND -> for each profile, set `ONEDNN_INPUT_DEP` and `ONEDNN_BARRIER` to
the selected 0/1 pair, then run:

```bash
IMG=b70-sglang-xpu-int8-runtime@sha256:fd9c806d517c073336d63a35b03eb552452c4c63d60b16bf55ec322de37bbc7d \
  NATIVE=1 ONEDNN_INPUT_DEP=<0-or-1> ONEDNN_BARRIER=<0-or-1> \
  NAME=<profile-name> PORT=18081 \
  ./bin/gpu-run bash sglang/w8a8/serve_qwen38_w8a8.sh start

./bin/gpu-run bash sglang/perf_regime.sh \
  <profile-name> 18081 qwen3.8-27b-W8A8-gptq \
  /models/qwen3.8-27b/w8a8-gptq <profile-label>
```

RESULT -> all four profiles returned the exact non-thinking Rayleigh control
SHA256 `a4dd7bbb...` twice and returned exact answers 45, 78, 93, and 189 under
four concurrent requests. Logs confirmed only the selected dependency marker
on each single-dependency arm and neither marker on both-off.

RESULT -> the matched warm results were:

```text
profile          c1 decode  c1 agg  c1 TTFT    c4 decode  c4 agg  c4 TTFT
both-on             6.04      5.25   2153.64       3.60    11.34   3498.44
input-only          6.04      5.25   2182.22       3.66    11.37   3549.23
completion-only     6.13      5.31   2137.97       3.63    11.41   3476.79
both-off            6.13      5.30   2177.33       3.66    11.44   3502.95
```

The largest apparent decode difference from both-on was 1.7 percent, while
TTFT moved in both directions. No removal profile showed a consistent material
advantage across c1, c4, aggregate throughput, and TTFT. The deleted historical
soak probe again did not run and is excluded.

RESULT -> the least-conservative both-off arm completed the same supported
300-second c4 soak at 48/48 requests, zero degeneracy, zero errors, and 20.3
aggregate output tok/s. This is effectively the same as both-on's 20.4 tok/s,
not a speed win. Every profile shut down gracefully. Exact-image per-card and
compiled P2P-off collective checks passed between arms and after both-off.
The input-only, completion-only, and both-off startup log SHA256 values are
`3f24cd30bfee29128cad59ef57fb58380ce0a047dd58502ebc2360c1ceb134d1`,
`7a43896b1ec227ddca3d06446b9022e20d86b3a6bba2f9ef09aa9b96f5f5d402`,
and `08b22002bf23b3370051715fa181d4699fadcc69fd8148d2e1d1727e29f51a64`.

VERDICT -> retain both asynchronous device dependencies. Their measured cost
is noise-scale, while they encode the intended cross-stream producer/consumer
ordering. `NATIVE=1` now defaults both dependency flags to 1; experiments may
still override either explicitly. Dependency tuning is exhausted for this
native dense route. Move to the next independent optimization lever.

### 2026-08-26v - Breakable XPU graph more than doubles Qwen3.8 c1 decode

CONFIG -> exact native INT8 ABI image `aeb939fa...` with a tracked Python-only
overlay at image digest
`f6aed4f45a922500ff286563e148bb5e13f05cd9c35d5177ba00204e18451770`;
dispatcher SHA256 `128c636fc78f411e34808d3428312ecfd5eb8b652394b83ba743e1bd62458bb2`.
Qwen3.8-27B compressed-tensors W8A8 GPTQ; TP=2; native per-token INT8 plus
oneDNN GEMM; both stream dependencies enabled; BF16 residual, KV, and output;
8K context; target only; no MTP, radix cache, overlap scheduler, or prefill
graph. The qualified graph arm captures decode batch sizes 1, 2, and 4 with
SGLang's segmented `breakable` backend. All TP all-reduces and all-gathers run
eagerly between graph segments through source-default c10d, SYCL kernels off,
P2P off, and the qualified pidfd-to-drmfd fallback transport profile.

COMMAND -> build the Python overlay, run exact-image health, launch and qualify
the graph arm, then run an identical-payload eager control:

```bash
TAG=b70-sglang-xpu-int8-runtime:20260826-breakable3 \
  bash sglang/refresh/build_int8_runtime.sh

./bin/gpu-run bash -lc '
  img=b70-sglang-xpu-int8-runtime@sha256:f6aed4f45a922500ff286563e148bb5e13f05cd9c35d5177ba00204e18451770
  ./bin/xpu-health --img "$img"
  ./bin/xpu-collective-health --p2p 0 --img "$img"
'

IMG=b70-sglang-xpu-int8-runtime@sha256:f6aed4f45a922500ff286563e148bb5e13f05cd9c35d5177ba00204e18451770 \
  NATIVE=1 DECODE_GRAPH=breakable \
  NAME=sglang_qwen38_w8a8_breakable PORT=18081 \
  LOG=/mnt/vm_8tb/b70/sglang_qwen38_w8a8_native_breakable3.log \
  ./bin/gpu-run bash sglang/w8a8/serve_qwen38_w8a8.sh start

./bin/gpu-run bash sglang/perf_regime.sh \
  sglang_qwen38_w8a8_breakable 18081 qwen3.8-27b-W8A8-gptq \
  /models/qwen3.8-27b/w8a8-gptq qwen38-w8a8-native-breakable-tp2

./bin/gpu-run python3 bin/serve-soak.py \
  --base-url http://localhost:18081/v1 \
  --model qwen3.8-27b-W8A8-gptq --concurrency 4 \
  --duration 300 --max-tokens 128 --timeout 300
```

RESULT -> a card-0 native-op XPUGraph oracle first proved that the exact
quantizer plus oneDNN GEMM chain is capturable: 16 replays were bit-identical,
with eager/graph medians of 0.076032/0.064169 ms for M1-K5120-N5120, a 15.6
percent graph reduction. The oracle SHA256 is
`4341f26dcc410158a93c71725c242c9c7bb02a8e3886c5ae3dfa18d1e905c8d1`.

RESULT -> FULL TP=2 capture was rejected after three bounded arms. With oneCCL
SYCL kernels off, capture rejected the first embedding all-reduce because
scheduler algorithms do not support SYCL graph recording. With SYCL kernels
on, both pidfd and drmfd failed before or during the first all-reduce with
`mem_to_ipc_handle: device_fd is invalid value`. The sockets exchange passed an
eager plus ten-iteration compiled two-rank collective preflight, but failed at
the same device-fd boundary specifically during FULL graph recording. Every
failed TP arm was followed by xe rebind recovery and exact-image per-card plus
compiled collective health. No P2P arm was run.

RESULT -> current SGLang already contains XPU-aware segmented graph primitives,
but two conservative XPU gates reject the backend. The narrow overlay permits
only explicitly selected XPU `breakable`, wraps the TP all-reduce and all-gather
boundaries with `eager_on_graph`, and adds buffer allocation, row counting,
slicing, and copying for `LogitsProcessorOutput`. A direct dataclass buffer
oracle passed. No retained runtime binary, push-AR graph mode, or old backend
patch was restored. One first attempt was correctly rejected because the
platform gate silently disabled graph; a second reached capture and exposed
the missing output buffer support. Neither produced a speed claim.

RESULT -> the final arm loaded 16.91 GB per rank and retained 14.99 GB after
load. It explicitly resolved decode backend `breakable`, captured bs 4, 2, and
1 in 9.65 seconds, and reported nonzero decode graph startup time. `/v1/models`
returned only `qwen3.8-27b-W8A8-gptq`. The startup log SHA256 is
`9d198e8aff9f8233e4d57a3e50ddafae70d2595b2a1f23c44815112a901a29c9`.

RESULT -> two non-thinking greedy Rayleigh responses were byte-identical at
SHA256 `e6e39dc2bf6864a1fcb7c78e89dcb2b3defbadb0507f2ae17a4673269b0a9ece`.
Four simultaneous arithmetic requests returned exactly 45, 78, 93, and 189.
A fresh eager server from the same final image returned the same Rayleigh text
and SHA256 twice under the identical payload, closing the graph/eager identity
comparison without relying on the earlier incompletely recorded payload.

RESULT -> matched p2048/o128 warm c1 measured 13.65 per-stream decode tok/s,
10.02 aggregate output tok/s, and 2167.92 ms TTFT. Against the qualified native
eager control at 6.04, 5.25, and 2153.64, decode improved 126.0 percent and
aggregate output improved 90.9 percent while TTFT changed by only +0.7 percent.
Warm c4 measured 5.62 per-stream decode tok/s, 16.08 aggregate output tok/s,
and 3268.35 ms TTFT. Against eager 3.60, 11.34, and 3498.44, those are +56.1
percent decode, +41.8 percent aggregate, and 6.6 percent lower TTFT. The deleted
historical soak helper failed as expected and is excluded.

RESULT -> the supported 300-second c4 soak completed 92/92 requests with zero
degeneracy and zero errors at 37.6 aggregate output tok/s. The eager control was
20.4 tok/s, so the graph arm improved soak throughput by 84.3 percent. Live
scheduler logs repeatedly reported `cuda graph: True` throughout the soak and
did not show the historical replay-degradation signature. Both graph and eager
control servers shut down gracefully. Exact-final-image per-card checks and the
compiled ten-iteration P2P-off collective passed before capture, after the graph
arm, and after the eager identity control.

VERDICT -> qualify the environment-gated XPU breakable decode graph as the new
Qwen3.8 native W8A8 performance control. This directly answers why the earlier
c4 soak looked strong while single-stream decode remained low: batching hid a
per-token Python/launch and submission tax, while segmented capture removes
most of that tax and leaves only the rank-coupled collectives eager. Keep FULL
capture rejected on this stack because oneCCL graph IPC remains broken. The
breakable route is qualified for bs <= 4 only and is not yet a shelf promotion;
MTP and the remaining campaign still require separate matched qualification.

### 2026-08-26w - Qwen3.8 NEXTN s1 is coherent but remains slower than target-only graph

CONFIG -> exact refreshed SGLang and native INT8 stack from 2026-08-26v, with
the final Python-only overlay image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`.
Dispatcher SHA256 is
`e255ef23b507767bf4e26f607e253f0894f3f80c1be2d2cfbd72e4e896354b76`.
Qwen3.8-27B compressed-tensors W8A8 GPTQ target plus the grafted official BF16
`model-mtp.safetensors`; TP=2; NEXTN; topk=1; explicit unquantized speculative
draft model; native oneDNN INT8 target linears; both dependency controls on;
P2P off; source-default c10d; pidfd falling back to drmfd; radix and overlap
disabled; 8K context; maximum batch four. The primary arm uses one speculative
step and two verify tokens. Only greedy serving is in qualification scope
because current SGLang deliberately sends XPU speculative verification through
the greedy branch even for sampling requests.

COMMAND -> launch eager NEXTN, close greedy identity/concurrency and matched
p2048/o128 performance, then repeat with breakable target-verify and
draft-extend capture. Run an s3/draft4 arm only as a coherence probe:

```bash
IMG=b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78 \
  NATIVE=1 MTP=1 SPEC_STEPS=1 SPEC_DRAFT=2 \
  SERVED=qwen3.8-27b-W8A8-gptq-nextn DECODE_GRAPH=breakable \
  NAME=sglang_qwen38_w8a8_mtp_breakable PORT=18082 \
  bash sglang/w8a8/serve_qwen38_w8a8.sh start

./bin/gpu-run bash sglang/perf_regime.sh \
  sglang_qwen38_w8a8_mtp_breakable 18082 \
  qwen3.8-27b-W8A8-gptq-nextn /models/qwen3.8-27b/w8a8-gptq \
  qwen38-w8a8-native-mtp-s1-breakable-tp2
```

RESULT -> current upstream already contains the XPU Triton tree builder, XPU
greedy verifier, XPU cache-location assignment, native Qwen3.5 MTP model, and
XPU draft graph runners. It still required four deliberate XPU source ports:
speculative Mamba scratch allocations and MTP weight-sharing synchronization
used CUDA explicitly; the XPU GDN wrapper omitted the kernel's
`stride_h0_source`; two portable Triton state-commit wrappers rejected non-CUDA
tensors; and the new fused multi-conv commit packed high XPU pointers into a
signed int64 tensor and overflowed. The last route is disabled only under the
XPU MTP gate in favor of the per-conv Triton loop. Exact source-pattern checks,
a GDN launch-contract oracle, and state-allocation/commit code oracles passed.

RESULT -> four bounded TP2 bring-up failures exposed those defects in order.
No failed arm produced output or a speed claim. Each crashed TP2 arm was
stopped, followed by xe rebind recovery and exact-image per-card plus compiled
P2P-off collective health before retry. The final eager s1 arm loaded the
target at 16.91 GB/rank and the BF16 MTP worker at an additional 2.64 GB/rank.
The two 96-token non-thinking greedy Rayleigh runs were byte-identical at
SHA256 `29d0e3f47e7937187acfaf6cdd0a8f67aed17f7781d530bc022b0f38f62993cb`.
A fresh target-only server on the same overlay generation returned that same
hash twice. Four simultaneous arithmetic requests returned exactly 45, 78,
93, and 189. The eager startup log SHA256 is
`01ab2f1266ec96fa03b80175607a6e86bfdfcb934c93e6317f710d7de8e28a8b`.

RESULT -> eager s1 p2048/o128 measured c1 6.29 decode tok/s, 5.55 aggregate
output tok/s, and 2228.19 ms TTFT. C4 measured 3.44 decode, 10.78 aggregate,
and 3809.22 ms TTFT. Against the qualified target-only eager control
6.04/5.25/2153.64 at c1 and 3.60/11.34/3498.44 at c4, s1 gained only 4.1
percent c1 decode while losing 4.4 percent c4 decode and worsening TTFT.
Observed accepted length was about 1.5 on the short prompt and usually 1.2-1.4
in the long-prompt regime.

RESULT -> the first breakable MTP capture was correctly rejected by the
upstream assertion that XPU full attention does not support speculative graph
metadata. The narrow port preserved that boundary by making speculative XPU
full-attention metadata and forward calls eager graph breaks, while leaving GDN
and surrounding dense INT8 computation capturable. Target verify then captured
bs 4, 2, and 1 in 7.06 seconds and draft extend in 1.30 seconds. First replay
exposed that the prior `LogitsProcessorOutput` adapter allocated one row per
request instead of one per verify token. A direct bs4/draft2 oracle then proved
all eight logits and hidden-state rows survive allocation, copy, and slice.
The corrected arm returned the exact target/eager-MTP hash twice and passed the
same c4 arithmetic test. Its startup log SHA256 is
`a6008b5365b9bbf160092f7ca6ef5881fad95e7ea6e80d368ba6ae9c7c5df341`.

RESULT -> breakable s1 p2048/o128 measured c1 12.01 decode tok/s, 9.23
aggregate, and 2252.33 ms TTFT; c4 measured 5.23 decode, 14.95 aggregate, and
3468.07 ms TTFT. This nearly doubled eager MTP, but remained below the qualified
target-only breakable control: -12.0 percent c1 decode versus 13.65, -6.9
percent c4 decode versus 5.62, and -7.0 percent c4 aggregate versus 16.08. The
removed historical `sglang/soak_probe.py` failed as expected and is excluded;
no long soak was justified for an arm already slower than the target-only
control.

RESULT -> s3/draft4 captured target verify, draft decode, and draft extend, but
failed exact greedy coherence. Three repeats deterministically returned SHA256
`4ff762ab129e74142980c1aa99a82f81bc7cec32133b8c7d01a51207c9701a24`
instead of the exact target/s1 hash. Accepted length was only about 1.57 of four
verify tokens. The arm was rejected without benchmarking. Startup log SHA256 is
`0a992eb4940e6e1c7120be2bb13de703f58d6b713624680dcd28b686cec42e35`.

RESULT -> all successful and rejected arms tore down. Final exact-image
per-card health and the compiled ten-iteration P2P-off collective passed.

VERDICT -> retain NEXTN s1 eager and breakable as coherent greedy research
controls, not shelf or default serving configurations. S1 graph is materially
faster than eager MTP but still loses to target-only breakable graph because
acceptance is too low to repay remaining eager full-attention and state-commit
work. Reject s3 because exact target coherence fails before performance is
considered. Do not claim sampled-serving correctness on the current XPU greedy
verification fallback. Return to MTP only if a target-exact multi-step GDN
state oracle or materially better draft acceptance changes this decision.

### 2026-08-26x - Refreshed Ornith W8A8 graph gains 4.15x; MTP is not target-exact

CONFIG -> exact refreshed SGLang runtime
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`,
SGLang `bede6bc`, sgl-kernel `2d10888`, Torch 2.13.0+xpu, Triton XPU
3.7.2, Compute Runtime 26.22, and vllm-xpu-kernels `2dd55f3`. The model was
the local Ornith-1.5-35B-A3B Quark-compatible RTN W8A8 checkpoint with
dynamic per-token INT8 activations and the BF16 Shisa MTP sidecar. All serves
used TP=2, P2P disabled, 8192 context, overlap/radix off, and capture batch
sizes 1, 2, and 4. Routed experts stayed on Triton W8A8. The control kept
eligible dense/shared projections in the proven load-time BF16 dequant route.

COMMAND -> exact-image per-card and compiled ten-iteration P2P-off collective
health bracketed each risky phase. Serves used `./bin/gpu-run` with
`sglang/w8a8/serve_ornith15_w8a8_refresh.sh`. The matched client protocol used
the retained `phase_bench.py` source from the pre-cleanup commit through
`git show`, one same-shape warmup, three unique entropy-prefix requests,
approximately 4200 actual prompt tokens, 128 forced output tokens, and true
client post-first timing. Separate c4 batches used the same prompt generator,
SSE timing, and four simultaneous forced-length streams.

RESULT -> current SGLang moved both the INT8 activation kernel and fused-MoE
module and renamed Quark's online quantized-layer bookkeeping. Two bounded
constructor-only failures exposed those API seams before weight load. The
ported loader now supports both old and current module paths and bookkeeping
names. Each failed TP2 attempt was torn down, followed by xe rebind recovery,
exact-image per-card health, and compiled P2P-off collective health.

RESULT -> eager no-MTP loaded 17.91 GB/rank and selected Triton INT8 W8A8 for
all 40 routed-expert layers. Two greedy Rayleigh responses were byte-identical
at SHA256 `1deaa216c21e626b549b9a6b4d8a05ef113761275a4196bfb2247b5bee3db3d9`;
four simultaneous arithmetic canaries returned 42, 78, 93, and 189. The
matched c1 median was 6.0923 post-first tok/s with 2.5948 s TTFT and 4219
actual prompt tokens. Runtime log SHA256 is
`07a4092b1087be6d45feb6416ebc000666103a7069a389c9b152199d33e19eca`.

RESULT -> target-only breakable capture kept TP collectives eager and captured
bs 4, 2, and 1 in 10.7 seconds. It returned the exact eager hash twice and
passed the same c4 arithmetic gate. The matched c1 median was 25.2616
post-first tok/s with 2.5644 s TTFT and 4218 prompt tokens: 4.15x the eager
decode rate with unchanged prefill latency. Three timed c4 batches returned
128 tokens on all 12 streams. Median per-stream post-first decode was 19.74
tok/s, aggregate post-first throughput was 29.42 tok/s, aggregate including
TTFT was 24.66 tok/s, and median TTFT was 6.853 s. This distinguishes c4
per-request latency from server aggregate capacity. Runtime log SHA256 is
`65b5ee45343e969b615c8b5fda3c39c52b1ea8ef91b1b8c6ee420b06e51f4555`.

RESULT -> Shisa NEXTN s1 loaded another 1.75 GB/rank and captured target verify
in 9.63 seconds plus draft extend in 2.33 seconds. It passed the four arithmetic
canaries, but its two deterministic Rayleigh responses had SHA256
`ecc5e331daf7886fe3831eef5a7de6722ef111f2edda75563dc51ff1dbe8c4dd`,
not the exact target hash. It was rejected before speed measurement. Runtime
log SHA256 is
`f93b416dd559d9af188a156c55df1fd3d8d471cc5133d7ac4ee022cc44c10bf1`.

RESULT -> an opt-in native dense route kept Quark dense/shared weights INT8 and
used the already-qualified oneDNN per-token quant plus W8A8 GEMM. It was
deterministic and passed c4 arithmetic, but changed the expected W8A16-fallback
hash and measured only 24.5356 c1 post-first tok/s with 2.6074 s TTFT: 2.85
percent slower than the matched breakable control. It remains default-off.
Runtime log SHA256 is
`04e48548042f82d2823a3c0d756ef228104105a7dd120a8631b98a1c2ffa3c39`.

RESULT -> full target capture with P2P still disabled failed at its first
in-graph embedding all-reduce because oneCCL could not export the allocation:
`mem_to_ipc_handle: EXCEPTION: device_fd is invalid value`. No endpoint or
speed result was produced. Runtime log SHA256 is
`efc290c67354cc7a0c9b71fdafa7b8e171c7488acf5166820943d528f57286a1`.
The required rebind recovery preserved the boot ID. Final exact-image per-card
health and compiled ten-iteration P2P-off collective health passed, and no GPU
server remains running.

VERDICT -> retain target-only breakable graph with load-time BF16 dense dequant
as the refreshed Ornith W8A8 research winner. It removes the dominant host
launch floor without putting oneCCL inside the graph. Do not promote Shisa MTP
until speculative greedy output is target-exact. Do not use the native dense
route by default because it is slower, and do not retry full TP2 capture until
the oneCCL graph IPC export has an isolated passing oracle. The missing B70
`E=256,N=256` Triton MoE tuning files remain a smaller post-graph opportunity.

### 2026-08-26y - Pi xhigh works; Ornith long-agent graph replay is not stable

CONFIG -> exact refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
and the target-only Ornith W8A8 route from 2026-08-26x. All model traffic was
TP=2, P2P off, source-default c10d, one request maximum, 65,536 context,
qwen3 reasoning parser, and qwen3_coder tool parser. Harbor 0.22.0 ran the
official `terminal-bench/terminal-bench@3.0.0` `bun-sourcemap-leak` task with
Pi 0.84.3. The dataset and job outputs remained outside git under
`/mnt/vm_8tb/b70/evals`. Pi `xhigh` was mapped to Qwen chat-template native
thinking; no unsupported OpenAI `reasoning_effort` field was sent.

COMMAND -> install-only and payload oracles first proved the custom adapter,
then run the official task through the exact local endpoint. The final safe
arm used eager decode and the task-agnostic concise prompt:

```bash
PYTHONPATH=/mnt/vm_8tb/github/b70_ai_things \
OPENAI_BASE_URL=http://192.168.10.5:18080/v1 OPENAI_API_KEY=EMPTY \
harbor run -d terminal-bench/terminal-bench@3.0.0 \
  -i terminal-bench/bun-sourcemap-leak -l 1 \
  -a evals.terminalbench.harbor_pi:SglangReasoningPi \
  -m openai/ornith-1.5-35b-a3b-W8A8-rtn-shisa-target-eager \
  --ak model_api=openai-completions --ak thinking=xhigh \
  --ak version=0.84.3 --ak context_window=65536 --ak max_tokens=16384 \
  --ak prompt_template_path=/mnt/vm_8tb/github/b70_ai_things/evals/terminalbench/pi_concise_prompt.j2 \
  --allow-agent-host 192.168.10.5 -n 1 -k 1 --yes
```

RESULT -> Harbor's stock custom-endpoint model description classified the
local model as non-reasoning and silently reduced `--thinking xhigh` to off.
The retained adapter fixes that metadata. A Pi-AI mock payload contained
`chat_template_kwargs.enable_thinking=true` and `preserve_thinking=true`, kept
the system role, and omitted `reasoning_effort`. The pinned Pi install smoke
passed. A direct live request then returned `finish_reason=tool_calls`, bash
arguments `{"command":"pwd"}`, separate reasoning content, and exact model
identity.

RESULT -> the first official arm lacked a tool parser. Ornith emitted the
correct Qwen XML bash call as plain text, Pi could not execute it, and the
official verifier scored 0.0. Adding qwen3_coder converted the same format to
structured tool calls. The uncapped diagnostic completed 11/11 tool turns and
reported 73,830 input plus 8,788 output tokens before cancellation; its largest
completed turn was 5,715 output tokens. It was not scored.

RESULT -> setting only `SGLANG_MAX_THINK_TOKENS=4096` did not enforce a cap
because the launcher still selected grammar backend `none`. Source inspection
showed that SGLang applies the token filter only with XGrammar strict thinking.
The launcher now couples a nonempty `THINKCAP` to
`--grammar-backend xgrammar --enable-strict-thinking`; empty `THINKCAP` keeps
the prior no-grammar route. The official strict arm bounded its two long
completed turns at 4,223 and 4,278 total output tokens including close and tool
overhead, proving the private-thinking cap on real xhigh traffic.

RESULT -> the strict 4,096 breakable arm made nine structured tool calls, then
the TP scheduler aborted during `torch.xpu.graphs.replay` with the Intel runtime
assertion `linear_stream.h:90` at about 17,664 live tokens. Pi saw the endpoint
close and the official verifier returned 0.0 after 19m55s; this is a runtime
failure, not a model-quality score. The server was not OOM-killed. Xe rebind,
exact-image per-card health, and the compiled ten-iteration P2P-off collective
all passed afterward.

RESULT -> a matched strict 2,048 arm proved that the setting is not a hard
completion ceiling. One turn returned at 2,135 total output tokens, but a later
turn continued its plan as visible text after strict thinking closed. It was
cancelled at a 16K live sequence to avoid the known replay boundary. SGLang did
not stop the disconnected request; it continued server-side and later hit the
same breakable replay assertion. A second xe rebind plus exact-image per-card
and compiled collective checks passed. The concise breakable arm began while
that disconnected request was still running and received no model tokens, so
it is invalid rather than a scored result.

RESULT -> the final eager arm removed the failing replay path and remained
stable. Its first three tool turns used 65, 61, and 117 output tokens. A long
turn returned a structured bash call at 2,198 output tokens, followed by a
141-token tool turn. Eager scheduler decode stayed near 6 tok/s at these short
contexts. A subsequent turn again spilled visible planning after the soft cap;
the feasibility arm was cancelled after 13m45s with 18,027 input and 2,582
completed output tokens because the remaining official 30-minute window could
not cover implementation and verification. It is unscored. The eager server
shut down gracefully, and final exact-image per-card plus compiled TP2
collective health passed. No GPU server remains running.

VERDICT -> the local Pi xhigh and structured-tool integration is valid, but
Ornith is not qualified for TerminalBench 3.0.0. Disqualify breakable graph for
long agent trajectories until an isolated replay/command-stream oracle passes;
the short-context 25.2616 tok/s winner from 2026-08-26x remains valid only in
its measured regime. Eager avoids the crash but is too slow when Ornith emits
multi-thousand-token plans. `SGLANG_MAX_THINK_TOKENS` alone is not total-output
control. The next agent-quality work should test a real per-request completion
policy or a more tool-eager agent/model strategy on eager decode before any
full dataset campaign. Do not spend time on a c=4 soak here: the observed low
rate is a c=1 long-trajectory/eager problem, not a concurrency qualification.

### 2026-08-27a - Qwen3.8 NVFP4 ported to vLLM 0.28; TP1 graph reaches 21.81 tok/s

CONFIG -> official vLLM XPU image
`vllm/vllm-openai-xpu@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3`,
vLLM 0.28.0, Torch 2.13.0+xpu, vllm-xpu-kernels 0.1.13.2, Triton XPU
3.7.2, Compute Runtime 26.27, Level Zero 1.32, and IGC 2.38.2. The model was
RadixArk Qwen3.8-27B NVFP4 at pinned revision
`319f741cce68d7914884900c138a1fbb70a42f30`. The source port used current
`vllm-xpu-kernels` commit
`a397c58eb7781e6fe0d6b3fb7c25d21b5f658784`. All TP2 work kept direct P2P
disabled. No result was promoted to the live shelf.

COMMAND -> inspect the exact release image and checkpoint identity, run a
stock v0.28 Qwen3.8 W8A8 smoke, and run the unmodified release image against
the NVFP4 checkpoint with `vllm/nvfp4/serve_qwen38_v028.sh`. Apply
`kernels/nvfp4_v028_integration.patch` to a dedicated clean source tree and
build with `vllm/nvfp4/build_nvfp4_v028.sh`. The build image was
`b70-sglang-xpu:20260826-bede6bc-2d10888-torch213-umd2622`; SHA256 comparison
first proved its Torch shared libraries byte-identical to the release image.
The tracked build enables XPU-specific and GDN kernels only.

RESULT -> stock v0.28 served the compressed-tensors W8A8 checkpoint coherently
at TP2. The stock NVFP4 control rejected the model before weight load with
`modelopt_mixed quantization is currently not supported in xpu`. vLLM's dSpark
path is CUDA/ROCm-only in this release, so it is not an XPU control.

RESULT -> the first source build exposed an upstream option-forwarding gap:
requesting MHC off still produced an undefined MHC symbol. The integration
patch now forwards the MHC option. An extension-only corrected build loaded
the NVFP4 model but full generation then failed because replacing the release
extension also removed its `gdn_attention` registration. The final build
therefore includes an ABI-matched GDN sidecar. Its artifacts are
`_xpu_C.abi3.so` SHA256
`96e33b4e66f4eba6a2108c5a4f3aef5fba505f3696ba876e60b6ddeb08a87549`
and `libgdn_attn_kernels_xe_2.so` SHA256
`323547ed36f4821ccba6fbbc75ced8fd6e9837e268891d6488d62825002279a8`.

COMMAND -> run `vllm/nvfp4/oracle_v028.py` on card 0 against the real layer-0
gate projection, N=17408 and K=5120, under the exact release runtime. Then run
`vllm/nvfp4/serve_qwen38_v028_nvfp4.sh smoke` at TP2. Exact-image per-card
health and the compiled ten-iteration TP2 collective probe bracketed risky
work.

RESULT -> for M=1, folded-BF16 scales returned cosine 0.99999410, relative L2
0.00365781, and max absolute error 0.015625 against explicit dequantization.
Native E4M3 scales returned cosine 0.99999720, relative L2 0.00240853, and max
absolute error 0.015625. For M=8, folded and native relative L2 were 0.00363943
and 0.00240471. Repeated folded output was exact. The TP2 serve selected the
custom NVFP4 W4A16 kernel, loaded about 10.69 GiB/rank, exposed exact ID
`qwen3.8-27b-NVFP4-radixark-vllm028-onednn`, produced coherent Paris/Berlin
text, and shut down gracefully. Per-card and compiled P2P-off collective
health passed before and after.

COMMAND -> with the same TP2 eager descriptor, run eight 512-input/512-output
requests at c1 and 32 at c4 through `bin/35_sweep_bench.sh` under one
`bin/gpu-run` lease. Result CSV:
`/mnt/vm_8tb/b70/results/sweep_qwen3.8-27b-NVFP4-radixark-vllm028-onednn-tp2_20260827_083803.csv`.

RESULT -> eager TP2 measured c1 4.31 aggregate output tok/s, 229.96 ms mean
TPOT, 4.35 per-stream tok/s, and 1293.04 ms mean TTFT. At c4 it measured 16.32
aggregate output tok/s, 238.81 ms mean TPOT, 4.19 per-stream tok/s, and 3462.77
ms mean TTFT. All requests completed, the server stopped gracefully, and final
exact-image per-card plus compiled TP2 collective health passed. The outer
wrapper returned 1 only because a post-stop `35_sweep_bench.sh` health check
reported `server not healthy`; the completed CSV, teardown, and separate
post-health evidence remain valid.

COMMAND -> on card 0, set `VLLM_USE_BREAKABLE_CUDAGRAPH=1`,
`VLLM_USE_AOT_COMPILE=0`, TP=1, PIECEWISE, capture size 1, no compile sizes,
Inductor graph partition disabled, one sequence maximum, and 4096 context. Run a
coherence probe plus two identical 64-token deterministic requests, then a
separate eight-request 512-input/512-output c1 benchmark. Result CSV:
`/mnt/vm_8tb/b70/results/sweep_qwen3.8-27b-NVFP4-radixark-vllm028-onednn-tp1-graph_20260827_091920.csv`.

RESULT -> breakable mode reported compilation mode NONE, kept Flash attention
and Triton GDN outside graph segments, and captured size 1. The two replay
texts were byte-identical with SHA256
`6a19e3fd220b4de31e008acd8c95ac2ce72ea3ce07d34ba590327b5755894f7a`.
The separate TP1 graph benchmark measured 21.81 aggregate output tok/s, 45.20
ms mean TPOT, 22.12 per-stream tok/s, and 376.55 ms mean TTFT. Runtime log
SHA256 is
`5153f5afdc075b6fb77038d2a6ea743f7afc898690b497c06ae30cc9a3363e2e`.
Both TP1 runs stopped gracefully and card-0 health passed before and after.

VERDICT -> the v0.28 XPU NVFP4 kernel port is numerically and functionally
valid, but eager TP2 is far below the 40 tok/s objective and is not shelf
quality. The TP1 graph result is not a matched speedup comparison to TP2, but
it proves substantial capture leverage on one card. Do not attempt a full TP2
breakable model capture yet: v0.28 documents XPU graph as single-GPU and its
breakable path does not eject oneCCL collectives. First qualify an exact-image
two-rank capture/replay oracle or implement a stable out-buffer eager
collective boundary with P2P disabled. Keep FULL, MTP, and direct P2P out of
the first TP2 graph arm.

### 2026-08-27b - Qwen3.8 NVFP4 reaches 78.07 tok/s at TP1 c4

CONFIG -> exact vLLM 0.28 XPU and Qwen3.8 NVFP4 identities from 2026-08-27a.
The graph work used PIECEWISE breakable capture with AOT compilation disabled,
P2P disabled, default Flash attention, Triton GDN, no MTP, and native E4M3
NVFP4 scales through M=8. The TP2 oracle and model arms also disabled oneCCL
SYCL kernels. Every GPU command used `bin/gpu-run` and exact-image health
bracketing. No result was added to the live shelf.

COMMAND -> add an opt-in Python boundary to the v0.28 compatibility layer.
`GroupCoordinator.all_reduce` preserves the XPU communicator's out-of-place
semantics by cloning inside the preceding graph segment, then a function
decorated with `eager_break_during_capture` performs synchronous in-place
oneCCL on that stable output buffer. Unexpected in-capture all-gather fails
closed. Run `vllm/nvfp4/breakable_allreduce_oracle_v028.py` through exact-image
torchrun at two ranks on a BF16 `[1,5120]` tensor.

RESULT -> the first standalone oracle attempt stopped before graph capture
because vLLM's CUDA-to-XPU graph wrapper was not installed. The second stopped
before graph capture because TP group creation lacked a current `VllmConfig`.
Both setup-only failures were followed by healthy per-card and compiled TP2
collective probes. The corrected oracle used the canonical XPU wrapper and
real vLLM TP group. It captured two graph segments around one eager break and
passed 16 synchronized replays exactly. Each rank kept one fixed output
address, inputs were unchanged, the helper ran outside graph capture, and its
call count was exactly 17: capture plus 16 replays. Exact-image per-card and
compiled P2P-off collective health passed afterward.

COMMAND -> run the first full-model TP2 graph smoke with capture size 1,
`BREAKABLE_AR=1`, one sequence maximum, 4096 context, and 0.85 memory
utilization. Then run eight forced 512-input/512-output c1 requests with the
same descriptor. Result CSV:
`/mnt/vm_8tb/b70/results/sweep_qwen3.8-27b-NVFP4-radixark-vllm028-onednn-tp2-graph_20260827_094901.csv`.

RESULT -> TP2 captured in 3 seconds using 0.67 GiB/rank, exposed exact model
identity, produced the expected Paris/Berlin completion, and stopped
gracefully. No all-gather guard fired. The matched c1 mean was 5.06 aggregate
output tok/s, 195.90 ms mean TPOT, 5.10 per-stream tok/s, and 1134.50 ms mean
TTFT. This is 17.4 percent above the matched eager TP2 4.31 tok/s result, but
far below the objective. More importantly, live per-request decode declined
from about 12 to about 3.1 tok/s across the serial sample while remaining
coherent and responsive. Runtime log SHA256 is
`ea03c13e2b0bcf65855b23fe22217d3c64fcdf28611f17a76d90e25b6c42fb17`.
Per-card and compiled collective health passed after graceful teardown.

COMMAND -> remove TP collectives by running the full model on card 0 with
TP=1, breakable PIECEWISE capture sizes 1, 2, and 4, four sequences maximum,
4096 context, and 0.92 memory utilization. Run 32 forced
512-input/512-output requests at c4. Result CSV:
`/mnt/vm_8tb/b70/results/sweep_qwen3.8-27b-NVFP4-radixark-vllm028-onednn-tp1-graph_20260827_100637.csv`.

RESULT -> the TP1 c4 route exposed exact model identity, passed its coherent
generation probe, completed all 32 requests, and measured 78.07 aggregate
output tok/s, 49.18 ms mean TPOT, 20.33 per-stream tok/s, and 1103.08 ms mean
TTFT. Runtime log SHA256 is
`b1a24707a38eefdce242f610ef71bbd8ea626fd60e76b2a2dd7804fa8c8419ff`.
The server stopped gracefully and card-0 health passed.

COMMAND -> compare four serial deterministic factual completions against the
same prompts issued concurrently at c4. An initial arithmetic gate was invalid
because its regex was over-escaped and two prompts elicited poor continuations.
Repeat with direct text comparison at 64, 24, and 8 forced tokens.

RESULT -> at 64 tokens, three of four concurrent texts were byte-identical to
their serial baselines; the fourth diverged after the shared coherent opening
`four sides`. At 24 tokens, three of four were again exact; the Jupiter case
shared the correct `Jupiter. It is a gas giant` prefix and then selected two
different factual continuations. At 8 tokens all four concurrent results were
byte-identical to their serial baselines with coherent France, gold, sequence,
and Jupiter text. Canary runtime log SHA256 is
`452809c6036c6905844937dbfe5f7a66bd04466435360b5ee8654a1923e74cbd`.
Every gate shut down gracefully and card-0 health passed.

VERDICT -> the qualified Qwen3.8 NVFP4 capacity candidate is TP1 breakable
graph at c4, reproduced by
`vllm/nvfp4/serve_qwen38_v028_nvfp4_graph.sh`. Its measured 78.07 aggregate
tok/s exceeds the 40 tok/s objective with exact identity, coherent generation,
32 completed benchmark requests, an exact four-stream 8-token canary,
graceful teardown, and post-health. Do not claim long-horizon batch-exact greedy
equivalence: measured 24/64-token continuations can diverge after coherent
common prefixes. Reject TP2 eager-all-reduce graph as a performance route;
although functionally correct and isolated-oracle clean, its many host
collective boundaries retain cumulative slowdown.

### 2026-08-27c - Ornith W8A8 graph reclaim sustains 87.80 tok/s at c4

CONFIG -> exact refreshed SGLang runtime
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`,
kernel 7.1.0-070100, SGLang `bede6bc`, sgl-kernel `2d10888`, Torch
2.13.0+xpu, Triton XPU 3.7.2, Compute Runtime 26.22, and
vllm-xpu-kernels `2dd55f3`. The model was the local
Ornith-1.5-35B-A3B Quark-compatible RTN W8A8 checkpoint. All accepted arms
used TP=2, P2P disabled, source-default eager oneCCL collectives, MTP off,
load-time BF16 dense/shared dequant, Triton routed-expert W8A8, breakable
decode graph sizes 1, 2, and 4, maximum concurrency 4, 8192 context, and no
radix, overlap, tool parser, or strict-thinking grammar. The candidate mounted
tracked overlay `sglang/refresh/b70_xpu_w8a8.py` SHA256
`083aea56045cc91dc66dd01e561e3a3876ce86ab9d5dfba76277e14534e41f31`
over the image copy and set graph reclaim to 500 replays per graph.

COMMAND -> first screen configurable graph sizes 1, 2, 4, 8, and 16 with the
current random-serving benchmark, then use
`sglang/w8a8/bench_forced_concurrent.py` for exact served-ID validation,
two repeated greedy Rayleigh responses, concurrent arithmetic/factual
canaries, exact forced completion-token accounting, and true client
post-first timing. The matched historical shape used 515 prompt tokens, 512
forced output tokens, and c4. Every risky TP2 run was fully enclosed by
`bin/gpu-run`, per-card health, and the exact-image compiled ten-iteration
P2P-off collective probe.

RESULT -> the variable-length screen measured 65.15 aggregate output tok/s at
c8 and 71.36 at c16, but it is not an accepted speed claim because prompts
averaged only about 1000 tokens and completions stopped at variable lengths.
The strict 4172-prompt/128-output c8 arm passed all eight canaries, returned
exactly 128 tokens on every stream, and produced byte-identical repeated greedy
output at SHA256
`c633eb39a51efdcb78f62e9db561cc4d157b64029369ee1097bc15beb0128c65`.
Its three measured aggregate post-first rates were 39.5081, 39.2342, and
38.9456 tok/s, so long prefill serialization does not meet the 65 tok/s
capacity objective.

RESULT -> the matched 515/512 arm initially looked successful. Three c4
batches measured 85.2119, 81.8794, and 79.6712 aggregate post-first tok/s;
c8 measured 87.1168, 83.6927, and 79.0298. C4 was the better serving point
because it had lower latency for nearly the same aggregate capacity. The
required 12-batch c4 control then exposed cumulative replay degradation:
84.0771, 81.4030, 78.1459, 74.6536, 73.0380, 69.5854, 66.9912, 64.6683,
62.7426, 60.2803, 58.3836, and 56.7261 tok/s. Its median was 68.2883, but the
tail crossed below the objective and lost 32.5 percent from first to last.
Control JSON SHA256 is
`6986a8e6abc48a6f1957e8af14f4634a413c5e3b9df44988819023a73a4dfa99`;
runtime log SHA256 is
`b597b3bc483a88ef41d944631a0cdae19f32525a2f485f764503d2a20039a689`.

COMMAND -> port the retained vLLM `B70_XPU_CG_RECLAIM` mechanism into the
refreshed SGLang overlay. When enabled, each `torch.xpu.XPUGraph` retains its
modifiable graph and calls `instantiate()` before every 500th replay, resetting
the accumulating Level Zero executable command-list state without retracing.
The launcher mounts the tracked overlay over the image copy, passes
`B70_XPU_CG_RECLAIM`, and defaults the qualified breakable route to 500. Two
setup-only attempts correctly stopped before traffic when the enable marker
was absent: the first modified the unused legacy shim, and the second had not
yet mounted the tracked refreshed overlay. Both tore down and passed card and
compiled collective health; neither produced a speed result.

RESULT -> the corrected candidate emitted the enable marker on both ranks and
16 sampled live re-instantiation markers. The same 12 measured c4 batches
returned 88.6186, 87.7131, 89.3556, 87.7184, 88.2516, 88.3519, 87.2723,
87.8725, 87.7168, 87.2393, 86.8321, and 88.1472 aggregate post-first tok/s.
Median was 87.7954, range was 86.8321-89.3556, aggregate including TTFT median
was 86.1822, and the first-to-last delta was only -0.53 percent. All 48 streams
returned exactly 512 tokens, for 24,576 measured output tokens. Repeated greedy
output remained byte-identical at the control hash, all four concurrent
arithmetic canaries passed, exact model identity passed, and no fatal runtime
marker appeared. Candidate JSON SHA256 is
`b8d58bd5c6e9ce9a3c60029c732dd4a9cb3c9949fba7bd2236d32c761d0166ba`;
runtime log SHA256 is
`399fddaaf7f3c437c9d6c1cf3c972eb2f3f8dd55f6201af83e0e2e93af3ff7a1`.
The server stopped gracefully. Final per-card and compiled P2P-off collective
health passed, no container remains, and both GPU leases are free.

VERDICT -> qualify Ornith target-only breakable graph with reclaim500 as the
current W8A8 MoE serving winner for the matched p515/o512 c4 regime. Its
sustained 87.7954 tok/s median exceeds the 65 tok/s objective by 35.1 percent
and removes the rejected control's cumulative slowdown. Keep MTP rejected
because Shisa output is not target-exact, keep dense-native off because it is
slower, and keep direct P2P and FULL TP2 capture rejected. Do not generalize
the 87.80 number to long-prefill traffic: the separately measured p4172/o128
c8 regime is 39.23 tok/s. This is a performance-control qualification, not a
live-shelf promotion.

### 2026-08-28a - SGLang FULL capture brings Qwen3.8 NVFP4 TP2 to 30.17 tok/s

CONFIG -> kernel 7.1.0-070100, refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`,
Torch 2.13.0+xpu, SGLang `bede6bc`, sgl-kernel-xpu `2d10888`, Compute
Runtime 26.22, and the local RadixArk cache revision
`554ebba9b5f1b79dc11246341960360e6ef05ef4`. The source-built XPU operator
SHA256 was
`96e33b4e66f4eba6a2108c5a4f3aef5fba505f3696ba876e60b6ddeb08a87549`;
its matching GDN sidecar SHA256 was
`323547ed36f4821ccba6fbbc75ced8fd6e9837e268891d6488d62825002279a8`.
The serve used TP=2, FULL decode graph at batch size one, prefill graph off,
bf16 KV cache, chunked prefill 128, maximum one request, P2P disabled, pidfd
IPC, SYCL collective kernels enabled, Triton linear attention, and a text-only
runtime copy of the multimodal checkpoint config.

COMMAND -> add `sglang/refresh/b70_xpu_nvfp4.py` and its `.pth` loader, mount
the exact XPU operator pair through
`sglang/nvfp4/serve_qwen38_nvfp4_refresh.sh`, and run every GPU action inside
`bin/gpu-run`. The overlay admits the checkpoint's ModelOpt format, preserves
packed E2M1 weights, folds or retains group-16 scales for the two current XPU
operator paths, and narrowly enables quantized lm_head dispatch. Query
`/v1/models`, run repeated greedy and arithmetic coherence gates, then run
`sglang/w8a8/bench_forced_concurrent.py --concurrency 1 --prompt-repeat 35
--output-tokens 512 --batches 3` for the tokenizer-derived p879/o512 shape.

RESULT -> the first eager load exposed SGLang's raw-matmul fallback for the
packed lm_head. The narrow runtime-state gate fixed that bug. The corrected
eager route passed identity and coherent generation, and the FULL route served
exact ID `qwen3.8-27b-NVFP4-radixark-sglang-full-tp2`. Its three post-first
rates were 30.2751, 30.1665, and 30.0855 tok/s; median was 30.1665 tok/s and
median including TTFT was 27.6079 tok/s. The result JSON SHA256 is
`618e99288361be4dfa88119bc2ef4a71bac52fca1a3c38d1f31a9c2dddc7bece`;
runtime log SHA256 is
`e09f3995fcc289b8d98c7280b095d4e462342a07a972e3d280e49818932dc217`.
The server stopped normally and post-card plus compiled P2P-off collective
health passed.

VERDICT -> qualify the refreshed SGLang FULL route as the first coherent TP2
execution baseline for this NVFP4 checkpoint. It is 5.96x the retained vLLM
0.28 TP2 graph result of 5.06 tok/s, but it remains 24.6 percent below the
40 tok/s single-stream objective and is not a shelf promotion.

### 2026-08-28b - XPU FP8 W8A16 decode raises matched NVFP4 speed by 8.14 percent

CONFIG -> the exact 2026-08-28a stack and TP2 serve shape. Source accounting
found, per rank and target token, 129 NVFP4 calls, 128 FP8 calls, 48 tiny bf16
linear calls, 129 all-reduces, and one logits all-gather. Approximate compulsory
weight and scale traffic was 8.197 GiB per token per rank. The stock FP8 route
issued a static activation quantization plus `torch._scaled_mm` for each of
the 128 FP8 projections.

COMMAND -> use `sglang/refresh/bench_qwen38_decode_linears_xpu.py` under a
one-card `bin/gpu-run` lease to compare real fused TP2 checkpoint shapes for
stock scaled_mm, direct `_xpu_C.fp8_gemm`, and
`_xpu_C.fp8_gemm_w8a16`. Test GDN qkvz M1x5120x8192, full-attention qkv
M1x5120x7168, and common output M1x3072x5120. Validate numerical agreement,
determinism, and XPUGraph replay before adding an environment-gated M<=1
W8A16 branch to the SGLang overlay. Repeat the exact p879/o512 c1 three-batch
serve with ID `qwen3.8-27b-NVFP4-radixark-sglang-w8a16-full-tp2`.

RESULT -> direct W8A8 was bit-identical to stock. W8A16 versus dequantized
weight references had cosine at least 0.9999965 and relative L2 at most
0.00264, and its XPUGraph replay was bit-identical to eager. Representative
W8A16 GEMM times were 0.0728 ms for GDN qkvz, 0.0616 ms for full-attention
qkv, and 0.0320 ms for the common output projection. The matched end-to-end
post-first rates were 32.7553, 32.6206, and 32.3642 tok/s; median was 32.6206
tok/s and median including TTFT was 29.7873 tok/s. Exact identity, repeated
greedy determinism, and the arithmetic canary passed. Result JSON SHA256 is
`71b18391e8fe545b52c8f16a640fcb93888a88e32e16ebaa208ab856e8853a99`;
runtime log SHA256 is
`6cbb837e8c9e8b0fda5107e12ddb3800e688ee524f43d41ff258abfaaba1829d`.

RESULT -> two setup attempts used prompt repeat 260 and correctly received an
HTTP context-length error because the resulting 6049-token request exceeded
the configured 4096 context. They are not performance results. The accepted
run used the tokenizer-derived repeat 35. The runtime log's only traceback is
SGLang's post-warmup self-call to `/freeze_gc` before its endpoint was
reachable; the endpoint subsequently opened, served every gate and benchmark,
and stopped normally. Kernel logs show no OOM, GPU hang, reset, fault, panic,
or reboot. Final per-card and compiled P2P-off collective health passed.

VERDICT -> make FP8 W8A16 at M<=1 the refreshed NVFP4 launcher's default. It
removes 128 activation-quant kernels per token and improves the matched median
by 8.14 percent. The result remains 18.4 percent below 40 tok/s, so retain it
as the current single-stream research winner, not a live-shelf entry.

### 2026-08-28c - M1 GEMV, native GDN, and direct P2P controls are rejected

CONFIG -> exact Torch 2.13 XPU stack from 2026-08-28a. The ESIMD source was
the retained M=1 prototype rebuilt against the current ABI; artifact SHA256
was `f44197b4d3a40f363375fd60d65bf570fc763cb265aadd47d442979436a67a7d`.
The native GDN arm changed only decode linear attention from Triton to
`intel_xpu`. The collective A/B used the exact current oneCCL libraries,
bf16 shape [1,5120], 64 direct iterations, 128 graph iterations, pidfd IPC,
and otherwise matched P2P-off/P2P-on environments.

COMMAND -> first compare ESIMD and current oneDNN NVFP4 output and timing on
the exact TP2 gate/up, down, and lm_head M=1 shapes. Then start a W8A16 FULL
serve with native decode GDN and require two byte-identical greedy responses
before any timing. Finally run the retained Steve-derived two-rank collective
oracle inside one outer `bin/gpu-run`, P2P off first and the explicit guarded
P2P-on arm second, followed by per-card and compiled P2P-off health.

RESULT -> ESIMD was correct and deterministic but slower: gate/up was 0.2605
ms versus 0.0554 ms, down was 0.2545 ms versus 0.0531 ms, and lm_head was
2.2596 ms versus 0.6340 ms. The native GDN endpoint exposed exact identity but
the two greedy responses were not byte-identical, so it was stopped before a
speed claim. Its runtime log SHA256 is
`e9cf1e2ad652c9c932383844ed0731471715fb437a9b9d6820b835516f3ca261`.

RESULT -> both collective arms were bit-exact with zero mismatches. P2P off
measured 0.35118-0.35132 ms per graph iteration; P2P on measured
0.36503-0.36531 ms, about 4 percent slower. P2P-off JSON SHA256 is
`f1bdeb63163b46e9aea1a59573ea65f9a22379ca46cbfd39190cdb704f5fca40`;
P2P-on JSON SHA256 is
`a5fb6015c699ea5e9ece783bf8cbf18f1feb580dff013f6d8169d0ce79d1849b`.
All post-run health passed.

VERDICT -> reject the ESIMD M=1 kernel, native GDN backend, and direct P2P as
current model optimizations. Keep oneDNN NVFP4, Triton GDN, and P2P disabled.
The exact-shape P2P oracle removes any justification for risking a model-level
P2P-on arm on this stack.

### 2026-08-28d - Current-stack oneDNN W8A16 dense route is rejected

CONFIG -> exact SGLang runtime
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`,
Torch 2.13.0+xpu, vllm-xpu-kernels source `2dd55f3`, and the real Qwen3.8
W8A8 GPTQ TP2 rank shapes. The candidate adds a source-built oneDNN
INT8-weight/BF16-activation operator. No quarantined binary or source was
restored.

COMMAND -> build `kernels/int8_gemm_w8a16_2dd55f3.patch` through
`sglang/refresh/build_int8.sh`, then run
`sglang/refresh/bench_qwen38_w8a8_linears_xpu.py` on card 0 through
`bin/gpu-run`. Compare eager and graph execution for the real M1 gate/up,
down, qkv, and output projections, and require numerical agreement and exact
replay before any model integration.

RESULT -> image
`b70-sglang-xpu-int8-w8a16@sha256:91cc53fab0e683a27735667ff0802ee065d328ad66f1d0d2d6c7236d0e1475f3`
built successfully. The patched tree was `a2559481686fcabeb95d5a315c73f87c3c4f5fe9`;
wheel SHA256 was
`4b62ca2bc19588dbe5b644260a33a8dbb1f2e26a75d1d304a067297ddbc60087`,
and installed shared-object SHA256 was
`1fd3cb680d46b50338bdb1ed71883ec73423183b55126011de99aa653ed2b3df`.
Correctness and exact graph replay passed, but performance failed decisively.
Gate/up graph time was 2.0775 ms versus 0.1911 ms for current W8A8, down was
3.5929 versus 0.1111 ms, qkv was 2.0649 versus 0.08027 ms, and output was
1.3099 versus 0.03346 ms. The 160-call weighted estimate was 416.899 ms
versus 21.1619 ms, or only 0.05076x current speed. Result JSON SHA256 was
`55fd3b4892fd4ca1ef45c083adc3ac42272624eab5b053a84c62525dbee26ba7`.
Pre- and post-card health passed.

VERDICT -> reject oneDNN W8A16 for Qwen3.8 dense decode before full-model
integration. Retain the tracked source port and exact artifact identity as a
negative control; do not use it as a speed route.

### 2026-08-28e - Selective GDN INT8 reaches 24.56 tok/s after OOM recovery

CONFIG -> exact runtime image `adc915d...`, Qwen3.8 W8A8 GPTQ, TP=2, FULL
decode graph at batch size one, p879/o512, maximum one request, 4096 context,
Triton GDN, source-default c10d, pidfd IPC, SYCL collective kernels, and P2P
disabled. Only the 48 GDN `in_proj_qkvz` and 48 GDN `out_proj` weights per
rank change from ignored BF16 to load-time per-output-channel RTN INT8; all
checkpoint W8A8 projections retain the qualified native route.

COMMAND -> first run `sglang/refresh/bench_qwen38_gdn_int8_xpu.py` under a
card-0 lease on real BF16 checkpoint projections. Then launch the selective
route through `sglang/w8a8/serve_qwen38_w8a8.sh`, require two conversion
markers and exact `/v1/models` identity, and run
`sglang/w8a8/bench_forced_concurrent.py --concurrency 1 --prompt-repeat 35
--output-tokens 512 --batches 3`. Enclose every TP2 attempt in `bin/gpu-run`
with teardown, per-card health, and compiled P2P-off collective health.

RESULT -> the one-card gate passed correctness, determinism, and 16 exact
graph replays. GDN qkvz improved from 0.154898 to 0.089870 ms and output from
0.048875 to 0.028697 ms. The 48-plus-48 weighted estimate fell from 9.7811 to
5.6912 ms per token per rank, a 1.7186x projection speedup. Cosine was at
least 0.999911 and relative L2 at most 0.01334. Result JSON SHA256 was
`82ac550616e73d524a797dafe0019728a5d273ef6461ee41dd64cecfc4c67817`.

RESULT -> the first full-model arm used `mem-fraction-static=0.90`. Both
ranks loaded and converted all 96 projections, then stalled for 7 hours 36
minutes after Mamba-cache allocation while attempting a 462,976-token KV
pool. At 09:21 the kernel reported a global OOM with about 59 GiB
`gpu_active` and killed the user D-Bus service. At 16:57 another global OOM
killed user systemd, closing tmux and Codex, and killed rank 1. The host did
not reboot. The dead container reported `OOMKilled=true`; full log SHA256 was
`accc095d3086fd2a0a811f15ceb3a6d27fb0638245e5de32cee14b31d7607cc0`.
The orphan process was gone but left stale owner text in the unlocked lease
files. The normal stop path preserved the log and removed the container.
`bin/gpu-run bin/xe-reset` completed a non-reboot rebind; both card probes and
the compiled ten-iteration P2P-off collective passed.

RESULT -> the guarded launcher now defaults to `mem-fraction-static=0.75`,
adds container OOM score adjustment 500, and the qualification command used a
ten-minute startup ceiling. A first guarded attempt allocated 306,176 KV
tokens, leaving 8.04 GB per rank, then exposed and cleanly rejected an
argument-binding bug in the GDN adapter during graph capture. Fixing the
adapter's call into the shared native helper produced a healthy endpoint in
125 seconds. Both conversion markers and exact served ID
`qwen3.8-27b-W8A8-gptq-gdn-rtn-full-tp2` passed. Repeated greedy output was
byte-identical, the arithmetic canary returned exact answer 45, and all three
measured streams completed 512 tokens. Post-first rates were 24.6409,
24.5628, and 24.5433 tok/s; median was 24.5628 tok/s and median including
TTFT was 23.0228 tok/s. This is 9.40 percent above the prior 22.4513 tok/s
W8A8 FULL control and 1.75 percent below the 25 tok/s objective. Result JSON
SHA256 was
`85045f85825c0ef27975856eab315b5ffe269b1ebdc7ecceccab91185f34e7fb`;
runtime log SHA256 was
`1f1db8c0380472b852bfd493b45700e843119848b83d5c97b96eb35136a05f9e`.
Graceful teardown, both card probes, and compiled P2P-off collective health
passed; no container remains and both leases are free.

VERDICT -> reject the 0.90 memory fraction as unsafe for this host-visible
VRAM configuration. Retain the 0.75 OOM-guarded selective GDN route as the
new coherent Qwen3.8 W8A8 single-stream winner. It is a material improvement,
but the strict 25 tok/s objective remains narrowly unmet, so continue with the
next measured bottleneck rather than promoting a shelf entry.

### 2026-08-28f - Qwen W8A8 LM-head INT8 is fast in isolation but not target-exact

CONFIG -> exact runtime image `adc915d...`, the qualified selective-GDN W8A8
route, and the rank-local TP2 BF16 LM head shape [124160,5120]. The candidate
used load-time per-output-channel RTN INT8 plus the existing dynamic
activation-INT8 oneDNN operator. It was default-off, target-only, and scoped
to the exact Qwen3.5 conditional-generation class, TP2/PP1, untied BF16 head,
and normal SGLang logits gather path.

COMMAND -> benchmark both real TP vocabulary shards on card 0 through
`bin/gpu-run`, requiring finite output, cosine at least 0.999, relative L2 at
most 0.02, local argmax equality, deterministic eager output, and 16 exact
XPUGraph replays. Then capture an eight-prompt, twice-repeated, fixed-seed
target corpus and compare the full candidate completions before any model
speed benchmark.

RESULT -> rank 0 reduced graph time from 2.14470 to 1.13681 ms and rank 1
from 2.14010 to 1.12884 ms, saving about 1.01 ms/token/rank. Both local
argmax checks and graph replay passed. Result SHA256 values were
`d6b00e12502ae404d71705c43f2eef20fac77d425dcbecbdd4b0e40fcb53d945`
and `a6277f303cd9d540e4a3b4a528ce45f6d625f4401a47bc1ef17a576f93b0801e`.
The full candidate loaded and converted both ranks, but changed prompts 6 and
7 of the eight-prompt target corpus. Reference and candidate JSON SHA256 were
`b33d8afecade1ccd13afc6f330b58d16013ed57976730c8f365e2085ee888802`
and `3a720e4adc6334d149c75cd7312ace49cc3835c938e4253f10e768b6eba89d67`.
It was stopped before timing. The model A/B inadvertently omitted the outer
`gpu-run` lease, although no other GPU work overlapped; therefore it is a
screening rejection, not qualification evidence. Post-card and compiled
P2P-off collective health passed.

VERDICT -> reject activation-W8A8 for the output-sensitive LM head. Retain the
default-off source and microbenchmark as a negative control. A future LM-head
candidate must avoid dynamic activation quantization and still pass the full
target corpus before performance measurement.

### 2026-08-28g - New official Ornith MTP is verified but rejected on SGLang

CONFIG -> the official `ornith-ai/Ornith-1.5-35B-A3B` repository remained at
revision `10fbf86fed7ecee4a061f8b499a618f46001cac1`, updated 2026-08-23. No
newer official release existed. Its 19 BF16 MTP tensors were already merged
into the local W8A8 target as
`w8a8-rtn-mtp-official-10fbf86`; contract SHA256 was
`4e286e6f85e868f60f07b8b1cc4adcc4bd875fe274d8b43d658952f3308a7150`.
The Shisa MTP-only repository remained the separate 2026-08-21 revision
`2b19b31`. The serve used TP2, P2P off, breakable graph size one, reclaim500,
4096 context, maximum one request, memory fraction 0.80, and no tool parser or
thinking grammar.

COMMAND -> hold both GPU leases for the full sequence, run per-card and
compiled collective health, capture the official-checkpoint target-only
eight-prompt corpus, tear down and recheck health, then start official MTP1
and compare every completion hash before benchmarking.

RESULT -> target-only was repeat-exact. Official MTP1 loaded, shared the
head, captured its draft graphs, and reported acceptance lengths around
1.70-1.98 with acceptance rates around 0.70-0.97. It nevertheless changed 7
of 8 target completions: indices 0,2,3,4,5,6,7. Reference and candidate JSON
SHA256 were
`0adfb91ca9d3d4e4d3c98936c64c5600ba8781c6aa19b20b719694b4f6e47b8f`
and `1a173a86c29d97b679c1526dce659bcfc3b1b60921f19bffab987f7821f66736`;
runtime log SHA256 values were
`261bcd81dd3f07041a9e96f9439854d461a6e9b59cf494b39a0f672a07307211`
and `ea0f3caf8efd41b1ca16b31f44ebc886f052c09a306811a51ea8d2e113ebec1f`.
No speed benchmark ran. Graceful cleanup and final card plus compiled
collective health passed.

VERDICT -> reject the official Ornith MTP head on the current SGLang
speculative path. High draft acceptance does not compensate for target-output
divergence. Retain target-only Ornith W8A8 as the coherent serving route.

### 2026-08-28h - XeCores Qwen GPTQ INT4 BF16-MTP transfers to vLLM 0.28

CONFIG -> exact vLLM 0.28 image
`vllm/vllm-openai-xpu@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3`
and XeCores recipe artifact
`SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` revision
`9d189a60e4c0ad7f9f47cd94bfa393ca10b3924e`. The exact 16-file tree was
downloaded and verified: five shards totaling 19,559,450,216 bytes, 2,399
tensors, GPTQ INT4 symmetric group 128 with desc_act false, 400 quantized
weights, and all 15 MTP tensors BF16. The current vLLM dynamic exclusion
already preserves MTP, so no legacy BF16-draft patch was used. All arms were
TP1 eager on leased card 0, text-only, maximum one request, 4096 context,
memory utilization 0.75, fixed seed 20260828, and thinking disabled.

COMMAND -> create the dedicated current-stack launcher, capture the target
eight-prompt reference, and run matched 839-prompt/512-output c1 tests. Then
test BF16 MTP depths 1, 2, and 4 in sequence, requiring exact equality to the
saved target corpus before speed measurement. Add optional fixed-seed and
thinking-disabled controls to the serving benchmark without changing its
historical defaults.

RESULT -> target-only was repeat-exact and measured 7.9642, 7.9304, and
7.8909 tok/s, median 7.9304. MTP1 was target-exact and measured 13.4763,
13.8193, and 13.4805 tok/s, median 13.4805, 70.0 percent above target-only.
MTP2 was target-exact. MTP4 was target-exact and measured 19.6595, 21.4223,
and 20.3339 tok/s, median 20.3339, 2.56x target-only. MTP4 mean acceptance
length varied about 2.65-4.29 and average draft acceptance about 41-82 percent.
Target, MTP1, and MTP4 result SHA256 values were
`df838a34667e2717d7ae4c7c00d7b98f6c169018974bbd6677d79a1c424b7e16`,
`6c7739c5140aba5af02fcc269145db83900dbcbce77eb7c39d5545ebc8d2b50b`,
and `d6cdc954866b9a927414fd8c3269ffca8e6aa024a0f0c91da5ab65532ae5d1da`.
Every teardown and card-0 health check passed.

VERDICT -> the pinned XeCores checkpoint and upstream vLLM 0.28 BF16-MTP
handling transfer correctly. MTP4 is the coherent eager winner but remains
49.2 percent below 40 tok/s. Proceed to the recipe's PIECEWISE/breakable graph
arm with legacy partitioning before considering draft-side quantization or
TP2.

### 2026-08-28i - vLLM graph is neutral; draft-only LM-head INT4 reaches 21.55 tok/s

CONFIG -> the exact XeCores/vLLM 0.28 TP1 MTP4 setup from 2026-08-28h. The
graph arm used PIECEWISE breakable capture, sizes 1,2,4, no compile-size
padding, legacy partitioning, AOT disabled, and the normal BF16 MTP draft. The
second arm returned to eager and changed only the separately loaded draft LM
head from FP16 to load-time per-output-channel RTN GPTQ INT4 group 128. A
v0.28-specific default-off overlay prevented later target-head sharing,
installed a draft quant method before compilation, preserved normal
LogitsProcessor and TP gather semantics, released only draft FP16 storage,
and left the target head untouched.

COMMAND -> require the eight-prompt target corpus before timing each arm. Run
the matched 839-prompt/512-output c1 three-batch benchmark. For the draft-head
arm require exact class, unquantized FP16 head, shape and group compatibility,
NT packed layout, target/draft isolation markers, deterministic canary, and
post-card health.

RESULT -> PIECEWISE remained target-exact but measured 20.3793, 21.5927, and
20.5439 tok/s, median 20.5439, only 1.03 percent above eager MTP4. It saved
about 0.503 ms per emitted token. Source accounting explains the ceiling:
breakable capture keeps all 48 GDN and 16 full-attention target cores eager,
plus four eager MTP attention passes, while replaying many small graph
segments around them. Graph corpus, result, and log SHA256 values were
`879a2787bbf9b1b4ccd96b07d717e17c6d86c83bcead70ffcf64b288d44d6685`,
`b9c69abf2d17557d67f4c9a94f725c74c600d228addc4209507661697072e932`,
and `9647ce13db446f82a3b77822305ea9031626272e9e94a6d36849bbcac3c676eb`.

RESULT -> the draft-head overlay packed [248320,5120] into 635,699,200
qweight bytes plus 19,865,600 scale bytes, released the draft FP16 parameter,
and emitted explicit target-untouched and no-share markers. It remained
target-exact and measured 20.9408, 22.9737, and 21.5520 tok/s, median 21.5520,
6.0 percent above BF16-head eager MTP4. Average draft acceptance remained
about 43-83 percent, so the gain came without a material acceptance collapse.
Corpus, result, and log SHA256 values were
`4a16198e4c1bdd24fc19fc1c4738377d405c0b61b31f86a9a2b3725177daf109`,
`12db1887cac4e9492936f3af56e2cfea96252817e4662e5e988061aad1c52618`,
and `4081d7faf8793735eac19d831189a397c6eeb09d8f47da1fb55b536218d2491a`.
Both arms stopped gracefully and card-0 health passed.

VERDICT -> retain the draft-only LM-head INT4 overlay as the coherent TP1
winner. Reject further PIECEWISE-only tuning: attention/GDN eager breaks cap
its benefit near one percent. At 21.55 tok/s the strict 40 tok/s objective
still requires a faster target path; advance to guarded P2P-off TP2 eager
qualification before deeper draft quantization.

### 2026-08-28j - GPTQ TP2 regresses and draft-MTP INT4 fails exactness

CONFIG -> the exact XeCores artifact and vLLM 0.28 image from
2026-08-28h. The topology arm used TP2 eager, the multiprocessing executor,
pidfd IPC, SYCL collective kernels disabled, P2P disabled, maximum one
request, 4096 context, and memory utilization 0.75. The draft arm returned to
TP1 eager MTP4 and converted only the five separately loaded MTP linears to
load-time symmetric INT4 group 128. Its v0.28 overlay accepted the runtime's
FP16 or BF16 source weights, used direct final packing buffers, cast FP16 or
BF16 activations to the W4A16 operator input, cast output back to the original
dtype, and left the shared target head and target model untouched.

COMMAND -> hold both GPU leases around the complete TP2 sequence; run card
and compiled P2P-off collective health before and after serving; require
exact equality to the saved TP1 target corpus before the matched
839-prompt/512-output c1 benchmark. For draft-MTP INT4, first require exact
class, five exact linears, unquantized source methods, shape and group
compatibility, successful conversion markers, exact served identity,
repeat-deterministic generation, and equality to the same target corpus.
Stop before timing on any mismatch.

RESULT -> TP2 target-only was repeat-exact and byte-identical to TP1 on all
eight prompts, but measured only 4.4903, 4.5041, and 4.4768 tok/s; median was
4.4903 tok/s, 43.4 percent below the 7.9304 tok/s TP1 target median. Corpus,
result, and runtime-log SHA256 values were
`9ace957f871a6d5726835399bb409610fdc4954c6cd2e1df6c92f2b2dbf7d70e`,
`65eed9515f44a9eba2f9a29202c2cb29ddedb9cb4c1467438999944fe0ec2834`,
and `16aba8cb32c1eff3bce8a06d5742069e9883f2879f39f29e72fec2170f3da43e`.
The server stopped gracefully and post-card plus compiled collective health
passed.

RESULT -> the first draft-MTP setup attempt failed closed before inference
because vLLM had materialized the nominal BF16 checkpoint tensors as FP16;
its log SHA256 was
`7ee42aab3079a1ed7511f8756c119fd4e09f2de9852a9dc55479fe361b0678cc`.
After widening the strict source guard to FP16 or BF16, all five linears
converted: 849,346,560 source bytes became 218,972,160 packed bytes. A
separate healthy retry proved coherent generation but a host invocation
mistake supplied `/v1` twice to the corpus tool and returned HTTP 404 before
the corpus; its preserved runtime log SHA256 was
`28b0b5f6937ac7220541bb247da70e13d9ceb04ec44e2c7653b4b5b88ca82181`.
The corrected run was repeat-exact but changed target prompts 2, 5, and 6.
Candidate corpus and runtime-log SHA256 values were
`8b58ad3d9bae80cd870db54cc92dfc22cd008244ddcee89d797035b8099aee75`
and `978cdf28686be07094eb28e7bea06395fec9df5d1580a92d9fb5c67d2b02248f`.
It stopped before timing. Graceful cleanup and card-0 post-health passed.

VERDICT -> reject TP2 for this GPTQ target: communication and duplicated
small work overwhelm the split GEMMs even before MTP. Do not risk a TP2 MTP
arm. Reject draft-MTP INT4 because it changes target output despite stable
repetition and plausible acceptance. Retain the default-off implementation
as a reproducible negative control, do not combine it with the accepted
draft-head path, and continue from TP1 MTP4 plus draft-head INT4.

### 2026-08-28k - BF16-KV vLLM 0.27.2 plus draft head clears 40 tok/s

CONFIG -> the pinned XeCores image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
vLLM `0.27.2rc1.dev77+gac7509e2b`, XPU graph enabled, PIECEWISE capture,
TP1 on card 0, 4096 context, maximum one sequence, memory utilization 0.75,
and the same GPTQ INT4 group-128 checkpoint. The published recipe used FP8
KV. The qualified local route instead used BF16 KV, per the standing campaign
preference. MTP4 kept all five draft linears BF16. Its final arm added only
the default-off cookbook draft LM-head INT4 patch; target weights and target
verification head remained unchanged.

COMMAND -> first reproduce the published image with FP8 KV at target-only,
requiring two identical greedy completions for every corpus prompt before
MTP or speed. Repeat FP8 KV on current vLLM 0.28 eager plus the accepted
draft-head route to isolate cache numerics from graph/runtime effects. Then
return to the pinned 0.27.2 image with BF16 KV: capture its target corpus,
benchmark target-only, require MTP4 equality to that target, benchmark MTP4,
and finally require the draft-head candidate to equal the same target before
the matched 839-prompt/512-output three-batch benchmark. Enclose every GPU
operation in a card-0 lease with graceful teardown and card health.

RESULT -> target-only FP8 KV on the published PIECEWISE recipe was not
repeat-exact on prompt 6, so it stopped before MTP or speed. Runtime-log
SHA256 was
`320c56a7d5225f136f139c61e989b9462249a4ef77f6bdb334519654a89b4848`.
Current vLLM 0.28 eager with FP8 KV independently failed repeat-exactness on
the same prompt, proving that cache precision rather than the old graph was
the common cause. Its log SHA256 was
`105ba4eabf03e94183c11f43fc359b149fab05175e050e9c8b653a54ed68f65b`.
Neither FP8 arm was timed.

RESULT -> BF16 KV restored repeat-exact generation on vLLM 0.27.2. Its
target corpus differed from the vLLM 0.28 target on prompts 2, 5, and 6,
which is a runtime-stack numerical boundary rather than an MTP change.
Target-only measured 14.5867, 12.8728, and 11.4988 tok/s; median was 12.8728
tok/s, 62.3 percent above the vLLM 0.28 target median. MTP4 was exactly equal
to its same-stack BF16 target and measured 39.2604, 41.4526, and 35.8199
tok/s; median was 39.2604 tok/s. Target corpus, target result, MTP corpus,
MTP result, target log, and MTP log SHA256 values were
`42364f1e7a01b9298c40e21ac821924eb7796cfa2abf94f50405abe302077f7d`,
`7e872623835cea154784dba76275bf466502519c4f29c8a32bff2e29208da35f`,
`34a772ff156cda64c818f7dff1b303fabbf68bfc1747bf2faf2516cefb4dfc03`,
`3bd5323d230a29b084171d2894449dec305809e6fefceba87d944d6a29d4d538`,
`0541e8e93ba673edae25fc83f038369775650bc3e52f7d05da9d3a3529bf308c`,
and `b705443edb765e8b7a5b90aebc24b6aa527788af03f0bdba553a43bbf695c6c8`.

RESULT -> changing only the draft LM head to INT4 remained exactly equal to
the vLLM 0.27.2 BF16 target corpus. It measured 45.7872, 48.0495, and
42.2354 tok/s; median was 45.7872 tok/s, 16.6 percent above the same-stack
BF16-head MTP4 median and 2.25x the current vLLM 0.28 eager BF16-draft MTP4
median. Corpus, result, and runtime-log SHA256 values were
`699818c8629c783a2cfd727f94a5c9529963a494532e0233459f9abdf6b2cfe4`,
`dd81a07f01a8caada16121f01b0d9477d46fb8f1e9fd0214b7540d1bce6202c0`,
and `b92f8368ce0e220ff39bc0a4bb5c99ff3e4d41179164fc1b166ab062e6ebe81e`.
All three measured streams completed 512 tokens. Every teardown and card-0
health probe passed.

VERDICT -> the 40 tok/s Qwen 4-bit objective is achieved at a 45.7872 tok/s
median with exact same-stack target output and BF16 KV. Retain vLLM 0.27.2 as
the pinned qualified serving control while treating the 2.25x vLLM 0.28 gap
as a regression to bisect. Permanently reject FP8 KV from the campaign path:
it violates repeat determinism on both tested runtime stacks.

### 2026-08-28l - Qwen 4-bit winner passes long C1 and C2; C4 aborts

CONFIG -> the 2026-08-28k winner with BF16 KV, MTP4, draft-only LM-head
INT4, and PIECEWISE graph on the pinned vLLM 0.27.2 image. The concurrency
arm raised maximum sequences to four and applied the cookbook mixed
speculative/non-speculative GDN split patch. The long arm returned to maximum
one sequence and used 839 prompt tokens plus 2,048 output tokens.

COMMAND -> on one card-0 lease, require the eight-prompt target corpus again,
then run three matched C2 batches followed by three matched C4 batches, each
request producing 512 tokens. Preserve the full runtime log if the engine
fails. After cleanup and health, launch the C1 configuration separately and
run one same-shape 2,048-token warmup plus two measured 2,048-token streams.

RESULT -> the max-sequence-four server remained exactly equal to the BF16
target corpus. C2 passed both arithmetic canaries and all six measured
streams completed 512 tokens. Aggregate post-first rates were 42.3087,
39.8084, and 33.9486 tok/s; median was 39.8084 tok/s. Median TTFT increased
from 6.72 to 7.96 seconds across the three batches. Corpus and C2 result
SHA256 values were
`873c11fb6810181b0af3880d60c399b3f85d8181f3a77ba714a04eaca954c7d9`
and `e5aa366ab290c9879f8624afb15e74e25bdae8a403ca04f5c9815e2eb3c55ca5`.

RESULT -> C4 passed all four arithmetic canaries and its first eight measured
streams completed, but the patched scheduler ran only one request while
three waited. Aggregate post-first throughput fell from 25.8180 to 22.7985
tok/s across the first two measured batches. During the third, the engine
aborted in Level Zero `linear_stream.h:90`; the API returned an engine-dead
stream without timing or usage fields. Runtime-log SHA256 was
`c4a93aa1e96cf1f8cda2a70c9e7d925fcb010072a8aea59a6bae0cc4b540f438`.
The container was removed and the card-0 health probe passed.

RESULT -> the separate long C1 arm completed the 2,048-token warmup and both
2,048-token measured streams. Measured post-first rates were 43.1751 and
47.0693 tok/s; median was 45.1222 tok/s. Result and runtime-log SHA256 values
were
`7846bc1e09611178e77e27bf984432d6109fa3b02e7dcb2e15748e34d7dd49b5`
and `9ff3180c108a884b4eeaf01f2915c9bbc85969904c517a72d0aeca0e00c9cd2c`.
Graceful teardown and final card health passed.

RESULT -> a dedicated pinned launcher was ported to
`vllm/gptq_int4/serve_qwen38_gptq_int4_v0272.sh`. It defaults to the C1
winner, refuses non-BF16 KV, requires the mixed-split patch above one maximum
sequence, and records logs before graceful removal. Its first smoke failed
closed at CLI parsing before model load because speculative JSON quoting was
lost across the container shell. After correcting only that quoting, the
launcher returned the exact served ID and the canary response `45`; graceful
stop and card health passed. Corrected smoke-log SHA256 was
`95d1e7801ab3c35e6aed529d46f029467a0b1bc1773f5bc824f688eb85df3359`.

VERDICT -> qualify the BF16-KV winner for sustained C1 and bounded C2 use.
Reject C4 and do not shelf-promote a max-sequence-four configuration: the
mixed-batch patch serializes work and the old Level Zero command stream still
aborts under repeated C4 load. The pinned launcher defaults to maximum one
sequence, requires explicit mixed-split enablement above one, and refuses a
non-BF16 KV override.

### 2026-08-28m - Ornith reclaim500 survives TB3 but Pi times out

CONFIG -> exact refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
and local Ornith W8A8 RTN checkpoint. The accepted retry used TP2, P2P off,
source-default eager collectives, target-only decode, BF16 KV, breakable graph
size one, graph re-instantiation every 500 replays, qwen3 reasoning,
qwen3_coder tools, strict-thinking cap 4096, maximum one request, 65,536
context, and memory fraction 0.70. Harbor 0.22.0 ran official
`terminal-bench/terminal-bench@3.0.0` task `bun-sourcemap-leak` with Pi 0.84.3
at xhigh and the retained concise prompt.

COMMAND -> first retry the old failing task at memory fraction 0.90 inside a
whole-box `bin/gpu-run` lease, then inspect the kernel and server evidence after
the user tmux session disappeared. Recover with exact-image per-card and
compiled two-rank P2P-off health. Retry at memory fraction 0.70, require exact
served identity, and let Harbor run through its official 1,800-second agent
budget and verifier. Record server-start, ready, Harbor-finish, and post-health
teardown times. Preserve the server, Harbor, trial, and lifecycle results under
`/mnt/vm_8tb/b70/evals`.

RESULT -> the 0.90 attempt never reached Harbor. Each rank allocated 10.60 GiB
of BF16 K/V cache for 1,112,192 tokens and left only 3.22 GiB/card before graph
capture. Kernel evidence at 21:14:54 UTC reported about 58 GiB `gpu_active`,
global OOM, and killed the user dbus, user systemd, and rank-1 SGLang scheduler.
That is what closed the user's tmux session. Both card probes and the compiled
collective passed immediately afterward without reset or reboot.

RESULT -> memory fraction 0.70 allocated 443,392 BF16 KV tokens per rank, left
9.67 GiB/card after graph capture, and became healthy with exact identity in
165 seconds. Sixteen sampled replay-500 re-instantiation markers appeared. The
server crossed the former approximately 17,664-token `linear_stream.h:90`
failure and remained healthy through a maximum logged live sequence of 42,112
tokens. There was no scheduler death, Level Zero assertion, abort, OOM, or
engine-dead marker. Server-log SHA256 was
`449d64fa9c67731fceba3885b59ecbed6734847951ae3da5998dbf2ce3941f36`.

RESULT -> Pi performed many valid structured reads, edits, and bash calls but
spent too long debugging source-map VLQ rewriting. Harbor stopped the agent at
exactly 1,800 seconds with 712,735 input and 30,290 output tokens. The official
reward was 0.0 with `AgentTimeoutError`. Harbor job wall was 2,074 seconds
(34m34s), server-start through Harbor finish was 2,245 seconds (37m25s), and
server-start through graceful teardown plus post-health was 2,319 seconds
(38m39s). Harbor log, trial result, trajectory, and lifecycle SHA256 values
were `c6df00e34a1b5dcb4679aee4ff1378ff24a0eb5ba65ed932cbd4a149ba6c9060`,
`d07097f72f3d65b3a90f31669fb8b88774e86c19594f112db9656fa49e2ee615`,
`b64c4921dcdcf98d1dc0d1e39eb7103af5998579499a3664dc2e59a0f7d31a9f`,
and `b7ea7f360e208f8b6924766ca8ff21d8dc7ae72730a1ea56f0acb5c3bff82a91`.
Graceful teardown, both card probes, and the compiled P2P-off collective passed;
both leases are free and no GPU server remains.

VERDICT -> qualify reclaim500 as the isolated fix for Ornith's prior long-agent
graph replay failure, but reject the current Ornith Pi/xhigh recipe on this
task because model verbosity consumed the official time budget and scored
zero. Use memory fraction 0.70 for 65K TP2 agent work; 0.90 is host-unsafe.
Retain Ornith as the fourth matched campaign arm, but pilot all four arms before
committing multiple days to the full 74-task set. The new campaign driver and
summarizer record score plus Harbor and full lifecycle wall time.

### 2026-08-28n - Qwen W8A8 TB3 pilot faults at 17K context and scores zero

CONFIG -> exact refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
and Qwen3.8-27B compressed-tensors W8A8 GPTQ checkpoint. The matched campaign
arm used TP2, P2P off, source-default eager collectives, FULL target decode
graph at batch one, BF16 KV, qwen3 reasoning, qwen3_coder tools,
strict-thinking cap 4096, maximum one request, 65,536 context, and memory
fraction 0.70. Harbor 0.22.0 ran official Terminal-Bench 3.0.0 task
`bun-sourcemap-leak` with Pi 0.84.3 at xhigh and the retained concise prompt.

COMMAND -> run
`INCLUDE_TASK=terminal-bench/bun-sourcemap-leak N_TASKS=1 STAMP=20260828-bun-pilot evals/terminalbench/run_arm.sh qwen-w8a8`
inside the driver's whole-box lease. Require clean per-card and compiled
two-rank P2P-off health before serving, exact `/v1/models` identity, Harbor
completion even if the endpoint fails, teardown, and the same post-health.
After any TP2 failure, run the non-reboot `bin/xe-reset` recovery ladder and
repeat both health probes.

RESULT -> startup took 140 seconds. Each rank allocated 253,696 BF16 KV tokens
with 3.87 GiB K plus 3.87 GiB V and retained 9.68 GiB after graph capture. Pi
read the app, produced a substantial release-script rewrite, caught and fixed
its first template-string error, and passed the base runtime and provenance
checks. The last completed model response reported 16,875 input plus 434
output tokens. During the following response at 22:44:07 UTC, card
`0000:0b:00.0` reset both CCS and BCS engines and reported two unsuccessful GPU
virtual-memory faults. The SGLang container died, and Pi ended after three
endpoint connection errors. This was not a host OOM and did not close the new
tmux session.

RESULT -> Harbor preserved and graded the edited task. It scored 0.0: 24 of 36
tests passed, 10 failed, and two errored. The dynamic application variants
exposed incorrect private-module stubbing and path handling, so the answer was
not merely denied a score by the endpoint crash. Agent execution was 13m51s,
Harbor wall was 18m23s, server-start through Harbor finish was 20m49s, and
server-start through teardown plus post-health was 21m52s. The job used 114,413
input and 11,801 output tokens across requests. Job result, trial result,
trajectory, lifecycle, and server-log SHA256 values were
`1303b93337ed0d5c40715b10368f8057c0dc5b3550fd2bab255eafa8582e4368`,
`360c647988d1acc91c7c3b5b36a6ecbc113d0416b27fdf2097f809f6c619dfc5`,
`03317a82c2dc22a58ba700f050d0259739a3b0b435bb03e3305bbaf895851714`,
`d3170fb3529cce8cdbb107a1381c4757398da7fce85230e36d74c4ea4b0fb99a`,
and `3fa7ce7e4f54d128a93b407d2f7db7b59a39801e643906bb8eb4b3a1e3acd7ec`.

RESULT -> immediate teardown health passed on both cards and the compiled
two-rank P2P-off collective. The mandated recovery then re-bound both xe
endpoints without rebooting; both card probes and the compiled collective
passed again under the unchanged boot ID
`e2d5777d-f6bb-4d92-a718-0fb07ae17919`.

VERDICT -> reject this Qwen W8A8 FULL-graph configuration for the 65K agent
campaign. Its one-task score is zero and, independently, its endpoint is not
stable through a roughly 17K-token tool conversation. Keep the result in the
matched pilot table, retain BF16 KV and memory fraction 0.70, and proceed to
the NVFP4 and GPTQ INT4 pilots before deciding whether a safer graph mode is
worth a separate Qwen W8A8 diagnostic.

### 2026-08-28o - Qwen NVFP4 TB3 pilot aborts FULL replay at 19K context

CONFIG -> exact refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
and Qwen3.8-27B RadixArk NVFP4 checkpoint. The matched campaign arm used TP2,
P2P off, source-default eager collectives, FULL target decode graph at batch
one, BF16 KV, qwen3 reasoning, qwen3_coder tools, strict-thinking cap 4096,
maximum one request, 65,536 context, and memory fraction 0.70. Harbor 0.22.0
ran official Terminal-Bench 3.0.0 task `bun-sourcemap-leak` with Pi 0.84.3 at
xhigh and the same concise prompt as every other arm.

COMMAND -> run
`INCLUDE_TASK=terminal-bench/bun-sourcemap-leak N_TASKS=1 STAMP=20260828-bun-pilot evals/terminalbench/run_arm.sh qwen-nvfp4`
inside the driver's whole-box lease. Require clean per-card and compiled
two-rank P2P-off health, exact served identity, Harbor completion after any
endpoint failure, teardown, and matched post-health. Apply `bin/xe-reset` and
repeat both probes after a failed TP2 serve.

RESULT -> startup took 117 seconds. Each rank allocated 378,240 BF16 KV tokens
with 5.77 GiB K plus 5.77 GiB V and retained 9.56 GiB after FULL graph capture.
Pi used tools early but repeatedly generated multi-minute plans before simple
build experiments. The 4,096-token strict-thinking cap was only soft: the model
continued its analysis as visible text. Decode began around 30 tok/s and
remained about 27 tok/s near the failure boundary.

RESULT -> the endpoint crossed the W8A8 arm's approximately 17K-token failure
boundary, but at 19,328 live tokens both ranks aborted at Level Zero
`linear_stream.h:90` while replaying the XPU FULL graph. The scheduler processes
exited with signal 6 and the SGLang parent shut down after its five-second crash
diagnostic delay. There was no kernel engine reset, GPU VM fault, or host OOM.
Pi ended after three connection errors without ever editing the task.

RESULT -> Harbor graded the preserved baseline and scored 0.0: 17 of 36 tests
passed and 19 failed. Agent execution was 10m15s, Harbor wall was 14m46s,
server-start through Harbor finish was 16m48s, and server-start through teardown
plus post-health was 17m52s. The job used 59,760 input and 9,497 output tokens.
Job result, trial result, trajectory, lifecycle, and server-log SHA256 values
were `d8d94bfb972d1589e29d0f13e47ef9c08640a9aadbf997ad68b868983dfdf63b`,
`7ec7bd091c8890e0dafb52c1b726f3b164ebca7c5a15ea25ce6be4679d7406e6`,
`480e091d3b08569ed2b786a7faa8c7f02c0c06eca2ad832a789de4def80e556d`,
`fba21206504af31aacfc5d31ba87368012b083dedd403b4423350d4d0862db56`,
and `1f71b02f53ca34a474187e7536875665df3cb78fc558609091ae16cc2e913b31`.

RESULT -> immediate teardown health passed on both cards and the compiled
two-rank P2P-off collective. The mandated recovery re-bound both xe endpoints
without rebooting; both card probes and the compiled collective passed again
under unchanged boot ID `e2d5777d-f6bb-4d92-a718-0fb07ae17919`.

VERDICT -> reject this Qwen NVFP4 FULL-graph configuration for the 65K agent
campaign. It is more graph-replay-stable than the W8A8 arm on this trajectory,
but still aborts well below the configured context and its agent policy spent
most of the available time planning rather than implementing. Keep its zero
score and full failure time in the matched comparison; continue with the TP1
GPTQ INT4 arm, which avoids this TP2 SGLang FULL-replay path.

### 2026-08-28p - Qwen GPTQ INT4 TB3 pilot aborts PIECEWISE and scores zero

CONFIG -> pinned image
`vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`,
vLLM `0.27.2rc1.dev77+gac7509e2b`, and the local Qwen3.8-27B GPTQ INT4
group-128 checkpoint. The matched campaign arm used TP1 on card 0, PIECEWISE
graph, MTP4 with the accepted draft-only LM-head INT4 patch, BF16 KV,
qwen3 reasoning, qwen3_coder tools, maximum one request, and 65,536 context.
Harbor 0.22.0 ran official Terminal-Bench 3.0.0 task `bun-sourcemap-leak`
with Pi 0.84.3 at xhigh and the same concise prompt as every other arm.

COMMAND -> start with the qualified short-context launcher's full 65,536-token
batched-token limit, then change only the prefill compile window if compilation
prevents startup. If BF16 KV still cannot fit the 65,536-token model window,
raise only the vLLM memory-utilization bound. Require exact served identity,
preserve Harbor grading after an endpoint failure, stop the container, and run
card-0 plus compiled two-rank P2P-off post-health inside the whole-box lease.

RESULT -> the first startup used 65,536 batched tokens and failed before KV
sizing when compilation attempted a 4.25 GiB allocation. Server-log SHA256 was
`1efaf70290b25378d6c964e80c63c84ca7becd2c662aedcf9a154ddb056c2f04`.
Restricting the compile/prefill window to 16,384 tokens completed compilation,
but memory utilization 0.75 left 1.0 GiB for KV while one 65,536-token request
required 5.07 GiB; estimated maximum model length was only 2,496 tokens.
Server-log SHA256 was
`0fad376683170524c85cb9b8d0866d0107cb989c6f51a5bde3008bb76608aca6`.
Both attempts failed closed before Harbor and their full lifecycle times were
289 and 246 seconds. Card and compiled collective health passed after each.

RESULT -> memory utilization 0.90 retained BF16 KV and made 6.44 GiB available
for 82,965 tokens, or 1.27 times the configured 65,536-token request. Startup
took 95 seconds and PIECEWISE graph capture used 0.91 GiB. The first substantive
model response spent 9,067 output tokens on a plan before running the baseline
release. It then inspected the leaking artifacts but made no edit. During the
next model request, at about 28 percent KV-cache occupancy, Level Zero aborted
at `linear_stream.h:90`; the engine core died and the API shut down cleanly.
There was no kernel engine reset, GPU VM fault, or host OOM.

RESULT -> Harbor preserved and graded the unchanged task. It scored 0.0 with
17 of 36 tests passed and 19 failed. Agent execution ended with the server at
7m57s; Harbor job wall was 12m25s, server-start through Harbor finish was
14m04s, and server-start through teardown plus post-health was 15m00s. The job
used 21,753 input and 9,459 output tokens.
Job result, trial result, Pi transcript, lifecycle, and server-log SHA256 values
were `c0a4f7bbbd0048584771f86a639711dd7a79088195904434f88a282bf240d51a`,
`bd4ff88b0b98332de69510838371284b808935a6cf25253a5d929ab7610e336b`,
`5901bd1ea7b2c6fe47a8c4e6cf3c3ee71554ccbc9c001f69c6d00a673e94ff3e`,
`4f90a7de40b2bf461d0ca2ccff56ee69de6c869599aa9c16616cdefd52e970cb`,
and `cf30912cf4d54629dcf38a5d3c2d9f529dbf2e255f87b68e0515688141badf48`.
Card-0 and compiled two-rank P2P-off post-health passed.

VERDICT -> reject the current GPTQ INT4 MTP4 PIECEWISE configuration for the
65K agent campaign despite its qualified short and 2K-token serving results.
Its official zero is an unchanged-baseline score caused by endpoint loss, and
the same Level Zero command-stream failure class now spans both vLLM PIECEWISE
and SGLang FULL long-agent runs. Keep BF16 KV, the 16K prefill window, and the
0.90 fit result as controls, but require a graph-safe long-context recipe before
running more official tasks.

### 2026-08-29a - Qwen W8A8 reclaim500 survives TB3 but xhigh times out

CONFIG -> exact refreshed SGLang image
`b70-sglang-xpu-int8-runtime@sha256:adc915d266eaa74f7bea164d97cb7870b04dd7eb4c613952c56f4fbff1584a78`
and Qwen3.8-27B compressed-tensors W8A8 GPTQ checkpoint. The diagnostic changed
the rejected matched arm's FULL decode graph to the previously qualified TP2
breakable backend and enabled executable re-instantiation every 500 replays.
It retained P2P off, source-default eager collectives, target-only decode,
BF16 KV, memory fraction 0.70, maximum one request, 65,536 context, qwen3
reasoning, qwen3_coder tools, strict-thinking cap 4096, Pi 0.84.3 xhigh, and
the same concise prompt and official `bun-sourcemap-leak` task.

COMMAND -> add a separately named `qwen-w8a8-reclaim500` campaign arm without
changing the historical FULL control. Run it inside a whole-box `bin/gpu-run`
lease with exact identity, per-card and compiled two-rank P2P-off pre-health,
Harbor through its terminal state, graceful teardown, and the same post-health.
Watch the server and kernel for the old approximately 17K fault, Level Zero
abort, replay-reset markers, throughput decay, OOM, or engine death.

RESULT -> startup took 146 seconds. Each rank loaded 14.33 GiB of weights,
completed breakable batch-one capture in 11.85 seconds, and retained 9.67 GiB.
Both ranks logged graph reclaim activation; the bounded log contained sixteen
first-500-replay re-instantiation markers across their graph segments. Decode
continued around 13-14 tok/s after reclaim. The server crossed the original
W8A8 hardware-fault boundary and the NVFP4 19,328-token abort boundary, then
remained healthy through a maximum logged live sequence of 26,368 tokens when
Harbor cancelled the final response. There was no Level Zero abort, scheduler
death, kernel engine reset, GPU VM fault, or host OOM. Server-log SHA256 was
`6804d13b8ce22a273940e32f1e77635b4bf2706934b65d75f54fe6406536d5df`.

RESULT -> the runtime fix did not fix agent efficiency. Pi used eight inspection
calls before editing, spent most of its budget on long plans, and finally wrote
a 14,649-byte replacement release script. Its only post-edit test found a `TypeError`
because the generated code called `.filter()` on a `Set`; the timeout arrived
before repair. Harbor stopped agent execution at exactly 1,800 seconds and
assigned official reward 0.0 with `AgentTimeoutError`. The job used 80,950
input and 20,144 output tokens. Harbor wall was 34m32s, server-start through
Harbor finish was 37m03s, and server-start through teardown plus post-health
was 38m17s. Job result, trial result, Pi transcript, and lifecycle SHA256 values
were `6432762b2f653b00ce6adde092a4da5cfc7e4d87aa7c477b74ac6af4d863aabe`,
`d49828c36c0f5ff19af6e6f0636fb6a855d22c51245883a52ed5f0dc43067cb9`,
`8da842a90d952c03439caf9f981dc4160382b591bf6dfbd2dc2ed55455b77ced`,
and `49fc18a050f0777e81c81fec2e1e06d7ae91e944eaf9b524ec37ab2cd495dba6`.
Graceful teardown, both card probes, and the compiled P2P-off collective passed.

VERDICT -> qualify breakable plus reclaim500 as the isolated long-agent runtime
fix for Qwen W8A8, replacing FULL for future 65K diagnostics. Reject the common
Pi/xhigh policy for this task: Qwen and Ornith both survived it but exhausted
the official budget through verbosity and scored zero. Do not interpret this
38m17s timeout as successful task speed. Before expanding beyond one task,
run a matched lower-thinking or hard-output-bound policy on the two stable
SGLang arms and require a nonzero result.

### 2026-08-29b - Qwen W8A8 no-thinking 4K cap ends before edit

CONFIG -> the qualified Qwen W8A8 breakable-reclaim500 BF16-KV runtime from
2026-08-29a. The new agent-policy arm changed Pi from xhigh to off, replaced
the xhigh-specific prompt with a matched concise prompt, and limited each model
response to 4,096 tokens. Model, quant, serving, task, and 65,536 context were
otherwise unchanged.

COMMAND -> run the same official `bun-sourcemap-leak` task under the whole-box
lease, preserve Harbor grading, and require normal teardown plus card and
compiled P2P-off collective post-health.

RESULT -> the policy sharply reduced early overhead: four focused inspection
tool calls completed before the first implementation response. That response
then reached the 4,096-token output bound and Pi settled without issuing an
edit. Harbor graded the unchanged baseline at 17 of 36 tests and reward 0.0.
Agent time was 6m26s, Harbor wall 10m51s, and full server-start through
post-health time 14m19s. The job used 15,480 input and 4,283 output tokens.
Job result, trial result, Pi transcript, lifecycle, and server-log SHA256 values
were `73cfe8d92fbf1061120b9544b4497d378ab8fd94569711d3d02180f7a021f4e4`,
`f1111cd7896b9624024c84088cf476b865120933dbfab67ec5e3696a373b8492`,
`905028b657def2b03667ffa5c1057ed19f82d3e15ea80d040aceb339ab85762d`,
`bcd9047e3095d32aad81e970f561f6ef79cae642c5ce0e805d56ed4ea4620ffa`,
and `cfc9929296d098dc870711e1e521beaf57d5116015152215d9b8e6f027e93d16`.
The server stopped normally and card plus collective post-health passed.

VERDICT -> reject the 4,096-token hard cap: it converts verbosity into premature
agent termination rather than a completed task. Retain thinking off as a
promising efficiency lever, but give the implementation turn 8,192 tokens in
the next Qwen pilot. Do not spend an Ornith run on this rejected 4K policy.

### 2026-08-29c - Terminal-Bench campaign audit finds two validity defects

CONFIG -> read-only audit of Pi 0.84.3, the retained Terminal-Bench adapter and
runner, all four arm launchers, preserved job transcripts and server logs, the
74 task manifests, and the current campaign summary. No endpoint was started
and no GPU was touched.

COMMAND -> compare the adapter metadata with Pi 0.84.3's installed
`getSupportedThinkingLevels`, `clampThinkingLevel`, and qwen-chat-template
payload construction. Cross-check the GPTQ launch command and runtime-reported
model and KV dtypes. Review stop-reason reporting, total-time boundaries,
per-arm graph failures, task GPU requirements, and configured timeout sums.

RESULT -> `thinkingLevelMap` set `off` to null. Pi treats a null mapping as
unsupported, clamps the requested off state upward, and sends
`chat_template_kwargs.enable_thinking=true`. The 2026-08-29b transcript's
4,096-token thinking block confirms that the job was native thinking with a
hard cap, not true thinking-off. Its conclusion about a true-off 4K policy is
invalid and must not guide another run before a payload oracle passes.

RESULT -> the Qwen GPTQ INT4 launcher hard-coded `--dtype float16` and left KV
dtype on auto. Its preserved runtime log reported `dtype=torch.float16,
kv_cache_dtype=auto`. The retained GPTQ fit, exactness, speed, and
Terminal-Bench results were FP16-KV results despite BF16 served IDs and
lifecycle metadata. Requalification must start at BF16 target-only, using
`--dtype bfloat16` and a runtime assertion of the observed cache dtype.

RESULT -> Qwen W8A8 and Ornith W8A8 have stable breakable-reclaim500 long-agent
runtimes at BF16 KV and memory fraction 0.70, but need a real thinking-off
policy qualification. Qwen NVFP4 FULL and Qwen GPTQ PIECEWISE remain rejected
for long agents. Four tasks require H100 environments, so the B70-local scope
is a labeled 70-task subset. Across all 74 manifests, agent timeout ceilings
sum to 201.69 hours per arm; agent plus verifier ceilings sum to 226.17 hours.

VERDICT -> block campaign relaunch until the Pi off/xhigh payload oracle,
policy-dependent strict-thinking configuration, observed-KV reporting, final
Pi stop-reason classification, endpoint-before-teardown health, and full
pre-health-through-post-health timing are implemented. Then calibrate true off
at 8,192 tokens on Qwen W8A8 breakable-reclaim500, transfer the matched policy
to Ornith, port breakable-reclaim500 to NVFP4 with eager fallback, and qualify
GPTQ BF16 target-only eager before reintroducing MTP. Use resumable matched
shards and the reporting contract in `evals/terminalbench/CAMPAIGN_RELAUNCH.md`.

### 2026-08-29d - Neural.Download/XeCores audit and serving roadmap recorded

CONFIG -> read-only synthesis of the current repository evidence, three
independent audits of Neural.Download and XeCores plus their linked source and
recipe repositories, the four-arm Terminal-Bench state, and the user's product
requirement to compare TP1, TP2, and two independent TP1 replicas as DP2. No
endpoint was started and no GPU was touched.

COMMAND -> normalize external results by model, quant, backend, target/KV
dtype, topology, graph mode, MTP depth, prompt shape, concurrency, and evidence
quality. Cross-check the claimed mechanisms with the local graph/runtime,
collective, recovery, and Terminal-Bench records. Write a dated evidence ledger
and a separate dated, gate-driven experiment matrix without changing prior
results.

RESULT -> `docs/20260829_neural_xecores_deep_dive_and_campaign_state.md`
records the complete literature synthesis and local campaign handoff. It
distinguishes Steve's graph-enabled dense-Qwen TP scaling from his negative
eager controls, records XeCores as measured TP1 evidence rather than TP2
evidence, and captures draft S+M1, prefix reuse, true Pi thinking-off,
model-specific MTP, collective completion, graph-boundary, topology, and
evidence-quality findings. It also records the current stable and rejected
states of the four Terminal-Bench arms and the invalid historical thinking/KV
labels.

RESULT -> `docs/20260829_local_serving_research_roadmap.md` defines harness,
source-oracle, per-model, graph, cache, MTP, topology, long-context, and
Terminal-Bench matrices. Every fitting one-card recipe receives TP1 and DP2
qualification; matched TP1/TP2 cells isolate scaling; final Pi tournaments
compare single-task time and two-user tasks per wall hour. The plan preserves
BF16 KV, P2P-off production safety, identity, target-exactness, lifecycle,
health, recovery, and local-70 reporting gates.

VERDICT -> use the two dated documents as the next-session campaign handoff.
Repair the harness and observed-dtype evidence before another official pilot,
secure a score-completing long route for each model/quant, then let matched
Terminal-Bench evidence select the TP1, TP2, or DP2 local-serving winner for
each workload. Do not assume TP2 wins when TP1 fits twice, and do not call a
DP2 product win a TP scaling result.

### 2026-08-29e - Roadmap objective corrected to single-stream Terminal-Bench

CONFIG -> user clarification after the first roadmap draft. TP1 and TP2 are
both candidates, but the next campaign's objective is the best fast, robust,
highest-scoring single Pi decode stream on Terminal-Bench 3.0.0. DP2 is only a
possible later concurrency benefit if a winning recipe fits one card.

COMMAND -> revise the dated evidence ledger and roadmap so TP1 and TP2 receive
equal single-stream qualification, remove DP2 experiments and the two-user
tournament from the active matrix, and preserve DP2 only as a deferred
post-selection deployment note.

RESULT -> the active matrix now ranks recipes by Terminal-Bench score and
normal completion first, then uses total task and machine time to distinguish
the speed of viable high-scoring routes. Server TTFT, prefill, decode, cache
reuse, MTP acceptance, failures, and health remain explanatory evidence. No
DP2 test consumes time before the single-stream recipe winner is selected.

VERDICT -> run the next campaign as a C1 recipe tournament across TP1 and TP2.
If the winner is TP1, evaluate DP2 separately afterward as a local-serving
concurrency bonus, not as part of model/recipe selection.

### 2026-08-29f - Terminal-Bench H01-H03 policy contract repaired

CONFIG -> Harbor 0.22.0 with exact Pi 0.84.3, the retained custom Qwen
chat-template adapter, no model endpoint, and no intended GPU work. True off
uses the concise-off prompt and an 8,192-token response cap. Xhigh retains the
concise prompt, 16,384-token response cap, and recorded 4,096-token private
thinking cap.

COMMAND -> add a real Pi subprocess oracle backed by a local mock OpenAI SSE
endpoint, add unit coverage for supported thinking levels and launcher policy,
and run `PI_0843_BINARY=/tmp/b70-pi-runtime-0.84.3/node_modules/.bin/pi
evals/terminalbench/phase0_preflight.sh`.

RESULT -> all four metadata/policy tests passed. The captured off payload set
`enable_thinking=false` and `preserve_thinking=true`; xhigh set the first value
true and preserved thinking. Neither payload contained `reasoning_effort`.
The launcher oracle resolved off to an empty `THINKCAP` and xhigh to 4096.
Intermediate levels fail closed. The result log SHA256 was
`d6ee63f433829770e43613a1a19583ec7db84f7092075a862f6f947ba5ac0e77`.

VERDICT -> H01, H02, and H03 pass. Keep official GPU pilots blocked until
H04-H07 also pass. The historical 4K job remains native-thinking evidence and
is not reclassified.

### 2026-08-29g - Aborted policy-oracle routing mistake

CONFIG -> Qwen3.8 W8A8 TP2 breakable-reclaim500, BF16 target/KV, P2P off,
65,536 context, memory fraction 0.70, target-only, and whole-box `bin/gpu-run`.
This was not a planned model experiment: the first `--print-config` check
incorrectly entered the normal lease path.

COMMAND -> interrupt the transaction, allow graceful server teardown, run card
and compiled P2P-off collective post-health, then terminate the accidentally
started Harbor setup before agent execution. Fix print-config to bypass the
lease and make INT/TERM exit through the cleanup trap instead of continuing.

RESULT -> both initial card probes and the compiled collective passed. A server
was briefly started, then stopped without a request or benchmark. Card and
compiled collective post-health passed. Harbor created an incomplete 74-task
job with zero completed trials before termination; it is not evaluation
evidence. Server-log and lifecycle SHA256 values were
`fac1e46095605f3b7a770546bdb9b3f2e716392717bfbd19c1e6bb5fa6e33b88` and
`272e1c52a8918875b8be987a1348ee9096594e21dc060e46b0cad4f775d28b8d`.

VERDICT -> reject the transaction as an experiment and retain only its cleanup
evidence. The corrected non-GPU print path now exits before the lease. Do not
use the incomplete job or its invalid lifecycle clock as campaign data.

### 2026-08-29h - Terminal-Bench H04-H07 evidence contract closed

CONFIG -> no model endpoint and no GPU. Harbor's Python environment, exact Pi
0.84.3, the preserved Terminal-Bench 3.0.0 task tree and five retained Pi
trajectories were used. Runtime identity fixtures were the accepted Qwen3.8
W8A8 SGLang BF16 log and the historically mislabeled vLLM GPTQ FP16 log.

COMMAND -> add fail-closed runtime identity and lifecycle parsers, replay the
preserved Pi session JSONL, generate and validate the deterministic local-70
manifest, and run `evals/terminalbench/phase0_preflight.sh`. Exercise lifecycle
ordering against a real local mock HTTP health endpoint. Feed both retained
runtime logs through the dtype validator and require the SGLang control to pass
and the vLLM control to fail.

RESULT -> 20 unit tests passed, the real Pi payload oracle passed for off and
xhigh, and the direct local-70 validation passed. The preflight log SHA256 was
`1acd9ed2d0758c6c398bf650dab486dd7b1816946ca530016f68656b3b872124`.
The SGLang control recorded target and observed KV dtype as BF16. The vLLM
control failed on target FP16 and missing observed BF16 KV evidence. The five
trajectory replays distinguished normal stop, length, Harbor timeout, endpoint
error, unique tool counts, confirmed source edits, and post-edit test state.

RESULT -> the tracked manifest contains exactly 74 source tasks, excludes only
`exam-pdf-eval`, `fp8-rmsnorm-gemm`, `jax-speedrun-gpu`, and
`math-eval-grader`, and locks 70 local tasks into fourteen stable five-task
shards. Its local-task digest is
`f42c7d0ac925d58d603dfd8f40ceebaac610d376ef1fa48bfe54f760a0970d3d`
and file SHA256 is
`b67a6fd54c4e3db8020f54891966b460230baeb8feec324faef21786526f3196`.
The runner now starts its clock before pre-health, validates `/v1/models` and
observed dtype before Harbor, checks the live endpoint and fatal markers before
teardown, requires endpoint disappearance, runs post-health, and closes the
clock afterward.

RESULT -> the official Qwen3.8-27B model card at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` describes xhigh as the default
deep policy and reports a 32,768-token ceiling for QwenSWEBench. This supports
a bounded higher-cap experiment but does not prove a Terminal-Bench benefit.
Local trajectories show that long native-thinking turns can also delay edits
until the 1,800-second timeout. The roadmap therefore adds one 8,192 private-
thinking-cap comparator instead of changing every launcher default.

VERDICT -> H04, H05, H06, and H07 pass; together with 2026-08-29f, Phase 0
H01-H07 is closed. Begin M01 source accounting and the isolated M02 P2P-off
collective oracle before porting Steve mechanisms. Keep true off uncapped at
the server, label local xhigh as native thinking because the endpoint does not
accept Qwen's `reasoning_effort`, and change policy caps only through matched
Terminal-Bench arms.

### 2026-08-29i - M01 Steve completion and state source ledger

CONFIG -> read-only comparison of Steve's qualified Qwen3.8 FP8 base at vLLM
`ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`, XPU kernels
`1e90ffa672ba02f17a909da11838a4c55b199783`, retained current vLLM
`44fc8fde09fc311d3099dab10366b672d9142ea4`, and the three published patch
hashes. No GPU, binary import, archive dependency, or source mutation.

COMMAND -> map explicit collective completion, GDN recurrent-state mutation,
cache binding, deterministic 256-row B/A projection, exact two-row RMSNorm,
and deterministic Inductor to retained line-level APIs. Classify every item as
equivalent, missing, or requiring an API-aware port in
`docs/20260829_steve_completion_state_port_ledger.md`.

RESULT -> eager async all-reduce plus `Work.wait()` exists as an opt-in branch,
but retained compiled paths bypass it. Compiler-visible recurrent state and
the old cache-binding hook are absent, and the current cache API requires a
deliberate port. The retained four-row B/A and RMSNorm diagnostics are not
equivalent to Steve's fixed-256 B/A and exact two-row publisher-MTP1 repairs.
No retained candidate launcher enables deterministic Inductor.

VERDICT -> M01 passes as a source-accounting gate, not as acceptance of any
patch. Close M02 and M03 on isolated BF16 P2P-off collectives first; then port
state visibility/cache binding, fixed-256 B/A, and exact two-row RMSNorm as
separate mechanisms before a combined target/MTP model qualification.

### 2026-08-29j - M02 P2P-off compiled collective boundary and pass

CONFIG -> kernel 7.1.0, host Compute Runtime 26.22, pinned vLLM image
`f01e24f6`, vLLM `ac7509e2b`, PyTorch 2.13.0+xpu, oneCCL `89438cc`, two B70
ranks, and `CCL_TOPO_P2P_ACCESS=0`. Exact BF16 shapes were all-reduce
`[1,5120]`, all-reduce `[4,5120]`, and all-gather input `[4,2560]`. Every
collective fed an immediate multiply-plus-add consumer.

COMMAND -> under one whole-box `bin/gpu-run` lease, run eager direct, compiled
functional plus `wait_tensor`, and compiled XPUGraph replay. Flush per-rank
entry/return JSONL with monotonic call IDs. Use three fresh containers and
compile caches, tear down each, and run card plus compiled P2P-off collective
health between lifetimes.

RESULT -> the first functional all-gather graph capture failed on both ranks
with `wait method cannot be used for an event associated with a command graph`.
Both all-reduce shapes had already passed eager, compiled, and 16 graph
replays; all-gather had passed eager and compiled execution. Teardown and
post-health passed. The required non-reboot rebind reset then completed with
clean card and collective health.

RESULT -> keeping compiled all-gather opaque through a direct custom op removed
the illegal Inductor event wait. Three fresh lifetimes passed. Each rank logged
102 matched calls per lifetime, every numerical comparison after the consumer
was exact, and no call remained open. All three teardowns and all intervening
health checks passed. Combined lifetime result SHA256 values were
`041b5d57729061b1650b8f36c6139488ff95edc01f4e50b45ff267485f26acf6`,
`e74ac321adea9565217dd33b75d8db210993f200588ff02fe13f9f1b43746be8`,
and `f3c59fbab483bffe1a364b9187358074ff08bed18e9216fde80b525e68dabe26`.

VERDICT -> M02 passes, with a route-specific constraint: functional-wait
all-reduce is graph-safe, but all-gather graph replay requires an opaque direct
boundary. Proceed to M03 blocking c10d versus `async_op=True` plus
`Work.wait()` as an isolated P2P-off all-reduce A/B. Do not claim endpoint
speed or accept a Steve model patch from this operator result alone.

### 2026-08-29k - M03 explicit collective completion A/B

CONFIG -> kernel 7.1.0, host Compute Runtime 26.22, pinned vLLM image
`f01e24f6`, vLLM `ac7509e2b`, PyTorch 2.13.0+xpu, two B70 ranks, and
`CCL_TOPO_P2P_ACCESS=0`. The exact BF16 shapes were `[1,5120]` and
`[4,5120]`. Blocking `dist.all_reduce` was compared with `async_op=True` plus
`Work.wait()` in balanced alternating order. Each result fed an immediate
multiply-plus-add consumer before any post-collective XPU synchronize.

COMMAND -> run two warmups and eight measured rounds per mode and shape in
each of three fresh process-group/container lifetimes under one whole-box
`bin/gpu-run` lease. Flush per-rank entry, completion, consumer, and validation
events. Tear down every lifetime and run card plus compiled P2P-off collective
health before, between, and after the matrix.

RESULT -> the first attempt reached exact equality but the evidence-only
fingerprint path failed while converting a nested byte list to `bytes()`. The
container tore down and post-health passed. `bin/xe-reset --method rebind`
completed on the same boot ID with clean card and collective health. Flattening
the byte view fixed the harness without changing the collective path.

RESULT -> three clean rerun lifetimes passed. Each rank completed 40 calls per
lifetime. External validation covered 240 calls and 1,080 flushed events,
strictly increasing per-rank monotonic times, exact blocking/async and
cross-rank fingerprints, matched call signatures, and no unreturned call.
All teardowns and pre/inter/final health gates passed. The sorted 15-file
evidence manifest SHA256 was
`dc19da09ffdcf2504775f574c54e1140616ae7dcc109fdebfd99b0c1c4d29210`.

RESULT -> exploratory host-boundary medians across 48 measured calls per cell
were 184.652 versus 239.362 us from entry through consumer return for blocking
versus async/wait at `[1,5120]`, and 182.628 versus 240.504 us at `[4,5120]`.
These are operator host timings, not device-kernel or endpoint measurements.

VERDICT -> M03 passes as a correctness and completion-ownership oracle. Both
routes safely support the immediate consumer with P2P disabled, but the
explicit route supplies no speed claim or model-patch acceptance. Proceed to
M04 graph-boundary census tooling before deciding whether a matched endpoint
completion-route control is worth running.

### 2026-08-29l - M04 structural census and host-stall classification

CONFIG -> Qwen3.8-27B compressed-tensors W8A8 GPTQ with GDN RTN, SGLang TP2,
P2P off, BF16 target/KV, breakable batch-1 decode graph, reclaim500, 4,096
context, memory fraction 0.75, radix cache off, MTP off, and native SGLang
decode annotations. The accepted structural capture used four decode steps.

COMMAND -> profile after first token, parse paired-rank native decode ranges,
count graph pieces, fences, host waits, submissions, and shaped collectives,
then compare post-first-token throughput with two unprofiled controls. Require
exact rank agreement and a profiled/control ratio of at least 0.75.

RESULT -> all four captured tokens on both ranks had the same signature: 131
graph pieces, 131 fence resets, 131 host waits, 262 submissions, 129 BF16
`[1,5120]` all-reduces, and one BF16 `[1,124160]` all-gather. The 10.1215
profiled tok/s divided by the 14.4349 tok/s control mean was 0.701183, or 29.9
percent loss, so the overhead gate failed. Teardown and card plus compiled
P2P-off collective post-health passed. The census JSON SHA256 was
`1ca603b54d1ce45a4e03ec385ab9a3e24ad1a29e88256ba3dee8ae56f41f7db7`.

RESULT -> a third, two-step attempt started at 06:50:08 UTC during 3.71 GiB of
swap use, active swap churn and reclaim, and a root-NVMe queue depth of 60.91
with 58.87 ms await. It reached TP2 weight loading but never endpoint health or
profiling. The host journal ended at 06:50:10 while the container log continued
to 06:50:34. The same boot already contained a directly observed global-OOM
episode that blocked root jbd2, journald, and Btrfs writeback for 122/245
seconds with about 56.4 GiB `gpu_active`. No final OOM, GPU fault, or crash dump
survived for the new incident.

VERDICT -> retain the exact structural census but keep M04 open because the
overhead gate did not pass. Reject the third attempt as experiment evidence.
Classify the unresponsive host as a likely memory-reclaim/swap/root-journal
stall, not a proven GPU wedge; the exact initiator remains unproven. Require
96 GiB MemAvailable, at most 1 GiB used swap, a 64 GiB no-swap container
ceiling, and persistent memory/PSI sampling before one bounded retry. Do not
relax those safety bounds to make the profile run.

### 2026-08-29m - M04 contained two-step pass

CONFIG -> committed Git identity `b6cc036`, the same Qwen3.8 W8A8 TP2
breakable-reclaim500 configuration, two profiled decode steps, 96 GiB minimum
host MemAvailable, at most 1 GiB preexisting swap, a requested 64 GiB
memory-plus-swap container ceiling, and five-second host memory/PSI sampling.

COMMAND -> on rebooted boot ID `868bc48dece94aa78569d5b6f38da02b`, first
pass both cards and the compiled P2P-off collective. Start one bounded server,
verify exact model identity, run warmup, control A, the profiled request after
first token, and control B, then require exact paired-rank census agreement and
a profiled/control ratio of at least 0.75. Tear down and repeat card and
collective health.

RESULT -> both profiled tokens on both ranks reproduced 131 graph pieces, 131
fence resets, 131 host waits, 262 submissions, 129 BF16 `[1,5120]`
all-reduces, and one BF16 `[1,124160]` all-gather. Profiled throughput was
12.6260 tok/s against a 14.6965 tok/s control mean. The ratio was 0.859115, or
14.1 percent loss, and passed. The census JSON SHA256 was
`41010eeb690c286b2629f2b46360b5c70d2715fa530728384e3c930c51abe144`.

RESULT -> all 48 host samples recorded zero swap. MemAvailable ranged from
123,996,420 to 61,283,424 KiB; memory PSI briefly reached 0.05 at 60 seconds,
then returned to zero. Exact model identity, endpoint teardown, both cards,
and the compiled P2P-off collective passed. The memory-monitor SHA256 was
`a68ee108d0743d4b4012d282493ea0309afd9cdc7aa56c5284ccc8fdd1c68190`.

VERDICT -> M04 passes. Use 131 pieces, 131 waits, 262 submissions, and 130
shaped collective calls as the Qwen3.8 W8A8 breakable TP2 boundary baseline.
Retain the host-safety gates for later Qwen3.8 work and proceed to the P0 W01
corrected long-output baseline; M03 supplies no reason to spend W05 endpoint
time on the explicit async/wait route before that baseline is stable.

### 2026-08-29n - Rejected W01 teardown-harness attempt

CONFIG -> committed W01 protocol `dadddf1`, Qwen3.8 W8A8 TP2,
breakable-reclaim500, BF16 target/KV, 65,536 context, memory fraction 0.70,
maximum one request, P2P off, and a 64 GiB no-swap container ceiling. Result
directory was
`/mnt/vm_8tb/b70/results/w01_qwen38_w8a8/20260829T193528Z/`.

COMMAND -> start fresh server A, verify identity/runtime/cgroup configuration,
capture the eight-prompt twice-per-prompt greedy corpus, then stop it before
the inter-server health gate.

RESULT -> server A and the corpus passed, but `stop_server` declared `label`
and a log path referencing `label` in the same Bash `local` statement. Under
`set -u`, expansion occurred before assignment and aborted both the normal stop
and cleanup paths. No server B or long request started. The corpus SHA256 was
`b5b01782764cc310f828e395e933471e555879cf317f85184915ae53d1fa47ff`.
All 39 host samples used zero swap and the minimum MemAvailable was
65,489,108 KiB.

RESULT -> the first recovery command targeted a mistyped timestamped name and
did not stop the real container. The actual container was then stopped and
removed before the compiled collective began. A fresh post-stop transaction
passed both card probes and the compiled P2P-off collective. The recovered
server-log SHA256 was
`ac1fe6a8eb6c8f341deadc3d63bfad887b6691ccf1d4d3d8ba44a487bd8cd8dc`.

VERDICT -> reject the attempt as W01 evidence. Split the dependent local
assignments into separate statements, retain the corpus only as harness-debug
evidence, and rerun both fresh servers plus the full 50K gate from the start.

### 2026-08-29o - Rejected W01 native-client parameter attempt

CONFIG -> corrected cleanup at Git identity `56d958c`; otherwise the exact W01
configuration and safety gates from 2026-08-29n. Result directory was
`/mnt/vm_8tb/b70/results/w01_qwen38_w8a8/20260829T194241Z/`.

COMMAND -> run the complete fresh server A corpus and teardown, inter-server
card plus collective health, host recovery gate, then fresh server B and its
cross-server corpus. Start the native `/generate` 50K stream only after those
gates pass.

RESULT -> both eight-prompt corpora were repeat-exact and server B matched all
server A hashes with an exact reference contract. Server A tore down normally;
inter-server and final card plus compiled P2P-off collective health passed.
All 93 host samples used zero swap and MemAvailable stayed at or above
65,320,196 KiB. Corpus A/B SHA256 values were
`b5b01782764cc310f828e395e933471e555879cf317f85184915ae53d1fa47ff`
and
`9fd1f1526ea92da9a70fb38f80926985ee54aa8db19a6f829c68fb31c246061d`.

RESULT -> the native request incorrectly included `seed` inside SGLang's
`sampling_params`. The endpoint returned HTTP 200 before its streaming body
raised `TypeError: Unexpected keyword argument 'seed'`; no model prefill,
decode token, milestone, or 50K evidence was produced. Server B remained
healthy, then tore down with no kernel fatal marker. Server B log SHA256 was
`4e9ec2a31d36171b95e3ce1fe5ef76a4d81ee056dca52cb85e769a97ae461efc`.

VERDICT -> reject the transaction as W01 evidence despite the useful corpus
gate. Native SGLang greedy sampling has no seed field; record seed as none,
retain `temperature=0`, remove the invalid parameter, cover its absence in the
mock SSE test, and rerun the full transaction rather than resume at 50K.

### 2026-08-29p - W01 corrected 50K baseline passes

CONFIG -> Git identity `a17eb6a`, Qwen3.8-27B compressed-tensors W8A8 GPTQ
with GDN RTN, pinned SGLang image digest `adc915d266e`, TP2, P2P off, BF16
target/KV, 65,536 context, memory fraction 0.70, maximum one request,
breakable batch-1 decode graph, reclaim500, radix off, and MTP off. The host
gate required 96 GiB available and at most 1 GiB used swap; each server had a
64 GiB no-swap container ceiling. Result directory was
`/mnt/vm_8tb/b70/results/w01_qwen38_w8a8/20260829T195551Z/`.

COMMAND -> pass per-card and compiled P2P-off collective health, start fresh
server A, verify exact identity/runtime/dtypes/resources, and capture the
eight-prompt corpus twice per prompt. Stop A, repeat health and the host gate,
then start fresh server B and require within-server and cross-server exact
corpus hashes. Send one native greedy 50,000-token `/generate` stream with
temperature zero, `ignore_eos=true`, and no unsupported seed. Preserve exact
token milestones, validate the full token array and preserve its SHA256, require
a length finish, and require final 5K/first 5K throughput of at least 0.80. Stop
B, scan server/kernel logs, and repeat card plus collective health.

RESULT -> both fresh-server corpora were repeat-exact and server B matched all
eight server-A completion hashes. Exact served ID, BF16 target/KV, image,
cgroup, P2P-off, breakable, and reclaim500 gates passed. Corpus A/B SHA256
values were `b5b01782764cc310f828e395e933471e555879cf317f85184915ae53d1fa47ff`
and `2740f737bf0e97b9900974e13f96ee69e67eb5fe75249ef9ead4ef4a9aba2163`.

RESULT -> the native stream finished by length with exactly 50,000 completion
tokens. TTFT was 323.074 ms, total response time 3,435.460 seconds, and
post-first-token throughput was 14.5552 tok/s. The first and final 5K windows
were 14.8396 and 14.2652 tok/s; the 0.961298 ratio passed. The full token-array
SHA256 was `01d78ddc5700922abcebc4ef5298df5c98840915eda72dcc3454c014860ca3a1`;
the replay JSON SHA256 was
`4300568c7a2da2d731124bf65284c5f10e40b6dfeb0484011727c5122557e349`.
The log crossed all prior graph-failure boundaries and contained 21 executable
re-instantiation markers without a configured fatal server marker.

RESULT -> all 775 host samples used zero swap. Minimum MemAvailable was
65,245,888 KiB (62.223 GiB), and memory PSI `some`/`full` totals did not move.
Both servers stopped and their endpoints disappeared. The kernel scan had no
OOM, hung task, GPU VM fault, dead engine, wedge, or failed reset marker. Final
card and compiled two-rank collective health passed; their SHA256 values were
`e9f3293cbccc9b9d07d5f665e37f940b1ea0f23da34b50468c052d459b52eeff`
and `93830e24e5201487f24df92401edb4e5054ec720ef64e6239a5d9e0325f5f614`.
Pre/inter card commands also returned success under `pipefail`, but their tee
files were empty because `xpu-health` wrote diagnostics to stderr; future
harness runs now capture both streams.

VERDICT -> W01 passes as a deterministic, contained single-stream long-output
baseline. It does not establish concurrent shelf readiness or attribute speed
to graph/reclaim. Proceed to matched W02 eager, breakable, and
breakable-plus-reclaim500 controls before concurrency qualification.

### 2026-08-29q - Rejected W02 measured-file selection attempt

CONFIG -> committed W02 protocol `ed33d07`, Qwen3.8 W8A8 TP2, P2P off, BF16
target/KV, 65,536 context, memory fraction 0.70, maximum one request, MTP and
radix off, and the 64 GiB no-swap container ceiling. The first arm was eager
with graph and reclaim disabled. Result directory was
`/mnt/vm_8tb/b70/results/w02_qwen38_w8a8/20260829T211047Z/`.

COMMAND -> pass pre-card and compiled collective health, start the eager arm,
verify exact identity/runtime/dtypes/resources, capture the eight-prompt corpus
twice per prompt, then run one 768-token warmup and three exact 2,048-token
native greedy measurements. Require identical text and output-token arrays
before teardown, inter-arm health, and the breakable arms.

RESULT -> eager corpus repeat exactness passed. The three measured streams all
finished by length with 2,048 tokens and the same literal output array, token
SHA256 `c64d070e5b79138c30386367506613066d38b9c9d3759207df71c57bfc021b0f`,
and text SHA256
`a59919ecafbb11ecd0c8fd2c2512fd3831dc4b3569461c70bb6130caf26d64a6`.
Rates were 6.0708, 6.0763, and 6.0467 tok/s, for a 6.0708 tok/s median; all
short flatness gates passed.

RESULT -> the comparison glob `measured_*.json` also selected each
`measured_*.partial.json` checkpoint. Those partial files have no final output
hash, producing a second blank unique value and tripping the fail-closed
within-arm hash count. Cleanup stopped the healthy eager server before either
breakable arm. Final card and compiled P2P-off collective health passed and
the kernel scan had no fatal marker. All 294 host samples used zero swap,
minimum MemAvailable was 65,538,172 KiB, and memory PSI `some`/`full` totals
increased by only 6 each.

VERDICT -> reject the attempt as W02 comparison evidence because only the
eager arm ran. Retain its numbers as harness-debug evidence only. Build the
measured-file list explicitly from repeat indices so partial checkpoints can
never enter exactness comparison, then rerun all three fresh arms from the
start.

### 2026-08-29r - W02 graph comparison closes on target divergence

CONFIG -> repair commit `7a3c2ac`; the matched W02 Qwen3.8 W8A8 TP2 protocol
from 2026-08-29q with eager, breakable without reclaim, and breakable plus
reclaim500 fresh-server arms. Result directory was
`/mnt/vm_8tb/b70/results/w02_qwen38_w8a8/20260829T213708Z/`.

COMMAND -> for each arm verify exact identity/runtime/dtypes/cgroup and P2P-off
state, require the repeat-exact eight-prompt corpus and eager-reference hashes,
then run one 768-token warmup and three 2,048-token native greedy measurements.
Persist literal arrays, require within-arm repeat equality, compare every graph
array to eager, and run card plus compiled collective health between arms and
after final teardown.

RESULT -> all three short corpora passed and every measured stream repeated
exactly within its arm. The stronger native comparison failed at zero-based
token index 24. Eager versus each graph arm differed at 2,011/2,048 positions;
breakable and reclaim500 matched one another at all 2,048 positions. Eager's
token SHA256 was
`c64d070e5b79138c30386367506613066d38b9c9d3759207df71c57bfc021b0f`;
both graph arms produced
`a1856299df39da9652f45a05a9f51475cf28384db6d354756087efa49a71109b`.

RESULT -> diagnostic medians were 6.0420 tok/s eager, 10.0590 tok/s
no-reclaim breakable, and 14.8028 tok/s reclaim500. No-reclaim fell from
11.7182 to 8.8579 tok/s across its three repeats, a repeat-3/repeat-1 ratio of
0.7559. Reclaim500 measured 14.7893, 14.8028, and 14.8269 tok/s, a ratio of
1.0025, while preserving the exact no-reclaim graph array. The analyzer marks
all cross-mode performance attribution unqualified. Comparison JSON SHA256 was
`736322d04b4044e584ddc1603caea372d02b188e40fb8b861005c4f02187ef23`.

RESULT -> all 612 host samples used zero swap; minimum MemAvailable was
62,545,508 KiB. Every inter-arm and final card plus compiled P2P-off collective
health check passed. The kernel transaction had no configured fatal marker.

VERDICT -> close W02 negatively at the target-exactness gate. Reclaim500 is a
real graph replay-stability mechanism, but breakable graph is not eager-target
exact for this native prompt. Cancel the conditional 50K no-reclaim canary and
do not advance cache/MTP work on this graph route before a source-level
numerical/state audit. Move next to the requested official-FP8 vLLM recipe port,
starting with tracked source identities and a P2P-off MTP0 control.

### 2026-08-29s - F01 Neural.Download official-FP8 port ledger

CONFIG -> user-requested Neural.Download Qwen3.8-27B official-FP8 vLLM TP2
candidate recipe, current host kernel 7.1, Compute Runtime 26.22, and the
standing vLLM direct-P2P queue-handoff quarantine. No GPU workload was run.

COMMAND -> resolve the reproduction and Hugging Face remote identities, create
a sparse external checkout under the retained `steve-repro` root, inspect the
MTP0/MTP1 build and launch wrappers, hash all correctness patches, verify the
exact base image is installed, and add the pinned FP8 checkpoint to the live
model manifest.

RESULT -> reproduction source pinned to
`0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7`, vLLM source to
`ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`, model to
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, and base image to
`f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
The exact base image was already local. The W8A16, deterministic GDN,
compiled-state/oneCCL-wait, and packed-RMS patch SHA256 values were
`5db7f1af1156f3490ca91d0d74a07aa2d0909e175eeb1ae23f2074c55c44ff8a`,
`cda7dd1e42a1e0fed2dd34f3936303cb038852a46d8d00786a1c2ebae326f8eb`,
`8f8febcd0abc59bc9b69830827cd7607c00870414b17bd02cf32e2d879858ac8`,
and `ff5b4f33f5596efbad75112bdbbca2bbf81b6c84688476bfa1c9ec9e546c78c4`.

RESULT -> the recipe is not safe to execute verbatim. Its qualified MTP0
command enables direct P2P, its strict MTP1 wrapper hardcodes direct P2P, and
both launchers permit 3 GiB of container swap. The page also labels clean-host
endpoint replay as missing. The local port must preserve source/compiler/model
settings while using the lease, P2P off, no container swap, host admission and
monitoring, and pre/post health.

VERDICT -> F01 passes as a source/identity ledger, not as a runtime
reproduction. Fetch and verify the exact 66-file, 30,866,866,928-byte model;
then build the deterministic MTP0 overlay from tracked source and qualify a
P2P-off graph-off target before MTP1 or any isolated direct-P2P oracle.

### 2026-08-29t - F01 checkpoint, deterministic overlay, and F02 harness

CONFIG -> Neural.Download Qwen3.8-27B official-FP8 reproduction source at
`0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7`, official model revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, pinned vLLM source
`ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`, and pinned base image
`f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
No GPU workload was run.

COMMAND -> download the immutable official checkpoint, verify both direct and
ordinary reads against the publisher manifest, build the deterministic MTP0
overlay in a dedicated external source root, compare every installed overlay
file to the patched checkout, trace the actual console-script import path, and
implement the leased F02 P2P-off/no-swap qualification wrapper.

RESULT -> all 66 Safetensors files and 30,866,866,928 bytes matched. The
basename-sorted aggregate manifest SHA256 was
`82fb8f84fa117c81c3e8639c4675709dfb667d70ddaa2fd097d35fc37d95453a`.
The local overlay image ID was
`dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291`,
different from the publisher's metadata-sensitive `d19f802b...` ID. The four
installed runtime files exactly matched the patched source, with SHA256 values
`f3273ccfb41be44c3c02080c26df10e8b200060366b900d940803f4221224c59`,
`5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d`,
`7c36e4a8dab4bfc06b1d5be2d8466e8cdc94099dd5409424fecc6dd8ffc2c208`,
and `7afb4de8b87d7f180d696f7cadad8b9d48d9ab7b706ae19616425c4f9456fb19`.
Import tracing confirmed that the real `vllm` console launcher selects the
patched `site-packages` tree; an initial interactive-Python observation of the
unpatched workspace tree did not describe the real launch path.

RESULT -> the new F02 harness fixes TP2 P2P off, MTP0, XPU Graph off,
deterministic Inductor, FP16 target/KV, official FP8 plus W8A16 runtime, one
request, 1,024 context, and a 32 GiB no-swap cgroup. It requires the whole-box
lease, 96 GiB host MemAvailable, direct model verification, continuous host
memory/PSI evidence, two fresh servers and compile caches, complete 12-prompt
raw-token equality, independent canaries, graceful teardown, and pre/post card
plus compiled collective health. Static syntax, ASCII, image-ID, and all four
runtime-file gates passed.

VERDICT -> F01 is complete and F02 is ready for its first GPU transaction.
The coming result is a deliberately different P2P-off safety port and cannot
be labeled a reproduction of the publisher's P2P-on 34.031596 tok/s MTP0 or
51.918757 tok/s MTP1 headline.

### 2026-08-29u - F02 official-FP8 P2P-off target closes negatively

CONFIG -> harness commit `7ccff19`; official Qwen3.8-27B-FP8 revision
`017b9c7`, local deterministic overlay `dce80db0`, vLLM `ac7509e2`, TP2,
P2P off, MTP0, XPU Graph off, deterministic Inductor, FP16 target and automatic
KV dtype, W8A16 runtime dispatch, one request, 1,024 context, prefix caching
off, fresh compiler cache per server, and a 32 GiB no-swap cgroup. Result root
was `/mnt/vm_8tb/b70/results/f02_qwen38_fp8_neural/20260829T231100Z/`.

COMMAND -> under the whole-box lease, verify all 66 model files through direct
and ordinary reads, pass card and compiled P2P-off collective health, run the
complete fixed 12-prompt natural suite plus independent canaries on two fresh
servers, stop each gracefully, repeat health, then compare every raw streamed
output-token array across lifetimes and against both publisher MTP0 references.

RESULT -> both cold performance workloads and both independent canary sets
passed. Diagnostic class-balanced rates were 11.351052 and 11.393397 tok/s,
with an 11.372225 median and 0.373% attempt spread. This is 33.42% of the
publisher's 34.031596 tok/s P2P-on MTP0 median. The performance result is not
qualified because only 7/12 complete arrays matched across the local fresh
servers.

RESULT -> mismatches began at zero-based token 392 for
`incident-retrospective`, 303 for `code-review`, 124 for `customer-email`, 169
for `performance-hypotheses`, and 77 for `decision-memo`. Against either
mutually exact publisher reference, local attempts matched 6/12 and 8/12.
Two locally repeat-exact prompts still differed from the publisher at tokens
341 and 160, showing a stable P2P-off route change in addition to the
fresh-lifetime instability. Both canary files had SHA256
`f234e605954b061e7f902eb92dd96739722df5437cadd9b2aceed79b976e45f8`.

RESULT -> both server lifetimes tore down cleanly and every card plus compiled
collective check passed. Across 300 host samples, swap stayed zero, minimum
MemAvailable was 113,409,448 KiB, and memory PSI `some`/`full` totals did not
move. Serving containers used about 7.7 to 8.1 GiB of host RAM. The kernel scan
had no configured OOM, hang, GPU fault, or wedge marker. Persisted negative
summary SHA256 was
`b5b522a45ea7b1b89663f87c9b1388a70300c794ac8762214835d5af563fe0b2`.

VERDICT -> F02 fails at target exactness despite clean infrastructure and
stable diagnostic speed. Do not advance to MTP1, long-agent, concurrency, or
shelf work. Run F02a as a bounded within-lifetime repeat of the five sensitive
prompts before considering the source-default completion diagnostic; keep
direct-P2P full serving quarantined.

### 2026-08-30a - F02a localizes FP8 instability to fresh initialization

CONFIG -> harness commit `e6e3ee9`; the unchanged F02 official-FP8 W8A16 TP2,
P2P-off, MTP0, graph-off, deterministic-Inductor, FP16 target/automatic-KV
route; one fresh compiler cache and server; and the five natural prompts that
diverged across the two F02 lifetimes. Result root was
`/mnt/vm_8tb/b70/results/f02a_qwen38_fp8_neural/20260829T235100Z/`.

COMMAND -> under the whole-box lease, pass card and compiled P2P-off
collective health, run the five prompts twice with raw streamed output IDs and
zero cached prompt tokens in the same server lifetime, compare complete
arrays against one another and both F02 lifetimes, gracefully tear down, and
repeat card plus compiled collective health.

RESULT -> all 5/5 prompt arrays were exact within the third lifetime. The two
diagnostic rates were 11.224449 and 11.095187 tok/s, with an 11.159818 tok/s
median. The third lifetime matched F02 attempt 1 for
`incident-retrospective`, `code-review`, and `customer-email`, but matched F02
attempt 2 for `performance-hypotheses` and `decision-memo`. Both repeats had
the same mosaic. The choice is therefore made at fresh compile/server
initialization and is prompt-specific, not a request-order drift or simple
whole-server A/B route.

RESULT -> all pre/post health passed. Across 149 host samples, swap stayed
zero, minimum MemAvailable was 113,335,124 KiB, and memory PSI `some`/`full`
totals did not move. Container host-RAM use peaked near 7.717 GiB under the
32 GiB no-swap limit. No configured kernel or server fatal marker appeared.
Summary SHA256 was
`a451ab90693be76eaab82bd44812721a24d0ff0edd9638c6e14ae58e1c79d404`.

VERDICT -> F02a passes its diagnostic gate but does not repair F02 or qualify
performance. Continue to F03: hold the P2P-off runtime fixed and compare the
source-default collective-completion route against explicit `Work.wait()`.
MTP, long-context, concurrency, direct-P2P serving, and shelf work remain
blocked.

### 2026-08-30b - F03 source-default completion also changes target

CONFIG -> harness commit `30888bc`; official Qwen3.8 FP8 plus W8A16 TP2,
P2P-off, MTP0, graph-off, deterministic-Inductor, FP16 target/automatic-KV
route; and a one-file image overlay restoring pinned vLLM source-default
synchronous all-reduce. Image ID was `c4fc0d65`; source-default communicator
SHA256 was `527cbfb250760abc62096ee7cd612307b821f21b72dee1687ad866620ec89b6d`.
Result root was
`/mnt/vm_8tb/b70/results/f03_qwen38_fp8_neural/20260830T004500Z/`.

COMMAND -> under the whole-box lease, verify all model and runtime bytes, pass
card and compiled P2P-off collective health, then run the complete 12-prompt
natural suite and independent canaries in two fresh servers with separate
empty compiler caches. Gracefully tear down and repeat health after each
lifetime; compare raw arrays against one another and both local F02 Work.wait
lifetimes.

RESULT -> source-default completion also matched only 7/12 arrays across its
fresh lifetimes. Mismatches began at tokens 392 for
`incident-retrospective`, 303 for `code-review`, 7 for
`architecture-tradeoff`, 127 for `risk-register`, and 479 for
`performance-hypotheses`. The latter two newly unstable prompts were exact in
both F02 Work.wait lifetimes. Across all four F02/F03 lifetimes, seven prompts
had two or three unique outputs and only five were invariant.

RESULT -> diagnostic rates were 11.722245 and 11.577714 tok/s, median
11.649980 tok/s. The apparent 2.442 percent increase over Work.wait is not
qualified because target arrays diverged. Both canaries passed. All pre/inter/
post card and compiled collective health passed. Across 293 host samples,
swap stayed zero, minimum MemAvailable was 113,374,204 KiB, and memory PSI
totals did not move. Container host-RAM use peaked near 7.716 GiB, with no
configured kernel or server fatal marker. Summary SHA256 was
`c7e542cafc6f095dbd9c39975a6f18e79aa9f51a988799f65cf2fc3a917debed`.

VERDICT -> close F03 negatively. Explicit `Work.wait()` is not the root cause,
and source-default completion has no qualified benefit. The separate compiler
caches had identical primary AOTAutograd graph keys but different secondary
artifact keys. Run F03a with two fresh Work.wait processes sharing the cache
created by lifetime 1 to discriminate compilation from later process/runtime
state. Keep all promotion work blocked.

### 2026-08-30c - F03a pins FP8 target selection to compiled artifacts

CONFIG -> harness commit `f33b223`; unchanged official-FP8 W8A16 TP2,
P2P-off, MTP0, graph-off, deterministic-Inductor, FP16 target/automatic-KV
Work.wait route. Lifetime 1 created one cache and lifetime 2 reused it after
clean teardown and inter-process health. Result root was
`/mnt/vm_8tb/b70/results/f03a_qwen38_fp8_neural/20260830T005000Z/`.

COMMAND -> under the whole-box lease, verify model and runtime identity, run
the full 12-prompt raw-token suite plus independent canaries in two fresh
processes sharing one compiler cache, and require card plus compiled P2P-off
collective health before, between, and after the servers.

RESULT -> all 12/12 complete arrays were exact across processes. Lifetime 1
reported 137.22 seconds in `torch.compile`. Lifetime 2 reconstructed 21
standalone artifacts and 65 submodules per rank, directly loaded both rank AOT
models, and reported 1.98 seconds total compile time. This is actual artifact
reuse, not a nominally shared directory followed by recompilation.

RESULT -> diagnostic rates were 11.303540 and 12.081169 tok/s, median
11.692355 tok/s. The 6.651 percent spread blocks a stable-speed headline even
though target coherence passed. Both canaries and all health checks passed.
Across 271 host samples, swap stayed zero, minimum MemAvailable was
113,127,392 KiB, container host-RAM use peaked near 7.718 GiB, and memory PSI
`some`/`full` totals moved by only 34.646/34.576 milliseconds. No configured
fatal marker appeared.

RESULT -> the 302 MiB, 2,250-file cache manifest SHA256 was
`ec1af4f6a06cc860da03e3bf7b359714efe6612e2b07d9083cb4cd30de19d64a`;
summary SHA256 was
`362c5b3ca2f5efaf53933cbf1e1f1723e1094b7de6c416907c6046af8024eabc`.

VERDICT -> F03a passes and localizes target selection to fresh compilation.
Treat the pinned cache as the deterministic MTP0 control; any fresh cache is a
new target. Proceed to the P2P-off packed-RMS MTP1 F04 comparison with shared-
cache discipline. Shelf, direct-P2P, long-context, concurrency, and stable-
speed claims remain blocked.

### 2026-08-30d - F04 MTP1 is restart-exact but misses the frozen target

CONFIG -> harness commit `dfe7ffd`; official Qwen3.8 FP8 W8A16, TP2, P2P
off, MTP1, graph off, deterministic Inductor, FP16 target/automatic-KV,
packed serial RMSNorm, persistent GDN scratch, and one shared fresh cache
across two server processes. Result root was
`/mnt/vm_8tb/b70/results/f04_qwen38_fp8_neural/20260830T012500Z/`.

COMMAND -> under the whole-box lease, verify all model/image/runtime bytes,
pass card and compiled P2P-off collective health, run the complete 12-prompt
raw-token suite plus canaries in two MTP1 processes sharing one cache, compare
both to the mutually exact frozen F03a MTP0 attempts, gracefully tear down,
and repeat health between and after servers.

RESULT -> the MTP1 lifetimes were 12/12 exact with one another. Lifetime 1
compiled the target in 141.48 seconds; lifetime 2 reconstructed 21 target
artifacts and 65 submodules per rank, directly loaded AOT key `ed4b9708...`,
and reported 1.92 seconds. Both attempts matched only 5/12 frozen MTP0 arrays.
The diagnostic rates were 18.076070 and 18.410930 tok/s, median 18.243500 and
1.836 percent spread. The apparent 56.029 percent gain over F03a is not
qualified because target identity failed.

RESULT -> canaries, all health, and teardown passed. Swap stayed zero,
minimum MemAvailable was 112,478,424 KiB, host-RAM use was 8.325 to 8.514
GiB, and the kernel/server scan had no configured fault marker. Runtime
accounting reported about 14.59 GiB model/non-Torch plus 8.3 GiB KV per card.
The 367 MiB, 3,081-file cache manifest SHA256 was
`8d85d9cc5e9f5d271048c0bd32863a489fe2e20c55dfd2e3d6f97c6a8a417e3f`;
the corrected summary SHA256 was
`4a0a2b38cd04691690729e71cb5fe1c2b7201fe02c3a57f49f540330065b042c`.

VERDICT -> close F04 negatively at target exactness. Do not attribute its
speed signal to MTP yet. Run F04a with an exact copy of F04's cache and a
synthetic zero-acceptance sampler, retaining MTP1 target verification while
forcing every draft rejection. Require reuse of target AOT key `ed4b9708...`;
keep long, concurrent, P2P-on, and shelf work blocked.

### 2026-08-30e - F04a clears MTP acceptance and implicates autotune selection

CONFIG -> harness commit `f59c6d9`; exact copy of F04's verified cache and
unchanged F04 MTP1 target/draft route, except synthetic acceptance was fixed at
zero. Result root was
`/mnt/vm_8tb/b70/results/f04a_qwen38_fp8_neural/20260830T020500Z/`.

COMMAND -> under the whole-box lease, verify the copied 3,081-file cache and
all model/image/runtime bytes; run two fresh server processes; require direct
loads of the same target and draft AOT artifacts; execute the full 12-prompt
suite and canaries at zero accepted drafts; gracefully tear down; and pass
card plus compiled P2P-off collective health between and after processes.

RESULT -> both processes directly loaded target key `ed4b9708...` and draft
key `aa87ccb...` on both ranks. Acceptance was exactly zero, yet both attempts
matched normal F04 12/12 and one another 12/12. Forced-rejection diagnostic
rates were 10.159880 and 10.178679 tok/s, median 10.169280 and 0.185 percent
spread. Normal F04's speed signal comes from accepted work, but its external
target gate remains failed.

RESULT -> F04a matched both local F03a MTP0 references 5/12, both publisher
MTP0 references 8/12, and both publisher MTP1 references 8/12. The publisher
MTP0/MTP1 references are mutually exact. Canaries, teardown, and every health
check passed; host RAM was 8.336 to 8.442 GiB, swap stayed zero, and minimum
MemAvailable was 112,926,784 KiB. The 3,131-file final cache manifest SHA256
was `ecf1d795d43494631134f8bbf943d42b5e2d91a5a68b1e257f96d75dab254a6c`;
summary SHA256 was
`911199dbce6e42cccd2ec7ba03e2fc7067ed0e045d3e4c82c6884d0880e7694b`.

RESULT -> read-only F02 cache comparison found the same primary graph key and
78 common Triton `.best_config` sites. Thirty-seven selected different block,
reduction, or warp configurations across fresh compiles; 41 differed only in
tuning time. The resulting rank AOT model binaries also differed.

VERDICT -> F04a passes and closes draft acceptance as a cause. Fresh compiler
autotune selection is now the leading local target-instability mechanism.
Proceed to F02b with XPU combo-kernel benchmarking disabled and separate fresh
MTP0 caches. Keep P2P-on full serving, long, concurrent, and shelf work
blocked.

### 2026-08-30f - F02b combo-off leaves lower-level autotune drift

CONFIG -> harness commit `e1221c1`; official FP8 W8A16, r15 Work.wait image,
TP2, P2P off, MTP0, FP16 target/KV, graph off, deterministic Inductor, and two
separate empty caches. Inductor `combo_kernels` and
`benchmark_combo_kernel` were false. Result root was
`/mnt/vm_8tb/b70/results/f02b_qwen38_fp8_neural/20260830T024100Z/`.

COMMAND -> under the whole-box lease, pass model/image/runtime identity and
pre-health; independently compile and run two 12-prompt plus canary server
lifetimes; tear down and pass card plus compiled P2P-off collective health
after each process; require cross-process and publisher token exactness.

RESULT -> clean negative. The attempts matched only 7/12 arrays. Attempt 1
matched the publisher 10/12; attempt 2 matched 6/12. Diagnostic rates were
11.465029 and 11.419766 tok/s, median 11.442398. Compilation took 105.74 and
105.87 seconds. Both canaries, all health gates, and teardown passed. Host RAM
peaked at 7.793 GiB, minimum MemAvailable was 113,617,324 KiB, and swap was
zero.

RESULT -> combo-off reduced `.best_config` sites from 78 to 44, but 22/44
common sites still selected different semantic block, reduction, or warp
configs. Twenty-one differed only in metadata and one was exact. All four
rank AOT model hashes differed across attempts. Cache-comparison summary
SHA256 was
`aa731b5a29e9b03b646c42746a8a67560caa26d9a9081421e73dae3a2f3db812`;
primary summary SHA256 was
`57500b75993cfe554cef6fb87214b77447de8c513923ce5b41544efaa77b3a7a`.

VERDICT -> combo benchmarking is not causal. F02c should retain combo-off and
also disable vLLM's default Inductor max-autotune and coordinate-descent
tuning, using separate fresh caches. Keep P2P-on full serving, long,
concurrent, and shelf work blocked.

### 2026-08-30g - F02c vLLM autotune flags do not control XPU tuning

CONFIG -> harness commit `8f9e9e5`; F02b configuration plus
`VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` and
`VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0`, with two separate empty
caches. Result root was
`/mnt/vm_8tb/b70/results/f02c_qwen38_fp8_neural/20260830T031700Z/`.

COMMAND -> under the whole-box lease, run the full two-lifetime identity,
12-prompt, canary, teardown, card-health, and compiled P2P-off collective
transaction. Require cross-process and publisher raw-token exactness.

RESULT -> clean negative: 8/12 cross-process exact and 5/12 versus each
publisher reference for both attempts. Diagnostic rates were 11.577039 and
11.346567 tok/s, median 11.461803. Canaries, teardown, and all health passed;
host RAM peaked at 7.789 GiB, minimum MemAvailable was 113,556,788 KiB, and
swap stayed zero.

RESULT -> both caches retained 44 common `.best_config` paths and exactly
22 semantic selection differences, unchanged from F02b. The vLLM flags
changed the AOT key to `eb5b1c57...` but not the XPU tuner. Cache-comparison
summary SHA256 was
`86134865a45f6d83ff006da881d68dac1dfd07f8e1dcaee51b9759b1beec2d63`;
primary summary SHA256 was
`d4eef66a854bba1482461e62aef11b2293adbb0ef147d369eb7c89f84e0998d1`.

VERDICT -> F02c is negative. PyTorch source shows
`triton.autotune_pointwise=True` independently creates multiple XPU
pointwise configurations. Run a bounded F02d two-cache compile oracle with
that control false and only proceed to a full suite if semantic cache
selection is exact. Keep P2P-on full serving, long, concurrent, and shelf work
blocked.

### 2026-08-30h - F02d isolates five variable reduction schedules

CONFIG -> harness commit `fae7351`; F02c controls plus
`triton.autotune_pointwise=false`, two empty caches, and a compile-only
16-token deterministic smoke per server. Result root was
`/mnt/vm_8tb/b70/results/f02d_qwen38_fp8_neural/20260830T035700Z/`.

COMMAND -> under the whole-box lease, independently compile two TP2 P2P-off
servers, verify model identity and short smoke coherence, tear each down, and
pass card plus compiled collective health. Compare every common
`.best_config` semantically and refuse the full suite on any difference.

RESULT -> pointwise-off reduced 44 sites to 16. Both attempts had the same
AOT key and exact smoke text, but five sites still differed semantically.
Every difference was `R0_BLOCK=2048` versus `8192` with XBLOCK, warps, and
stages otherwise equal. Compilation was 96.25 and 96.33 seconds. All health,
teardown, and zero-swap gates passed; host RAM peaked at 7.668 GiB and minimum
MemAvailable was 113,721,668 KiB.

RESULT -> generated kernel source recorded `deterministic: False` despite the
launcher environment `TORCHINDUCTOR_DETERMINISTIC=1`. The environment setting
did not survive into the AOT compilation patch. Summary SHA256 was
`135f482a392bb4367fafa25873e8bb1bfba33931167c02ab1e2c815e46357f58`.

VERDICT -> F02d correctly blocks a full run and narrows the target to five
reduction schedules. F02e should explicitly pass
`inductor_compile_config.deterministic=true`, invoking PyTorch's existing
deterministic reduction filter. Keep P2P-on full serving, long, concurrent,
and shelf work blocked.

### 2026-08-30i - F02e collapses fresh compiler selection

CONFIG -> harness commit `d8f4170`; F02d compile oracle plus explicit
`inductor_compile_config.deterministic=true`, with two separate empty caches.
Result root was
`/mnt/vm_8tb/b70/results/f02e_qwen38_fp8_neural/20260830T041300Z/`.

COMMAND -> under the whole-box lease, independently compile two TP2 P2P-off
servers, verify model identity and a 16-token smoke, tear down, pass card and
compiled collective health, and require semantic cache-selection exactness.

RESULT -> pass. Both fresh compiles used AOT key `5001f6c4...`; generated
reduction metadata recorded deterministic true; both final caches contained
zero `.best_config` files; smoke text was exact. Compilation took 92.11 and
90.91 seconds. All health and teardown passed. Host RAM peaked at 7.651 GiB,
minimum MemAvailable was 113,822,200 KiB, and swap stayed zero. Summary SHA256
was `47c53fe1719f8a83515027f7f26d3de21c2de4480378e56194e441b690147f23`.

VERDICT -> F02e passes its compile-selection discriminator. Proceed to F02f
with the full 12-prompt two-empty-cache target gate and identical compiler
controls. Require both cross-process and publisher raw-token exactness; keep
P2P-on full serving, long, concurrent, and shelf work blocked.

### 2026-08-30j - F02f creates a reproducible local target, not the publisher target

CONFIG -> harness commit `eb14d56`; official FP8 W8A16 r15 `Work.wait()`
image, TP2, P2P off, MTP0, FP16 target/KV, graph off, and two independent
empty caches. Combo tuning, vLLM max autotune, coordinate descent, and Triton
pointwise autotune were disabled; deterministic true was passed explicitly in
the Inductor compile config. Result root was
`/mnt/vm_8tb/b70/results/f02f_qwen38_fp8_neural/20260830T042700Z/`.

COMMAND -> under the whole-box lease, verify image, model, and served identity;
run two fresh compiles through the complete fixed 12-prompt 512-token-cap
suite and independent canaries; tear down and pass card plus compiled P2P-off
collective health after each; require cross-process and publisher raw-token
exactness.

RESULT -> local fresh-cache exactness passed 12/12. Both attempts used AOT key
`5001f6c4...`, took 92.19 and 92.38 seconds to compile, and left 621-file
caches with zero `.best_config` files. Diagnostic class-balanced rates were
11.637675 and 11.649289 tok/s, median 11.643482 and 0.100 percent spread.

RESULT -> the external target gate failed: both attempts matched only 8/12
arrays against each of the two mutually exact publisher references. Fresh
schedule selection caused prior local restart drift, but the publisher target
is a different autotuned compilation mosaic. Eleven suite prompts reached the
512-token cap, so this is not a high-thinkcap quality qualification.

RESULT -> all canaries, teardowns, card health, and compiled collectives
passed. Host RAM peaked at 7.696 GiB, minimum MemAvailable was 113,710,852
KiB, and swap stayed zero. Device accounting reported 14.24 GiB weights plus
non-Torch, 1.19 GiB peak activation, and 8.8 GiB KV per card. Summary SHA256
was `fc73b5bea7bb0e9c98361cd66e965591292c437fd8cee790a98e19c613703934`.

VERDICT -> close F02f negatively versus the publisher but positively as a
local deterministic oracle. Run F02g as an MTP0 bridge through the
MTP-capable packed-RMS image, requiring two empty caches to match F02f. Do not
add MTP1 until the bridge passes. Keep long, concurrent, P2P-on, speed
attribution, and shelf work blocked.

### 2026-08-30k - F02g passes the packed-RMS MTP0 bridge

CONFIG -> harness commit `58baa4e`; official FP8 W8A16, MTP-capable local
image, TP2, P2P off, MTP0, packed serial RMSNorm, FP16 target/KV, graph off,
and two empty caches. F02f's explicit deterministic compiler controls were
retained and both F02f attempts were frozen references. Result root was
`/mnt/vm_8tb/b70/results/f02g_qwen38_fp8_neural/20260830T050400Z/`.

COMMAND -> under the whole-box lease, independently compile and run two full
12-prompt lifetimes in the MTP-capable image; require cross-process and F02f
raw-token exactness; run canaries, graceful teardown, card health, and compiled
P2P-off collective health after each.

RESULT -> pass. The attempts matched one another 12/12 and each matched both
F02f references 12/12. Both used target AOT key `5001f6c4...`, compiled in
95.41 and 95.84 seconds, and left 621-file caches with zero `.best_config`
files. Packed serial RMSNorm and the MTP-capable image preserve the local
explicit-deterministic target.

RESULT -> diagnostic rates were 11.503855 and 11.434064 tok/s, median
11.468959 and 0.609 percent spread. The 1.499 percent difference from F02f is
below the attribution threshold. All canaries, teardowns, and health gates
passed. Host RAM peaked at 7.708 GiB, minimum MemAvailable was 113,609,756
KiB, and swap stayed zero. Summary SHA256 was
`a378bf0d71b9b4fd9ec9a62b89d460f87b0afb79cac4907661acabc1c56ef3bf`.

VERDICT -> F02g authorizes F04b: add MTP1 with the same image and compiler
controls, use two empty caches, and require exactness to frozen F02g MTP0
arrays before speed attribution. Keep long, concurrent, P2P-on, and shelf
work blocked.

### 2026-08-30l - F04b qualifies deterministic MTP1 at 17.65 tok/s

CONFIG -> harness commit `59f72c0`; official FP8 W8A16, TP2, P2P off, MTP1,
packed serial RMSNorm, persistent GDN scratch, FP16 target/KV, graph off, and
two empty caches. Explicit deterministic compiler controls were retained and
both F02g MTP0 attempts were frozen references. Result root was
`/mnt/vm_8tb/b70/results/f04b_qwen38_fp8_neural/20260830T053900Z/`.

COMMAND -> under the whole-box lease, compile and run two independent MTP1
lifetimes through the full 12-prompt suite and canaries; require cross-process
and F02g raw-token exactness; gracefully tear down and pass card plus compiled
P2P-off collective health after each.

RESULT -> pass. Both MTP1 attempts matched one another and both F02g MTP0
references 12/12. Target key `57e8f544...` and draft key `fe3112d...` repeated
across caches. Target compilation took 96.08/96.28 seconds and draft
compilation 9.67/9.55 seconds. Both 976-file caches had zero `.best_config`
files.

RESULT -> class-balanced rates were 17.648289 and 17.650913 tok/s, median
17.649601 with 0.015 percent spread. This is a qualified 53.890 percent gain
over F02g's matched 11.468959 tok/s MTP0 median. Acceptance commonly ranged
from about 65 to 93 percent in ten-second windows.

RESULT -> all canaries, teardowns, and health gates passed. Host RAM peaked at
8.399 GiB, minimum MemAvailable was 112,819,476 KiB, and swap stayed zero.
Device accounting reported 14.59 GiB weights plus non-Torch, 1.20 GiB peak
activation, and 8.45 GiB KV per card. Summary SHA256 was
`4c7a689698e32bd3865f6e3147637ada3eb8a040556c9ed2706a0c6cdaa8963e`.

VERDICT -> F04b is research-qualified for bounded 1K single-stream serving.
Proceed to F05a with a 32K-configured server, real growing prompts, forced 4K
decode, restart exactness, bounded memory, and full health. Long, concurrent,
P2P-on, agent, and shelf qualification remain blocked until their own gates.

### 2026-08-30m - F05a passes 30K context and 4K forced output

CONFIG -> harness commit `cbb24dc`; F04b MTP1 route at 32,768 model and batch
limits, one sequence, TP2, P2P off, FP16 KV, graph off, prefix cache off, and
two empty caches. Each lifetime ran the bounded target, actual growing context
points, and a forced 4,096-token output. Result root was
`/mnt/vm_8tb/b70/results/f05a_qwen38_fp8_neural/20260830T061000Z/`.

COMMAND -> under the whole-box lease, independently compile two 32K servers;
require 12-prompt exactness to one another and both F04b references; run cold
2K/8K/16K/30K prompts with 128 forced outputs and a 2K-prompt plus 4K forced
decode; compare all long raw-token arrays across restarts; tear down and pass
card plus compiled collective health after each lifetime.

RESULT -> the bounded target passed 12/12 everywhere at 17.746417 and
17.743088 tok/s, median 17.744753 and 0.019 percent spread. Actual context
counts were 2,070, 8,214, 16,407, and 30,023 tokens. All four 128-token arrays
were restart-exact. TTFT reached 40.151 and 40.101 seconds at 30,023 tokens;
decode remained 17.15 to 18.34 tok/s.

RESULT -> both 2,070-prompt plus 4,096-output runs were raw-token exact and
measured 19.186696 and 19.101269 tok/s after TTFT. Target key `80de0121...`
and draft key `be175b50...` repeated; compile times were 95.31/95.54 and
45.07/45.11 seconds. Both 976-file caches had zero `.best_config` files.

RESULT -> all canaries, teardowns, and health gates passed. Host RAM peaked at
9.663 GiB, minimum MemAvailable was 111,723,052 KiB, and swap stayed zero.
Device accounting reported 15.47 GiB weights plus non-Torch, 2.91 GiB peak
activation, and 5.86-5.87 GiB KV per card. Primary summary SHA256 was
`014fd18be7c66bda43b0d83e11c371c5bbe5c8837948297a1b249b36ee1c194d`.

VERDICT -> F05a passes synthetic C1 long-context and forced-output gates. Run
F05b for concurrent batch-shape coherence before shelf work, then the
higher-thinkcap growing-agent quality ladder. P2P-on full serving remains
blocked.

### 2026-08-30n - F05b reproduces the old GDN mixed-batch abort

CONFIG -> F05a's 32K deterministic MTP1 path with four service slots, P2P
off, graph off, and the inherited vllm-xpu-kernels 0.1.12.3. Result root was
`/mnt/vm_8tb/b70/results/f05b_qwen38_fp8_neural/20260830T065500Z/`.

COMMAND -> Run the normal target and canary gates, four serial 2K/512
controls, then synchronized C4 completion requests under the whole-box lease.
On failure, tear down and run card plus compiled P2P-off collective health.

RESULT -> the normal suite retained the target at 17.511970 tok/s. The first
C4 batch mixed active MTP decode and a new prefill, and both workers raised the
kernel's explicit `causal_conv1d does not support spec-decode and non-spec`
error. The engine stopped. Both cards and the compiled collective passed after
cleanup.

VERDICT -> F05b is a software-path negative, not a host wedge or RAM-spill
event. Concurrent MTP1 is blocked on the recipe's corrected mixed-path kernel.

### 2026-08-30o - F05c closes the engine abort and corrects the quality gate

CONFIG -> overlay the exact recipe wheel at XPU kernel commit `1e90ffa672`,
SHA256 `f3d999060c11ad6db5b4033d50d19c6b665492380075480d041ec4ee58fdfeb6`,
onto the qualified F04b/F05a image. Composite image ID was `8e0e3deb...`.
Keep TP2 P2P off, MTP1, graph off, deterministic Inductor, 32K capacity, and
C4. Result root was
`/mnt/vm_8tb/b70/results/f05c_qwen38_fp8_neural/20260830T073000Z/`.

COMMAND -> compile two empty caches and run serial plus synchronized C4 2K
prompts with 64 forced output tokens, teardown, and health around each server.

RESULT -> all 8 serial and all 8 concurrent streams completed. The fatal GDN
exception disappeared. Target/draft AOT keys `80de0121...`/`be175b50...`
repeated with zero `.best_config` files. Both teardowns and all card/collective
checks passed. Diagnostic C4 aggregate rates were 18.334303/18.192014 tok/s.

RESULT -> all four serial arrays were restart-exact. Two of four C4 arrays
changed across restarts because asynchronous arrival changed batch history;
the publisher discloses the same batch-shape dependence. Peak container RAM
was 9.816 GiB and minimum host MemAvailable was 111,388,204 KiB. The container
had no swap allowance. Global host swap rose to 28,652 KiB without OOM or a
GPU kernel fault.

VERDICT -> the pinned kernel fixes concurrent engine survival. Reject the old
asynchronous C4 byte-exact contract; run F05d with complete 512-token streams,
the fixed target suite, and concurrent exact-answer/isolation semantics before
shelf work. Direct-P2P full serving remains blocked.

### 2026-08-30p - F05d qualifies corrected-kernel C4 and finds a new target

CONFIG -> official FP8 W8A16, corrected 1e90 GDN kernel, TP2, P2P off, MTP1,
graph off, deterministic Inductor, packed serial RMSNorm, FP16 KV, 32K
capacity, and C4. Result root was
`/mnt/vm_8tb/b70/results/f05d_qwen38_fp8_neural/20260830T075200Z/`.

COMMAND -> under the whole-box lease, compile two empty caches; in each fresh
lifetime run the 12-prompt C1 suite, canaries, two C4 batches of four
2K-prompt/512-output streams, and 32 concurrent exact-answer/isolation
requests; then teardown and run card plus compiled collective health.

RESULT -> both attempts matched one another 12/12 at 17.574570 and 17.503311
tok/s, median 17.538941. All 16 long concurrent streams completed and all 64
semantic requests passed. C4 batch aggregates were 66.376149, 50.879935,
67.294763, and 49.450377 tok/s. Target/draft keys `80de0121...`/`be175b50...`
repeated with zero `.best_config` files.

RESULT -> both attempts matched old-kernel F05a only 10/12. Stable changes
were `customer-email` at token 124 and `technical-guide` at token 160. All
teardowns and health passed. Peak container RAM was 9.088 GiB, minimum host
MemAvailable was 111,507,896 KiB, and global swap stayed at its preexisting
28,652 KiB baseline. Summary SHA256 was
`373f32462f63db25a540c60e1e54afface84cece11585180358cfb0c4ef10f76`.

VERDICT -> corrected-kernel C4 engine survival, complete long streams, and
semantic isolation pass. Old-target promotion correctly fails. Run an MTP0
control against both corrected MTP1 references to determine whether the
target shift is caused by the kernel or speculative decoding.

### 2026-08-30q - F05e proves the target shift belongs to the corrected kernel

CONFIG -> exact F05d corrected-kernel image and deterministic P2P-off 32K
route, with only MTP changed from one to zero and maximum sequences reduced
to one. Two fresh server processes shared the newly compiled MTP0 cache and
used both F05d MTP1 performance files as required references. Result root was
`/mnt/vm_8tb/b70/results/f05e_qwen38_fp8_neural/20260830T083000Z/`.

COMMAND -> under `bin/gpu-run`, run the 12-prompt suite and independent
canaries twice; require cross-process and both-reference raw-token exactness;
gracefully tear down and run card plus compiled P2P-off collective health
after each lifetime.

RESULT -> pass. Both MTP0 attempts matched one another and both corrected
MTP1 references 12/12. They retained F05d's stable 10/12 relation to the old
F05a kernel target, including `customer-email` at token 124 and
`technical-guide` at token 160. Rates were 11.327250 and 11.742236 tok/s,
median 11.534743. Attempt 1 compiled target key `560096c7...` in 97.66
seconds; attempt 2 loaded it directly. The 1,747-file cache had zero
`.best_config` files.

RESULT -> canaries, both teardowns, and all card/collective checks passed.
Peak container RAM was 7.755 GiB, minimum host MemAvailable was 112,371,368
KiB, and all 291 host samples retained the preexisting 28,652 KiB swap
baseline. Summary SHA256 was
`a7385835dad957e386203465add4550ea4f5d57cd5f7af4972600bf1e62c9fe7`.

VERDICT -> the corrected GDN kernel, not MTP acceptance, causes the stable
two-prompt target shift. Treat F05d as the corrected-kernel MTP1 target and
concurrent qualification. Keep shelf promotion blocked on the growing-agent
gate; direct-P2P full serving remains blocked on its loaded-context oracle.

### 2026-08-30r - TB01 true-off policy times out with a healthy machine

CONFIG -> Qwen3.8-27B compressed-tensors W8A8 GPTQ under the stable SGLang
reclaim500 route, TP2, P2P off, target-only decode, breakable graph at batch
one, BF16 target and KV, 65,536 context, memory fraction 0.70, and one running
request. Pi 0.84.3 used the payload-verified true thinking-off policy, the
concise off prompt, 8,192 maximum output tokens, and the 1,800-second official
agent timeout on `terminal-bench/bun-sourcemap-leak`. Host admission required
96 GiB available and at most 1 GiB swap; the server container was limited to
64 GiB with no swap. Result root was
`/mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-w8a8-reclaim500-20260830T090700Z/`.

COMMAND -> under `bin/gpu-run`, pass card and compiled P2P-off collective
health, launch one fresh server, require exact `/v1/models` identity and
observed BF16 KV, run the one-task Harbor job, check the endpoint before
teardown, stop the server, and repeat card plus compiled collective health.

RESULT -> the agent edited after about 7 minutes 35 seconds and ran relevant
post-edit tests, but used 23 tool calls and remained in an iterative repair
loop. Harbor terminated it at exactly 1,800 seconds with
`AgentTimeoutError`. Pi recorded 235,752 input and 10,965 output tokens. The
separate verifier ran and returned zero; its captured state still made
`bun run release` reject private server identifiers. This is not a normal
zero-score completion.

RESULT -> exact model identity, configured and observed BF16 KV, endpoint
health before teardown, endpoint shutdown, both post-teardown cards, and the
compiled two-rank P2P-off collective all passed. There were no fatal server
markers. Full machine occupation was 2,311 seconds. Host spot checks retained
about 60 GiB available and only the pre-existing roughly 32 MiB swap usage.
Result, lifecycle, trajectory, and verifier-log SHA256 values were respectively
`ebe73b8d...`, `ed22d3ed...`, `6547c864...`, and `3a9ec0e9...`.

VERDICT -> TB01 fails the true-off policy gate through model-agent timeout,
not infrastructure, VRAM spill, or host exhaustion. TB02A does not apply and
accepted TB02B remains blocked because the baseline was not a normal
completion. Predeclare one unranked TB02X rescue diagnostic at the same
1,800-second timeout with native thinking, 16,384 maximum output tokens, and
an 8,192 private-thinking cap. It may diagnose completion rescue but cannot be
reported as a matched win over the censored TB01 baseline.

### 2026-08-30s - TB02X 8K native thinking completes but scores zero

CONFIG -> exact TB01 Qwen W8A8 reclaim500 server, model identity, TP2 P2P-off
topology, BF16 KV, 65,536 context, one request, host admission guards, and Bun
task. Change only the agent policy to Pi native thinking with 16,384 maximum
output tokens, the normal concise prompt, and an 8,192 server-side private
thinking cap. Keep the official 1,800-second agent timeout. This was an
unranked rescue diagnostic because TB01 was time-censored. Result root was
`/mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-w8a8-reclaim500-20260830T094700Z/`.

COMMAND -> under `bin/gpu-run`, pass card and compiled P2P-off collective
health, launch the guarded server, verify exact identity and observed BF16 KV,
run the one-task Harbor job, require an independent verifier, check the live
endpoint before teardown, then repeat card and compiled collective health.

RESULT -> normal completion with no exception, length stop, or endpoint loss.
Agent execution took 26 minutes 13 seconds, used ten tool calls, and recorded
146,926 input plus 13,728 output tokens. The first write-generation request
began after about 11 minutes 59 seconds, but the file write executed only after
about 15 minutes 56 seconds. The first post-edit test was issued after about
19 minutes 7 seconds. Public client/server behavior, relative source-map
provenance, the deobfuscated public trace, leak greps, and an idempotent release
passed in the agent's checks.

RESULT -> the independent verifier scored zero but passed 25/36 tests. Nine
tests failed and two errored, concentrated in private-client entry/helper,
many-file client, generated-policy, local-path, private-import identity, and
private-secret variants. Unlike TB01, this is a normal model-quality result.
Harbor wall was 30 minutes 43 seconds, and full machine occupation was 2,035
seconds. Exact identity, observed BF16 KV, pre-teardown endpoint health,
teardown, both cards, and the compiled two-rank P2P-off collective passed with
no fatal server markers. Host spot checks stayed near 60 GiB available with
the roughly 32 MiB swap baseline unchanged.

RESULT -> result, lifecycle, trajectory, and verifier-log SHA256 values were
`3967704c...`, `9342f4b7...`, `6bf11be7...`, and `1bf83c97...`.

VERDICT -> an 8,192 private-thinking cap rescued normal completion and a much
stronger captured implementation than TB01, but still scored zero and missed
the ten-minute edit gate. Reject 8K as the default 30-minute policy. Because
the run completed normally, predeclare one matched TB02C cap calibration that
changes only the private-thinking cap to 4,096 while retaining native thinking
and the 16,384 response ceiling. Require a roughly ten-minute edit, normal
completion, and at least 25/36 tests; stop Bun cap search after that run if it
times out or regresses.

### 2026-08-30t - TB02C 4K native thinking times out and regresses

CONFIG -> exact TB02X Qwen W8A8 reclaim500 server, model identity, native
thinking policy, 16,384 maximum output, Pi prompt, task, official 1,800-second
timeout, host guards, and lifecycle contract. Change only the server-side
private-thinking cap from 8,192 to 4,096. This isolates the cap from the
historical confounded 4,096-total-response run. Result root was
`/mnt/vm_8tb/b70/evals/harbor-jobs/tb3-qwen-w8a8-reclaim500-20260830T102200Z/`.

COMMAND -> under `bin/gpu-run`, pass card and compiled P2P-off collective
health, launch the guarded server, verify exact identity and observed BF16 KV,
run the one-task Harbor job and independent verifier, then require live
pre-teardown endpoint health, endpoint shutdown, both cards, and the compiled
collective after teardown.

RESULT -> timeout. The write-generation request began about 6 minutes 46
seconds after agent start, but continued generating for about 9 minutes 8
seconds; the 10,436-byte file write executed only at about 15 minutes 54
seconds. Later edits remained in a repair loop and no post-edit test ran before
the exact 1,800-second `AgentTimeoutError`. Pi used 13 tool calls and recorded
145,742 input plus 15,813 output tokens.

RESULT -> the independent verifier scored zero and the captured state
regressed to 21/36 tests, with 13 failures and two errors including a broken
server artifact. Harbor wall was 34 minutes 32 seconds and full machine
occupation was 2,252 seconds. Exact identity, BF16 KV observation,
pre-teardown endpoint health, endpoint shutdown, both cards, and the compiled
P2P-off collective passed with no fatal server markers. Host spot checks kept
about 60 GiB available and the roughly 32 MiB swap baseline unchanged.

RESULT -> result, lifecycle, trajectory, and verifier-log SHA256 values were
`f09a93d1...`, `2ed91bba...`, `bc01130e...`, and `dcd86b94...`.

VERDICT -> TB02C fails every predeclared retention gate: edit latency, normal
completion, post-edit test, and the TB02X 25/36 hidden-test count. Stop the Bun
thinkcap search. The private-thinking cap is a soft segment bound, not a hard
agent-turn bound, and cap size did not dominate trajectory quality. Among the
tested policies only 8K completed normally, but none scored; do not promote
off, 4K, or 8K as a task-effective default from this single task.

### 2026-08-30u - exact Qwen3.8 FP8 direct P2P passes through full C4 qualification

CONFIG -> pinned corrected Qwen3.8-27B official-FP8 image ID `8e0e3deb...`,
TP2, FP16 target and KV, packed serial RMSNorm, persistent mixed-path GDN,
deterministic Inductor, FlashAttention, graph off, MTP0 then MTP1, and the
publisher direct route: localhost TCP OFI, pidfd IPC exchange, direct
send/receive, `CCL_TOPO_P2P_ACCESS=1`, and simple thresholds at 4294967296.
Kernel was 7.1.0-070100 and all GPU touches used `bin/gpu-run`.

COMMAND -> advance only after each bounded gate and P2P-off post-health:
F06a compiled raw oneCCL; F06b vLLM `XpuCommunicator` with eager/custom-op and
40 compiled calls from 1 to 2,048 rows; F06c full-weight MTP0; F06d MTP1;
F06e 32 short C4 quality requests; and F06f two fresh 32K/C4 full lifetimes.
F06f result root was
`/mnt/vm_8tb/b70/results/f06f_qwen38_fp8_neural_p2p/20260830T130000Z/`.

RESULT -> every staged gate passed. F06b returned both ranks from every
entry/return pair with zero numerical error. F06c and F06d returned `READY`.
F06e passed all 32 exact-answer/isolation requests. The historical loaded
vLLM queue-handoff stall did not reproduce, including the real 32,768-size
F06f warmup in two fresh server processes.

RESULT -> F06f serial rates were 18.297860 and 18.377703 tok/s, centered at
18.337782 versus F05d P2P-off's 17.538941: +4.554670 percent. Direct C4
aggregate rates were 71.223289, 70.946863, 70.647401, and 66.240527 tok/s;
their mean improved 19.254966 percent and median improved 20.756433 percent
over F05d P2P off. Direct P2P is a material concurrency/prefill lever but
does not explain the publisher's 51.918757 tok/s single-stream headline.

RESULT -> both direct lifetimes matched one another and both corrected P2P0
references 12/12. Both independent canary sets, all 16 complete long C4
streams, and all 64 concurrent quality requests passed. Both teardowns and
all card/compiled-collective checks passed with no matching Xe fault. Peak
container RAM was 8.591 GiB, minimum host MemAvailable was 110,729,476 KiB,
and global swap peaked at the pre-existing 340,152 KiB. Summary SHA256 was
`a838f76e750b822b3f80b306559fca8f80e60f28771d688edfe950edc5f82c69`.

RESULT -> one aborted F06d preflight at `20260830T122300Z` mistakenly entered
card health without a lease and then interrupted the following P2P-off
collective. No model or P2P1 server started. It is labeled ABORTED and excluded
from results. The process was stopped, Xe rebind was run under `bin/gpu-run`,
and both cards plus the compiled P2P-off collective recovered without reboot.
The harness signal trap was fixed before the accepted F06d run.

VERDICT -> direct oneCCL P2P is qualified for this exact Qwen3.8 FP8,
corrected-kernel, MTP1, graph-off route. Keep P2P off as the generic launcher
default and require the explicit risk guard for this opt-in. Next compare
publisher and F06f process/profile, collective-count, target/drafter, and
kernel-selection evidence to locate the remaining speed gap. Keep push-AR as
a separate arm until its loaded-context first-submission oracle returns, and
keep long growing-agent/thinking-cap quality as a separate campaign gate.

### 2026-08-30v - Qwen3.8 FP8 FULL graph exceeds 45 tok/s twice

CONFIG -> corrected neural.download Qwen3.8-27B official FP8 W8A16 image ID
`8e0e3deb...`, TP2, FP16 model and KV, MTP1, deterministic Inductor, packed
serial RMSNorm, persistent GDN scratch, one slot, 1,024-token envelope, prefix
cache off, direct oneCCL P2P, explicit Triton target and draft attention,
`FULL_DECODE_ONLY`, and forced graph capture with communication. Kernel was
7.1.0-070100 and every GPU touch used `bin/gpu-run`.

COMMAND -> first reproduce the publisher graph-off compiler/service envelope
twice in F07a; then screen PIECEWISE, no-MTP FULL, MTP1 FULL with auto draft,
and MTP1 FULL with explicit Triton draft. Promote the last arm to two fresh
F07f complete fixed 12-prompt strict suites. Add the 32-request, eight-round
four-client exact-answer/isolation canary to the confirmation lifetime. Run
card and compiled P2P-off collective health before and after every risky TP2
transaction.

RESULT -> F07a remained at 17.488233 and 17.259319 tok/s despite matching the
publisher target and draft AOT keys. PIECEWISE was slower. No-MTP FULL reached
30.838216 full post-TTFT tok/s. MTP1 FULL with auto draft reached 40.341043,
and explicitly selecting Triton for the independently configured draft raised
the bounded full-stream screen to 42.034140.

RESULT -> the two F07f strict class-balanced first-100 rates were 46.721530
and 47.170372 tok/s, averaging 46.945951. Both workload gates passed with all
cached-token counts zero. Complete token arrays matched 12/12 across fresh
lifetimes. The confirmation canary passed 32/32 requests. Both teardowns,
card checks, and compiled TP2 collectives passed. The two performance SHA256
values were `058c26b9...` and `c9c5ce67...`; concurrent-quality was
`1d551218...`.

RESULT -> F07f matches only 7/12 publisher r32a and 7/12 r32b token arrays.
It is deterministic locally but is not an exact publisher-output route. The
two-run primary mean is 170.21 percent above the F07a graph-off center and
9.58 percent below the 51.918757 tok/s publisher headline.

VERDICT -> the 45 tok/s single-stream goal is met and reproduced. FULL decode
capture, rather than P2P alone or publisher compile-envelope matching, removes
the dominant local host-submission bottleneck. Qualify F07f as a local FULL
graph route with explicit Triton target/draft attention, one-slot scope, and
the existing P2P risk guard. Do not call it an exact publisher reproduction or
place it on the serving shelf until packaging and serving-policy review.

### 2026-08-30w - source-only MTP8 matches the publisher speed range

CONFIG -> qualified F07f base image and settings, plus only the 103-line
MTP1/MTP8 packed-serial RMSNorm Python-source change from pinned vLLM commit
`ac7509e2b...` and publisher patch SHA256 `98c26561...`. The derived image was
`b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a`, image ID `9ae697d4...`,
with installed layernorm SHA256 `d911627c...`. No quarantined wheel, shared
object, or other ABI-specific binary was restored. Runtime was official FP8
W8A16, TP2, FP16 KV, MTP8, one slot, 1,024 tokens, direct oneCCL P2P, explicit
Triton target/draft attention, and forced `FULL_DECODE_ONLY` graph capture.

COMMAND -> build the source overlay using
`vllm/fp8/build_qwen38_fp8_mtp8_rms_overlay.sh`; run the bounded F08a screen;
then run two fresh complete fixed 12-prompt F08b strict suites with
`vllm/fp8/run_qwen38_fp8_f08b_full_decode_triton_mtp8_strict_suite.sh`. Add
the 32-request, eight-round, four-client isolation canary to the confirmation
lifetime. Run card and compiled P2P-off collective health before and after
each direct-P2P model transaction.

RESULT -> the F08a cold screen measured 29.140815 tok/s on the first-100
metric and 44.262562 tok/s across the full completion while paying five first
inference Triton JITs. The two complete F08b lifetimes measured 64.965356 and
67.404052 tok/s primary, averaging 66.184704. This is inside the publisher's
later MTP8 range of 62.432362 to 68.049727 tok/s and 27.48 percent above the
public 51.918757 tok/s MTP1 headline. Full-response post-TTFT medians were
46.874718 and 47.519518 tok/s, averaging 47.197118.

RESULT -> both strict gates passed with zero cached tokens. All 12 complete
token arrays matched across local fresh lifetimes, and the confirmation
canary passed 32/32 requests. Local arrays matched 8/12 publisher MTP8 r1a
and 9/12 r1b arrays, so publisher process identity remains false. Both
teardowns, all card checks, and all compiled TP2 collective checks passed.
Performance SHA256 values were `100da68c...` and `21cbc97a...`; confirmation
concurrent-quality was `db19b6d1...`.

VERDICT -> qualify F08b as the local single-stream performance winner. It
reproduces the publisher MTP8 speed band and clears 45 tok/s even on the
full-response median, while F07f remains the simpler MTP1 control. Do not call
F08b the public graph-off MTP1 recipe or publisher-byte-identical. Keep the
one-slot scope and P2P risk guard; defer shelf promotion until max-concurrency
policy and packaging are explicitly qualified.

### 2026-08-30x - Qwen3.8 FP8 MTP1 daily-driver promotion and envelope limit

CONFIG -> derived image `b70-local/vllm-openai-xpu:qwen38-fp8-mtp8-rms-f08a`
ID `9ae697d4...`, official Qwen3.8-27B FP8 W8A16, TP2, FP16 KV, direct P2P,
FULL decode graph, explicit Triton target/draft attention, deterministic
compiler settings, four slots, 262,144 model length, 32,768 batched tokens,
0.96 utilization, prefix cache off, and fixed MTP1 or MTP8. No host swap
allowance or model/KV offload was used.

COMMAND -> screen MTP1 and MTP8 at c1/c2/c4; probe V1 dynamic MTP and V2 FULL;
run a 30,037-token semantic needle under the exact large envelope; then run
two fresh MTP1 F09f qualification lifetimes with full 12-prompt token arrays,
canaries, c2/c4 completion probes, 32-request concurrent quality per lifetime,
teardown, card health, and compiled TP2 collective health. Probe 260K once as
a bounded negative and use Xe rebind after its abort.

RESULT -> V1 dynamic MTP forced PIECEWISE and measured 27.642081 tok/s strict.
V2 retained FULL but failed the native GDN speculative-token graph invariant.
Fixed MTP1 exposed 323,202 aggregate KV tokens; fixed MTP8 exposed 303,414.
One full 262K request fits in VRAM, while four full requests do not.

RESULT -> MTP8 screens measured about 59 to 64 tok/s aggregate at c2 and 101
to 102 at c4, plus 61.485779 strict c1 in the 262K envelope. MTP1 F09f strict
rates were 46.610781 and 46.597152; c2 was 45.708003 and 48.045488; c4 was
89.094019 and 88.830444. Complete 12-prompt arrays matched 12/12 across MTP1
lifetimes and both 32-request concurrent quality gates passed.

RESULT -> fresh 30K TTFT was effectively independent of MTP depth: 221.968 s
at MTP8 versus 221.961 and 221.837 s at MTP1. The older 32K-envelope MTP1
route needed only about 40.1 s at 30K. A 260K request held its first 32,768
worker submission for more than ten minutes and was aborted. All recovery,
card health, and compiled-collective checks passed. The long-prefill regression
belongs to the large service/prefill shape, not MTP8 specifically.

RESULT -> F09f summary verdict and promotion authorization are true, strict
median is 46.603967 tok/s, long-agent token arrays match, and summary SHA256 is
`44da0a2d...`. Evidence root is
`/mnt/vm_8tb/b70/results/f09f_qwen38_fp8_mtp1_daily/20260830T230000Z/`.

VERDICT -> promote MTP1 as the conservative shelf default and retain MTP8 as
the explicit faster decode profile. Label 262K as a measured capacity ceiling,
not qualified near-window prefill throughput. Prefix caching is still off;
large-envelope prefill and prefix-cache qualification are the next campaign.

### 2026-08-30y - promoted FP8 shelf live smoke

CONFIG -> promoted `rdy_to_serve/vllm/qwen38-27b-fp8/serve.sh`, default MTP1
profile, separate seeded daily cache, otherwise exact F09f 262K/c4 runtime.

COMMAND -> run `vllm/fp8/smoke_qwen38_fp8_f09g_shelf.sh` through the whole-box
`bin/gpu-run` lease with card and compiled collective health before and after.

RESULT -> exact served ID passed, the completion endpoint returned output,
the container and listener tore down, and both card and compiled TP2
collective post-health passed. Evidence root is
`/mnt/vm_8tb/b70/results/f09g_qwen38_fp8_shelf_smoke/20260830T233000Z/`.

VERDICT -> F09g PASS. The new FP8 MTP1 shelf default is live; MTP8 remains the
explicit `PROFILE=fast` option.

### 2026-08-31a - graph-off publisher cgroup match remains at 17.55 tok/s

CONFIG -> closest runnable Steve r32 reproduction using local image ID
`8e0e3deb...`, verified packed-RMS and GDN runtime files, XPU kernel
`1e90ffa672`, official Qwen3.8 FP8 W8A16, TP2, FP16 KV, MTP1, direct P2P,
graph disabled, default FlashAttention v2, deterministic Inductor, publisher
compilation JSON, one slot, 1,024 model and batched tokens, 0.96 GPU memory,
fresh compile cache per server, and the publisher 9 GiB memory / 12 GiB
memory-and-swap cgroup. The publisher `r31` tag and image ID `ba42e928...`
were not locally present or publicly pullable, so byte-identical OCI identity
was unavailable.

COMMAND -> add an explicit memory-and-swap limit to the retained launcher and
run `I_KNOW_P2P_WEDGES=1
vllm/fp8/qualify_qwen38_fp8_neural_f10a_publisher_cgroup.sh` under
`bin/gpu-run`. Execute two independent fresh servers, the fixed 12-prompt
strict suite, canaries, teardown, card health, and compiled P2P-off collective
health.

RESULT -> class-balanced first-100 rates were 17.716072 and 17.381759 tok/s;
the center was 17.548916. That is 33.80 percent of Steve's graph-off r32
51.918757 tok/s and only 1.01 percent above local F07a. Cross-server complete
arrays matched 9/12; each local attempt matched 7/12 publisher arrays. Peak
container RAM was 8.581 GiB and host swap never rose above its pre-run value.
Both server teardowns, all per-card probes, and all compiled collectives passed
with no new Xe fault signature. Evidence root is
`/mnt/vm_8tb/b70/results/f10a_qwen38_fp8_neural_publisher_cgroup/20260831T173435Z/`;
summary SHA256 is `86bfdf34...`.

RESULT -> the 1,091.642460 tok/s c64 result belongs to a different older
profile: graph enabled with only size one captured, a 256-token model limit,
128 slots, and 512 batched tokens. It delivered 17.056913 tok/s per stream.
Graph capture did not prevent the scheduler from serving c64, and graph-off
did not create that aggregate result.

VERDICT -> published launch flags and cgroup are insufficient for a 10-percent
graph-off reproduction on this host. The unavailable publisher OCI and
unpublished host CPU/runtime boundary remain uncontrolled. The local MTP1
FULL graph route at 46.603967 tok/s is 165.57 percent faster than F10a and
remains the daily-driver choice. Do not mix the c64 short-service aggregate
with the strict c1 or 262K profiles.
### 2026-08-31a - Neural.Download r31 lineage correction and ordered rebuild

CONFIG -> Steve packet commit `6aab301f30912c87bfcc7b7982f2fab27eb1eca5`
present in preserved checkout `0948f7c2`; official vLLM XPU base
`f01e24f6c7ff`; kernel commit `1e90ffa672ba`; kernel wheel SHA256
`f3d999060c11`; vLLM source `ac7509e2b1db`; deterministic r15 four-file
overlay; packed-two-row MTP1 RMSNorm r31 overlay. This was a no-GPU image
construction and inspection transaction.

COMMAND -> inspect the public build scripts, Dockerfiles, patches, qualified
result, and raw r32-A container inspection; compare inherited image labels to
the nominal build defaults; verify the upstream kernel Actions artifact and
retained wheel; implement
`vllm/fp8/build_qwen38_fp8_r31_ordered_repro.sh`; then rebuild the lineage as
official f01e -> kernel r13 -> deterministic r15 -> RMSNorm r31 and verify the
seven known final runtime files.

RESULT -> the raw r31 labels prove that the publisher r15 inherited the
`1e90ffa` kernel wheel, while `build-deterministic-compiled-image.sh` defaults
directly to official f01e. The public displayed recipe therefore requires an
unstated caller `BASE_IMAGE` override. The ordered local kernel, r15, and r31
image IDs are respectively `90746d6d1f11`, `82cac0986b54`, and
`3e29569b6d15`. The final runtime manifest passed for the four r15 Python
files, both r31 layernorm copies, and `_xpu_C.abi3.so`. The publisher r31 OCI
ID remains `ba42e928e69c`, but its archive, image history, layer descriptors,
and complete build metadata are not public, so metadata-sensitive image-ID
equality is not attainable from the published inputs alone.

VERDICT -> the known executable lineage and layer order are reconstructed and
content-verified. This corrects the earlier direct-f01e r15 assumption. It is
not yet a runtime speed qualification or a byte-identical publisher OCI
reproduction. The reusable mechanisms and missing publisher evidence are
recorded in
`docs/20260831_neural_r31_image_provenance_and_transfer_ledger.md`.

### 2026-08-31b - Qwen3.8 native W8A8 INT8 closes behind FP8

CONFIG -> Qwen3.8-27B compressed-tensors W8A8 GPTQ, FP16 activations, BF16
KV, TP2, direct oneCCL P2P, FULL decode graph, maximum four sequences,
237,568 context, and image `5dad53a3...`. The tracked source adds a native
oneDNN/XMX `s8 x s8` GEMM, cached scratchpad, optional input dependency, and
an opt-in native per-token INT8 quantizer to `vllm-xpu-kernels` commit
`1e90ffa672ba`.

COMMAND -> pass real-shape FP16 M=1/M=4 numerical and latency oracles, run a
complete 12-prompt MTP1 performance suite and concurrent quality canary, run
bounded target-only Triton-quant and native-quant FULL screens, and attempt a
matched stock Triton FULL control. Inspect speculative metrics and the MTP
safetensor, then tear down and require card plus compiled collective health.

RESULT -> the native GEMM cosine was at least 0.99999988 and all tested FP16
M=1 native quant bytes matched Triton. MTP1 measured 22.7345 tok/s strict and
22.5071 tok/s class-balanced intervals, but accepted 0/6,076 draft tokens and
passed only 24/32 quality requests. Target-only measured 26.5623 tok/s strict
in a one-prompt screen and passed 32/32 quality requests. Native activation
quantization measured 26.0765 tok/s in the same bounded screen and did not
win. The stock Triton INT8 path failed before service at the known Torch FX
empty-`TreeSpec` partition boundary; the custom op is what makes FULL capture
possible. All final card and collective health checks passed.

RESULT -> exact 262,144 MTP1 with FP16 KV failed only the capacity gate:
8.69 GiB was required and 7.90 GiB was available, for an estimated 237,952
token maximum. The loaded MTP artifact is unquantized and incompatible with
useful speculation against this quantized target. The W8A8 model dynamically
quantizes activations before about 160 linears per token, while FP8 W8A16
retains FP16 activations and avoids that repeated reduction/quantization work.

VERDICT -> preserve native INT8 as a research control but do not promote it.
Remove MTP1 from the default route, retain target-only FULL as the best INT8
diagnostic, and return daily-driver work to the qualified FP8 W8A16 MTP1
route. Resume INT8 only around fused norm/quant or quant deduplication plus a
matching quantized MTP artifact. Full evidence is in
`docs/20260831_qwen38_w8a8_native_int8_result.md`.

### 2026-09-01a - FP8 daily driver and Open WebUI v0.11.1 live

CONFIG -> promoted Qwen3.8-27B official FP8 W8A16 shelf, TP2, direct P2P,
FP16 KV, FULL decode graph, fixed MTP1, four slots, 32,768 batched tokens,
262,144 model length, and exact served/container ID `hotschmoe-dd`. Open
WebUI was updated from the stopped June `main` image to official stable
v0.11.1, image ID `6bb1fbe8...`, with the existing `open-webui` data volume,
auth and Ollama disabled, and telemetry disabled.

COMMAND -> verify the pinned vLLM image and launch the promoted shelf through
the whole-box `bin/gpu-run` lease on loopback port 18080. Require per-card and
compiled TP2 collective pre-health, exact `/v1/models` identity, and a
thinking-off deterministic completion. Pull Open WebUI v0.11.1, recreate only
its disposable container on host networking at port 3000 so it can reach the
loopback-only vLLM endpoint, then require Docker health, `/api/config`, and a
backend model census from inside the WebUI container.

RESULT -> both card probes and the compiled collective passed before serving.
The model loaded 14.07 GiB per rank, restored both AOT caches, captured three
FULL decode graphs in four seconds, and exposed 323,202 KV tokens, or 1.23
times one full 262,144-token request. `/v1/models` returned only
`hotschmoe-dd`; the deterministic gate returned exactly `DAILY DRIVER READY`.
Open WebUI reports version 0.11.1, auth false, healthy status, and sees only
`hotschmoe-dd` through its configured backend. A consistent post-migration
volume copy is under
`/mnt/vm_8tb/b70/backups/open-webui-20260901T151600Z/`.

VERDICT -> GO live. The FP8 MTP1 full-context daily driver remains running
under its GPU lease at `127.0.0.1:18080`; Open WebUI is healthy at port 3000.
The WebUI volume was preserved. The backup was taken after v0.11.1 migration
and is not a pre-migration rollback image.

### 2026-09-01b - Open WebUI persisted backend correction

CONFIG -> the live `hotschmoe-dd` FP8 MTP1 server on loopback port 18080 and
Open WebUI v0.11.1 on host networking at port 3000. The preserved WebUI
database still contained its pre-relaunch OpenAI base URL despite the new
container environment.

COMMAND -> probe vLLM health, identity, and a real deterministic completion;
inspect WebUI connection errors and the redacted persistent OpenAI settings;
make a SQLite-consistent database backup; update only
`openai.api_base_urls` to `http://127.0.0.1:18080/v1`; restart WebUI; then
query `/api/models` with an in-container authenticated admin probe without
printing the token or secret.

RESULT -> vLLM never exited: `/health` and `/v1/models` returned HTTP 200,
the served identity remained `hotschmoe-dd` with model length 262,144, and a
generation returned exactly `OK`. WebUI logs instead showed connection
refusals to the stale persisted `192.168.10.5:18080` address. After the
single-key correction, WebUI became healthy and its authenticated model list
returned HTTP 200 with `hotschmoe-dd` present. No post-fix connection error
was logged. The immediate pre-fix database backup is at
`/mnt/vm_8tb/b70/backups/open-webui-20260901T155600Z-url-fix/webui.db`.

VERDICT -> the apparent backend outage was a WebUI persistent-configuration
override, not a vLLM or GPU-serving failure. The WebUI connection is repaired
without changing chats, users, other settings, or the running GPU server.

### 2026-09-01c - FP8 daily-driver agentic serving enabled

CONFIG -> the promoted `hotschmoe-dd` Qwen3.8-27B FP8 W8A16 MTP1 shelf at
262,144 context, vLLM `0.27.2rc1.dev77+gac7509e2b`, Open WebUI v0.11.1, and
local Pi 0.84.3. The checkpoint chat template supports low, medium, and xhigh
reasoning effort and defaults to xhigh.

COMMAND -> add configurable agentic arguments to the shared Qwen3.8 FP8
launcher; enable `--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
`--reasoning-parser qwen3`, and explicit xhigh default chat-template kwargs in
the shelf. Configure Pi's local OpenAI-completions provider, exact model
identity, 262,144/32,768 context/output limits, reasoning-level mapping, and
xhigh default. Stop only the old vLLM container, require both per-card probes
and the compiled P2P-disabled TP2 collective probe, relaunch through
`bin/gpu-run`, and exercise render, non-streaming tools, streaming tools,
reasoning override, authenticated WebUI discovery, and a real Pi read-tool
loop.

RESULT -> both card probes and the compiled ten-iteration `4x5120` collective
passed before launch. The server restored its AOT caches, captured three FULL
decode graphs, retained 323,202 KV tokens, and logged that auto tool choice
was enabled. Non-streaming and streaming `tool_choice: "auto"` requests both
returned `finish_reason=tool_calls`, the requested function name, and valid
city arguments without an error. A default rendered prompt contained the
xhigh instruction and `<think>` opener; a normal default request returned
separate reasoning and content without raw think tags. An explicit
`reasoning_effort: "none"` produced zero reasoning characters and the exact
requested answer. Open WebUI's authenticated model list contained
`hotschmoe-dd`. Pi loaded the configured model, completed a read-only tool
round trip with one tool execution, 63 valid JSON events, no error event, and
no stderr; its settings loader resolved xhigh as the default.

VERDICT -> GO for Open WebUI and local Pi agent use. The previous auto-tool
error is fixed at the vLLM boundary, xhigh is the server and Pi default, and
per-request lower effort or thinking-off overrides remain available. This is
an agentic compatibility qualification, not a new throughput measurement.

### 2026-09-01d - Authenticated LAN daily driver live

CONFIG -> the Qwen3.8 FP8 MTP1 `hotschmoe-dd` shelf, host address
`192.168.10.5/24`, port 18080, existing off-repository daily-driver API key,
and the stale installed `b70-daily-driver.service` that still named the
retired Qwen3.6 NVFP4 route and was enabled but failed.

COMMAND -> make the shared FP8 launcher publish address configurable while
preserving its loopback default; set the daily shelf to `0.0.0.0`; inject the
API key from a read-only mounted secret through a wrapper so neither the key
nor its value appears in Docker image configuration or host command
arguments; add a tracked Qwen3.8 systemd service and pre-health startup
wrapper. Stop the loopback-only server, treat its forced in-flight TP2 drain
as risky, require both per-card checks and the compiled P2P-disabled TP2
collective, then launch the secured LAN route and probe public health,
unauthenticated rejection, authenticated identity and generation, Open
WebUI, and Pi.

RESULT -> both card probes and the ten-iteration compiled `4x5120` collective
passed after teardown. The first authenticated process environment caused a
one-time AOT cache miss; target and draft compilation took about 129 and 27
seconds and saved new artifacts. The server then retained 323,202 KV tokens
and listened on `0.0.0.0:18080`. Health returned HTTP 200 through both
loopback and `192.168.10.5`; unauthenticated LAN `/v1/models` returned 401,
while the same request with the daily-driver key returned 200 and exact model
ID `hotschmoe-dd`. Authenticated LAN generation returned exactly `LAN READY`.
Open WebUI retained model discovery, and Pi completed an authenticated exact
reply with no error or stderr. Repeated health requests from LAN host
`192.168.10.50` reached vLLM. The secret value is absent from Docker image
configuration.

VERDICT -> GO for authenticated LAN clients at
`http://192.168.10.5:18080/v1`. The tracked replacement systemd unit passes
`systemd-analyze verify`, but it is staged rather than installed because the
current user has no non-interactive sudo and the installed unit is root-owned.
The live server is running through the same tracked startup wrapper; install
the replacement unit before relying on reboot persistence.

### 2026-09-01e - Grafana and Prometheus restored beside Open WebUI

CONFIG -> live authenticated `hotschmoe-dd` on port 18080, healthy Open WebUI
v0.11.1 on LAN port 3000, stopped `b70_grafana` and `b70_prometheus`
containers, persistent monitoring data under `/mnt/vm_8tb/b70`, and the
tracked anonymous-viewer Grafana provisioning for both vLLM and SGLang.

COMMAND -> run `bin/monitoring/up.sh`, retain host networking and persistent
data, wait through the fresh Grafana 13.1.0 SQLite migration, probe Grafana
and Open WebUI through `192.168.10.5`, verify Prometheus health, target state,
and `up` query, enumerate provisioned Grafana datasources and dashboards, and
set the existing Open WebUI container restart policy to `unless-stopped`.

RESULT -> Grafana completed its first database migration in about nine
minutes and returned HTTP 200 through loopback and LAN port 3001. Prometheus
returned HTTP 200, scraped `http://127.0.0.1:18080/metrics` without error, and
reported `up{job="b70_daily"}=1`. Grafana provisioned the default Prometheus
datasource plus the vLLM and SGLang dashboards. Open WebUI remained healthy
and returned HTTP 200 through LAN port 3000. All three UI/monitoring
containers now use Docker `unless-stopped` restart policy.

VERDICT -> GO. Open WebUI is live at `http://192.168.10.5:3000`; Grafana is
live at `http://192.168.10.5:3001`; and the current vLLM metrics path is
healthy end to end. The fresh Grafana 13 migration took longer than the old
four-minute launcher note, so the operator message now allows roughly ten
minutes for a new database.

### 2026-09-01f - Steve R50 FP8 package closes the public source gap

CONFIG -> stopped local `hotschmoe-dd`, Steve's September 1
`b70-optimization-lab` commit `6adab048...`, the new
`packages/qwen38-27b-fp8-tp2-b70` entrypoint, its R50 reproduction chain, and
the local requirement for cache-on long-agent use plus exact c2/c4 scaling.

COMMAND -> fetch the external source without changing its preserved detached
worktree; inspect the package README and package JSON, final builders and
launchers, R54-R63 qualification evidence, R77-R82 concurrency localization,
and the pinned/current vLLM Qwen3Next prefix-cache source. Do not touch either
GPU.

RESULT -> the package now publishes the complete R13 -> R15 -> R31 -> R49 ->
R50 source build and content-verification chain. A clean source rebuild
measured 51.579521 tok/s and matched 12/12 outputs against MTP1 and MTP0. The
stronger unrepeated-content R56 diagnostic kept MTP1 at 49.990-53.134 tok/s
through 32K and matched 18/18 target arrays. All publisher performance runs
still force cache zero. The pinned vLLM can enable Qwen3Next hybrid prefix
caching in block-aligned `align` mode, but that route has no publisher
qualification. The newer c2/c4/c64 audit also shows sequential-oracle
mismatches beginning at c2; R77 localizes the first meaningful difference to
layer-1 GDN, and the latest R81 repair remains negative.

VERDICT -> choose graph-off R50 FP8 W8A16 MTP1 as the primary campaign, with
same-image MTP0 as control and a separately gated fallback candidate.
Reproduce c1 and the natural 2K-32K matrix first, extend the 30 tok/s decode
floor through 262K,
then qualify cache-on `align` mode and exact c2/c4 sequential-oracle parity.
The older 1,091 tok/s aggregate curve and local semantic canaries are not
sufficient output-identity evidence. Full review and gates are recorded in
`docs/20260901_steve_r50_fp8_package_review.md`.

### 2026-09-01g - Steve R50 independent replay blocked by artifact identity

CONFIG -> latest public `b70-optimization-lab` main at
`0e8c4c577d40674f0aceb3c5005c24f3d305f951`, the complete public R13 -> R15
-> R31 -> R49 -> R50 build chain, exact Qwen3.8-27B model revision, Steve's
graph-off FlashAttention FP8 W8A16 MTP1 strict c1 launch and benchmark, and
the project two-card lease and health gates.

COMMAND -> run Steve's source-closure verifier and unchanged final-builder
preflight in a pristine detached worktree; inspect patch history, GitHub code
and release assets, and documented registry tags; build a clearly local
candidate from the tracked patch; compare complete and per-section ELF
hashes; compare its container arguments, 31 strict environment values, and
cgroup limits with the clean R55C result; then run the strict 12-prompt suite
with model verification, cache-zero and canary gates, teardown, post-health,
and recovery after the failed speed assertion.

RESULT -> public source closure passed, but the unchanged final builder
stopped before compilation because the tracked R50 patch hashes to
`08a3de4f...` while the builder, qualified image labels, and R55C result
require an unavailable `40ca8c3f...` patch. No final R50/R55C image, wheel,
or library pair was present in public releases, and both documented registry
tags required authentication. The tracked-patch build matched all six
published host `.text`, `.rodata`, and `.data` hashes but not the published
whole-library hashes; the public evidence omits hashes for the
`OFFLOAD_DEVICE_CODE` sections that hold the SYCL device images. The live
launch differed from R55C only in served-model display name, had zero changes
across the selected environment values and identical 9 GiB/12 GiB cgroup
bounds, and passed exact model identity, workload shape, all canaries,
cache-zero, teardown, and post-health. It measured a class-balanced median of
17.203380 tok/s versus Steve's 51.808087 tok/s and the 46.627278 10-percent
floor. The speed failure triggered the documented rebind recovery; both card
and compiled P2P-disabled collective health passed afterward and both leases
were free.

VERDICT -> NO-GO for the locally rebuilt candidate and defer prefix-cache,
long-context, and c2/c4 qualification. The exact missing publication input is
the R50 `40ca8c3f...` patch or, preferably, the clean R55C OCI/library
artifact. Device-code hashes and a complete build receipt are additionally
needed if source-only reproduction is intended. Launch flags, model files,
base sources, benchmark harness, and serving bounds are not missing. The
verified request and DM draft are in
`docs/20260901_steve_r50_reproduction_gap_ledger.md`.

### 2026-09-04a - corrected Steve FP8 recipe reproduces with graph recovery

CONFIG -> fresh full-history Steve source at `8319e096...`, corrected public
release assets, official Qwen3.8 FP8 weights, local no-compiler R156 image,
TP2 MTP1 c1, empty caches, and the exact strict natural-512 workload.

COMMAND -> validate public source and remote asset closure; build R55C, R62,
R139, and R156; verify image/model identities; run leased host and collective
probes; then run R187 graph-off, matched R156 graph-off, and R156
`FULL_DECODE_ONLY` sizes `[1,2]` graph-on with pre/post card and compiled
two-rank health. Compare complete strict token arrays.

RESULT -> closure and every content contract pass. The local host measured
`9.2399 us` async launch, `52.1136 us` launch+sync, `200.1721 us` RMSNorm,
and `74.6 us` two-row all-reduce. R187 graph-off measured `17.917452 tok/s`;
matched R156 graph-off measured `16.845797`; graph-on measured `49.675873`,
or `2.948858x` and `90.976%` of Steve's R156 MTP1 center. The matched R156
pair passed 12/12 complete token arrays; R187 versus R156 graph-on was 9/12
because that comparison also changes the Inductor split policy. All workload,
cache-zero, canary, identity, teardown, card-health, and compiled-collective
gates passed. No reset or reboot was needed.

VERDICT -> GO for continued local c1 work on the documented graph variant;
the former public artifact blocker is closed and the remaining graph-off
speed gap is explained by measured host submission/collective latency. Do not
shelf-promote yet: a second matched pair, concurrency, long-context, and
cache-on qualification remain open. Full evidence is in
`docs/20260904_steve_r187_independent_replay.md`.

### 2026-09-04b - R187 MTP5 XPU graph reaches 72.245 tok/s

CONFIG -> corrected Steve R156 image and official Qwen3.8 FP8 weights, R187
whole-graph compile, TP2 MTP5 c1, `FULL_DECODE_ONLY` XPU graph capture sizes
`[1,2,3,4,5,6]`, empty cache, 1K allocation, and the strict natural-512
suite.

COMMAND -> extend the fail-closed Steve replay harness with an experimental
MTP5 graph profile; run it through the two-card lease, image/model identity,
exact graph-config assertions, pre/post card and compiled P2P-off collective
health, cache-zero and canary gates, teardown, and kernel-journal check; then
compare complete token arrays with the prior R156 MTP1 graph attempt.

RESULT -> MTP5 graph measured `72.245076 tok/s`, `1.454329x` or `45.433%`
above MTP1 graph's `49.675873 tok/s`. All workload, cache-zero, canary,
served-ID, health, teardown, and kernel gates passed; swap remained at
`646792 KiB`. Outputs were 12/12 exact against the same-image, same-whole-
graph R187 MTP1 graph-off reference. They were 9/12 against R156 MTP1
graph-on, with the same three late divergences already observed between R187
whole-graph and R156 piecewise profiles.

VERDICT -> GO as a coherent experimental c1 speed lead, not for shelf
promotion or a controlled speed claim. The requested MTP1 graph comparison
changes both MTP depth and Inductor split policy; run and repeat same-depth
graph-off/on controls before promotion. Evidence is appended to
`docs/20260904_steve_r187_independent_replay.md`.

### 2026-09-04c - R187 MTP5 daily envelope is live with real cache hits

CONFIG -> R187 whole-graph FP8 W8A16 MTP5 XPU graph, 237,568 context, four
request slots, 32,768 scheduler budget, FP16 KV, 0.96 GPU utilization,
`align` prefix caching, graph sizes 1 through 24, agentic parsers, and an
explicit served model ID on loopback port 18080.

COMMAND -> add a tracked foreground daily launcher that holds both
`bin/gpu-run` leases, verifies image/model identity, and brackets the server
with card and compiled P2P-off collective health. Start it durably, verify the
live config and model ID, then run exact short-chat and repeated 3,098-token
cache smokes.

RESULT -> pre-health passed; the server allocated 290,188 KV tokens, captured
four c1-c4 graph descriptors in 3 seconds, and returned exactly `DAILY READY`.
The repeated long prompt returned identical output and changed from zero to
1,664 cached tokens; engine metrics agree. The server remains live with both
leases held. No matching Xe fault appeared before handoff. Post-health is
pending teardown.

VERDICT -> READY for the planned quality test, not qualified for speed,
stability, or shelf promotion. Use the tracked stop action so teardown and
post-health run. Full configuration and evidence are in
`docs/20260904_steve_r187_independent_replay.md`.

### 2026-09-04d - MTP5 daily candidate exposed through secured LAN frontdoor

CONFIG -> Keep the live R187 whole-graph FP8 W8A16 MTP5 server at 237,568
context with cache enabled. Move its Docker-published vLLM endpoint to
`127.0.0.1:18124`; expose an API-key frontdoor on `0.0.0.0:18080` using the
existing off-repository daily-driver key and the exact served model ID
`qwen3.8-27b-FP8-official-W8A16-mtp5-r187-xpugraph-cacheon-ctx237568-daily`.

COMMAND -> Add the tracked key-checking streaming proxy to the foreground
leased launcher; preflight it against the old loopback server; perform a
controlled stop; then launch a fresh two-card lifecycle with image/model
verification and pre-card/compiled-collective health. Probe health, auth
rejection, authenticated identity, exact chat, and streaming through the
host LAN address without putting the key in command arguments.

RESULT -> Proxy preflight passed. During the controlled stop, the old resident
Bash process read the newly edited cleanup body and lacked its new variables;
model workers drained cleanly, but the lifecycle returned 1. Fail-closed
cleanup ran rebind recovery, passed card and compiled P2P-disabled collective
post-health, and released both leases. The fresh lifecycle passed pre-health,
loaded the compiled graph target, retained 290,188 KV tokens, and listened as
intended: frontdoor `0.0.0.0:18080`, backend `127.0.0.1:18124`. Through
`192.168.10.5`, health returned 200, unauthenticated `/v1/models` returned
401, authenticated identity returned 200 with the exact ID, chat returned
exactly `LAN READY`, and streaming reached `[DONE]`. No key appears in process
arguments and no matching Xe fault was present. The server remains live with
both leases held. The LAN receipt SHA-256 is
`a2e12fbdbee37b90fd2f0f570abe0451d2eb302d68905027ae717b62740b2c0c`.

VERDICT -> READY for LAN quality testing at
`http://192.168.10.5:18080/v1` with the existing daily-driver API key. This is
not a speed, stability, or shelf claim. Post-health for the current lifecycle
remains pending until its tracked teardown.

## 2026-09-08 -- Deferred R187 KV offload and calibrated FP8 research

CONFIG -> User requests research/plan only while hotschmoe-dd is in use.
Current R187/MTP3 TP2 official Qwen3.8 FP8 W8A16 uses FP16 KV and a
292968-token shared pool; container memory limit is 32 GiB.

COMMAND -> Read launchers, installed container Python source with shell
tools only, Docker memory/image metadata, /v1/models, host RAM, retained
calibration code, July archival evidence, and upstream vLLM documentation.

RESULT -> Native OffloadingConnector has XPU/HMA/Mamba/preemption source
paths. Fresh model-specific calibration is required; old Qwen3.6 scales
are not transferable. July follow-up revised the early claim that
uncalibrated KV alone caused repetition. Wrote the deferred test matrix,
capacity/cgroup budgeting, calibration procedure, quality/performance gates
and rollback in vllm/fp8/20260908_kv_offload_calibrated_fp8_plan.md.

VERDICT -> Plan only; no inference, GPU workload, restart, runtime edit or
calibration. Execute only in a later maintenance window. Source support
is not qualification of the current hybrid/MTP3/graph combination.


## 2026-09-08 -- KV campaign started; baseline and DMA milestone

CONFIG -> Authorized R187/MTP3 TP2 official FP8 W8A16 cache campaign.
COMMAND -> Preserve service/runtime identity; stop original service through
its cleanup; run isolated FP16 baseline under bin/gpu-run and native DMA
oracles. Implementation/evidence in vllm/fp8/20260908_kv_campaign.md.
RESULT -> 48 short checks, 32 tool-history checks, 32K and four 150K retrievals
passed. Four evicted-history requests required 373.418 s total with no cache
hits. Two simultaneous 150K requests completed in 219.141 s but serialized
without preemptions. Native 4 MiB host round trips were exact on both cards.
A harness queue-file race ended the lifecycle before the new growth case;
model shutdown was clean and per-card/compiled two-rank post-health passed.
VERDICT -> Baseline/DMA checkpoint only; active recovery and offload still
unqualified. A1 adds a 32 GiB CPU tier with the same 64 GiB cgroup.
Commits a3f2e23, f926343, 66370ab preserve implementation and evidence.

## 2026-09-08 KV campaign: native offload and fresh calibration

CONFIG -> Pinned R187 image f46780e1a72c, Qwen3.8 official FP8 W8A16,
MTP3 TP2, auto KV, native CPU32, 64 GiB cgroup. Isolated campaign endpoint.
COMMAND -> vllm/fp8/kv_campaign_server.py with queued bounded probes;
raw root /mnt/vm_8tb/b70/results/kv_campaign_20260908/.
RESULT -> Stock A1 had no 150K A/B/C/A history hits. Forced 132K growth
completed 16384 output tokens in 394.110 s with six preemptions and two
native loads. A guarded Mamba draft-group fallback repair restored 148928
of 150045 tokens correctly; final revisit TTFT 8.857 s. Cold prefills were
slower in the early-import diagnostic, so it is not a promoted timing arm.
A1fix2 passed 48 short checks and 32 interleaved 44K tool-history checks;
clean teardown plus per-card and compiled collective post-health passed.
Fresh eager/no-prefix calibration passed 256 short samples and contexts
through 180K; sustained continuations are still running. Both ranks show
all 17 attention layers with finite observations. Scale rule frozen before
held-out eval: 1.10 * max observed amax across ranks / 448, floor 1e-6.
VERDICT -> Functional native history reload established; performance and
calibrated-FP8 qualification remain in progress. Daily-driver defaults have
not been promoted. Details: vllm/fp8/20260908_kv_campaign.md. Existing user
JOURNAL changes are preserved; milestone evidence is committed separately.

2026-09-08 KV campaign scope: FP8 deferred on user direction.
CONFIG -> Frozen calibrated E4M3 artifact, MTP3, prefix on/off controls.
COMMAND -> Eager repeat and graph/prefix-off repeat; clean post-health.
RESULT -> Eager/prefix-on corrupted one long response; graph/prefix-off
passed three exact 2048-token responses. Both post-health runs passed.
VERDICT -> No FP8 KV promotion. Preserve evidence, continue RAM offload with
existing auto/FP16 KV. See vllm/fp8/20260908_kv_campaign.md.

2026-09-08 KV campaign identity correction.
CONFIG -> Hook launcher included an empty trailing PYTHONPATH component.
COMMAND -> CPU import-origin probes and source/installed _xpu_ops.py hashes.
RESULT -> Hook arms selected the image's /workspace/vllm source checkout,
not the installed patched serving package. A0 and stock A1 were unaffected.
VERDICT -> Withdraw baseline-matched conclusions for all prior hook arms,
including FP8 diagnostics/calibration. Fix search path; qualify packaged
FP16 offload on the installed package. FP8 remains deferred. Full evidence:
vllm/fp8/20260908_kv_campaign.md.

### 2026-09-08 - Corrected-package offload candidate rejected

## Corrected offload candidate A1pack1: rejected on oversized tool history

CONFIG -> Installed R187 package, Python-only partial group repair, MTP3,
FP16 KV, 32 GiB CPU tier, 64 GiB cgroup. Raw evidence is
/mnt/vm_8tb/b70/results/kv_campaign_20260908/a1pack1/.
COMMAND -> Complete 12-job matrix, including three fresh-salt 150K A/B/C/A
traces, four concurrent 8000-record tool histories, full 164 HumanEval+,
forced 132K+8192 growth, 180K recall, clean teardown and post-health.
RESULT -> Reuse traces passed in 290.815/283.134/283.286 s, with confirmed
CPU loads and warm TTFT 10.466/2.570/about 2.6 s. Cancellation and xhigh
thinking checks passed. Oversized tool histories failed: session 1's first
tool turn emitted 512 literal exclamation marks and no tool call; only
25 checks passed. That workload recorded two preemptions and CPU reloads.
A zero cached_tokens field does not exclude a reload after preemption.
HumanEval+ scored 157/164 base and 151/164 plus. Forced growth completed
in 335.196 s and 180K recall passed in 106.705 s. Neither diagnostic
throughput nor independent coding success repairs the tool corruption.
Teardown and per-card/compiled two-rank post-health passed (exit.rc=0).
VERDICT -> Not promotable. A0b replays all 12 jobs on the frozen original
image with the same 64 GiB limit; it must classify the baseline behavior.
The three guide outputs have identical parsed Python ASTs but differ in
comments/explanation, so their strict byte-repeat gate remains failed.

## Next bounded candidate: merged upstream scheduler fixes

CONFIG -> Preserve the original native image; deliberately port scheduler
source changes from merged PRs 52807, 54288, and 52771. These correct the
fresh load-region scan, final committed-token store watermark, and shared
MTP group/tail handling respectively. This replaces the partial hook.
COMMAND -> build_merged_offload_image.py and the tracked three-PR patch;
CPU regression qualification precedes another leased GPU attempt.
RESULT -> Source hunks match the installed scheduler after mapping the
new upstream use_eagle_block_drop name onto this pinned package's existing
use_eagle predicate. No speculative capability refactor is included.
VERDICT -> Prepared, not GPU-qualified. Open PR 54165 is DFlash-specific;
open superseded draft 53479 is not an accepted patch source.
Primary sources: https://github.com/vllm-project/vllm/pull/52807,
https://github.com/vllm-project/vllm/pull/54288,
https://github.com/vllm-project/vllm/pull/52771. API metadata and diffs are
archived under the campaign source directory. User now permits a bounded
FP8 KV retest only AFTER the other tasks; none has been started.

### 2026-09-08 - Merged scheduler CPU regression qualification

## Merged scheduler backport: CPU qualification and queued GPU gate

CONFIG -> Derived image
sha256:eb852f45140db6f812dfda8867d59795a91a831266dedf37c8a666bf7b200f4f,
base f46780e1a72c, native bytes unchanged. Only installed scheduler.py is
replaced: 616e7fd4cb0d09064cbc4d5735f607b37964c6be3b81e26de00d5913e0a9a3e3
becomes 0d2fd9a20e02e2b1560d757658ded738ae6c5d7cc292bf75d20c49636f6338b0.
No PYTHONPATH hook or custom entrypoint is used.
COMMAND -> Port the image's scheduler test suite plus merged-PR regressions;
run Docker without GPU devices, then run the targeted tests against stock
as a negative control. test_merged_offload_image.py reproduces extraction,
strict source-hash checks and the tracked regression-port JSON.
RESULT -> 124/124 scheduler tests pass (23.47 s). The nine-test negative
control on stock fails seven and passes two, covering the missing sparse
load boundary, terminal-slot watermark, shared-MTP annotations and widened
lookup boundary. Initial fixture attempts failed before exercising these
paths; raw logs are retained. The final fixture only mocks platform hybrid
capability for CPU scheduling, uses the pinned partial_tail_offloads API,
and preserves the existing block-hash setter. No scheduler method is mocked.
Raw evidence: upstream-cpu-tests/pytest-v3.log and negative-control.log.
VERDICT -> CPU-qualified for one bounded GPU attempt, not serving-qualified.
A1merged waits for A0b's healthy teardown, then runs C1/C4 and oversized
tool histories first. Any failed gate stops further workload submission and
still performs teardown/health. If those pass, the remaining reuse, cancel,
thinking, coding, pressure, 180K and guide probes follow. A0b also fails
strict guide byte-repeat while passing coherence, so that variation is
not specific to the CPU connector. Critical tool corruption remains an
unconditional rejection, independent of whether the baseline also has it.


### 2026-09-08 - Matched reuse benefit and baseline overload deadline failure

CONFIG -> A0b original R187, FP16 KV, no offload, cgroup 64 GiB; same
12-job matrix as A1pack1. Tool test: four distinct 8000-record histories,
about 88K tokens each, four tool/answer turns, 600 s per-request deadline.
COMMAND -> Three A/B/C/A reuse repetitions, cancellation, xhigh thinking,
then oversized tools. Continue the queued coding/pressure/long-context
controls and clean teardown; they are still in progress at this entry.
RESULT -> Baseline reuse 373.923/374.079/374.203 s versus rejected A1pack1
290.815/283.134/283.286 s: 23.612 percent lower mean trace time. All recall
answers passed. A1pack1 peak cgroup 37050257408 bytes, no OOM/max events.
Baseline cancellation passed in 115.158 s and xhigh checks in 9.231 s.
Baseline tools failed a 600 s timeout awaiting a tool-call response header;
seven preemptions were already visible in the progress metrics snapshot.
The original probe's pool.map exception prevented writing its aggregate
response JSON. Do not claim baseline punctuation corruption or attribute
A1pack1's corruption to offload from this incomplete coherence control.
VERDICT -> RAM reload benefit is measured on the stated histories, but
A1pack1 remains rejected for corruption. The baseline also fails the
bounded overload responsiveness gate. Updated probe preserves each session
row immediately and records timeout/schema failures independently, without
changing valid requests, prompts or deadlines. Two CPU regression cases
confirm that one failed session preserves the other 24 successful checks.
Future probe runs record their changed source SHA. A1merged remains next;
INT4 follows its healthy teardown. No FP8 KV retest has been started.


### 2026-09-08 - Baseline raw coding outputs confirm corruption with FP16 KV

CONFIG -> A0b original FP8-weight R187 recipe, FP16 KV, prefix cache on,
no CPU connector; coding follows the timed-out oversized tool workload.
COMMAND -> Grade all 164 HumanEval+ tasks, then compare raw solutions with
A1pack1 and scan for long repeated punctuation before trusting sanitized code.
RESULT -> A0b scored 152/164 base and 147/164 plus, versus A1pack1's
157/164 and 151/164. Raw solutions are byte-identical on 152/164 tasks.
Five baseline outputs contain 1383-1658 consecutive exclamation marks:
HumanEval/81, /129, /130, /132 and /147. Their raw SHA/offsets are recorded
in a0b/10-code/raw-corruption-audit.json. A1pack1's coding outputs contain
no such runs. Sanitization can discard corrupted tails, so aggregate pass@1
alone is insufficient. Baseline coding corruption is now directly observed;
the prior tool timeout's missing responses still cannot establish tool
corruption. Causation by prior overload, prefix reuse or another state path
is not established by this ordering alone.
VERDICT -> A0b is invalid as a clean quantization-quality reference. Add one
fresh prefix-cache-off FP8-weight/FP16-KV coding control, before any long
stress, after A1merged's teardown. This is NOT an FP8 KV retest. INT4 strict
qualification is resequenced behind that reference. New code probes retain
raw symbol-loop audits and reject corruption even if sanitized code passes.
compare_code.py refuses a corrupted baseline or candidate. The daily INT4
coordinator can select the clean coding baseline independently of its
unchanged workload templates. All earlier raw grades/reports are preserved.


### 2026-09-08 - A0b teardown and matched forced-growth comparison

CONFIG -> Same 64 GiB cgroup, installed R187 package, FP16 KV, MTP3;
A0b has no connector, A1pack1 has the rejected partial repair and CPU32.
COMMAND -> Matched two-request 132071-token prompts plus 8192 forced output
tokens each; then 180K recall and leased teardown/health.
RESULT -> A0b forced growth completed 16384 output tokens in 467.463 s,
with 147 preemptions and no CPU loads. A1pack1 completed in 335.196 s,
with two preemptions and 5122293760 CPU-to-GPU loaded bytes. The external
prefix-hit counter did not increase in this pressure case; actual transfer
bytes establish reload activity. A0b 180K recall passed. A0b clean shutdown,
per-card and compiled two-rank post-health passed (exit.rc=0). Raw paired
summary: matched-forced-growth-summary.json; original response/metric files
remain under each arm. A1merged has acquired the lease and begun startup.
VERDICT -> This forced-length workload is a pressure diagnostic, not a
useful-output throughput or coherence claim. It demonstrates a substantial
recompute/preemption difference but cannot override either arm's recorded
quality failures. Both main reuse and active-growth benefits require a
candidate that passes the correctness gates and a second clean lifecycle.


### 2026-09-08 - Merged offload candidate rejected; bounded RAM campaign concluded

CONFIG -> A1merged: frozen native R187 stack plus merged scheduler fixes
52771/52807/54288, FP16 KV, MTP3, CPU32, cgroup64, prefix cache on. Same
10.13 GiB/rank and 292968 GPU-token capacity. No custom Python entrypoint.
COMMAND -> C1/C4, then four 8000-record tool histories with per-session
response preservation; fail-fast on correctness, clean stop and post-health.
RESULT -> C1/C4 passed in 9.058/7.467 s. Tool test failed after 239.608 s:
27 checks passed and one failed. Session0's second answer emitted 512
exclamation marks rather than 731. Prompt88474, reported cached87360,
37.054 s response latency. All three other sessions passed eight checks
each. metrics-before/after and metric-delta.json preserve transfer evidence.
Remaining long benchmarks were not submitted after rejection. Teardown and
per-card/compiled two-rank post-health passed, exit.rc=0.
VERDICT -> Merged scheduler fixes enable functional RAM reuse but do not
qualify this MTP/prefix-cache recipe. Do not promote either offload image.
Stop further RAM patch arms in this campaign and proceed to the clean coding
reference and exact INT4 recipe. This is not a claim that RAM offload is
universally broken; it is a rejection of the tested local serving combination.
The existing upstream Mamba prefix-state issue remains an unproven causal
lead. Keep FP16 KV. No FP8 KV retest has been started.


### 2026-09-08 - Clean FP16-cache reference and INT4 preflight correction

CONFIG -> A0clean: original FP8 weights/R187 native image, FP16 KV, MTP3,
200K/c4 shape, 64 GiB cgroup, prefix caching disabled; coding before long
stress. This is a clean quality control, not an FP8 KV experiment.
COMMAND -> C1/C4, all164 HumanEval+, three2048-token guides, clean teardown.
RESULT -> Every job and post-health passed, exit.rc=0. Coding157/164base,
152/164plus, no raw symbol loops; generation529.6s and grading53.0s.
Guides repeated byte-exactly in75.597s total. This control changes both
cache mode and preceding workload history, so it does not isolate the
baseline corruption's root cause. It supplies the coherent coding reference
for INT4: retain the frozen <=2 net additional failure limit and review
all newly failed tasks, with unconditional rejection of raw corruption.

CONFIG -> First INT4 strict preflight, exact published R276 image.
COMMAND -> Per-card health and compiled two-rank P2P0/P2P1 controls.
RESULT -> Original attempt strict/mtp0-a passed per-card/P2P0, then the
P2P1 shell guard refused to run because I_KNOW_P2P_WEDGES=1 was missing.
No model serving or P2P1 collective ran in that attempt. The generic error
path unnecessarily performed a non-reboot reset; recovery and post-health
passed. Preserve exit.rc=1 as an infrastructure failure, not hardware or
model evidence. The runner now supplies the opt-in only to the explicitly
requested scoped P2P1 preflight, after P2P0, and distinguishes probe rc2
from hardware failure. Two CPU lifecycle regressions pass. No bin/ changes.
The fresh strict-retry1/mtp0-a has now passed all three real preflights,
including P2P1, and is starting the model. Published serving settings and
12GiB/16GiB cgroup/swap limits are unchanged.
VERDICT -> Coherent reference accepted for bounded coding comparison.
INT4 model speed, parity, quality and serving qualification remain pending.

### 2026-09-08 - R276 INT4 strict replica qualifies at 100.21 tok/s

CONFIG -> Published R276 image521eb277/source54aefaf0, AutoRound INT4
bce40cac tensors with R212 GPTQ routing, TP2, FP16 KV, prefix caching off,
1024 context/seq1/batch1024/util0.95, published 12GiB/16GiB cgroup/swap.
COMMAND -> run_strict_replica.py --out
/mnt/vm_8tb/b70/results/int4_replica_20260908/strict-retry1; two fresh MTP0
and two fresh MTP4 lifecycles, scoped per-card/P2P0/P2P1 preflight and
teardown/per-card/P2P0 post-health for each. Author 12-prompt/six-class
suite, temperature0/seed42, natural EOS policy with 512-token cap.
RESULT -> All four workload/canary/cache-identity and lifecycle gates pass.
MTP0 pair, each MTP4 against MTP0, and MTP4 pair all match 12/12 complete
token arrays. No cache hits. Class-balanced first99-interval rates:
MTP0 47.08406/47.09156; MTP4 100.53594/99.89262 tok/s, pair mean100.21428.
MTP4 full-output decode medians96.73944/96.45728 and wall-throughput
medians93.99733/94.15117 tok/s; median TTFT154.00/141.62ms. These metrics
are different aggregations, not interchangeable. All exit.rc=0; PASSED
written only after the final health and token comparison.
VERDICT -> Exact short-context replica meets frozen >=95tok/s target.
Not yet a daily-driver qualification. The separately pinned 200K/c4 plain
prefix-off profile has started, with clean A0clean coding reference,
FP16 KV and no CPU offload. FP8 KV remains deferred. Prepared authenticated
frontdoor smoke is syntax-checked only; no public endpoint promotion yet.

### 2026-09-08 - INT4 daily coding and long tool review

CONFIG -> R276 AutoRound INT4/MTP4, FP16 KV, prefix-off 200K/c4 daily
profile, util0.96, batch32768, 64GiB cgroup. No CPU offload.
COMMAND -> run_daily_replica.py with strict-retry1 gate, A0b workload
prompts and clean A0clean/10-code coding comparator; raw root
/mnt/vm_8tb/b70/results/int4_replica_20260908/daily-prefix-off.
RESULT -> C1/C4 exact-answer checks pass; three2048-token guides pass
byte-exact repeat in44.362s total. Actual reported KV15.62GiB/rank,
451562tokens versus FP8 control10.13GiB/292968tokens. Four concurrent
8000-record (~88K-token) tool histories pass32/32 checks in1521.243s.
All164 coding outputs generated in339.0s, graded in49.2s, no raw symbol
corruption. INT4 scores158base/150plus versus clean FP8 157base/152plus.
Paired review in20260908_coding_failure_review.json: new base91 and
new plus97/125/151/154; recover33/132 in both metrics. Negative-modulo,
word-boundary, whitespace and empty-string mistakes are real regressions;
bool interpretation remains scored as a failure. Do not rescore or claim
no-loss parity. The FP8 task132 was incomplete at the shared2048 cap.
VERDICT -> Within frozen <=2 net additional failures, exactly on plus
boundary. Manual review accepts this bounded coding tradeoff for continued
qualification; no critical runtime/tool corruption found. This is not a
Terminal-Bench ranking. Reuse, cancellation, thinking,180K/199K,pressure,
profile,teardown/post-health and fresh serving lifecycle remain pending.

### 2026-09-08 - INT4 daily workload and profile qualification completes

CONFIG -> R276 INT4/MTP4, 200K/c4/batch32768/util0.96, FP16 KV,
prefix caching off, no CPU connector,64GiB cgroup; same daily-prefix-off
lifecycle as the coding/tool milestone.
COMMAND -> Complete planned long/reuse/cancellation/thinking/pressure and
single-profile jobs. After pressure, add95-postpressure-c4 and
96-postpressure-tools in the existing leased queue before98-profile.
RESULT -> All12 planned jobs pass. 150045-token A/B/C/A recall passes in
374.775s, TTFT93.25/93.61/93.70/93.72s; no cache reuse claimed.
Cancellation plus two exact repeated followups pass297.475s. Default
xhigh concurrent checks pass.180K and199037-token recall pass (199K
140.055s,139.887s TTFT). Four132071-token prompts forced to8192 outputs
each complete in639.777s (32768 output tokens), TTFT92.102/171.822/
539.101/239.155s. Actual preemption delta0: scheduler queued the fourth
request until capacity freed. This is overload/queueing evidence, NOT an
INT4 eviction/reload validation or matched FP8-C2 throughput result.
After pressure,24/24 C4 answers and32/32 short tool checks pass.
Single32753-token profile returns correct recall; both ranks have matching
948 c10d collective CPU calls plus948 enclosing vLLM wrappers each.
Per rank:597 allreduce calls (132 at[32753,5120],351 at[1,5120],114 at
[5,5120]);351 allgather calls including one[32753,2560] input. All have
completed CPU entry/return durations. Do not double-count the wrappers or
infer hidden graph device-collective counts. Observed host-event waits
382/343 differ; the trace does not establish a one-fence-per-token claim.
Peak cgroup14751510528bytes, no max/OOM events. Clean teardown, per-card
and compiled TP2 P2P0 post-health pass; exit.rc0, WORKLOADS_PASSED present.
VERDICT -> Daily benchmark qualification accepted with the documented
158/150 vs157/152 coding tradeoff. Keep RAM offload disabled after the
separate negative campaign; no FP8 KV test. Write pinned qualification
manifest and start a fresh authenticated hotschmoe-dd launcher lifecycle.
Startup/stop validation and final serving restoration remain in progress.

### 2026-09-08 - hotschmoe-dd restored on qualified INT4; startup unit ready

CONFIG -> Guarded R276 INT4/MTP4 daily-prefix-off launcher, pinned image,
source/model/config, FP16 KV, no CPU offload, 200K/c4, xhigh default,
API-key frontdoor18080/backend18124, stable alias hotschmoe-dd.
COMMAND -> Fresh production lifecycle20260908T104540Z: actual systemd
readiness helper, missing/wrong/valid API-key checks, four concurrent
answers, three2048-token guides,32 tool checks, exact new ExecStop command.
RESULT -> All pass. Guides match the previous daily lifecycle byte-for-byte
and repeat exactly,44.420s total. Stop retains the lease through clean
teardown/per-card/compiled P2P0 health; parent and server exit.rc0.
Then launch final detached session20260908T105516Z (parent PID189318).
Final readiness/auth/concurrent answers and24 xhigh checks pass. Immutable
image521eb277 verified running, /health successful, no exit marker.
Raw final-serving-validation.json records current identity and unit hash.
VERDICT -> hotschmoe-dd restored on INT4, retaining the documented two-net
extended-coding-test tradeoff. No RAM offload or FP8 KV promotion. The repo
systemd unit now uses the tested INT4 start/stop commands; model registry
maps the public alias to the physical INT4 identity. Unit verification and
shell syntax pass. Installed unit remains unchanged/inactive because sudo
needs the user's password. The installer backs up/enables boot configuration
without interrupting the manual instance. Actual systemd boot untested.
User command: sudo bash
/mnt/vm_8tb/github/b70_ai_things/vllm/fp8/install-hotschmoe-dd-systemd.sh
Handoff: vllm/int4/20260908_hotschmoe_dd_handoff.md. Preserve unrelated dirty
user files and earlier uncommitted journal entries. FP8 KV remains deferred.

### 2026-09-08 - GPU prefix caching is required for the INT4 daily driver

CONFIG -> User explicitly requests enabling conversational prefix reuse.
Start with exact qualified R276 image/INT4 weights/MTP4/FP16 KV/200K/c4.
Only configuration delta: --no-enable-prefix-caching becomes
--enable-prefix-caching. Environment, native stack and CPU tier0 unchanged.
COMMAND -> run_prefix_campaign.py, raw root
/mnt/vm_8tb/b70/results/int4_prefix_20260908/stock-mtp4. The idle serving
instance stopped cleanly through its owned lease and post-health. Candidate
starts only after prior exit.rc0. Register descriptive experiment IDs.
RESULT -> Candidate starting; no prefix qualification claim yet. Planned:
C4 answers, repeated guides,150K A/A/B/C/D/A/A (four distinct histories
exceed451K prior pool), four88K growing tool histories, cancellation,xhigh,
four110K+8K growth pressure, postpressure answers, four132K tool histories,
all164 coding problems after stress, final canaries and199K recall.
Inspect actual pool/admission/preemptions; do not infer eviction from request
count. Require matching recall, warm cached tokens >=90% of prompt and
warm TTFT <=25% of its cold request. Inspect growing tool-history hits,
all raw corruption/new coding failures and clean lifecycle health. Existing
cache-off INT4 controls are the paired correctness reference. No new profile
unless a changed native/runtime path justifies one; the matched R276 daily
collective profile is already qualified.
VERDICT -> Prefix-off serving is not the requested conversational outcome.
Promote GPU prefix caching only after measured reuse and correctness; if
stock MTP4 fails, diagnose the concrete failure and test a bounded fix or
MTP control. RAM offload and FP8 KV remain outside this campaign. Update
startup qualification and restore hotschmoe-dd, commit/push at milestones.
Upstream leads checked today (not applied or proven local fixes):
https://github.com/vllm-project/vllm/issues/53912
https://github.com/vllm-project/vllm/pull/48375
https://github.com/vllm-project/vllm/pull/43650

### 2026-09-08 - Review first prefix-on guide divergence

CONFIG -> Stock R276 MTP4, GPU prefix caching on, FP16 KV, no CPU tier.
COMMAND -> stock-mtp4 C4 and three2048-token guides, then fail-fast teardown.
RESULT -> C4 passes. Guides all coherent but byte-repeat flag false:
first two hashes3c52c745..., third99452102... exactly matches all three
cache-off controls. All text through the last closed code block is identical
across all six responses; only the trailing prose changes (syntax-note text
versus complexity heading). All three prompt-cache-hit counts are0, so no
cached prompt was exercised. Preserve raw job rc1. Lifecycle exit.rc0 and
post-health pass. Prefix-enabled pool446332tokens versus451562cache-off.
VERDICT -> This is not evidence of cache-state corruption or a prefix hit.
Allow a narrowly reviewed noncritical guide-tail variation only when every
coherence row passes and the complete text through the final closed code
block matches every cache-off guide. Changed code/earlier prose/loops still
fail. Keep byte-repeat claim false. Fresh stock-mtp4-b repeats and continues
the cache tests; retain all raw flags plus guide-review.json for any accepted
exception. No image or native-code change. Add one final cached-request
profile on this prefix-enabled graph to measure its actual collective shape
and paired rank entry/return counts; profiler inactive during timing tests.

### 2026-09-08 - Stock INT4 GPU prefix reuse works on long histories

CONFIG -> stock-mtp4-b, same R276 image/weights/MTP4/FP16 KV, prefix on,
446332-token pool, CPU tier0. Profiler configured but inactive during tests.
COMMAND -> 150K A/A/B/C/D/A/A and four8000-record growing tool histories.
RESULT -> All seven recall responses exact/correct. First cold A TTFT94.203s;
immediate repeat1.648s with148928/150045 cached prompt tokens. After four
distinct histories, A reports0 cached and93.756s (evicted); its next repeat
again hits148928 tokens and takes1.635s. No CPU reload involved.
Four growing tool histories pass32/32 checks in218.022s versus1521.243s
for cache-off INT4. All28 followups reuse98.53-98.85% of prompt tokens;
followup wall latency1.200-19.825s (initial concurrent prefills overlap).
VERDICT -> Actual GPU prefix reuse and post-eviction repopulation measured,
with correct answers/tool state so far. This addresses full recomputation
between turns while history remains cached. Cancellation, tighter growth
pressure, larger eviction tool histories, poststress coding/profile and
post-health are still running; no serving promotion yet. The reviewed
zero-hit guide-tail divergence remains recorded separately.

### 2026-09-08 - Reject stock INT4 prefix under oversized tool-history churn

CONFIG -> stock-mtp4-b; exact R276 image, MTP4, FP16 KV, prefix enabled,
446332 nominal GPU KV tokens, four admitted sequences, no RAM offload.
COMMAND -> run_prefix_campaign.py followed by manually queued postfailure
coding, coherence, 199K recall and warm collective-profile diagnostics.
RESULT -> Normal four88K growing histories passed32/32; an additional unsalted
(shared-cache) four2000-record control passed32/32 in32.292s. Four110071-token
prompts plus8192 output tokens each finished in519.040s but scheduler
preemption delta was0: this did not force active-sequence preemption.
The larger four132K-history run failed:23 checks in1378.876s, one session's
first answer timed out after900.009s (no response captured), and another
session's third tool request returned512 exclamation marks, finish=length,
prompt132502, cached_tokens0, created_cache_tokens131456. Two sessions
completed8/8. This is actual corruption under cache churn, not merely slow
queueing, and not proof that the bad request itself reused a corrupt hit.
Subsequent coding completed without symbol loops (157 base/149 plus of164),
short C4 coherence and199K recall passed. Lifecycle/profile evidence pending
when this entry was written. Raw evidence: results/int4_prefix_20260908/
stock-mtp4-b under the runtime root; manual-diagnostics.json records a paused
coordinator so the leased runner could finish postfailure diagnostics.
VERDICT -> Reject stock MTP4+prefix for daily promotion. Investigate the
installed Mamba manager ignoring drop_eagle_block using a Python-only
backport of upstream PR48375, with CPU regression and exact native-hash
comparison before GPU retest. This is a candidate, not a proven root cause.
Public serving has not been promoted; restoration remains required.

### 2026-09-08 - Native-preserving Mamba boundary candidate starts

CONFIG -> PR48375 aligned Mamba lookup backport into the installed R276
Python module only. Base image521eb277..., derived image43e77a22...;
source hash1a0dedb7 -> a3ab3679. No PYTHONPATH or entrypoint override.
COMMAND -> build_mamba_eagle_image.py; test_mamba_eagle_drop.py in CPU-only
base and candidate containers; run_prefix_campaign.py --churn-first --profile.
RESULT -> Base regression confirms Mamba retains80 tokens where full
attention drops to64. Candidate matches64, including zero/one-block and
no-drop controls. Six native/core routing file hashes match exactly.
Stock-b final warm profile has948 completed c10d calls per rank plus948
wrappers (not1896 independent collectives), matched shapes/counts; cached
prefill1137 rows, decode1/5 rows. Both stock-b teardown health gates pass.
VERDICT -> CPU bug is confirmed and patched; GPU root cause and serving
correctness remain unproven. Candidate must pass large-history churn on a
fresh process and again after pressure, plus existing reuse/quality/lifecycle
gates. Do not promote based solely on the CPU regression.

### 2026-09-08 - Mamba drop backport does not clear oversized-history gate

CONFIG -> Derived image43e77a22 from exact R276 image521eb277, MTP4,
FP16 KV, prefix on,200K/c4, no CPU tier; unchanged native hashes.
COMMAND -> mamba-eagle-mtp4, --churn-first. Coordinator CPU checks also
exercise the installed HybridKVCacheCoordinator: eagle groups0/1, aligned
832-token blocks, partial-hash path false. Base hit3328, patched2496.
RESULT -> First four132K histories captured14 checks:13 correct, session3's
first answer timed out at900.017s with responseNone. No partial output was
captured for that request, so do not label this timeout a punctuation loop.
The original stock-b run remains the actual512-exclamation corruption proof.
SIGTERM cancelled the client after this decisive failed gate; raw remaining
sessions are incomplete, job rc=-15, manual-failure.json records the action.
Normal server teardown/post-health is underway, followed only on exit.rc0
by stock MTP0+prefix control. No serving promotion and no root-cause claim.
VERDICT -> Reject this backport as a sufficient daily fix. Retain the proven
CPU regression separately from unsuccessful GPU qualification. Next test
changes only speculative decoding from MTP4 to0 on the original image.
Add optional SSE capture to the tool probe to preserve partial timeout
output; fragmented tool arguments, final usage and incomplete-stream
retention passed CPU checks. The first MTP0 churn diagnostic uses streaming;
its final postpressure churn remains the original nonstreaming API path.
Raw roots: results/int4_prefix_20260908/mamba-eagle-mtp4 and stock-mtp0.

### 2026-09-08 - Diagnose mixed-prefill blocking; prepare 4K-batch candidate

CONFIG -> Stock R276 MTP0 with prefix caching, FP16 KV,200K/c4,32768
prefill budget. Nominal pool535222 tokens, versus446332 with MTP4.
COMMAND -> Streaming four132K tool histories, then profile_mixed_prefill.py
with one cold132K recall and three short128-token guides, under the same
owned lease. Pause coordinator only while the diagnostic queue completes.
RESULT -> Captured10/10 completed tool/answer checks before deliberately
cancelling this incomplete campaign. One otherwise correct tool-call stream
had about17-second fragment gaps. No MTP0 corruption was observed, but this
is NOT a completed correctness qualification. Planned154K oversubscription
extra was cancelled, not run. DRM fdinfo snapshot did not establish a
residency shortfall. All four mixed-profile responses pass: long recall
TTFT76.62s, subsequent max gap0.04s; short-guide TTFT0.32-0.65s and max
chunk gap23.17s. These are profiled diagnostic timings, not a speed result.
Both ranks match2324 completed CPU collective events (1162 c10d calls plus
wrappers); actual allreduce rows include32451 four times per129-layer-call
set, plus1667/30/594/61. The large prefill chunks support head-of-line
blocking as a cause of long gaps; they do not explain the original stock
punctuation corruption on their own. Coordinator resumed for normal health.
VERDICT -> Do not promote or call MTP0 failed for corruption. Next candidate
retains the scoped Mamba backoff image43e77a22 and MTP4, changes only
--max-num-batched-tokens32768 ->4096 in Config.json (Env identical), and
keeps200K/c4/FP16 KV/no CPU tier. Normalize extra churn workloads from the
measured new pool, since lower prefill budget may grow KV capacity. Require
both original132K case and capacity-normalized churn, before and after
pressure. Prepared authenticated150K cold/warm public-frontdoor smoke for
final serving validation; it is not yet run. Daily qualification unchanged.

### 2026-09-08 - MTP4 still corrupts with the backoff fix and 4K prefill

CONFIG -> mamba-eagle-mtp4-b4096, image43e77a22, MTP4, FP16 KV,
200K/c4,4096 prefill budget. Nominal GPU pool524324. Prepared normalized
four151839-token histories (13775 records, estimated83032-token excess).
COMMAND -> First streaming four132K tool histories, before any broader
campaign. Inspect per-session SSE rather than waiting for a900s timeout.
RESULT -> Four complete checks correct, then session3's initial tool request
streamed396 exclamation marks before cancellation. This is actual corruption,
not just slow queueing. It was the first request for that session's unique
cache salt; final usage was not received, so do not invent a cached-token
counter for the cancelled response. Preserve full SSE and manual-failure.json.
Client SIGTERM and runner STOP ended the failed workload; normal teardown,
per-card and compiled two-rank post-health passed, exit.rc0. Normalized
extra was not reached. Small-batch MTP4 plus the backoff patch is rejected.
VERDICT -> Prefill blocking explains long gaps but not this correctness
failure. Next is original-image MTP0 with4096 prefill and prefix enabled;
no patch-image promotion. Dynamically grow both tool-churn sizes and the
number of distinct150K reuse histories from the measured pool, and allow
appropriate total job duration without weakening the900s request gate.
The launcher now pins the qualified prefill budget as well as MTP depth;
its existing qualification is unchanged. Also clarify the preceding profile
entry:129 is a collective-call count per forward-shape set, not129 layers.

### 2026-09-08 - Retarget prefix serving to MTP2/MTP3

CONFIG -> User prioritizes MTP2/MTP3 toward >80 tok/s and permits rare
punctuation loops with bang-guard recovery. FP16 KV and GPU prefix caching
remain required; CPU offload stays disabled. Keep4096 prefill budget fixed
for the first depth comparison.
COMMAND -> End stock-mtp0-b4096 control through STOP and client SIGTERM;
run_prefix_campaign.py --mtp3 --screen on the original pinned R276 image.
Inspect installed engine/arg_utils.py and hotschmoe/bang-guard source.
RESULT -> Incomplete MTP0 control recorded12/12 correct completed checks;
normal teardown and post-health exit0. No full qualification claimed.
Installed R276 API-server batch default for32GiB cards is2048; the prior
FP8 and INT4 daily32768 setting was explicit. Chunked prefill permits
prompts larger than the scheduler token budget. Bang-guard detects32
consecutive exclamation marks, excludes the failed attempt, rotates cache
salt and permits up to3 consecutive retries by default. Its own README
explicitly leaves real vLLM recovery unverified. MTP3 screen is running.
VERDICT -> Replace zero-loop promotion preference with measured raw loop
frequency and recovery evidence per user instruction. Do not infer rarity
from a small sample or silently count recovered attempts as clean. A screen
is only a bounded comparison (coherence, decode, growing tools, shared cache,
thinking and150K warm reuse), not a shelf qualification. Preserve all earlier
failures. Source/default evidence and incomplete-control record are under
/mnt/vm_8tb/b70/results/int4_prefix_20260908/.

### 2026-09-08 - Measure rare loops and recovery separately

CONFIG -> User target is under1 percent raw bang-loop requests. User reports
old FP8 incidents clustered into about30 minutes between long clean periods;
this is an anecdote to guide testing, not proof of poisoned cache state.
COMMAND -> Inspect bang-guard d341fe6c9a3b53061e9936b0cb78b6f08e6f6438;
run its real Pi CLI offline integration with installed Pi0.84.3. Add bounded
live-model recovery probe and optional diagnostic tool-probe cancellation,
salt rotation, timestamped attempts and up to3 retries.
RESULT -> Offline actual Pi test passes: stream cancelled, corrupt context
excluded, salt rotated, recovered answer returned. CPU diagnostic checks
pass across stream chunks, reasoning, escaped tool JSON, and four concurrent
sessions with injected failures; raw failures remain separate from successful
checks. Live Pi probe is queued under the MTP3 server lease: first response is
explicitly injected; subsequent response goes to the actual88K-history model.
VERDICT -> Neither an injected recovery nor a small clean sample establishes
natural fault recovery or an under1 percent long-run rate. Measure natural
incidents, retry outcomes and clusters separately. Guard emulation in the
Python history probe is explicitly not execution of the Pi/OMP extension.
The new live Pi probe executes the actual pinned extension. No guard source
or client installation was changed.

### 2026-09-08 - MTP3 speed screen and functional guide review

CONFIG -> Original R276 image, MTP3, FP16 KV, GPU prefix on,200K/c4,
batch4096,util0.96,cgroup64GiB,CPU tier0. Pool530468 tokens.
COMMAND -> stock-mtp3-b4096-screen; author's fixed12-prompt benchmark,
actual Pi injected-failure/live-recovery probe, concurrent canaries and
three2048-token LRU guides. Review guide code in the pinned CPU grader
container with no network, devices, writable root or extra capabilities.
RESULT -> Author metric95.363772 tok/s (class-balanced first99 intervals),
zero prompt-cache hits and all canaries pass. Complete token arrays12/12
match the earlier strict MTP0 control; this cross-shape comparison alone is
not a fresh matched restart qualification. Actual Pi recovery into the live
88K-history model passes after the explicitly injected initial failure.
All3 guides coherent, but raw exact-repeat gate fails: generated custom LRU
variants differ in implementation, not only prose. All6 extracted complete
Python blocks pass18000 operations against an OrderedDict reference. Raw
failure remains preserved. Runner stopped before growing-history screens;
normal teardown and post-health exit0. MTP2 matched screen is running.
VERDICT -> Functional guide variation is reviewed, with no byte-exact claim.
MTP3 cache qualification remains incomplete. User-scoped final campaign now
keeps prefix reuse/eviction, bounded132K concurrent histories with separately
counted bang retries, coding, cancellation and health; it omits the previous
forced32K-output offload-pressure workload. Launcher accepts qualified MTP2
or MTP3 manifests, but the promotion manifest remains unchanged.

### 2026-09-08 - MTP2 meets speed target;4K history reuse is incomplete

CONFIG -> Original R276 MTP2, FP16 KV, GPU prefix on,200K/c4,batch4096,
util0.96,cgroup64GiB,CPU tier0. Nominal pool536758 tokens.
COMMAND -> stock-mtp2-b4096-screen fixed author benchmark, concurrent
canaries, four88K tool histories, isolated150K A/A/B/A reuse, xhigh canaries,
guide functional review and CPU /tokenize prefix comparison.
RESULT -> Author metric84.557267 tok/s; all12 complete token arrays match
strict MTP0 control. Initial and xhigh concurrent canaries pass. Incomplete
history control6/6 completed checks correct, but both completed followups
report zero cached tokens. Cancelled the remaining history requests and
pending4K promotion; no corruption claim. CPU tokenizer counts match actual
usage88314 ->88380, with all88314 initial tokens preserved as a prefix.
Isolated150K A/A/B/A answers4/4 correct: TTFT95.260,1.671,95.176,1.687s;
cache hits0,148928,0,148928. Thus GPU caching is functional for these repeated
prompts, but normal growing-history reuse was not established. All3 guides
coherent but non-identical; all6 Python blocks pass18000 reference operations.
Normal teardown and post-health exit0. MTP2/batch32768 direct comparison is
running, with an early8-check88K history arm before full qualification.
VERDICT -> Do not promote4K batching based on nominal capacity or isolated
reuse alone. The explicit old32K budget remains a candidate; do not infer
that smaller is always better. Add an early warm-hit/latency gate to stop
qualification if correct responses are recomputing. A new raw bang-rate
audit separates workload families, retries and natural incidents; CPU tests
verify the denominator and32-bang detection. It makes no long-run rarity or
independence claim and excludes injected Pi failures and intentional cancels.

### 2026-09-08 -32K batching restores the MTP2 tool-history reuse check

CONFIG -> Original R276 MTP2, FP16 KV, GPU prefix on,200K/c4,batch32768,
util0.96,cgroup64GiB,CPU tier0; profiler configured but inactive during tests.
Nominal KV pool456916 tokens (15.62GiB per rank), versus536758 at4K.
COMMAND -> stock-mtp2-b32768-daily/00b-strict and00a2-tool-boundary:
four concurrent88314-token histories, one tool call and answer each.
RESULT -> Author metric83.150939 tok/s with all canaries and zero cache-hit
checks passing; complete outputs12/12 token-exact against the4K MTP2 screen.
Boundary test8/8 correct,8 attempts,0 bang incidents,194.231s total. Cold tool
calls189.405-192.061s; all four followup answers reused87360/88380 tokens and
finished1.602-2.473s. The4K control's two completed followups had zero cached
tokens despite a full88314-token common prefix. Three long guides remain
coherent but not byte-identical; six Python blocks pass18000 reference ops.
Initial00a attempt failed in argparse before inference because a single-dash
salt looked like an option. Raw rc2 preserved;00a2 uses explicit --salt=value.
Queued legacy salt arguments now parse literally; CPU parser check passes.
VERDICT ->32K is the selected batching candidate on direct measured reuse
benefit, with slower cold concurrent first responses as the known tradeoff.
Do not claim the internal cause or that eight clean requests establish the
under1 percent long-run target. Full quality, eviction, recovery, teardown
and fresh public-serving qualification remain in progress.

### 2026-09-08 - MTP2 prefix reuse survives eviction and growing tools

CONFIG -> stock-mtp2-b32768-daily, original R276, FP16 KV, GPU prefix on,
MTP2,200K/c4,batch32768,CPU tier0; same candidate as preceding entry.
COMMAND ->03-reuse:150K A/A/B/C/D/A/A;04-tools:four88K histories with four
tool/answer turns each, streaming evidence and optional bang retry detection.
RESULT -> Reuse7/7 correct and exact. Cold TTFT93.648s, warm1.668s with
148928/150045 cached tokens. Four distinct histories exceed the456916-token
pool; revisiting evicted A cached0 and took93.881s, then next A hit148928 and
TTFT1.675s. Growing tool histories32/32 checks,32 attempts,0 bang incidents,
218.761s total; all28 followups reused98.5315-98.8459 percent of prompt tokens.
VERDICT -> GPU prefix reuse and correct recomputation after eviction are
measured on this candidate. Larger-context churn, shared-prefix clients,
cancellation, coding quality, final health and fresh public deployment remain
in progress. No long-run under1 percent incident-rate claim is made.

### 2026-09-08 - One unsalted shared-prefix bang; continue with measured recovery

CONFIG -> Same MTP2/batch32768 candidate;04b-shared-tools omits cache_salt,
four22K histories and32 tool/answer checks. This arm did not enable retries.
COMMAND -> Inspect raw session0 turn3 answer and audit_bang_rate.py. Continue
on a fresh matched lifecycle with a fixed128-check shared-prefix recovery
soak, then the still-outstanding cancellation/churn/coding/profile checks.
RESULT ->31/32 shared checks correct; final session0 answer is512 bangs,
finish=length,21632/22662 prompt tokens cached. No timeout or hardware fault.
Normal teardown and post-health exit0. Independent raw audit counts1/106
across completed test mix (0.9434 percent), but1/32 shared (3.125 percent).
Old summary bang_attempts=0 reflected disabled live detection, not absence
of raw corruption; new summaries also scan completed unguarded responses.
CPU tests pass for32-bang detection and four concurrent shared requests that
switch to private salts on an injected failure, preserve clean histories and
count raw retries. No natural recovery was observed in the stopped process.
VERDICT -> Per user's rare-loop tolerance, one incident is not automatic
rejection. Do not use the aggregate to establish a long-run under1 percent
rate or hide the shared-workload result. Fresh continuation fixes the soak
size at128 checks in advance; natural failures and successful recovery remain
separate. It resumes after the prior measured prefix/eviction/growing-tool
checks, retains all old failures, and has its own continuation marker rather
than falsely marking the prior campaign fully passed. Actual Pi recovery is
also queued for MTP2; its initial failure is explicitly injected.

### 2026-09-08 - Compare MTP decode and accept user under5 percent tolerance

CONFIG -> User accepts an observed bang fraction below5 percent and asks
for reproducible diagnosis. This supersedes the earlier under1 percent target
for current acceptance, without rewriting earlier failed gates. FP16 KV and
GPU prefix caching remain required; CPU offload remains disabled.
COMMAND -> Review fixed12-prompt strict results and matched2048-token guide
workload, audit fixed128-check MTP2 shared recovery soak, and continue MTP3
qualification under --bang-fraction .05. Reattach only the coordinator to the
same leased server PID272967; no GPU process restart or runtime change.
RESULT -> INT4 strict MTP2=83.15094tok/s at32K prefill and84.55727 at4K;
MTP3=95.36377 at4K. Prefix-off MTP4 two-fresh mean100.21428 has a different
profile and is not a fully controlled MTP comparison. Clean-package FP8 MTP3
guide result a0clean/03-decode=6144/75.59694=81.273tok/s, exact repeats.
INT4 guide MTP2/4K=98.24 and MTP3/4K=123.75tok/s, coherent but variable
outputs; do not compare these full-output rates to strict first100 timing.
MTP2/32K has456916 shared KV tokens; current MTP3/32K reports451562,
versus old FP8 daily292968. All use FP16 KV. Completed INT4 prefix-off
HumanEval base/plus158/150 versus clean FP8 157/152; MTP2/3 quality pending.
MTP2 fixed soak:128 correct checks after134 raw attempts,6 bangs (4.4776
percent), all recovered. Every incident is logical session0; two occur after
private-salt rotation. MTP3 already reproduces session0 bangs. Add full raw
request capture and logical session-order controls to distinguish content,
submission order and concurrency. Thread submission order is not proof of
actual backend batch-row order. Queue reverse/shift/singleton/isolated arms.
VERDICT -> MTP2 observed rate is within the newly accepted tolerance. This
small correlated sample is not proof of a long-run below5 percent rate.
A shared-cache-only explanation is insufficient; root cause is unproven.
Finish matched MTP3 speed, coding and lifecycle qualification before public
promotion. Retain all prior failures and report workload-specific fractions.

### 2026-09-08 - MTP3 preferred; historical mixed-batch NaN lead recovered

CONFIG -> User leans toward INT4/MTP3/GPU prefix;32K prefill and FP16 KV.
Current stock-mtp3-b32768-daily retains base R276 identity and451562 KV tokens.
COMMAND -> Fixed128 shared checks; strict suite; reversed/shifted logical
submission IDs, singleton, private salts, and logprobs; inspect archived
June26-27 journal read-only and git commit ca762fa. No old runtime restored.
RESULT -> MTP3 fixed soak128/128 correct after132 attempts,4 bangs=3.0303
percent, recovered. Strict94.82656tok/s and12/12 complete output token arrays
exact against MTP3/4K. Three2048-token guides byte-exact,6144/49.98355=
122.9204tok/s full wall. Current150K cold prefill93.6631s (~1602tok/s) versus
valid installed FP8 a0b/04-reuse93.2081s (~1610tok/s), both32K budgets.
This isolated difference is about0.5 percent; not a meaningful prefill win.
Reverse3,2,1,0:32/33 attempts, one bang on session3 answer0. Shift1,2,3,4:
32/33, one bang on session1 answer0. Singleton0:8/8, no bang. Four private
salts:32/33, one bang on session0 answer2. Each bad answer's first bang
preceded the other three answers' first deltas while their requests were
outstanding (bang-timeline.json). This supports order/overlap sensitivity,
not a unique session0 prompt or a requirement for cross-session cache hits.
Logprobs arm32/32, no bangs,2112 finite logprob values;113.386s including
initial latency. Its different timing/path means no current NaN confirmation.
Historical June26-27 Qwen3.6/vLLM0.23 evidence (ca762fa) reported clean isolated
replay and homogeneous prefill, reproducible mixed-load degeneration, NaN
JSON errors with logprobs, and eventual persistent failure cleared by restart.
That older finding is a lead, not proof that R276 has the same defect.
VERDICT -> Continue full MTP3 qualification. Direct cold-prefill-over-active-
decode probes queued with and without logprobs; client overlap is measurable
but does not prove backend co-batching. Broader coding and lifecycle gates
remain required before promotion. Cache-salt recovery is mitigation, not a
root-cause fix; no new FP8 KV work or native/driver changes.

### 2026-09-08 - Bound the current overlap trigger without reviving old fixes

CONFIG -> Same R276 MTP3/32K prefix candidate. Installed _xpu_ops.py already
has VLLM_XPU_GDN_SPLIT_MIXED=1; server log confirms split-mixed execution.
This is not an absent-old-flag diagnosis. No runtime changes in these arms.
COMMAND -> probe_mixed_decode.py with and without --logprobs: two waves per
arm, each one2048-token anchor plus three cold32K recalls, prefills submitted
after an actual anchor text delta. Also32 shared tool checks with
--serial-requests and four logical histories; preserve SSE and raw requests.
RESULT -> Both direct mixed arms8/8 correct,0 bangs,6/6 client-observed
overlaps; logprob arm25116 finite values,0 nonfinite. The serialized four-
history control32/32 correct,0 bangs; timestamp audit confirms maximum one
observed active stream. Earlier concurrent counterpart was32/33 with one
bang. These are bounded controls, not representative long-run rate tests.
Current03-reuse completes7/7 correct; warm fraction0.9925556 and TTFT ratio
0.0173275. Existing prefix cache works while the growing-tool trigger remains.
VERDICT -> Generic cold-prefill/long-decode overlap alone is insufficient to
reproduce current bangs. Concurrency plus cached growing tool conversations
is the stronger reproducer. Next instrumentation should correlate actual
scheduler request/row order, cache hit length, accepted speculative tokens,
and recurrent state indices at the first bad step, on both ranks. Finite-
value observation must avoid changing the timing enough to hide the bug.
Do not claim NaNs, a particular kernel, or global persistent poisoning on
R276 from the older stack's evidence. Continue coding/churn/health and fresh
public MTP3 deployment under the user's under5 percent observed tolerance.

### 2026-09-08 - Promote user-accepted INT4 MTP3 prefix candidate

CONFIG -> R276 base image521eb277/source54aefaf0, AutoRound W4A16 g128,
MTP3, FP16 KV, GPU prefix on, CPU tier0,200K/c4,batch32768. User explicitly
accepts current INT4-vs-FP8 coding results and directs moving forward.
COMMAND -> Complete stock-mtp3-b32768-daily; analyze_prefix_campaign.py,
compare_code.py against both baselines, audit_bang_rate.py, paired profile,
normal teardown and per-card/two-rank post-health. Prepare qualification,
registry and systemd unit; start fresh leased public lifecycle20260908T220558Z.
RESULT -> Candidate WORKLOADS_PASSED and exit0; both GPU and collective
post-health pass. Pool451562 tokens. Four88K histories32/32 correct in217.876s,
all28 followups98.53-98.85 percent cached. Cancellation112.712s with exact
repeats and warm TTFT1.62-1.65s. xhigh24/24 and199K recall pass. Actual Pi
injected-failure recovery passes on MTP3/32K; no user client config changed.
Oversubscribed four132K histories:8/8 correct after9 attempts,1 bang recovered,
645.822s; all four final answers cached0, no active preemptions, capacity queue
observed. Its1/9 raw fraction exceeds5 percent in this small stress sample;
do not dilute it with easy requests. Fixed shared soak remains4/132=3.0303
percent. Known concurrency-sensitive tool-history bug is NOT fixed.
Coding164 raw outputs contain no bangs. Base/plus157/149 versus FP8 157/152,
previous INT4 158/150. Only additional failed task versus previous INT4 is132,
truncated at2048 tokens in comments. The raw outputs for33,91,97,125,151,154
are byte-identical to previous INT4. Original <=2-net-loss score gate stays
false; user's explicit acceptance is recorded separately, not a rescoring.
Extra task132/4096 diagnostic generated output but its one-task grading hit
missing-problems assertion; excluded and inconclusive, no additional GPU work.
Warm profile paired1666 completed CPU collective events each (833 c10d plus
833 wrappers), equal shapes/counts:525 all-reduces,308 all-gathers per rank,
1137-token cached prefill and1/4 decode rows; host event waits477/433. Captured
internals may be opaque; no one-fence/per-token claim.
VERDICT -> Promote the user-preferred guarded conversational recipe with
explicit quality/rate/oversubscription limitations. Public alias hotschmoe-dd
unchanged. Fresh startup/authentication, concurrent canaries,150K warm reuse
and strict restart parity queued under the serving lease. Startup pending;
boot unit prepared and statically verified, host systemd installation still
requires the documented user sudo command. Do not claim all clients have
bang-guard or that the server itself retries natural loops.

### 2026-09-08 - hotschmoe-dd restored on INT4 MTP3 with GPU prefix caching

CONFIG -> Fresh public lifecycle20260908T220558Z, same qualified R276 image,
AutoRound INT4/MTP3/FP16 KV/GPU prefix on/CPU tier0/200K/c4/batch32768.
Stable public alias hotschmoe-dd, authenticated port18080, backend18124.
COMMAND -> Leased launcher start; ready helper; smoke_daily_frontdoor.py;
smoke_prefix_frontdoor.py through the authenticated public API; author's
fixed12-prompt strict suite and compare-strict-attempt-outputs.py against the
qualified MTP3 candidate. No driver/image/native changes. Leave serve running.
RESULT -> All four leased validation jobs exit0. Missing/wrong keys return401;
authenticated identity and C4 arithmetic pass. Fresh pool451562 tokens.
150K public recall correct twice: cold TTFT94.10277s, warm1.622735s,
99.2542 percent cached. Fresh strict94.43162tok/s versus candidate94.82656;
all12 complete output token arrays exact. Candidate had clean teardown and
per-card/compiled two-rank post-health; fresh startup pre-health passes.
Public evidence: runtime root/public-validation.json. No STOP or exit marker
in the active lifecycle. Qualification and registry now reference live proof.
VERDICT -> Requested conversational prefix-serving outcome is live. Bang
bug remains unresolved: client recovery is validated; shared-soak4/132 and
oversubscribed1/9 raw incidents remain explicit, with no long-run below5
percent guarantee. User accepts coding157/149 versus FP8 157/152; primary
score and failed conservative score gate remain unchanged. CPU RAM offload
and FP8 KV remain disabled. This is not a no-loss or global-stability claim.

Boot configuration is prepared and statically verified; host systemd unit
installation was not performed. User command (does not interrupt this serve):

```bash
sudo bash /mnt/vm_8tb/github/b70_ai_things/vllm/fp8/install-hotschmoe-dd-systemd.sh
```

The installer backs up units, installs/enables the new launcher and disables
the retired boot recipe. Current manual serving continues; subsequent
service ownership changes require draining/stopping it and waiting for the
owned teardown, as the installer explains. User worktree changes preserved.

### 2026-09-08 - Manual serving lost on user-session shutdown; system start required

CONFIG -> Previously validated INT4/MTP3/FP16 KV/prefix recipe unchanged.
COMMAND -> Check public health, backend models/metrics, Docker, installed
system unit, user journal and previous lifecycle. Run bin/xe-reset through
its GPU lease; try systemctl --no-ask-password --no-block start hotschmoe-dd.
RESULT -> Ports18080/18124 refuse connections; no serving container. Grafana
and Prometheus containers remain up; Prometheus b70_daily reports connection
refused at18080/metrics. Manual launcher exit.rc=-1 denotes runner SIGHUP;
its server/exit.rc is absent. At22:23:55 UTC the user manager stopped the
tmux scope; the API subsequently performed SIGTERM shutdown. This supports
session-lifetime termination, not a demonstrated bang/GPU crash. The correct
MTP3 system unit is installed/enabled but inactive, with no service start log.
Recovery rebind completed, both card probes and compiled two-rank collective
health passed, exit0; no reboot. System service start was denied because
interactive authentication is required. No privilege bypass attempted.
Add SIGHUP to the runner's existing SIGTERM/SIGINT graceful-stop handler;
Python compilation passes. This repairs cleanup but is not a substitute for
system-service ownership. Historical performance/quality proof is retained;
deployment state now explicitly records the outage instead of claiming live.
VERDICT -> Endpoint is currently down. The earlier detached manual process
was still tied to the user-session scope; enabling the boot unit did not
transfer ownership of that process. User must start the installed service:

```bash
sudo systemctl start hotschmoe-dd.service
```

The service owns both GPU leases and repeats matched startup health before
serving. It is already enabled for boot. Verify API and Prometheus scrape
recovery after it becomes ready. Recovery evidence:
/mnt/vm_8tb/b70/results/int4_prefix_20260908/20260908-hangup-recovery.log.

### 2026-09-09 17:59 UTC - Scheduled S0c CPU diagnosis

CONFIG -> Same-image SGLang FP16 INT4 control crashed on both TP2 ranks;
no test or coordinator remains running. COMMAND -> Inspect lifecycle/logs;
export hashed installed source and test dtype selection without GPU devices.
RESULT -> Default BF16 convolution state conflicts with FP16 input in the
Triton branch merge. Explicit float16 conv selects FP16 while temporal state
stays FP32 in the CPU helper test. Teardown and both post-health probes
completed, but STOP-after-job races crash detection and skipped xe-reset.
VERDICT -> No GPU retry this review. Repair/test lifecycle detection, complete
leased crash recovery/health, then qualify an explicit FP16-conv S0d arm.
Full evidence: vllm/cache800k/20260909_campaign.md and raw
/mnt/vm_8tb/b70/results/cache800k_20260909/s0c-cpu-diagnosis-1758/.

## 2026-09-09 18:11 UTC review: S0c lifecycle repair

CONFIG -> No campaign coordinator, server process or serving container alive.
S0c retains quality rc=1, both-rank dtype crash and false lifecycle exit0.
Actual prior post-health logs pass both cards and compiled P2P-off collective;
no crash recovery log exists. Production remains intentionally offline.
COMMAND -> Repair sglang/cache800k/server.py and run CPU-only mocked lifecycle
regressions plus existing cache800k coordinator tests. No GPU commands run.
RESULT -> Failed or timed-out jobs now raise after preserving their done rc,
triggering failure accounting, owned teardown, reset and post-health. Check
server exit before honoring STOP and again before intentional Docker stop.
Five lifecycle tests pass (crash/STOP race, failed probe before server exit,
timeout, startup exit/STOP, clean stop); three coordinator tests pass.
Source hashes and test logs: results/cache800k_20260909/
s0c-lifecycle-repair-1811/ under /mnt/vm_8tb/b70. git diff --check passes.
VERDICT -> Bounded tested lifecycle repair complete, not GPU qualification.
Next tick must perform leased xe-reset and per-card/compiled collective health
for the prior crash, explicitly set and CPU-test FP16 convolution state with
FP32 temporal state, then run a fresh S0d via one named systemd user coordinator.
No new image/native bytes, dtype setting, serve or GPU recovery in this review.
Four-active-200K FP8, prefix, MTP, graphs and real RAM recovery remain pending.
Timer remains enabled; no campaign test is currently running.

## 2026-09-09 18:23 UTC review: S0d FP16 convolution-state qualification started

CONFIG -> No live campaign coordinator/server at entry. S0c quality rc=1,
prior scheduler dtype failure and false lifecycle exit0 preserved. Exact
SGLang adc915d266 image and same INT4 tensor path. FP16 model/KV, MTP0,
eager, radix-off; change convolution storage BF16 to FP16 explicitly with
SGLANG_MAMBA_CONV_DTYPE=float16; retain FP32 temporal state.
COMMAND -> Add launcher environment and manifest assertions. Run eight CPU
lifecycle/coordinator tests and the installed dtype helper in a CPU-only
container without GPU mounts. Run bin/gpu-run python3 recover.py, serial
rebind reset, both-card health and compiled TP2 P2P-off collective. Start
run_arm.py with fresh s0d-sglang-fp16-conv16.plan.json via named systemd user
unit b70-cache800k-s0d.service, coordinator PID 674545, lease owner 674562.
RESULT -> Eight CPU tests pass; installed helper confirms FP16 conv/FP32
state. Initial unittest invocation had two import-path errors; original log
and corrected PYTHONPATH passing run both preserved. Recovery and both health
checks exit0; compiled collective world_size=2, shape=4x5120, 10 iterations.
Single coordinator confirmed active; server holds both leases and begins its
own pre-health. No inference result yet. Raw hashes, source snapshots, CPU
logs and recovery: s0d-preparation-1822/ under the campaign raw root.
VERDICT -> Configuration repair is CPU-tested, not numerically qualified.
Fresh bounded arm runs c4 short quality, two c1 guides, then 32K/c2 retrieval;
failed jobs must stop and recover through the repaired lifecycle. Next tick
inspect actual processes, jobs, scheduler logs, teardown and post-health.
Coordinator log: s0d-sglang-fp16-conv16.plan.log; server and job logs are under
s0d-sglang-fp16-conv16/. No new image/native bytes or production restart.
Four simultaneously progressing 200K FP8 contexts, prefix/MTP/graphs and real
RAM reload remain pending. Review timer remains enabled.

## 2026-09-09 18:41 UTC review: S0d long-prefill device loss

CONFIG -> Same S0d FP16 convolution/FP16 KV/FP32 temporal-state arm,
INT4/MTP0/eager/radix-off, TP2 P2P disabled. Single systemd coordinator
b70-cache800k-s0d.service PID 674545, leased server PID 674570.
COMMAND -> Read actual processes, job summaries, scheduler traces and kernel
journal; preserve timestamped raw snapshots and SHA256 hashes in
s0d-failure-review-1840/ under the campaign raw root. No new GPU workload.
RESULT -> Short quality passed 24/24 with exact repetition. Both 2048-token
guides passed and repeated exactly (4096 completion tokens). The 32K/c2 job
started at 18:34:56 but has no passing retrieval result. Kernel records
0b:00.0 queue timeout/reset at 18:35:01-02, blocked TTM fence waits, then
44:00.0 timeout/reset at 18:39:34. Both schedulers report DEVICE_LOST at
18:39:35 in embedding all-reduce during extend. This is the observed error
site, not established initiating cause. Job rc=1 and failure.txt preserved;
final metrics connection reset masks request detail in the probe traceback.
Owned lifecycle performed rebind recovery successfully under its existing
lease. At 18:41 post-health is running; final lifecycle rc remains pending.
VERDICT -> FP16 conv change passes short/guides but SGLang baseline is not
qualified: long-prefill execution failed. Bounded read-only failure diagnosis
complete; no unchanged retry, source/image change or production restart.
Next tick verify final per-card/compiled collective post-health, lifecycle rc
and teardown first. Then inspect the long-prefill collective shape/source
before designing a changed bounded control. Four-active-200K FP8, prefix,
MTP, graphs and RAM reload remain pending. Timer stays enabled.

## 2026-09-09 18:52 UTC review: S0d teardown verified and collective source traced

CONFIG -> Exact SGLang adc915d266, S0d INT4/FP16 model and KV/FP16 conv/
FP32 temporal state, TP2 P2P-off, eager, radix-off, chunked prefill 8192.
No campaign coordinator, serving container or health process remains alive.
COMMAND -> Inspect actual processes, systemd state, arm and recovery logs.
Export seven installed Python source files CPU-only with no device mounts,
network disabled, 2 CPUs/2 GiB; preserve relative paths and SHA256 hashes.
Trace embedding, scheduler chunk budget and distributed all-reduce source.
RESULT -> S0d unit failed with MainPID=0 and ExecMainStatus=1; lifecycle rc=1.
Short quality and guides rc=0; long32k rc=1. Owned rebind recovered both cards;
both per-card probes and compiled P2P-off collective completed successfully
at 18:41. The prior continuation JSON was stale, not an active test.
Embedding gathers and masks local-shard output before TP reduction. The XPU
branch enters inplace_all_reduce, then _all_reduce_in_place falls through to
torch.distributed.all_reduce(device_group), matching both crash stacks.
There is no clone or explicit producer synchronization in that inspected path.
Scheduler source truncates oversized input to the remaining chunk budget.
With hidden_size=5120, a configured 8192-row FP16 embedding would be
83886080 bytes (80 MiB) per rank. This is an inferred candidate shape, NOT
measured per-rank entry evidence; no successful long-prefill batch log or
collective entry/return trace establishes actual rows, strides or call count.
Passing collective health uses 4x5120 BF16 (40960 bytes), with different
runtime environment/preload settings; it is recovery evidence, not matched
large-prefill qualification. Device loss may surface after earlier queued
work; neither the traceback nor this source inspection isolates its cause.
VERDICT -> Bounded CPU source diagnosis complete; no GPU touch or runtime
change this review. Raw snapshot/source hashes: s0d-collective-diagnosis-1852/
under the campaign root. Next prepare a CPU-tested source-only trace of
per-rank embedding producer and collective entry/return (shape, dtype,
stride, bytes, sequence number), hash the overlay, and use one bounded fresh
leased arm via a named systemd coordinator. Record asynchronous return versus
completion distinctly. A smaller-prefill-budget control is a possible next
single-factor intervention; do not copy the historical Qwen3.6 clone fence
or repeat S0d unchanged. Failure stops and recovers before any subsequent arm.
SGLang long-context baseline remains unqualified. Four-active-200K vLLM FP8,
prefix/MTP/graphs and RAM reload remain pending. Timer enabled; production
offline; no campaign test running and no input needed.

Source preservation note: files listed in source-encoding.json use ASCII
JSON strings (.py.json) to preserve upstream non-ASCII comments losslessly;
decode JSON before checking their original source SHA256.

## 2026-09-09 19:04 UTC review: embedding trace prepared on CPU

CONFIG -> No campaign coordinator, serving container or GPU process alive.
S0d lifecycle rc=1, short/guides pass, long32k rc=1 preserved. Rebind and
both-card/compiled P2P-off post-health logs pass; unit MainPID=0/status1.
COMMAND -> Inspect processes, containers, systemd and actual lifecycle logs.
Prepare exact-source-hash-guarded embedding trace overlay; run CPU lifecycle,
trace and AST-extracted patched-forward tests against the exported image source.
RESULT -> Nine tests pass, including eight actual forward branch/rank cases.
Trace emits producer host return and TP-wrapper entry/return with TP rank,
PID, monotonic timestamp, sequence, shape, stride, dtype, device and bytes.
No tensor reads or device synchronization. Return is explicitly NOT device
completion; scope is embedding wrappers, not every model collective.
Source review caught nonexistent self.tp_rank in the first undeployed overlay;
corrected to installed get_tp_group().rank_in_group. Initial mock-only results
and overlay retained but superseded by overlay-v2 and cpu-tests-v2.log.
Both versions were CPU-only; no GPU deployment or image/native change.
VERDICT -> Bounded tested instrumentation preparation complete. Authoritative
raw overlay and hashes: s0e-trace-preparation-1904/overlay-v2/ under campaign
root. Next tick integrate this read-only overlay and helper into the campaign
container with recorded mount/import identity, CPU-test launcher command, and
prepare a fresh bounded arm through run_arm.py and one named systemd user
coordinator. A 2048-row prefill budget versus S0d's 8192 is a possible single
configuration control; record tracing as an additional timing perturbation.
Do not launch an unchanged uninstrumented S0d or infer a root cause from host
return records. SGLang long context, four-active-200K FP8, prefix/MTP/graphs
and RAM reload remain unqualified. No test running; timer enabled, production
offline. No user input needed.

## 2026-09-09 19:13 UTC review: S0e traced prefill2048 control launched

CONFIG -> S0d stopped with lifecycle rc=1 after long32k failure; no active
campaign server/coordinator at entry. Its owned rebind and both per-card and
compiled P2P-off collective post-health passed. Exact adc915d266 SGLang,
same INT4/FP16 KV/FP16 conv/FP32 temporal state, MTP0/eager/radix-off/TP2.
COMMAND -> Integrate reviewed overlay-v2 with exact-image and two-file hash
guards, read-only mounts and recorded target paths. Run ten CPU tests,
including actual patched-forward branches, mounts, tamper rejection and
lifecycle failures. Verify base source hash and mounted helper import/path
and source hashes in CPU-only containers without GPU devices. Start fresh
s0e-sglang-fp16-conv16-trace-p2048.plan.json through run_arm.py in named user
unit b70-cache800k-s0e.service, coordinator PID 697856, lease wrapper 697873,
server PID 697881. All GPU health/serving remains under the server's lease.
RESULT -> Ten CPU tests pass; exact base and overlay source identities pass.
One coordinator active and both card leases acquired; pre-health underway.
No new inference result yet. Prefill budget changes 8192 to 2048; tracing is
an additional timing perturbation, not a synchronization fix. Plan preserves
c4 short quality, two exact guides and 32K/c2 retrieval, with failure-stop,
owned teardown/recovery and post-health. No image/native bytes changed.
VERDICT -> Changed bounded diagnostic arm launched, not SGLang long-context
qualification. Next tick inspect actual jobs, both-rank trace shapes and
entry/return sequence, scheduler errors and lifecycle health. Host return
records do not establish device completion or total model collective count.
Raw CPU tests/source hashes/container commands: s0e-integration-1914/ under
the campaign root. Coordinator log: s0e-sglang-fp16-conv16-trace-p2048.plan.log;
server/job logs: s0e-sglang-fp16-conv16-trace-p2048/. Four-active-200K FP8,
prefix/MTP/graphs and RAM reload remain pending. Timer enabled; production
offline; no user input needed.

## 2026-09-09 19:31 UTC review: S0e measured embedding collective stall

CONFIG -> S0e same-weight SGLang FP16 KV/conv, FP32 temporal state,
TP2 P2P-off, eager, radix-off, traced prefill2048. Single coordinator
b70-cache800k-s0e.service PID 697856; leased server PID 697881.
COMMAND -> Inspect live processes, completed jobs, watchdog stacks, rank
traces and kernel journal. Preserve raw snapshot and SHA256 manifest in
/mnt/vm_8tb/b70/results/cache800k_20260909/s0e-watchdog-review-1931/.
RESULT -> Short quality passed 24/24 exactly; both 2048-token guides passed
and repeated exactly. Long32k/c2 began 19:24:43 and has no completed result.
Both ranks enter embedding collective sequence 4274 with shape 2048x5120,
stride 5120x1, FP16, 20971520 bytes each; neither records host return.
Previous sequence 4273 returned on both ranks. Both watchdogs time out at
19:29:49 after 300 seconds, with stacks inside c10d all_reduce and Level Zero
queue synchronization. No kernel timeout/reset/fault matches since arm start
at inspection. Watchdog announces a 60-second coredump wait before exit;
coordinator, leased server and probe remain alive, with no final lifecycle
or recovery marker yet. Per-card and compiled collective pre-health passed.
VERDICT -> Bounded read-only diagnosis complete. Smaller prefill plus tracing
did not avoid the long-prefill stall. Measured host entry localizes the wait,
but does not prove producer completion or establish initiating root cause.
Allow the existing bounded lifecycle to account for failure, tear down,
recover and run post-health; launch no competing workload or unchanged retry.
Next tick verify actual exit/recovery/post-health first, then design a focused
CPU-tested producer/collective completion control using this measured shape.
SGLang long context and four-active-200K FP8 remain unqualified; prefix, MTP,
graphs and RAM reload pending. Timer enabled; production offline.

## 2026-09-09 19:42 UTC review: S0e teardown verified; completion control prepared

CONFIG -> S0e traced prefill2048 SGLang INT4/FP16 KV/FP16 conv/FP32 state,
TP2 P2P-off, eager, radix-off. No campaign process or serving container alive;
unit failed, MainPID=0, ExecMainStatus=1. Prior running JSON was stale.
COMMAND -> Inspect actual processes, results, failure/recovery and post-health
logs. Prepare a separate source-only completion trace helper and exact-source
hashed overlay; run CPU trace, lifecycle and coordinator regressions.
RESULT -> S0e lifecycle rc=1; quality/guides rc=0, long32k rc=1. Owned rebind
recovered both cards; both per-card and compiled 4x5120 P2P-off collective
post-health pass. Sixteen CPU tests pass, including pre-fence failure stopping
collective submission, collective failure stopping post-fence, and post-fence
failure never claiming completion. Initial run skipped one mount test; full
run supplies the existing overlay and has no skips. New overlay adds device-
wide XPU synchronize before/after embedding reduction, with distinct entry
and successful-return records. Collective host return remains explicitly
not device completion. No GPU touch, image/native change or deployment.
VERDICT -> Bounded CPU diagnostic preparation complete, not a serving repair
or numerical qualification. Raw source snapshots, hashes and tests under
s0f-completion-preparation-1943/ in the campaign root (directory label only;
actual preparation was 19:41-19:42 UTC). New helper hash eb732b739ff7e594d6707294aa902291fe841d844a475d0fcd6209f0f648b494.
Next tick integrate this exact helper hash and completion metadata into the
launcher, CPU-test mounts/import identity, then one fresh bounded S0f p2048
arm via run_arm.py and a named systemd coordinator. Current launcher correctly
rejects the new helper until explicitly integrated. Fences include all prior
device work and perturb timing; neither a fence stall nor a pass alone proves
an embedding root cause. Keep all failed-arm accounting. SGLang long context,
four-active-200K FP8, prefix/MTP/graphs and RAM reload remain unqualified.
No test running; review timer enabled, production offline, no input needed.

## 2026-09-09 19:52 UTC review: S0f completion diagnostic launched

CONFIG -> S0e stopped rc1; no campaign workload alive at entry. Owned rebind,
both per-card probes and compiled P2P-off post-health passed. Same SGLang
adc915d266 INT4/FP16 KV/FP16 conv/FP32 temporal state, TP2 P2P-off, eager,
radix-off, prefill2048. Add device-wide pre/post embedding reduction fences.
COMMAND -> Integrate exact reviewed completion-helper hash into launcher,
record completion scope and timing perturbation in manifest. Run 16 CPU
trace/lifecycle/coordinator tests, including mount and tamper checks. Import
helper and verify both mounted hashes in CPU-only network-disabled container
with no GPU mounts, 2 CPUs/2 GiB. Start fresh S0f through run_arm.py in named
systemd user service b70-cache800k-s0f.service, coordinator PID 716359,
lease wrapper 716382, server PID 716390.
RESULT -> All 16 CPU tests pass without skips; mounted hashes/import path
match. Single coordinator active, both GPU leases acquired, pre-health
underway. No inference result yet. Raw CPU evidence/source hashes under
s0f-integration-1952/ (directory label; work started 19:51 UTC). Coordinator
log: s0f-sglang-fp16-conv16-completion-p2048.plan.log; plan uses the matching
.plan.json and server/jobs use s0f-sglang-fp16-conv16-completion-p2048/.
VERDICT -> Bounded changed diagnostic launched, not a serving repair. Plan
runs c4 short quality, two exact guides and 32K/c2 retrieval with failure-stop,
owned recovery/teardown/post-health. Next tick inspect actual progress and
per-rank pre-fence/collective/post-fence records. Successful fence return
covers all preceding device work; collective host return alone is not device
completion. Neither pass nor stall alone identifies embedding root cause.
SGLang long context, four-active-200K FP8, prefix/MTP/graphs and RAM reload
remain unqualified. No native/image change or production restart. Timer enabled.

## 2026-09-09 20:11 UTC review: S0f pre-fences pass; collective still stalls

CONFIG -> S0f exact SGLang INT4/FP16 KV/FP16 conv/FP32 temporal state,
TP2 P2P-off, eager, radix-off, prefill2048, device-wide embedding pre/post
fences. At entry its coordinator remained alive performing owned post-health.
COMMAND -> Inspect live processes, job results, both-rank completion traces,
watchdog stacks and recovery logs. Preserve raw snapshot and hashes under
s0f-completion-diagnosis-2011/ in the campaign root. CPU-parse final sequences
and assert the matching rank/stage/shape records. No new GPU workload.
RESULT -> Short quality 24/24 and both 2048-token guides pass exactly.
Long32k/c2 fails rc1. At sequence 4274 both ranks record successful pre-fence
return (0.530201 ms rank0, 0.662844 ms rank1), then collective host entry
for contiguous 2048x5120 FP16, 20971520 bytes each, without host return.
Both ranks completed post-fences at sequence 4273. Watchdogs time out at
20:08:26 inside c10d all_reduce/Level Zero queue synchronization. Recovery
rebound both cards; both card probes and compiled P2P-off collective pass.
Lifecycle exit.rc=1; unit failed with MainPID=0/ExecMainStatus=1 by snapshot.
VERDICT -> Device-wide producer completion before this collective does not
avoid the observed stall. This narrows the diagnostic but does not establish
the initiating root cause or qualify large collectives from small health
probes. Bounded CPU trace diagnosis complete; no unchanged retry or runtime
change. Next prepare a CPU-reviewed matched 2048x5120 FP16 collective control
with exact serving process environment and allocation/producer provenance;
compare fresh-process versus loaded-context behavior before another full arm.
Preserve P2P-off and bounded leased recovery. SGLang long context and four
active 200K FP8 contexts remain unqualified; prefix/MTP/graphs and actual RAM
reload pending. No campaign test running; timer enabled; production offline.

## 2026-09-09 20:22 UTC review: collective environment and source audit

CONFIG -> S0f stopped rc1, unit MainPID=0; no campaign coordinator/server or
GPU container alive. Both card rebinds and per-card/compiled P2P-off post-health
passed. Exact adc915d266 image, unchanged source/native stack.
COMMAND -> CPU-only Docker imports with serving versus health environment,
network disabled, no devices, 2 CPUs/2 GiB, 60-second subprocess bounds.
Export three installed SGLang sources and hash libraries mapped after Torch
import. Compare health source against measured S0f 2048x5120 FP16 reduction.
RESULT -> Both CPU containers exit0; source hashes and mapped-library SHA256
multisets match (libccl.so.1, libccl.so.2, libsycl and UR loader). Eight relevant
environment differences remain: SYCL kernels, kernel path, logging, topology
vertex override, pidfd IPC, LD_LIBRARY_PATH, LD_PRELOAD and Level Zero V2.
Serving explicitly sets topology-check=0, pidfd and V2=0; health does not.
Health uses WORLD with 4x5120 BF16 and compiled functional reductions;
SGLang fallback reduces in place on device_group created by new_group.
Embedding producer can select fused Triton or gather plus masked_fill;
existing shape/fence traces do not identify which branch actually executed.
VERDICT -> Bounded CPU diagnosis complete. Matching CPU-loaded hashes does
not establish GPU-loaded identity or rule out environment/queue differences.
Raw commands, source exports, hashes, assertions and comparison are under
collective-env-diagnosis-2023/ (label only; actual audit 20:20-20:22 UTC).
Next implement/CPU-test a fresh-process 2048x5120 FP16 subgroup control using
the recorded serving environment, with bounded leased lifecycle/recovery.
Record actual embedding branch/allocation and GPU-loaded library identity
before treating a later loaded-context comparison as matched. Keep health
unchanged; vary environment factors individually only after that control.
No GPU work or unchanged full-model retry this tick. No test running; timer
enabled, production offline. SGLang long context, four active 200K FP8,
prefix/MTP/graphs and real RAM reload remain unqualified; no input needed.

## 2026-09-09 20:33 UTC review: fresh-process subgroup control prepared

CONFIG -> No campaign coordinator, serving container or GPU workload alive.
S0f rc1 and long32k failure preserved; owned rebind, both per-card probes
and compiled P2P-off post-health passed. Exact SGLang adc915d266 image.
COMMAND -> Prepare collective_control.py and run its four CPU tests inside
that image, serving environment, network disabled, no devices, 2 CPUs/2 GiB,
90-second outer bound. Preserve command, source hashes and lifecycle snapshot
under collective-control-preparation-2034/ in the campaign raw root.
RESULT -> Four tests pass, including full 2048x5120 FP16 CPU SUM arithmetic
and deliberate output corruption detection, environment mismatch rejection,
subgroup forwarding and pre-fence/reduction/post-fence failure accounting.
Control uses XCCL new_group([0,1]), contiguous FP16 4x5120 then three
2048x5120 reductions, 45-second process-group bounds, explicit completion
stages and mapped native-library hashes from the future GPU process.
VERDICT -> CPU-tested diagnostic source prepared; no GPU execution or image
change. Synthetic arange/add allocation is explicit and does not reproduce
loaded embedding provenance, group creation history or earlier model work.
Next integrate this control into a bounded leased lifecycle with serving
environment reference and exact image/source hashes, pre/post-health and
crash recovery; use one named coordinator and a new output directory.
The outer lifecycle timeout is required; group timeout alone is insufficient.
After fresh-process evidence, compare loaded-context behavior and capture its
actual producer branch/allocation before claiming a matched reproduction.
No campaign test running; timer enabled; production offline. SGLang long
context, four-active-200K FP8, prefix/MTP/graphs and RAM reload remain pending.

## 2026-09-09 20:45 UTC review: fresh subgroup lifecycle integrated and launched

CONFIG -> No campaign workload alive at entry; S0f rc1 with owned rebind,
both per-card and compiled P2P-off post-health passed. Exact adc915d266
SGLang image and recorded serving environment; synthetic FP16 subgroup
control source 8ea40270a46d3d0cb9bc426df227c3526a8a3083ef03c820c5d3b0def3ca2dc1.
COMMAND -> Add campaign-local queued lifecycle compatible with run_arm.py.
CPU-test outer timeout, STOP race, removal verification, recovery ordering
and health failure accounting. Run exact-image CPU tests without GPU mounts,
network disabled, 2 CPUs/2 GiB; assert serving environment including absent
LD_PRELOAD/kernel-path/log overrides. Launch one fresh named coordinator.
RESULT -> Eleven CPU tests pass. First CPU mount at /control was too shallow
for server.py repository-root resolution; original import failure retained,
then /workspace/control mount passes without source changes. Environment
assertion passes. b70-cache800k-c0.service coordinator PID740790, lease wrapper
PID740802 and lifecycle PID740810 are active; both leases acquired and
pre-health underway. The frozen job permits one 240-second diagnostic;
container removal is verified before crash recovery or post-health. Source
snapshots, hashes and commands: collective-lifecycle-integration-2047/ (label
only; actual preparation 20:43-20:44 UTC). Plan/log prefix:
c0-sglang-fresh-subgroup-fp16-p2048.plan; raw arm directory has the same base.
VERDICT -> Bounded changed diagnostic launched; no GPU result yet. Next tick
inspect numerical results, per-rank completion stages and GPU-loaded library
hashes, then actual teardown/recovery/post-health. Synthetic allocation and
fresh group history differ from loaded SGLang; even a pass cannot establish
loaded-context correctness or isolate root cause. No native/image changes,
production restart or promotion. Timer enabled. SGLang long context, four
active 200K FP8 contexts, prefix/MTP/graphs and RAM reload remain pending.

## 2026-09-09 20:52 UTC review: fresh subgroup stalls without model load

CONFIG -> C0 exact adc915d266 image and serving environment, P2P off,
synthetic contiguous FP16 tensors, XCCL new_group([0,1]), 240-second bound.
COMMAND -> Inspect lifecycle, per-rank stages, source, kernel log, removal,
rebind and post-health; CPU-parse/assert both rank traces.
RESULT -> 4x5120 SUM passes on both ranks. First 2048x5120 reduction
(20 MiB/rank) completes both producer pre-fences, then both ranks enter
all-reduce without host return. Outer timeout rc124; the 45-second group
bound did not end the call. GPU-mapped library dictionaries match ranks.
Container absence verified; rebind, both cards and compiled P2P-off
post-health pass. Unit failed, MainPID=0, lifecycle rc1; no workload remains.
Raw assertions/snapshots: c0-failure-diagnosis-2052/. Initial line JSON parse
failed on concatenated rank records; stream parsing succeeds, raw preserved.
VERDICT -> Bounded CPU diagnosis complete. Fresh synthetic work stalls
without loaded-model history; same root cause as S0f is not established.
Next CPU-validate one C0 environment change: topology recognition enabled
(CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=1), retaining P2P=0 and exact
source/image/shape/group/other environment. Use a new directory and one
named leased coordinator with bounded recovery and health. The topology
override is a candidate, not a diagnosed cause. No GPU attempt this review.
SGLang long context, four-active-200K FP8, prefix/MTP/graphs and RAM reload
remain pending. Timer enabled; production offline; no input required.


## 2026-10-08 - Flash-Next Unsloth Q4 campaign authorized and prepared

CONFIG -> User selected Unsloth UD-Q4_K_XL for the dual-B70 plus host-RAM
campaign, authorized current GPU-service downtime and any backend/kernel/profile
work. Target revision766911a6b7369840a91dbcd95f9f997acaab6cd6; four target
shards111334654784 bytes. Publisher MTP Q4_K_M and BF16 vision sidecars retained.
Host kernel7.1 and UMD26.22 unchanged; old native5802 image/config retained.

COMMAND -> Audit current Steve/Sergio/Strata evidence and neural.download
methodology; fetch current Strata and fresh llama.cpp sources; snapshot live
stack/model identities; start bounded hash-verified intake; stop native5802 via
its owned leased STOP lifecycle; prepare/start pinned oneAPI2026.1 build.
Write docs/20261008_flashnext_udq4xl_campaign.md, exact model/backend locks,
header-only inventory tooling and a lease-enforced current llama.cpp build.

RESULT -> Strata fb58e0d and llama.cpp de7fa0a pinned. Current llama.cpp has
layer-split GPU MoE cache over host experts, so it is now a primary candidate
as well as fidelity reference. Strata needs native-Q8 hyper-connection fidelity
and potentially per-stage host mirrors. Current vLLM0.31.0 and SGLang0.5.21
source audits show blockers for this exact GGUF/XPU combination. The old
service/backend exited0; per-card and compiled two-rank10-iteration P2P0
post-health passed. Worker XPU shutdown completed; executor sent SIGTERM after
its worker grace period. Service inactive; original lease released.

CPU validation: manifest/lock agreement, syntax/ASCII/whitespace checks,
verified-byte acceptance and same-size corruption rejection, bounded GGUF
header parsing with complete/incomplete split and truncation checks. The first
target shard is metadata-only; the downloaded MTP header has34 tensors. Model
intake and compiler build remain running under separate user units; no complete
model hash verification, new-backend GPU qualification or serving pass claimed.
Raw receipts/logs: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/.

VERDICT -> Research plan and reproducible preparation established. Next gates
are completed intake/build, actual tensor memory map, primitive/runtime health
and deterministic no-MTP dual-card baseline. No shelf promotion. Historical
Steve FP8 and Sergio different-checkpoint speeds are not this model's results.


## 2026-10-09 - Flash-Next Q4 dual-card screening baseline established

CONFIG -> Unchanged pinned Unsloth UD-Q4_K_XL; llama.cppde7fa0a; oneAPI2026.1,
F16 compute OFF, FP16 KV, C1/8192, no MTP, explicit48/52 layer split and
first/last-eight host expert blocks plus host lazy PLE. Original weights
and user changes preserved. No shelf promotion or benchmark speed claim.

COMMAND -> Complete intake/header inventory; repair double oneAPI startup;
build current source; prepare hash-pinned NEO26.22 runtime; build test-only
production-shape overlay; run per-card comparisons; localize runtime crash
with host debugger; compare cache0; launch bounded repeated model screens.

RESULT -> All115028467617 selected bytes hash-verified. Actual expert71.729GiB,
PLE26.822GiB and main total103.678GiB. Build/runtime complete. Initial primitive
process crashed in SYCL persistent-code-cache getSortedImages/strcmp; isolated
small cases reproduce, upstream Intel fix49dc9346 closely matches. Cache0
changes no arithmetic; full36/36 per card, exact group counts, normal exit and
per-card/compiled-pair post-health pass. Earlier wrong count5 corrected to9
with original plan archived; failed/cache-enabled evidence remains failed.

First no-warmup model screen:12 individual checks pass, prose repeat differs;
strict verdictFAIL preserved. Default upstream warmup clears memory/state and
primes first-use paths. Warmupv2 and independent verbosev3 each pass12 checks
and within-run repeats; all six text hashes match across starts. Normal
container exit/removal plus per-card and ten-iteration compiled P2P0 collective
post-health pass. No GPU-fault signature. Actual GPU model buffers25994.56/
26861.90MiB;48 routed expert tensors overridden to host, intended layers
0-23/24-48 confirmed. CPU mapped coverage53.22GiB is not resident RAM.

VERDICT -> Bounded six-case C1 text-screen authority established, not broad
quality/token authority, occupied8K retrieval, concurrency, MTP or speed
qualification. New source-only MoE-cache audit identifies a current view-based
reorder safeguard; evictions/refills and larger batches still need tests. Next
step is measured cache candidate/coherence, then profiling and paired cold
performance. Servers stopped after tests; GPU leases released. Research and
reproduction: docs/20261009_flashnext_intake_and_primitive_qualification.md.
Raw F01 receipts: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f01-20261009/.

## 2026-10-09 FlashNext F02 cache prerequisites and history controls

CONFIG -> Pinned de7fa llama.cpp/server f5a0f84c, NEO26.22 runtime39992d70,
Unsloth UD-Q4_K_XL revision766911a6, dual-B70 layer split48/52, warmup on,
FP16 KV, no MTP, ctx8192, graph/persistent-code-cache off. Cache arm hosts
all experts with32768MiB total GPU expert cache. Static arm retains F01
placement. No shelf promotion.

COMMAND -> run_primitives.py cache-prerequisites-v2 and cache-opt0-prerequisites;
run_control.py cache32768-candidate-v1, cache32768-opt0-history-v2,
static-opt0-history-control; analyze_history.py offline captures. Commands,
source/binary identities and raw responses in results/flashnext_udq4xl_20261008/
f02-20261009. Full methodology in docs/20261009_flashnext_gpu_expert_cache_comparison.md.

RESULT -> Corrected test registration gives47/47 fixtures and40 exact cache
mutation phases per card under OPT1 and OPT0. Initial disabled-registration
attempt failed coverage and is preserved. Cache candidate passes individual
answer checks but fails strict prose repetition; disabling optimization does
not repair it. Static OPT0 also fails prose repetition and immediate native
history repeat diverges at token0. Cache later35 output IDs repeat, yet top-five
probabilities drift. Equal string prompts/params/progress are recorded; input
IDs are not independently established. All three model runs stop/remove
normally and pass per-card and compiled P2P0 post-health, no GPU fault signature.
Profiler CPU-only preflight proves PID1 attach/stop kills target; supervised
child survives. No GPU profiler result or cache byte-verifier execution yet.

VERDICT -> Bounded cache layout/refill prerequisites pass; release cache
correctness does not. Static OPT0 result prevents attributing drift solely to
cache. Static OPT1 history comparison is the next matched control. Keep failed
receipts and diagnostic timing separate from performance claims.

## 2026-10-09 FlashNext static OPT1 and checkpoint exclusion

CONFIG -> Same pinned model/runtime/server and static layer placement, warmup
on, OPT1, detailed logging5. Second arm adds only --ctx-checkpoints 0.

COMMAND -> run_control.py static-opt1-history-control and
static-opt1-checkpoint0-history-control, followed by offline analyze_history.py.

RESULT -> Both strict twelve-case screens fail prose repetition. OPT1 native
history diverges after JSON at output22; checkpoint-off native history shares
35 IDs but probability drift remains. Progress changes31+4 to35. Both runs
stop/remove normally, pass per-card/compiled P2P0 post-health, no GPUfault.
Default-off cache byte verifier now has CPU ASan/UBSan self-tests and syntax
checks, not GPU qualification; independent builder prepared but not executed.

VERDICT -> Earlier F01 bounded pass does not establish general repeatability.
Cache is unnecessary for this failure; checkpoint-off alone does not fix it.
Shape-dependent dispatch is a confound. Preserve failed evidence and investigate
shared request state/kernel paths before any speed or shelf claim. Detailed
records: docs/20261009_flashnext_gpu_expert_cache_comparison.md.

## 2026-10-09 FlashNext Intel AutoRound INT4 source investigation

CONFIG -> User requested a curiosity/research lane; UD-Q4_K_XL remains selected.
CPU-only HF metadata/header and primary-source review; no model download or GPU
work. Agents reviewed Intel artifact and Lumnus four-B70 implementation.

COMMAND -> HF Intel/Qwen3.8-Flash-Next-W4A16-AutoRound model card/config/API,
64KiB bounded safetensors header; Lumnus/b70-flash-next source/docs at b103e0f.

RESULT -> Official Intel W4A16 tuning release exists. Reported four-task average
BF16 0.8362 vs INT4 0.8332; no local reproduction or UD-Q4 comparison. Full
17-shard artifact168.75193GiB includes95.36789GiB BF16 PLE tensors. Remaining
files exceed paired VRAM before workspace. Lumnus uses4B70/256GiB RAM and
substituted INT8 PLE from another source; exact PLE identity unverified.

VERDICT -> Promising alternative for future measured offload lane, not a
requested format switch or ready TP2 deployment. Primary evidence and quality
limitations: docs/20261009_flashnext_autoround_int4_sources.md.

## 2026-10-09 FlashNext F03 fusion exclusion prerequisites and engine direction

CONFIG -> User retains UD-Q4_K_XL and permits NVMe placement only for measured
performance benefit. Static OPT1 original checkpoint/ubatch settings; only
SYCL fusion disabled. Exact runtime/model/server remain pinned.

COMMAND -> run_primitives.py --sycl-fusion 0, all four primitive plans, through
pair lease; token-evidence client/source audit and Strata source review CPU-only.
Host storage audit uses df/findmnt/lsblk.

RESULT ->47/47 plus40 cache mutation phases/card pass, per-card and compiled
P2P0 post-health pass. Model history control running; no correctness verdict.
Tokenizer IDs now captured for native diagnostics. Recurrent-reset diagnostic
plan prepared only. Strata tips unchanged; Q8 HC execution and stage1 mirrors
still missing. SATA8TB has5TB free; /mnt/cache NVMe has671GiB free.

VERDICT -> Continue llama correctness lane with parallel Strata fidelity/source
planning. No drive relocation, new model download, performance or shelf claim.
Evidence: docs/20261009_flashnext_gpu_expert_cache_comparison.md,
docs/20261009_flashnext_strata_next_steps.md, and
llamacpp/flash-next/recurrent-reset-diagnostic-plan.json.

## 2026-10-09 FlashNext fusion exclusion result and latency priorities

CONFIG -> Static UD-Q4_K_XL OPT1, checkpoint-on, ubatch256, warmup, logging5;
only SYCL fusion disabled. Native diagnostic adds tokenizer evidence without
changing generation payload. Current user priority1/2streams, rare4, max4-6.

COMMAND -> static-fusion0-history-control via pairlease and offline history
analysis, following94fixture/80mutation-phase prerequisite pass.

RESULT -> Strict screen fails prose repeat. Native three-prose outputs share35
IDs but probability drift persists. Exact tokenizer IDs/counts and progress
match. Normal removal, per-card and compiled P2P0 post-health pass; no GPUfault.
Source audit finds graph-reuse rs_z/head/count safeguards and owned probability
snapshots; neither proposed source bug is established. Strata has genuine
batched slots and cross-GPU groups but fairness/residency needqualification.

VERDICT -> Fusionoff is not a correctness fix. Stop toggle stacking and use
bounded reset/activation instrumentation for the reference. Strata-style tiers
are the development target, startingnativeQ8HCfidelity then2slot/2group and
stage1mirrors. Profile latency/criticalpath, not bandwidth saturation. User's
newer llama.cppHexagonrepo is required methodology source; Speculais old and
not accepted as its substitute. Awaiting correctrepo while otherworkcontinues.

## 2026-10-09 Hexagon latency method and first Strata Q8 source increment

CONFIG -> User identifies newer profiling reference as hotschmoe/x2-nvfp4-lab.
Pinned e0756d2115dd22af0b7399515de41d9390004422, CPU-only clone/read. Strata
fb58e0d primitive-only nativeQ8HC/F32injection patch, defaultOFF and no model
callsites connected. Selected UD-Q4_K_XL unchanged.

COMMAND -> Read current Hexagon timing/negative-control/ABBA evidence; adapt
latency methodology. test_native_q8_hc_cpu.py; parent-leased icpx2026.1 -fsycl
-O2 object compilation in39992d70 image with no GPU devices exposed.

RESULT -> Current7edd8631 patch passes SYCL object compile. CPU tests pass12
production descriptors,10reject cases,63488 finiteFP16scale fixtures and F32
injection preservation. Earlier426bc4ac object receipt remains prior-snapshot
evidence only. No device arithmetic or fullmodel integration qualified.
Latency plan uses criticalpath attribution, trace output equivalence, negative
controls, interleaved clean measurements and1/2/4/6stream latency/fairness gates.

VERDICT -> Concrete Strata fidelity source increment is reviewable; GPU
primitives and composed HC integration are next. Strata-style tiers/concurrent
slots are the development target; llama remains diagnostic reference. No final
engine winner or speed claim. Evidence in docs/20261009_flashnext_latency_methodology.md
and strata/flash-next/native-q8-hc-design.md; finalcompile receipt under
/mnt/vm_8tb/b70/build/strata-q8-hc-final-20261009-tc6wgt0y/.

## 2026-10-09 Native Q8 HC GPU primitives and alternative source audit

CONFIG -> Continue saved full FlashNext goal; preserve unrelated dirty changes.
Pinned UD-Q4_K_XL, Strata fb58e0d, patch7edd8631 and runtime39992d70; synthetic
native Q8/F32 projection fixtures. Parent owns GPU execution; agents prepare
source/audits. Previous saved source/CPU/object work is verified progress.

COMMAND -> test_native_q8_hc_cpu.py; run_native_q8_hc_gpu.py native-q8-hc-v1/v2;
CPU source audit of mature GGUF/Intel/tier/prefix alternatives; prepare default-off
composition patch and FP64 intermediate fixtures. GPU runs use pair lease with
per-process card pin, strict per-card and compiled P2P0 pre/post health.

RESULT -> CPU descriptors/dequant pass again. v1 device discovery fails because
nonroot Docker process lacks DRM groups; no HC kernels execute, post-health passes.
v2 adds actual device group IDs:72/72 fixtures and22 rejection cases per card,
72 numeric negative controls/card, guard corruption control; exact repeat/batch
bytes,128-byte guards and source/input immutability pass. Worst NMSE3.05296e-10
and normalized maximum error1.74727e-5 are below frozen1e-6/1e-4 gates. Both
normal exits/removals and full post-health pass; no selected kernel fault signature.
Composed patch0002 passes CPU32 valid/1120 invalid preflights;48-case/344-stage
fixture reference construction passes, GPU composition qualification in progress.
Source audit confirms Strata per-stage in-memory prefix snapshots exist; disk
sessions and extra message-boundary checkpoints have distinct split restrictions.
Current vLLM/SGLang exact GGUF-XPU blockers remain; Kobold batching rejects smartcache.

VERDICT -> Native projection device arithmetic passes bounded synthetic gates,
not full-model fidelity or speed. Strata stays a development candidate and llama
an unqualified history-drift diagnostic reference. Full route wiring, composed
math, dual-stage RAM tiers, prefix/concurrency/fairness and shelf gates remain.
No promotion. Raw F04 receipts under results/flashnext_udq4xl_20261008/f04-20261009;
source audit in docs/20261009_flashnext_backend_source_audit.md.

## 2026-10-09 Native HC composition/write, exact source contracts and full engine

CONFIG -> Pinned model/source/runtime; native source patches default off.
Parent GPU ownership and preserved dirty worktree. CPU source agents prepare
HC/PLE/mirror/API changes; no Strata full-model or shelf claim.

COMMAND -> Leased composed48-case/344-stage and shared-write21-case GPU gates
on each card; fresh header + actual HC tensor byte audit; CPU artifact-derived
contract tests; source route TUs and full CMake engine builds. Build independent
llama reset observer, add test-only production fixtures preserving all serving
binaries, qualify same-build primitives, run diagnostic-off history control.

RESULT -> Composed and shared-write GPU gates pass frozen1e-6/1e-4 thresholds,
exact repeat/chunk/single/pending bytes, guards, immutable inputs and negatives;
normal teardown/per-card/compiled P2P0 post-health pass. Shared-write worst
NMSE6.25260e-16/maxnorm6.15842e-8. Actual387 HC source descriptors/tensorSHA256
verified; original norm rank1[10240] corrects a loader/binding assumption. Actual
96 injection tensors are BF16-exact, while some97 F32norms are not. Stage range
ownership correction prevents duplicated HC/head uploads. Full engine v1 fails
PLE size_t initializer, v2 fails shadowed SYCL verifier header; separate0008/0009
fixes make v3 full engine/native-expert/snapshot binaries build successfully.
Original PLEQ8value/F32conv compatibility rounding is identified; native source
key/value/conv patch and device fixture prepared, GPU PLE not yet qualified.

Reset observer full server build and same-build36/36 primitives/card pass.
Original build plan exact bytes recovered by removing only later experiment
metadata, hash matches receipt; pinned snapshot/provenance preserved. Test-only
augmentation keeps all serving libraries/server hashes. Diagnostic-off strict
screen still fails; native P/P/JSON/P immediate prose shares35 IDs, after-JSON
prose diverges at22, probability drift starts0. Normal teardown/post-health pass.
Metadata-only arm is running next, not yet a numerical reset verdict.

VERDICT -> Concrete native math/source/build progress, no full serving winner,
quality, prefix, tier, concurrency, latency or shelf qualification. Strata is
primary serving-development lane; llama remains unresolved divergence control.
New raw evidence F04 and docs/20261009_flashnext_native_hc_gpu_qualification.md.

## 2026-10-09 PLE GPU math and first reset mapping evidence

CONFIG -> Same pinned source/model/runtime. Native PLE production-shape fixtures;
llama same-build static OPT1/warmup/history observer metadata lane.
COMMAND -> Leased native-ple-v1; offline reset-metadata-history-v1 history/mapping
analysis after independent diagnostic-off control. Numeric readback arm follows.
RESULT -> PLE7cases/70stages/57historyrows per card pass frozen1e-6/1e-4 gates,
repeat/chunk bytes, guards, source/input immutability and negative controls;
worst NMSE1.22648e-13/maxnorm4.55320e-7. Clean teardown and full post-health pass.
Metadata lane records32 FNRESET entries, actual prompt31+4 token hashes match
immediate repeats. First layer0 R/S mapping has rollback0/src0=rs_z=0 on fresh
request; continuation has rs_z=-1 and zero-sized reset views. This proves
selection/mapping, not numeric clearing. Metadata strict screen passes in this
one launch but native probability drift begins0 despite shared35 output IDs;
previous off arm strict screen failed/after-JSON diverged22. Clean post-health.
VERDICT -> No history correctness fix established. Selected state rows can now
be observed directly in numeric arm; layer1 PLE and later layers remain outside
this observer. Full Strata model/tier/prefix/concurrency/latency/shelf gates open.

## 2026-10-09 Actual reset rows and original-source upload gate

CONFIG -> Same pinned model/runtime; reset observer numeric arm and original
Strata source upload oracle. Fifth engine source generation adds strict API
identity and complete PLE allocation accounting. User changes preserved.
COMMAND -> reset-numeric-history-v1 plus independent raw bit/history analysis;
full native-source engine v5 build; compatibility metadata/tokenizer pack with
explicit conversion audit; actual source-upload-v1 first-card narrow owners.
RESULT -> Numeric arm selected6 paired layer0R/S rows have finite/nonzero pre
state and all positive-zero post state;12 raw files independently verified.
Native three-prose outputs share35 IDs but probability drift remains from0.
Normal teardown/post-health passes. PLE/later layers/fourth request unobserved.
Full engine v5 builds including0010 accounting. CPU pack host regex missing and
container NumPy missing failures preserved; isolated host dependency path makes
intake-v4 complete, exact GGUF tokenizer preqwen35/special IDs retained. All
196 rounded compatibility copies are explicitly marked as requiring native
source overrides; expert and PLE table bytes unchanged. Not fidelity authority.
Actual card0 upload oracle verifies27HC+3PLE source images against preregistered
SHA256/raw bytes, stage ownership and allocation accounting. It then fails
freed-pointer type-registration check; normal process removal/full post-health
pass. Pooling versus leak is being localized, not waived or labelled solved.
VERDICT -> First-layer reset hypothesis narrowed; history drift remains.
Actual source upload bytes pass first-card scope, lifecycle oracle unqualified;
second-card/pair and model/tier/prefix/concurrent/latency/shelf gates remain open.

## 2026-10-09 USM post-free query gate localized independently

CONFIG -> Pinned runtime/source header, default allocator controls and separate
UR trace arm; no model/weights/inference. Previous goal turn classified progress:
source/model-byte and reset evidence changed next action; preserved checkpoint4b2f783.
COMMAND -> run_usm_free_control.py default and --trace on both cards,42cases/card;
independent chronological trace audit with missing/duplicate free negatives.
RESULT -> All plainSYCL/guardedSYCL/rawZe owning frees/fresh probes pass; every
post-free type query remains device. SYCL/guarded keep backend ranges; rawZe
range query fails/null/0 while cached type/properties remain. Trace records60
successful matched alloc/free pairs/card,0 outstanding logical allocations;
deleting/duplicating free events rejects. Normal process removal and full
per-card/compiledP2P0 pre/post-health pass; no traced fault signature. Earlier
compile missing fcntl/unistd includes is retained; corrected snapshot compiles.
VERDICT -> Source-upload-v1 remains failed. Its pointer-type teardown gate is
not a valid logical-live test on this installed runtime. New owner-specific
matched-free/byte/lifecycle oracle required, not a silent gate waiver. Model,
full390source coverage, tier/prefix/concurrent/latency/shelf gates remain open.

## 2026-10-09 Logical source frees, cached-page recovery and first full serve

CONFIG -> Exact selectedGGUF/Stratafb58e0d/runtime39992d70, sourceenginev5,
strict source/lifecycle oraclev2 and bounded C1 diagnostic. Goal remains full
twoGPU/prefix/concurrent/latency qualification. Prior status-only turn classified
no progress; terminal84809 revalidated before continuing. User dirt preserved.
COMMAND -> F06/source-upload-v2; independent full buffered/direct shard3 hashes
and full49GB comparison; targeted onepage reload then all4full hash C1prepare;
qualify_c1_serving.py firstonecard; host-alloc-limit-v1; immutableenginev6build.
RESULT -> Source30-image owner lifecycle passes card0/card1/pair, exact27HC3PLE
readback/accounting and62alloc/free pairs,53owners,3contextbridges with negatives.
Full strict/compiledP2P0 pre/posthealth and normalremoval. Not full390/modelmath.
Buffered shard3 differed from publisher despite unchangedinode; directfullhash
matched. Full comparison finds onecachedpage/onebit at3857881836,0x65/0x45,
Q4_K blk13up expert79. Captures retained; targeted4KB invalidation reload and
fullbufferedhash match, then all4fullhashes pass. Cause unknown/no historylink.
Firstmodel loaded691native images3671.29MiB and8083expert slots23.59GiB, then
one49293MiB host allocation failed; strict coverage refused serving. No ready
or inference. Normalremoval/posthealth; changedpage sentinel remains exact.
Independent untouched hostalloc1MiB/1GiB/8GiB succeeds,48GiB returnsNULL under
8MiBmemlock; device maxalloc32530182144. Normalexit/removal/percard/compiled
posthealth, nofaultsignatures. Initialmissingheader/nestedshell failures retained.
CPU segmented plan preserves16493expert extents51688243200B in49segments, source
bytes/globalbudget/lifetime/context/free-retry checks pass. New0011/0012engine
build launched; fullbuild/GPUmirror/source390/model qualification still pending.
VERDICT -> Logicalfree gate localized and newlimitedscope passes. Wholeartifact
identity restored withoutweightwrites, corruptionorigin unproven. Allocation
evidence supports boundedsegments, not serving stability. Fullgoal staysactive.

## 2026-10-09 Complete original HC/PLE loader and lifecycle qualification

CONFIG -> ImmutableStrataenginev6 with0011/0012, exactmodel/runtime/pack;
full390source roster and v2 chronologicalfree gate. Prior continuation changed
allocator/identity/source state and is classified progress.
COMMAND -> fullenginev6 build, independentall4fullhash postfirstserve scan,
fullsourceoracle link and run_source_upload_oracle_full.py fivecases.
RESULT -> Engine threeexecutables build/link pass277sec; source/planunchanged.
All4fullhashes publisherexact with unchangedpre/poststatincludingctime. Actual
387HC3PLE sourcebytes/SHA/type/shape/offset pass fullcard0/fullcard1/sameGPU24_24/
twoGPU24_24/actualmodelstaticbounds. 300ordinarynative matrices enumerated but
ordinaryGPU payloadreadback outside thisoracle. Ownedimagebytes3821772800:HC
695132160,PLE34979840,ordinary3091660800; priorPLEestimate understated640B.
Fullcases690imagepointers+scratch/694allocfreepairs; split345images/stage+
scratch/698pairs total. Exactstageownership192HC3PLE150ordinary then195HC0PLE
150ordinary; allownerdestructorsreturn, ledger0live, missing/double/failedfree
negativesreject. Knownpage sentinel exact before/aftereverycase. Normalremoval
and fullstrict/compiledP2P0 pre/posthealth pass, nofaultsignatures; terminal
session72782 exit0 in195sec. NewsegmentedC1prepare hashing/identity inprogress.
VERDICT -> FullHC/PLE source loader/ownership gate passes. No fullmodel inference
result. GPUmirror/sourceexpertmath/state/logits/prefix/concurrent/latency/shelf
requirements remainopen; goalactive with parallelCPU research and parentGPUowner.

## 2026-10-09 Segmented expert GPU consumption, arithmetic and graph lifetime

CONFIG -> ExactGGUF/Strataenginev6/runtime39992d70;10actualexperts/3tokens each,
Q4_K/Q5_K GU andQ5_1/Q8_0 down, source-defaultgroupedmode;64MiB globalmirror
cap/4MiBsegments. Previousfull390goalturn classifiedprogress. Userdirt preserved.
COMMAND -> Immutablemirrorfixtures1/3/4 compile; parentcontrollerschema1 then
schema2 oncard0/card1/pair; originaloffsetsourceSHA roster beforeGPU.
RESULT -> v1compilefails toolchainheader;v2unexecuted;v3correctactualgraphvoid
APIs builds. FirstGPUrun fails hiddenpacketgate becauseoracle readsunwrittenh
scratch indefaultfusedkernel, notamodelfixfinding. Normalremoval/posthealth.
v4keepsmodelkernel/mode andadds separateGPU sameexpressionh reconstruction,
explicitlynotobservedrawfusedh; independentCPUQ8packets matchactualfusedHQ.
All3casespass10sourceexpertbytechecks/40strictmetrics, worstNMSE1.03152073e-13
andnormalizedLinf4.85313131e-7. GPUvsF32originalactivationrelativeL1max0.01019814
isdeclaredquantizationdifference withinupstream0.03, notimplementationerror.
Mirror/resident/repeatedgraph bytesexact; graphreplayafter sourcecloseexact.
33587200mirrorbytes6+4segments,10storage/20mirrorreads percase; foreign/missing/
budget/segment/partialread negativesreject. 126successfulalloc/free pairs,12
registeredsegment/tableowners,2contextbridges, ledger0live, all3freenegatives
reject. Normalexit/removal/fullstrict+compiledP2P0preposthealth,nofaultsignatures.
New18reportvalidatornegativespassCPUonly. ActualfullmodelsegmentedC1launched
underparentlease session5156; readiness/coherence/capacitystillpending.
VERDICT -> Boundedsource/tier/defaultkernelmath/graphlifetime qualificationpasses;
rawfusedh remainsunobserved. Full48GiB/all-expert/modelstate/logit/prefix/
concurrent/latency/shelf gatesremainopen. Fullgoalactive, no productionclaim.

## 2026-10-09 First native full-model segmented readiness

CONFIG -> Exactartifact/enginev6/one-card-segmented,8092variable-format resident
experts and16484overflow experts. Full390andboundedGPUmirror gatespassed.
COMMAND -> qualify_c1_serving.py C1onecardsegv1, independentraw-evidenceaudit;
combined0013..16sourcebuild preparation.
RESULT -> Fullhostmirror51,660,083,200B in49segments fits; INFOready andboth
verifiergraphs captured. Sixcorrectconstrained APIanswers; rendered/submitted
IDs exact, allpromptcounts consumed, actualgenerated/DONEcounts equal, marker
andhistoryrepeat IDs exact. Nativeengine exit0/normalcontainerremoval/full
strict+compiledP2P0preposthealth,nofaultsignatures; knownsourcepage exact.
Automatedscreen remainsFAILED: normalEOS GeneratorExit taggedaserror inall6
requests; launchsupervisor races explicitownedstop duringcontainerremoval.
Independentaudit recordsactualevidence, qualifiedfalse, originalfailureunchanged.
CPUagentpreparescorrectedtracer/stopsynchronization/negativecontrols; fresh
run required. Combinedprefix/observer firstplan failspristine-sourceidentity
validationbeforebuild/GPU becauseconsumedhashes wereinpristineledger. Oldplan
preserved; parentnewplanv2 correctsonlypristinehashes, fullbuildsession41878
running. No sourcepatch/math changedfor thatconfigurationcorrection.
VERDICT -> Actualfullmodelreadiness/capacity andboundedanswers observed.
Qualificationtools mustbefixedandrechecked; fullstate/logit/prefix/concurrent/
latency/shelf requirementsstillopen. Goalremainsactive withGPUownershiplocal.

## 2026-10-09 Full observer and explicit prefix controls source generation

CONFIG -> Unchangedmodel/runtime/externalStrata; separateimmutable0013..17
source increment: fullfirstlogits/layerresidualobserver, actualevaluationledger,
serialcomplete-stage pin andfreshreset controls. Diagnosticsdefaultoff.
COMMAND -> Fullcombinedplansv1/v2/v3, actualprefill/verify objectchecks, CPU
ledger/collector/serialharness negatives; currentcorrectedC1rerun session64837.
RESULT -> Firstplanidentitygate failsbeforebuild becausepatchedgenerate/new
header wereinpristine ledger; preserved, v2correctspristineledgeronly. Fullv2
build fails because0013memberswereinPeerPrefill notactualPrefill::Impl. New
0017movesmembers usingexplicitstructcontext, actualconsumedprefill+verify TUs
compile0; oldpatch/buildpreserved. Fullv3 buildsall3requiredexes exit0/279sec,
receipt /b70/build/strata-native-hc-engine-20261009T080647Z-6_ja3xd5/receipt.json,
source/plansnapshot unchanged. NoGPUinference fromnewgeneration. CPUhelper
contracts/byteflip/ledger/cached-reportcontrols pass, notmodel/cachefidelity.
Newserialqualifier enforcesfresh/cache withinonewarmedprocess, root/pin/turn/
park/eviction/cancel groups, actualspans andfullfinite248320logits plus48layer
residuals;6armedcaptures/process, rawflagofflogits explicitlyunobserved.
Fresh all4fullshardhashes aftersegmentedmodelserve publisherexact/unchanged
statinclctime, knowncachedbit originstillunknown. CorrectedC1tooling rejects
incomplete/cancelled/error closes andnonzerosupervisors, serializesownedstop;
CPU16negative/race casespass. Newpreparedv2 launched; actualrerunpending.
VERDICT -> Fullnewsourcegenerationbuilds; originalartifact identitystillvalid.
Actualfullmodelstate/logit/cache/concurrent/latency/shelf qualificationspending.
PriorfailedC1 remainsfailed, notretroactivelypromoted. Goalactive.

## 2026-10-09 Bounded one-card C1 passes; two-card stream ownership localized

CONFIG -> Exactmodel/sourceenginev6; correctedtracer/serializedstop; two-card
layer32split followsmatchedonecardqualification. No driver/kernelchange.
COMMAND -> C1onecardsegv2, C1twocardsegv1 withmatchedreceipt, bin/xe-resetrebind
underpairlease plusexplicitpinnedstrict/compiledP2P0health.
RESULT -> Correctedonecard screen passes all6answers/transport/counts/repeatIDs,
native+supervisorexit0/removal/health; explicitlynotfullfidelityorconcurrency.
Twocardcompleteplacementloads:17041resident+7535RAM experts23557120000B in
stage0,layers0..31; stage1layers32..47all8192resident/missing0. Readybutfirst
requestfailsmirrorqueue/residencyguard afterstage0T2capture; unwindingUR37.
Poststrict/compiledhealthpass,nofaultsignature; processremoved. Sourceaudit
finds SYCLVerifierext_stream_ storesGPU0defaultqueue atconstruction, thenGPU1
initmistakesitfor explicitsuppliedqueue. Guardretaintoexposecontext/ownership
errors. New0018sourcefix+capturequalificationpending, nofullmodelresponses.
Rebindbothendpoints/explicitpinnedhealth passeswithoutreboot,kernel7.1same.
Combinedobserver/prefixenginev3alreadybuildspass; itsfull390oraclelinkedpass,
GPUupload/newfullstatebasicyetpending. Serialharnessv2requires0017+matching
newenginegate; trackedpromptfixture replacesgitignore-hiddenfilename with
identicalbytes. OnecardpostGPUall4hashrefreshpreviouslymatchespublisher.
VERDICT -> Onecardscreenqualifiedonlyboundedscope; realtwoGPUsourcebugnow
localized. Recoveryhealthdoesnotfixsource. Fullgoalactive, allremainingmodel/
cache/concurrent/latency/shelf gatesopen, userdirty1043journallinespreserved.

## 2026-10-09 Init-owned verifier queues and corrected32/16 source gates

CONFIG -> Exactmodel/runtime, freshengine0013..18; nullableconstructorstreams,
explicitqueue validationbeforealloc, owningcapture/teardown, mirrorguardkept.
Previousgoalturn classifiedprogress: actualonecardpass/twocardfail/localization
andnonrebootrecovery changedauthoritativeevidence. Userdirtpreserved.
COMMAND -> Fullenginebuild277sec, boundedactualVerifier constructor0/init0_1
control, correctedfull390upload5cases withactualmodelstatic32/16bounds.
RESULT -> Full3executables linkpass/sourceplansunchanged. Actualcontrol4graphs/
8replaysT2thenT1/oppositecaller,9negatives/callerrestored/14matchedUSMfrees
withallfreenegatives pass; notfullrecord_window math. Fullstrict/compiled
P2P0preposthealth/normalremoval/nofault. All387HC3PLE readbacks byte/SHAexact
percase,300ordinarycount/account only. Actual32/16 ownercounts256HC3PLE200
ordinary then131HC0PLE100ordinary, bytes2554839040+1266933760=3821772800.
Sourceowners/alllogicalfrees/negatives/zeroledger/normalexit/fullhealth pass.
All4freshpost-twocard/rebind wholehashes publisherexact, statinclctimeunchanged.
Strictobservercoverage testsnowrejectempty/incompletehead/stage/48layers;
V4serialpilot keepssame2048/64/65536geometry bothcards, exact18binding and
honesttopologyaliases. No GPUstate/cache resultfromthoseCPUtests. Corrected
onecardC1preparation session18630 running; actualmodelrerunnext.
VERDICT -> Actualownershipcontrol/sourcegate passes, fullmodeltwoGPU/state/
logit/prefix/concurrent/latency/shelf stillunqualified. Goalstaysfullandactive.

## 2026-10-09 Cached source bit recurs; direct reader prefault discriminated

CONFIG -> Exactartifact/correctedengine18; allsource/controlgatespassed, newC1
preparedhashes/sentinelgood, strictidentityguardbeforeanynewmodelcontainer.
COMMAND -> CorrectedC1attempt, preservedviews/C-Python-GNUdirectmatrices, full
prefaultedshard3hash, targeted4KBreload, idle+separatehealthstage sentinelcontrol.
RESULT -> Sameknownpage3857879040 byte2796 changes0x45->0x65 xor0x20 again.
Launchfailsbeforecontainercreation at09:16:22 afterhealthonly; all sourcefile
statinclctimeunchanged. CPUtwocardpreparestopped, parentnormalposthealthpass.
Untoucheddirectbuffersalso showbadpage, butinitializeddirect C/Python buffers
original. Readcall/allocation/alignment/CLOEXEC crossedmatrices isolateprefault;
BtrfsFIEMAP unencoded, upstreamnofault->cachedfallbacksupportsonlyinference,
exact7.1branchunobserved. Fullinitialized16MiBdirectreader49,376,141,504Bhash
matchespinned56758f40... in158sec. No diskcorruptionclaim frombadO_DIRECTreads.
Bad/originalviews preservedthenonlyknown4KB FADVDONTNEEDreloadrestoresoriginal
SHA2780fef9... unchangedstat,no filewrite/globaldropcache/redownload.20sCPU
idle/card0/card1/compiledP2P0health controls allkeepgoodcached/directpage; no
reproducedcause. Healthd556processNeo26.27 differsmodel39992lane26.22; package
identityfactnotcausality. Newall4bufferedhash C1preparedv2 passes; guarded
rerunsession23893live. PriorfailedC1/sourceidentityevent remainsunchanged.
VERDICT -> Currentidentityrestored, originunresolved, strictguardsretained;
wholemodelcorrectedone/twocard/state/logit/cache/concurrent/latency/shelf open.
Actualprogress ratherthanblockedimpasse; goal remainsfullandactive.

## 2026-10-09 Corrected full-model serving passes one and two B70s

CONFIG -> Exactartifact/0013..18engine/FP16KV/MTP0; identityrecovered andfull
source32_16/captureownershipgates passed. ScopeC1boundedAPI, notshelf.
COMMAND -> C1onecard-corrected-streams-prepared-v2 followedmatchedtwo-card-v2;
postonecardall4wholehashes; actualV4serialprepare attemptedbeforeGPU.
RESULT -> Both6requestAPI screenspass rendered/submittedIDs, consumedcounts,
modelIDaliases, constrainedanswers andgreedyrepeatIDs. Bothstages captureT2
andT1; previousGPU1ownershipguardfailureabsent. Actual2GPUcoverage17041VRAM
experts+7535RAM stage0=23557120000B, stage1all8192resident. Native+supervisor
exit0/normalremoval/fullstrict+compiledP2P0preposthealth/sentinelexact. Fresh
postonecard4hashes publisherexact/statinclctime unchanged. ActualV4prefix
prepare fails tuplekey17_18 inengine_identity beforeGPU; failurepreserved.
SeparateV5single17lookup+separate18guard usesgenuinepositiveengineidentity
andone/twocardCPUprepare, samegeometry/prompts/tolerances. NoV5GPUresults yet.
VERDICT -> Fullmodel2B70tieredserving passesboundedcoherence/lifecycle. No
fullstate/logit/prefix/concurrency/matchedlatency qualificationorproduction
claim. Cachedbitcorruptionorigin stillunresolved; guardsretained. Goalactive.

## 2026-10-09 Frozen serial-prefix parent lifecycle wrapper

CONFIG -> Exact corrected0018 engine and V5 serial controller; parent wrapper
SHA256 6f2a28ec6aadbce4989a10ed25048f57fd84b9694fd37cea4af1ca21a1c350f6.
COMMAND -> python3 strata/flash-next/test_qualify_serial_prefix_cpu.py; Python
compilation; actual prepared-plan identity validation recorded in the CPU receipt.
RESULT -> CPU tests pass for success and failed-child paths, inherited lease
descriptors, stdout forwarding, owned teardown, post-health retention, fresh
four-shard buffered hashes, and terminal-time/finalization negative controls.
The first actual one-card basic suite is running under the parent-owned pair
lease at f06-20261009/serial-basic-onecard-v1; no numerical result is claimed.
VERDICT -> Wrapper source checkpoint only. Full model mathematics, complete-state
prefix reuse, concurrency, clean latency, and shelf promotion remain unqualified.
Evidence: strata/flash-next/qualify-serial-prefix.md and external CPU receipt
/mnt/vm_8tb/b70/results/strata-prefix-harness-cpu-20261009/parent-wrapper-receipt.json.

## 2026-10-09 Optional health-runtime source audit and preparation

CONFIG -> Preserved 09:14..09:16 cached-source incident; health image NEO26.27 /
IGC2.38.2 / loader1.32.0 versus model image NEO26.22 / IGC2.36.3 / loader1.28.2.
COMMAND -> CPU source/image metadata review and
python3 strata/flash-next/test_health_page_control_cpu.py.
RESULT -> ASCII/syntax/JSON/source bindings pass; eight preserved-page negative
controls pass. Historical invalid-source prose is retained with later restored
four-shard and corrected C1 receipt identities appended. Optional pure-SYCL source
and official-package derived-image recipe are prepared only; neither was built
or run. Loader provenance, IGC byte match and Torch/CCL compatibility remain open.
VERDICT -> No cause inference, stack replacement or matched-health qualification.
Keep current guards and current images; optional controls remain separate source
preparation. See strata/flash-next/health-source-preparation-review-v1.md.

## 2026-10-09 Serial prefix qualification coverage audit

CONFIG -> Frozen V5 and actual corrected0018 source, while parent basic GPU
suite runs; no edits to consumed model/controller/plan sources.
COMMAND -> Read-only goal-to-controller/cache/state source audit.
RESULT -> Identified missing nonfresh replay after decode cancellation, actual
generated-assistant/new-user continuation, exact snapshot admission/victim and
retained budget checks, and independently measured per-stage span provenance.
Recorded bounded next-version rosters and source locations in
strata/flash-next/serial-prefix-goal-gap-audit.md. The live basic suite's clean
and logits-only probes match output IDs and LP20 for A/B/C; layer suite and
final health/hash gates remain pending, so this is not a finalized GPU pass.
VERDICT -> Preserve V5 as bounded internal-consistency evidence. Complete-state
prefix, API session isolation, concurrency and independent mathematics remain
open; new coverage must be separately versioned and qualified.

## 2026-10-09 Finalized one-card full-logit observer equivalence

CONFIG -> Frozen corrected0018 engine/V5/parent, exact original model,
card0 all48 layers, context2048/prefill64/PLE65536 FP16KV MTP0.
COMMAND -> Parent qualify_serial_prefix.py --group basic; result directory
F06/serial-basic-onecard-v1.
RESULT -> Parent and child finalized PASS, exit0, 837s. A/B/C IDs and LP20
match diagnostics-off/on; full248320 first logits are bitwise identical with
activations0/1. All48 first-window residuals per probe have required coverage,
but activations-off residuals are unobserved and provide no reference comparison.
All native exits0/removals, pre/post strict+compiled P2P0 health, kernel gate,
and new complete four-shard publisher hashes pass. Matching two-card basic
suite acquired both leases and started; no result yet.
VERDICT -> Bounded observer consistency/lifecycle only. Independent mathematics,
complete-state prefix/cache/API/concurrency, clean latency and shelf remain open.
See docs/20261009_flashnext_serial_diagnostic_qualification.md.

## 2026-10-09 Independent original-GGUF mathematical reference gap audit

CONFIG -> Frozen corrected0018 source and embedded GGML3cf03257, exact selected
original four shards, qwen4exp architecture.
COMMAND -> CPU source census and independent-reference route audit; all eight
recorded source/receipt hashes rechecked.
RESULT -> Missing ref/model.py, gdn.py and qsa.py; existing ref/load.py omits
actual Q5_1 expert-down decoder. Full390 readback proves HC/PLE source ownership,
not ordinary dense/head/router/GDN/QSA or end-to-end mathematics. Recorded
bounded lazy original-GGUF reference plan with separate original-FP64 and actual
Q8_1/storage contract lanes; conditional replay cannot prove upstream state.
VERDICT -> Independent math remains open. Prepare a separate lazy exact-artifact
reference; do not treat same-engine topology/cache equality or fresh llama prose
as an authoritative mathematical oracle. See
strata/flash-next/independent-math-reference-route-v1.md.

## 2026-10-09 Finalized two-card diagnostics and whole-layer topology parity

CONFIG -> Frozen corrected0018/V5, matching2048/64/65536 pilot, actual32/16
physical split; stage0 resident8897/RAM7487, stage1 all8192 resident.
COMMAND -> Parent two-card basic qualification, then V5 compare-topologies
against finalized one-card report.
RESULT -> Two-card parent+child PASS exit0,498s with all native removals, strict
and compiled P2P0 pre/post health, kernel gate, and fresh all4 publisher hashes.
CPU comparison PASS all9pairs: exact IDs/LP20/natural stops;6 observed full
first-logit pairs bitwise equal;3 layers-enabled pairs all48 rows bitwise equal
(144paired residuals, max_abs0/RMSE0). Two-card root prefix suite started under
pair lease at F06/serial-root-twocard-v1; no prefix result yet.
VERDICT -> Topology/observer consistency, not independent full-model math or
complete-state prefix/concurrency/latency/shelf qualification. See
docs/20261009_flashnext_serial_diagnostic_qualification.md.

## 2026-10-09 Finalized two-card root cache reuse

CONFIG -> Frozen0018/V5/parent,32/16 two-card2048/64 pilot, parking disabled.
COMMAND -> --group root at F06/serial-root-twocard-v1.
RESULT -> Final PASS exit0,388s including native teardown, strict+compiled
P2P0 pre/post health, kernel gate and NEW complete all4 publisher hashes.
Both matched cache hits reuse21 tokens and evaluate28 rather than49 rows.
Two cached/fresh pairs plus fresh-repeat match IDs/LP20/natural EOS, full248320
first logits and all48 residuals bitwise (144pairedrows max_abs0/RMSE0).
Two-card191-token pin/209-token-input suite started at
F06/serial-pin-twocard-v1 under pair lease; no result yet.
VERDICT -> Bounded actual root reuse; independent whole-state proof, stage
measurement, main cache bytes, real continuation/eviction/cancel/API isolation,
concurrency and clean latency remain open. See serial diagnostic qualification doc.

## 2026-10-09 Independent exact-GGUF CPU decoder and scalar foundations

CONFIG -> New bounded original-GGUF reference files, pinned original1224-role
inventory/lock and official GGML3cf03257 source/library; no model payload reads
or GPU devices. Independent expectation lanes retain original-FP64 values and
explicit F32 dequant storage separately.
COMMAND -> python3 strata/flash-next/test_original_gguf_reference_cpu.py and
python3 strata/flash-next/test_original_math_scalar_cpu.py.
RESULT -> Parent rerun PASS:84 synthetic blocks across7 actual formats match
official gguf-py;60 quantized blocks match compiled official C decoder;15 rival/
role/MTP controls and13 synthetic reader admission controls pass. Nine scalar
properties pass, including labelled GDN modulo and QSA division head mappings.
The parent initially misread QSA's conditional without its default; source
Alt::kv_divide=true and consumed h/G confirm existing reference division, and
modulo is rejected as rival. No production math was changed. Prior PLE57-row
wording corrected by appended evidence: retained history9, dilation3;57 was
cumulative fixture token updates.
VERDICT -> CPU reference groundwork, not actual original model/operator/full
forward qualification. No native Q8_1 reduction emulation or full accumulated
error contract yet. Actual math requires current parent full-hash/sentinel
admission because receipt/stat alone cannot rule out cached-byte changes.
See strata/flash-next/original-gguf-reference-foundations-v1.md.

## 2026-10-09 Finalized two-card191-token pinned-prefix reuse

CONFIG -> Frozen0018/V5,209-token inputs with191-token shared pin,32/16
two-card2048/64 pilot, parking disabled.
COMMAND -> Parent --group pin at F06/serial-pin-twocard-v1; additional CPU
matching-prefill-roster compare_vectors on repeated fresh requests.
RESULT -> Final PASS exit0,439s. Both hits actually reuse191/evaluate18 vs209
fresh; cached/fresh/repeat full first logits and48 residuals bitwise match
(144pairedrows). Fresh prefill-last coverage96 rows each;192 repeated-fresh
prefill rows separately bitwise equal. Health/teardown/kernel gate and NEW
all4 publisher hashes pass. Parked snapshot suite numerical childPASS actual
ram_snapshot/42reuse twice; parent healthPASS but hash/finalization pending.
VERDICT -> Bounded pin/prefill seam observable consistency, not independent
whole-state/complete-cache/concurrency/latency qualification. See serial doc.

## 2026-10-09 Finalized two-card parked conversation restore

CONFIG -> Frozen0018/V5 two-card32/16 pilot,512MiB/one-slot parking.
COMMAND -> Parent --group parked at F06/serial-parked-twocard-v1.
RESULT -> Final PASS exit0,467s; two actual ram_snapshot hits reuse42/evaluate7
vs49 fresh. Both pairs match full logits/all48residuals bitwise (96pairedrows),
IDs/LP20/natural EOS. Logical selected/finished cache bytes peak356658400/one
entry, restored0entries/1361344retainedbytes; not total transient memory. Native
exit/removal, pre/post strict+compiled P2P0 health, kernel gate and NEWall4
publisher hashes pass. Bounded eviction suite started under pair lease at
F06/serial-eviction-twocard-v1; no result.
VERDICT -> Bounded actual two-stage snapshot restore observable consistency,
not raw persistent-state/complete-cache/API/concurrency/latency qualification.

## 2026-10-09 Cache lifecycle and stage slot ownership source checkpoint

CONFIG -> Separate0019/0020 source plus V6 controller/fixture/parent; original
fb58/GGML3cf/image39992 unchanged. Combined engine plan SHA
05041bd19eed14a9758ab00c9b02784308a126ce7a6b6414727668bda6c9fa1c.
COMMAND -> Parent CPU tests: prefix_lifecycle, serial_prefix_qualification_v6,
stage_slot_owner and qualify_serial_prefix_v6 (test_*.py).
RESULT -> PASS actual portable cache/header ASan/UBSan; V6 fullraw/phase/stage/
budget/victim negatives and tokenizer synthetic continuation; pristine+20-patch
reconstruction/all38 consumed hashes and actual owner-header/release/rebind
mocked allocator controls; V6 parent success/failure/old18 rejection.
Default-off lifecycle metadata adds observed snapshot instances, live chain and
completed stage spans. Slot owner adds null/partial cleanup, resize/fallback
frees, shifted QSA ordinal release and identical RoPE rebind.
VERDICT -> CPU/source checkpoint only. Full SYCL build, real one/pair ownership
oracle, NEW full390/prepared source gate, observer off/on model qualification
and actual V6 cache tests remain required. Existing V5 receipts do not qualify
this new generation or concurrent serving.

## 2026-10-09 Independent original first-GDN layer source preparation

CONFIG -> NEW source versions, immutable decoder/scalar foundations retained.
Original-FP64 first-layer composition owns embeddings/routing/expert selection
and temporal state; native-storage emulation remains unsupported.
COMMAND -> Parent test_original_vector_decoder_cpu_v2.py,
test_original_first_gdn_layer_cpu_v1.py and test_first_gdn_numeric_probe_cpu_v1.py.
RESULT -> PASS112 vector decoder blocks across both lanes vs frozen scalar and
official Python;17 asymmetric first-layer composition controls;18 source/position/
state/packet/identity metadata rejections for1/2/4/8-token numerical probe plans.
L2 uses sum(q*q)+eps, not RMS or floor; qwen4exp output uses sigmoid, not SiLU.
Expert scale absent tag is literal0 sentinel/effective1, nonidentity rejected.
All projection tiles bounded; actual model source payloads were not read and
actual original forward/GPU comparisons were not executed.
VERDICT -> Source and synthetic equations only. Production Q8_1 packets, native
F32 reductions/routing precision, captured outgoing state and remaining PLE/QSA/
48-layer/head independent mathematics remain prerequisites. Preregistered
conditional implementation-error limits stay separate from reported activation
quantization cost. No model/latency/natural API completion qualification.

## 2026-10-09 Combined20 full build and stage-owner oracle source preparation

CONFIG -> Committed combined20 engine plan05041bd1, unchanged39992 image,
fb58 source/GGML3cf. New no-weight owner oracle source recipe frozen separately.
COMMAND -> build_native_hc_engine.py --plan
strata/flash-next/serial-lifecycle-slot-owner-engine-build-plan-v1.json --jobs4
under automatic pair lease; parent rehash38 source files and3 executables;
python3 strata/flash-next/test_stage_slot_owner_trace_cpu.py.
RESULT -> Full SYCL build PASS exit0,284s, isolated source/plan unchanged.
Actual patched38-file ledger exactly matches expected; strata,
native_expert_parity and conversation_snapshot_test built. Receipt:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T112318Z-4uag6j5x/receipt.json
SHA1647fdb0941d6e960bdbded793849b998de9de963eb5a6e6c687f18ea5547b00.
Synthetic82-owner chronological context/free collector and missing/double/
failed-free/empty/wrong-roster negatives pass. Owner oracle not compiled/run.
VERDICT -> ABI/source build only. Real one/pair slot ownership, new full390
source upload and model/API/V6 qualification remain open. Old18 cache receipts
are not new20 qualification. Compile frozen owner oracle next under lease.

## 2026-10-09 Finalized bounded eviction reconstruction and combined20 build

CONFIG -> Old18 V5 eviction pilot remains separate from newly compiled20.
COMMAND -> Parent --group eviction; F06/serial-eviction-twocard-v1.
RESULT -> Final PASS exit0,387s; counter2-to6, target0reuse/55eval. Two full
head/all48residual comparisons bitwise equal, IDs/LP20/natural EOS equal;
health/teardown/kernel/NEWall4 publisher hashes pass. Combined20 full build
then PASS284s with38 expected source hashes and3 executables; owner oracle
compilation started under lease, not runtime-qualified.
VERDICT -> V5 bounded cold reconstruction, not exact target admission/victim
proof. New20 requires independent ownership/full390/C1/V6 qualification; old
receipts cannot establish it. See serial diagnostic qualification doc.

## 2026-10-09 New20 whole390 source-upload gate and owner runtime supervisor

CONFIG -> Actual combined20 receipt1647fdb0; full upload oracle
/mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-tkbp0s5d/receipt.json;
actual32/16 source plan and unchanged pinned model/source/runtime.
COMMAND -> run_source_upload_oracle_full.py under pair lease; result
F07/source-upload-combined20-v1. Parent CPU test_run_stage_slot_owner_oracle_cpu.py.
RESULT -> Source-upload parentPASS208s, all5 cases exit0 with original387 HC/
3 PLE payload readback, logical frees/negative controls, context bounds and
strict+compiled pre/post health. Ordinary300 payload readback and model math
remain false. New owner runtime supervisor genuine actual20/oracle source/ABI
checks and mocked3case success/failure/cleanup/health/fd8_9 controls pass.
VERDICT -> New20 source upload/lifetime qualification only. Commit runtime
supervisor before actual no-weight card0/card1/pair ownership execution; no
slot runtime/concurrent/model-math result from CPU mocks or compilation.

## 2026-10-09 Independent activation packet and prefill storage contracts

CONFIG -> New CPU-only Q8_1 references; frozen18/20 consumed sources pinned.
COMMAND -> Parent python3 strata/flash-next/test_independent_q8_1_activation_cpu_v1.py.
RESULT -> PASS28 native profile/shape cases vs independent scalar,21 official
compiled GGML variant cases,16 rival/rejection controls. Native F32 division
plus raw XOR-tree sum differs from official reciprocal multiplication/code-sum
(witness native16 versus official15). QFUSE unclamped fields are separate and
excluded by current QFUSE0. Native HC uses directF32. Ordinary prefill narrows
both dequantized weights and activations toF16, with distinctBF16 projection
seams; these arithmetic storage costs are separate from decode packet identity.
VERDICT -> CPU packet contracts and bounded33-field/28MiB capture layout only.
No GPU packets, raw fused hidden, actual model payloads or full math qualified.
New default-off layer0 hook0021 requires later source/build/off-on/context gates.

## 2026-10-09 New20 real slot ownership and one-card API screen

CONFIG -> Actual new20 build1647fdb0/source38 ledger, unchangedmodel/runtime.
COMMAND -> Parent no-weight owner oracle; genuinefresh-hash C1/V6 preparations;
parent onecard C1 screen.
RESULT -> F07/stage-slot-owner-gpu-v1 finalizedPASS178s,82/82/92 owners on
card0/card1/pair, actualisolation/rollback/resize/freshprobe, chronologicalfree
contexts, negative controls, exits/removals/health/kernel/sentinels pass.
F07/c1-onecard-combined20-prepared-v1 API screenPASS381s identity/coherence/
repeat/consumption/teardown/posthealth. Genuine new20 one/pairCPU and V6
source-chain preparations pass with independentfreshall4 hashes. Pair API
screen now running; no result claimed.
VERDICT -> Ownership and bounded onecard API qualification only. New20 V6
cache/model off/on math, full independentfidelity, batchgraphs/concurrency,
physical backing release and clean latency remain open. See combined20 doc.

## 2026-10-09 New20 two-card API screen passed; V6 basic started

CONFIG -> New20 two-card32/16 API8192/128 profile, finalized onecard prerequisite.
COMMAND -> Parent qualify_c1_serving.py two-card screen.
RESULT -> FinalizedPASS330s, served/registry identity/hotschmoe-dd first,
coherence/repeat/consumption, normal teardown and post-health. V6 one-card
basic suite started under pairlease at F07/serial-basic-onecard-v6-v1 with
matched2048/64/65536 native geometry; no numerical result yet.
VERDICT -> Bounded API screen only. V6 observer/caches, independent wholemodel
math, batch/concurrency and clean latency remain open. See combined20 doc.

## 2026-10-09 New20 V6 one-card basic and stricter record validation

CONFIG -> Frozen20/V6, card0 native2048/64/65536 pilot.
COMMAND -> Parent basic suite and external actual-record auditor.
RESULT -> FinalPASS724s, exact IDs/LP20/natural stops/full first logits off/on,
all48 residual coverage, teardown/health/kernel/NEWall4 hashes. Stronger final
auditPASS exact9 roster, cross PREFIX/PCL/SFD identities and exactstage
entry/return coverage; auxiliary unbound descriptions explicit. Initial external
auditorv1 falsefailure preserved (missing PID on descriptive reset event),
not a runtime/source violation. Matchingtwo-cardV6basic started, cleanPASS
remainingarms/finalgates pending.
VERDICT -> Bounded diagnostic and observedwork provenance; independentmodel
math/completecache/concurrency/latency remain open. See combined20 doc.

## 2026-10-09 Frozen read-only V6 strengthened audit and hash provenance

CONFIG -> Separate auditor/helper source; V6/controller/live runs unchanged.
COMMAND -> Parent test_audit_v6_actual_request_records_cpu.py and
test_hash_combined20_preparation_identity_cpu.py; actual completed onecard
records independently audited outside frozen run tree.
RESULT -> CPU negatives for unmatched stage entries, cross PID/request/source/
position/device and committed suffix pass. Actual9 onecard records pass stronger
checks and final expected roster. Hash helper now cross-binds genuine upload
oracle to exact20 engine/source/library recipe and rejects old18; tiny controls
pass without another full scan. Original helperv1 preserved as historical.
VERDICT -> Stronger observed record evidence, not full hidden state/commit proof.
Auxiliary reset/pin descriptions remain unbound; observed-only bool cannot
prove missing process/partial roster. Future V7 stricter parser must preserve
these gates; no V6 source/hash changes during actual runs.

## 2026-10-09 New20 two-card V6 basic and exact topology parity

CONFIG -> Frozen20/V6 matched2048/64/65536 one/pair controls.
COMMAND -> Parentpairbasic, externalstrengthenedactualrecords, CPUcross-topology.
RESULT -> PairfinalPASS557s all3processes/numeric/teardown/health/kernel/NEW4
publisher hashes. Exact9records perstage/request sourcebinding pass. Cross9pairs
PASS fullobservedheads and144 pairedresiduals bitwise, IDs/LP20/naturalstops.
Real_live pairchildnumerical+teardownPASS genuine51reused/28evaluated vs79
fresh, exactgeneratedassistant/template/tokenprefix; parentfinalgates pending.
VERDICT -> Internal observer/topology and record provenance, not independent
math/concurrency/cleanlatency. See combined20 qualification doc.

## 2026-10-09 Real generated continuation with actual live cache

CONFIG -> Frozen20/V6 pair, exactoriginaltemplate/tokenizer, four-requestroster.
COMMAND -> Parent real_live; F07/serial-real-live-twocard-v6-v1.
RESULT -> FinalPASS320s, seednatural42/newreply2; actualcommitted51IDs exactly
renderedprefixof79, live51reused/28evaluated. Fullhead/all48residuals bitwise
matchfresh with IDs/LP20/naturalEOS; strong4-record/stage/sourceaudit and
teardown/health/kernel/NEW4publisherhashes pass. Decodecancel isolationstarted
atF07/serial-cancel-decode-twocard-v6-v1 underpairlease; no resultyet.
VERDICT -> Bounded real-live continuation, not fullmath/hiddenstate/API/session/
concurrency/latency. Source/template/tokens not changed toforcehit.

## 2026-10-09 Decode cancellation with nonfresh cache isolation

CONFIG -> Frozen20/V6 pair, six bounded requests.
COMMAND -> Parent cancel_decode atF07/serial-cancel-decode-twocard-v6-v1.
RESULT -> FinalPASS327s. Actualdecode STOP/DONEcancel/PCLphase; nonfreshsame
42reuse/7eval vs49fresh, unrelated21reuse/24eval vs45fresh. Three fullhead/
all48residual comparisons bitwise, IDs/LP20/naturalstop equal; external6-roster
source/stage/request audit, teardown/health/kernel/NEW4publisherhashes pass.
Cancelledpartialwork/stateunobserved, not zero. Prefill isolationstarted under
pairlease atF07/serial-cancel-prefill-twocard-v6-v1; no resultyet.
VERDICT -> Bounded actualdecode-cancel nonfresh isolation, not reset-onlypass
or fullmath/completecache/concurrency/latency qualification.

## 2026-10-09 Prefill cancellation with same/unrelated nonfresh replay

CONFIG -> Frozen20/V6 pair,209/45-token isolation prompts.
COMMAND -> Parent cancel_prefill_isolation F07/serial-cancel-prefill-twocard-v6-v1.
RESULT -> FinalPASS321s; actualprefill STOP/DONEcancel zerooutputs;21completed
rows lowerbound, in-flightwork/stateunobserved. Nonfreshsame21reuse/188eval
vs209fresh; unrelated21reuse/24eval vs45fresh. Three fullheads/all48residuals
bitwise, IDs/LP20/naturalstop equal. Strong6-recordaudit, teardown/health/kernel
and NEW4publisherhashes pass. ExactV6evictionstarted7136 pairlease; no resultyet.
VERDICT -> Bounded actualprefill cancellation isolation, not reset-onlypass or
fullmath/persistent-state/concurrency/latency qualification.

## 2026-10-09 Exact snapshot admission/victim and cold reconstruction

CONFIG -> Frozen20/V6 pair,512MiB/one-entry cache, six-requestroster.
COMMAND -> Parent eviction F07/serial-eviction-twocard-v6-v1.
RESULT -> FinalPASS312s; targetinstance2 admittedrequest2 andsameinstance/
source/semantic hashes evictedrequest3 acrossbothstages beforecoldrequest5.
Snapshot356867440 bytes, allobservedheld+retained<=536870912/entries<=1.
Cold0reuse/55eval matchesfresh fullhead/all48residuals bitwise, IDs/LP20/EOS;
external6/source/stageaudit, teardown/health/kernel/NEW4publisherhashes pass.
New20V6pin191/209 started underpairlease atF07/serial-pin-twocard-v6-v1.
VERDICT -> Actualidentity/coldobservable consistency, logicalbudgets only;
fullmath/rawstate/OSpeak/concurrency/latency remain open.

## 2026-10-09 Layer0 numerical capture21 source checkpoint

CONFIG -> Separate0021 on frozen20; standalone21 engineplan SHA
5367d37727a4e3df8c53288ad96cdf22660cab4c448845a818cdd70a3197e016.
PatchSHA3c282ced590a5a71e5354c22438728cf326cbf78201497cdd1de708ce0f51081.
COMMAND -> Parent test_layer0_numerical_capture_cpu_v1.py.
RESULT -> PASS exact reconstruction and actualhelper under mockSYCL: defaultoff
zeroalloc/copies/frees,14 stale/partial/nonce/context/extent/quota negatives,
three frames26observed/7UNOBSERVED. Persistentnoncebacking/4sealedgraphkeys/
exactactualreplay nonce+keystamp gate; incoming/outgoingstate requires existing
T1selfcommit, refuses unsupportedflag ratherthan togglingmath.
VERDICT -> CPU/source only; fullSDKbuild/newsource390/off-on/output/state/owner
GPUqualification required. Shared/expert7producerfields remainunobserved;
partial26fieldcapture cannotqualify fullfirstlayer/wholemodel equations.
Frozen20/V6/reference/packet contracts preserved.

## 2026-10-09 Versioned batch identity22-v2 CPU source checkpoint

CONFIG -> Separate22-v2 on frozen20; first22 review retained. Identity-only
strict requested2/4/6 capacities, request/engine/slot generation metadata,
existinglastTLS extended with per-requestprogress/reuse/segments; no source
math/cache/scheduling policy changes claimed.
COMMAND -> Parent test_batch_request_identity_v2_cpu.py and
test_batch_identity_transport_v2_cpu.py.
RESULT -> PASS trackedsource reconstruction/hashes; actualAPI AST threadedTLS/
socket/pump/control, restart/BYIELD/solo migration/detached-drain; native
capacity/RID/BSTOP blocks hostg++ASan/UBSan and malformed/overflow/stale peer
cancellation negatives. ExistinglastTLS correction appendedsourceaudit.
VERDICT -> CPU/transportsource only. FullC++/ABI build and actual simultaneous
2/4/6 model/state/rawhead/spans/cache/fairness/teardown remain required. Source
pin/fresh guards remain; cannot qualifyconcurrency from metadata orcapacity.
Future concurrent artifact/controller mustfingerprint newPythonhelper.

## 2026-10-09 New20 longpin requalification and capture21 build

CONFIG -> Frozen20/V6 pair,191sharedpin/209inputs,2048/64 pilot.
COMMAND -> Parent pin F07/serial-pin-twocard-v6-v1.
RESULT -> FinalPASS331s both191reuse/18eval, three fullhead/all48residual
comparisons bitwise, IDs/LP20/naturalstop. Actualsource/stage/prefill84/148
coverage, exact6-roster/sourceaudit, teardown/health/kernel/NEW4publisher
hashes pass. New21 standaloneSDKbuild started underpairlease withtracked
sourceplan40 expectedfiles; no SDK resultyet.
VERDICT -> Boundednew20pin/prefill observable consistency. New21 source
compilation/off-on/ownership/originalpacket/math gates remainopen;7fields
unobserved and fullmodelmath/concurrency/latency stillunqualified.

## 2026-10-09 Capture21 fullSDK and source-upload oracle builds

CONFIG -> Frozen21 standaloneplan5367d377, source/model/runtime unchanged.
COMMAND -> Leased fullenginebuild, parenthashverification, leased full390
oraclecompile, then neworacleGPUruntimegate.
RESULT -> SDKPASS279s,40expectedsourcefiles and3exes; actualreceipt
48a677e70c58686ba68299d2607b7c37d41f1774fdb9eb811ab6a3d6f8771de7 at
/b70/build/strata-native-hc-engine-20261009T130403Z-8b4zuyvp. Oraclecompile
PASS47s chhk0m3k. F08/source-upload-capture21-v1 GPUgate running1293; no
runtime source/capture resultyet.
VERDICT -> ActualSDK/ABI compile only. Genuine newsourceidentity/readback,
20vs21off+same21off/on GPUstate/output/nonce/lifetime gates remain required;
26/33capture partial, fullmodelmath/concurrency/latency stillunqualified.

## 2026-10-09 Capture21 whole390 source gate finalized

CONFIG -> Actual21SDK48a677e7/chhk0m3k ABI/sourceoracle, unchangedmodel/runtime.
COMMAND -> Parent GPU sourcegateF08/source-upload-capture21-v1.
RESULT -> FinalPASS207s,5layouts original387HC+3PLE sourceidentity/readback,
logicalfree/context/negative gates, nativeexits/removals/strict+compiled
pre/posthealth pass. Ordinary300payload/readback andmodelmath false.
VERDICT -> New21sourceupload/lifetime only. Genuine21C1/identity/observer
off-on/packet/state/math andfullconcurrentlatency gates remainopen.

## 2026-10-09 Capture21 genuine manifests and onecardAPI screen

CONFIG -> Actual21/newsource390/Pythonruntime chain, diagnosticflagoff.
COMMAND -> GenuineC1CPUfullhashprepares; parentonecardAPI qualification.
RESULT -> OnecardfinalPASS279s identity/coherence/repeat/consumption/normal
teardown/posthealth. Bothfresh21one/pair manifests launchallowed; pairAPI
screen14312running withactualonecardprerequisite. No old20sourcegate transfer.
VERDICT -> Boundedactual21API only. Numericalequality/actualframe/packet/state/
nonce/lifecycle, independentmath, concurrency and cleanlatency remain required.

## 2026-10-09 Capture21 two-card API qualification

CONFIG -> Actual21/newsource390/Pythonruntime/model chain,32/16API8192 pilot.
COMMAND -> Parent two-cardC1 screen withactual21onecardprerequisite.
RESULT -> FinalPASS229s, live/registry identity/hotschmoe-dd first, coherence/
repeat/consumption, normalAPI/native/supervisor teardown and post-health.
VERDICT -> BoundedAPI only; newnumerical20vs21off/same21off-on/source/nonce/
state/packet/owner checks next, fullmath/concurrency/latency stillunqualified.

## 2026-10-09 Numerical capture21 orchestration frozen and genuinely prepared

CONFIG -> Newchild9fd3b713/parent7a3c2098, genuine21source/full390/C1/full4
identity plus finalized20native reference, oldcontrollers unchanged.
COMMAND -> Parent threeCPUtests: test_layer0_numerical_qualification_cpu_v1.py,
test_qualify_layer0_numerical_cpu_v1.py, test_layer0_packet_orchestration_cpu_v1.py.
RESULT -> PASS actual21/20bindings/old20candidate rejection; source-layout
mutation,26/33/provenance/nonce/length/EOS controls, targetall7023104B incl56
control/key owningURfreerejections,12independentpacket corruption/roster tests,
retainedparentAST/health/fd8_9/identity/finalization gates. Actualone/pairCPU
plans prepared underF08/layer0-numerical-{onecard,twocard}-prepared-v1 with
genuine source/ABI/sourceidentity. No numericalGPUframes yet.
VERDICT -> CPU/source/preparation only. Nextactual20vs21off and21off/on raw
prefix1/2/4/8 comparisons, fields/nonce/packet/owner plus pre/posthealth/NEW4
hashes. No natural/API/latency claim; fullmath false and7fields unobserved.
Runtime graph-handle retirement remains explicitlyunobserved.

## 2026-10-09 Capture21 numerical observation with failed lifetime collection

CONFIG -> Frozen21/20/raw1,2,4,8 onecard,26/33scope.
COMMAND -> Parentnumericalrun F08/layer0-numerical-onecard-v1, readonlypacket
checks and conditionaloriginalHC checker.
RESULT -> ParentFAILED724s: URIstdout/observermarkersstderr split means target
allocation/context/free chronology unproven; no posthocmerge. Childall3exit0/
removed and4fullheads/all48residual comparisons bitwise pass,12actualpackets
exact,26fieldnonce/source checks valid. Health/kernel/NEW4publisherhashes pass.
Conditionaloriginal HCweights+suppliedinput/blockoutputs32checksPASS worst
NMSE3.018986267244434e-14/maxnorm3.5706573414878165e-7, unchangedlimits.
VERDICT -> Preservefailure andvalidboundedmath, not overallqualification.
Newversion collector orderedproducer stream required; no mathflag/tolerance
changes. Raw7producerfields/fullGDN/MoE/ownstate/fullmodel remainopen.


## 2026-10-09 - Ordered numerical collector v2 checkpoint

CONFIG -> Keep frozen20/21 engines, original model shards, math flags and failed
v1 numerical evidence. Change host collection only: producer FD2 joins FD1
before engine startup, one reader saves canonical engine.combined.log.

COMMAND -> Independently run test_merged_numerical_protocol_cpu_v2.py,
test_layer0_numerical_qualification_cpu_v2.py and
test_qualify_layer0_numerical_cpu_v2.py; verify all11 source freeze hashes and
genuine one/pair v2 plan hashes; review actual launch and parent audit paths.

RESULT -> All3 CPU checks PASS, freeze/plan hashes PASS. Real OS producer fixture
preserves allocation/owner/free order; reversed order and missing free reject.
Parent audits the canonical stream. No GPU run by these CPU checks.

VERDICT -> Commit reviewed collector before new one-card GPU rerun. Failed v1
remains failed; full-model math, concurrent serving and shelf remain unqualified.


## 2026-10-09 - Producer and batch capture source checkpoint

CONFIG -> Preserve frozen21 runtime and active numerical-v2 run. New0023
producer callbacks and0024-v2 batch observer remain CPU/source only. All SYCL
ABI callers must rebuild before runtime use. Preserve earlier draft plans.

COMMAND -> Independently run producer capture mock, batch observer v2 CPU
ASan/UBSan graph controls and strict coverage collector; verify frozen hashes.
Review producer scratch boundaries, actual entry mapping and source coverage.
Run new producer manifest v2 positive and six rejection controls.

RESULT -> Producer mock PASS eleven controls; batch graph mocks PASS21 controls
and strict synthetic196-vector/six-replay coverage PASS with17 negatives.
Root found old observer header payload hash in v1 manifest; corrected immutable
v2 agrees with final source hash. Manifest positive and six negatives PASS.
Coverage is31 source values plus2 DERIVED values; raw fused hidden is unobserved.

VERDICT -> Source checkpoint ready for full pinned SDK rebuild and actual
source390, off/on, numerical and lifecycle gates. CPU fixtures establish no real
2/4/6 concurrency, full-model math, latency improvement or shelf qualification.


## 2026-10-09 - Conditional original GDN checker source

CONFIG -> Original GGUF Q8_0/F32 weights and frozen21 supplied capture fields.
Keep original FP64 operator lane and existing NMSE1e-6/max-normalized1e-4 limits.
Incoming recurrent/convolution state is supplied, not independently reconstructed.

COMMAND -> Independently run test_verify_layer0_gdn_original_v1.py; inspect
actual native convolution tap/history layout and L2 epsilon/S caller. Review
metadata/SFD/L0 nonce, field provenance, packet and original-role binding.

RESULT -> PASS18 CPU controls. Seven conditional checks per prefix cover original
Q8_0 projections, convolution shift/taps/SiLU/qk normalization, recurrent update
and gated output. Actual original-weight evaluation waits for parent terminal
and the genuine new full four-shard source identity.

VERDICT -> Source checkpoint before actual evaluation. Supplied-input operator
evidence cannot qualify own-state/full-model math, concurrency or lifecycle.


## 2026-10-09 - Capture21 ordered one-card numerical PASS

CONFIG -> Genuine numerical-v2 one-card plan, frozen20 reference and21 candidate,
original source shards, same matched native serial geometry and prefixes1/2/4/8.
Failed v1 stays preserved. Producer-merged chronological log fixes collection.

COMMAND -> qualify_layer0_numerical_v2.py to new F08/layer0-numerical-onecard-v2;
verify all finalized parent/child/lifecycle/packet/health/source receipts. After
terminal exit0 run verify_layer0_gdn_original_v1.py with actual21_on requests
and NEW full-shard identity to layer0-conditional-gdn-onecard-v2/receipt.json.

RESULT -> Parent PASS761s, all processes exit0/removed. Eight full248320 heads
and384 full residual rows bitwise equal across20/21-off andsame21 off/on.
Four26-field frames,12 native packets, chronological target allocation/context/
free, pre/post strict/compiled health, fault gate and all4 model hashes PASS.
Conditional original-weight seven-operator checks per prefix PASS28, worst
NMSE2.126246967028125e-14/max-normalized2.9579823077046734e-7. Limits unchanged.

VERDICT -> Bounded one-card capture diagnostic qualified; conditional supplied
state math is not independent own-state/full-model qualification. Actual pair,
remaining producer coverage, real2/4/6 concurrency and matched latency still
required. See docs/20261009_flashnext_capture21_numerical_v2.md for receipt hashes.


## 2026-10-09 - Combined numerical and batch full rebuild plan

CONFIG -> Fresh pinned pristine Strata source plus reviewed20/21/23/22-v2/24-v2
patch chain. Preserve active frozen21 pair numerical run and prior artifacts.
Full ABI rebuild includes8 executable targets and transitive kernel libraries.

COMMAND -> Independently reconstruct pristine files with git-show, validate all
patch SHA values, apply/check all24 patches and compare every51 final source and
22 consumed-header hash. Run combined manifest CPU positive plus6 negatives.
Review CMake definitions for required engine, grouped/native/shared/verify and
conversation parity callers.

RESULT -> All independent reconstruction hashes and six rejection controls PASS.
API fingerprint binds batch_request_identity.py. Runtime/source expected ledger
includes previously omitted prefill/fidelity headers. No SDK or GPU build yet.

VERDICT -> Immutable combined plan source checkpoint before full pinned rebuild.
Compilation, new source390/upload identity, numerical off/on, real2/4/6 capacity
and per-request coherence/lifecycle remain required. Earlier runtime receipts
cannot qualify new combined artifacts; full-model math and concurrency false.


## 2026-10-09 - Strict capture21 cross-topology comparator source

CONFIG -> Actual finalized one-card numerical-v2 receipt; pair numerical-v2
still live. Compare identical2048/64/65536 native serial geometry and exact
prefix1/2/4/8, allowing only declared32/16 split device and weight trim options.

COMMAND -> Review compare_layer0_topologies_v1.py; reconstruct all26 source
field records from actual one-card metadata/log/frame and saved hashes.
Check wrong physical card roster, unfinalized pair and4 malformed topology
controls. Initial pending-run check read child report before parent gate; move
parent finalized check before dependent file reads and rerun all controls.

RESULT -> Actual finalized one-card provenance PASS; wrong cards, pending pair
and4 topology controls reject. Comparator requires finalized parent/child,
health/fault/source/lifecycle/packet and canonical log hashes, matching source
engine and non-topology args, fullhead/all48 residual equality and26 field hashes.
No cross-topology result is claimed before pair termination.

VERDICT -> Commit comparator before actual pair comparison. Exact diagnostic
self-consistency cannot qualify independent own-state/model math, concurrency,
API behavior or clean latency.


## 2026-10-09 - Pair source failure and new cached-page discriminator

CONFIG -> Exact frozen21 pair numerical-v2,32/16 split and matched one-card
geometry. Full source identity remains independent of numeric/health gates.

COMMAND -> Pair parent terminal; preserve failed all4hash receipt. New read-only
audit_shard_cached_direct_v1.py hashes every49376141504B buffered and initialized
aligned direct view, saving differing pages. Independent initialized C readers
repeat6 page views. After evidence, target only4096B DONTNEED and rehash all4.

RESULT -> Pair FAILED604s solely full buffered shard3 hashb9351423... !=56758f40...;
other3 pass. Eight heads/384 residuals,26field frames/12packets, owner/context/free,
pre/post health, fault gate and owned normal teardown pass. Whole direct hash
matches publisher; exactly one cached page differs at39437303808, byte2796
absolute39437306604, cached0xfb/direct0xdb XOR0x20; source stat unchanged. Original
inventory maps blk.35.ffn_down_exps.weight Q5_1. Independent C readers confirm.
Same byte-within-page andbit as prior different source page; cause unknown.
Targeted reload restores original page; NEW all4 buffered hashes PASS.
CPU-only own-prefix storage estimate9controls and source/dependency pins PASS;
realprefill2/4/8 rejected, native reducers/exp unimplemented, no payload read.

VERDICT -> Preserve pair FAILED; current source recovered, no retrospective
qualification. Expanded two-page guards and full hashes mandatory. Combined
SDK rebuild may proceed without devices/modelweights; serving qualification
remains open. See docs/20261009_flashnext_capture21_pair_source_failure.md.


## 2026-10-09 - Combined SDK and full oracle compiled; V3 prototypes preserved

CONFIG -> Fresh combined24patch plan8d708bff,51 expected sources,8 targets,
pinned image39992 and clean GGML3cf03257. Pair leases for both builds; no GPU
devices/modelweights exposed. Failed pair source evidence stays FAILED.

COMMAND -> build_native_hc_engine.py --jobs4 then rebuild actual full source
upload oracle against new libraries. Independently compare every51 source and
8 binary hash; retest C1combinedV3 (14CPU controls) and numericalV3 parent/child
source controls, check all frozen source maps.

RESULT -> SDK PASS305s, full oracle compile/link PASS44s,51source/8binary identities
PASS; matching engine receipt0c58553d... andoraclef55f12e9... . CPU controls PASS.
Two-page watchdog/new7023304 allocation/33layout coverage and6Python source
fingerprints are prototype features; no genuine V3 prepare/runtime executed.
Root found legacy upload receipt could omitnewpost4hash in C1V3; strictnewV4
versions are being prepared, preserving V3. Optional source-page PFN diagnostic
could not run because noninteractive sudo authentication is unavailable; PFNs
remain unobserved and repeated pagebyte/bit is not a hardware attribution.

VERDICT -> Source/compile checkpoint only. New source390 GPU/readback, strict
upload/posthash gates, modelmath, API/concurrency/cache/fairness remain open.
See docs/20261009_flashnext_combined24_sdk_checkpoint.md and raw F09 evidence.


## 2026-10-09 - Strict upload/preparation gates and CPU RAM discriminator

CONFIG -> Preserve all frozenV2/V3 controls and failed pair source evidence.
New uploadrunnerV2, C1combinedV4 andnumericalV4 require genuine full postterminal/
posthealth four-shard identity, matching fresh SDK/oracle,6Python source hashes
and both known-page guards. Admit4 detailed unqualified research aliases only;
hotschmoe-dd stays primary. No shelf addition.

COMMAND -> Independently run uploadrunner10CPU controls, C1V4 27controls and
both numericalV4 source tests; verify source freeze maps. Compile CPU-only
host_ram_pattern_probe_v1.c;16MiB injected byte2796 XOR0x20 must fail. Run actual
64GiB eight-pattern anonymous allocation with noGPU/modelmount, capture VmRSS/
VmSwap and terminal state. After removal run NEW complete all4 source hashes.

RESULT -> All CPU gate tests/freeze maps PASS, legacy passed-upload rejection
confirmed. Injected control detected. Real64GiB patterns00/ff/55/aa/01/fe/df/20
all zero mismatches, VmRSS67110696KiB/VmSwap0, exit0/noOOM/ownedremoved. Both known
pages remain original before/after. PostRAM full all4publisher hashes PASS.
Oracle UR tracing explicitly routes tostderr with flush:info, as do its owner
markers; unchanged Docker commands retain this single ledger producer stream.

VERDICT -> Cached bit incident not reproduced by this bounded CPU allocation;
PFNs/fullhost/driver stability unqualified and source cause unknown. Strict
newsource upload GPU qualification may now start with fresh identity; no model
math, actual2/4/6/cache/fairness or shelf qualification follows from CPU gates.
See docs/20261009_flashnext_cpu_ram_discriminator.md and source plans.


## 2026-10-09 - Strict new source390 PASS and genuine C1 schema integration

CONFIG -> Combined24 actualSDK8targets/51sources, rebuiltsourceoracle and strict
uploadV2. New sourceproof parent4/numeric5 prototypes preserved, canonicalstat
C1V5/parent5 andnumeric6 use list5 withctime. Cache25 remainsCPU source only.

COMMAND -> Actual sourceuploadV2 all5cases plus ownertraces, pre/posthealth and
NEWpostterminal/posthealth full4 hash. GenuineC1V4prepare attempted with actual
newSDK/upload/oracle/runtime. After refusal, test actual tinyfile uploadproducer
to new strictC1/parent consumers withctime negative. Independently run C1V5
31controls/parent5 20controls and numericalV5/V6 tests. Run cache25 actual-body
ASan/UBSan controls and7config negatives; verify allsource freezes.

RESULT -> Actualsource390 PASS284s; eachcase387HC+3PLE, ordinary300 allocations
counted notpayload-qualified. Five cases/preposthealth/ownedterminal/twopages/
NEWall4publisher hashes PASS. C1V4refused beforepayload/output: list5stat producer
versusdict4 consumer despite equal physicalvalues. C1V5canonical5 withctime
producer-consumer PASS, physicalctime changed/restoredmtime rejects; allCPU
controls PASS. Cache25 old spare999/1999 vs corrected111/211, active checkpoint
donor, idleparking and22transfer controls+7config negatives PASS.

VERDICT -> Source390 qualified; preserve failedC1V4attempt and prototypes.
Corrected strict5workflow may perform genuine preparation next. Cache25 needs
full SDK/device/complete cache tests; publicfresh/pin and observer26 remain
required. Fullmodel math, actual2/4/6 serving and matchedlatency/shelf false.
See docs/20261009_flashnext_combined24_source_upload_strict_v2.md.


## 2026-10-09 - Actual API import failure and source28 entrypath control

CONFIG -> Genuine C1preparedV5 with compiled24/API22 six-source identity. Trace
wrapper uses package import; source22 imports helper before root path setup.

COMMAND -> Actual onecard C1parent5 launch; inspect terminal/log/stop/health.
Separate0028 sourcepatch plus newfull SDKplan; actual pinnedPython venv tests
package classidentity/module/script/trace help andoldpackage negative.

RESULT -> API C1FAILED149s pre-native: ModuleNotFoundError batch_request_identity.
Containerexit1/noOOM/removed; pre/posthealth andtwopages PASS. No actualnative
modelwork, dependentposthash UNOBSERVED. CPUfirst testusedsystemPython/Jinja
missing; corrected toactuallaunch venv, preservedoldtest. Corrected4entrypaths
PASS andoldfailure reproduces. All25patches/51 finalsourcehashes agree.

VERDICT -> Source28 fixes bothpackage/script imports without native math/ABI
orflags change. Commitnewplan before fullrebuild; preservefailedAPI/source
snapshots. Modelmath/concurrency/cache/latency/shelf remainunqualified.


## 2026-10-09 - API28 full build and genuine oracle linkage PASS

CONFIG -> Source28 packageimportfix only, newimmutablefull SDKplanv2 d496d463;
pinnedimage39992/sourcefb58/GGML3cf03257. Fresh8targets,51 sourcefiles.

COMMAND -> FullSDKbuild underpairlease/no devices/no modelmounts; compare
everyfinalsourcehash andall8 nativebinaries topreviouscompiled24. Rebuild
actual fullsourceuploadoracle againstnewlibraries. Completefresh4modelhash
scan then launch new strictuploadV2 withtwoknownpageguards andpostfull4gate.

RESULT -> SDK PASS308s; oracle compile/link PASS49s; all51 planhashesmatch.
Onlyserve/server.py sourcechanged andall8 native executables bitwiseequalold24.
Newall4publisher sourcehashes PASS. Sourceupload GPUparent8356 islive, no
source390/device/model/APIqualification result yet.

VERDICT -> Compile/source identitycheckpoint only, no native mathematical/ABI
change claimed. Newupload/actualAPI qualification required; priorfailedAPI5
retained. See docs/20261009_flashnext_api28_sdk_checkpoint.md.


## 2026-10-09 - Public batch fresh/pin source27 CPU checkpoint

CONFIG -> Separate default-off27 afterfull-chain25, strictnative2/4/6/group1
noMTP/nonpipeline fixedresidentFP16, no context/slotshrink. Compiled28baseline
remains unchanged andactualGPUupload8356 remainsinprogress.

COMMAND -> Independently run test_batch_public_prefix_cpu_v1.py against actual
session_zero/QSAzero/publicreset/native/API extracted source; verifyfreeze.
Review compositionreceipt for25/currentdraft26/27/28, retainunqualifiedscope.

RESULT -> CPUcontrols PASS: allmainstage persistentstate/indexerstaging cleared,
activepeer unchanged, malformed/partial/wrongqueue rejected. API/nativefresh
andpin0..prompt-1 fields/ceilings validated; earlier validstate evaluatedtoN
without inventingstate/tokenrewrite. Sharedreset closespreviousQSA metadata
omissions. Compatibility isCPUonly; no SDK/GPU/model payload byagent.

VERDICT -> Commitimmutable27source before integratedbuild; actualfresh-vs-hit
fullheads/all48/state/count/lifecycle and1/2/4/6 cachecoherence stillrequired.
No runtime/math/fairness/latency or shelfqualification.


## 2026-10-09 - Import28 new source390 qualification PASS

CONFIG -> NewSDK28 7baca0cc source/APIimportfix,51sources/8nativebinaries same
asprior24 exceptserver.py; newlylinked sourceoracle andactual32/16 sourceplan.
NewC1V6/parent6/numericalV7 source generation pins newSDK/six Python files.

COMMAND -> ActualstrictuploadV2 withfreshall4identity, fiveGPUcases and
newpostterminal/posthealth full4scan. Independently run C1V6 33controls/parent6
20controls and numericalV7 parent/child tests; verify sourcepins. Register4
newimport28 researchaliases with hotschmoe-dd first, no shelfentry.

RESULT -> UploadPASS277s; all5cases387HC+3PLE, ordinary300 countednotreadback.
Logicalowner/context/free/negativecontrols, terminal/removal, pre/posthealth,
twoknownsourcepages andNEWall4publisher hashes PASS. CPUcontrols PASS. No
genuineC1V6 preparation/runtime or numericV7 positive fromtheseCPUtests.

VERDICT -> Source390/lifecycle gatequalified fornewimport28generation.
GenuineC1one/pair, originalmodelmath, completeprefixcache, real2/4/6 serving
andmatchedlatency remainrequired. Priorfailedruns remainfailed; sourcecause
ofearliercachedbit remainsunknown. See docs/20261009_flashnext_api28_source_upload_qualification.md.


## 2026-10-09 - Import28 actual one-card C1 API/source qualification PASS

CONFIG -> GenuineV6 prepared e5e13288, import28 SDK7baca0cc,6 Python sources,
onecardsegmented context2048/prefill64 FP16KV/MTP0/adapt0/no borrow. Primary
hotschmoe-dd first, exactimport28 secondaryalias. Pairlease for pre/posthealth.

COMMAND -> qualify_c1_serving_combined_v6.py; inspect real/live models, six
APIresponses/token traces/repeats, native/API stop, health andNEWall4fullhash.
Independently call public validate_final_source_proof with actualpreparedmetadata.

RESULT -> Parent terminal PASS516s. APIidentity/coherence/consumption/EOS and
repeat token/LP20 checks PASS. API/nativeexit0/noOOM/removed, pre/post strict
percard+compiledP2P0health PASS. Bothsourcepagesoriginal andNEWall4publisher
hashes afterterminal/posthealth PASS. Publicstrongproof validatescrossbindings.

VERDICT -> Bounded single-cardAPI/source/lifecycle passed, notwholemodel math,
prefix/concurrency/clean latency/stability/shelf. PriorAPI5 andpairV2failures
remainfailed; sourcecauseunknown. Matchedpair actualqualification next.
See docs/20261009_flashnext_import28_c1_onecard_qualification.md.


## 2026-10-09 - Original conditional FFN checker CPU/source checkpoint

CONFIG -> Newimmutablechecker for source23 all33fields, originalselectedQ8_0/
Q4_K/Q5_1 consumers plus explicitF32toBF16-RNE router/sharedgate seams. Use
suppliedquantizedinputs/HQ/routerweights; rawhidden andindividualdown outputs
remainUNOBSERVED. PreservefrozenFP64/GDN/packet/reference sources.

COMMAND -> Independently run test_verify_layer0_ffn_original_v1.py andvalidate
allimplementation/dependency/consumedsource hashes. Require actualfuture33
field/source/nonce/rank/tier/liveowner bindings before anyweightverification.

RESULT -> PASS19 CPUcontrols. 27conditional originalweight checks perprefix
areprepared, notexecutedonactualweights. Selectedaffine minima use d*codesum,
storedsumunused onlyforadmittedformats. Aggregatechecks canmaskdown cancellation;
noindividualdownclaim, nativeFMA/exp/reducer/topktiesunsupported. Existing
NMSE1e-6/max-normalized1e-4 limitsunchanged. Zeroactualweights/GPUbyagent.

VERDICT -> Sourcecheckpointbeforeactualconditionalverification. Ownstate/full
layer/model/math/lifecycle/concurrency remainunqualified; actual33capture
qualification andtrueprefill-ownedhistory stillrequired.


## 2026-10-09 - Import28 actual paired C1 API/source qualification PASS

CONFIG -> GenuineC1V6 paired32/16/card0+1 context8192/prefill128 FP16KV/MTP0,
exactnewSDK/source390/import28 sixPython fingerprint/primaryhotschmoe-dd.
Matchedactualonecardqualification/sourceproof is prerequisite.

COMMAND -> ParentV6 --prepared paired --one-card-receipt actualonequalification;
inspectAPI IDs, sixcoherentresponses/repeats/consumption/EOS andownedstop.
Independently validate publicfinalsourceproof afterfresh all4hashes/health.

RESULT -> ParentPASS327s; allAPI/trace/coherence/repeat gatesPASS, native/API
exit0/noOOM/removed, pre/posthealth andNEWpostterminal/posthealth all4publisher
hashesPASS. Publicstrongproof verifiesactualartifact crossbindings.

VERDICT -> BoundedpairedAPI/source/lifecycle only, notmodelmath/completecache/
concurrency/fairness/clean latency/shelf. One/pairC1geometrydiffers; no speedclaim.
Oldfailedrunsstayfailed. Numericcomparison harnessreference selectorfix next.
See docs/20261009_flashnext_import28_c1_pair_qualification.md.


## 2026-10-09 - Reference21 current-source guarded rerun preparation

CONFIG -> PreservefailedoldpairV2 andfrozen21 SDK/childV2/nativeflags; newparent3
adds bothknownpageguards only. ActualnewAPIpair6 currentpostfull4 identity binds
genuine newreferenceplan, old20qualified numericalcontrol retained.

COMMAND -> GenuineV2prepare withactualoldengine/upload/C1/reference20 andcurrent
sourceidentity. Run test_numerical_reference_parent_v3_cpu.py /py_compile.

RESULT -> PreparationPASS; actualsource_watch secondpagewrong rejects/preserves
bothsamereadviews andstopsbeforelegacycheck. Old lifecycle/fullhash helperASTs
identical; no GPUexecution ornewreferencepositive yet.

VERDICT -> Commitparentbeforeactualpairedreference rerun; fullmatched heads/
48residuals/packet/owner/health/teardown/NEWall4hash gates mandatory. Candidate
comparison must notusefailedoldreference orwrongnewmanifest validator.


## 2026-10-09 - Guardedreference PASS and numerical8/observer26/prefill source

CONFIG -> Frozen21 control/frozen20 baseline, matchedpair nativegeometry with
newtwo-page parent3. Newcandidate33captures needstrictnewC1validator; old21
references explicitlyusefrozenoldvalidator/fullqualified chain, notnewmanifest.

COMMAND -> Actualguardedpairedreference withfreshsource; inspectactualparent/
child/lifecycle/packet/health/full4. StrictcrossTopology comparison toqualified
onecard. IndependentCPUtests V8 threecommands, observer26 actualheaders/mock/
collector/graphmap tests, prefillownedstorage12controls andsourcefreeze checks.

RESULT -> Actualreference PASS433s/NEWall4publisher hashes. CrossTopology full
heads/all48 and104fields bitwiseequal; priorfailedpairstaysfailed. CPUtests PASS:
actualold/newmetadata/wrongvalidator/failurelanes;26 synthetic294vector22negative
producer/ARMloss/solo/warmmap controls;12trueprefillstorage seam/history controls.
GenuineV8onecardpreparation passed usingnewC1/source andactualqualified reference.

VERDICT -> Commit beforeactualnew33numerics. Observer26/cache25/public27 require
integratedfullSDK/actual2/4/6/state/cache gates. Prefillreference owns history but
exactdevicearithmetic/runtimebranch/fullmath unqualified; no actualweightsread
byagents. Modelmath/concurrency/latency/shelf false.


## 2026-10-09 - Integrated fullcache/fresh/pin/migration SDK source plan

CONFIG -> Actual28baseline plusimmutable25completechains/26migrationobserver/
27publicfreshpin, allfeatureflagsdefaultOFF. Originalmodelbytes andmath/
sampling settings retained; reset/cache/diagnosticpaths separatelyscoped.

COMMAND -> Independently run test_integrated_batch_prefix_source_cpu_v1.py;
reconstructpinned pristinefb58 andapplyall28patches incheckedorder, compare
all60 finalsource/24headers andsixPython-source closure/eightABI recipes.

RESULT -> Independentpristine reconstructionPASS. All source/header hashes
match; no missinghelper/fallback slot/context/budget guards. No SDK/GPU/model
payload byagent or thisCPUtest. PlanSHA b4a7d4baa7ae654b3888354240cc541290a218caa82c3b62d7bcf74f1ec5bd42.

VERDICT -> Immutableintegrated sourcecheckpointbeforefreshSDK. Existing28
source390/C1/num proofs cannotqualifynew60-source artifacts. Actualactivated
fullcache/fresh/pin/cancellation/migration/private1/2/4/6/rawmath/fairness/latency
gates remainrequired. Parent ownsGPU/lifecycle; numeric8onecurrentlylive.


## 2026-10-09 - Actual33field onecard numerical/conditionalFFN PASS

CONFIG -> Newimport28 SDK/strongC1/source390, frozen21 reference, matchednative
2048/64/65536 geometry andfour freshprefix1/2/4/8. 31source+2DERIVEDobservations.

COMMAND -> GenuineV8prep andparent; afterterminal originalconditionalFFN
checker withactual33requestframes andNEWpostfull4sourceidentity.

RESULT -> ParentPASS813s; eightfullheads/384residualcomparisons bitwiseequal,
all3processesexit0/removed. Four33frames/rank/tier/liveblob/nonce,20packetarray
pairs/DERIVEDgates, entire7023304B owner/context/free, pre/posthealth andNEWall4
publisherhashes PASS. ConditionaloriginalFFN108checks PASS, worstNMSE
3.1926730082874146e-14/maxnormalized3.5706573414878165e-7, thresholdsunchanged.

VERDICT -> Scopedonecardcapture/equivalence/conditionalconsumer proof only;
rawhidden/individualdown/ownstate/fullmodelmath/cache/concurrency/latency/shelf
unqualified. GenuineV8pairprepPASS; pairexecutionnext withmatchedone receipt.
See docs/20261009_flashnext_import28_numerical_onecard_v8.md.


## 2026-10-09 - Integrated serving gates and owned local QSA source

CONFIG -> Newintegrated60/24headers/eighttargets/sixPython C1controller/parent7
withcanonical5stats/uploadV2/finalpost4/twopages/strongsourceproof, default
batch0/cacheflagsOFF. IndependentQSA lane suppliedlayerinput butownedhistory.

COMMAND -> Independently run combinedV7 CPU61controls andQSA15synthetic tests;
verifysource/headerinventory/frozenreference/implementation SHA receipts.

RESULT -> CPUcontrols PASS. Newserving generation rejectsoldSDK/proof before
modelscan; no genuinepreparation/runtime. QSA ownFP16KV/4cellpool/tail/dead/
spare/blkpos/RoPE/causal2051selection/headdivision andcheckpoint replay/order
negativecontrols pass;64/65 history boundary isnotpool64. Existingsource
resetidxmetadata omission explicitlyretained, notmasked. Noactualweights/
GPU/SDKbyagents. No nativeintrinsic/topk/FMA exactness ornewtolerance inferred.

VERDICT -> Sourcecheckpoint only. IntegratedfreshSDK/source390/C1/2/4/6 cache/
freshpin/migration andfulloriginalmodelmath remainrequired. Actualpaired33
numerical8 GPU25190currentlylive; no prooftransfer or shelfpromotion.


## 2026-10-09 - Paired33 numerical and cross-topology PASS

CONFIG -> Import28 pinned SDK/C1/source390; matched serial1/2/4/8 context2048
prefill64, pair32/16 split,31source+2DERIVED fields.

COMMAND -> Actualpaired numericalV8 parent; originalconditionalFFN checker;
new read-only all33 cross-topology comparator andfour rejection controls.

RESULT -> PairPASS658s, eightheads/384residual comparisons bitwise, allthree
processesexit0/removed,7023304B logicalowner/free, packets/sourcebindings and
pre/posthealth plusNEWallfour fullpublisherhashesPASS. ConditionalFFN108PASS
worstNMSE3.1926730082874146e-14/maxnormalized3.5706573414878165e-7.
Cross-topology fourheads/192residuals/132fields bitwisePASS. Initialobsolete
armdirectory invocationfailedbeforeoutput; correctedsource/newreceiptPASS.

VERDICT -> Scopedserialequivalence/conditionalconsumers; fullownedmodelmath/
cache/concurrency/latency/shelf unqualified. FrozenfailedpairremainsFAILED.
See docs/20261009_flashnext_import28_numerical_pair_v8.md.


## 2026-10-09 - Default-off slot ownership source and fresh ABI build

CONFIG -> Frozen29 patch on integrated cache/migration/public fresh/pin source;
newplan94aef305dbc6a31f023afa55573d52a9ce6af9443cfc0182eb7c324255159094.
No source/model/image replacement; inherited pair lease for isolatedSDK build.

COMMAND -> Independently run actualarena ASan/UBSan hostmock lifetimecontrols
andpristinefb58 reconstruction ofall29 patches/60source/24headers/sixPython/
eighttargets. Start build_native_hc_engine.py withnewplan/exactcleanGGML.

RESULT -> Both CPUcontrol suites PASS. Defaultoff zeroextraGPU alloc/copy/
wait/nativecontextquery; enabled authoritativePID/stage/slot/device/allocation
generation/context/pointer/extent plusactualgraphretirement/release markers.
FreshSDK build currentlylive; no runtimeGPU/model/source390 qualificationyet.

VERDICT -> Source checkpoint only; no old28 qualificationtransfer. NewSDK,
source390, genuineC1 andactual2/4/6 slot-owner/completecache/migration/API/full
math/fairness remainrequired. Unrelated userchanges preserved.


## 2026-10-09 - Integrated29 serving admission and original PLE source

CONFIG -> New C1controller/parent8 strict integrated29 SDK admission with60
source/24headers/eightABI/sixPython, batch0/cacheobserverOFF. PLE conditional
ownedoriginalIQ4NLlookup/F32history/keyvalueQ8 directF32 source_exact lane.

COMMAND -> Independent C1generation8 unittest64 andPLE15 source/synthetic
controls, frozenfile pins/ASCII checks. No modelpayload/GPUbyagents.

RESULT -> Both suitesPASS. New controller refusesoldSDK/newmissing29/source
package mutations before modelaccess andrequiresnew390/full4/proof8. PLE owns
uint64hash16rows/lasttwo/9rowhistory/dilation3 andcheckpointreplay with source/
order/input negatives; tablehas90paddingrows, convoriginalF32 despite oldlabels.

VERDICT -> CPU/sourcecheckpoint; genuineC1/newSDK/upload/runtime stillpending.
PLE suppliedlayerresidual, nativecontracts/GPUcheckpoint/full48math unqualified.
No newtolerance or shelf/speedclaim. SDKparent79123 stilllive.


## 2026-10-09 - Actual integrated29 fresh SDK and oracle link PASS

CONFIG -> Pinned integrated29 plan94aef305, newsource60/24headers/sixPython/
eighttargets, unchanged39992image/fb58/3cf03257, inheritedpairlease noGPUdevice.

COMMAND -> Fullnativeengine buildjobs4 thenfreshsource390oracle link fullv6.

RESULT -> SDKPASS337s receiptbd17b41b0074b816ab88d3804af78727480291f96d2510e5759b9788091b3c2d;
eightactualABI targets/external/snapshotpass. C1generation8 actualsourcegatePASS.
OraclefreshlinkPASS46s; librarypins unchanged. Newstrictsourceupload running.

VERDICT -> Compiler/sourcecheckpoint, no modelruntime/slotcache qualification.
See docs/20261009_flashnext_integrated29_sdk_checkpoint.md.


## 2026-10-09 - Integrated29 actual whole390 strict upload PASS

CONFIG -> Fresh29 SDK/oracle, fixedmodel/images, bothknownpageguard/newpost4.

COMMAND -> SourceuploadV2 fivecases innewF12, inheritedpairlease strict/compiled
P2P0 health andactualownedterminal thenindependentfull4hash.

RESULT -> PASS276s; caseHC387+PLE3 payloadcoverage, ordinary300 accounting;
exit0/removed/context/URlogicalfree, preposthealth/knownpages/newall4 PASS.
ReceiptSHA9f4f52a3b057ffcd0728fafc4ac4ef89a11e063728b6b7e577c6d29bbc72e0fc.
NewgenuineC1generation8 onecardpreparation running.

VERDICT -> Scopedsource/lifecycleproof; ordinarypayload/nativefullmath/slots/
cache/concurrency/latency/shelf remainunqualified. See integrated29sourceupload
qualification doc. Earlierfailedruns unchanged.


## 2026-10-09 - Actual33 conditionaloriginalHC/GDN one/pair PASS

CONFIG -> Explicit33 admission/frozenHC/GDN scalar consumers, completedV8
one/pair31source2DERIVED, original13roles/no guessed nativecontract.

COMMAND -> New20CPU controls andactualone/pair checkers afternew390/CPUprep
terminal; noGPU arithmeticchecker. Thennew29 C18 APIone parent41425started.

RESULT -> PASS60each =32HC+28GDN/fourprefixes, worstNMSE3.018986267244434e-14
maxnormalized3.5706573414878165e-7, limits1e-6/1e-4 unchanged. All33source
bindings/actualproofchain andtwo nativepacketcontracts admitteddirectly.

VERDICT -> Conditionalcapturedincomingstate/activation consumers only. No
filtered26admission/ownprefix/upstream/full48/lifetime/cache/speed inference.
See docs/20261009_flashnext_original33_conditional_hc_gdn.md.


## 2026-10-09 - Integrated29 unsegmented startup allocation refusal

CONFIG -> Rootselectedone-card unseg profile; priorqualifiedC16 used1GiB
segmented/adapt0/no-borrow control. Same2048/64 context/prefill/modelprecision.

COMMAND -> GenuineC18 oneprep/API parent41425 innewF12directory.

RESULT -> FAILED154s pre-ready:49266MiB pinnedmirror allocationrefused,16484
expertsuncovered -> nativeNO_HOST coveragegate refused. Native/APIexit1/noOOM/
removed; preposthealthPASS. Finalcomplete4afterfailedserve UNOBSERVED.

VERDICT -> Preservefailedrun; rootprofile mistake correctedwithfreshsegmented
preparation3335/directory, no coveragebypass/math/speedclaim. See integrated29
unsegmentedlaunchrefusal doc. No evidenceofhardwarecrash/sourcecorruptionhere.


## 2026-10-09 - Full48 dependency/capture plan and numerical29 source gates

CONFIG -> Original1224 tensor inventory/all7formats,36GDN/12QSA/PLE schedule;
NEWnumericalV9 new29 SDK/C18 finalproof sourceadmission, old21 explicitreference.

COMMAND -> Full48 syntheticownershipfixture5CPU tests; eachthree executable
V9CPU scripts. Initial unittestdiscovery reportedzero (customscript controls),
so explicitCLI suites runPASS. Sourceplan commandlistcorrectedbeforecommit.

RESULT -> Syntheticfull48 schedule/replay/ownership PASS atdims4/vocab16 only.
Originaldownexceptions/Q5K layer2 recorded. Nativefull48contract gaps explicit;
nextprefix8 all48 three residualviews47,185,920B route/source bound planned.
V9CPU actualnew29 SDKmetadata/source60/24headers/eight/six admits; old28/oldC1
proof rejects before modelaccess, frozen21 dispatch preserved. No GPU/modelreads.

VERDICT -> Sourceplan/controls only; no faithfuloriginalownedfullmodel composition
ornewactualnumerical proof. C18segmentedone parent97548 live. New bounded
allprefill-row capture/sourceadapter work assigned; concurrentharness inprogress.


## 2026-10-09 - Integrated29 segmented APIone actualPASS

CONFIG -> GenuineC18/source390/new29SDK, one-card-segmented2048/64,1GiB mirror
segments/adapt0/no-borrow, batch0/featuresOFF, exactprimaryhotschmoe-dd/alias.

COMMAND -> Freshsegmentedprepare andparent97548; independentlypublicproof8.

RESULT -> PASS490s sixboundedresponses/actualconsumption/EOS/repeats/LP20/API
identity/oneengine. API/nativecleanexit0/noOOM/removed/preposthealth/kernel/
twopages/NEWallfourpublisherhashesPASS. Independentproof8PASS. Qualification
SHAd8f43a305f188ebaa340be3a7c73799aa76d3d803bc189977eb2f2a420f307d1.

VERDICT -> BoundedAPI/coherence/lifecycle/sourceproof; fullmath/cache/2/4/6/
latency/shelf stillunqualified. Failedunsegmentedretained. Pairseg genuineprep
99606live; context8192/128 notmatchedspeedcontrol. See integrated29C1one doc.


## 2026-10-09 - Integrated29 segmented APIpair actualPASS

CONFIG -> GenuineC18/source390/new29SDK,32/16 split8192/128/1GiBmirrors,
completednewonecardcontrol, batch0/featuresOFF/exactprimaryhotschmoe-dd alias.

COMMAND -> Genuinepairprepare andparent77204; independentpublicsourceproof8.

RESULT -> PASS383s sixboundedresponses/actualconsumption/EOS/repeats/LP20/
APIidentity. Native/APIcleanexit0/noOOM/removed/health/kernel/twopages andNEW
allfourpublisherhashesPASS. Independentproof8PASS; qualification
SHA324d88f4fc4d50617af55957cde88100c2fe10794c0e1f056e256b815e29abde.

VERDICT -> BoundedAPI/coherence/lifecycle/sourceproof. 8192/128 vsone2048/64
isnotmatchedspeedcontrol. Nativefullmath/completecache/concurrency/fairness/
latency/shelf required. NewV9oneprepare58354 nowlive; failedunseg retained.


## 2026-10-09 - Integrated29 actualone numericalV9 PASS

CONFIG -> Fresh29SDK/source390/completedC18, explicitfrozen21 reference;
serial1/2/4/8/context2048/prefill64,31source+2DERIVED, newcache/slotflagsOFF.

COMMAND -> GenuineV9 preparation thenparent2681, actualcard0/pairhealthlease.

RESULT -> PASS753s eightfullheads/384residual comparisons bitwise, allthree
exit0/removed, four33frames/20packet-array pairs/7023304B logicalowner/free
andpreposthealth/kernel/knownpages/NEWallfourpublisherhashesPASS.

VERDICT -> Boundedserialsource/numerical/lifecycle proof; rawhidden/ownedfullmath/
cache/concurrency/latency/shelf incomplete. Neworiginalconditional consumersnot
yetexecuted. Pairedparent7497 nowlive usingthisactualone childreport.
See docs/20261009_flashnext_integrated29_numerical_onecard_v9.md.


## 2026-10-09 - Serving mirror ownership source and preserved CPU receipts

CONFIG -> NewdefaultOFF31 source onfrozen29; actualserving mirror table/host
segments/contiguous payload owners, stage/device/context/pointer/generation.

COMMAND -> Independent actualheader/factory ASAN/UBSAN sourcebody controls50130;
new explicitnonoverwriting --output controls14484 andexistingoutput rejection.
Verifyfinalsourceplan/frozenpatch/parser/recipe andinitialreview snapshots.

RESULT -> PASS CPUactualsource/lifetime/rollback/quota/defaultOFF0nativequeries,
exactcallerdeclared verifier roster includingmissingone ofmultiplepairs rejection.
Twentytrace negatives; owncontext actualUR frees andorderedgraph/source retirement.
RootnewoutputSHAced515e5ecdcee009ecd2f77fcda170e5bd11ebbf2425bbec1bfc1bd73aafe7d
preservedbyteexact onrejectedrerun. EarliermutableCPUtest regeneratedanuncommitted
receipt; finalmanifest transparentlyreboundto retainedreceipt, initialreview intact.

VERDICT -> Source/CPUcheckpoint only. No servingmirror/runtime/source390proof
transfer/wholeengine/physicalreclamation claim. Need exact30+31 composition,
freshABI andactualcontext/owner/free qualification. Pairednumerical9 parent7497
remainslive; noGPU/modelpayload reads byresearchagents.


## 2026-10-09 - Paired29 numerical/originalconsumers andnextsourceclosure

CONFIG -> Qualified29/C18/390, paired32/16 numericalV9 withcompletedone;
NEW30+31 source62/header26/eight/six; V2concurrency source-onlyprototype.

COMMAND -> Actualpairedparent7497; afterterminal conditionaloriginalcheckers
foreachnewone/pair andV3cross-topology/fournegatives. Independently sevenV2CPU
suites and30capture23/ownedFFN6; pristineall31patchapplication/full62hashes.

RESULT -> PairPASS498s, allthree exit0/removed/bitwiseeightheads384residuals/
packets/ownerfree/health/knownpages/NEWall4PASS. Crossfourheads192res132fields
bitwisePASS; HC/GDN60+FFN108EACH originalconditionalPASS, unchangedthresholds.
V2CPU7suitesPASSincludingrawmutation/seals/collector294+49/aliaschecks. Source
review findsreread_to=-1 whennativeCKPT_REREADoff vsV2zeroassumption; no V2
actualGPUpositive attempted. V3counter-admission correction/specs assigned.
30CPU23+FFN6/pristine62/26PASS; combinednewSDK66369 nowrunningleased,noDRI.

VERDICT -> Bounded29serial/conditionalproof plusCPUprototype/newsourcecheckpoint.
V2 knowncounterincompatibility precludesactualpositive; preservefrozenprototype.
30/31 runtime/fullmath/completecache/concurrency/latency/shelf unqualified; all
proof generations remainexplicit. No old29 prooftransfer or tolerancechange.


## 2026-10-09 - New62source C1generation9 admission READY

CONFIG -> Combined30+31 plan82951242, source62/header26/eightfreshABI/sixPython
andall31 orderedpatches. Newobserver/cache flagsOFF, unchangedsegprofiles.

COMMAND -> Independent combinedC19 unittest67; verifyfrozenfiles/sourceplans.

RESULT -> PASS67 CPU admission/tinyfile/sourceproof controls. PriorC18bytes
unchanged. NewSDK/source390/uploadV2/current4/genuinefinalproof9 mandatory;
no actualgenuineprep/GPU/modelweights byagent. FreshSDK66369 stilllive.

VERDICT -> Sourcecheckpointonly. Fullmath/30capture/31mirror lifecycle/newSDK
serving/concurrency/cache/latency remainrequired. V3source29 counterfix/specs
beingprepared; source29 uses-1 no-reread sentinel ratherthanV2assumedzero.


## 2026-10-09 - Source33 resume, fresh SDK and isolated runtime PASS

CONFIG -> Original Flash-Next objective unchanged; runtime/Git/network writes
allowed again. Fixed7.1/26.22/LevelZero1.28; exact33 source/63files/27headers/
eightfreshABI/sixPython. Unrelated dirty changes preserved.

COMMAND -> Fresh pair-leased SDK29238 with pinned local ggml; isolated C111
Python runtime91133; actual C111 SDK/source admission; independent CPU controls.

RESULT -> SDKPASS312s/eight executables/clean source/recipe isolation; actual
C111 source gatePASS. Python runtimePASS, GPU libraries unchanged. CPU147+75
tests and58 batch negativesPASS. New experimental aliases retain hotschmoe-dd.
Two extra Docker mock controls aborted/terminal/no receipt or surviving owned
container; no evidence claimed. New standalone390 oracle54593 linking live.

VERDICT -> Native compilation/runtime source checkpoint, not correctedmodel
GPU/math/cache/concurrency/latency/shelf qualification. No old29/31 proof
transfer. Next fresh390/uploadV2/C111 serving then actual source33 PLE input
proof and original48 requalification. See docs/20261009_flashnext_source33_resume.md.


## 2026-10-09 - Source33 new whole390 upload PASS and C111 onecard live

CONFIG -> Fresh source33 SDK0b32d570/new oraclee2edeb20, originalmodel/stack
unchanged; allfive whole390cases and own terminal/posthealth/source4 required.

COMMAND -> Oracle54593/link43s; pair-leased upload47722; genuine onecard
segmented C111prepare58899 then parent62244; new readonly originalprefix1
P30V2 explorer/admission CPU14 and full22file closure.

RESULT -> UploadPASS274s/fivecases/ownedterminal/strict+compiledposthealth/
NEWallfourpublisherhashes, receiptbe7661c6e7e95a5e6b82c54b6a5bbe3c70c3e1f5e7241aeeac4159c7f0009e4e.
Genuine C111onecard preparationPASS. Parent62244 nowlive under pairlease.
NewexplorerCPU14PASS;144P30phase+head comparison uses originalweights/IDs
only, no captured state as mathinput. Actual newexplorer admission/run pending.

VERDICT -> Newcompiler/upload/source checkpoint, not correctedmodel math
qualification. C111/runtime/P30V2/actualoriginalinput/nativefidelity then
cache/concurrency/latency/shelf stillrequired. Source32-only batchV4 needs
new source33/C111 V5 admission; no silent priorproof transfer.
See docs/20261009_flashnext_source33_resume.md.


## 2026-10-09 - Corrected source33 onecard C111 PASS; P30V2 live

CONFIG -> New source33/native SDK and exact C111onecard-segmented2048/64,
all new cache/observer flagsOFF; originalmodel/stack/clientname unchanged.

COMMAND -> Pair-owned C111parent62244, fresh terminal/posthealth/full4
then actual publicproof validation. Prepare4203 and launch P30V2parent12328.
V5 concurrency source-only port on source33/C111 with17CPU tests,37hash closure
and root actual onecard baseline admission.

RESULT -> C111PASS508s/APIidentity/coherence/tokenconsumption/repeats/exit0/
removed/strict+compiledposthealth/NEWfourpublisherhashes. QualificationSHA
80d33583efac5ed1562ec0dbf58316ecf009b8ae61c8224c34f399075708f288;
parentSHA05aec7d374fa2b7a1dcceece1b020886b8b18abb2fff4f0219ed4d10f60c1696.
P30V2onecard12328 live with hostinput33 ONbotharms/P30onlyOFF-ON toggle.
V5CPU17/37closurePASS and actual SDK/C111admissionPASS; no batch runtime.
V5 rejects EAGER presence0 (native getenv presence enableseager), keeps
input33/P30OFF; source32/V4 evidence unchanged.12experimental aliases added.

VERDICT -> Bounded correctedonecard service/lifecycle/sourceproof, not native
originalmodel math or speed. P30 actualcurrentinput/original48 then corrected
pair/cache/concurrency/fairness/latency/shelf remainrequired.
See docs/20261009_flashnext_source33_resume.md.


## 2026-10-09 - P30V2 real observer failure; source34 SDK PASS

CONFIG -> Source33/C111onecard baseline qualified; actual P30V2/input33 diagnostic
with originalmodel/source/stack fixed. New34 patch limits publication to final
accepted prompt verifier window; source32 gathering/math unchanged.

COMMAND -> Actual P30parent12328; preserved terminal/posthealth/source4 onerror.
New34 pristine/CPU11,C112CPU85,P30V3CPU30,originalexplorerV3CPU15. Fresh leased
SDK9860 then source390 oracle90862; newupload43045 currentlylive.

RESULT -> P30V2FAIL551s: prefix2 earlierpos0 and finalpos1 bothpublished same
requestordinal, duplicatecurrentordinal exception/nativeexit139/noOOM/removed.
Posthealth/kernelgate/NEWallfourpublisherhashesPASS; no P30/inputqualification.
Source34SDKPASS309s/eightABI, actualC11263/27/34sourcegatePASS, receipt
d2af73f0062a44a1c2809870071f5d28e2e77d310e07e0ae2c16c40776fce30a.
Newlinked390oraclePASS47s. CPU141/sourceclosuresPASS;4C112aliasesregistered.
Actual source34upload43045 live under pairlease, f15-source34-20261009.

VERDICT -> Real diagnosticfailure localized/newcorrected observer compiled,
not fullmodelqualification. Newupload/C112/P30V3 actualreplay/originalIQ4
input/owned48/futurepair/cache/concurrency/latency/shelf remainrequired.
No old33compiled/upload/runtimeproof transfer to34. No reset needed on
healthy singlecardfailure; no driver/kernel/sourceweight change made.
See docs/20261009_flashnext_source33_multiverifier_observer_failure_and34_fix.md.


## 2026-10-09 - Source34 one/pair C112 PASS, rough decode and source35 build

CONFIG -> Exact source34/Unsloth UD-Q4_K_XL, batch0/MTP0, FP16 KV,
segmented static experts, observers/cacheOFF. One2048/64/PLE65536; pair
8192/128/PLE1048576/split32-16. Geometry differs; no scaling-speed comparison.

COMMAND -> Source34upload43045; C112one48168/pair3339 with new source4/
health/ownedteardown; actual V12 publicproof validators. Read engine timing
logs. Source-audit normal short_read64/P30 earlierrows; new35 SDK17447 and
new source390link52494, upload6067 nowlive. Independent newCPU183/closures.

RESULT -> UploadPASS276s/allfive/newsource4. C112onePASS479s/pairPASS320s,
identity/coherence/repeats/consumption/cleanexit0/removed/posthealth/full4PASS.
Pair short warm2-9token screens13.1-18.4 tok/s; final repeats16.3/18.4.
One warm5.9-10.7/final9.8/10.7; earlier C1118.1-10.7. No clean/100token metric.
P30source34 guaranteed missing earlierprompt_verify rows; new35 preserves
normalT1/T2 math and adds actualrow/group observer graphs/schema2 nonce/collector.
SDK35PASS302s/eightABI/actualC11363/27/35 admissionPASS, engineSHA
7e3e4c1e61bcee7bc0ea1a15c0b2d8347bfdbd0c8f9fb16296476625bff58c12.
Newsource390linkPASS46s; upload6067 live. CPU183/sourceclosuresPASS.
Q5K200tiny algebra/projector+5existing controls show no wrong storedS/dFloat
reference term; no kernel/reference replacement justified.

VERDICT -> Real paired boundedservice/source proof and rough Strata decode
readings, not fullfidelity or matched performance. New35/C113/P30V4 original
inputs/owned48 then completecache/concurrency/fairness/profiling/latency/shelf
required. C112/P30V3 remain frozen; no new math or sourceweight change.
See docs/20261009_flashnext_strata_decode_readings_and_prompt_capture35.md.


## 2026-10-10 - Source35 original PLE/fullprefix PASS and FFN seam evidence

CONFIG -> Same exact source35/C113 SDK/model, one2048/64/PLE65536, P30V4
OFF/ON only with SFD/input33 bothON; original ownzero prefix1, oneBLASthread.
Separate conditional FFN nativeattention diagnostic, never a fullown reference.

COMMAND -> Actualsource35upload6067/C113one32077/P30one23058/original22283/
conditional45761. NewC113pair36108 then P30pair25718 nowlive. NewseamCPU16.

RESULT -> UploadPASS273s/C113onePASS434s/P30onePASS556s, allsource/identity/
health/ownedterminal gatesPASS.11normalrouteframes,192P30-SFD/196OFF-ON
vectors bitwise; original stagedPLE4framesEACHarm BITWISE independentIQ4/hash.
ActualONexplicitpeak99,297,312B andowningfreesPASS. Originalown48compute
complete/noerrors/newpostCPUfull4PASS. L1FFNnmse2.92457e-14,L2FFN1.59384e-14
(oldhugePLE/layer2counterfactualgapgone); laterFFNgrowth/headnmse8.40917e-5/
max.002253 remains, no numericalPASS/tolerance assigned.
Seam8cases/newpostCPUfull4PASS; nativeattention conditional FFN4/6/10
nmse1.2882e-14/1.06207e-15/4.41218e-17, ownreplay8casesBITWISE. Independently
derivedmixedpacketSHAs differat4/6/10, sameat3; actualGPUmixed/packetsunobserved.
C113pairPASS345s/APIcoherence/identity/exit0/removed/posthealth/newsource4;
publicfinalproofPASS. Matchedreducedgeometry P30pair25718 liveunderpairlease.

VERDICT -> ActualoriginalPLE andfullprefixobservation nowqualified; major
semantic omissionremoved. Conditionalagreement supports propagatedquantization
seams, not a Q5Kstoredsum fix or fullmodelPASS. Remainingnumeric/quality/
route/state/cache/concurrency/latency/shelf gates retain originalscope.
See docs/20261010_flashnext_source35_original_ple_and_ffn_seams.md.


### 2026-10-10 - Seam case-count clarification

The preceding source35 seam entry has eight total cases: four conditional
native-attention cases and four owned-attention replays. The four owned
replays are bitwise exact. The phrase "ownreplay8cases" was a count typo;
all saved metrics, scope restrictions and source proofs remain unchanged.


## 2026-10-10 - Paired source35 P30 and matched topology equality

CONFIG -> Exact source35/C113/P30V4 onecard and pair split32/16;
matched CTX2048/prefill64/PLE65536/FP16KV, accepted prefixes1/2/4/8.
Same SDK/image/math flags; only topology/split/trim/affinity differs.

COMMAND -> Paired P30V4 parent25718; independent public finalized_binding;
frozen cross_topology_prefix35_v1.py session91448; fresh CPU14 controls
and12-file dependency closure. Earlier session11445 report retained but
non-authoritative because helper docstring/sourceplan finalized during run.

RESULT -> Paired PASS450s,22frames, original input33/wholeprefix/lastrow/
logical lifecycle/owned teardown/posthealth/kernel/newfull4 PASS. ON peak
99,297,344B/OFF4,925,440B. Frozen comparison completes with errors[]:
all4 fullheads and2160 absolute phase rowvectors BITWISE equal onevs pair.
Report SHA eb9b244df7b85dc079fa7c88df5a8b3b7baf11eb181dc992aeaaf311fd5db743.
Report: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/cross-topology-prefix35-v1-frozen.json

VERDICT -> Actual matched short-prefix native topology control passes.
No tolerance, originalmath or speed claim. Independent route-aware larger
prefix reference, complete statecache, concurrent fairness/profiling/latency
and verified shelf remain required. Original full campaign stays active.


## 2026-10-10 - Saved original/native prefix1 distribution diagnostic

CONFIG -> Completed independent original48 prefix1V4 report SHA
c4cc4ca6caae9e67ad57867adf9f65dc10f8cea03474c984e7fb78d082e2a244;
its bound original/native full248320 F32 heads, no new weight or GPU access.

COMMAND -> original_head_distribution_v1.py with exact report SHA and new
original-head-distribution-v1.json; CPU5 analytic/shift/extreme/negative controls.

RESULT -> Same argmax and10/10 top-token overlap. KL(original||native)
8.552987271077312e-7; total variation0.0006415135246005365. Actual saved
head SHA/extent/finite checks and completed original provenance pass.
Report SHA ae71beb8fa6a2eedffc19cd97f26a111a2f2d3bc1c03488e24a3e9078fdf9223.

VERDICT -> Diagnostic probability difference on one short prefix only.
No tolerance, quality or fullmodelmath PASS. Longer independent prefixes,
statecache/concurrent serving and matched latency remain required.


## 2026-10-10 - Source35 batch V6 preparation and genuine admission

CONFIG -> NEW immutable V6 port of frozen batchV5 to source35/C113.
Source32/33/34/35 required, normal graph/EAGER absent including0, P30/input33
OFF, unchanged case2/4/6 corpora/cancel/survivor and private all48/head gates.
Requested1 retains separately qualified serial/control scope.

COMMAND -> test_batch_current_source35_cpu_v6.py;38-file frozen dependency
closure; genuine_baseline on actual C113 onecard and pair, session33131.

RESULT -> CPU20/closure PASS. Genuine one[0]/pair[0,1] admission PASS through
exact pinned C113 public final validator and actual fresh source/ELF/all8 SDK
checks. Both engine receipt SHA
7e3e4c1e61bcee7bc0ea1a15c0b2d8347bfdbd0c8f9fb16296476625bff58c12.
New V6 plan SHA82bde63d7badee998da6ba03a811c76b40253b207c3a71b89d669d9090b48744.
No batch process/GPU/model payload execution in this preparation.

VERDICT -> Source and actual prerequisite admission ready for future batch
runtime. No concurrent/cache/latency proof, no oldsource33 transfer. New registry
aliases remain a proposal until admitted. Original full fidelity and serving
campaign remain active; source-only serial cache V7 and originalrouteV2 ongoing.


## 2026-10-10 - Source35 serial cache V7 CPU and genuine preparation

CONFIG -> New immutable V7 cache controller/parent bound to exact source35/C113,
matched2048/64/65536 FP16 one[0] andpair[0,1] split32/16. OldV6 frozen.
P30/input33/batch observersOFF, EAGER absentincluding0. Exact cache/cancellation/
victim/continuation policies and separate OFF/logits/layers basic gates retained.

COMMAND -> Direct CPU controller and parent scripts30340/70287;18-file closure;
actual prepare98831/22256 using finalC113 source35 and pinned tokenizer image.
Admit12 batchV6 research aliases in evals/configs/models.yaml, exactgate checks.

RESULT -> Both direct CPU suites/closure PASS. unittest discovery was initially
used on script-style tests and reported0tests/exit5; corrected direct scripts
PASS. Initialprepare85547 hadwrong identity filename and failed before output/
Docker; corrected c1-post-model-identity-v13.json then both actual preparesPASS.
NoGPUexecuted. Same freshSDK/tokenizer/tokens/matched geometry bothtopologies.
Oneplan SHA fee5886f99a05ffa614780960ed2fd88e66fbb57633681378a08be66a24747dc;
pairplan SHA cb1b8297e92c7afa912ed0715359ace7646cea49c0af2eec8e822272bb3f743f.
Plans under f16-source35-20261009/serial-cache-v7-{onecard,pair}-prepared.
All12 source35 batchresearch aliases appearonce; hotschmoe-dd primary retained.
V7 sourceplan SHA c731157e99b07e310289ac643fd3a9c782d7aad9b1931275ba294e9f079e3394.

VERDICT -> Actual source/tokenizer-bound plans ready, not cache/runtimequalification.
Root must run onebasic thenpairedbasic before cachegroups; originalownroute
fidelity remains next. No shelf/cache/concurrency/latency claim. Fullgoalactive.


### 2026-10-10 - V7 sourceplan whitespace follow-up

The preceding V7 checkpoint had an extra blank line at EOF in the parent CPU
test; diff-check caught it. Removed only that blank line and refreshed its
sourceplan test hash. New V7 sourceplan SHA
41aec78ab833362668070841c8c67f599328d79f018c8f354a19c41b23f0a513.
Controller/parent/math/tokenizer and both actual prepared plans are unchanged.
No runtime proof changes. git diff --check now passes.


## 2026-10-10 - Original route V2 ready and two live correctness experiments

CONFIG -> Exact source35/C113/P30V4, independently zero-owned original48,
acceptedprefix1/2/4/8 with source-derived earlierprompt_verify/finalverifier.
Frozen originalprimitives/V4 unchanged; native inputs/routes never mathfeeds.
Separate native onecard cacheV7 basic OFF/logits/layers control underbothhealth.

COMMAND -> Root routeCPU20 session88984 PASS14.5s; independentreview20PASS;
full frozen36-file dependency closure. Actual originalprefix2 CLI61391 with
OMP/OpenBLAS/MKL/Numexpr1, P30one post-model-identity.json, newoutput
f16-source35-20261009/original48-routes-v2-prefix2. Cachebasic parent71545
at f16-source35-20261009/serial-cache-v7-onecard-basic via bin/gpu-run.

RESULT -> Route source/CPU/source3 schedule/zero-state/input-isolation controls
PASS. Sourceplan SHA7676d9ebf641df4b41be36aa9286a5f65188681fa91d37c0fe30536a86c12ec6.
61391 confirmed live, no originalprefix2 numerical result yet. 71545 confirmed
live, strictpercard/compiledP2P0 prehealthPASS, owned basic_off containeractive.
No cachebasic terminal or finalqualification yet; preserve/poll exacthandles.

VERDICT -> Real original-weight computation and native fresh-state controls
started. Need originalprefix2 terminal/postCPUnewfull4, then independent1/4/8;
cachebasic terminal/teardown/posthealth/new4 beforepairedbasic/cachegroups.
Native actualT/group rounding, fullmodelmath/cache/concurrency/latency/shelf
remain unqualified. Originalfullcampaign active; no speed/stability promotion.


## 2026-10-10 - Real route-prefix1/2 and onecard basic complete

CONFIG -> Same source35/nativeHC/exact locked UD-Q4_K_XL, routeV2 original
ownzero state with singleBLASthread, onecard P30V4 native comparison targets.
Separate source35 cacheV7 basic fresh OFF/logits/layers matched2048/64/65536.

COMMAND -> Originalprefix2 CLI61391 andprefix1 92243 terminal0, actual full4/
source3/page proofs; rawsha/all278 savedownarrays compareV2prefix1 vsfrozenV4;
V2 head diagnostic +CPU4 failclosed controls. Cacheone71545 and realpublic
basic_receipt_binding recollection47770. Newprefix4 77913 andpairedbasic1748.

RESULT -> Both original explorations complete/errors[], newpostCPUall4/pages/
source3 PASS; no numericalPASS. Actualprefix1 ALL278 ownsavedarrays BITWISE
V4, reportSHA dca1fb225e25422c6cc4b5ca89bc3cffefd42ff0f0b90bf94d20f60971a08301.
Prefix2 ALL289 phase/head comparisons completed; row0L1/L2FFN~1e-14.
Row1L0FFN2.9379143166e-14 -> L1attention1.6997006209e-10 -> L1FFN
1.7311205814e-7. HeadNMSE1.0125105723874907e-4/maxnormalized.00433114524646.
Prefix2 reportSHA b622572574690680cbe74763fd537aa255169c122a5ceba8b0a61afc7f151f1d.
Same headargmax/top10; distribution KL2.4398092047e-11/TV7.5985260683e-10.
This is a near-certain short header prediction, not a broadquality result.
Cacheonebasic PASS765s, natural output/LP20/observerfullhead bitwise,
all48 finalwindow coverage, owned cleanexit/removal/posthealth/kernel/new4PASS;
publicrawrecollection PASS, childSHA
bb301e9f2b479124249740d76dd8d182b7d74061bcde8061cf664e12d21beb13.
Paired basic1748 live requires this completedone; prefix4 77913 CPUlive.
BatchV6 onecardnative2diag1 genuinely prepared72372, no actualbatchrun yet.
Trace sourceaudit finds inert gpu_stamp0, profile disablesoverlap and existing
hostcounter doublecharges precollectedPLE; existingtimers notstageGPU evidence.

VERDICT -> Actual route-reference backward compatibility and prefix2 estimates,
plus firstsource35 serialbasic prerequisite qualified. Investigate row1FFN seams
with explicitlyconditional diagnostic, preserve fullown reference. Prefix4/8,
complete-state cachegroups/concurrency and x2-method criticalpath/clean latency/
verifiedshelf remain required. No tolerance, fullmath, quality or speed claim.
See docs/20261010_flashnext_source35_trace_coverage_audit.md.


## 2026-10-10 - Longer original prefixes expose fidelity gap; paired basic PASS

CONFIG -> Same source35 exactoriginal routeV2 prefix4/8 ownzero mathematics;
no native values used in complete reference. Separate row1 conditional FFN V2
prefix2 atlayers1/2/4/10, source35 paired cacheV7 basic2048/64/65536 FP16.

COMMAND -> Original4 77913/8 37298 terminal0 +newpostCPUfull4/source3/pages;
conditionalCPU21/11-file closure, real8case35701 terminal0/newfull4/pages.
Paired basic1748 terminal0 andgenuine rawrecollection72525. Actual native
prefix2vs4 first2 rows publicrecollection80855. Newrootcacheparent95907.

RESULT -> Prefix4 headNMSE.00932149853017/max.0627542661573, argmaxDIFF,
KL.106027145542/TV.225238467082. ReportSHA
ad23c7444e7436594e740ce696968d6bac82b796f3c978e138ec16403a5fabe4.
Prefix8 headNMSE.0524038742028/max.0779862658651, top10overlap7.
ReportSHA c846fd45044a2b7b47000b91982797e1ea739ccaa9409008025278a642139445.
Both source3/page/newfull4 proofsPASS/errors[], NO numericalqualification.
Near-certain prefix8 headerargmax agrees/tinyTV, which doesnot negate logitsgap.
Native sharedfirst2 rows ALL288 vectors BITWISE acrossprefix2T1 andprefix4T2;
firstbigger prefix4 row3L0 attentionNMSE2.6214e-6/FFN8.8696e-6.
Conditionalrow1 FFN4cases NMSE1.24e-15..3.93e-15; all4 owned replaysBITWISE,
estimatedmixedpacketSHAs differ atall4layers. ActualGPU packets unobserved.
SeamreportSHA e42050023c435e86a2343073777c706ade77e32c910aa2e6d090b19200cfac61.
Pairedbasic PASS500s/naturaloutputs/LP20/observerhead/full48/teardown/posthealth/
kernel/newfull4 PASS; publicrawrecollection PASS. ChildSHA
e27fd832c47de33e7ee38961a7f649a0b77d64ccfbd2d3b4377b31bbbc915bf9.
Rootcache one parent95907 nowlive, sharedrootfreshvsreuse diagnostic.

VERDICT -> Larger reference discrepancy ismaterial; cannotdismiss as harmless
rounding or assign PASS. ConditionalFFN agreement supports incomingdifference/
quantization amplification, notcomplete-modelquality. Sourceaudit finds GDN
T2 deferredcommit versusT1selfcommit, bothF32/sameintendedrecurrence; no proved
semantic/storage bug justifieschanging reference/kernel. FreshSYCL state-transition
leaforacle preparation willtest actualstate/output path parity independently.
Cache/shared-root native consistency remains a diagnostic, notoriginalfidelity/
production cache proof. Fullgoal/cache/concurrency/profiling/latency/shelf pending.


## 2026-10-10 - Cache pin0 harness diagnosis; real GDN transition fixture PASS

CONFIG -> Current source35 SDK/native model unchanged. Separate synthetic GDN
leaf with actualmetadata S128/HK16/HV48/C10240/conv4, F32 state3MiB/history120KiB.
New cacheV8 omits defaultpin; explicit0/positive retained; oldV7 evidence frozen.

COMMAND -> CacheV7root95907 terminal1/strictfailure, posthealth/newfull4/pages.
Inspect actual requests/selection and source batch_public ceiling. CPU V8pin9/
source22closure; newone/pair V8prepare2003/44170. GDNsource/collectorCPU, parent17,
leafcompile75774 failure then65125 PASS20s; realowned GDNparent25432 PASS223s;
independent raw27 recollection. V8onebasic92970 nowlive.

RESULT -> Cachefailure was harness defaultpin0 forcing lookupceiling0, not
observed cache math corruption. Actual refB fresh1/establishA fresh1/hitB fresh0
allpin0; hit candidate/reused/read_from0 and49fresh rows, intendedroot21 valid.
Engineexit0/removed/posthealth/kernel/newall4 PASS; failed gate/evidence retained.
V8 sourceplan SHA f37e86eca829fda1c2fd6433a6d7290195b4bca555efd5165a8ebd58e4b9ad0e;
newplans ready, newbasic mustqualify changed protocol, no V7 transfer.
Firstleafcompile11s lackedELF becauseimagebash-lc entrypoint ignored nestedbash;
failedreceipt/log/oldcontrollersnapshot retained, cleancontainer0/removed.
Explicitentrypoint fixed, genuinefreshELF link PASS20s/after-source unchanged.
Compile receipt: /mnt/vm_8tb/b70/build/gdn-state-transition35-leaf-v1-entrypoint-fixed-20261010/receipt.json.
ActualGDN all27 fullraw qkv/state/history/output comparisons BITWISE,
modifiedstate negative detected; realURlogicalfree/cleanexit/removal/strictpair
posthealth/kernel/newfull4/page/source checks PASS. ParentSHA
4084a293ae40fc8293dfac00589cd272e7887ca7912e6054e5774371c11d901c.
Staticarchives freshly pinned byleaf but absentfromhistoricalSDKreceipt;
associationlimit recorded, no retroactive archive proof or model math claim.

VERDICT -> Synthetic T2commit/T1carry parity succeeds, notrealrow3/fullmodel
fidelity. Existing retained21/23 hooks can observe real GDNstate/conv before/
after row3 and qkv/decay/beta; newsource35 NUM10 wrapper is beingprepared,
no backendpatch needed. Realprefix4/8 discrepancy remains material. V8basic
one92970 live; completecache/concurrency/criticalpath/latency/shelf remain pending.


## 2026-10-10 - V8 onebasic PASS; real-state NUM10 capture started

CONFIG -> Exact source35/C113 SDK/model unchanged. V8 defaultpin absent,
explicit zero/positive preserved. New NUM10 matched same-source OFF/ON,
rawfinalT1 prefixes1/2/4/8, 33 L0fields including true before/after GDNstates.
Separate original-owned layer0 microreplay never receives native mathinputs.

COMMAND -> V8one92970 terminal0/publicrawrecollection64261; NUM10 genuine
prepare98335, CPU34field/packet +10parent +7admission/24file closure PASS;
ownedL0microCPU13/55frozen closure PASS. Real NUM10parent98868 nowlive.

RESULT -> V8onebasic PASS713s: absent-pin commandsource checks, naturaloutputs/
LP20/matchedobserverfullhead/all48, ownedcleanexit/removal/posthealth/kernel/
newfull4/pages/source PASS. ChildSHA
0d32f4717a4c0e2f3e7b4ae1acd47107e1c199386cdf8f0a1ba6082342ea245c.
NUM10 actualplan under f16-source35-20261009/num10-onecard-prepared,
sourceplan SHA2026b454e437eef53155fd2dd05121beb630db630e3e5487b19fce6ab65c6210.
Parent98868 at num10-onecard-run holds pairlease, strict/compiled prehealth
running; no finalcapture/state result yet. Retained21/23 hooks already compiled
in35; no backendpatch/ABIchange. Native actual 31 fields+2derived retain scope.
New ownedL0replay sourceplanSHA
890a5e5b5ffa5a76e2dced293555beea2cfc01c9b0b80ad2ed0dd59c4d3fd1d3.
Original-only embeddings/HC/GDN preservezero ownhistory and physicalstate
permutations; full48 savedrow/state replay guards prepared, no payloadrun yet.

VERDICT -> Corrected client prerequisite qualified and actualmodelstate
observation begun. GDNsynthetic27 parity doesnotsettle realprefix4/8gap.
Require NUM10actual terminal/rawpacket/lifetime/OFFON/source4 proof, then
original-own row3 state/params comparison. V8paired basic/cachegroups,
concurrency/criticalpath/cleanlatency/shelf remain required. Fullgoalactive.


## 2026-10-10 - Real row3 activation-packet boundary localized

CONFIG -> Genuine source35 NUM10 OFF/ON same2048/64/PLE65536/FP16KV,
independent original-owned L0prefix4 zero-state replay, original primitives unchanged.
Separate explicitly CONDITIONAL native-mixed diagnostic, never fullreference.

COMMAND -> NUM10one98868 PASS643s/publicrawreader92859; ownedL0prefix4 67857
terminal0/source3/newfull4/pages/ten originalV2 replay checks; raw Q81 block
inspection and F32 scale/quotient atindex1123. ConditionalCPU5 andreal3case32507
terminal0/newfull4/source3/pages. V8pairedbasic65259 PASS454s/publicrecollect50866;
correctedone rootcache99168 nowlive.

RESULT -> NUM10 completeactual4prefixes x33fields, packet/lifecycle/OFFON/full49/
ownedteardown/health/kernel/new4 PASS. Owned L0replays savedV2 allrowinput/
attention+finalstate/history BITWISE10checks, reportSHA
947057466d9a0d23f81aa65c2b77e9d6211098bb427336e0c8e73472c0bbcd83.
Realrow3 incomingstateNMSE2.2303e-14/conv3.1805e-15/mixed3.4049e-15;
attn_input_q81 differsONEcode inblock35,index1123: own71/native72.
The value1.143454909324646 itself isidentical; blockmax isown2.031032085418701
versusnative2.031031847000122, oneF32ULP. F32scales .015992378816008568 vs
.01599237695336342 yieldquotients71.49999237060547 vs71.5; storedhalfD andS
bothidentical. This isactual observedpacket evidence, not predictedGPUbytes.
GDNoutputpacket differs92blocks/34codes/19scales/68sums. Own originalblockout
NMSE2.6158372698e-6. ConditionalnativeMixed+OWNbeforestate reducesblockout
NMSE7.9254321969e-15 andboth actualpackets BITWISE; qkv/z~3e-15.
Addingnativebeforestate also givesblockout7.9254e-15; ownbaselineall8fields/
bothpackets replaysBITWISE. ConditionalreportSHA
76d9986a97f736ad78cb2f2371d988042c6a5ced7c77b3417a7ac628dfea7432.
Whole-vector substitution cannot isolateonly the singlecode apartfromsmall
accompanyingmixed changes. No complete-reference nativeinput substitution.
V8paired basic passes naturaloutputs/LP20/head/full48/teardown/posthealth/new4.

VERDICT -> Incoming HC arithmetic/activation packing is a concrete local seam;
no statecorruption or GDNkernelreplacement justified bythese data. Still no
fullmodelmath/tolerance/quality/speed PASS. Native HC FP32 FMA/reduction versus
originalFP64-dot/F32store audit underway; broaderfidelity and meaningfulquality
remain required. Correctedrootcache99168 running; allstatecache/concurrency/
criticalpath/cleanlatency/shelf retain originalfullscope.


## 2026-10-10 - Corrected V8 shared-prefix root one-card qualified

CONFIG -> Exact source35/C113 SDK and UD-Q4_K_XL identity unchanged; V8
absent default pin, context2048/prefill64/PLE65536/FP16KV; serial native GEN,
all48 phase observer. Root boundary21, divergent A/B suffixes, 49-token prompt.
Frozen original mathematical reference remains unchanged.

COMMAND -> Owned parent99168 terminal0 in450s; public extract/compare_requests
recollected allsix rawrequests and allthree comparisons. Read final parent,
child, actual ledgers, teardown, health, full4 identity and both page guards.
Next paired root owned parent93880 launched with finalized matched pairbasic
and onebasic receipts; acquired cards0/1 and confirmed pre-strict stage.

RESULT -> Onecard root PASS: both nonfresh B hits select checkpoint21,
actually reuse21 and evaluate28 promptrows; cold B references evaluate49.
Allthree comparisons preserve natural completion, outputIDs, LP20, full
248320-logit head and all48 residual vectors BITWISE. Container normalexit0,
removed, no forcedcleanup/errors; strict and compiled-pair pre/posthealth,
kernelgate, postterminal/posthealth newcomplete four-shard hashes and two
knownpages before/after PASS. Public raw recollection PASS.
Parent: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v8-onecard-root/parent-qualification.json.
Parent SHA256 9953bebc7c1dafe4229442cba0e8de0cb1189ffc1989da241ad5c951e41667d5.
Paired root output: same runtime parent/serial-cache-v8-pair-root; session93880.
Independent HC audit records F32 lane/FMA/XOR versus frozen FP64 accumulation,
post-SiLU capture semantics and unresolved exp/rsqrt/contraction contracts.

VERDICT -> Genuine serial checkpoint reuse demonstrated after protocol repair.
This does not prove all cache modes, original-model quality, concurrent state
isolation, cache latency or speed. Paired root is live and unqualified pending
terminal/raw/lifecycle/health/new4. Pin/turn/parked/eviction/cancellation/live
continuation, concurrency2/4/bounded6, fidelity, critical-path profiling,
matched per-stream latency and verified shelf remain required. Fullgoalactive.


## 2026-10-10 - Paired checkpoint reuse PASS and host arithmetic controls

CONFIG -> Exact source35/C113 SDK/UD-Q4_K_XL, paired serial root boundary21;
separate source-indexed HC host prototype, original FP64 reference unchanged.
No GPU/model payload touched by host arithmetic compilation or controls.

COMMAND -> Paired root93880 terminal0 in331s; public raw extract/compare of
sixrequests/allthree comparisons plus final health/identity chronology checks.
Host compiler preflight failed (no c++ installed), retained failed receipt;
compiled fresh host C++ in pinned image39992 with no GPU device/model mounts,
FE_TONEAREST/FTZ-DAZ OFF, -fno-fast-math/-ffp-contract=off/-frounding-math.
Independent review found directPython subnormal and XORorder test gaps;
preserved frozenV1 and prepared V2 mandatorybehavioral probes/ordernegative.
Final V2CPU18 PASS and actualcompiled adapter3 PASS afterfinalsource freeze.

RESULT -> Pairedroot bothcheckpoint hits reuse21/evaluate28 of49prompttokens;
full248320-logit heads/all48 residuals/outputIDs/LP20/naturalfinish BITWISE
for allthreecomparisons. Normalownedexit0/removal, noforcedcleanup/errors,
strict/compiledpair pre/posthealth/kernel/newfull4/pages/source PASS.
Parent path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v8-pair-root/parent-qualification.json.
ParentSHA 5bf7f8035b826844ef4dfdd02f1e4584edab0f09c24740e4f3737883503ea0a5.
CPUhost helperfreshELF d97ea12b9b805e3e8d9520ae8452dcf2a8129216e7666af6f5e44926e80b48e8;
compile receipt1fba58350b0fe82987bf1916e7f1a663cbadfee5ee8bad985d2cb51cf9bdbac1;
15 directanalytic controls receiptb05cf281b7bc1ec63a4164459194e465589b488951c974e06b9d4feb65bff1cb.
Host root: /mnt/vm_8tb/b70/build/hc-f32-arithmetic35-host-v1-container-20261010.
FinalV2plan b5eca831ca6725b0674d8e06584e0fbb2c42da551756b9f6d63876ea24cf9aae;
finaladapter3 receipt425d397b0a5d6228f5990af0392132d0e9fe68c8fad3e95be2f4fb6a7ef22309.
Earlier draftadapter receipt retained, not substituted for finalized binding.
Next onecard explicitpin parent71624 acquiredpair lease at
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v8-onecard-pin.

VERDICT -> Both topologies qualify this serial shared-prefix checkpoint case.
CPUanalytic/source-shaped arithmetic controls are established; exp/rsqrt/mix
contraction/fullHC/GPU equivalence/modelquality remain unqualified. Newpin
run is live, not yet qualified. Othercachegroups, concurrent2/4/bounded6,
fullfidelity/quality, criticalpath/cleanlatency and shelf remain required.
Fullgoalactive; no speed or production qualification claim.


## 2026-10-10 - Onecard explicit pin qualified; host runtime closure established

CONFIG -> Exact source35/C113 SDK and original UD-Q4_K_XL, V8 explicitpin191
on209token divergent-suffix fixture; serial native GEN/FP16KV/ctx2048/pref64/
PLE65536. Separate unchanged HC host helper and finalV2 arithmetic adapter.

COMMAND -> Onepin71624 terminal0 in499s; rawrecollection85731 checksall6
requests/threecomparisons; inspectfinalparent/health/modelhash chronology.
Review found dynamic-library/rawtinyfixture closuregap; new supplementary
hostproducer qualifies15knownanalytic+3actualadapter rows and preservedraw
fixtures/commands/source/compilerlog/hostELFdependencies. CPU8 PASS and
separate finalized_binding reread PASS. Pairedpin99833 now acquired0/1 leases.

RESULT -> Bothonecard explicitpin hits actuallyreuse191 andevaluate18 of209
promptrows, freshreferences209. Allthreecomparisons naturalfinish/outputIDs/
LP20/full248320 firsthead/all48 firstwindow residualvectors BITWISE.
Ownednormalexit0/removal/noforcedcleanup/errors, strict/compiledpair pre/post
health/kernel/newfull4/pages/source PASS. ParentSHA
bde62853bcb27cab1786ac67c2e078f0b747bd202d2884eff888f0fe31e5dcfc.
Parent: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v8-onecard-pin/parent-qualification.json.
Supplementary hostroot: /mnt/vm_8tb/b70/build/hc35-host-runtime-v1-20261010.
ReportSHA 7ab01bf3efae5daac463a56b3ba620fb2ee20e883a6d104808e99c5ad17e8931.
SourceplanSHA 4e41ed7824a9fe2aef95c78ca0cd7c8c222df218a55e988b678cc819b02c898f.
Pre/posthelper/runtime/source bindings match; all5resolved ELF libraries,
Pythonmappedctypeslibm, CPU/kernel/affinity/loaderBLASenv, exacttinyfixtures/
expectedbytes andactualoutputs pinned. NoGPU/modelpayload touchedbyhostrun.
Pairedpin active99833 at same runtimeparent/serial-cache-v8-pair-pin;
synthetic GPU HC publicprojection fixture and separate conditionaloriginal
HC row3 projections source-only preparations underway undercoordinator.

VERDICT -> Explicitonecard pin case and reproduciblelimited hostarithmetic
controls qualified. Deviceprimitive arithmetic/fullHC/modelquality remain
unqualified; conditional inputs cannot enter frozen originalreference. Other
cachegroups, concurrency2/4/bounded6, criticalpath profiling, matchedstream
latency and verified shelf remain required. No trusteddecode/productionclaim.
Fullgoalactive; unrelateddirtychanges preserved.


## 2026-10-10 - Paired explicit pin PASS; HC count mismatch and dev-loop audit

CONFIG -> Exact source35/C113 SDK/UD-Q4_K_XL; paired native V8 pin191 of209.
Separate synthetic public HC projection leaf/currentlibraries; originalmodel
math/weights unchanged. Root owns all GPU work; research agents source/CPUonly.

COMMAND -> Pairedpin99833 terminal0 in350s, rawrecollection ofall6/threepairs,
finalhealth/full4/pagechronology checks. HC V1actualexpectedexport/CPU12+parent18,
freshcompile47460 PASS21s, ownedparent26925 terminal1 in225s. Independently
recollectactualGPU raw13cases/allwords/negatives. Dispatch dev_loop_audit at
userrequest, inspect actualparent timing/source and existinggroupall. Preserve
V1; NEWV2 fixture CPU14 andparent CPU20 derive/check counts beforeGPUhealth.
V2 realexpectedexport PASS, freshmetadata-linked compile87264 nowlive.

RESULT -> Pairedpin actuallyreuses191/evaluates18 inbothhits; full248320
firsthead/all48 firstwindow residuals/naturalfinish/outputIDs/LP20 BITWISE
for allthreecomparisons. Ownedexit0/removal, health/kernel/newfull4/pages PASS.
ParentSHA dc82d6a3c80588911ceb7bedbc4c6781bf87be84a117ea025d20241c8b6ef56f.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v8-pair-pin/parent-qualification.json.
HC V1all13cases/actual36 outputfloats match CPUFMA/XOR BITWISE; two wrongorder/
weight negatives detected. ParentFAIL solely becausemetadata/guard claimed38
words instead36. Leafexit0, removed, strict/compiledpair pre/posthealth/kernel/
postsource/newfull4/pages PASS; errors retainedexactcount guard failure.
V1 path: same runtimeparent/hc-projection35-v1-owned/parent-qualification.json.
No oldfalse->true rewrite or modelqualification. NEWV2 data/CPP/librarymath
unchanged; corpuscount derivesactualm*t andsourceplan/input/recipe/parent
counts mustagree beforeevenhealth/fullscan admission; wrong38 negativepasses.
V2fixtureplan1236831f2f48b355d2c15be5e3475b9013be2eaa8643c79eb626765317875547;
V2compileplan05a6a14b175029a1c1850a2bd34de1bd397068d90138b9248c7d6a03c4cd9b8a.
V2package: /mnt/vm_8tb/b70/build/hc-projection35-fixture-inputs-v2-20261010.
V2compileoutput: /mnt/vm_8tb/b70/build/hc-projection35-leaf-v2-20261010.
ConditionaloriginalHC selectedprojection CPU14/all70dependencies PASS, pending
qualifiedGPUprimitive; no nativeinputs enter frozen fullreference.
Devloopaudit measured sixfinalizedruns: health+newfull4 cost231-294s; pairroot
74percent gates. Modelbytes111334654784 (103.69GiB), not198GB. ExistingV8ALL
suite retainsfullgates andcouldsave about17min onpairedremainingcachecampaign
versussevenseparateparents, projectiononly. Parallelstrict health26-30s/parent
opportunity; actualparallelhashgain unproven. Countpreflight wouldavoid225s
knownfalsefailure. No bins/shelf/activecontroller changes orgate removal.

VERDICT -> Sharedroot andexplicitpin cases qualified onboth topologies;
othercachemodes/concurrency/fidelity/quality/profiling/latency/shelf stillpending.
HC numericaldata ispositivebutV1wholequalificationFAIL; V2freshactualruntime
mustfinish raw/free/identity/health/ownedteardown beforeprimitivequalification.
Nextcompletecache execution shoulduseexistingALLsuite afterHCchecks. Fullgoal
active; no productiondecode speed claim, unrelateddirtychanges preserved.


## 2026-10-10 - Corrected HC V2 fresh runtime started

CONFIG -> V2 derives13cases/36words/2negatives; original CPPv1 and current
SDK/library mathematics unchanged. V1 failed-count history remains preserved.
COMMAND -> V2 real CPUexpectedexport PASS; rootCPU14+parent20 PASS; leased
freshcompile87264 terminal0 in20s; ownedruntime72004 now acquiredpairleases.
RESULT -> Compile receipt SHA256 8d1eca82ed1dc7d936be2f8df08e553780662ad774085b7c824fa1efffb1d670.
Build: /mnt/vm_8tb/b70/build/hc-projection35-leaf-v2-20261010/receipt.json.
Runtime: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/hc-projection35-v2-owned/parent-qualification.json.
Runtime72004 confirmedlive atpre-strict/pre-compiled-pair; no finalresult yet.
VERDICT -> Current actualV2 qualification pending raw36/free/ownedteardown/
health/newfull4; no modelqualclaim. Devloopaudit checkpoint24d86ef pushed.
Next cachetesting usesexistingALLsuite toreduce repeatedparentgate overhead,
after HCfidelity diagnostics. Fullgoal remains active.


## 2026-10-10 - HC primitive PASS; cache reader recovered without GPU rerun

CONFIG -> Source35/C113 exact UD-Q4_K_XL and native math unchanged. HC V2
fresh synthetic public projection; separate conditional original-weight HC
prefix4 row3. V9 remaining-seven one-card serial cache suite, NEW CPU reader.

COMMAND -> HC parent72004 terminal PASS215s; conditional51256 complete with
new post-CPU source/full4 evidence. V9 parent55814 terminalFAIL1521s; NEW strict
read-only adjudication34115 PASS with all19CPU controls and independent review.
CPU two-stream replay58238 reproduces event-scope defect, scoped audit PASS.
Fresh paired V10 prepare and leased47979 acquired0/1, pre-health now underway.
CPU-only fresh llama.cpp V3 build64755 live, source completeness PASS3579files;
V1 missing nested model source and V2 unused-option failures preserved.

RESULT -> HC13cases/36words/2negatives BITWISE; owned cleanup, health/kernel/
source/new4 gates PASS. ParentSHA
1fccb38232e9e586019dba06d4382907105e0a0f11492bc71df269f6ede0f34d.
During manual HC reread, raw_proofs rewrote two derived JSON files with identical
bytes; hashes were checked unchanged. Future revalidation uses readonly collect.
Conditional eight selected up/inject projection values BITWISE under native
F32 FMA/XOR contract; same-input F64 reference differs in F32 bytes. These are
supplied-native-input conditional diagnostics, not fullmodel math proof.
Conditional reportSHA95f6cac73e7586ec20be391a7301ad7f9eef90e7b2366901be2b0b00870ce604.
V9 allseven groups numerical/teardown PASS but finalizer falsely rejects raw
JSON saved before extract adds derived committed IDs. Original failure remains.
NEW reader derives complete copies before equality and replays40requests,
seven production branches, exact tokenizer continuation, source/SDK/identity,
owned commands/health/teardown/kernel/new4/page chronology; PASS and evidence
tree unchanged. Turn reuse63, RAMrestore42/49, live51/79; all comparisons match.
Reader reportSHA 0ba8b1eacb88a687fdf058e702656d8307a74521ab611a677d386a7c940f44a2.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v9-onecard-readonly-adjudication-v1/adjudication/report.json.
Two-stream V6 parent failed solely whole-trace duplicate event namespace after
warm/ARM reset; exact replay observes warm32/32, cancel6 and survivor32 length,
scoped31events/196vectors. No native64/natural/concurrency/math/speed claim.
V3 CPU configure recipe removes only two unused FetchContent cache variables;
networknone/backendOFF/sourceV2 provenance preserved; no model mounted/read.

VERDICT -> Primitive HC and bounded serial cache proof advance full goal;
whole model/quality, paired remaining cache, API stale identity/concurrency,
1/2/4/bounded6 streams, profiling/latency and shelf still incomplete. Paired GPU
47979 and CPU build64755 are live, not restarted on observation timeout. Group
bundling and reader replay reduce repeated work; no matched timing gain claim.
Preserve all failed history and unrelated dirty files. Full goal remains active.


## 2026-10-10 - Fresh CPU reference binaries built; runtime loader qualified

CONFIG -> Verified V2source3579files/98MB with V3CPU-only recipe; pinned39992
compiler/runtime, clean env/networknone/2cores/8GiB/no model mounts/devices.
COMMAND -> Actual64755 compile/link all6 exit0/ownedremoved. Host ldd postreader
fails missing libgomp; preserveFAILED original. NEW35997 all6 ELF/currentCPU
container dependency+cache/source closure PASS; NEW14260 actual server and
completion version checks plus test-quantize-fns PASS/noGPU/no model read.
RESULT -> Original receiptSHA2df28740819c1b0869395fb8ff52d000cf167e4993b434a3e76f5fc48a5969ac remains false.
NEW all6 closureSHA26e460378f77e5e6ae0e76d1293d1694991a491483cf628c9d37f096a9f6bfb6.
Failure associationSHAe2126b4e67bb51548f5b1647bb1d08e9b50ba139146de6bebaa0d821b7f8b18e pins exactoriginalerrors/hostldd/boundlogs/all6.
Runtime smokeSHA2cead24f1c108f07fb36c9feefb4c06aae2ecd1f5ecd3e2a137f74433a981937.
Path: /mnt/vm_8tb/b70/build/flashnext-cpu-build-v3-20261010/.
Rootbuildscript preserved withreceipt-matchedSHA; supplemental source script
also preserved. All6 derivefrom --target ratherthan oldslice omittingserver.
Intendedpinnedcontainer resolveslibgomp; hostlibrary absence needs no rebuild.
Pairedcache47979 confirmedlive; turn/parked/eviction partialPASS withowned
removal. Cancellation/live/teardown/freshposthealth/new4 stillpending.
VERDICT -> Real CPUbinary/runtime foundation readyfor exclusive-RAM functional
pilot afterGPUruncloses. No CPUmodel/quality/nativeexactmath/productionclaim.
NextV7concurrency parent preparedbyagents, independentreviewcaught serialjob
budgetcount and preflight-finally healthissues beforeGPU. Fullgoal active;
originalfailures/unrelateddirtyfiles preserved. No stack/package changes.


## 2026-10-10 - Initial V7 concurrency prototype checkpoint before hardening

CONFIG -> NEW source-only V7 harness, source35 backend/maths unchanged;
old V6 failed runtime and frozen V5 event auditor/pilotV7 preserved.
COMMAND -> Root14integration and11eventscope CPUtests PASS; independent source
review verifies ARMscope/native32/API64/serialGEN1 roles and raw49 joins.
Checkpoint initial frozen prototype before remaining identity-admission edits.
RESULT -> No genuine V7 plan prepared and no V7 GPU/modelforward executed.
Initial sourceplanSHA3623297cbf45f2ed94301d8b9a0ee2da6dd2b5f428435863e2c302c7f1b830d3.
Review still requires publicreader currentC113/SDK/shardstats/pages admission,
complete input/childplan+postidentitypath/terminal chronology crossjoins and
parent knownpage receipts bracketing NEW posthealth full4. Existing direct
parent codechecks help but do not substitute closed publicreader evidence.
VERDICT -> Source-only checkpoint, NOT runtime-ready or qualified concurrency.
Agent will harden source-onlyV7 beforeactualplan/runtime, update sourceclosure
and CPUnegatives transparently. This gitcheckpoint preserves original source
and test provenance; do not rewrite oldexecutedV6/V5/pilot/failedreceipts.
Pairedcache47979 remainslive, partial turn/parked/eviction/cancel/cancel_decode
passed; cancellationprefill/live/finalhealth/new4 remain pending. Fullgoal active.


## 2026-10-10 - Restore frozen V3 design binding; preserve actual runtime evidence

CONFIG -> V3CPU plan pins designSHA80d9f21da87ae691374fb97a6a03847010e544161029a9ccd13f1128dad78610.
COMMAND -> Root noticed actualruntime append in e64912d changed frozenrecipe
document. Restore exact plan-declared design bytes and move completeappend to
NEW docs/20261010_flashnext_cpu_reference_build_runtime_v3.md. Verify original
recipeSHA against plan; no recipe/plan/source/binary/runtime receipt changes.
RESULT -> Frozen designbinding restored; actualfailure and supplementalPASS
evidence preserved verbatim in newruntime document. Existing compile/source
checks had not detected this documentationbinding mismatch. Pilot will check
all design/source closure beforemodel execution. Paired47979 all7 numerical/
teardownpassed, posthealth/new4 still live and pending.
VERDICT -> Documentationprovenance correction, no experiment relabel or model
qualification claim. Use separateactualevidence documents for frozen designs.
Fullgoal remains active; unrelatedworktree changes preserved.


## 2026-10-10 - Paired remaining cache PASS and matched one/pair raw parity

CONFIG -> Source35/C113 exact original UD-Q4_K_XL; V10paired remaining-seven
suite, V9onecard originalfailure retained plus qualified readonlyadjudication.
COMMAND -> Leased47979 terminal0/PASS926s including ownedteardown/posthealth/
kernel/newfull4/page/source gates. NEW pinnedCPUread-only2122 recollects both
closed trees/all40 requests each/sevenproductionbranches, source/SDK/full4/
ownedcommands/currentraws, then compares matched uncancelled requests.
RESULT -> PairedparentSHA edc5eeb6e306e3f5a882c1c2714ed07c7fb142bd70d069fe03f4d7fa751801a2.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v10-pair-remaining/parent-qualification.json.
Both layouts all40requests recollected, fullhead/all48 firstwindow vectors,
outputs/LP20 bitwise for36uncancelled crosslayout pairs. Four asynchronous
cancel requests skippedforcrosslayout parity but eachlayout isolation/cleanup
was qualified. Both originalevidence trees unchanged. ComparisonSHA
6d6c45d9b6e80fbe5c4040990a121cdf085c1113b4c95342f9ed089f6827377a.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/serial-cache-v9-v10-remaining-topology-recollection-v1/comparison/report.json.
No GPU/model-forward inreader, no oldfalse flagflip. V7genuine2native prepare
1168 underwayCPUonly after36CPU controls/39fileclosure independentreviewPASS.
CPUfunctionalpilot initialsourceV1 root11CPU PASS, unexecuted; independent
review found memorymonitor errors do notinterrupt blocking1200sHTTP. Checkpoint
prototypebeforefix; no inference launch until ownedimmediatestop is verified.
InitialpilotplanSHA2a4e1abc032583d02cf765a375738d0e89b1f448e9f8385c2dbf833057437573.
VERDICT -> Bounded serial remaining cache qualified onone/pair with matched
noncancelled numericalparity. CompleteAPI/concurrentcache/staleidentity/cache
memorylatency/fullmodelquality/profile/latency/shelf stillrequired. CPUmodel
functionalpilot next afterprelaunchmemorycancel fix, no broadquality/speedclaim.
Source-only prototypes may be hardenedbeforeactualplans/runtimes withinitial
source/hashes preserved in git; do not rewrite actualoldexperiment evidence.
Fullgoal remainsactive; unrelatedworktree changes preserved.


## 2026-10-10 - CPU functional V1 wrapper failure before inference

CONFIG -> Revised source-only V1 memoryguard fix, independent/root15CPU PASS;
exact pinned CPU build, c388 metadata/39992 inference images. No GPU grants.
COMMAND -> Leased10681 actualV1 terminal1 in156s. Pre-full4/preguard PASS;
metadata container starts/exit1/normalremoved; root postterminalfull4/pages PASS.
RESULT -> Metadata script mount /harness/pilot.py has onlytwo parents, so module
ROOT=Path(__file__).resolve().parents[2] raisesIndexError2 beforeany inference.
No CPUcase/server forwardexecuted; reportactual_CPU_inference_observed false.
Original reportSHA e77d201da127383eb5eb7410c1ff303f613392a720c134ada433e8a8a643c024.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v1-20261010/report.json.
Source/plan/build/weight identity preserved, noGPUtouch. MockCPU tests and source
review didnotcover realmountedwrapper moduleimport. Allseven pairedcache and
36matcheduncancelled crosslayout bitwise proofs remain valid and separate.
V7native2 planCPUpreparedSHA2040aa96d982f139faead0ce105b92fd926ef0a233e6eac5ccd8cd8fc9892f0c; no actualV7GPU forward.
VERDICT -> PreserveactualfailedV1source/plan/receipts. NEWV2mustuse correctdeep
mount forbothmetadata/server helpers and actualCPUmetadata+server-version
wrapper preflight BEFOREexpensivefull4 scan, thenretainfull4pre/post around
actualinference. Agent/source-only fixes underway; rootaloneexecuteswrappers/
model. No rebuild, driver/package/registry change required. Fullgoal active.


## 2026-10-10 - V2 actual CPU wrappers PASS; functional model pilot started

CONFIG -> NEWV2 fixes shareddeep /harness/llamacpp/flash-next/pilot.py topology;
oldactualV1metadata failure source/receipts unchanged. Same exactCPU build,
originalmodel/tokenizer/template and pinned c388/39992 images; noGPU access.
COMMAND -> Root/independent16CPU tests+closure PASS. Actualwrapper-preflight
36590 terminal0/PASS: realmetadata export+productioninside-server --version
beforeanymodelscan, CPU2GiB/noswap/networknone/no model mounts/device grants.
Leased15024 startsNEWV2modelrun, repeatswrapperpreflight thenfull4 identity
beforefirstcase; allfour fresh-process functional requests stillpending.
RESULT -> ActualwrapperreportSHA 8594a7a1d57f75d168e20055f96c31c09b3169decb850949106e5a57c9b8347f.
Path: /mnt/vm_8tb/b70/build/cpu-functional-v2-wrapper-preflight-20261010/report.json.
V2sourceplanSHAa99252b366bb7167e983305c58269a56a1cd9e898796f0f0b860ec052e746448.
ROOT-depth bug nowactualcontainer-tested, source35 exports two exact declared
fixtures; CPUserver ELF/dependencies/version wrapperworks. No modelinference
bywrapper test. Modelrun15024 confirmedliveholdingpairleases onlytoexclude
GPUserving/RAMcontention; allinferencecontainers CPUonly/noGPU/health calls.
VERDICT -> Prelaunchexecution gap closed withoutABI rebuild/model mutation.
Currentactualmodelpilot muststillcomplete alias/template/token acceptance,
naturalEOS functionaltwo-prompts/freshrepeats, memory/no-swap/ownedteardown,
outputdecode andNEWpostterminalfull4/pages/sourcebindings. Noquality/math/
latency/productionclaim beforethoseactualresults. V7native2genuineplan ready,
notGPUexecuted; fullgoal intact/unrelatedworktreechanges preserved.


## 2026-10-10 - First real CPU function PASS; port guard corrected for repeats

CONFIG -> Exact originalCPUmodel/V2 parameters, no GPU grants. Two prompts/
twofreshprocess repeats declared; helperpreflights verifiedbeforemodelscan.
COMMAND -> V2 actual15024 terminal1 in196s. Case0repeat0 semantic/naturalEOS/
model-template-inputIDs/memory/ownedteardown PASS. Repeat1 fails BEFOREstart
on bare socket.bind OSError98 despite priorservernormalexit/removal; newpost4/
pages/source PASS. PreservewholepilotFAIL and partialpositivecase.
RESULT -> Actualresponse is fenced Python def add(a,b): return a+b;18output
IDs, naturalEOS, no truncation/no memoryerrors/normalexit0. Originaltemplate/
accepted30promptIDs match. No broadquality or deterministicrepeat proof yet.
V2 reportSHA 3148828582e927021f5d1c4c3ecf27df455c4fcdde1d657b61ef7e6d05d872ce.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v2-20261010/report.json.
Pinned httplib uses SO_REUSEPORT onLinux; ADDR-onlyprobe also rejects its old
TIME_WAIT. NEWV3 explicitlyrefuses procTCP/TCP6 LISTEN, thenprobes bothADDR/
PORT reuse andclosesbeforelaunch. Root/independent21CPU controls include4real
OS loopback socket cases (oldTIMEWAITfailure/newreuse/liveforeignrefusal).
Allnonport helpers/classes/modelparameters ASTunchangedfromV2; no rebuild.
ActualCPUwrapperpreflight14929 PASS, no modelmount/inference/GPU; SHA
180141cae0a0ccb64238e9844c1963decabb73a1f0463275f25fa6dcd80a5e74.
V3sourceplan890f6393a285449ebb54fb24e13b734db7b178a06c09236c7d25b77012d4de0f.
Leased41294 modelpilotstartsNEWoutput cpu-functional-v3-20261010 withsame
model/CPU/sampling/memory settings; allfour naturalrequests stillpending.
VERDICT -> Independentexact-model CPUmeaningfulfunction nowactuallyobserved,
wholepilotstillunqualifieduntilfreshrepeats/arithmetic/identity/cleanup pass.
V2 failure is lifecyclepreflight, not NNoutputfailure; preserve bothtruths.
MatchingactualOS/networkbehavior in cheappreflight avoids expensivefalse
modeltestfailure. Source-onlynative counterpart preparation delegated; no
GPUexperiment besideCPUrun. Fullgoal remainsactive/unrelatedchanges preserved.


## 2026-10-10 - Independent CPU functional pilot PASS; native2 collection started

CONFIG -> Exact original four UD-Q4_K_XL shards/frozen3579file CPUsource/V3build;
CPUfunctionalV3 hostloopback, originaltokenizer/template, temp0/seed1234/max64,
8threads/norepack/noGPUaccess, freshprocess perprompt/repeat. No registry edits.
COMMAND -> Modelpilot41294 terminal0/PASS286s. Independentlyinspectactual four
responses/IDs/naturalstop/teardown/memory/source identity. Root starts leased
V7native2 diagnostic88303 from genuineplan2040aa96 afterCPUrun fullycloses;
no concurrentRAM-heavyCPUmodel/GPUserve. Agents nativefunctionalprep sourceonly.
RESULT -> Bothfresh functioncases produce identicalfenced def add(a,b): return
a+b (18tokens), bothfresh arithmeticcases exact12 (3tokens). AllnaturalEOS,
inputtemplate/tokenagreement and original outputtoken decodePASS, deterministic
freshprocess repeatsPASS, no memory/swap/forcedcleanuperrors, normalownedexit0/
removal all4. Pre/postcomplete4/pages/source/build/runtimebindings PASS.
ReportSHA1b98ee53d20896840215da7c1f97b4245de86b3955abe80ce0cef09e1727e3f8.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/cpu-functional-v3-20261010/report.json.
PeakobservedRSS64.770GiB from1s samples; minimumsampledhost MemAvailable116.166
GiB. RSS includes mmap/file-backed pages; these arenot totalresidentweights/
capacity estimates or speed results. Originalmodel filebytes103.69GiB unchanged.
Native2actual88303 pending fresh health/collection/cancellation/ownedteardown/
new4/source andsubsequent4exactprefix serialjobs/196vector numerical join.
VERDICT -> Trustworthy LIMITED independent CPUfunctional/freshrepeat baseline
nowestablished onselectedartifact. Not registeredbroadeval, full48 native math,
history/concurrency/stale-cache/latency or productionqualification. Two-prompt
Strata counterpart and V7private2stream equivalence next, then4/bounded6 and
fullcriticalpath/matchedlatency/shelf work. Preserveall V1/V2failedhistory and
qualifiedone/pairserialcache proofs. Fullgoal remains active.


## 2026-10-10 - Native2 closed collector and actual serial equivalence arm

CONFIG -> Exact source35/C113 native2diagnostic32 onphysical0; correctedV7
source/pilot budget/ARMscope; no underlyingmath/ABI/registry changes.
COMMAND -> 88303 terminal0/PASS467s. Currentpublicreadonlyparent_arm/vectors
37300 verifiescompleteclosedsource/identity/health/page/raw joins,196vectors
andactual4 consumed-prefix jobs. GenuineCPUserial0prepare99325 PASS; leased
GPU9143 startsall4fresh GEN1 jobs andpassedpre-strict/compiled-pair/kernel.
RESULT -> NativecollectorparentSHA 1f9d763dec77be9b0136ae0d397261973639b3a0c9751486dfd39b9c367617e5.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native2-diag1-run/parent-qualification.json.
SerialplanSHA 0c8e8a0535fa91d786f183439bec8d9532c4e5dd4b98cb6065ebc3b814050d12.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native2-serial0-prepared/plan.json.
Actualserialbudget[1,1,1,1], not32/64; nativebudget[32,32] diagnosticlengthcap.
Serial9143 confirmedlive, numerical196vector/privatehead join andpostgates
remainpending. No freshAPI/OFFON/4/6/latency/concurrentcache proof transferred.
Source-onlynativefunctional screen has12CPUcontrols afterstrictrecipe/command/
end-sourcejoin hardening; manifest1d353f69d619cca48545d9794a1697e502d450d7265fc475b77368ca8c5d5203 awaitsindependentfinalreview.
No genuinefunctionalplan/GPUrun; frozenV7runtime/serialsource unchanged.
VERDICT -> Realcurrent native2collector proved andfreshserialreference arm
nowrunning. CPUtwo-promptreference PASS stands; fullmodel/nativequality and
concurrency/completecache/profiling/latency/shelf remainopen. No speedclaim.
Fullgoal remainsactive; alloldfailures and unrelateddirtychanges preserved.


## 2026-10-10 - Cache-off serial reader mismatch; functional screen prepared

CONFIG -> V7freshserial GEN1/cacheOFF, actualnative2collector qualified. Separate
source35Stratafunctional screen exactCPUV3two-prompts/freshGEN64 observerOFF.
COMMAND -> Serial9143 terminal1/FAIL421s. Firstjob engineexit0/normalremoved,
posthealth/kernel/new4/pages/source gates; no numericalcomparison attempted.
Existingcache.extract fails becauseitrequiresnoncancelled live-reusablechain.
Root12functionalCPU controls+57fileclosure/independentreview PASS; genuine
onecardfunctionalprepare16788 PASS withsuccessfulCPUreport/30+39IDs bound.
RESULT -> SerialactualfirstGEN1 fresh1 pin0 finishlength,59inputs/59evaluated,
reused0/generated1/decode0. Genuine committed_live tuple (publishedfalse,
chain_updatedtrue,live_reusablefalse) isexpectedwithcacheOFF; oldcachevalidator
isappropriateforcachequalification butnotpurefreshnumericalreference.
ParentSHA ce8ab07df9a4cb4be2b4f84f0d16a7c82eb92cd2f7d3978146b7b49f20be424f.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native2-serial0-run/parent-qualification.json.
Agentreadonlynumericextract+firstprivatecomparison finds49vectors/fullhead+
48residuals BITWISE; preliminaryonly, remaining3jobs uncaptured. NEWspecific
cacheOFFcollector/mergedProtocol adapter preservesactualtuple andstrictsource/
freshconsumption/spans/data; no falsifiedcacheflag oroldproofwaiver. Review
caughtwrongtwo-FD protocolforfrozenexec2>&1 beforeanynewGPUattempt; fixpending.
Functionalsourceplan1d353f69d619cca48545d9794a1697e502d450d7265fc475b77368ca8c5d5203,
actualgenuineonecardplan prepared; no functionalGPUexecution yet.
VERDICT -> SerialFAIL isreader-scope mismatch, notNN mismatch orhardwarefault;
oldfailedV7retained. FreshremainingserialGPUjobs neednewpurpose-specificparser
beforefull196join. Functionalnativecounterpart readyfornextlease; source/math/
registry frozen, wholemodelquality/completecache/concurrency/latency/shelf open.
Fullgoal remainsactive; unrelatedworktreechanges preserved.


## 2026-10-10 - Native functional PASS and196 numerical pairs; metadata join gap

CONFIG -> Exact source35/C113 original artifact. Onecardmeaningfultwo-prompts/
freshGEN64 observerOFF, fourrepeats; separate cacheOFF GEN1 numerical serial
supplement for4actual consumed-prefix jobs fromnative2diagnosticcollector.
COMMAND -> Functional23905 terminal0/PASS418s; currentreadonlyfinalizedreader
66141 PASS completecommands/currentC113/source4/pages/health/raw4/semantics.
Manualsamecase CPUV3/NativetokenIDs+texts compareall4 EXACT. Serial97749
terminal0/PASS479s, all196freshserial/private comparisons BITWISE; engineowned
normalexit/removal/posthealth/kernel/newfull4/pages/currentSDK gates PASS.
RESULT -> FunctionalparentSHA c2100ece8cc8e7e96bbb3ca4557448bcac7cb06cf42de185a44d145a9aa79fe1.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/functional-screen-v1-onecard-run/parent-qualification.json.
SerialparentSHA ef2f3b99f2554c1e5cc77712ff15af02904eb13362f3419b4783fb8afc21c184.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-serial-cacheoff-v8-onecard-native2-serial0-run/parent-qualification.json.
Functioncode18tokenIDs andarithmetic12/3tokenIDs exactlysameasCPUeachrepeat;
allnaturalstopunder64/freshreuse0. Not fullmodelmath/registeredbroadeval/speed.
Serialreadonlyvalidator32782 thenfailsKeyError plan_sha256: V8producer omitted
childreport planhash although actualexternal/input/childsnapshot bytes+parent
planhash exist. Originalsuccessfulparent/child/data/source remainunchanged.
NEW readonlysupplement mustderiveONLYthatmissinghashfromactualsnapshots in
explicitderivedview, labeloriginalmissingfield andretainall196/source/health/
commands/page/currentstat gates andoriginaltreehash/stat immutability.
NoGPUrerunneededfor metadataassociation. Source758d49 files checkpointedhere;
actualoldV7failedcached-validator run staysFAILED. FouractualnewV8jobs captured.
Pairfunctional81109 genuineCPUprepare PASS; leased38570 nowprehealth/live on
bothB70s afterpriorrun closes. Source-onlyadjudicator/futureV9 producer underway;
no activeV8/V7/functional source orregistry edits. Diagnosticwarmuptiming audit
inNEWdoc confirmslazycapture butnocause/tok/s/optimization claim.
VERDICT -> Meaningfulnativeonecard semantics nowqualifiedwithin twopromptscope;
196numericalmeasurements passed butpublicclosed-evidence joinpendingmetadata
supplement. Pairfunctional runtimepending. Fullmodel/nativearithmetictolerances,
API2/4/bounded6/concurrentcache/staleidentity/memory/profiling/matchedlatency/
verifiedshelf remainrequired. Fullgoal active/unrelateddirtychanges preserved.


## 2026-10-10 - Paired semantic reference and closed196-pair numerical proof PASS

CONFIG -> Exact source35/C113 originalUD-Q4_K_XL; pairedfreshGEN64 semantic
counterpart andcacheOFF4fresh GEN1 originalV8numericaldata, samebackend math.
COMMAND -> Pair38570 terminal0/PASS318s; currentreadonlysemanticreader97627
PASS. All4 nativepairedoutputIDs/texts exactlyCPU V3 andonecard. NEWreadonly
adjudicator14365 terminal0/PASS all196vectors usingexplicitderivedmissing
childSHA; currentsource/SDK/health/modelstats/pages/fullcommands/raw49 and
originaltreehash/stat5 predicates retained. Root9CPU metadata tests+peerreview.
RESULT -> PairedparentSHA c80881eee2a0524b0dc2b43e3f0cbf2a147619f8692c8bbdb058fd6c0ed3dcec.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/functional-screen-v1-pair-run/parent-qualification.json.
ReadonlynumericaladjudicationSHA 0f6128e863f394ba1aa8170150d40b437f2f6033d0f9d6b71dc34a49263f61e1.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-serial-cacheoff-v8-adjudication-v1.json.
Fouractualconsumed-prefix jobs:4full248320heads+192layerresiduals BITWISE with
actualN2privatecollector. Onlymissingmetadatahash derivedfromidenticalactual
external/input/childsnapshots(parentSHAec35f9c56d115b847661defab55884d9c71abc7e5f9daeefd45374d173d177d6).
Originalchildfield remainsabsent; originalsuccessfulparent/child hashes/data
unchanged, no fakecacheflags/noGPUrerun. ActualC0tuplefalse,true,false preserved;
no cachequalification orfullmodel arithmetic transfer. OldV7failedrun preserved.
Future V9producer emitsrealchildsnapshotSHA +actualjob/paircounts, unexecuted.
API2diag genuineCPUprepare63120 PASS; leased46522 nowlive prehealth/collector,
APIbudget64 distinctnative32. Actualcancel/solo-migration/APIboundary/rawjoin/
source4/health qualification stillpending. Allruntime/backend/registryfrozen.
VERDICT -> Declaredlimitedsemanticbaseline nowworks CPU/oneB70/twoB70 with
exactoutputparity, andprivateN2numericalequivalence hasclosedreadonlyproof.
Broadeval/nativefullmath, API2/4/bounded6+concurrent/stalecache/memorylatency,
criticalpath/cleanABBA/P50P95/fairness andverifiedshelf remainrequired. Fullgoal
active; userdirtyworktreeandalloriginalfailed/passingevidence preserved.


## 2026-10-10 - API required capability disabled; explicit newlane andnative4

CONFIG -> OldV7API2 diagnosticpcache0/chain0/public0, genuineINFOslot_cache0;
source35backend/weights unchanged. Separate fullyadmittednative4diag32 profile.
COMMAND -> API46522 terminal1/FAIL504s afterwarm requestsbeforetargetARM;
actualengineexit0/normalremoved/posthealth/kernel/new4/pages/source gates PASS.
Childrequires slot_cache1 forplannedsolo migration butoldprofiledisabledit.
Source+peerreview identify cache3/chain25/public27/freshtrue support; no NNpatch.
Rootgenuinenative4prepare30098 PASS thenleased88372 startsnative4collector.
RESULT -> OldAPIparentSHA 22d7cae5f76efc99d70b6c9716234c092ddb7534fa4c9b48cc423756694f0893.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-api2-diag1-run/parent-qualification.json.
Targetcancel/migration/numerical proofabsent, oldFAILED runpreserved. NEWAPI
proposal enablestruthfulcapability andexplicitfresh throughout forrealowned
slot->main continuation withfullprefixrecompute andstrict0/0/-1 counters.
Migrationlineageevent precedescacheselection; itdoesNOTprove cachedstatecopy.
Positivecachedhandoff remainsseparatefullgoal gate. Proposal mustrejectold
impossibleprofile beforeGPU/hash, emitactualchildplanSHA/schema/copycounts and
admitallstatic/chain/state limits. Registryaliasproposal/coupling assessment
pending; no canonicalglobalregistry/oldsource edits orpolicywaivers.
Native4planSHA 54612361d00df9ee81ba331897199d3bdd50448409e412c66770a64757e72e49.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native4-diag1-prepared/plan.json.
Actual88372 pending GPUcollection/ownedhealth/postsource4 thenfreshserialNN
joins; no4/6/APIcoherencequalified yet. Native2actual196closedreadonlyproof
andCPU/one/pair2promptsemantic parity remainprovedseparately.
VERDICT -> Exactcapability/configuration conflict localized; honest scopednew
APIrecompute lane preparedinparallel while4stream testing advances. Fullmath,
broadquality,4/bounded6/API/concurrentcache/cachedmigration/memory/profiling/
cleanlatency/shelf stillrequired. Goalactive; allolduserdata/evidence preserved.


## 2026-10-10 - Critical-path tracing implementation admitted; native4 raw capture

CONFIG -> Exact source35 current engine and pinned x2 methodology. Separate
new default-OFF host tracer proposal; no active SDK, registry or bin changes.
COMMAND -> Inspect frozen trace proposal, current native4 child report and
poll live parent88372. Run nine cacheoff V9 metadata/adjudication CPU controls.
RESULT -> Native4 child completed all392 raw vectors for four requests and
normal engine exit0/removal; fresh serial comparison is not yet performed.
Parent remains live in final post-health/identity work, so no native4 PASS
claimed. Nine CPU controls PASS. Trace proposal frozen SHA
62eb46304676344334b9deaedc0f9831391d0b6589c0af3729b78229cf926726.
Agent now prepares separate patch36/header/CPU controls/new build plan, with
no extra waits/barriers and OFF original control flow. API source lane and
registry add-only association remain under review; registry unchanged.
VERDICT -> Actual four-stream raw evidence collected, final qualification and
serial parity pending. Profiling moved from source audit to implementation;
no latency, fullmath, broad quality, API/cache4/6 or shelf claim. Goal active.


## 2026-10-10 - Native four-stream collector qualified

CONFIG -> Exact source35/C113 onecard four private native slots, diagnostic32
budget and current original UD-Q4_K_XL; serial numerical parity still separate.
COMMAND -> Poll88372 terminal exit0/PASS477s. Inspect actual finalized parent,
child raw392 vectors, normal engine exit0/removal, posthealth and new full4 scan.
RESULT -> Parent PASS/errors empty, child collection/source lifecycle qualified.
Parent SHA 4e0e5e61e1cd81023d728b53ef275b065634be3e1a150b010c9b84c165b88984.
Path: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native4-diag1-run/parent-qualification.json.
CPU preparation now running for V9 cacheOFF freshGEN1 groups0/1 (eight actual
consumed-prefix jobs, all392 vectors). No serial comparison PASS claimed yet.
VERDICT -> Fourstream raw/current source/identity/health/lifecycle gate passed.
Fresh serial parity, OFF/ON API concurrency, bounded6, cached state handoff,
fullmath/broad quality, criticalpath/matched latency and shelf remain required.
Goal active; canonical registry and consumed SDK unchanged.


## 2026-10-10 - Sixstream and paired private concurrency admitted on CPU

CONFIG -> Exact source35/C113 unchanged; native bounded32 perrequest,
onecard6 and paired2 independent plans. No extra GPU run during serial0.
COMMAND -> Execute V7 prepare with exact v2 case6/2 and actual finalized
native4 post-model identity. Both CPU preparation commands terminal0/PASS.
Run14 new API fresh migration CPU controls; allPASS (source draft not frozen).
RESULT -> Actual new plans:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-onecard-native6-diag1-prepared/plan.json SHA 40602c555819a8a952ab2d48560f45e9cd25cf8a03646b390e32b3b1e1c2f6c1
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-pair-native2-diag1-prepared/plan.json SHA 585db533682848de83b68a770e4d1b5e1d7f0f395636c2d5122b8fa00c870aac
V9 native4 serial0 live78416 under pairlease, currently prehealth/kernel gates.
Serial1 two-job plan already admitted; current serial0 six jobs not completed.
API review confirms in-place alias recipe updates propagate exact cache3 args
and retains strict new profile admission. Additional producer/health/command
joins being hardened; registry remains unchanged. Positive cached handoff and
concurrent prefix reuse remain separate required work after API scheduling.
VERDICT -> Native6 and twoGPU2 ready for coordinated runtime after current
serial comparisons. No native6/pair concurrency numerical PASS or speed claim.
Fullgoal remains active, with fullmath/cache/API/profiling/latency/shelf open.


## 2026-10-10 - API capability-on source lane frozen; 294 native4 pairs matched

CONFIG -> NEW API source35 cache3/fullstatechain/public/fresh1 recompute,
explicit8192/512MiB budgets and unchanged0/0/-1 counters; no cachedhandoff claim.
COMMAND -> Root+independent17CPU controls PASS; final50 SHA/ASCII/AST checks
PASS. Final sourceplan e9c56e0b6dcb82a7a7186de2daeee3cac6dd862e2e82009b7db632cf1bc3ec15.
Review actualproducer/CLI/interpreter/health and prefixjobs/nativehistory joins.
Inspect current V9 native4serial0 child; parent78416 stilllive posthealth.
RESULT -> Serial0 six requests and294 full49 comparisons BITWISE, normal
child teardown. Final parent new4/health stillpending; no complete392 claim.
API source files frozen, canonicalregistry still56a6 unchanged. Add-only alias
proposal and preservedbaseline included; named derivative registry admission
changes only viewdigest and explicitly disclaims oldglobalgate/evidence PASS.
Pairednative4 CPUprepare48882 alsoPASS, no pairednative4 runtime yet.
VERDICT -> Concrete reviewed API lane ready for later registry/CPU/runtime
admission. Positive cachehandoff source route localized but unqualified. Full
native4 final392, native6/pair/API/fullmath/cache/profiling/latency/shelf open.


## 2026-10-10 - Paired bounded6 admission and immutable numerical recollection

CONFIG -> Exact source35/C113 paired32/16 sixslot bounded32 diagnostic.
Separate new readonly recollection outputs outside original V9 run trees.
COMMAND -> CPUprepare47787 terminal0/PASS. Save/compile root readonly reader;
strict V9 validator + complete original-tree fileSHA/stat inventory before/after.
RESULT -> Paired6 plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-v7-pair-native6-diag1-prepared/plan.json
SHA 10869feb927707126fc19e94bb009e089e6532f31019415a8424c9b108793361.
Reader /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/native4-v9-readonly-recollection-v1/reader.py
SHA de13fa7fc7cfe637ee08325dc4b051873bb3c85e2c321fa7663b625a2e192db7.
Reader not executed yet: serial0 parent78416 confirmed live in full4 scan,
child sixrequests/294 comparisons alreadyPASS; serial1 remains unexecuted.
Root identified drafttracer export shortcircuit fclose bug and graph-pointer
reuse stale-generation risk; agent notified for CPU controls before freeze/build.
Positive cached slot restore source route established; sourcecase preparation
ongoing, no cachedhandoff numerical claim. Currentcanonicalregistry unchanged.
VERDICT -> All one/pair2/4/bounded6 native diagnostic plans nowready ormeasured
within declaredscopes; actualremaining runtime+parity and publicAPI/cache,
fullmath, profiling, matchedlatency and shelf remain required. Goalactive.


## 2026-10-10 - Native4 serial0 closed294 proof and authentic cache-positive inputs

CONFIG -> Frozen V9 freshcacheOFF numerical serial0 sixjobs, exactsource35.
Positive cachedhandoff source proposal55files/6CPUcontrols, no runtime claim.
COMMAND -> Parent78416 terminal0/PASS493s; NEWreadonly40252 PASS294 pairs,
originaltreehash/stat unchanged. Lease74445 acquired for remainingtwo serial1.
Root CPU-only tokenizer V1 failed stdlibshadow tokenize.py; preservedreceipt,
normalexit1/removed/noGPU. NEWV2 samecode safe render_cache_fixture.py, exact
c388 image/source tokenizer+frontend+originalpacktokenizer, noGGUF/GPU mounts.
RESULT -> V2 terminal0/PASS normalremoved, no devices/device requests. Genuine
warm inputs231/231 andtarget235/235, seed/tokenizer hashesmatch. Fixture:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/cache-positive-tokenizer-fixture-v2/fixtures.json
SHA 1662f43ee1a5edf380bb1f602b1d2b0e3ca996faa922dc7433d8c1026cf53f42.
Initialcheckpoint nonreuse remainsunqualified; source LCP/boundary checks and
positivecontroller required. Source-only571be0d9d72a4aa3a44ce78a0cc12bfa175bd3cc875c00f31e4fd2910d047444
closure verified55files ASCII/AST/SHA; CPU6PASS. New proposalrequires genuine
last-live restore/donorRID/gen/chain bytes+positive counters andfresh49control;
no recompute/checkpointfallback counted aspositive cachedhandoff.
VERDICT -> Actual294 native4 comparison proofclosed; remaining98 underway.
Authenticpositive cache corpusnowexists, fullstate cachedhandoff runtime still
required. Fullmath/API/4-6concurrency/cache/profiling/latency/shelf goalactive.


## 2026-10-10 - Remaining98 child comparisons and owned HC CPU control

CONFIG -> Exact current native4 serial1 and new independent token-driven HC
block35 arithmetic prototype. Preserve all historical math/helpers/SDK.
COMMAND -> Poll live74445; inspect child PASS98/raw2requests/normalexit.
Read actual HC norm/silu/mix source; run new ownHC9 CPU controls, allPASS.
RESULT -> All392 native4 comparisons nowmeasured BITWISE across294+98,
but remaining98 parent stilllive posthealth/new4 and readonly finalreplay
pending. Update current completion audit with exact incomplete scope.
Nine ownedHC controls establish synthetic FMA/tree/shape/contracts only;
sourceplan/runtime originalpayload control stillpending. Cache tokenizerLCP
and checkpointboundary analysis inprogress; no cachedhandoff runtime proof.
VERDICT -> Concrete numerical data and source-math control advance original
scope; no full392/API/cache/fullmath/latency/shelf qualification inferred.


## 2026-10-10 - Native4 serial1 parent PASS and owned HC control frozen

CONFIG -> Current originalsource35 native4 remainingtwo cacheOFF GEN1 jobs,
separate new independently originaltoken-driven HCblock35 reference control.
COMMAND -> Serial1 parent74445 terminal0/PASS435s, remaining98 BITWISE plus
normalowned/posthealth/kernel/new4/pages. Readonly77049 nowlive final98 proof.
Lease39834 acquired for onecardnative6 diagnostic32/current unchanged SDK.
Root verifies frozenownedHC76files SHA/ASCII/AST and9CPUtests PASS.
RESULT -> OwnedHC sourceplan cdd62f2d8e87d3319673d8c8fbe01802a9632c61f305c2e3c1c27f5156ad9b13.
Originalembedded tokens exclusively drive RMS/projections/selected32columnmix;
no capturedinput/state substitutions. Host rsqrt/exp/div/mixcontraction remain
explicit hypotheses, no deviceintrinsic/fullmath/tolerance PASS. Adapter ready
for originalpayload run after currentGPU arm, with newpostCPUfull4/pages.
Native4 all392 comparisons+bothparents measured; final98 readonly pending.
VERDICT -> Fourstream dataset near finalclosure, bounded6 runtime begins.
WholeHC deviceoracle source preparation next. Fullmodel/API/cache/memory/
profiling/latency/shelf remainrequired; canonicalregistry remainsunchanged.


## 2026-10-10 - Fourstream392 closed numerical proof

CONFIG -> Actual source35 onecard native4, eight exact consumedprefix fresh
cacheOFF serial jobs across groups0/1; bounded32 diagnostic, not publicAPI.
COMMAND -> Readonly serial0 andserial1 validators PASS294/98 respectively,
all original evidence treeSHA/stat5 unchanged. Derive eightjob coverage summary
with no overlap and complete392 full49 comparisons. Current report hasheschecked.
RESULT -> Summary:
/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/native4-v9-readonly-recollection-v1/coverage-summary.json
SHA dedc49903476e32508dc0ccaf65fc6f08f124f31808c685821668d1369ee207f.
Eightjobs/392vectors BITWISE; bothserial parents actualhealth/kernel/new4/pages/
source/ownedteardown PASS. This closes declarednative4 serialnumerical dataset.
Native6 leased39834 remainslive; no6 parent orserial qualification inferred.
Roottracer review requests actualTprintf/flush spans and explicitpreexisting
warmgraph identities. NewwholeHC deviceoracle preparation separates exact
compiledkernel outputs fromshadowintrinsic probes; no deviceexecution yet.
VERDICT -> Native2/4 private numerical coherence nowboundedqualified, not
fullmath/API/concurrentcache/cleanlatency/quality. Fulloriginal goalactive.


## 2026-10-10 - Sixstream588 raw capture completed; collector postgates pending

CONFIG -> Exact source35/C113 onecard native6 diagnostic32, no cache/MTP;
source/registry/weights unchanged. Separate device-free hosttracer CPU fixture.
COMMAND -> Poll39834 confirmedlive; parse actual combinedlog READY1/BDONE12/
SBFrow12/SBFvector588. Actualchild then reports collection/teardown true,
errorNone, normalengineclosure. Parent enters poststrict; full4 stillpending.
RESULT -> All warm andsix target requests emitted real actualdata. Native6
serial12-job comparison and finalparent qualification remainunobserved. Root
reviewpositivecache controller/fixture checkpointdecls beforefreeze, requests
exactsourceboundary bindings and frozenregistry association. Hostcompiler
missing; CPU-only pinned39992 compilercontainer nowauthorized for tracer
fixture with2CPU/2GiB/noswap/networknone/sourceonly/noGPU/noSDK/model mounts.
VERDICT -> Sixstream dataset advances; no source/full588 parity/cleanlatency
qualification claimed before finalgates. Independent originalHC runtime next
once this parent closes; fullmath/API/cache/profiling/shelf stillrequired.


## 2026-10-10 - Sixstream collector PASS; positiveAPI source ready; ownedHC guardFAIL

CONFIG -> Exact source35 sixstream native32 diagnostic; NEW positiveAPI cached
last-live statehandoff; independentoriginalHCblock35 arithmeticprototype.
COMMAND -> Sixparent39834 terminal0/PASS545s actual588raw/fullsource/health/
new4/pages/normalownedexit. Both V9 serial6 plans CPUprepare0/PASS; leased8220
nowruns firstsixjobs. OwnedHC69970 actualCPU FAIL atmetadataepsilon guard;
postCPUnew4/pagesPASS. Provider exactF32 epsilon9.999999974752427e-07 was
compared incorrectlytoPythonF64 literal1e-6; originalmetadata/kernel unchanged.
RESULT -> V1source/runtime preservedfailed; newV2 guard/tests beingprepared.
Root preflight hostdefaultenv also correctlyrejected threadsettingmismatch;
exactdeclared1-threadenv source/host admission PASS. NUM10/currentidentity
admission PASS reads original known-page guards, not computationtensorpayloads.
Earlier diagnostic stdout 'no original payload reads' should be read with this
explicit known-page exception; no originaltensor computation occurred there.
PositiveAPI sourceplan423fa3be817ff3ab27d80f9151a595e44934a1601088ba8f1142dfd543d0d395
68SHA/ASCII/AST +independent10CPU tests PASS. No actualcachedmath/transfer claim;
newcacheOFF schema6 serial counterpart source required before qualification.
Tracer standalonehost CPU9cases PASS; durable identical35-file evidencecopy:
/mnt/vm_8tb/b70/build/hosttrace36-cpu-actual-v1-20261010.
Originaltmp kept; copyreceipt pins allbytes, not a newruntime execution.
VERDICT -> Native6 source/lifecycle gatepassed; 588serialproof stillpending.
PositiveAPI and tracer actualCPU evidence advance source readiness. Fullmath,
publicAPI/cache, profiling/matchedlatency/shelf remainrequired. Registryunchanged.


## 2026-10-10 - Exact F32 epsilon admission repaired in newownedHC V2

CONFIG -> NEW V2 token-driven originalHCblock35 control, exact original GGUF
FLOAT32 epsilon9.999999974752427e-07/LEbytesbd378635. V1 remainsfrozenfailed.
COMMAND -> Sourceplan427a44571c4573253d573be61ce0d9685b4dbd131e8f8df85cf3d6bf36034cdf
82SHA/ASCII/AST closure PASS. Root+agent6newCPU tests PASS: actualsavedmetadata
ctoracceptance withouttensorreads, +/-1ULP/F64/type/shape/digestfail before
sourcebinding/rows. V1mathfunctionAST unchanged. Source/host/CLIadmissionPASS.
RESULT -> Control60346496.../adapter562449a6.../receipt9723d160... frozen.
ActualV2runtime pending untilcurrentGPU armterminal; original69970failedreport
andnew4/page sourceproof pinned innewclosure. Sixserial0 childhas294 full49
BITWISE comparisons/normalteardown; leased8220 live finalposthealth/new4.
Remaining sixserial1 planalreadyadmitted. No full588 serialparity yet.
VERDICT -> Avoidable metadatafailure nowcaughtcheaply withreal sourceencoded
value, no tolerancewaiver orNN/modelchange. Tracercollector token/print/flush
coverage reviewadvances. FullAPI/cache/fidelity/profiling/latency/shelf open.


## 2026-10-10 - Firstsix native6 serial parent PASS; correctedownedHC V2 starts

CONFIG -> Frozen source35 native6 firstsix freshcacheOFF GEN1 jobs; separate
V2 exact-F32 originalweight HCblock35 control in declared one-thread CPU env.
COMMAND -> Serial0 parent8220 terminal0/PASS430s, actual294BITWISE comparisons,
normalownedexit/posthealth/kernel/new4/pages. Independent readonly25757 starts
withcomplete originaltree inventory/currentV9 finalproof. CPU V2 payload35758
starts NEWoutput owned-hc35-f32-block-prefix4-v2; noGPU running concurrently.
RESULT -> Remainingnative6 serial1 sixjobs alreadyprepared, unexecuted. Root
newpositive cacheOFF counterpart9CPU controls PASS; sourceclosure/reviewpending.
Existing native6 readonly script saved outsideoriginalruns withidentical generic
reader bytes andcopyreceipt under native6-v9-readonly-recollection-v1.
VERDICT -> Actual294 serial6 vectorpairs havecompleteparent gates, independent
readonly proofpending. Correctedoriginal arithmetic nowexecutes; no native
intrinsic/fullmodel/tolerance PASS before realresults. Fullgoal remainsactive.


## 2026-10-10 - Independent original HC rounding seam reproduced; serialpositive ready

CONFIG -> Exact originalweight token-driven V2 HCblock35, observed prefix4row3.
COMMAND -> Root35758 terminal0/errors[], originalsource/host/NEWpostCPU4/pages
PASS. Report6634cea052a0b5b9942a82a6f1b1dfec788f20d62b42faf5b6c206837790b495.
RESULT -> Ownnormalized10240+inject4 BITWISE native. LowNMSE1.53e-17 and
selectedgate128NMSE5.69e-16 nonbitwise. Separate32mixNMSE4.715e-15/oneQ81code
71vs72. Preregisteredfused32mix+entire36Bpacket BITWISE native, no capturedinput
orstate used. Not devicecontract/fullmodel/tolerance qualification. Curated
result docs/20261010_flashnext_owned_hc_f32_block35_result_v2.md.
Native6 serial1 leased34910 nowlive posthealth/new4; source12job588 closure
stillpending. Positive serialV10 CURRENTfb7e869c304bc0d8ff24015a662e612a9127b41e2f08c6a71e855472a0bc64ce
92SHA/ASCII/AST +independent11CPU PASS, exactproducerPID/CLI/snapshots/full49/
allgroups+actualsolorole. Originalnumericalproducer/extractor/math unchanged;
actualpositiveAPI and freshcounterpart stillunexecuted. TracerV2 CPU9PASS,
durable identical35files+buildsnapshot copied; nativepatch/buildplan pending.
VERDICT -> Actualoriginal arithmetic evidence explains localized threshold,
not wholemodel fidelity. Fullcache shared/divergent/stale/memory sourcework
assigned; registry willremainunchanged until native6 closedproofs complete.


## 2026-10-10 - All588 sixstream pairs closed; positive cache API runtime admitted

CONFIG -> Source35 native6 bothfresh serialgroups, twelvejobs/all588 comparisons.
NEW genuinepositiveAPI2 cache3/chain/public/freshfalse target, exactsource/state
budget/CPUfixture andcurrentexperimentalregistry; primaryhotschmoe-dd retained.
COMMAND -> Serial1 parent34910 terminal0/PASS437s; readonly34369 PASS294,
originaltreeunchanged. Bothgroups nowindependentPASS294+294/all12jobcoverage.
Summary /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/native6-v9-readonly-recollection-v1/coverage-summary.json
SHA 0dba88609f5eea279eae04989ecc962238ff92d042a9432540f7d24749c72db7.
Append exactpreserved56a6 registrybytes +reviewedpositive12 only, YAML parsed,
alloldrecords/defaults byte/semantic unchanged; fresh12 remainunregistered.
RESULT -> Actualcurrentregistry SHA e0abd69b8b991793cc2c2b78d0522aaec6c185a362309ed47982713f66b4dd28.
Older strictV7 current-globalSHA gate nowfalse; namedentry/source association
retainsoldbytes, doesNOTreportoriginalvalidator/rawrequalification PASS.
Genuinepositiveprepare91156 terminal0/PASS; plan /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/batch-api-positive-v2-onecard2-diag1-prepared/plan.json
SHA c8a3b47666fc3bdad73630297f5c464c1286b110c7a461a1e636610e49a22c42.
Leased10435 nowlive APIpositivecollector; no cachedtransfer/math/latency claim.
Pairedold-registry unexecutedplans preserved; newgenuineplans canpinnewdigest.
VERDICT -> Native2/4/6 private numericaldatasets closedwithin boundedonecard
scope. FulltwoGPUconcurrency/API/cache/stale/memory/fidelity/profiling/latency/
shelf required. RealpositiveAPI nowadvances nextgate. Alluserchanges preserved.


## 2026-10-10 - Bounded defaultOFF host tracer source frozen

CONFIG -> New36 onexact35, onlygenerate/verify hosthooks +testedtypedV2header;
ONfirstonecardnative serial, max4requests/2048input/64new/32768events, no new
SYCLwait/queue/graphnode ormodelmath changes. OFFoldmodes retained.
COMMAND -> Root14source/preserved/buildplan SHA +basefiles PASS; actualCPU9
C++controls androot10source/collector tests PASS. Engineplanc2b80b7e...;
sourceplanee895bd90c2c70c0176acc77c9e1c1110ee2707c75150a03ac046e49685c7d82.
RESULT -> All64reconstructed sources/36patches/28addedheaders/8freshABI/6Python
readyforrootnewbuild, no nativecompile/modeltrace yet. DurableV2receipt binds
all35files/sourceSnapshot/binary; V1lifetime evidence retained ashistory.
Newpaired2 plan69519 genuinelyCPUpreparedagainstactuale0abd registrydigest.
WholeHC synthetic4cases/36frames/19fields3802788words inputexport8380PASS;
independentfinalsource reviewpending. PositiveAPI10435 remainslive, actual
BCHAINstartup slots2/cap3/point236159904/required4250878272/chain8589934592/
transfer536870912 admitted; warmtransfer/numerical proof stillpending.
VERDICT -> Concrete hosttracing implementation readyforfresh ABI andexact
OFF/ON requalification; GPUtest unchanged currentlyleasedpositiveAPI. Full
originalgoal active; no performance/shelf orfullcache/fullmath claim.


## 2026-10-10 - PositiveAPI warm transport completed; native terminal guard failed

CONFIG -> ExecutedpositiveV2 onecard API2, fullchain/public/cache3, authentic
231token warm corpus, UR2 completeAPI/native diagnostic line recording.
COMMAND -> Poll10435 confirmedlive posthealth; inspectsealedchild+warmclient
andstreamfilteractualAPI event log. Newpaired4 prepare88681 terminal0/PASS.
Root wholeHC lifecycle7/source18 CPU checks passed; independentreview found
journalreceipt/logSHA andleaf-logSHA closure gaps; NEWV2 fixesbeforecompile.
RESULT -> Warmboth HTTP200/[DONE]/finishstop and21 visibletokens, no client
error/cancel; bothactualengine_end GeneratorExit/error 'Noncancelled consumer
close without pinned EOS native stop'. ChildFAIL Warm API/native terminals
incomplete before ARM/targets; parenthealth/new4 stillpending. Sourcefrontend
breaks onstopID beforegenerator resumes BDONE, so rawowner/terminal association
mustbe investigated before assigning backendmath/transport fault. Oldsource/
measurement unchanged. Sourceaudit actualGEN->STOP->BGEN promotion re-evaluates
prefix underfresh1; long231 enters differentprefill path than prior59 corpus.
V2 observer opens/appends trace+mergedlog perURline; millionsoflines canback-
pressure producer. NEW bufferedV3+semanticflush/owneddrain/CPUequivalence being
prepared; no matchedtimings orspeedup established. No change tocurrentrun.
VERDICT -> Actualpublicwarm replies observed, positivecachedhandoff untested.
Fullgoal active; failednativeboundary kept, independentdeviceoracle closure
fixes anddev-loop traceIO implementation progressinparallel. Noshelf/speedclaim.


## 2026-10-10 - Nativepaired observer crash and wholeHC runtime V3 ready

CONFIG -> Source35 paired32/16 native2diagnostic targetARM. Separate compiled
syntheticwholeHC oracle withV3 strictsupervision; no model/weight changes.
COMMAND -> Pair66128 terminal1/FAIL361s, childexit139/OOMfalse afterwarm and
at targetcapture; posthealth/kernel/new4/ownedremoval PASS. Trace exception:
batch observer graph seal/stamp mismatch. Sourceaudit early-stage returnat
verify1637..1642 skips sole snapshotstamp1741; sealer1856 rejectsunstampedstage0.
RESULT -> NEW guardedobserverstamp37 prepared CPU10; existingmodel/head/OFF
ops unchanged, freshABI/twoGPUrequalification required. H36+37 newbuildoption.
WholeHC V1journal/logSHA gaps fixedinV2; V2launchcomp1/ownershipcomp2 mismatch
caughtbeforeexecution. V3 consistentlycomp3 +actualargv→ownership CPUcontrol,
alljournal/log/currentproof gates retained. Sourceplanf33fe98f106fileclosure,
18original+8journal+10V3CPU tests PASS, independentreviewREADY. CPP/fixture/
compileplan unchanged. Preliminaryrootleaf compile86242 PASS22s undercard0,
sourcefullySHAfrozen butrepo checkpointwaspending; preserveprelimreceipt and
performfresh tracked-source linkafterthischeckpoint beforeactualdeviceoracle.
API10435 failedwarmguard FAIL1449s; actual HTTP/nativestop21token evidence
provesstale engine_last observerpredicate, no rawrewrite; warmalsoZERO2row
body, so sourcefixture/overlapneeds separatedeclaredrepair. BufferedV3 work
retains exactrawschema/flush proof andseparatetruthfulterminal association.
VERDICT -> TwoGPUstate/math unqualified; fixeslocalizedtoobservers/admission.
WholeHC runtimecanproceedafterfreshlink; allfullAPI/cache/fidelity/profiling/
latency/shelf requirements remainactive. Allfailed/prototype evidence preserved.


## 2026-10-10 - Earlier-stage batch observer stamp repair and wholeHC device run

CONFIG -> NEW metadata-only stamp37 afterexistingthree earlystage handoffcopies,
beforele<n_layers return; oldhead stamp/modelops/defaultOFF unchanged. Separate
wholeHC source35 compiledactual/shadow arithmetic undercorrectedV3supervisor.
COMMAND -> Root10CPU stampcontrols/11SHAclosure PASS. Standalone37 plan2cb8c255
andcombined36+37 plan2e940d51 frozen, all8 freshABI andsameNNmodel required.
Root tracked-source freshleaf23651 PASS22s afterd1db0a9 checkpoint; preliminary
untrackedintake build86242 retained. Runtime48761 pairleased/currentcard0pin
nowlive, usesnewtracked-source receipt, 36frames/19fields andstrictV3ownerlabel.
RESULT -> Pair35failed66128 retainsEOF/exit139/observerseal exception and
posthealth/new4PASS. Real twoGPUcoherence stillunqualified, newC137 intake
beingprepared beforepair2OFF/ON. PositiveAPIfailed10435 warmnative21EOSs have
producer-owned stopBDONE beforeengine_end, staleengine_last causedV2guard
rejection; no rewrite/pass transfer. Warmactual38rowevents ALLrows1/zero2row,
so honestnewobserver alonecannotgrantrequiredwarmoverlap. BufferedV3 and
separateterminalsource admission +newcase planning ongoing; targetsunexecuted.
VERDICT -> Specificobserverbug repairedinsource; needsfreshbuild/runtime.
ActualwholeHCdevice qualification nowprogresses. Fullmodel/API/cache/profiling/
latency/shelf goalactive, no productionor speed claim. Allolder evidence kept.


## 2026-10-10 - ActualwholeHC36frame proof and freshcombined37 ABI build PASS

CONFIG -> Source35 actualcompiled syntheticHC36frames/19fields3802788F32,
separate knownURmetadata log format adjudication; newcombined36+37engine.
COMMAND -> WholeHC48761 terminal0/PASS218s allnumerical/lifecycle/health/new4;
originalpublic79819 failedonly capabilityline fragmentedbyactualURquery.
NEWreadonly84261 PASS preservesexactoriginaltreeSHA/stat5 andallfrozenV3
raw/current/journal/owned/post4 predicates, acceptsONLY uniqueactualknownUR
insertion. Originalreader/run/source untouched. Adjudication111closure/CPU6.
RESULT -> All36 selfconsistency/replay/alias/write framesPASS;18analyticbothmix
variantsmatch,18heldoutonlyfusedmatch; allshadowgradualFMAprobes exact. Capability
observations donotqualifygeneralFPmode ororiginalwholemodel. Newcombined SDK
77196terminal0/PASS308s,8freshELFs/sourceclean/pin/snapshot unchanged:
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T084541Z-zqrph_y1.
ActualC137combined_generation_gate PASS64/28/37/8/6. Root90baselineCPUtests
PASS (60+30); additional10newcontrols agentPASS. Newupload37 recipe ready;
actual390/normalownedhealth/new4 andC137 serving notexecuted. Inheritedjournal
association gaps mustbe fixedinnewbaselineparent beforequalification.
VERDICT -> Concrete currentcompiled HC arithmetic evidence +deliberatefresh
ABI checkpoint. Fullmodel fidelity/privatepair/API/cache/profiling/latency/
shelf goalactive. No capture input/state enters independentoriginalreference.


## 2026-10-10 - Source37 baseline andsource390 intake tracked beforedeviceuse

CONFIG -> Freshcombined37 SDK084541Z-zqrph_y1, receiptSHA d39c75cff27ba53309384f4369c15f9046903c5649769ba7fa4f9050165c9dfd.
COMMAND -> ActualC137sourcegate andNEWupload37 metadataadmission PASS64/28/37/
8/6. Root90inherited baselineCPUcontrols PASS; agentadditional10PASS. Source
andparent localSHA closures PASS. RootNEWupload4CPU controls PASS. TrackNEW
modelsource intake/controllers/recipe beforeactualoracle build/runtime.
RESULT -> Existingfrozen137parent isprototypependingNEW137v2 journal command/
rc/error/logSHA chronology hardening; no modelserving/upload qualificationyet.
WholeHC84261 readonlyadjudication PASS111closure/6CPU/originaltreeunchanged;
all36 actualcomponent frames pass, device/universal/fullmodelflags false.
VERDICT -> FreshABI andboundsourceintake readyfornew390 upload/requalification.
FulltwoGPU/API/cache/fidelity/profiling/latency/shelf goal remainsactive; no
priorC113 model/success orABI artifacts transferred toNEWgeneration.


## 2026-10-10 - New37 selected390 upload and authenticshared272 prefix PASS

CONFIG -> Fresh source37 SDK/oracle eu0sgjai, NEW boundHC/PLE390 sourcebytes;
separate17-row readonly CPUtokenizer fixture, noGPU/GGUF mounts.
COMMAND -> Oracle21507 terminal0/PASS48s, explicitnew37compile admissionPASS;
pairleasedupload72133 terminal0/PASS317s. ReceiptSHA0b8a1e8169d8ca543c7e48d417bd26361742dc2264019f5e3e60af9570037eba.
CPUfixture6683 terminal0/PASS ownednormalexit/removal/tokenizer/source/image
hashes; authenticsharedboundary272tokens and17rows/7phases. FixtureSHA
93df9a5d46aa4332f080a84cdc326cb67e0ed6486b1d29b122260105b72e9efd.
RESULT -> Source37 selectedHC387/PLE3 upload/readback pluscard0/card1/24-24
ownership/staticbounds/health/new4/sourcepage/lifecycle gates passed. It does
NOTqualify ordinarypayload/fullmath/newpairedbatch/rootcache/serving. Shared
fixture creates realtokens only, not cachedreuse/runtime. Fullcache V2 source85/
CPU12 retainsall shared/divergent/cancel/stale/victim/memory requirements.
StaleINFO public_pin_fresh unsupported literal isNOTprotocolauthority: actual
source implementsfresh+pin, combination runtimeunqualified; conservativeV2
case excludesitbyCASEscope, not falsebackendunsupportedclaim. C137V2chrono
crossjoin gapcaughtbeforebaseline, new1373 parent/source19CPU controls ready
independentreview. No actualbaseline serve untilstrongparentreviewcomplete.
VERDICT -> NewABI deliberately requalifiedonselectedsource bytes; genuine
sharedprefix input foundation improvescachetests. Fulloriginalgoal active,
metadata/prototypefailures kept, no speed/shelf/cache/fullmodel promotion.


## 2026-10-10 - Strong C137parent1373 ready; semanticregistry bug blockedbeforewrite

CONFIG -> NEW C137parentV3 exactjournal/health/identity/snapshot crossjoin,
samefresh37 SDK/controller/source/recipe; no old1371/1372prooftransfer.
COMMAND -> Root+independent19CPU controls and12SHA/ASCII/AST closure PASS,
sourceplan cdf2a877c3d2d3c3e9b3d7a74b08a36c683ade49907762860d194099e4f258ce.
RootYAMLsemantic registrationpreflight rejects proposedbatch delta2acfc1:
appended top-level models key replacesold149+baseline4 list with12 inparse.
RESULT -> Exactcanonical E0abd unchanged, no registrationwrite. Source-only
SHA/byte preservation tests missed YAML semantics; NEW listentries-onlyV2
proposal +duplicate-key-reject/alloldrows+topkeys tests beingprepared. CPU
prepare32468 inadvertentlystartedafterfailedfirstshellstatement; rootstopped
only ownedverifiedPID865040 SIGINT, terminal130 duringhash, noGPU/modelserve.
Partialc137-onecard-segmented retained+abortreceipt; nextgenuineprepare newdir.
VERDICT -> Actualcaughtsemantic configurationbug; userregistry/weights untouched.
Strongbaseline supervisor nowreadyafter correctedregistration. Freshupload390
andrealshared272token fixture remainvalid; fulloriginalgoal active.


## 2026-10-10 - Semantic-safe C137/batch37 identities registered; genuineprepare

CONFIG -> ExactE0abd149existingmodels +NEWbaseline4fragment08043ca +batch12
ded5e8; no duplicate YAML keys; alloldentries/order/topkeys/values preserved.
COMMAND -> RootstrictYAML prospectivefile associationPASS beforecanonicalwrite;
CPU4 semantic tests/123SHAASCIIAST closure PASS, plan94520cf69c9428dd55e3ecef079f1cb92841baf977dcb24ddbc205cc2de0c4a3.
RESULT -> Canonical165rows, actualSHA86621c71b829c75cc2f7312248932e9a8bb276018db3c0bef77f1fdb58678052.
Oldfull-document proposals+weakassoc remainhistorical, no oldwholeglobalgate
successclaim. Genuineonecard C137prepare68050 nowlive withfresh37SDK/new390,
currentruntime/canonicalaliases/newdirectory/port18337, originalfull4scan.
ActualV2shared fixture91801 PASS19rows/7phases/shared272; V2casegenerated after
strictproducer/seed/image/terminal/tokenizer admission, runtimeunqualified.
EarlierV1fixture→V2case refusal retained asvalid provenancegate, notmodelerror.
VERDICT -> Correctedconfiguration concreteandreviewable; nextstrong1373baseline
GPUserve afterpreparationterminal. FulltwoGPU/API/cache/fidelity/profiling/
latency/shelf goal remainsactive; unrelateduserchanges preserved.


## 2026-10-10 - GenuineC137 onecard preparation andstrong1373 serving start

CONFIG -> Freshcombined36+37 SDK eightnewABI +source390 qualifiedneworacle;
exactoriginalUDQ4XL/originaltokenizer/template/runtimec388, onecardsegmented
2048/64/PLE65536/defaulttracerOFF/static/noBorrow/noMTP.
COMMAND -> Metadata/full4prepare68050 terminal0/PASS launch_allowedtrue;
preparedSHA a867bfe779bc685f1f8d13c86ed6e28c7fe3dbf00f09157d09bb5a390cebee9d.
Root19strongparentcontrols+independentreviewREADY cdf2a877 sourceplan. Start
qualify_c1_serving_combined_v137_v3 underpairlease90362, source/profile unchanged.
RESULT -> Actualbaseline90362 live, no C137GPUserve/speed/quality/pairedbatch
PASSyet. Existingpartialunregisteredprepare32468 retainedabortedCPUonly.
Bulkhost6sourcecontrols PASS but compile/runtime/original48refinement still
pending. Source37full390+authenticshared19rows272prefix remainseparateproofs.
VERDICT -> Concrete freshservingrequalification nowexecutingafterall newsource
prerequisites. FulltwoGPU/nativeOFFON/API/cache/fidelity/profiling/latency/shelf
stillrequired; oldbaselineABI/proof not borrowed. Alluserchanges preserved.


## 2026-10-10 - Source37 onecard screen passed; final parent receipt failed

CONFIG -> Genuine C137 strongparent1373, fresh combined36+37 SDK, prepared
onecard-segmented-v2 at port18337; original pinned UD-Q4_K_XL and tracerOFF.
COMMAND -> Actual leased session90362 terminalexit1 after501s. Read original
parent, controller qualification, screen, identity proof and engine log. Root
source38 CPU18 and62 source-file SHA checks PASS, plan def9ec0046a5e2b760f5296e34bdfd06b999526fee7d2a62755913ad20713ed4.
RESULT -> Screen identity/token consumption/coherence/repeat PASS; controller
bounded qualification PASS; launch supervisor0, owned normal terminal, pre/post
health, kernel journal, newcomplete4 and knownpages PASS. Final parent1373
passed=false with post_error KeyError started_epoch during proof binding.
Original failed artifacts preserved; independent source/receipt investigation
requested before any rerun. No final C137 receipt or pair prerequisite PASS.
VERDICT -> Actual bounded serving evidence advances current-source baseline,
but receipt binding remains unqualified. Full fidelity, pair/API concurrency,
complete prefix cache, profiling, matched latency and shelf goal stays active.


## 2026-10-10 - Actual receipt localization and current completion audit

CONFIG -> Read-only actual source37 C137 artifacts; CPU-only source38 and
buffered API V4 closures; full original campaign scope unchanged.
COMMAND -> Independently verify source38 CPU18/62 SHA and bufferedV4 CPU17/135
SHA/ASCII/AST; current C137 SDK source gate PASS. Inspect actual parent/proof
initial shape and update completion audit from closed runtime evidence.
RESULT -> Parent omits started_epoch while proof retains it; exact helper
parent crossjoin raises KeyError. Independent buffered review found missing
postjournal-before-identity and prejournal-before-launch chronology joins,
so NEW V5 successor requested before device use. Frozen artifacts unchanged.
Bulk CPU helper resource/observed-container bounds are being finalized.
VERDICT -> Evidence changes next actions: strict read-only baseline validation
and future producer regression before pair; no unnecessary GPU restart. Full
original48/API/cache/profiling/latency/shelf requirements remain open.


## 2026-10-10 - Buffered API V5 actual producer chronology source checkpoint

CONFIG -> NEW V5 buffered observer parent/reader; frozenV3/V4, native model
math and raw failed API10435 evidence unchanged. No device execution.
COMMAND -> Independent/root terminal8+journal11+adapter4 CPU controls PASS;
144 SHA/ASCII/AST exact closure, sourceplan1c00933d11a82a340d7fdfd5d5ffe58b20e9c645ab35ca75f3781e3394087e75.
Production command function tested with mock Popen, matching actual row schema.
RESULT -> Actual command start/end and launch epoch are recorded; prejournal
finishes before launch, postjournal before actual identity.started. Original
health/EOF/source/terminal predicates retained. Current longwarm case has no
observed two-row warm event; new shortwarm authentic fixture remains needed.
Root source35 code confirms standalone and pending HC writes use explicit FMA;
original48 refinement must cover writes as well as RMS/projections/fusedmix.
VERDICT -> Concrete source hardening before GPU use, no measured IO speedup
or cache/model/runtime qualification. Original full campaign remains active.


## 2026-10-10 - Bounded bulk HC host helper tracked before fresh compile

CONFIG -> Ordinary CPU-only scalar/bulk FMA helper, frozen scalar body included
verbatim, no SDK/device/model inputs. Exact2CPU/2GiB/no-network compile recipe
and observed image/mount/device/resource checks, current host library binding.
COMMAND -> Agent/root nine CPU controls and20 SHA/ASCII/AST closure PASS;
sourceplan c752635b2cd3d653eae2a43c515569ba757a04c6551c8cc7b00b02fd75461a1e.
Root read-only actual C137 journal79active+pre/post and strict/compiled health
crossjoins PASS; missing parent start field remains independently unqualified.
RESULT -> Bulk source ready for root fresh compile, fifteen scalar known
answers and nine bulk fixtures plus each row's actual previously qualified
scalar ELF equivalence required. No actual helper runtime or model math yet.
VERDICT -> Enables deliberate original48 arithmetic refinement after actual
qualification; source tests do not substitute for compiler/runtime evidence.
Full fidelity/cache/concurrency/profiling/latency/shelf goal stays active.


## 2026-10-10 - Fresh CPU bulk HC helper compile and actual equivalence PASS

CONFIG -> Tracked0630fce sourceplan c752635b, pinned ordinary compiler image,
no devices/model/SDK mounts, observed2CPU/2GiB/no-network container; original
qualified scalar executable retained as actual same-operand comparator.
COMMAND -> Root fresh compile/qualification14222 terminal0/PASS. Newreadonly
finalized_binding independently recollects preserved operands/commands/outputs
and current source/build/host runtime, PASS. Root build hc35-host-bulk-fma-v1-
20261010; reportSHA9fb2fcb0c7651fa8f62df7638c9e79beecf7f5672e10e7e40666f7c8c9de184f.
RESULT -> Fifteen scalar known answers, nine bulk fixtures and17 actual
qualified-scalar equivalence rows BITWISE. CompileSHA082c3d6f8f2e1f9ae18e8178bfbc814d90b577db0764765ac79f03cd5384caee;
helperSHA1aea3a954d2a7dd0b90cecbe5d7f906ad2cecf37858737480ebf5ba1140391d2.
Normal owned exit0/noOOM/removal and identical pre/post hostruntime gates PASS.
VERDICT -> Actual CPU arithmetic prerequisite established for efficient
original48 HC-only refinement. Device intrinsic/fullmodel math flags false;
no measured dev-loop or serving speed claim. Original full campaign active.


## 2026-10-10 - Genuine source37 two-card preparation started

CONFIG -> Same fresh combined36+37 SDK/source390/runtimec388, registered
two-card-segmented profile, new directory and port18338; CPU-only preparation.
COMMAND -> Strict current165-entry YAML/alias/hash preflight PASS. Root starts
prepare21840 with --verify-model-shards; polls same handle and confirms live.
Directory f17-source37-20261010/c137-two-card-segmented-prepared-v1.
RESULT -> Complete4 scan in progress; no preparation or pairserving PASS yet.
Independent actualone1373 adjudicator draft found no further evidence blocker,
but CPU/source freeze and actualreadonly validation still pending. Future
parent must explicitly accept bound adjudicatedone without rewriting failure.
VERDICT -> Advance pair prerequisites while CPU-only source work continues;
no GPU launch before correctedparent and actualone evidence admission. Full
original fidelity/cache/concurrency/profiling/latency/shelf goal unchanged.


## 2026-10-10 - Pair preparation and authentic shortwarm PASS; original48 starts

CONFIG -> Fresh source37 pair baseline preparation; separate source35 CPU
shortwarm fixture; independent original48 HC-only prefix4 refinement.
COMMAND -> Pairprepare21840 terminal0/PASS launch_allowedtrue, preparedSHA
8aa7a4a15d1eca94edf07722678cdae6dbf807f7469844041b1d342f031afbaf.
Shortwarm93054 terminal0/PASS and freshreadonlyadmission PASS; genuinecase
generated port18339. Independent/root7 HC-refinement CPU and94SHAASCIIAST
PASS, sourceplan6f3e29e263d58636b439d0abc3703846645bd2d03122a44b440c7fa8b7b5c33b;
trackedf1c58d7 beforeactual root CPUoriginal48 session51315 started.
RESULT -> Original47/47 warm and exactunchanged235/235 target tokens; fixture
SHAb05d5528afa5b1103eef3180a2569790292563854f841c6f91f6ad63af0e4d52, receipt
ad0fac499906031d87a56b356c82afaf0dc28b7fcaf658aff6cfb1b9c48a3942. Normal
owned CPUcontainer removal and source/tokenizer checks PASS. PairGPUserve
pending strictone/newparent admission. Original48 HC exploration nowlive;
no numerical outcome/tolerance claimed. Capturednative data comparisononly.
VERDICT -> Concrete prerequisites closed and fullmodel fidelity experiment
advancing. Actualshortwarm2row/cachehandoff/completeAPI andcleanlatency/shelf
remain unqualified. Full original goal active, unrelateduserchanges preserved.


## 2026-10-10 - Strict actual90362 receipt adjudication PASS; pair1374 live

CONFIG -> Exact failedactual90362 source37 onecard artifacts; NEW read-only
adjudicator, original files and failed parent preserved. Futureparent1374
records producer started_epoch and retains strong source/health chronology.
COMMAND -> Root/independent11adjud+21parent CPU and8/14 SHAASCIIAST closures
PASS; tracked48763a8 beforeactualadjudication29891 terminal0/PASS. Explicit
baseline consumer reexecutes actualstrictbinding, PASS, savedadjSHA
3842d6e9eb5f811e83db93e17a02bd80d2530c2ab14ebcf4b9115009b9f3485e.
RESULT -> All383 originalfiles SHA/stat5 unchanged; original_parent_passed
false/error retained. Namedmemoryview supplies only omittedstart fromactual
proof, allotherstrictV3 source/full4/pages/79activejournals/health/ownednormal/
bounded6screens gates recollected. No fullmodel/cache/concurrency/latency PASS.
Root starts pair1374 leased73796 atport18338 with exactnewadjud prerequisite
and genuinesameSDK prepare. Original48HC prefix4 actual51315 stilllive.
VERDICT -> Validated boundedone evidence without GPU rerun; freshpairbaseline
nowexecuting with correctedproducer. Full goal intact, no shelf/speed claim.


## 2026-10-10 - Original48 HC prefix4 closed without fidelity improvement

CONFIG -> NEW independent HC-only FMA candidate; original source35 P30V4
comparison/nativecapturesTARGETONLY. Original GDN/QSA/FFN/PLE/head retained.
COMMAND -> Actual51315 terminal0/exploratory complete, noerrors/new4/pages.
Rootreadonly12605 recomputes577 saved metrics andpostidentity, originaltree
SHA/stat5 unchanged. ReportSHAdce363ee8e533595ba42733bbc0e55c423fdd85f1e5a24954d78404fc4f5b27e.
RESULT -> HeadNMSE0.01330947575 versus original0.00932149853; firstbitwise
difference p0_l0_attention NMSE3.9609885e-7. Candidate doesnot improve fullhead
agreement. No tolerance/numericPASS assigned; hostexpf/rsqrt remainunqualified.
Curated result docs/20261010_flashnext_original48_hc_fma_prefix4_result_v1.md.
V6 shortwarm source10CPU/158SHA closure PASS/tracked9470172; docCLIcasepath
erratum: actualcase OUTSIDEfixture at f16/api-shortwarm-authentic-case-v1.json.
Actualpair73796 screencoherence/repeat PASS, ownednormal/posthealth PASS;
parent finalfull4 stilllive, no completepairbaseline PASS yet.
VERDICT -> Important negative fidelity evidence narrows nextlocalization;
fullgoal remainsactive. No rawfailure rewritten or syntheticPASS promoted.


## 2026-10-10 - Fresh pair1374 PASS and shortwarm API diagnostic started

CONFIG -> Actualnewsource37 pair baseline, strictreadonlyadjudicatedone
prerequisite; separate source35 bufferedshortwarm API case usesactual47/235 IDs.
COMMAND -> Pair73796 terminal0/PASS419s; root currentfinalizedbinding PASS.
parentSHA70a131195dc47b0e7936d377e631c015c446f4f4d7177cd5afb1cd2e649a767a,
qualificationb9e318d5f772a436fd340b8d73525e16cce8ba3b3f2b8a1304992ddc471eb9aa.
V6CPUprepare66271 terminal0/PASS usingactualcase/currentfull4; GPUparent69544
started underpairlease, real warm2row/nativecachedtarget gates unchanged.
RESULT -> Freshpairbounded identity/coherence/repeat/normalteardown/health/
journal/new4/pages complete, fullfidelity/concurrency/shelf flagsfalse. New
firstHC source7CPU/100closure PASS but conditionalGDN would execute AFTER
earlyterminal stamp; rootblocked runtime andrequestedNEWV2 producerfix.
VERDICT -> Actualpairprerequisite closed; advanceAPIcache andfirstHC fidelity
without weakening chronology or fullgoal. No speed/fullcache/modelclaim.


## 2026-10-10 - First HC scale boundary found; shortwarm API missing cancel

CONFIG -> Independently owned firstHC V2 +explicitconditional GDN; separate
source35 API V6 target235/warm47. Source37 pair2 OFF/ON currentSDK40 controls.
COMMAND -> FirstHC73859 terminal0/exploratory/errors[]/new4/pages PASS. API69544
terminal1/FAIL674s but ownednormal/noForcedCleanup/health/journal/new4 PASS.
Root current40 CPU18/66SHA closure PASS, source574bf88 tracked; actualOFF
prepare20908 PASS andleasedOFF78197 starts; ONCPUprepare22930 live.
RESULT -> FirstHCprior normalized andfullQ81 BITWISEnative. Candidate differs
one scale-D byte block59 acrosshalfmidpoint, codes/S unchanged; conditional
GDN NMSE6.00347e-15. Curated docs/20261010_flashnext_first_hc_scale_boundary_result_v2.md.
APIwarm sevenreal completed2row events passed; both targets returnedOne+EOS
and stoppedbeforeBGEN. No target2row, cancellationgate correctlystayedclosed.
Underlyingcause ofearlyEOS UNDETERMINED pendingmatchedsame235 fresh control;
no promptsubstitution or fixture/model/state blame. OriginalfailedAPI intact.
VERDICT -> Concrete arithmetic/scheduling evidence advancesfullgoal; no
fullmodel/cache/latency/shelf claim. Next RMS/native rsqrt sourcecontrol and
exactfresh comparison, plus newpair2OFF/ON/serial49 proof. GPU rootcoordinated.


## 2026-10-10 - Source37 paired2 OFF closed; matched ON starts

CONFIG -> Same fresh source37 SDK, actualone-adjudication/current1374pair
baselines, harness40 native2/32token diagnostic, OFF/ON settings matched.
COMMAND -> ActualOFF78197 terminal0/PASS462s. Rootreadonly audit.parent_arm
PASS, parentSHA4120b46018352452148a1ab6d3271dcbde373d7487053f3717f0e0656bedb8a6.
ONCPUprepare22930 PASS; rootleasesON15143 afterOFF terminal, prehealthPASS and
child896984live. Newprivate reader12CPU/72SHAASCIIAST PASS, trackedb5cb991.
RESULT -> OFF actualcollection/normalownedterminal/ownerfrees/health/journal/
new4/pages complete. OFF raw49/Nrowevents remainunobserved bydesign. ON
requestedNrow/currentgraph/frame/raw49 andeveryactualselected serial49 job
stillrequired before pairednumerical comparison or4/6 progression. Private
reader states stdin-cancel timing inferred fromsourcepredicate+actualACK, not
independentlylogged. Same235 fresh controls/newshared runtime source pending.
VERDICT -> Actualpairedcontrol closes; matchedobserver arm nowexecuting.
Originalfullmath/APIcache/quality/profiling/latency/shelf goal stays active.


## 2026-10-10 - Source37 paired2 ON closes; actual serial49 starts

CONFIG -> Samecurrent SDK37/native2 harness40 after OFF462s PASS; actual
one-adjudication/currentpair1374 baselines; diagnostic toggles only.
COMMAND -> ON15143 terminal0/PASS386s; rootreadonly current parent_arm +vectors
PASS196 vectors, parentSHA13462ad721085c3d410aeafe461ffce58713819cda4970e3fa29abe4eb6ec9d5.
Four actualselected prefixes/each48 residual+head; serial0CPUprepare15661 PASS.
Root starts leasedserial0 withsameSDK/corpus/currentON identity.
RESULT -> ON collected/normalteardown/ownerfrees/health/journal/new4/pages
complete. Actual196 exactfreshserial comparisons stillpending; no paired
numerical progression4/6 or fidelity/latency claim yet. Same235 draft130dc
review rejects globalwarms-only chronology and droppingERR/SERR/FATAL; new
freeze required beforeanyactualcontrol run. OriginalV6 failure preserved.
VERDICT -> Concrete twoGPU observer/runtime corrected37 proof advances;
complete originalmath/cache/quality/profiling/latency/shelf goal active.


## 2026-10-10 - Serial cache-policy parser failure closed; fresh-state controls live

CONFIG -> New37/40 sameSDK freshserial exactprefix comparison; separate
source35 same235 state controls preserving actualtarget/warm/configuration.
COMMAND -> Serial62605 terminal1/FAIL471s, engine0/removed/noForcedCleanup,
posthealth/kernel/new4 PASS; parentSHAbe7f5eddc39d4cdc14f48a106dcee871156c6c9df10961a9ac7879d0ac0b43f6.
Root/independent corrected same23516CPU/170SHAASCIIAST closure PASS, sourceplan
a18dee2d6d5868b409a4379a7cbf7164fe2c754974f474e69436d1eb0bed8080;
tracked5172251 beforegenuineprepare15013 terminal0/PASS and leased31395start.
RESULT -> Serial firstjob has49 captures, fails inheritedcacheON reusable
predicate. Actualsource37/PCL tuple is publishedFALSE/chain_updatedTRUE/
live_reusableFALSE; earlier proposed chainFALSE wasincorrect. LegacyPIN0
request mustremain explicitlyhistorical, not falselyabsentPINqualified. New
strictreadonlypartialadjudicator and absentPIN successor beingprepared.
Same235 controller enforces owncompletedwarm2 beforeeachscalar target,
disjointcohorts, oneengine/PID and nativeerrors; actualprehealthPASS/childlive.
Sharedobserver process64MiB budget preventsmonolithic16phaseactor; newbounded
scenarioactors mustretainfullrequirements withcoldpriming peractor and no
assumed state transfer. Realvictim/stale/history/fresh49 requirements remain.
VERDICT -> Realfailure preserved andlocalized; no paired196comparison yet.
Actualexactinput fresh-state test nowrunning, no EOS/state/cache/math claim.
Fulloriginal goal active; source37 kernels/weights untouched.


## 2026-10-10 - First49 recovery and exact235 serial fresh matrix closed

CONFIG -> Strictreadonly historicalV40 firstjob, PIN0 exception only; actual
source35 same235 fourcontrol matrix afterownwarm47cohorts, unchangedNNconfig.
COMMAND -> Firstadjud18377 terminal0/PASS49 andunchangedoriginaltree, saved
receiptSHA4caabd8e11098bf2833e9d6787d2bd38f8f9fae9e324bebbcd8097b78d76938b.
Same23531395 terminal0/PASS1144s; rootcurrent publicreader65883 PASS. ParentSHA
f9644629c1f2d38846447f42b571b288bb006a100948ff16cf5b5c5093370d65.
RESULT -> Four serial fresh0/fresh1 xexacttarget0/1 allproduce3833,248046
(One+EOS), full235read/reused0/reset markers afterdistinctproperwarmcohorts.
Normalowned/EOF/health/journal/new4/pages/source gates complete. No full49
state/cache/math qualification; queuedfresh1 andindependentreference remain.
First49 historicalbyte comparisons BITWISE, originalparentfalse/PIN0 scope
retained; historicalsupervisorEOF unobserved. Remaining147 newparent42 needs
prospectiveEOF/rawfile/sourcepostjoin proofs, sourceV2review stillpending.
Sharedruntime checkpoint211SHA/13CPU tracked6cc5663 NOTlaunch-ready: actual
static64MiB plus6RID ledger bounds require boundedcapture actors andseparate
qualifieddiagnostic extension forfullhistory/eviction/victim/solo evidence.
VERDICT -> Concrete numericalrecovery +exactflagdifferential closes, no
underlyingEOScause orcache/fidelity/speed/shelf promotion. Fullgoal active.


## 2026-10-10 - JSON receipt preflight rejects; exact235 CPU reference starts

CONFIG -> Prospective remaining147 serial controllerV2, saved strictfirst49
receipt; separately qualified freshCPU reference exacttarget235 corpus.
COMMAND -> CPUprepare88193 terminal1 aftergenuineV40 originprepare, beforeGPU
execution. Rootreadonly83250 recomputes allfirstreceipt gates: rawPython
equalityFALSE, completeJSON-valuesequalityTRUE. Original95/adjud unchanged.
CPUreference root/independent9CPU/42bindings PASS, tracked627a624. Wrapper
50000 terminal0/PASS no-model. ActualexclusiveRAM CPU71408 startedpairlease
with noGPUdevices; noGPUexperiment overlapsit.
RESULT -> Receipt tuples/intobjectkeys round-trip toJSON lists/stringkeys;
directPython comparison iswrong. NEWstrictcanonical JSON consumer required
with duplicateconverted-key/nonfinite/type checks, preserving allfields.
Partialremaining3prepared-v1 retains genuineoriginonly, no acceptedV2 plan.
ExactCPUreference twofreshprocesses/natural64/temp0/seed1 retainoriginal235
render/IDs andsource/runtime/memory/ownednormal/new4/pages checks; no prior
prompt-success transfer. Actualobservations stillpending.
VERDICT -> Realpreflight datatypeissue localized beforedeviceuse; current
model/kernel/weights unchanged. NewCPUreference progressesEOS/localfidelity
question whilefuturemetadatafixed. Fullcache/quality/latency/shelf goal active.


## 2026-10-10 - Exact235 CPU reference closes; strict serial V3 frozen

CONFIG -> Two independent fresh CPU llama.cpp processes, exact original235
input corpus, natural64/temp0/seed1; serial/private V3 JSON representation fix.
COMMAND -> Actual CPU71408 terminal0/PASS435s, owned processes removed and new
completefour publisher hashes/pages PASS. Root V3 unittest26+20 PASS and
114/127 dependency SHA closures PASS; genuine V3 preparation15962 started
from preserved genuine V40 origin without repeating origin preparation.
RESULT -> Both CPU cases output One and EOS, matching bounded Strata targets.
This establishes an independent exact-input observation, not native arithmetic
equivalence, model quality, underlying EOS cause, or speed qualification.
V3 strict canonical comparison retains all fields/types and rejects collisions,
duplicates and nonfinite values; original failed receipts remain unchanged.
VERDICT -> Independent reference evidence complete; remaining147 numerical
comparisons and complete cache/concurrency/fidelity/latency/shelf remain open.


## 2026-10-10 - Remaining147 V3 genuine preparation completes

CONFIG -> Source37 paired2 cacheOFF fresh serial jobs1/2/3, budget1 each;
strict historical first49 is a separate named receipt, no scope expansion.
COMMAND -> Genuine CPUprepare15962 terminal0; planSHA
4f81326368b315815840b2f03a7df42c60746770f879ce14a2d2a70a196a4613.
Prospective parent43 actual70200 started; PID915796 confirmed live in CPU
admission, before GPU health/inference. Source checkpointfc15bc4 pushed.
RESULT -> V3 canonical roundtrip now accepts the exact saved JSON receipt
while all source/baseline/adjudication gates rerun. No runtime numerical
result yet; same active70200 handle must be polled, not restarted on silence.
VERDICT -> Genuine preparation advances remaining147; goal stays active.


## 2026-10-10 - Live serial admission and next diagnostic CPU review

CONFIG -> Same live serial70200, unchanged frozen parent43/source37. New
source38 defaultOFF selected-RID/cache-victim/memory proposal, CPU-only.
COMMAND -> Poll70200 confirms lease wrapper915796 and leased child916020
live; prelease CPU admission repeated after exec, leased child observed
CPU100percent at90s with parent snapshot created and no reported errors.
Root source38 final12 CPU controls PASS;53-file closure SHA matches sourceplan
8c8cac3a6e5a2d77ecd263284b6b9439f9dfc630ca26a3f05c30039d717f1461.
RESULT -> Repeated admission is a concrete dev-loop optimization candidate;
future operation-local reuse must retain fresh pre/post integrity and mutation
rejection. Current live frozen code unchanged. Source38 proposal preserves
static6 RID/64MiB bounds, observes victims/host logical memory and adds
truthful directGEN diagnostic phase; native compile/runtime remain unobserved.
VERDICT -> Verified live wait plus new CPU evidence; no source38 launch or
cache/fullmath/speed claim. Independent review and fresh ABI checks remain.


## 2026-10-10 - RMS diagnostic hypotheses preregistered; serial engine live

CONFIG -> Unchanged SDK37 nativeHC RS diagnostic proposal; saved independent
original embedding/residual, synthetic normones/zero down/up weights only.
COMMAND -> Root9 CPU controls and113-file closure PASS, sourceplan
31d294c1412e79444a45d1a8dd90ea856712fc1a9a2799b562e6964618ea6602.
Metadata-only prepare completed at f17-source37-20261010/
native-rms-rsqrt37-preregistered-inputs-v1; inputSHA
9913fa74fb647cbe9dd92a4d55f834177bc927d48226b13a1930f4bbb576ba1e.
RESULT -> Two argument families times three RS hypotheses recorded before
native observation. No fresh model reads/compile/GPU. Object precise flags
and production correctly-rounded device-link flag have separate authorities.
Actual70200 prehealth PASS; child917640 owns live model engine and is loading
source37 native HC/PLE and stage mirrors, numerical observations pending.
VERDICT -> Diagnostic fixture prepared, actualRS/fullmath unqualified; new
owned lifecycle qualifier required. No parallel GPU experiment launched.


## 2026-10-10 - Remaining147 comparisons pass; diagnostic38 source frozen

CONFIG -> Same actual serial70200/source37; bounded source38 diagnostic only.
COMMAND -> Child report147 comparisons PASS, engine0/removed/noerror. Parent
poststrict started; finalposthealth/journal/new4 and readonlyjoin stillpending.
Root source38 CPU12/53SHA PASS; c112 independent consumed-member/defaultOFF/
deque identity/two-graph-lifetime source review completed.
RESULT -> Source38 checkpoint8c8cac3a6e5a2d77ecd263284b6b9439f9dfc630ca26a3f05c30039d717f1461
remains source-only. FC38 main-body work is GEN/admission, not full BSTEP/BDONE
or HTTP terminal authority; entries bytes omit reusable/held/incoming and
physical accounting. New successor39 is adding fullbatch terminal and logical
reservation observers, preserving38. No source38 nativebuild/runtime executed.
Dev-loop census binds10 pack files1495822427bytes; nested validate_prepared
rehashes dense.bin1485688320bytes repeatedly. Actualleased admission rchar
124161860824bytes observed; exact callcount/cost stillunmeasured.
VERDICT -> Numericalcollection advances, final147 parent unproven. Freeze
reviewed boundeddiagnostic without fullcache/math/latency/physicalmemory claim.


## 2026-10-10 - Remaining147 parent closes with full integrity

CONFIG -> Same source37 paired2 fresh cacheOFF absentPIN serial jobs1/2/3.
COMMAND -> Actual70200 terminal0/PASS724s under pairlease; parentSHA
a2524d4ab09067a952f344cd74077a971b86688fea92155ad5fc96ede8b0c195. Child147 full-head/all48 vector
comparisons PASS, normalowned engine0/removal, strict percard/compiledpair
pre/posthealth, kerneljournal, NEW completefour publisherhashes and knownpage
brackets PASS. Strictreadonly combined-roster publication53284 started.
RESULT -> Actual remaining147 numerical arm closes. First49 stays named
historical recovery with PIN0 and unobserved supervisorEOF restrictions.
All196 private OFF/ON/serial join remains pending; no fullmath/cache/latency
or serving shelf promotion follows from scoped native comparisons.
VERDICT -> Parent qualification advances concurrency numerical prerequisites;
full original objective remains active and readonlyjoin mustfinish.


## 2026-10-10 - Combined private V3 metadata join rejects page epoch

CONFIG -> Actual immutable source37 OFF/ON parents plus first49 and new147.
COMMAND -> Readonly roster53284 terminal1 at privateV3 matched_plans before
publication; root recursive exactsnapshot diff saved external receiptSHA
d98461cc1aa4829d199755917aa2e9f7eff16f17ea1baffcdb2fdb3ea3259621.
RESULT -> Only differences: declared diagnostic/alias/registry servedID and
model_identity.current_known_pages.epoch (OFF1791627892.37529 vs
ON1791628030.9351199). Full identity path/hash/revision, page digests and stat5
match. V3 compares observation timestamp as matched semantic configuration.
NEW V4 must independently prove each timestamp/page observation and compare
all substantive fields strictly. FailedV3/no-roster and original arms unchanged.
Successor39 root12CPU/61SHA PASS, independent review pending; no nativebuild.
VERDICT -> Actual147 parent remains PASS, private196 remains unqualified.
Metadata consumer repair required, no GPUrerun justified by this mismatch.


## 2026-10-10 - Exact per-arm page-epoch private V4 frozen

CONFIG -> Same immutable source37 OFF/ON/first49/remaining147; no GPUrerun.
COMMAND -> Root actualmodel_page_epoch_binding botharms and matched_plans
PASS. Root19 newV4 CPUcontrols and131-file SHA closure PASS; sourceplan
28e9041801d23e506d98b5369491907c1792eb5e06a780c97dee637b8429fc40.
RESULT -> Original observedpage epoch proves full4.finished<=epoch<=actual
parent.started. Every other modelidentity field matches current independent
guard; canonical allotherplan fields match exactly with declareddiag aliases.
V3 failure remains unchanged; savedtimestamp is not recomputed fromclock.
Immutable V3 canonical20 and serialcanonical26 prerequisites remain pinned.
VERDICT -> Corrected metadata matcher passes actualsnapshots; full196 runtime
recollection remains unproven until genuineV4 readonlyroster/finaljoin finish.


## 2026-10-10 - Cache memory39 source clears independent review

CONFIG -> Separate defaultOFF diagnostic39 atop frozen38;66sources/30headers/
39patches/8ABI/6Python. No native math/cache-policy/scheduling change intended.
COMMAND -> Root and independent c11215 CPU controls/61SHA closure PASS,
sourceplan4a2bfa146b26d3b7f2883a111277bcb38b88bdd68f0393dd03020222ca3b302c;
engine86bc189bbb2805ffcb989974146fe8263a00beeb109cd44ef41af77eb1db6940.
RESULT -> MultiBT within onewindow and nextBSTEP grammar corrected before
runtime. Resource rows explicitly describe main-body context, not an invented
active-slot owner; native lifetime retains actualRID/slotgen. Logical reusable/
held/incoming/stage-image accounting and retained capacities independently
reviewed; full physical/transient/HTTP/fullcache authority remains false.
Actual privateV4 roster87154 confirmed live PID921931 revalidating oldcorpus.
VERDICT -> Source39 ready for fresh full8 ABI build and subsequent runtime
qualification. Old37 evidence cannot qualify new39 code. Full goal active.


## 2026-10-10 - V4 serial roster closes; source39 fresh ABI build starts

CONFIG -> Frozen privateV4 28e90418; independentlyreviewed source39 eb5fa4a.
COMMAND -> Metadata-only roster87154 terminal0; external schema4 rosterSHA
139b6b3eb489576bd2623a0595c5c04773565e456d82fa2193b5c14987a41255.
Actualfinal private reader4616 started with original OFF/ON artifacts. Fresh
ABI build3527 started underpairlease with no devices exposed, clean pinned
ggml3cf03257 source and39engine86bc189; PID922891 confirmedlive.
RESULT -> Rosterpublication makes noqualificationclaim (CLI passedfalse is
expected for this association object); final196 fullnumerical/histories/owner
join remains pending. Source39 nativebuild is genuinelyrunning, no oldSDK
proof transfer. No fullmodel serving or GPU diagnostic overlaps compilation.
VERDICT -> Actualmetadataassociation and freshbuild advance; full original
objective remains unproven until native/fidelity/cache/concurrent/latency gates.


## 2026-10-10 - Source39 fresh eight-target build completes

CONFIG -> Frozen39engine86bc189, compiler39992, clean ggml3cf03257, fourjobs.
COMMAND -> Actual3527 terminal0/PASS324s; buildroot
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T122458Z-_0gr_xk8,
receiptSHA5908e669eefe63d25d12f97dfebea4d9d73bdf92f1b072f74b3bf37f4e7052e8.
Root actual66 source hashes and8 fresh ELF hashes/header checks PASS.
RESULT -> Native39 compilation established, external sources unchanged/no
devices exposed. Upload/newC139 baseline/OFFON/cache/runtime remain unqualified.
Root newoverlap corpus7CPU/82SHA PASS, sourceplan
c31a56276794c1e94fd3f01d48e6596630f640d0ebaa0c655568352e1ed52e72.
Three coherent finite promptpairs plus2warmups preregistered; no output
continuation/model inference/token IDs claimed before authentic export.
VERDICT -> Compilation advances current39, fullgoal active; genuine tokenizer
export and new-purpose CPU continuation screen precede positiveAPI overlap.


## 2026-10-10 - Authentic new overlap corpus exported

CONFIG -> Frozen c31 overlap corpus, exactoriginal tokenizer/template and
source35 renderer; metadata-only c388 container/noGPU/modelmount/network.
COMMAND -> Actual36119 terminal0/PASS, owned normalremoval; fixture receiptSHA
74b6072df8d3fdc9c2565e9983d99b7da0a65f5b0ad7678529eeb202e9f67b2b.
Genuine screen-plan metadata preparation terminal0 at f17-source37-20261010/
api-positive-overlap-cpu-screen-plan-v1.json, samplingseed1234/temp0/natural64.
RESULT -> Warm IDs50/50; target IDs178/179/177/178/183/184. All8 exactroles
remain preregistered; no continuation/modelquality/overlap/cache proof.
Current CPUcompletion API has stop_type, not legacy stopped_eos/stopped_limit;
NEW actualscreen producer must source-bound interpret rawresponse and preserve
it fully, rather than assume those absent fields. All16 fresh repeats required.
OwnedRMS draft10CPU/134SHA nowPASS, but runtimeimage must match actualserving
c388 rather than compiler39992 before usingit for nativecontract diagnosis.
VERDICT -> Authentic inputs prepared; CPUmodel screen and actualAPI/raw49
checks remain required. V4final reader4616 stilllive; fullgoal staysactive.


## 2026-10-10 - Owned RMS V1 review checkpoint preserved before launch

CONFIG -> Frozen31d RMS math proposal and savedindependent residual; new
owned compiler39992/runtimeC388 qualifierV1, no actualnative execution.
COMMAND -> Root13 CPU controls/134SHA PASS, ownedsourceplan
7ee7968d44c030d8c99ccd72acf3f8c460736a8e446859c86ab41299be1b4354.
RESULT -> Actualbefore/after maps, typed image/argv/mount/device/resource
ownership, EOFretirement/health/journal/new4/page chronology prepared. Root
review found finalreader must additionally assert ordered publisher bytes and
sha/expectedsha againstlock for every row, not merely trust producerpassed
and currentstat5. NEWV2 plus mutationnegatives and explicitcachepersistent0
beingprepared; frozenV1 preserved unexecuted, not launchready.
VERDICT -> Source/CPU review advances; no RS/math/runtime proof assigned.
PrivateV4 final4616 remains confirmedlive CPU revalidating existingcorpus.


## 2026-10-10 - Private196 closes; actual C139 header-ledger rejection

CONFIG -> FrozenV4 source37 actual OFF/ON/first49/new147, current165registry.
COMMAND -> Actualfinal4616 terminal0/PASS196; readonlyreportSHA
c05d9416ac7b92a78f9d770283c22dd118017a4ef6281c2740b46eb6eb4bc9f1.
Rootactualreport confirms full196 bitwise, actual2rows, histories/ownership
joins; firstfailedparent remainsfalse/PIN0/unobservedhistoricalEOF limitations.
RootC13933CPU/85+87 sourceSHA PASS, then actual sourcegate61868 terminal1
BEFOREGPU: consumed batch_fidelity_contract header payload differs.
RESULT -> Actual39code all66finalhashes/8ELFsPASS, but inherited added_header
payload ledger records two historicalpre38 headerbytes while finalexpected
source66 reflects source38 contract/observer changes. Newexplicitconsumer
repair required; actual39build86bc/receipt/code remain unchanged. Current
C139source-only proposal preserved, not runtimequalified.
VERDICT -> Scoped private2 numerical control established, not originalmath/
fullcache/latency. Metadata source-admission issue localized withoutGPUrerun.
Full goal active; new139consumer and ownedRMSV2 peerreview pending.


## 2026-10-10 - RMS V2 review preserved and CPU screen source committed

CONFIG -> OwnedRMS V2 c842/139, CPU all8x2 e633/115, no actualmodel/GPU run.
COMMAND -> Root17RMS CPU/139SHA and13screen CPU/115bindings PASS. RMS V2
source checkpointdbcc337 preserves NONREADY lifecycle review. Initial combined
staging stopped on an external runtime receipt; Git rejected it. Corrected
staging verifies everyexternal binding but commits only repository sources.
RESULT -> V2 timeout container-stop-before-EOF and exacthealth/journal argv
gaps require NEWV3. No V2 launch despite narrowed intrinsic review PASS. CPU
screen actualrawstop_type/all16/seed1234/source/normalowned/memory proof source
nowcommitted; oldresponse/continuation/GPU proofs do nottransfer.
VERDICT -> Rejected drafts preserved; CPU wrapper smoke and independent
screen review pending. Fullgoal staysactive with scoped private196 established.


## 2026-10-10 - C139 final-header gate succeeds; CPU wrapper smoke passes

CONFIG -> New C139 V2 consumer, unchanged fresh39 code/build86bc/receipt5908.
COMMAND -> Root37CPU and100/101 sourceSHA PASS; actualmetadata gate18001
terminal0/PASS. Original28 headerpayload provenance plus new2 retained, all30
actualconsumedheaders match final66 ledger; exactly2 source38 modifications.
Genuine CPU wrapper27433 terminal0/PASS at api-positive-overlap-cpu-wrapper-
preflight-v1, with all8 authentic renders/CPUsoftware but no model/GPU.
RESULT -> C139 oldmetadatafailure preserved; no nativeABI rebuild required
for corrected consumer. Fresh oracle/upload/new1392 baselines remainpending.
CPU continuation remainsfalse/unobserved until all16 actualfreshcases finish.
Registry86621 unchanged while pending source37 RMS experiment depends on it.
VERDICT -> Source-admission and non-model smoke advance; full original goal
active. No serving/cache/latency/quality/shelf claim assigned.


## 2026-10-10 - Fresh source39 oracle compiled; actual upload390 starts

CONFIG -> Unchanged actual39 SDK86bc/receipt5908 with C139 V2 headerconsumer.
COMMAND -> Freshoracle3304 terminal0/PASS48s, root
/mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-fwgm5sx0,
receiptSHAb88effbd1e8b11e31f390bd491bd1123f60752084da7faabf39ad64a50d4678f.
Actualmetadata oracle admissionV2 terminal0. Freshactualupload58683 started
underpairlease with source39planca334, originalpackintakee9a7 and current
postterminal/posthealth identity fromremaining147; five source390 cases.
RESULT -> Freshcompile/link/library proof established; actualdevice upload,
ordinarypayloads/health/teardown/new4/pages stillpending. No registrychange,
source37 model/math/concurrency proof transfer or parallelGPU workload.
VERDICT -> Concrete native39 qualification underway; fullgoal staysactive.


## 2026-10-10 - Source39 upload390 closes with healthy full-source lifecycle

CONFIG -> Fresh39 SDK86bc/receipt5908 and newly linked oracle b88effbd;
source390 plan ca334, allfive card/topology cases, original four shards.
COMMAND -> Actual58683 terminal0/PASS288s, receiptSHA
08f17c66aa836aeaa0c517d0f12c8746d4b0c2d01f73feedcea863e6a99b28fc.
Rootactual C1392 upload_gate73590 terminal0/PASS, postfull4/currentpages
and source/owner/normalterminal metadata joins reproduced.
RESULT -> Five cases all exit0/source_and_probe_passed/owner destructors
returned; strictpercard/compiledpair preposthealth, ownedterminal and NEW
completefour identity/pages PASS. Ordinary_payload_readback_qualified remains
FALSE, as does fullmodelmath. Selected HC387/PLE3 upload/lifecycle only.
RMS V3 source145/23CPU PASS but peer found report commandrow must exactly
match savedproducer receipt; successor in preparation, no nativeRMS launch.
VERDICT -> Current39 source-upload prerequisite advances. NewC1392 serving,
originalfidelity/fullcache/API/concurrency/latency/verifiedshelf remain open.


## 2026-10-10 - RMS owned V4 receipt/lifecycle source frozen

CONFIG -> Frozen31d saved-independent RMS fixture and unchanged native37HC,
compiler39992/runtimeC388; all earlier ownedV1/V2/V3 source remains preserved.
COMMAND -> Root30 CPUcontrols/150SHA PASS, V4 sourceplan
7cdae45fc0b7bb2bdd74e701876bca1fb31b33198bd7aaa970285467b09bd02d.
RESULT -> Exactowned timeout and normal-exit stdout-drain cleanup precede
EOFretirement; strict health/P2P0/journal argv/epochs reconstructed. Every
commandreport row canonically joins original savedreceipt with duplicate/
nonfinite/type rejection, explicitcleanup_errorNONE and artifact-roster proof.
Originalpublisher4 bytes/hashes/stats/orderedpaths and preservedpage brackets
remain strict. No nativecompile orGPUrun yet; independentreview pending.
VERDICT -> Review gaps addressed in prospective source; actualbounded RMS
observation remains required. Full originalfidelity/cache/latency/shelf open.


## 2026-10-10 - CPU screen memory failure; RMS V5 clears full peer review

CONFIG -> ActualCPU all16 e633 recipe with strictnohostswap guard; RMS V5
source-only comprehensive successor, unchanged31d math and savedown input.
COMMAND -> CPU95049 terminal1/FAIL246s at case0repeat0. Actual memory monitor
reports Hostswapuseincreased, ownedstop/RemoteDisconnected/no completion,
cleanupnonzeroexit. Fresh postterminalfull4/pages PASS; originalfailure kept.
RMS V5 root40CPU/157SHA and independent49CPU (40+9math) PASS, sourceplan
a8488530a6e02fd35b70e64be6dfaa9b0129c94b396653fcaae2fd210e17a2bb.
RESULT -> CPUcontinuation unobserved, no pair selected and guard notrelaxed.
Memory causality/newexplicitrecipe under investigation. RMS rawfaults/native
markers/originalinspection+receipt/library/currentELF/prepostpublisher/pages/
Docker/EOF/health chronology comprehensivepeer READY, no knownsourceblocker.
VERDICT -> Preserve actualfailure; proceed fresh boundedRMS GPU experiment
after nowterminalCPU lease. Full fidelity/cache/latency/shelf remain open.


## 2026-10-10 - RMS isolated compile succeeds after entry/init repair

CONFIG -> Frozen31d math retained; V6guardd87a/V7entry265e prospective code.
COMMAND -> V5 actual73415 terminal1/FAIL317s: compileexit3, imageEnv
SETVARS_COMPLETED1, nocompiler/stdout/RS. Preposthealth/new4 PASS; reportSHA
a4f3dcf10b888417c03a946c849c99768a565cee20971f744f2a47847dee4dee.
Rootconditionalsetvars smoke bothimages version0/0 PASS, no devices/model.
Root61CPU/170SHA and independententry derivation review PASS. Actualisolated
compile55085 terminal0/PASS22s with no devices/modelmounts, fresh ELF.
RESULT -> V6 skipalreadyinitialized oneAPI; V7 callable copy changesONLY
soleentry identifier, avoiding nestedmacro reset, all mathbytes unchanged.
CPU95049 hostswap audit: ownedVmSwap/cgroupSwap0/noOOM, globaldelta10592256
bytes; no causalowned-memorypressure finding or guard/recipe relaxation.
VERDICT -> Actualsoftware compile closes before fullGPUrepeat; no nativeRS/
fullmath/cache/latency qualification yet. Full original goal remains active.


## 2026-10-10 - Fresh RMS V7 full qualification starts

CONFIG -> Unchanged compiled37 HC and independent saved40960byte residual,
new entry-only callable adapter, conditional oneAPI init, compiler39992/
runtimeC388, card0 leaf plus pair-health, source170/61CPU/peer review PASS.
COMMAND -> Rootactual35193 started, PID947207 confirmedlive underpairlease
aftercurrentsemantic admission. New output f17-source37-20261010/
native-rms-rsqrt37-owned-v7-run-v1. Previous73415 FAILED andisolated55085
compile-only PASS22s remain distinct; no oldhelper runtimeproof transfer.
RESULT -> Fullpre-source/health/freshcompile/direct+replay/posthealth/new4
sequence is genuinelyrunning; no RS observation or finalparent result yet.
Registry165 remains unchanged; no GPU/model run overlaps this owned parent.
VERDICT -> Currentboundednative arithmetic diagnostic underway; fulloriginal
fidelity/cache/concurrency/latency/verifiedshelf goal remains active.


## 2026-10-10 - Actual RMS direct/graph outputs and full reader close

CONFIG -> Unchanged compiled37 HC, original independent residual, synthetic
normones/zero down/up; exactobject/link flags and servingimageC388.
COMMAND -> Actual35193 terminal0/PASS317s. Publicreadonly83248 terminal0/
PASS; originalreportSHA58da27b488ba02c77d9717977d68152a25fd6cff21f278a2336149daa56b462c,
readonlybindingSHAc45571e440fb72e1a9a863c35331fa4b79800b15275692dcdfee2a05344c2ba1.
RESULT -> All3 routes4fields bitwiseequal. ActualHC_RS allfour430aba1f
(138.72703552246094), SOURCEargument shadow3859f0be, legacyarg3859f0bd.
NativeRS matchesall3 legacyargument hostcandidates; sourceargument roundedRS
430aba1e is1ULP lower. Argumentshadow is NOT nativeinternalargument witness;
no reduction/intrinsiccause or tolerance inferred. Originalcore/math unchanged.
Ownednormalterminal/posthealth/journal/new4/pages/source/library/receipt joins
PASS. Newdevice-operation control preregistered separately, unexecuted.
VERDICT -> Real bounded HC normalization observation narrows firstscale-byte
investigation; no fulloriginalmath/cache/concurrency/latency/shelf promotion.


## 2026-10-10 - Device-operation follow-up source frozen

CONFIG -> Original unchangedHC and own input/flags preserved; source-only
three eager deviceRS expressions at the same separate sourceargument.
COMMAND -> Root77CPU/182SHA PASS, ownedV8plan
4ed12398b46e490ea1a85fef902f1441e76f42d83dcfc818c179b82ee735108c;
independentpeer16 newCPU/addition/lifecycle controls READY. Rootisolated
compile8008 started underpairlease with no devices/modelmounts.
RESULT -> Frozenproposal6543 adds rsqrt/native::rsqrt/1-over-sqrt candidates,
all original4 fields must bitwise match hardpinned actualV7, all7 fields repeat
on3 routes. Capturedfields onlytargets, neverinputs. Allprior math/source/
health/timeout/receipt/inspection/knownpage/publisher gates retained.
VERDICT -> New diagnostic source ready, compile/runtime results pending.
No fittedtolerance/internalargument/instruction/fullmath/latency claim.


## 2026-10-10 - Device-operation V8 compiles and actual run starts

CONFIG -> Frozen6543 seven-field proposal +owned4ed123/182, actualV7 targets
58da/c455 hardpinned. All originalHC/input/flags/library/math unchanged.
COMMAND -> Rootisolatedcompile8008 terminal0/PASS19s no devices/model,
newELF under native-rms37-isolated-device-ops-compile-v8-20261010.
Rootactual94059 started; PID952142 confirmedlive afterpairlease acquisition
in f17-source37-20261010/native-rms-rsqrt37-owned-v8-run-v1.
RESULT -> native::rsqrt API compiles on pinnedcompiler, but actual device
operation outputs/fullparent proof pending. No oldsoftware-only success
transferred to runtime. Registry165 unchanged and no overlappingGPU/modelrun.
H45 source-only controller held under unresolved corpus/spec exactjoin;
NEW successor beingprepared without changing priorcase/evidence.
VERDICT -> Actual arithmetic discrimination progresses; no internalargument/
fullmodel math/cache/latency/shelf claim. Full goal remains active.


## 2026-10-10 - H46 prospective case/recipe ownership source frozen

CONFIG -> Exactsource37 native4/bounded6 numericalcontrol, current165registry
and actualprivateV4 closed196 prerequisite; all historical40/43/44/45 retained.
COMMAND -> Root31CPU/184SHA PASS, sourceplan
4d1bbcb5e9d4385154eb14613320c74f01278b19f23dbfb596f5b2c2287a58ee.
RESULT -> Exactauthenticated casepath/SHA plus tokens/messages/APIinputIDs/
cancel/counters/budgets/port/privatecontrol/tokenizer rejoin beforedevice.
Baseline args/env/image/pack/cards reconstruct strictly. Exactadmitted bytes
survive lease/health/child snapshots. Parent-only duplicateadmission removal
is explicit; children/runners retain currentsemantic checks. APIcache0 purpose
refusedprelease; separatecorrect APIpurpose stillneeded forfullgoal.
VERDICT -> Source/CPU only, peerreview and actualprepare/run pending; no4/6
completion or dev-loop/serving speed claim. V8deviceop actual94059 stilllive.


## 2026-10-10 - Device expressions resolve bounded RS difference

CONFIG -> Same independentoriginal input, unchangedactualHC/library/flags;
three eagerdevice expressions at separateargument3859f0be, original4V7targets.
COMMAND -> Actual94059 terminal0/PASS325s; publicreadonly28938 terminal0/
PASS. ReportSHA8f070f780a63eae2d92abdd98e62d2ecfc36ab1a95a81ade055d3b184fdf9c90;
bindingSHA3129cfd8c78a29e3cf9bd6f8792d9f93154439efe92c950ab871afbd5b9902bc.
RESULT -> All7 fields/3routes bitwise repeat, original4 exactlyV7. Device
rsqrt/native::rsqrt both430aba1f matchactualHC; explicit1/sqrt430aba1e
matches sourceargumentroundedhostcandidate. No actualHCinternalargument or
instruction lowering observed, no generalaccuracy/fittedtolerance inferred.
Posthealth/ownednormal/journal/new4/pages/source/runtime/receipt joins PASS.
Root genuineH46native4OFF CPUprepare79819 started fromsameC137/private196
actualprerequisites andnewfullysource-guarded modelidentity.
VERDICT -> Deviceoperation observation narrows independentreference math
contract; no backendmathchange/fullmodel/cache/latency/shelf qualification.


## 2026-10-10 - Passive CPU swap observer V3 source review checkpoint

CONFIG -> Unchanged e633 CPU continuation recipe, strict memory guards and
root pair exclusion discipline; actual95049 swap failure remains immutable.
COMMAND -> Root14 CPU controls PASS;128 source bindings and new ASCII/AST
checks PASS. New source plan0b4fa9e0b5ae483a3705064442e368a8dba5a79c7958b60ff28b90c6725db77b.
RESULT -> Reader now requires errors[], exact regular nonsymlink artifacts,
recursive original idle rejoin and end-of-work artifact/current full source
checks. V1/V2 historical files unchanged. Independent review requested;
no actual host observation, CPU model retry or GPU workload executed.
VERDICT -> Source/CPU progress; passive attribution and same-recipe all16
completion still pending. Original full goal, numerical/cache/API/latency/
verified shelf requirements remain active. H46 preparation79819 still live.


## 2026-10-10 - Genuine paired native4 H46 OFF arm starts

CONFIG -> Frozen H46 4d1bbcb5/184 source closure, actual source37/C137 paired
baseline and closed private196 prerequisites; registry165 unchanged.
COMMAND -> Preparation79819 terminal0/PASS; generated plan
SHAeb857c056dac88148dce006ae942a3580269f80584484219e63f473265218d8c.
Root qualifier19472 starts with fresh batch46-pair-native4-off-run-v1 output.
RESULT -> Pair exclusion lease acquired; current semantic admission precedes
health/device. Same original preparation handle was polled to completion,
never restarted on silence. No other model/GPU actor live at launch.
VERDICT -> Genuine four-stream experiment advances; actual runtime/parity/
health/postidentity results pending. No dev-loop speed, fullmath/cache/API/
latency/shelf claim. Full original goal remains active.


## 2026-10-10 - H47 frozen and observed CPU retry cleanup corrected

CONFIG -> Original goal intact; H46 actual native4 parent19472 remains live
with165 registry. No new simultaneous GPU/model actor. Explicit H47 pack
byte epochs retain original C137/private/serial producer provenance.
COMMAND -> Root51 CPU controls and224 SHA bindings PASS, plan
aa00f18bc1b06f27f646bfe4d6aa0d5dcaf1e3528818a17ae3dd783690318d24;
source checkpoint9ef95a6 pushed. Independent review requested before runtime.
RESULT -> Corrected subprocess cleanup kwargs and old dynamic selection call
are covered by exact callable-signature and source equivalence controls.
Witness parent/child byte boundaries are independent and process-local;
between-boundary mutation remains explicitly unobserved. No speed measured.
Passive observerV3 peerREADY0b4fa9e0/128/14CPU. RetryV1 peerheld for unprotected
idle and timeout escaping cleanup/report. NEW V2 f050f41c/136/19combinedCPU
protects idle, installs handlers first and records timeout while retaining
exclusion through actual owned joins. V1 remains unchanged/unexecuted.
VERDICT -> Source/CPU progress, review/runtime pending. Same-recipe all16 CPU
continuations, original math, fullcache/API1/2/4/6, x2 critical path, matched
latency/fairness and verified reproducible shelf remain required.


## 2026-10-10 - Observed CPU retry startup failure and owned recovery closed

CONFIG -> Exact e633 screen, observerV3/observed wrapperV2; no GPU grant and
strict memoryguards unchanged. Original CPU95049 swap failure preserved.
COMMAND -> Actual14032 terminal1/130s. Root ownedrecovery45070 terminal0/9s;
fresh pairleased post-recovery identity81692 terminal0/75s.
RESULT -> Observer missing-object parser rejected current lowercase Docker
absence before firstowned sample. Wrapper interrupted modelparent, whose
absent-container cleanup failed; pending launch created container after
wrapperterminal. Recorded emptyserverlog later changed to1242bytes. No
completion/candidate and no normal originalteardown qualification.
Root reacquiredpair, checked exactownedimage/labels/CPUrecipe, stoppedexit0/
OOMfalse andremoved. Recoveryreceipt54a4debb; newfull4/pagesPASS f4de9e015a.
Currentonly monitoring/UI containers live. No oldreport rewritten.
VERDICT -> Actual orchestrationfailure retained; explicit recovery/integrity
closed. NEW observerV4/wrapperV3 requires passivefailure isolation and actual
launch/descendant/census containment. Fulloriginalgoal remainsactive.

CONFIG -> H46 failed actualpaired2 savedbinding comparison; exact c05 report
content unchanged. Corrected mandatory fixed-point pack48 gate routes.
COMMAND -> Root14 CPU/241SHA PASS; actualtype recollector1992/PID962747 live
under CPU pair exclusion, newoutput outside all originalproof trees.
RESULT -> New actualfresh private196 and3completepackbyte boundaries pending.
No GPU/model inference by thisdiagnostic and no source/runtime prooftransfer.
VERDICT -> H46/H47 remain FAILED; native-vs-JSON representation diagnosis
pending before NEWH48 fixes. No numerical4/cache/API/latency/shelf claim.


## 2026-10-10 - Representation-only prerequisite diagnosed and fresh CPU retry live

CONFIG -> Frozen cf263 corrected routes, immutable eb857 H46 plan and c05
private196 report; original H46/H47 failures remain FAILED.
COMMAND -> Actual1992 terminal0/PASS559s, report6ed05cab321e67733043fd22541911d3b320aa0695b57fb4b41dcafc78739930.
RESULT -> Native Python equality false; typed canonical equality true. Exactly
81 tuple/list ownership fields differ, canonicalSHA both e06681a1. All3080
packdigest consumers and3freshcompletebyte boundaries closed. No GPU/model
inference. Fullmodel/cache/latency or oldparent PASS never inferred.
VERDICT -> Actual representation diagnosis closed; NEW H48 19c47079/263 and
69CPU+peerREADY preserve strict values. Sourcecheckpointbbc3340 pushed;
actual H48 prepare/runtime stillpending.

CONFIG -> Frozen observerV4 78792b6c/132 and wrapperV3 2e36b8ab/146; exact
unchanged e633 recipe/strictguards. Root32CPU and independentpeerREADY.
COMMAND -> Sourcecheckpointf9d1522 pushed; rootactual50448 starts fresh
api-positive-overlap-cpu-observed-retry-v3-run-v1 underpair CPU exclusion.
RESULT -> Idle30 passed. First case0/repeat0 passed actual response/identity/
ownedteardown, response64token limit so overlapeligibility false. Observer
captures actualownedcontainer. Remaining all8inputs/twofreshrepeats and final
full4/source/ownership/independentdecode/select gates remainpending.
VERDICT -> Concrete unchangedrecipe progression beyond startup failure;
no fullcontinuation/candidate/API/cache/math/latency/shelf qualification.


## 2026-10-10 - Independent HC device arithmetic V2 compiles and starts

CONFIG -> NEW de8dd545/227 source closure, own original weights/residuals and
source FMA/XOR arguments. V1 remainsheld; prior original references unchanged.
COMMAND -> Root95CPU/227SHA/ASCII/AST PASS, source33peerREADY. Isolatedcompile
17428 terminal0/PASS21s underpairlease, pinned39992 compiler/SDK37 flags/libs.
Savedfullimage/label/argv/mount/env/bounds recipe readonlybinding ca2167f4 PASS.
RESULT -> Fresh helper ELF compiled with no model mounts/inference. No device
arithmetic result transferred from compilation. Rootactual76881/PID1037680
started genuine currentbaseline/V8/Num10/P30/hostbulk inputs, source-only
preflight confirmedlive. Exact first normalized10240 and fullmixedQ81 gate
precedes optional freshprefix4; no capturedoperands/ULPfit/tolerancechange.
H48 metadataonly preparation87440 remainslive; no secondmodel/GPUactor.
VERDICT -> Actual independent arithmetic experiment progresses; firstgate,
prefix4, normalowned/health/new4/currentreader results pending. Fullgoal active.


## 2026-10-10 - H48 native4 plan prepared and first HC exact gate observed

CONFIG -> Exactsource37 H48 19c47079/263 with actual6ed05cab type diagnosis;
original H46/H47 failures remainimmutable. Independent ownHC V2 de8dd545.
COMMAND -> Genuine H48prepare87440 terminal0/PASS, plan
SHA8f440447fc7065f7aaddbd63176568c833659e30ab4c5e7f0ca2b2faaed7e60c,
native4 OFF/bothcards. Root actual76881 modelwork completed386RSrequests and
577 prefix4 comparisons; final parent/postidentity/publicreader stillpending.
RESULT -> Firstownnormalized10240 and complete2880-byteQ81 packet exactly
nativeNum10, confirmed root consumedraw comparisons (preliminarye5602992).
Workreport firstgate=true permitsfreshprefix4. Early p0L0attention NMSE5.45e-15,
p0L2FFN1.426e-14; finalprefix4head NMSE0.007888246485733295 remainsnonbitwise,
numeric_gate_assigned=false. No tolerance or backendmath patch introduced.
ActualruntimeEOF/normalhelper exit0 and poststrict+compiledpair healthPASS;
newfull4 scan live, so full parent admission and finalized evidence unproven.
VERDICT -> Concrete independent component agreement and remaining trajectory
mismatch. H48 actualmodel runtime queuedaftersoleHC actor releaseslease.
Fulloriginalmath/cache/API/profile/latency/verifiedshelf requirements intact.


## 2026-10-10 - HC V3 independently admitted and finite natural EOS starts

CONFIG -> OwnHC V3 93d9329b/234, exactown postPLE operands/no arithmeticchange;
recorded OMP/MKL/OpenBLAS1. H49 SDK+pack475f47f0/313 preserves300s freshness.
COMMAND -> Actual18877 terminal0/PASS696s report502566b296bb5809f01c5c5cd3b3938f460a7c862d61aae8573bdd7028983d2a;
matchedenvironment independentreader41715 terminal0/PASS binding770fe1a023fe893afee718bc8aaeb5943d661d829e8e2ef766965ffaeb3ff020.
RESULT -> All386 own operands/device requests rejoined, exact first10240norm/
2880Q81 target agreement,577prefix4 diagnostics, normalEOF/health/new4/pages.
Prefix4head NMSE0.007888 remainsnonbitwise/unqualified; no internalHCarg or
universalrounding witness. Initialreadonly invocation omitted threadenv and
wasrefused; corrected invocation matches actualruntime. V2FAILED unchanged.

COMMAND -> H49correctedprepare40222 terminal0/PASS, genuine native4OFF plan
c1c96f0adee7728c0487874a368cadf2641b9e5887bd98dd988e14e27707acc3.
Original mistaken v11case command73180 exitedmissingfile withoutpublication;
corrected commandusesfrozenv10 andnewoutput. Pack+SDK witnessesclosed, no speed
comparison or modelruntime yet; repeated source/evidence/log admissions persist.

CONFIG -> PeerREADY finiteV3 a3de9021/160+observer6/wrapper5,66rootCPU PASS.
COMMAND -> Authenticexport50031PASS f13b14ed0ad03832be3e00317877b2049824f5ed83a7443295d67109c2a8d34e;
actualwarm57/56,target185/186/185/185/191/189; realprepared4c9bcd56.
Rootactual15719 solemodel/pairCPUexclusion, freshfinite observedv5run.
RESULT -> Idle30PASS; firstwarmcase0repeat0 healthyresponse/terminal EOS21 and
eligibletrue. Allremaininginputs/repeats/decode/currentidentity/full4/ownership
and candidate selection remainpending. No oldresponses or counts transferred.
VERDICT -> Independent component agreement and genuine continuationprogress;
fullmodel/cache/API1/2/4/6/profile/latency/verifiedshelf remainrequired.


## 2026-10-10 - Finite CPU candidate admitted; passive observer failure preserved

CONFIG -> Frozen finiteV3 a3de9021/observer6/wrapper5, actual15719 pairedCPU
exclusion, exactselectedmodel and unchanged memory/sampling/no-swap guards.
COMMAND -> Wrapper15719 terminal1/853s; actualscreen child0; unchanged public
screen finalized_binding independently PASS6149753f3aedff89995f6fde1271175616c4734cb58657d0fccf57aea372f701.
RESULT -> All16 cases/repeats/originaldecode/identity/source/new4 passed;
water-cycle selected. ObserverFAIL Missing kernel byte fields makes wrapper
FAILED. Allownedlaunches absent/sessionempty/no recovery/noartifact changes.
VERDICT -> CPU candidate only admitted; observerfreshfix pending, no GPU/API
cache/fullmath/latency/shelf transfer. Rootstarts actualH49OFF7881 using
prepared c1c96f0a underpairlease; model-runtime result stillpending.


## 2026-10-10 - Authentic selected-input admission prepared

CONFIG -> NEW finite_overlap_candidate_admission_v1, successful actualfiniteV3
all16 screen, unchanged original corpus and exactscreen report pin.
COMMAND -> Root3tinyCPU PASS; currentaccessor actualadmission reruns unchanged
screenpublicreader+authenticexport and rejoins savedselectedinput657841f3.
RESULT -> water-cycle inputs warm57/56,target185/186; inputsonly, noCPUoutput/
state/length/counter/cache/observerPASS transfer. Source39actualactor separate.
VERDICT -> Concrete usefulGPU-input preparation, not GPU orfullmath proof.
H49OFF7881 andONprepare83996 remainlive; no restart orsecondmodelactor.


## 2026-10-10 - Own layer3 QSA projection source proposal independently reviewed

CONFIG -> NEW frozen8b615f3a9528f0f1e7eb84d9ea8481056f142a04e9720b79de16181da1e45567,
249repo+14consumedsourcebindings, inheritedHC V3 unchanged. Actualruntimefalse.
COMMAND -> Root18CPU PASS0.806s/allhashes; independentpeer18PASS0.792s/READY
for SOURCE PROPOSAL scope only. No payload/GPU/build by research agents.
RESULT -> Exactprefix4/identity/one-shot/ownzero-history guards, ownoriginal
projectioninput/output/gamma rawSHA/stat/shape joins. Production devicehelper
contract preservesQstride512 and2/1/1windows; no executablequalifier yet.
VERDICT -> Sourcecheckpoint only; helper+actualoriginalcompute/device/lifecycle
stillrequired. Prefix4 originalhead mismatch notqualified/tolerance unchanged.
H49ONprepare83996terminal0 plan1f0469671123bc06eaef860bcfd1cf0fb3004085d7f076c24d448c60e0aeae18;
OFF7881prehealthPASS/child1068991live, no finalmodelruntime result yet.


## 2026-10-10 - H49 actual freshness rejection fully closed

CONFIG -> FrozenH49 475f47f0 pairednative4OFF c1c96f0, unchanged300s checks.
COMMAND -> Actual7881 terminal1/1143s, parent0e405b27bf7053d27e9048936965baf7f2839f06817b9e0770a7b78a7cfd34c2.
RESULT -> Child467.63s admission exceeded freshhealth300s; line43 rejected
beforeoutput/NativeStream/modelleaf. Parentpre/poststrict+compiledhealth,
kernel/ownedterminal/noforcedcleanup/newcomplete4 PASS192b7010415cea8dc27d4fe0dd7a706a2cbca77853c0e0dc863ad2a79528664f.
VERDICT -> Failedoriginal retained; no numerical orspeed proof. ONplanprepared
but unexecuted; NEW H50semanticREADY/realhealthACK scheduling required.
Pairlease released. ObserverV7sourceaae9eac19CPU PASS/noactualobservation;
QSA V4helper/qualifier sourcework pending, fullcampaign goal remainsactive.


## 2026-10-10 - H50 frozen handoff source and genuine preparation starts

CONFIG -> NEW5f719918c0f6c6a0b817d2da33500bfd543bb0a232c86006ffaae58d3404d620,
331sourceclosure, native4/6-only, source37/private196/registry165 unchanged.
COMMAND -> Rootall331SHA/ASCII/AST andfull109CPU PASS1.100s. Peerreview pending;
rootCPU-only actualOFFprepare15923 andONprepare80579 started distinct outputs.
RESULT -> Parent/child READY->actualhealth->ACK->freshbytes->leaf protocol,
PID/start/lease/currentplan/source/realhealthrawjoins and unchanged300s checks.
Child4 byteboundaries andparent3 independent; oldH49failure unchanged. NoGPU
model actor started; both prepare processes mustfinish before admittingplans.
VERDICT -> Sourcecheckpoint and realCPUpreparation, no runtime/coherence/speed
qualification. PeerREADY and exactpreparedplan/ownedhealth gates precedeGPUrun.


## 2026-10-10 - Both H50 native4 plans prepared and actual OFF parent starts

CONFIG -> Frozen5f719918/331, root+peer109CPU PASS, native4 source37 unchanged.
COMMAND -> ActualCPU OFF15923 andON80579 terminal0; exactpreparedplans
3da5d55ec5521aabbba19befb624059fe77c5e5dd7511f09d5c759934446b3e0 and
8b46355e819d18a25319227b250bfa05dab181a0b3821118ac99af3c835da329.
RESULT -> Pack/SDK preparationwitnesses saved. RootactualOFFparent1874 acquired
pairlease andstartsfulladmission; modelleaf/runtime results remainpending.
N6OFFmetadata prepare5705 runs withcProfile, noGPU/inference/performanceclaim.

CONFIG -> Unintegrated immutableparser V1 4ee4aa45/19, freshisolated worker.
COMMAND -> Root25CPU PASS4.843s/all19SHA; localPythonPopen.__exit__ inspected.
RESULT -> Fixedparser synthetic5case80caller counts20 scans, no actuallogdata.
Interruptcleanup gap confirmed: KeyboardInterrupt context onlybrieflywaits,
whileworker has ownsession; V1 catchesTimeoutExpiredonly. Outputcap ischecked
aftercommunicate, so not a harddrain bound. FreshV2correctionrequested.
VERDICT -> H50runtime progresses independently; parserV1 HOLDintegration,
oldsource/failures unchanged. No modelmath/cache/latency/shelf qualification.


## 2026-10-10 - Own original QSA V4 producer/device source checkpoint

CONFIG -> Frozene12200531601cab9e403222c4ccb0e2d287b2a1065bad34615cbbd86dd6dcc25,
261repo+19consumedsource/header/build bindings, source37 SDK/HC V3 preserved.
COMMAND -> Rootall280SHA/ASCII/AST andfull134CPU PASS2.231s; independentpeer
review pending. No actual model, helpercompile or GPU work in source tests.
RESULT -> OwnfirstHCgate thenprefix4 projectionproducer; independent10ownfiles
feed separate unchanged-productionfunction QSAhelper. Full14fields/3routes,
2/1/1windows/zero-state resets, originalresolvedconfig, exactstdin/output/raw/
source/mappedlibs/health/publisher/normalteardown reader gates. No nativecaptured
operands substituted; internalnorm/score witnesses remainunobserved.
VERDICT -> Sourcecheckpoint only; peerREADY andisolatedcompile needed before
actualproducer/device/reference requalification. RootH50OFF1874 solepairactor;
N6instrumentedCPUprepare5705 remainslive. Fullmodel/cache/latency/shelf open.


## 2026-10-10 - Immutable parser V2 failure and ownership rules reviewed

CONFIG -> NEW75f2433bd27d7eb0ca11088f281daa7273279b122f637d1ba0aeb0ebd9524f91,
27sourcebindings; V14ee4 bytes preserved andunintegrated; H50 unchanged.
COMMAND -> Root34CPU PASS8.185s/all27SHA; independentoldtimeoutcounterexample
rerun oncorrectedV2: caughtfailure->finalize passedfalse, onefailure retained,
parsecounts unknownNone. ActualCPU workers use synthetictrace bytes only.
RESULT -> AllBaseException/timeout retirement, emptyownsession/closedregular
sinks, RLIMIT_FSIZE32MiB whileproducing output. Failed epoch cannotreusesuccess;
worker transport/protocol/parse failures retained. Fivecase synthetic80callers
use20actualscans; no actualupload/SDK/model payload or speed evidence transferred.
VERDICT -> Sourceproposalcheckpoint only, deliberately narrowfutureconsumer
port pending. Genericpurity contract unchanged. RootH50OFF1874 andinstrumented
N6prepare5705 remainlive; fulloriginalmath/cache/latency/shelf unqualified.


## 2026-10-10 - Actual H50 handoff reaches native leaf; CPU profile closes

CONFIG -> Exact frozenH50 5f719918, native4OFF actual1874; independentpairlease.
COMMAND -> ChildREADY1791655462.557, parentrealprehealthfinish1791655526.266,
ACK1791655526.358; finalbyte sealchecks1791655526.807/5527.911 thennativeleaf.
RESULT -> Actualchild1075593 terminal0, initialcollection/normalteardown true,
errorNone andownedcontainer removed. Parentposthealth/new4/finalreader pending.
No originalmath/cache/numeric/cleanlatency proof inferred fromcollection.

COMMAND -> N6instrumentedCPUprepare5705 terminal0 plan
a62de9a86715a9dcfbdc3bebac461ad904a20c695ce0acad7bd37c59035ca580.
RESULT -> ActualPStats1455.567s/2.2019Bcalls: original_upload_gate624 invokes;
parse_trace15600calls/cum1070.613s (~73.55percent), allfromthatgate's5cases
andduplicatedpositive+3negative controls. Runtimepayload untouched byagent.
VERDICT -> RealmeasuredCPUbottleneck; immutableparser consumerport nowjustified.
No matchedspeedup or cleanlatency result yet. Fullcampaign remainsactive.


## 2026-10-10 - H50 OFF parent passes and isolated QSA helper compiles

CONFIG -> FrozenH50source37 native4OFF withactualpostREADYhealth andnewbytes.
COMMAND -> Actual1874 terminal0/PASS1339s; posthealth/kernel/knownpages/new4/
normalownedretirement passed. Independentpublicparent_arm5003 stilllive.
VERDICT -> Parentcollection/lifecycle PASS only; OFFON/all49numeric/cache/
originalmath/latency remainseparate mandatorygates.

CONFIG -> PeerREADY e122QSA V4; exact isolatedsource/SDK/flags/image/recipe.
COMMAND -> Actual29705 bin/gpu-run --card0 pairedwithaffinity0; terminal0/
PASS22s; compileronly, nohelperexecution/modelmounts. ReceiptSHA
346257541cf291ceacf07352e147e3614ab1c1f27eeb6caed2d7777202a68600.
RESULT -> FreshhelperELF/log/currentfullownedinspection/exit0/normalremove.
OriginalQSAproducer launch70239 mistakenlywrappedself-leasingqualifier without
--leased. Verified3ownedlease-only processes/nooutput/noGPUleaf; SIGTERMgroup
andterminal124/emptySID confirmed, failuremetadata preserved. Corrected8162
usesoneouterpairlease+explicit--leased/newproducer-v4-run-v2; currentlylive.
VERDICT -> Compileproof only; originalownprojection/device/math/runtime and
posthealth/source/new4/publicreader results pending. Fullgoal remainsactive.


## 2026-10-10 - H50 OFF independent admission and native QSA seam audit

CONFIG -> H50native4OFF actual1874 parentPASS, frozen5f719918/currentSDK37.
COMMAND -> Independentpublicparent_arm5003 terminal0/PASS binding
57caeb3d86dd901fa9349a343b0aeccf32b1085217230e4ec8a71ae88ea33975.
RESULT -> Actualsource/READY/health/ACK/freshbytes/EOF/normalowner/new4/pages
rejoined. NumericalOFFON/all49serial andoriginalmodelmath remainunqualified.

CONFIG -> Observerproposal58f94e94322363e3625b81016f5c0af5cbbeb34d03d6301ba4ebc214a10f1145,
2repo+32actualsource/buildbindings. Sourceonly, no patchorarrayread.
COMMAND -> Rootall34SHA/ASCII matched; actualQSA37/39 kernel/header/callbranch
andflags byteaudit checked. Wholeverify paths differ, no runtimeprooftransfer.
RESULT -> Exactfutureprojection/normRoPE/indexer/KV/attention/gate targethooks,
2/1/1 route/frame/nonce/quota/owner gates andexplicitpair stage1zero specified.
InternalRMSargument/rsqrt/score/softmax registers remainunobserved.
VERDICT -> Proposalonly. CorrectedQSAproducer8162 solepairactor live.
Consumer36beb0peerHOLD reproducedexactclass collectshadow bypass; prospective
successorneeded. ExistingH50/QSA/immutableV2 source/runtime unchanged.


## 2026-10-10 - Own QSA projection producer passes; H50 ON starts

CONFIG -> FrozenQSA e122/source37 +closedHC V3, exactownoriginalweights,
firstnorm/Q81 gate thenstrictprefix4, no capturedmathinputs.
COMMAND -> Correctedactual8162 terminal0/PASS790s, report
61041a2990c6f1b172ff1e8de8f9187805ac4279349e9082fd48136816c31ab3.
RESULT -> First10240normalized/2880Q81 bitwisePASS, fouroriginalownQSA rows/
projection/gamma/state snapshots,577prefix4 comparisons, normalhelperEOF/
retirement/posthealth/new4/source gates. Prefix4head NMSE0.007888246485733295
andmaxnorm0.05251488383778164 remainnonbitwise/numeric_gate_assignedfalse,
matchingunchangedHC V3 arithmetic. Publicproducerreader49222 stilllive.
VERDICT -> Producer/lifecycle proof only; prepareown10helperfiles aftercurrent
reader, thenactualnativeQSA device comparison andnativeobserver remainrequired.
RootH50ONactual1577 acquiredpairlease usingprepared8b46355e; parent/child
admission nowlive. OFF57ca independentadmission preserved. Fullmath/cache/
matchedlatency/shelf unqualified; no secondmodelactor oroldproof transfer.


## 2026-10-10 - Own QSA helper inputs fully prepared

CONFIG -> Originalproducer61041 andindependentreader c028a99b35d3b512cbc490077aad23d755f05073824e55bca531143a6d066b6c;
frozenQSA V4e122, no nativecapturedoperands asmathinputs.
COMMAND -> ActualCPUprepare24675 terminal0; owninputbinding
97a012c1569e8fb93f688ac054bb50790bd9f5da587e5abfd1080f0d5ef4b1d0.
RESULT -> Ten exactown files226514B, independentcurrentSHA/extent rechecked.
Actualdevicecomparison pending H50ON1577solepairactor terminal/posthealth.
VERDICT -> Inputpreparation only, no QSA/norm/RoPE/attention/fullmath gate.
Memorycollector draft root5CPU PASS; task-widechildren/fullactualrecipe/bounded
read gapsclosed, physicalmodel/expert/transient coverage remainsunqualified.
Consumer36beb0HOLD successorpending; allfrozenlive source remainsunchanged.


## 2026-10-10 - Observer40 independently reviewed source checkpoint

CONFIG -> NEW2b4d54291397a2f982a9a687e612e0c23f4c035e2a66c740b37f63b4f1b9f25e,
73dependencyclosure, independentengine72ad297b67sources31headers40patches.
COMMAND -> Root14CPU PASS0.428/all73SHA; peer14PASS0.404/READY forsource/
freshbuild scope. Header-in-patch exact, originalmodellines retained.
RESULT -> DefaultOFF no newobject/alloc/copy/wait/graph; ON actualgroup-local
29targets/2,1,1 ownedwindows/nonces, actualpage0/selectedwidth refusal, stage1
zeroallocation andgraph-before-frees discipline. Captures notmathinputs;
internalnorm/score registers unobserved. No source39 runtimeproof transfer.
VERDICT -> Freshcompile/upload/model/ONOFF/all49/source/health/lifecycle/new4
qualification stillrequired; physicalresidency/fullmodel/latency/shelf open.
H50ON1577child0/collectiontrue actualrequestedN4/3serialjobs; parentposthealth/
kernel pass, finalfull4/currentreaders pending. QSA97a inputs fullyprepared.


## 2026-10-10 - H50 native4 ON parent passes; QSA device control starts

CONFIG -> Frozen5f719918/source37, native4ON prepared8b46355e, sameOFF profile.
COMMAND -> Actual1577 terminal0/PASS1324s, parentSHA
d1b46b2dc0f58d9ca352c9692e86cb24908e43f375c8fabad844e513ef88d5b4.
RESULT -> ActualrequestedN4/multirow warm/currentcounters/rawcoverage,3actual
serialjobs, ownednormalteardown/poststrict+compiledpairhealth/kernel/new4 PASS.
Publicreadonly44836 stilllive. OFF57ca independentadmission alreadypassed.
VERDICT -> Collection/lifecycle only; matchedOFFON andexact-prefixall49serial
comparisons stillrequired. H50serialblockedbydesign; freshsuccessorpending.

COMMAND -> RootactualQSAdevicecontrol46835 acquiredpairlease with--leased,
preparedown97a/sourcee122/producer61041+reader c028. Actualdirect+twofullgraph
replay/raw14field/10echo comparison andposthealth/source/new4 gates pending.
No capturedoperands orIRQfitting/tolerancechange. No secondmodel/GPUactor.
VERDICT -> Realrequirednumericexperiment starts, not numericalqualification.
AllocationV2heldtype/VA gaps; freshV3root10CPU PASS/all7SHA, peerreviewpending.
Fulloriginalmath/cache/interactiveprofile/latency/shelf goal remainsactive.


## 2026-10-10 - Preliminary native4 OFF/ON histories match

CONFIG -> Frozen H50 source37 native4 OFF and ON actual saved runs; matching
slots, input tokens, cancellation slot, max_new, SDK receipt, prepared baseline,
and card topology. OFF independent parent reader passed; ON reader remains live.

COMMAND -> Original native_histories and off_on_histories functions recollected
both saved requests.json files. All eight consumed plan/request/child/parent
files were stat-bracketed, SHA256 bound, and rechecked after comparison.

RESULT -> Three uncanceled requests have exact generated IDs and finish reasons.
Canceled request has four common generated IDs and actual cancellation terminal
in both arms. Complete cancellation counts are deliberately not an equality
requirement under the existing frozen cancellation timing contract.

Artifact: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/batch50-native4-offon-history-preliminary-v1.json
SHA256: 43b1e6a55058753b45e1ec65e014ec69c001454be3fc406f3ba69046945345c5

VERDICT -> Preliminary saved token-history equality only. Independent ON parent
admission, three exact-prefix serial controls and all49 vector comparisons still
remain. No complete model math, cache, fairness, latency or shelf qualification.


## 2026-10-10 - Native4 ON independent reader passes

CONFIG -> Frozen H50 source37 native4 ON actual parent, current operation pack
and SDK entry/predevice/post witnesses; OFF independent57ca already admitted.
COMMAND -> Original reader44836 terminal0/PASS; ON readonly receipt
2b340fe2750bbe3067ee69a7840197935d4135dc06fbf83764b80693435544c8.
RESULT -> Both collection/lifecycle arms independently admitted. Saved history
artifact e2750de4c6cdddc90505edf656cec9704f69eea846b5b553acfce75432f75fce
binds original preliminary43b1 histories, both reader receipts and rechecked
original plan/request/parent/child hashes and stats. Three uncanceled exact
ID/finish trajectories; canceled common four tokens. No new GPU execution.
VERDICT -> Exact-prefix serial all49 remains required. QSA46835 remains live;
fullmath/cache/interactive latency/fairness/shelf goal remains incomplete.


## 2026-10-10 - Own QSA device executes with coherent replays

CONFIG -> Frozen original QSA V4 e122, own input fixture97a, production source37
QSA functions. Exact own original projection operands; direct plus two graph
replays, each restored to own zero state. No captured operand substitution.

COMMAND -> Actual leased control46835 compiled and ran helper. Original
compare_routes recollected all14 full output fields and ten consumed input
SHA echoes against the unchanged own fixture/producer records.

RESULT -> Compile and runtime receipts passed: return_code0, EOF and reader
retirement true, no command or phase-cleanup error. All14 direct/replay fields
bitwise equal. Own inherited candidate versus native RoPE Q/K/IQ NMSE about
3-4e-15; gated attention6.773201341454446e-15. Indexer final tail NMSE
3.5521200573815874e-8/maxnormalized0.00018098385583797445; dead
3.3517877136024584e-8/0.0003151492046816616; pooled
2.3670205512285337e-8/0.00029471256831234486. Comparisons are nonbitwise;
no numerical acceptance gate was assigned. Internal arguments/scores unobserved.

VERDICT -> Actual raw replay coherence and descriptive candidate comparison.
Parent post-health/new4/current-source/final report and independent public
reader remain pending. Larger indexer-state difference requires source-level
localization before changing the independent reference. No full-model math,
normal-model graph, cache, serving latency or shelf qualification follows.

Runtime artifacts: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-layer3-qsa-control-v4-run-v1/


## 2026-10-10 - QSA indexer tail storage discrepancy localized

CONFIG -> Own original raw input97a and actual source37 QSA46835 output;
small384-element final tail only, original independent arithmetic unchanged.
COMMAND -> Agent small-array audit plus root independent NumPy bitwise reads
of raw.f32 and route0/indexer_tail_windows.f32 final window.
RESULT -> Device final tail exactly equals original F32 raw rows0..2. It differs
from FP16-RNE then F32 widening in all384 words. Agent independently confirms
own tail equals that half-roundtrip and own raw matches prepared input. Consumed
native indexer source explicitly includes F32-to-half-RTE-to-F32 conversion.
VERDICT -> Observed storage-contract discrepancy before normalization; compiler
or lowering cause is not qualified. New minimal expression-roundtrip versus
materialized-half discrimination proposed. No tolerance adjustment/reference
shortcut. Parent46835 final admission remains live; full goal incomplete.


## 2026-10-10 - Half-discrimination proposal reviewed, runtime held

CONFIG -> Frozen086b/267repo+19consumed source/build; original own raw512.
COMMAND -> RootSHA/ASCII/AST and7CPU PASS0.013; peer7PASS0.009/allbindings.
RESULT -> Exact production expression versus materializedF16 store/load source
reviewed. Additional root synthetic negatives show truncated finalmarker and
extra routefile accepted byV1collector. Both author and peer source-only scope.
VERDICT -> Preserve frozenV1/nonruntime; new owned lifecycle successor must
close exactmarker/roster/stat/reread/F16widening joins and all original
source/runtime/health/EOF/teardown/full4 gates before execution admission.
QSA46835 stilllive finalsourceadmission, no secondGPU/modelactor.


## 2026-10-10 - Correct actual serial roster; half V2 peer ready

CONFIG -> Actual native4 ON child/serial-jobs.json, unchanged original output.
COMMAND -> Agent metadata inspection plus root independent JSON/SHA read.
RESULT -> Eight actual jobs, not three: admission RID2001..2004 prefix59 and
later RID2001..2004 prefix60. Actual SHA256
5765f0701d4fd7a779c951583cf7a85fa34dbb397e427e220aff292ce2c55f7e.
Two groups under the original six-job bound;392 all49 comparisons required.
Earlier three-job statements counted top-level metadata fields and were wrong.
No solo_migration job exists; do not invent one or claim that gate passed.
VERDICT -> H51 must consume all eight actual jobs; full cancellation/solo
migration qualification remains a separate explicit full-goal requirement.

CONFIG -> New half lifecycle52dbb91d/274repo+19source.
COMMAND -> Root154CPU PASS2.318s; peer154PASS2.299s/allbindings0mismatch.
RESULT -> Strict input/marker/roster/raw/F16widening and original lifecycle
source/health/EOF/full4 gates reviewed. CPU fixture preparation14303 is live;
QSA46835 remains sole pairedGPU/model coordinator in final source validation.
VERDICT -> Source-ready for isolated compile and fresh diagnostic after leases
free. No actual half compile/run, lowering cause or candidate correction.


## 2026-10-10 - Own QSA actual control finalizes PASS

CONFIG -> Frozen QSA V4e122, currentSDK37, independently own projection/input97a.
COMMAND -> Actual46835 terminal0/PASS1324s; reportSHA256
c50be35d4e36af91f4894cc980fcd4d22a2187efa91c34fda9276c5b46a14d13.
RESULT -> Fresh compile and actualruntime0/EOF/owned removal;14 fullfields
bitwise direct+two graphreplays; owninputecho/metadata/flags/library/source/
prepoststrict+compiledpairhealth/kernel/freshcompletefour/pages PASS.
Descriptive owncandidate differences retained, including F32-versus-F16 indexer
tail discrepancy. Fullmath/nativeintrinsics/normalmodelgraph flags false.
VERDICT -> Actual diagnostic/lifecycle only. Independent public32721 is live;
fullmodel native seam via freshobserver40 remainsrequired. Root fresh source40
all8build93799 acquiredpairlease with--leased and clean localggml exactpin;
no runtime ABI transferred. HalfinputCPU14303 remainslive.


## 2026-10-10 - Source40 compile failure and half isolated compile

CONFIG -> Frozen observer40 engine72ad, compiler39992, exact clean source/dependency
pins,40 ordered patches and all8 fresh ABI targets. Compiler exposes no devices.

COMMAND -> Actual fresh build93799 under paired lease, terminal1/197s.
Receipt: /mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T192405Z-dycexnrk/receipt.json

RESULT -> verify.cpp:769 fails on three undeclared identifiers in the observer
native-feature guard: native_qsa_enabled, native_rope_enabled and
native_qsa_indexer_enabled. Compiler identifies kernels namespace declarations.
No device/model execution or fresh SDK admission follows from this failed build.
Original source remains clean and frozen V1 evidence is preserved.

NEW SUCCESSOR -> Engine planV2 d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6,
patch0040V2 3a4e239c81635fc33651944648a4a621e36428789a409fbe29a39e0f8a3df0b8.
Root pristine source reconstruction applies/checks all40 patches and validates
all67 final source hashes. Removing the three added kernels:: qualifiers returns
exact failed V1 verifier bytes. Model arithmetic and all other verifier code
are unchanged. Fresh compile and new C140 controller/parent/oracle bindings
must use V2 explicitly; the old failed build cannot transfer.

VERDICT -> Localized compilation error and source correction only. Independent
review, fresh all8 compilation, source admission, new oracle/upload, model-target
observations, full fidelity/cache/concurrency/latency/shelf remain required.

CONFIG -> Frozen half52dbb91d/card0 isolatedcompile, no helper execution.
COMMAND -> Actual18096 terminal0/PASS21s; isolatedbinding
21c3221523c3b8c99b9722a3fe0a886912d6cae1e469e1048285d1be6467b9c1.
RESULT -> Fresh ELF/exactsource/argv/image/boundedrecipe/normalownedremoval PASS.
VERDICT -> Compile-only; ownfixture14303 stilllive, realhalfdiagnostic pending.
QSApublic32721 remainslive. H51root346SHA/ASCII/AST+25CPU PASS0.239;peerpending.


## 2026-10-10 - Initial H51 reader held after independent review

CONFIG -> Frozen ffccc8227bd375af73c837592f4f31a94ce497155154fa93db8a982d73f7b3ab,
346 source bindings, original source37/H50 collector and new serial51.

COMMAND -> Root346SHA/ASCII/AST and25CPU PASS0.239s; peer same closure and25
CPU PASS0.248s, plus a synthetic contradictory-command/terminal control.

RESULT -> serial_vectors returns all49 vectors without reading or validating
group result.json and command.json. Peer supplied result passedfalse/engine_rc9/
removedfalse/error present and foreign command recipe, while raw vectors were
otherwise consistent. Artifact hashes do not establish these semantic joins.
READY/ACK freshness, source/byte and individual raw joins passed scoped review.

VERDICT -> Initial H51 remains frozen NONREADY for execution/public admission.
New successor must independently validate actual group terminal/exit/OOM/removal/
error, reconstruct full command from original plan/group and actual child PID,
and join child success/collection/counts/plan SHA. Preserve all eight actual
jobs,392 vector comparisons and absent-PIN/cacheOFF semantics. No runtime
serial qualification, concurrency/cache/latency/fullmodel claim follows.


## 2026-10-10 - C140V2 source consumer freeze reviewed

CONFIG -> NEW3121080fd2663582a3fbffbc814c3ae403b09002b5131c03d5803143a907bd46,
138-source closure; corrected source40V2 engine d87740b5, all67/31/40/8/6.
COMMAND -> Root138SHA/ASCII/AST and15CPU PASS0.966s; author15PASS0.954s.
Pristine namespace-only reconstruction/Undo previously independently passed.
RESULT -> Newcontroller0fb8fc38/parenta6955761/uploadplan56fae38e requirefresh
corrected SDK/neworacle/upload/new1402 onecard and pair admission. FailedV1/
1401source or runtime cannot transfer. New research alias proposal retainsIDs
and corrects launch toV2; canonicalregistry is unchanged.
VERDICT -> Reviewed SOURCE/preparation only. Actual build71854 remainslive;
neworacle/upload/modeltarget/math/cache/concurrency/latency/shelf remainrequired.
Own half fixture14303 completed0, inputbinding2d9862b6/raw2048 de82f9d4.
Isolatedhalfcompile18096 passed; realdeviceexperiment awaits solebuildterminal.
QSA public32721 stilllive; existing reportc50be35d diagnostic scope retained.


## 2026-10-10 - Corrected source40 build and fresh oracle pass

CONFIG -> Corrected d87740b5/67source/31header/40patch/all8/sixPython, exact
compiler39992/clean localggml3cf03257; no devices exposed during SDK build.
COMMAND -> Actual71854 terminal0/PASS321s, SDK
/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261010T193019Z-1kl6uepo,
receiptb3fe2a932995f5d7b52e89c85a6a4212a1458d3a513dbd4d92b883c89afbbec3.
RESULT -> All8 fresh ELF targets/currentsource admission PASS c80c492e.
Fresh whole390 oracle99195 terminal0/PASS45s against exact newSDK/56fae38eplan,
receipt0cd9a3c7f08afcba509b525f9aad74d53b497a603ff85722d2fe44e716fffd76
at /mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-bhya9nil/receipt.json.
Independent metadata oracleadmission287cd52e passed currentactualarchive/ELF.
VERDICT -> Compilation/source-only; actualupload, C140modelbaseline/targets,
complete math/cache/concurrency/latency/shelf stillrequired. FailedV1 preserved.

COMMAND -> Actualhalf27860 acquiredpairlease with--leased/current52dbsource/
own2d9862b6fixture/card0baseline. Source/producer validation live beforedevice.
PublicQSA32721 remainslive CPUonly. No secondGPU/modelactor.
VERDICT -> Actualdirect/materialized512 comparison andteardown/posthealth/new4
not yet observed; no rounding/lowering cause or reference correction assigned.


## 2026-10-10 - QSA public admission closes; H52 peer ready

CONFIG -> ActualQSA control c50be35d/current frozene122 and original owninputs.
COMMAND -> Public32721 terminal0/PASS; readonlybinding
24077dfc6bc33699fffec871a705e715241f44ad9f98d384bbc98af3534037f7.
RESULT -> Original source/producer/fixture/14field/direct-replay/actual recipe/
receipt/EOF/normalterminal/health/journal/full4/page joins independently admitted.
VERDICT -> Diagnostic replay/lifecycle only; original fullmodel math/cache/
concurrency/latency/shelf remain incomplete.

CONFIG -> NEW H52 944e252bb89d088d23cfb1c29b86e1a639e402e9dc4df6fb17462b53c1eeb0c5
362-file closure, serial-only source37 versus genuine originalH50 collector50.
COMMAND -> Root362SHA/ASCII/AST+30CPU PASS0.257s; peer30PASS0.254/allbindings.
Root independently exercised real group.binding synthetic files beforefreeze:
consistentpositive admitted, foreignrecipe and contradictoryDockerError refused.
RESULT -> H51terminal/recipe gap closed innewH52; all8actualjobs/2groups6,2/
392full49 pairs preregistered, absentPIN/fresh1/cacheOFF unchanged; no solojob
invented. Fullactualcommand/counts/individualJSON/raw49/group/source/lifecycle
joins retained. CPUprepare77157group0 and61871group1 nowlive, no model actor.
VERDICT -> Source/preparation only. Actualtwo serialarms andfinal392comparison
remainrequired. Half27860 solepairGPU/modelparent live beforedevicevalidation.


## 2026-10-10 - Actual half expression/storage discrimination

CONFIG -> Frozen half52db, same production source37 indexer object flags and
compiler/runtime identities;512 independently original-projected F32 values.
Original reference and QSA V4 remain unchanged.

COMMAND -> Actual27860 fresh helper compiled/executed, normal runtime exit0,
EOF/reader retirement passed. Root original strict compare_routes independently
recollected all4 fullfields, own input echoes and direct+two graphreplay bytes.

RESULT -> expression.f32 equals original F32 input BITWISE; SHA
 de82f9d41a59b874c1973cd95d9a51091ccbcc3911b4fa01e51ee7e338f39226.
materialized.f16 equals independently host FP16-RNE bytes BITWISE; SHA
 e7439a01697a2d1c107bda3c5f58d56f123f921d3a806990406c0d709bcd41ee.
materialized.f32 equals host half-widened F32 BITWISE and independently rejoins
actual materialized half bytes; SHA
 5b4022292463cb2ee026537ff257c48fabe22b8e28c711bd928297021488bf2e.
All512 own input echoes and all4 fields match across direct and two replays.
Post strict/compiled-pair health, kernel and newfull4 receipts passed. Parent
final source/fixture/baseline admission remains live; public reader not yet run.

VERDICT -> Actual expression-versus-storage behavior discriminated on this
precise input/source/compiler/device configuration. Compiled lowering/IR cause
remains unobserved and is the next diagnostic. No candidate modification,
tolerance fitting, universal FP16 claim or full-model/native QSA qualification.

Runtime root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/owned-indexer-half-control-v2-run-v1/

CONFIG -> Frozen compactconsumerV2 c756/400.
COMMAND -> Root400SHA/ASCII/AST andexact52CPU PASS21.049s;peerpending.
RESULT -> Source/CPU only; no matchedreal admission speedup/integration yet.
VERDICT -> Preserveold400-independent scopes; benchmarksource workcontinues.
Hostallocator CPU-only -fno-sycl/noGPU fixture: OFF0 andON14balancedJSONrecords,
originalvectorbytes/copy/move preserved. Initialhostg++unavailable and
unconditionalimage setvars attempt exited3; correctedconditionalCPUrun passed.
Draftsource wasnotfrozen and no closednative/source/runtime proof isclaimed.


## 2026-10-10 - Compact parser consumer V2 independently reviewed

CONFIG -> NEW c75684185ff69f4174c4fc2e7ce1f2025640177246403562f68cb209a313e9a9,
400-file closure; immutable workerV2 source/runtime owned operation evidence.
COMMAND -> Root400SHA/ASCII/AST+exact52CPU PASS21.049s; peer52PASS21.094s.
Extra root exploratory53-suite pass is not the frozen52 command/result.
RESULT -> Exact expected method/global/alias/guard code/default/namespace checks
before dispatch; ordinary shadow/cache injection refused; genuine own worker
packet/PID/start/1positive+3negative/normalretirement provenance retained.
All nonparser guards/returns reconstruct underASTUndo; live source/stat/current
SDK/pack/model/health/lease predicates remain outside immutable parse reuse.
Original V1 c36beb0 is held and preserved; H50/H51/H52 unchanged.
VERDICT -> Reviewed source/CPU only. Actual data matchedABBA benchmark and new
prepare/runtime integration stillrequired before any dev-loop speedup claim.
Half27860 andtwo H52CPU preparations77157/61871 remainlive.


## 2026-10-10 - Half final PASS; corrected source40 upload plan

CONFIG -> Actualhalf52db/ownfixture2d9862b6/source37.
COMMAND -> Actual27860 terminal0/PASS1349s; report
5e66575dcf49c66fd1118c9af14e118217fa672ea5ed42217e506cba1b110cc3.
RESULT -> Expression preservesF32; actualmaterializedF16 matcheshostRNE;
allfour/direct+two replay fields, normalownedterminal/EOF, current source/
preposthealth/kernel/freshfour/pages PASS. Public68272 remainslive.
VERDICT -> Diagnostic/lifecycle only; lowering cause and fullmodel notqualified.

CONFIG -> Source40V2 upload56fae38e versus actualSDKd877 patch0040V2.
COMMAND -> Actualsource40-v2-upload390-run-v1 failed0s beforehealth/device.
RESULT -> Requirednewpatch stillV1/f99d whileSDK containsV2/3a4e239c. Frozen
V2 and failed receipt preserved; no GPU recovery needed for predevice failure.
VERDICT -> NEW C140V3 b7e6f1da/150 andupload20e22d enforce exactcorrected
patch beforeSDK/oracleadmission. Root18CPU PASS0.999s/allSHA;peer18PASS1.033.
SDK d877 unchanged, no ABI rebuild. Freshoracle63898 passed45s, receipt
92202b035ea15efd545b0e878e2b112f6cc0aaa042a361e2539c492204be15dc,
at /mnt/vm_8tb/b70/build/strata-source-upload-oracle-full-ji1eroif/receipt.json;
currentmetadataadmissionc1c8ae4e PASS. Actualnewupload65169 acquiredpairlease,
all390sourcepayload/health/normalretirement/newfour gates pending. No registry
mutation or modelbaseline/target/math/cache/latency proof transferred.

CONFIG -> HalfLLVMrecipeabee/278+19source, originalsource/mathunchanged.
COMMAND -> Actual74202 normalEOF/cleanup butLLVMphase1, 'IR output is not
supported'; toolsphase0. No newIR/ISA or runtimehelperexecution.
RESULT -> Absolutecompiler-root metadata inventory foundbundledclang++/
clang-offload-bundler/extract/llvm-objcopy/llvm-spirv/sycl-post-link outsidePATH.
Only frontend retained flags can be compared; omittedbackend/linkcontrols
remainseparate, so no identicaldevice-lowering claim. NewrecipeV2 pending.
VERDICT -> Preservefailure; compiledcause stillunobserved. TwoH52CPU prepares
andhalfpublicreader live. Fulloriginalgoal remains incomplete.


## 2026-10-10 - Actual matched admission benchmark passes

CONFIG -> Frozen benchmarkV2 d321495c/412 and compactconsumerV2 c756/400,
current actual source37 SDK/pack/upload evidence. Two ABBA rounds, eight trials,
first plus four repeated calls per trial. Existing outer upload guards run every
call; fresh logical/SDK/pack entry/predevice/post byte edges for both arms.
Known concurrent test/compiler jobs were held; perfect host isolation and OS
page-cache coldness are not claimed. No GPU/model inference.

COMMAND -> Actual60152 terminal0/PASS. Report
172ff115da00da97fb95aec646eb852684ad1a8e63c696ab4a79b6d0e90ea7f3.
All412 frozen source hashes rechecked after execution; all trial artifact SHAs
checked. Old/new outputs equal, wrong-owner latch and owned-copy mutation
controls pass, original inputs remain unchanged.

RESULT -> Original A repeated-gate median1.255748557s; reused B0.225575324s:
5.566870235x faster. First-call median A1.2387491175s versus B2.6596827925s,
so the first call is slower. Complete operation median (setup, first+four
repeats, generation guards and final byte/provenance closure) A9.9216711521s
versus B7.5933243036s,23.467285percent lower. Four trials/twenty timed calls per
arm; sixteen repeated observations per arm. Independent final reader pending.

VERDICT -> Measured CPU admission reuse benefit on this configuration, with
first-call cost. No complete harness preparation, decode tok/s, TTFT, serving
latency, model correctness/cache/shelf or OS-cold claim. New H54 integration
f7a473a4/480 passed root39CPU2.700s and peer scope review; root group0/1 CPU
preparations66898/73653 are live before any model execution.

Report root: /mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010/matched-upload-logical-gate-benchmark-v2-run-v1/
Summary SHA: c59e0a52ef16ea8d7092d3a3f6eb0536f538a8dc61496d3a81b18ba3e293317b

Historical failures preserved: benchmarkV1 da627 failed before trial0 because
its factory expected absent ledger log_sha256. H53 6a74 remains held because
its saved reader accepted mutated packet/argv/start fields. Successors use the
actual ledger schema and exact saved-worker provenance; prior files unchanged.

Actualsource40V3upload65169 terminal0/PASS295s af2cad6be95ce8ad7ede3d1df7b6be31150c787d448a0f688b6cc541eb74f5d1;
all390source/health/normalowned/newfour gates pass, no modelmath transfer.
Halfpublic68272 terminal0/PASS e1789d5c870a522cbb23931a96b3e87c470f5dffc16046f93e711b134c9cb727.
H52originaltwo6/2plans prepared0 e03c9982/ffb052ab; no execution.
Driverprobes36246 all10 commands0, collector admissionfailed missing
command_sha256. Originalrecords preserved; newcollector62049 live withproper
commandhash field, no IR/JIT/helperexecution or referencechange.


## 2026-10-10 - Benchmark full recollection and H54 actual serial start

CONFIG -> ActualmatchedV2 report172ff115/rootderived summaryc59e0a52;
newexplicit readonlyV3 source cbe1d9d8/419 retains originalnine-file V2 refusal.
COMMAND -> Agentmetadata/log recollection06afd345 PASS; rootfullreader56730
terminal0/PASS, receipt67464601f4ccc8897883c262715f2410d0066055091a846fe486f70b48714a54.
RESULT -> All8trial SHA/order/source/old-newresult/workerpacket/PID/start/count/
retirement/currentlogical bytes rejoined. Root fullSDK/pack threebyteedges and
live old/new gate/freshnegatives passed; original negative temporarycopy bytes
are unavailable and explicitly not currentlyrejoined. Summary metrics recomputed.
VERDICT -> CPUadmission measurement only, no inference/serving/wholeprepare claim.

CONFIG -> H54 originalN4 collectorH50, exact8jobs in6/2 groups.
COMMAND -> CPUprepare66898/73653 terminal0; plans2f547043 and3352b279.
Actualgroup0 qualifier2165 selfleasedpair/expectedSHA2f547 beforemodel work.
RESULT -> Newmemo integration keeps alloriginal model/health/source/byte/
absentPIN/group392 gates; actualserial results pending. Group1 not launched.
VERDICT -> Full392 compares/normalteardown/posthealth/new4/public joins required;
original model math/cache/latency/shelf remain open.

CONFIG -> Originalhalf full-driver metadata/support b5c/282+19.
COMMAND -> Correctedrootprobe62049 terminal0/PASS101s, binding733dc7aa.
Freshsameflag save-temps33497 terminal0/PASS22s db27e681/198artifacts; helper
unexecuted and originalexecutedELF unchanged. Source/helper reference unchanged.
RESULT -> Saved sixdirect/wrapper Expression/Store/Load bitcodes identified.
Textview99657 failednormal10s 'IR output is not supported'; no textualIR/cause
or originalruntimeJITISA claim. Root/probe failures preserved, supported next
bitcode/SPIRV extraction source work delegated.
VERDICT -> Compiledartifact capture only. Actualconversioncause remainsunproven.


## 2026-10-10 - Allocator producer V2 source/CPU and real host fixtures

CONFIG -> NEW00f903c277f6bce58a34d6de82c0e416f4549cae5403e8cafde017c2e3eb76c3,
162repo+4references/73proposedfiles; no integratedgeneration assigned.
COMMAND -> RootallSHA/ASCII/AST and31CPU PASS0.876s; peer31PASS0.863.
Rootsnapshot C++fixtures79638 undercompiler39992/-fno-sycl, noGPUdevices,
networknone/1CPU/1GB, explicitsyntheticgeneration1000000; terminal0/PASS.
RESULT -> OFFzeroevents; hostON14 balancedactualstdallocator/free records;
original vectorbytes/copy/move/size/alignment assertions pass. Concurrent
regression emits4balancedrecords, actuallyreuses same pointer and proves
registration waits untilsuccessful free+retirement. Scopedownersallreleased.
Receipt30fb903a58548dbc499e75b82d0202492e6f3d8174a4c38953859e3d796eb2cb
at f17/allocator-v2-host-cpu-fixture-v1/receipt.json; exactsource/tool/argv pinned.
V2 per-role/context/stage placement coverage and payload/directory tags close
V1peer gaps; original9a1 held/preserved.
VERDICT -> ActualhostCPUfixture and sourceproposal only. No nativeSYCL model/
physicalresidency/fullcache/wholepeak proof. Freshall8 ABI/oracle/actualmodel/
ONOFF numerical/healthy teardown qualification remainsrequired.

CONFIG -> Supported newSPIRVtext704ff628/286repo+19source+15smallcode/help.
COMMAND -> RootallSHA/ASCII/AST+3CPU PASS0.148s; independentreview pending.
Derivedtranslations retain actualoriginaltranslator options; no actualnew
SPIRV/text, originalembeddedELF/JIT or conversioncause observed.
VERDICT -> Execute onlyafterliveH54group0 qualifier2165terminal. Alloriginal
math/reference andcanonical165registry remainunchanged; fullgoal stillactive.


## 2026-10-10 - SPIRV text recipe reviewed; H54 real model starts

CONFIG -> NEW704ff628d0768f02cd36573c457f12bcf3932031265774a7fa618859cd56eed4,
286repo+19consumedsource+15smallcode/help bindings, sixsavedpostlinkmodules.
COMMAND -> RootallSHA/ASCII/AST+3CPU PASS0.148s;peer3PASS0.162s/all286SHA.
RESULT -> Exactobservedtranslator options and supported to-text preregistered,
inputbitcode/symbol/capture receipts bound; newtranslations explicitlydistinct
from originalexecutedELF/JIT. No actualSPIRV/text/cause/NNreference change.
VERDICT -> Source-ready for rootcompile onlyafterlive2165leasesfree.

CONFIG -> ActualH54group0 originalsource37/SKD/full6-prefix vector controls.
COMMAND -> 2165 childsemantic READY thenfreshstrict/compiledpairhealth/kernel
andACK observed; originalrecordedhealth/ACK epochs retained, source/byte seals.
RESULT -> Actualserial modelprocess launched, ARM/capture/command/engine log
exists. Requests and294 vectorcomparisons stillpending; no earlyPASS claimed.
VERDICT -> Retainpairlease until normalownedteardown/posthealth/full4/current
source andpublic joins. Group1 two-prefix/98comparison arm unexecuted.


## 2026-10-10 - H54 group0 actual294 vectors match bitwise

CONFIG -> Exact6 observed H50N4 consumed prefixes, original source37 SDK/math,
H54sourcef7a4 and prepared2f547; absentPIN/fresh1/cacheOFF serial controls.
COMMAND -> Actual2165 child finished; root reread serial-comparison.json all
294 rows, bitwise flags/equal batch+serialSHA/zeroNMSE/maxnormalized checked.
RESULT -> ChildpassedTrue/6jobs/294pairs; actualmodelnormalexit0/errorNone/
removedTrue. ComparisonSHA14b7f3c088397a0aafcc364566357aaed4d8f4f0503d5f6dadfe3df27df2978b.
Parentpost-strict andcompiledpairhealth are live; finalsource/newfour/public
admission stillpending. Remaininggroup1 twojobs/98pairs hasnotexecuted.
VERDICT -> Preliminary actual batch-versus-serial numerical equality only.
No complete392suite/lifecycle/public/originalmodelmath/cache/latency claim.
Normalparentfinalization and group1 remainrequired; fullgoal unchanged.


## 2026-10-10 - H54 group0 parent finalizes PASS

CONFIG -> Frozengroup0 plan2f547043/H54f7a4/source37,6actualprefixes/294pairs.
COMMAND -> Actual2165 terminal0/PASS1013s, parentSHA
2d007308900ab7d4275376307bdc1029e9bf169c67494e1b05bd0d918539b960.
RESULT -> All294 exactbatch-versus-serial bytecomparisons; modelnormalexit0/
removal, READY/freshhealth/ACK/predevice/currentbyte/EOF/poststrict+compiledpair/
kernel/newcompletefour/source/page gates passed. Independentpublic9370 islive.
No forcedcleanup. Group1 98pairs hasnotexecuted.
VERDICT -> Actual scoped serial numerical/lifecycle result; independentreader
andremaining98/full392join stillrequired. Fullmodelmath/cache/latency/shelf open.

COMMAND -> RootSPIRV65718 acquiredcard0lease/pin0 forreviewed704f sixsaved
moduletranslations; no model/kernel execution, originalELF/reference unchanged.
VERDICT -> ActualSPIRV/text/cause observation pending; no new inference claim.


## 2026-10-10 - Fresh saved SPIRV retains half conversions

CONFIG -> Reviewed704f recipe, genuine db27 saved bitcode/symbols, exact observed
original translator options. Six direct/graph-wrapper modules. Fresh translation
is separate from original executed ELF and original runtime JIT ISA.

COMMAND -> Actual65718 under card0 lease/pin, terminal0/PASS60s. Capture binding
ceb15c5f0f2f95690bad1b21cc1714afff080ccfc5fab697ef4c48047fd5afa9.

RESULT -> Direct Half37Expression(module33) and graph wrapper(module21) contain
32-bit float Load, FConvert to TypeFloat16, FConvert back to TypeFloat32 and
32-bit Store. Both results have FPRoundingMode0 decorations. Materialized
store modules36/24 contain FConvert32-to16 and16-bit Store; load modules38/25
contain16-bit Load, FConvert16-to32 and32-bit Store. All six current text/code
artifacts and original input/module/source hashes retained; helper not executed.

VERDICT -> The fresh post-link translation retains the declared conversion
roundtrip; it does not show frontend deletion. Runtime expression identity versus
materialized rounding is already observed, but original embedded image/JIT ISA
and execution cause remain unproven. No model/reference/tolerance modification
or broad original-model fidelity claim. Next extraction must distinguish original
executed ELF from fresh/offline artifacts and pin actual tools/target/IGC.

Runtime artifacts: /mnt/vm_8tb/b70/build/half37-spirv-text-v3-20261010/

ActualH54group1 qualifier82490 selfleasedpair/prepared3352b279 forremaining
2jobs/98pairs; no secondGPUactor. Group0currentpublic9370 remainslive.
Complete392final/public/fullmodel/cache/latency qualifications stillrequired.


## 2026-10-10 - H54 group0 independent current-byte reader passes

CONFIG -> Original H54f7a4/source37/group0 plan2f547, six actual H50N4
prefixes and294 full49 batch-versus-serial comparisons.
COMMAND -> Poll original reader9370; terminal0/PASS. Reread saved parent/child
status and exact output SHA256.
RESULT -> Current logical/pack/SDK and finalized original numerical/lifecycle
joins pass. Receipt f17/batch54-native4-serial-group0-readonly-binding-v1.json
SHA2565fb305c046e5199815bfbb07a5d87c1c0ac692348bca4148e5d9724fb8b318e0.
Group1 original qualifier82490 remains live; actual child1142789 in admission.
VERDICT -> Scoped294 qualification closed; remaining98/full392 join pending.
Original model fidelity/full cache/interactive serving latency/shelf unqualified.


## 2026-10-10 - Original executed ELF extraction V4 source ready

CONFIG -> New frozen a674e0f070259b717ba65d67f4e22750750a1adcb629265e03b8ef0ad2c3f2ed,
292 repository/19 consumed source/18 small code and help bindings. Original
executed helper4d86d9b8 and original report5e66575d/publice1789d5c unchanged.
COMMAND -> Root verifies every hash/ASCII/AST and exact ten CPU controls PASS
0.162s. Original pair qualifier82490 child reaches READY after admission; root
starts CPU-only extract/current public recollection37439, no GPU tool launch.
RESULT -> Source/parser/publication negative controls pass. Actual extraction
and independent review pending. Full current original runtime/source/model
identity joins are required; this operation is not merely a small-code read.
VERDICT -> Supported original image extraction only, no original JIT/ISA or
conversion cause/model fidelity qualification. Tool translation waits until
the sole pair actor finishes. Full original goal remains active.


## 2026-10-10 - Original executed embedded half conversions observed

CONFIG -> Frozen originalELF extractor a674e0f0, unchanged executed helper
4d86d9b8d50655949cd30ded27e20337e7ed03ab20549ba29915dae9abcedb63.
COMMAND -> Root bounded symbol-extent ELF/SPIRV parsing; all six original
entrypoints and module SHA256 identities retained. No tool/kernel execution.
RESULT -> Original direct33 and graph-wrapper21 Expression each retain two
FConvert instructions forming float32->float16->float32. Original store36/24
and load38/25 each retain one conversion and distinct float16 types. Receipt
f17/half37-original-elf-bounded-code-observation-v4.json SHA256
8c2b23561ae502a8302f8d173c0913d2de33e3f0790c76b42760f30862c03f68.
Full current original public/runtime/source recollection37439 still live; this
bounded-code receipt explicitly does not claim that admission is complete.
VERDICT -> Conversion deletion before original embedded SPIRV is unsupported
by these observed instructions. Runtime JIT/ISA and discrepancy cause remain
unproven; original model/reference/tolerances unchanged. H54group1 original
qualifier82490 has actual READY/health/ACK; remaining98 qualification pending.
