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
