# Qwen3.8-27B W4A8 full-send -- campaign journal

Standing prompt: `docs/20260820_qwen38_w4a8_campaign.md`.
Ledger: `docs/20260820_qwen38_w4a8_loops.md`.
Dead-ends: `docs/20260820_qwen38_w4a8_deadends.md`.

This file is the **evidence log for this campaign only**. Newest
entry at the bottom. Root `JOURNAL.md` gets a one-line pointer per
loop, not the full writeup. ASCII only.

Shape (copy):

```
### YYYY-MM-DD<letter> - LOOP N: <one line>

CONTEXT ->
CONFIG ->
COMMAND ->
RESULT ->
VERDICT ->
```

LOOP 0 (plan, no GPU) is in root JOURNAL `2026-08-20bv`. Day-1 GPU
starts here.

---

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
  Log: results/logs/151_qwen38_w4a8_20260821_001757.log

VERDICT -> GO. Pipeline smoke artifact
  exists. GPTQ fire 2 after load-gate.
  Do not bake. Do not start DD.

---

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
  Log: results/logs/k1_w4a8_shapes_20260821T001857Z.log
  CSV: results/logs/k1_w4a8_shapes_20260821T001857Z.csv

VERDICT -> GO. Split-M: decode Path H,
  large-M Path X (or W8A8 TOPS). N=96
  keep BF16. Do not mix H/X unnamed.

---

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
  Log: results/logs/k0_census_w4a8_rtn_gdn_20260821.txt

VERDICT -> File census GO. Next: GRAPH=0
  vLLM smoke card 1, served id
  qwen3.8-27b-W4A8-rtn-gdn. Do not GPTQ
  until load-gate. Do not start DD.
  P2PACCESS=0.

---

### 2026-08-21d - LOOP 4: campaign journal + arm 30m loop

CONTEXT -> Operator: journal this campaign
  in its own doc; commit+push at milestones;
  when does next loop start; /loop 30m on
  the campaign doc if not armed.

CONFIG -> scheduler_list was empty. Cards
  free. NEXT PICK still GRAPH=0 smoke.
  Durable 30m, fire_immediately (slash
  /loop semantics). P2PACCESS=0. DD PARKED.

COMMAND -> wrote this file. Pointed
  campaign read-order + section 13 here.
  Root JOURNAL.md one-line pointer only
  from this heading on. scheduler_create
  30m.

RESULT -> scheduler `01a021be5649` 30m
  durable fire_immediately. First fire
  is GRAPH=0 smoke (LOOP 5+).

VERDICT -> GO (steer). Next loop starts
  on arm (fire_immediately), then every
  30m. Do not start DD. Do not bake.

---

### 2026-08-21e - LOOP 5: GRAPH=0 vLLM smoke load-gate GO

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  GRAPH=0 smoke of w4a8-rtn-gdn. Cards free.
  18080 down. Artifact on disk as CausalLM
  config with VLM tensor names.

CONFIG -> IMG=int8g-v0260 TP=1 GRAPH=0
  DTYPE=float16 NOMM=1 B70_NOMTP=1
  B70_W4A8_HYBRID=0 P2PACCESS=0
  PORT=18081 NAME=qwen38_w4a8_rtn
  DEVICE=1 MAXLEN=8192 UTIL=0.85
  CKPT=/models/qwen3.8-27b/w4a8-rtn-gdn
  SERVED=qwen3.8-27b-W4A8-rtn-gdn
  3.6 shelf serve.sh via vllm/w4a8/serve_qwen38_w4a8.sh
  VLM wrapper spliced from bf16 (graft_qwen38 pattern).

COMMAND ->
  ```
  python3 vllm/w4a8/wrap_vlm_config.py
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l5-smoke \
    ./bin/gpu-run --card 1 \
    env GRAPH=0 NOMM=1 B70_NOMTP=1 PORT=18081 \
      NAME=qwen38_w4a8_rtn P2PACCESS=0 \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  # first start EXITED 52s: fused in_proj_qkvz got W4A8
  # patched config group_0 += in_proj_qkvz; ignore in_proj_ba
  # retry w4a8-l5b-smoke -> HEALTHY 66s
  ```

RESULT -> First start: AssertionError
  load_merged_column_weight. vLLM packed
  in_proj_qkvz=[qkv,z]; 151 regex was
  in_proj_qkv$ / in_proj_z$ so fused
  Linear fell through to W4A8 packed int4
  then loaded I8 shards.
  Retry after target/ignore patch:
  HEALTHY 66s. Served id exact.
  Paris: "Paris." 391: "391". Fib:
  iterative a,b=0,1. Coherence OK.
  Kernels: XPUW4A8IntLinearKernel +
  XPUInt8ScaledMMLinearKernel. int4_gemm_w4a8
  fake already registered (op present).
  VLLM_W4A8_PREPACKED unknown-env warn on
  0.26; prepack still engaged (66s load,
  no 28 GiB unpack).
  Logs: results/logs/l5_w4a8_rtn_graph0_20260821T003813Z.log
        results/logs/l5_w4a8_rtn_graph0_crash_engine.log
        results/logs/l5b_w4a8_rtn_graph0_20260821T004125Z.log
  Serve left Up :18081.

VERDICT -> GO. K0 dispatch+load-gate green.
  Leave GRAPH=0 up. GPTQ fire 2 is unblocked
  for card 0. Do not start DD. Do not bake.

---

### 2026-08-21f - LOOP 6: 151 GPTQ fire 2 STARTED

CONTEXT -> 30m fire. Load-gate GO. NEXT PICK
  card 0 GPTQ. Card 1 serve Up, leave it.

CONFIG -> DATAFREE=0 METHOD=gptq
  SMOOTHQUANT=selective SAMPLES=128 CARD=0
  IMG=int8g-v0260
  OUT=models/files/qwen3.8-27b/w4a8-gptq-gdn
  P2PACCESS=0. DD parked.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-151-gptq \
    ./bin/gpu-run --card 0 \
    env DATAFREE=0 METHOD=gptq SMOOTHQUANT=selective \
      SAMPLES=128 CARD=0 \
    bash scripts/151_quantize_qwen38_27b_w4a8.sh
  ```

RESULT -> STARTED. pid=353913. GDN hit 144/144.
  selective-sq attn=16 mlp=64 mappings=80.
  SequentialPipeline inferred. Same root
  "Could not match re:.*linear_attn" red herring
  as RTN. Log:
  results/logs/151_qwen38_w4a8_20260821_010557.log
  Hours remaining. Do not start a second 151.

VERDICT -> STARTED. Next fire ATTACH this job.

---

### 2026-08-21g - LOOP 7: GRAPH=0 attach tok/s

CONTEXT -> Card 1 serve already Up. Attach,
  do not start a second serve. GPTQ on card 0.

CONFIG -> :18081 qwen3.8-27b-W4A8-rtn-gdn
  GRAPH=0 TP=1 eager HYBRID=0 NOMM=1

COMMAND -> timed /v1/completions greedy
  warm 8 tok, then 128 tok, then Paris 24.

RESULT -> warm 6.39 tok/s. dec128 6.34 tok/s
  (128 tok / 20.19s wall, includes TTFT).
  Paris still "Paris." Not bench_code c1.
  Honest GRAPH=0 eager 1-card wall.

VERDICT -> GO as an attach number. Do not
  demote 31.9 / 43.8 / 65.08. GRAPH=1 is
  the decode lever. Leave serve Up.

---

### 2026-08-21h - LOOP 8: ATTACH GPTQ layer 9/64

CONTEXT -> 30m fire. NEXT PICK attach 151 GPTQ.
  Do not start a second 151. Serve :18081 Up.

CONFIG -> card0 HELD w4a8-151-gptq pid=353913
  since 01:05:57. q151_quant Up 29 min.
  DATAFREE=0 gptq selective 128.

COMMAND -> read log
  results/logs/151_qwen38_w4a8_20260821_010557.log
  ps -p 353913; docker ps q151_quant

RESULT -> RUNNING. 62 GPTQ module steps,
  layers 0-9 of 0-63. Last:
  layers.9.mlp.down_proj. Mean step 11.8s
  max 54s (down_proj). ~2.8 min/layer ->
  ~2.5h remaining + save. No Traceback.
  OUT not written yet. :18081 still
  qwen3.8-27b-W4A8-rtn-gdn. 18080 down.
  gate_proj recon error ~5e3 vs out_proj
  0.44 (scale, not a stop).

VERDICT -> RUNNING. ETA ~2.5h. Next fire
  attach again. Do not steal. Do not GRAPH=1
  while this serve is the research holder.

---

### 2026-08-21i - LOOP 9: ATTACH GPTQ layer 30/64

CONTEXT -> 30m fire. NEXT PICK attach 151.
  Do not start a second 151. :18081 Up.

CONFIG -> card0 still HELD pid=353913
  elapsed 59m. q151_quant Up 59 min.

COMMAND -> parse
  results/logs/151_qwen38_w4a8_20260821_010557.log

RESULT -> RUNNING. 188 Quantizing lines,
  layers 0-30. Last:
  layers.30.linear_attn.in_proj_qkv at
  02:05:46Z. 187 GPTQ steps, mean 11.6s
  max 55s, sum 36 min compute. Pace since
  LOOP 8: +21 layers / 30 min. Remaining
  ~34 layers * ~1.4 min ~ 50 min + save.
  No Traceback. OUT not on disk. Serve
  id still qwen3.8-27b-W4A8-rtn-gdn.
  18080 down.

VERDICT -> RUNNING. Faster than LOOP 8
  ETA. Next fire attach; if DONE then
  wrap_vlm_config + census. Do not steal.

---

### 2026-08-21j - LOOP 10: ATTACH GPTQ layer 49/64

CONTEXT -> 30m fire. NEXT PICK attach 151.
  pid=353913 elapsed 89m. :18081 Up.

CONFIG -> same GPTQ job. Do not start a
  second 151.

COMMAND -> parse
  results/logs/151_qwen38_w4a8_20260821_010557.log

RESULT -> RUNNING. 308 Quantizing lines,
  layers 0-49. Last:
  layers.49.linear_attn.in_proj_z at
  02:35:34Z. 307 GPTQ steps, mean 11.8s
  max 56s, sum 60 min compute. Pace +19
  layers / 30m. Remaining ~15 layers
  * ~1.5 min ~ 25 min + save. No
  Traceback. OUT not on disk. Serve
  still qwen3.8-27b-W4A8-rtn-gdn.
  18080 down.

VERDICT -> RUNNING. May finish before
  next 30m fire. Then wrap_vlm_config
  (in_proj_qkvz) + census. Do not steal.

---

### 2026-08-21k - LOOP 11: 151 GPTQ fire 2 GO + census

CONTEXT -> GPTQ gpu-run exited 0 in 7133s.
  NEXT PICK was wrap + census after Stage B.

CONFIG -> DATAFREE=0 gptq selective 128
  OUT=models/files/qwen3.8-27b/w4a8-gptq-gdn
  wrap_vlm_config.py (qkvz + ba ignore)

COMMAND -> Stage B already in 151 log.
  docker chown + rm RAW. wrap_vlm_config.py.
  CPU census.

RESULT -> calib+quant 6757s. pack 256 int4
  + 144 I8 GDN. graft vis 333 mtp 15.
  20.616 GiB. Census matches RTN byte
  table (MLP I32 7.969, GDN I8 5.156,
  hot 18.967). VLM wrapper + in_proj_qkvz
  on INT8 group. is_prepacked_w4a8 True.
  Log: results/logs/151_qwen38_w4a8_20260821_010557.log

VERDICT -> GO. GPTQ artifact load-ready.

---

### 2026-08-21l - LOOP 12: GPTQ GRAPH=0 smoke GO

CONTEXT -> Census green. Card 0 free. RTN
  GRAPH=0 stays on card 1 :18081.

CONFIG -> GRAPH=0 NOMM=1 B70_NOMTP=1
  PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  HYBRID=0 P2PACCESS=0 IMG=int8g-v0260

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l11-gptq-smoke \
    ./bin/gpu-run --card 0 \
    env GRAPH=0 NOMM=1 B70_NOMTP=1 PORT=18082 \
      NAME=qwen38_w4a8_gptq DEVICE=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  ```

RESULT -> HEALTHY 61s. Paris exact.
  17*23=391. Fib iterative a,b=0,1.
  XPUW4A8IntLinearKernel +
  XPUInt8ScaledMMLinearKernel.
  Log: results/logs/l11_w4a8_gptq_graph0_20260821T030559Z.log
  Serve left Up :18082. RTN still :18081.

VERDICT -> GO. Both artifacts load-gated.
  Next: GRAPH=1 on GPTQ (stop :18082 first).
  Do not start DD. Do not bake.

---

### 2026-08-21m - LOOP 13: GPTQ GRAPH=1 smoke GO ~24.5 tok/s

CONTEXT -> NEXT PICK GRAPH=1 on GPTQ.
  D09 is W4A16-autoround hang, not oneDNN.
  GRAPH=0 already GO. Stop :18082 then
  restart GRAPH=1. Leave RTN :18081.

CONFIG -> GRAPH=1 PIECEWISE capsizes 1,2,4
  NOMM=1 B70_NOMTP=1 HYBRID=0 CGRECLAIM=1000
  PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  P2PACCESS=0 IMG=int8g-v0260

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l13-graph1 \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 NOMM=1 B70_NOMTP=1 PORT=18082 \
      NAME=qwen38_w4a8_gptq DEVICE=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  ```

RESULT -> HEALTHY 299s (compile+capture).
  Paris exact. 17*23=391. Fib iterative.
  dec128 wall 24.52 tok/s (128/5.22s) vs
  GRAPH=0 ~6.3 (~3.9x). Not bench_code c1.
  3.6 W4A8 GRAPH was 27.3. Do not claim
  vs k1bar 31.9 (TP=2 W8A8).
  Log: results/logs/l13_w4a8_gptq_graph1_20260821T030938Z.log
  Serve left Up :18082 GRAPH=1.

VERDICT -> GO. GRAPH=1 is the decode lever
  on 3.8 W4A8 oneDNN. Next: HYBRID=1 A/B
  or bench_code c1. Do not start DD.

---

### 2026-08-21n - LOOP 14: bench_code c1 25.0 GRAPH=1 GPTQ

CONTEXT -> NEXT PICK attach live GRAPH=1
  for bench_code c1 (or HYBRID=1 restart).
  Attach, do not start a third serve.

CONFIG -> :18082 qwen3.8-27b-W4A8-gptq-gdn
  GRAPH=1 TP=1 HYBRID=0 NOMM=1 NOMTP=1
  bench_code out=256 reps=3 conc=1

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> c1 avg=25.0 best=25.0 t/s
  agg=25.0 out~256 wall~10.3s. Matches
  LOOP 13 24.5 wall-128. 3.6 hybrid 27.3
  is still ahead. k1bar 31.9 is TP=2
  W8A8 -- do not demote.

VERDICT -> GO. First 3.8 W4A8 bench_code
  c1 is 25.0. Next: HYBRID=1 GRAPH=1 A/B.
  Leave both serves Up.

---

### 2026-08-21o - LOOP 15: HYBRID=1 e2e 25.0 NO-GO as win

CONTEXT -> NEXT PICK Path H A/B vs 25.0.
  Isolated bar 1.10x. K1 M=1 w4a16 ~tied
  w4a8_op. Stop :18082, GRAPH=1 HYBRID=1.

CONFIG -> GRAPH=1 B70_W4A8_HYBRID=1
  NOMM=1 B70_NOMTP=1 PORT=18082 DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  P2PACCESS=0. RTN :18081 left Up.

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l15-hybrid \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_W4A8_HYBRID=1 NOMM=1 B70_NOMTP=1 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> HEALTHY 56s (compile cache).
  Paris exact. 391 exact. w4a16 fake
  registered (hybrid xpu.py path live).
  bench_code c1 avg=25.0 best=25.0
  wall~10.3s. 25.0/25.0 = 1.00x < 1.10x.
  Log: results/logs/l15_w4a8_gptq_hybrid1_20260821T040547Z.log

VERDICT -> NO-GO as e2e speed win.
  GRAPH=1 already ate the act-quant tax
  so Path H does not move c1. Score stays
  HYBRID=0 25.0. Split-M: decode may use
  either op at M=1. Next: MTP3. Do not
  demote 25.0 / 31.9.

---

### 2026-08-21p - LOOP 16: GRAPH=1 MTP3 !!!! false 61.7 D14

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  GRAPH=1 + MTP3 on GPTQ vs 25.0. HYBRID=1
  :18082 was Up (equal, not the score).
  Card 0 lease w4a8-l16-mtp3 from prior
  fire; this fire ATTACH. P2PACCESS=0.
  DD PARKED. Do not bake.

CONFIG -> GRAPH=1 PIECEWISE capsizes 1,2,4
  B70_NOMTP=0 MTPTOK=3 B70_W4A8_HYBRID=0
  NOMM=1 CGRECLAIM=1000 PORT=18082
  NAME=qwen38_w4a8_gptq DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  speculative-config method=mtp n=3
  P2PACCESS=0 IMG=int8g-v0260
  Restore: same minus spec (B70_NOMTP=1).

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l16-mtp3 \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_NOMTP=0 MTPTOK=3 B70_W4A8_HYBRID=0 NOMM=1 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 CARD=0 P2PACCESS=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  # then restore
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l16-restore \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_NOMTP=1 B70_W4A8_HYBRID=0 NOMM=1 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  ```

RESULT -> MTP3 HEALTHY 340s (compile cache
  80c41e5072; torch.compile 241s + 25s
  drafter). Shim: Qwen3_5MultiTokenPredictor
  forced unquantized; MTP shares embed/lm_head.
  Built-in gen probe Paris OK at start.
  Completions Paris exact, 17*23=391 exact.
  Then fib continuation and chat/code
  collapsed to "!!!!". bench_code c1
  avg=61.7 best=61.8 t/s wall~4.1s
  (61.7/25.0 = 2.47x). SpecDecoding:
  mean accept length 4.00, 100% draft
  accept, per-position 1.000,1.000,1.000.
  LRU chat dump: n=256 top='!' frac=1.000
  GARBAGE. After collapse even Paris
  completions were bangs. Restore NOMTP
  HEALTHY 56s, gen probe Paris OK.
  Re-probes: Paris exact, 391 exact,
  fib iterative a,b=0,1, LRU thinking
  coherent (top='e' frac=0.123).
  Logs: results/logs/l16_w4a8_gptq_mtp3_20260821T043551Z.log
        results/logs/l16b_w4a8_gptq_nomtp_restore_20260821T045155Z.log
  30m loop already ARMED 01a021be5649
  next ~2026-08-21T05:05:17Z.

VERDICT -> NO-GO as speed win. Packet D14.
  61.7 withdrawn. Score stays GRAPH=1
  HYBRID=0 NOMTP 25.0. Next: K16 c=2 on
  the live restore, or isolated K4 M=4,8.
  Do not retry MTP3 e2e without a
  hypothesis. Do not start DD. Do not bake.
  Do not demote 25.0 / 31.9.

---

### 2026-08-21q - LOOP 17: K16 c=2 agg 47.7 G1 OK

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  K16 c=2 on live GRAPH=1 NOMTP :18082.
  Attach, no third serve. D14 closed for
  MTP3 e2e. P2PACCESS=0. DD PARKED.

CONFIG -> :18082 qwen3.8-27b-W4A8-gptq-gdn
  GRAPH=1 TP=1 HYBRID=0 NOMTP=1
  speculative_config=None max_num_seqs=2
  capture sizes 1,2,4. bench_code c=2
  out=256 reps=3. Dual LRU chat 128 G1.

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 2 256 3
  ```

RESULT -> Paris completions OK (not bangs).
  c2 avg=23.9 best=24.2 t/s | agg=47.7
  | out~256 tok | wall~10.9s.
  Per-stream 23.9/25.0 = 0.956x.
  Agg 47.7/25.0 = 1.91x.
  Dual LRU chat G1 OK (top e ~0.10-0.12).
  18081 still RTN. Leases free.

VERDICT -> GO for K16 c=2. Score stays
  c1 25.0. Do not treat 47.7 agg as a
  c1 north-star (not vs 31.9 / 65.08).
  Next: MAXSEQS=8 restart then c=4, or
  card1 isolated M=4,8. Do not retry
  MTP3. Do not start DD. Do not bake.

---

### 2026-08-21r - LOOP 18: K16 c=4 agg 91.4 G1 4/4

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  K16 c=4. Live :18082 was MAXSEQS=2.
  Stop and restart MAXSEQS=8, re-gate c1
  vs 25.0, then c=4. D14 closed. DD PARKED.
  P2PACCESS=0. RTN :18081 left Up.

CONFIG -> GRAPH=1 B70_NOMTP=1 MAXSEQS=8
  HYBRID=0 NOMM=1 CAPSIZES 1,2,4
  PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  P2PACCESS=0 IMG=int8g-v0260
  bench_code c1 then c4 out=256 reps=3
  plus 4x LRU chat 128 G1.

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l18-maxseqs8 \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_NOMTP=1 B70_W4A8_HYBRID=0 NOMM=1 MAXSEQS=8 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 CARD=0 P2PACCESS=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 4 256 3
  ```

RESULT -> HEALTHY 56s. vllm --max-num-seqs 8.
  Built-in gen probe Paris OK. Completions
  Paris OK. c1 avg=24.9 best=25.0 wall~10.3s
  (holds vs 25.0). c4 avg=22.9 best=23.2
  agg=91.4 wall~11.5s.
  Per-stream 22.9/25.0 = 0.916x.
  Agg 91.4/25.0 = 3.66x.
  G1 4/4 OK (top e ~0.12-0.14).
  Log: results/logs/l18_w4a8_gptq_maxseqs8_20260821T053608Z.log
  18081 still RTN. Leases free.

VERDICT -> GO for K16 c=4. Score stays
  c1 25.0. Do not treat 91.4 agg as a
  c1 north-star. Next: c=8 needs
  CAPSIZES+=8 (capture max is 4 today)
  or isolated M=4,8 (stop RTN on card1).
  Do not retry MTP3. Do not start DD.
  Do not bake. Do not demote 25.0 / 31.9.

---

### 2026-08-21s - LOOP 19: K16 c=8 agg 145.8 G1 8/8

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  K16 c=8. Live :18082 capture max was 4.
  Stop and restart CAPSIZES=1,2,4,8,
  re-gate c1 vs 25.0, then c=8. D14 closed.
  DD PARKED. P2PACCESS=0. RTN :18081 left Up.

CONFIG -> GRAPH=1 B70_NOMTP=1 MAXSEQS=8
  CAPSIZES=1,2,4,8 HYBRID=0 NOMM=1
  PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  P2PACCESS=0 IMG=int8g-v0260
  bench_code c1 then c8 out=256 reps=3
  plus 8x LRU chat 128 G1.

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l19-cap8 \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_NOMTP=1 B70_W4A8_HYBRID=0 NOMM=1 \
      MAXSEQS=8 CAPSIZES=1,2,4,8 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 CARD=0 P2PACCESS=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 8 256 3
  ```

RESULT -> HEALTHY 279s. New compile cache
  f6dd77a115 (capture 8). vllm
  cudagraph_capture_sizes [1,2,4,8].
  Built-in gen probe Paris OK. Completions
  Paris OK. c1 avg=25.0 best=25.0 wall~10.2s
  (holds). c8 avg=18.2 best=18.5 agg=145.8
  wall~16.0s.
  Per-stream 18.2/25.0 = 0.728x.
  Agg 145.8/25.0 = 5.83x.
  G1 8/8 OK (top e ~0.11-0.14).
  Log: results/logs/l19_w4a8_gptq_capsizes8_20260821T060557Z.log
  18081 still RTN. Leases free.

VERDICT -> GO for K16 c=8. Concurrent row
  complete (c2/c4/c8). Score stays c1 25.0.
  Do not treat 145.8 agg as a c1 north-star.
  Next: K4 isolated M=4,8 on card1 (stop
  RTN). Leave GPTQ score serve. Do not
  retry MTP3. Do not start DD. Do not bake.
  Do not demote 25.0 / 31.9.

---

### 2026-08-21t - LOOP 20: K4 M=4,8 still BW; D15 pad-M

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  K4 isolated GEMM M=4,8 on card1. Stop RTN
  :18081. Leave GPTQ :18082. D14 closed.
  DD PARKED. P2PACCESS=0. ONEDNN_VERBOSE
  M=1 vs M=8 to see DPAS.

CONFIG -> IMG=int8g-v0260 --entrypoint bash
  ZE_AFFINITY_MASK=1
  B70_XPU_C_SO=w8a8_kernel_v0240_fusedq
  CKPT=/models/qwen3.6-27b/w4a8-sqgptq
  ONLY_MS=4,8. ONEDNN_VERBOSE=1 down_proj.

COMMAND ->
  ```
  NAME=qwen38_w4a8_rtn bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l20-k4 \
    ./bin/gpu-run --card 1 \
    env ONLY_MS=4,8 bash -c '...bench_w4a8_shapes.py;
      ONEDNN_VERBOSE=1 python3 vllm/w4a8/onednn_verbose_m1_m8.py'
  ```

RESULT -> 13s. Fat GEMMs M=8 still ~94-96% of
  581 GB/s, ~17 TOPS. Path H ~= Path X
  (down_proj w4a16 0.080 / w4a8_op 0.0815 =
  1.00x). o_proj/gdn_out w4a8_op ~60% roof
  (quant tax on small N). gdn_ba <3% roof.
  oneDNN: both M=1 and M=8
  `gpu,matmul,jit:gemm:any` src:s8 wei:u4
  dst:f16. No `dpas` token. Same impl.
  Log: results/logs/k4_w4a8_m48_20260821T063757Z.log
  GPTQ :18082 still Up. RTN stopped.

VERDICT -> GO as measurement. Isolated 1.10x
  Path X vs H at M=4,8: NO-GO. Packet D15
  pad-M dummy 8. K4 e2e DSpark accept still
  open (do not night-train this pick).
  Next: K5 VNNI16. Do not retry MTP3.
  Do not start DD. Do not bake. Do not
  demote 25.0 / 31.9.

---

### 2026-08-21u - LOOP 21: K5 stock sycl-tla D16

CONTEXT -> 30m fire 01a021be5649. NEXT PICK
  K5 VNNI16 isolated vs K1, bar 1.10x or
  packet. Card1 free. GPTQ :18082 Up.
  D14/D15 closed. DD PARKED. P2PACCESS=0.
  Stock sycl-tla mixed-dtype examples are
  the scaffold baseline, not the paper kernel.

CONFIG -> IMG=vllm-xpu-env:v0240
  ZE_AFFINITY_MASK=1
  ONEAPI_DEVICE_SELECTOR=level_zero:0
  binaries build_bmg examples 02 mixed-dtype
  EXAMPLES=bf16_s8,f16_s8 ITERS=50
  roofline in harness 608 GB/s.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l21-k5 \
    ./bin/gpu-run --card 1 \
    docker run --entrypoint bash vllm-xpu-env:v0240 \
      python3 /bench/bench.py --examples bf16_s8,f16_s8 --iters 50
  ```

RESULT -> 96s, all Disposition Passed.
  down M=1 bf16_s8 13471 us vs K1 w4a16
  79 us (171x slower). gate_up M=1 22840 us
  vs 161 us (142x). M=1..16 wall flat
  (down ~13.4 ms, gate_up ~22.8 ms).
  ~1.1-1.4% of 608. f16_s8 == bf16_s8.
  Log: results/logs/k5_sycltla_m148_20260821T070719Z.log
  GPTQ :18082 still Up.

VERDICT -> NO-GO as 1.10x. Packet D16.
  Paper rectangular TiledMMA is the retry,
  not a re-run of example 02. Next: K10
  prefill on live :18082. Do not retry
  MTP3/pad-M. Do not start DD. Do not bake.
  Do not demote 25.0 / 31.9.

---

### 2026-08-21v - LOOP 22: K10 prefill ~2870 / ~2750 tok/s

CONTEXT -> Operator: dual-card + 15m loop.
  NEXT PICK K10 on live :18082. Card1 LOOP 23
  in parallel. Files already on disk. D14-16
  closed. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 qwen3.8-27b-W4A8-gptq-gdn
  GRAPH=1 NOMTP HYBRID=0 MAXLEN=8192
  bench_prefill_ttft lens 2048,8000 reps=3
  max_tokens=1.

COMMAND ->
  ```
  python3 -u vllm/w4a8/bench_prefill_ttft.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 2048,8000 3
  ```

RESULT -> 2045 tok TTFT_avg=0.713s
  prefill_avg=2868 tok/s. 7995 tok
  TTFT_avg=2.908s prefill_avg=2750 tok/s.
  First token " The" (not bangs).
  Log: results/logs/k10_w4a8_prefill_20260821T071519Z.log
  15m loop armed 01a021be5649.

VERDICT -> GO. First 3.8 W4A8 prefill
  number. Do not vs 27.3 (decode). Score
  stays c1 25.0. Next: K8 card1.

---

### 2026-08-21w - LOOP 23: M=2048 Path X 1.4-1.7x H

CONTEXT -> Dual with LOOP 22. Card1 free.
  Does large-M switch oneDNN impl? Isolated
  1.10x Path X vs H at prefill tiles?

CONFIG -> IMG=int8g-v0260 ZE_AFFINITY_MASK=1
  ONLY_MS=256,2048 ONLY_SHAPES=gate_up,down_proj
  ONEDNN_VERBOSE=1 ONLY_MS=1,256,2048 down_proj.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l23-largeM \
    ./bin/gpu-run --card 1 env ONLY_MS=256,2048 \
    ONLY_SHAPES=gate_up,down_proj bench_w4a8_shapes.py
  ```

RESULT -> gate_up M=2048 w4a8_op 3.71 ms
  197 TOPS 1.39x w4a16; w8a8_full 261 TOPS.
  down M=2048 w4a8_op 1.65 ms 221 TOPS
  1.70x w4a16; w8a8_full 227 TOPS.
  oneDNN jit:gemm:any s8xu4 at M=1,256,2048.
  CSV: results/logs/k10_w4a8_m256_2048_20260821T071519Z.user.csv
  Log: results/logs/k10_w4a8_onednn_m2048_20260821T071519Z.log

VERDICT -> GO. Split-M is real at M=2048.
  Decode stays H; prefill/c>1 stays X.
  Do not demote 25.0 / 31.9. Do not bake.

---

### 2026-08-21x - LOOP 24: attach c1 hold 25.0

CONTEXT -> 15m dual fire 01a021be5649. Serve attach
  while card1 runs K8. Live :18082 ~1h.
  D14-16 closed. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 qwen3.8-27b-W4A8-gptq-gdn
  GRAPH=1 NOMTP HYBRID=0. Paris + bench_code
  c1 1 256 3 vs 25.0.

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris OK. c1 avg=best 25.0
  wall~10.2s. Log: results/logs/l24_w4a8_c1_hold_20260821T072116Z.log

VERDICT -> GO. Score holds. Leave serve Up.

---

### 2026-08-21y - LOOP 25: K8 lm_head g32 isolated 1.27 ms

CONTEXT -> NEXT PICK K8 isolated. Dual with
  LOOP 24. Do not 151 whole-model. GROUP=32
  as catalog. Synthetic pack (ckpt lm_head BF16).

CONFIG -> IMG=int8g-v0260 ZE_AFFINITY_MASK=1
  INCLUDE_LMHEAD=1 ONLY_SHAPES=lm_head
  ONLY_MS=1,8 GROUP=32. N=248320 K=5120.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l25-k8 \
    ./bin/gpu-run --card 1 env INCLUDE_LMHEAD=1 \
    ONLY_SHAPES=lm_head ONLY_MS=1,8 GROUP=32 \
    bench_w4a8_shapes.py
  ```

RESULT -> 9s. M=1 w4a16 1.274 ms 499 GB/s
  85.9%roof 3.35x bf16. w4a8_op 1.293 ms
  (~1.01x H). M=8 still BW ~84% / 15.6 TOPS.
  Log: results/logs/k8_lmhead_g32_20260821T072116Z.log
  CSV: results/logs/k8_lmhead_g32_20260821T072116Z.user.csv

VERDICT -> GO as kernel. Isolated 1.10x X vs
  H NO-GO. e2e lm_head INT4 still open (file
  is BF16). Next: K12 N-trap card1. Do not
  rewrite 151. Do not demote 25.0 / 31.9.

---

### 2026-08-21z - LOOP 26: attach c2 agg 48.5

CONTEXT -> 15m dual fire. Serve attach while
  card1 K12. c1 held last fire. D14-16 closed.
  DD PARKED. P2PACCESS=0.

CONFIG -> :18082 GRAPH=1 NOMTP HYBRID=0
  bench_code c=2 out=256 reps=3 vs 47.7.

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 2 256 3
  ```

RESULT -> Paris OK. c2 avg=best 24.3 t/s
  agg=48.5 wall~10.6s. Log:
  results/logs/l26_w4a8_c2_hold_20260821T073611Z.log

VERDICT -> GO. c2 holds. Score stays c1 25.0.

---

### 2026-08-21za - LOOP 27: K12 N-pad 96->128 D17

CONTEXT -> NEXT PICK K12. Dual with LOOP 26.
  Isolated bar 1.10x. Not a serve change.

CONFIG -> IMG=int8g-v0260 ZE_AFFINITY_MASK=1
  EXTRA_SHAPES=gdn_ba_pad:5120:128
  ONLY_SHAPES=gdn_ba,gdn_ba_pad ONLY_MS=1,8,256

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l27-k12 \
    ./bin/gpu-run --card 1 env EXTRA_SHAPES=gdn_ba_pad:5120:128 \
    ONLY_SHAPES=gdn_ba,gdn_ba_pad ONLY_MS=1,8,256 \
    bench_w4a8_shapes.py
  ```

RESULT -> N=96 M=1 w4a16 0.0380 ms 1.1%roof;
  N=128 0.0389 ms 1.4%roof. Wall tied. 6s.
  Log: results/logs/k12_gdnba_pad128_20260821T073611Z.log
  CSV: results/logs/k12_gdnba_pad128_20260821T073611Z.user.csv

VERDICT -> NO-GO as 1.10x. Packet D17. Keep
  ba BF16. Next: K13 GROUP=32 vs 128. Do not
  demote 25.0 / 31.9. Do not bake.

---

### 2026-08-21zb - LOOP 28: attach c8 agg 147.8

CONTEXT -> 15m dual fire. Serve attach while
  card1 K13. c2 held last fire. 2h soak.
  D14-17 closed. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 GRAPH=1 NOMTP HYBRID=0
  MAXSEQS=8 CAPSIZES=1,2,4,8
  bench_code c=8 out=256 reps=3 vs 145.8.

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 8 256 3
  ```

RESULT -> Paris OK. c8 avg=best 18.5 t/s
  agg=147.8 wall~15.4s. Log:
  results/logs/l28_w4a8_c8_hold_20260821T075051Z.log

VERDICT -> GO. c8 holds. Score stays c1 25.0.

---

### 2026-08-21zc - LOOP 29: K13 g32/g64 slower D18

CONTEXT -> NEXT PICK K13 isolated. Dual with
  LOOP 28. 128 default unless >=1.10x.
  Do not 151 requant.

CONFIG -> IMG=int8g-v0260 ZE_AFFINITY_MASK=1
  ONLY_SHAPES=down_proj ONLY_MS=1,8,2048
  GROUP=128 then 64 then 32.

COMMAND ->
  ```
  for G in 128 64 32; do
    GROUP=$G ONLY_SHAPES=down_proj ONLY_MS=1,8,2048 \
      bench_w4a8_shapes.py
  done
  ```

RESULT -> M=1 w4a16 g128 0.0789 ms 97.2%;
  g64 0.0828 (0.95x); g32 0.0957 (0.82x).
  M=2048 w4a8_op 220 / 176 / 121 TOPS.
  g64/g32 synthesized (ckpt g128). 20s.
  Log: results/logs/k13_group_size_20260821T075051Z.log

VERDICT -> NO-GO as 1.10x. Packet D18.
  GROUP=128 stays. Next: K15 TP=2 PUSH_AR
  P2PACCESS=0 (stops :18082). Do not 151
  g32. Do not demote 25.0 / 31.9. Do not bake.

---

### 2026-08-21zd - LOOP 30: K15 TP=2 GRAPH=0 PUSH_AR c1 3.7

CONTEXT -> 15m fire. NEXT PICK K15 TP=2.
  Stop TP=1 GRAPH=1 :18082. Both cards.
  GRAPH=0 first (avoid capture wedge).
  PUSH_AR=1 PUSH_AR_GRAPH=0 (eager).
  P2PACCESS=0. B70_NOMTP=1. xpu-health GO.
  D14-18 closed. DD PARKED.

CONFIG -> TP=2 GRAPH=0 PUSH_AR=1
  PUSH_AR_GRAPH=0 MIN_NUMEL=65536
  P2PACCESS=0 NOMTP=1 HYBRID=0 NOMM=1
  MAXSEQS=8 PORT=18082
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  IMG=int8g-v0260

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l30-tp2 ./bin/gpu-run \
    env GRAPH=0 TP=2 PUSH_AR=1 P2PACCESS=0 B70_NOMTP=1 \
      MAXSEQS=8 PORT=18082 NAME=qwen38_w4a8_gptq \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> HEALTHY 81s. PUSH_AR patched
  XpuCommunicator.all_reduce. world_size=2.
  Paris OK. 391 OK. c1 avg=best 3.7 t/s
  wall~68.7s. LRU G1 OK. vs TP=1 GRAPH=0
  ~6.3 and score 25.0. P2PACCESS=0.
  Logs: l30_w4a8_tp2_graph0_20260821T080715Z.log
        l30_w4a8_tp2_c1_20260821T080911Z.log

VERDICT -> GO as TP=2 load-gate. Not a
  speed win vs 25.0. Next: GRAPH=1 TP=2
  PUSH_AR_GRAPH=1 IGP=false. Do not MTP.
  Do not P2PACCESS=1. Do not demote 25.0.

---

### 2026-08-21ze - LOOP 31: K15 GRAPH=1 TP=2 D19 segfault

CONTEXT -> 15m fire. NEXT PICK K15 GRAPH=1
  TP=2 after GRAPH=0 load-gate 3.7. Stop
  GRAPH=0 first. PUSH_AR_GRAPH=1 graph.so
  MIN_NUMEL=0 IGP=false P2PACCESS=0.
  B70_NOMTP=1. Do not MTP (D14). xpu-health
  GO. D14-D18 closed. DD PARKED.

CONFIG -> TP=2 GRAPH=1 PUSH_AR=1
  PUSH_AR_GRAPH=1 MIN_NUMEL=0
  P2PACCESS=0 NOMTP=1 HYBRID=0 NOMM=1
  MAXSEQS=8 CAPSIZES=1,2,4,8 PORT=18082
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  IMG=int8g-v0260 IGP=false

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l31-tp2g1 ./bin/gpu-run \
    env GRAPH=1 TP=2 IGP=false B70_NOMTP=1 B70_W4A8_HYBRID=0 \
      NOMM=1 MAXSEQS=8 CAPSIZES=1,2,4,8 PUSH_AR=1 \
      PUSH_AR_GRAPH=1 P2PACCESS=0 PORT=18082 \
      NAME=qwen38_w4a8_gptq \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> HEALTHY 381s. PUSH_AR patched.
  Graph capture 4 sizes in 3s. Paris OK.
  391 OK. c1 avg=23.5 best=24.7 wall~10.4s
  (warmup HTTP 500). LRU chatcmpl:
  Worker-0 segfault in
  XPUGraphImpl::instantiate /
  urCommandBufferReleaseExp /
  exec_graph_impl dtor. EngineDead.
  Connection reset. container exit 0.
  xpu-health HEALTHY after (not
  DEVICE_LOST). 23.5 = 0.94x vs 25.0.
  Logs: l31_w4a8_tp2_graph1_20260821T082108Z.log
        l31_w4a8_tp2_graph1_c1_20260821T082752Z.log
        l31_w4a8_tp2_graph1_engine_20260821T082108Z.log

VERDICT -> NO-GO. Packet D19. Score stays
  GRAPH=1 TP=1 25.0. Do not chain another
  GRAPH=1 TP=2 start. Dual restore next:
  card 0 TP=1 GRAPH=1, card 1 D04 probe.

---

### 2026-08-21zf - LOOP 32: GRAPH=1 TP=1 restore STARTED

CONTEXT -> 15m dual fire after D19. Restore
  the 25.0 score path on one card. W4A8
  files already on disk (no 151). DD PARKED.
  P2PACCESS=0.

CONFIG -> GRAPH=1 TP=1 DEVICE=0 CARD=0
  NOMTP=1 HYBRID=0 NOMM=1 MAXSEQS=8
  CAPSIZES=1,2,4,8 PORT=18082
  NAME=qwen38_w4a8_gptq
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  IMG=int8g-v0260

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l32-tp1g1 \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 TP=1 NOMM=1 B70_NOMTP=1 \
      B70_W4A8_HYBRID=0 MAXSEQS=8 \
      CAPSIZES=1,2,4,8 PORT=18082 \
      NAME=qwen38_w4a8_gptq DEVICE=0 CARD=0 \
      CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-gdn \
      P2PACCESS=0 \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  ```

RESULT -> HEALTHY 56s (compile cache hit).
  Built-in Paris OK. Manual Paris exact.
  391 exact. bench_code c1 avg=best 25.0
  t/s wall~10.3s. Served id
  qwen3.8-27b-W4A8-gptq-gdn. KV auto.
  Logs: l32_w4a8_tp1_graph1_restore_20260821T083346Z.log
        l32_w4a8_tp1_c1_20260821T083621Z.log

VERDICT -> GO. Score restored. Leave
  :18082 Up. Next: K17 DSpark 10-sample.
  Do not demote 25.0.

---

### 2026-08-21zg - LOOP 33: D04 still gated oneAPI<2026

CONTEXT -> dual with LOOP 32. Campaign
  listed one woqgemm compute_type=int8
  re-probe on runtime 26.22. Card 1.

CONFIG -> IMG=sglang-xpu:woq-0515
  ZE_AFFINITY_MASK=1. First synthetic
  int4 5120x5120. Then real
  qwen3.6-27b/int4-autoround layer 20
  down_proj via QuantLinearGPTQ.

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l33-d04 \
    ./bin/gpu-run --card 1 docker run --rm \
    --name k_d04_joint sglang-xpu:woq-0515 \
    python ark.woqgemm sweep
  ```

RESULT -> synthetic: all ct FAIL
  ValueError blob dtype int8 vs int32.
  real: "XMX int8 is not supported on
  B70 with oneAPI < 2026. Falling back
  to fp16." ARK libsycl.so.9 missing.
  post_init NotImplementedError device
  xpu:0. 39s exit 1.
  Logs: k_d04_joint_matrix_20260821T083346Z.log
        k_d04_joint_matrix_real_20260821T083621Z.log

VERDICT -> NO-GO. Leave D04 closed.
  oneDNN Path H/X stay the kernels.
  Do not re-probe on 2025.3.

---

### 2026-08-21zh - LOOP 34: K17 off-shelf DSpark pos0 66%

CONTEXT -> 15m dual fire after D04. NEXT PICK
  was K17 10-sample overfit. SpecForge is
  not on disk. Dump off-shelf accept first
  (W8A8 sequence). Score serve :18082 stays.
  Card 1 new W4A8+DSpark GRAPH=0. D13 KV
  auto. D14 no MTP. D19 no TP=2. DD PARKED.

CONFIG -> GRAPH=0 TP=1 SPECTOK=7 method=dspark
  DEVICE=1 PORT=18083 MAXSEQS=1 MAXLEN=8192
  UTIL=0.88 NOMTP=0 HYBRID=0 P2PACCESS=0
  CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn
  DRAFTER=dflash-drafter-fp8-b70
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  IMG=int8g-v0260 readout patches v0260
  KV auto/bf16

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l34-dspark \
    ./bin/gpu-run --card 1 \
    env GRAPH=0 TP=1 SPECTOK=7 PORT=18083 \
      NAME=qwen38_w4a8_dspark DEVICE=1 CARD=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  ```

RESULT -> HEALTHY 97s. Paris exact. 391 exact.
  c1 avg=14.6 best=15.8 t/s wall~16.2s.
  vs GRAPH=0 NOMTP ~6.3 = 2.3x. vs score
  25.0 = 0.58x. metrics drafts=398
  accepted=682 pos0=264/398=66.3%
  mean_len=2.71 tok_rate=24.5%.
  :18082 Paris still holds.
  Logs: l34_w4a8_dspark_graph0_20260821T084152Z.log
        l34_w4a8_dspark_c1_20260821T084341Z.log
        l34_w4a8_tp1_attach_20260821T084152Z.log

VERDICT -> GO as accept gate. Not a score
  win. Do not night-train. Next: GRAPH=1
  DSpark vs 25.0. Garbage-test. Do not
  demote 25.0.

---

### 2026-08-21zi - LOOP 35: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK GRAPH=1
  DSpark on card 1. Attach :18082. D14-D19
  closed. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 out=256 reps=3 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.2s. Log:
  l35_w4a8_tp1_c1_hold_20260821T085055Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21zj - LOOP 36: GRAPH=1 DSpark 8192 KV miss

CONTEXT -> NEXT PICK K17 GRAPH=1 DSpark.
  Stop GRAPH=0 :18083. Leave :18082.
  CAPSIZES=1 MAXSEQS=1. KV auto. D13/D14/D19.

CONFIG -> GRAPH=1 TP=1 SPECTOK=7
  MAXLEN=8192 UTIL=0.88 DEVICE=1 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  CGRECLAIM=1000 P2PACCESS=0

COMMAND ->
  ```
  NAME=qwen38_w4a8_dspark bash \
    vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l36-dspark-g1 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=1 CAPSIZES=1 \
      PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  ```

RESULT -> EXITED EARLY ~321s. EngineCore
  ValueError: 8192 needs 2.06 GiB KV, have
  1.59 (est max 3328). Not DEVICE_LOST.
  xpu-health HEALTHY. Logs:
  l36_w4a8_dspark_graph1_20260821T085055Z.log
  l36_w4a8_dspark_graph1_engine_20260821T085055Z.log

VERDICT -> NO-GO at 8192/0.88. Retry
  MAXLEN=4096 UTIL=0.90.

---

### 2026-08-21zk - LOOP 37: GRAPH=1 DSpark c1 34.7

CONTEXT -> LOOP 36 KV miss. Dual with LOOP
  35. Garbage-test vs D14. Do not night-train.

CONFIG -> GRAPH=1 TP=1 SPECTOK=7
  MAXLEN=4096 UTIL=0.90 MAXSEQS=1 CAPSIZES=1
  DEVICE=1 PORT=18083 KV auto P2PACCESS=0
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l37-dspark-g1 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXLEN=4096 UTIL=0.90 \
      MAXSEQS=1 CAPSIZES=1 PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  ```

RESULT -> HEALTHY 299s. Paris exact. 391
  exact. LRU n=128 G1_OK bang_frac=0.
  c1 avg=34.7 best=35.1 wall~7.3s.
  1.39x vs NOMTP 25.0. 2.38x vs GRAPH=0
  DSpark 14.6. drafts=431 accepted=775
  pos0=298/431=69.1% mean_len=2.80
  tok_rate=25.7%. Not D14 100% accept.
  :18082 still Up. Logs:
  l37_w4a8_dspark_graph1_ml4096_20260821T085659Z.log
  l37_w4a8_dspark_graph1_c1_20260821T090214Z.log

VERDICT -> GO. Spec e2e row 34.7.
  NOMTP honesty stays 25.0. Leave both
  Up. Next: k=4 A/B. Do not demote 25.0.

---

### 2026-08-21zl - LOOP 38: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK k=4
  A/B on card 1. Attach :18082. DD PARKED.
  P2PACCESS=0. D14-D19 closed.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 out=256 reps=3 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. Log:
  l38_w4a8_tp1_c1_hold_20260821T090546Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21zm - LOOP 39: k=4 c1 33.5 NO-GO vs k=7

CONTEXT -> NEXT PICK k-sweep. Stop k=7
  :18083. GRAPH=1 MAXLEN=4096 UTIL=0.90.
  Isolated 1.10x vs 34.7. Garbage-test.
  Leave :18082. DD PARKED.

CONFIG -> GRAPH=1 TP=1 SPECTOK=4
  MAXLEN=4096 UTIL=0.90 MAXSEQS=1 CAPSIZES=1
  DEVICE=1 PORT=18083 KV auto P2PACCESS=0
  SERVED=qwen3.8-27b-W4A8-gptq-dspark4

COMMAND ->
  ```
  NAME=qwen38_w4a8_dspark bash \
    vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l39-dspark-k4 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=4 MAXLEN=4096 UTIL=0.90 \
      MAXSEQS=1 CAPSIZES=1 PORT=18083 DEVICE=1 \
      SERVED=qwen3.8-27b-W4A8-gptq-dspark4 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark4 1 256 3
  ```

RESULT -> HEALTHY 305s. Paris exact. 391
  exact. LRU G1_OK bang_frac=0. c1 avg=33.5
  best=34.5 wall~7.4s = 0.97x vs k=7 34.7.
  drafts=464 accepted=746 pos0=67.5%
  mean_len=2.61 tok_rate=40.2%. Logs:
  l39_w4a8_dspark_k4_graph1_20260821T090546Z.log
  l39_w4a8_dspark_k4_c1_20260821T091113Z.log

VERDICT -> NO-GO as 1.10x. Keep SPECTOK=7.
  Restore k=7. Do not demote 34.7 / 25.0.

---

### 2026-08-21zn - LOOP 40: restore k=7 STARTED

CONTEXT -> k=4 lost. Restore spec champion
  on card 1. :18082 stays.

CONFIG -> GRAPH=1 SPECTOK=7 MAXLEN=4096
  UTIL=0.90 MAXSEQS=1 CAPSIZES=1 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l40-dspark-k7 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXLEN=4096 UTIL=0.90 \
      PORT=18083 DEVICE=1 \
      SERVED=qwen3.8-27b-W4A8-gptq-dspark7 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  ```

RESULT -> HEALTHY 61s (k=7 compile cache).
  Paris exact. c1 avg=35.2 best=38.1
  wall~6.7s. Holds vs 34.7. Logs:
  l40_w4a8_dspark_k7_restore_20260821T091204Z.log
  l40_w4a8_dspark_k7_c1_20260821T091318Z.log

VERDICT -> GO. Spec path restored. Leave
  both Up. Next: MAXSEQS>1 if KV fits.
  Do not demote 34.7 / 25.0.

---

### 2026-08-21zo - LOOP 41: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK MAXSEQS=2
  on card 1. Attach :18082. DD PARKED.
  P2PACCESS=0. D14-D19 closed.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 out=256 reps=3 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. Log:
  l41_w4a8_tp1_c1_hold_20260821T092056Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21zp - LOOP 42: MAXSEQS=2 c1 32.4 / c2 47.2

CONTEXT -> NEXT PICK MAXSEQS>1 if KV fits.
  GRAPH=1 DSpark k=7 MAXLEN=4096 UTIL=0.90.
  Leave :18082. Garbage-test. vs 34.7 and
  NOMTP c2 48.5.

CONFIG -> GRAPH=1 TP=1 SPECTOK=7
  MAXSEQS=2 CAPSIZES=1,2 MAXLEN=4096 UTIL=0.90
  DEVICE=1 PORT=18083 KV auto P2PACCESS=0
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  NAME=qwen38_w4a8_dspark bash \
    vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l42-dspark-ms2 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=2 CAPSIZES=1,2 \
      MAXLEN=4096 UTIL=0.90 PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 2 256 3
  ```

RESULT -> HEALTHY 285s. KV fitted. Paris
  exact. 391 exact. LRU G1_OK. c1 avg=32.4
  best=34.1 = 0.93x vs 34.7. c2 avg=23.6
  agg=47.2 vs NOMTP c2 48.5 = 0.97x.
  pos0=64.2% mean_len=2.65. Logs:
  l42_w4a8_dspark_mseqs2_20260821T092056Z.log
  l42_w4a8_dspark_mseqs2_bench_20260821T092603Z.log

VERDICT -> GO as KV-fit. NO-GO as 1.10x.
  Restore MAXSEQS=1. Do not demote 34.7 / 25.0.

---

### 2026-08-21zq - LOOP 43: restore MAXSEQS=1 STARTED

CONTEXT -> MAXSEQS=2 lost isolated c1.
  Restore spec champion. :18082 stays.

CONFIG -> GRAPH=1 SPECTOK=7 MAXSEQS=1
  CAPSIZES=1 MAXLEN=4096 UTIL=0.90 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l43-dspark-ms1 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=1 CAPSIZES=1 \
      MAXLEN=4096 UTIL=0.90 PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  ```

RESULT -> HEALTHY 61s (cache). Paris exact.
  c1 avg=33.8 best=35.9 wall~8.1s. Holds vs
  34.7. Logs:
  l43_w4a8_dspark_mseqs1_restore_20260821T092800Z.log
  l43_w4a8_dspark_mseqs1_c1_20260821T092920Z.log

VERDICT -> GO. Isolated spec restored.
  Leave both Up. Next: SPECTOK=3 or hold.
  Do not demote 34.7 / 25.0.

---

### 2026-08-21zr - LOOP 44: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK k=3 A/B
  on card 1. Attach :18082. DD PARKED.
  P2PACCESS=0. D14-D19 closed.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 out=256 reps=3 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. Log:
  l44_w4a8_tp1_c1_hold_20260821T093544Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21zs - LOOP 45: k=3 c1 31.0 NO-GO vs k=7

CONTEXT -> NEXT PICK k-sweep SPECTOK=3.
  Stop k=7 :18083. GRAPH=1 MAXSEQS=1
  MAXLEN=4096 UTIL=0.90. vs 34.7.
  Garbage-test. Leave :18082.

CONFIG -> GRAPH=1 TP=1 SPECTOK=3
  MAXLEN=4096 UTIL=0.90 MAXSEQS=1 CAPSIZES=1
  DEVICE=1 PORT=18083 KV auto P2PACCESS=0
  SERVED=qwen3.8-27b-W4A8-gptq-dspark3

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l45-dspark-k3 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=3 MAXLEN=4096 UTIL=0.90 \
      MAXSEQS=1 CAPSIZES=1 PORT=18083 DEVICE=1 \
      SERVED=qwen3.8-27b-W4A8-gptq-dspark3 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark3 1 256 3
  ```

RESULT -> HEALTHY 294s. Paris exact. 391
  exact. LRU G1_OK. c1 avg=31.0 best=33.4
  wall~8.1s = 0.89x vs k=7 34.7.
  drafts=500 accepted=700 pos0=65.6%
  mean_len=2.40 tok_rate=46.7%. Logs:
  l45_w4a8_dspark_k3_graph1_20260821T093544Z.log
  l45_w4a8_dspark_k3_c1_20260821T094059Z.log

VERDICT -> NO-GO as 1.10x. k-sweep closed
  7>4>3. Restore k=7. Do not demote 34.7.

---

### 2026-08-21zt - LOOP 46: restore k=7 STARTED

CONTEXT -> k=3 lost. Restore spec champion.
  :18082 stays.

CONFIG -> GRAPH=1 SPECTOK=7 MAXSEQS=1
  CAPSIZES=1 MAXLEN=4096 UTIL=0.90 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l46-dspark-k7 \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=1 CAPSIZES=1 \
      MAXLEN=4096 UTIL=0.90 PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  ```

RESULT -> HEALTHY 61s (cache). Paris exact.
  c1 avg=34.1 best=36.7 wall~7.0s. Holds vs
  34.7. Logs:
  l46_w4a8_dspark_k7_restore_20260821T094146Z.log
  l46_w4a8_dspark_k7_c1_20260821T094303Z.log

VERDICT -> GO. k-sweep closed 7>4>3.
  Isolated spec restored. Leave both Up.
  Do not demote 34.7 / 25.0.

---

### 2026-08-21zu - LOOP 47: attach NOMTP c1 25.0 / PP 2880

CONTEXT -> 15m dual fire. Both campaign serves
  Up. Attach, do not steal. NEXT PICK was
  hold or sglang. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 + prefill 2048 vs K10 ~2870

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  python3 -u vllm/w4a8/bench_prefill_ttft.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 2048 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. prefill pt=2045 TTFT 0.710s
  2880 tok/s. Log:
  l47_w4a8_nomtp_attach_20260821T095057Z.log

VERDICT -> GO. NOMTP score and prefill hold.

---

### 2026-08-21zv - LOOP 48: attach DSpark c1 33.2 / PP 2615

CONTEXT -> dual with LOOP 47. Hold isolated
  spec. Prefill A/B vs NOMTP 2880. MAXLEN
  4096 so 2k only. D14-D19 closed.

CONFIG -> :18083 GRAPH=1 DSpark k=7
  MAXSEQS=1 MAXLEN=4096
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  bench_code c1 vs 34.7 + prefill 2048

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  python3 -u vllm/w4a8/bench_prefill_ttft.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 2048 3
  ```

RESULT -> Paris exact. 391 exact. c1 avg=33.2
  best=37.3 wall~9.3s (holds vs 34.7).
  prefill pt=2045 TTFT 0.782s 2615 tok/s
  = 0.91x vs NOMTP 2880. mean_len=2.71
  tok_rate=24.4%. Log:
  l48_w4a8_dspark_attach_20260821T095057Z.log

VERDICT -> GO as hold. Spec decode wins
  e2e; prefill pays ~9% drafter tax. Leave
  both Up. Next: sglang Path H. Do not
  demote 34.7 / 25.0.

---

### 2026-08-21zw - LOOP 49: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK sglang
  Path H on card 1. Attach :18082. DD PARKED.
  P2PACCESS=0. D05 no torch.compile AQ.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.2s. Log:
  l49_w4a8_nomtp_attach_20260821T100710Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21zx - LOOP 50: sglang float16 GDN triton crash

CONTEXT -> NEXT PICK sglang 3.8 Path H.
  Stop DSpark. GRAPH=0 first. COMPILE=0
  (D05). B70_XPU_W4A8=1 B70_XPU_W8A8=1.
  IMG=sglang-xpu:mtp. --dtype float16
  (w4a8 op emits fp16).

CONFIG -> GRAPH=0 DEVICE=1 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-sglang
  CKPT=w4a8-gptq-gdn CTX=4096

COMMAND ->
  ```
  NAME=qwen38_w4a8_dspark bash \
    vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l50-sglang \
    ./bin/gpu-run --card 1 \
    env GRAPH=0 PORT=18083 DEVICE=1 \
    bash sglang/serve_qwen38_w4a8.sh start
  ```

RESULT -> shims installed. Scheduler
  CompilationError GDN Triton mismatched
  col0 bf16 vs fp16. container exit.
  Logs: l50_w4a8_sglang_graph0_20260821T100710Z.log
        l50_w4a8_sglang_graph0_engine_20260821T100710Z.log

VERDICT -> NO-GO as float16. Retry bf16.

---

### 2026-08-21zy - LOOP 51: sglang bf16 GARBAGE

CONTEXT -> LOOP 50 hypothesis: 3.6 Path H
  uses bfloat16. COMPILE=0. GRAPH=0.

CONFIG -> DTYPE=bfloat16 GRAPH=0 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-sglang

COMMAND ->
  ```
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l51-sglang \
    ./bin/gpu-run --card 1 \
    env GRAPH=0 DTYPE=bfloat16 PORT=18083 DEVICE=1 \
    bash sglang/serve_qwen38_w4a8.sh start
  ```

RESULT -> HEALTHY ~282s after Triton JIT.
  Paris x2 garbage 'onatitable...猜到猜到'.
  391 garbage. W4A8 layers did wire.
  Log: l51_w4a8_sglang_bf16_20260821T101630Z.log

VERDICT -> NO-GO as coherent sglang 3.8
  Path H. Do not GRAPH=1 this body.
  Restore vLLM DSpark.

---

### 2026-08-21zz - LOOP 52: restore DSpark k=7 STARTED

CONTEXT -> sglang GARBAGE. Restore spec
  champion. :18082 stays.

CONFIG -> GRAPH=1 SPECTOK=7 MAXSEQS=1
  MAXLEN=4096 UTIL=0.90 PORT=18083
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7

COMMAND ->
  ```
  NAME=qwen38_w4a8_sgl bash sglang/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l52-dspark \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=1 MAXLEN=4096 \
      UTIL=0.90 PORT=18083 DEVICE=1 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  ```

RESULT -> HEALTHY 72s (cache). Paris exact.
  c1 avg=33.4 best=34.1 wall~7.6s. Holds vs
  34.7. Logs:
  l52_w4a8_dspark_k7_restore_20260821T102148Z.log
  l52_w4a8_dspark_k7_c1_20260821T102319Z.log

VERDICT -> GO. Spec path restored. Leave
  both vLLM Up. Do not GRAPH=1 sglang until
  GRAPH=0 is coherent. Do not demote 34.7 / 25.0.

---

### 2026-08-21aaa - LOOP 53: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK hold
  vLLM pair. Both serves Up. Attach, do
  not steal. DD PARKED. P2PACCESS=0.
  Do not GRAPH=1 sglang.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. Log:
  l53_w4a8_nomtp_attach_20260821T103546Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21aab - LOOP 54: attach DSpark c1 35.2

CONTEXT -> dual with LOOP 53. Hold isolated
  spec vs 34.7. D14-D19 closed.

CONFIG -> :18083 GRAPH=1 DSpark k=7
  MAXSEQS=1 MAXLEN=4096
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  bench_code c1 vs 34.7

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  ```

RESULT -> Paris exact. 391 exact. c1 avg=35.2
  best=41.6 wall~8.3s. mean_len=2.72
  tok_rate=24.5%. Log:
  l54_w4a8_dspark_attach_20260821T103546Z.log

VERDICT -> GO as hold. Leave both Up.
  Do not GRAPH=1 sglang. Do not demote
  34.7 / 25.0.

---

### 2026-08-21aac - LOOP 55: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK hold
  vLLM pair. Both serves Up. Attach, do
  not steal. DD PARKED. P2PACCESS=0.
  Do not GRAPH=1 sglang.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.2s. Log:
  l55_w4a8_nomtp_attach_20260821T105042Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21aad - LOOP 56: attach DSpark c1 35.5

CONTEXT -> dual with LOOP 55. Hold isolated
  spec vs 34.7. D14-D19 closed.

CONFIG -> :18083 GRAPH=1 DSpark k=7
  MAXSEQS=1 MAXLEN=4096
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  bench_code c1 vs 34.7

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  ```

RESULT -> Paris exact. 391 exact. c1 avg=35.5
  best=38.9 wall~7.5s. mean_len=2.77
  tok_rate=25.2%. Log:
  l56_w4a8_dspark_attach_20260821T105043Z.log

VERDICT -> GO as hold. Leave both Up.
  Do not GRAPH=1 sglang. Do not demote
  34.7 / 25.0.

---

### 2026-08-21aae - LOOP 57: attach NOMTP c1 hold 25.0

CONTEXT -> 15m dual fire. NEXT PICK hold
  vLLM pair. Both serves Up. Attach, do
  not steal. DD PARKED. P2PACCESS=0.
  Do not GRAPH=1 sglang.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP HYBRID=0
  SERVED=qwen3.8-27b-W4A8-gptq-gdn
  bench_code c1 vs 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-gdn 1 256 3
  ```

RESULT -> Paris exact. c1 avg=best 25.0
  wall~10.3s. Log:
  l57_w4a8_nomtp_attach_20260821T110541Z.log

VERDICT -> GO. NOMTP score holds.

---

### 2026-08-21aaf - LOOP 58: attach DSpark c1 37.9

CONTEXT -> dual with LOOP 57. Hold isolated
  spec vs 34.7. D14-D19 closed.

CONFIG -> :18083 GRAPH=1 DSpark k=7
  MAXSEQS=1 MAXLEN=4096
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7
  bench_code c1 vs 34.7

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7 1 256 3
  ```

RESULT -> Paris exact. 391 exact. c1 avg=37.9
  best=41.7 wall~6.1s. this-bench mean_len=3.15
  tok_rate=30.7%. Log:
  l58_w4a8_dspark_attach_20260821T110541Z.log

VERDICT -> GO as hold of same config.
  Do not replace 34.7 with 37.9.

---

### 2026-08-21aag - LOOP 59: K8 e2e int4 lm_head g32 27.0

CONTEXT -> 15m fire after dual attach.
  Isolated K8 already GO. 25.0 locked by
  holds. 3.6 sglang LMHEAD=1 was +7.9%.
  Runtime RTN, do not rewrite 151 (D12).
  Keep DSpark :18083. DD PARKED. P2PACCESS=0.

CONFIG -> GRAPH=1 TP=1 NOMTP HYBRID=0
  LMHEAD=1 LMHEAD_GROUP=32 MAXSEQS=8
  CAPSIZES=1,2,4,8 PORT=18082 DEVICE=0
  SERVED=qwen3.8-27b-W4A8-gptq-lmhead32
  CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn
  IMG=int8g-v0260

COMMAND ->
  ```
  NAME=qwen38_w4a8_gptq bash vllm/w4a8/serve_qwen38_w4a8.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l59-lmhead \
    ./bin/gpu-run --card 0 \
    env GRAPH=1 B70_NOMTP=1 B70_W4A8_HYBRID=0 NOMM=1 \
      MAXSEQS=8 CAPSIZES=1,2,4,8 LMHEAD=1 LMHEAD_GROUP=32 \
      PORT=18082 NAME=qwen38_w4a8_gptq DEVICE=0 CARD=0 \
      P2PACCESS=0 CKPT=/models/qwen3.8-27b/w4a8-gptq-gdn \
      SERVED=qwen3.8-27b-W4A8-gptq-lmhead32 \
    bash vllm/w4a8/serve_qwen38_w4a8.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-lmhead32 1 256 3
  ```

RESULT -> HEALTHY 76s. RTN
  language_model.lm_head N=248320 K=5120
  g=32 int4 0.64GB bf16 kept 2.54GB.
  apply hit M=8 then M=1. Completions
  Paris exact, 391 exact, fib exact.
  LRU bang=0. c1 avg=best 27.0 wall~9.5s
  = 1.08x vs 25.0. Logs:
  l59_w4a8_lmhead_graph1_20260821T111454Z.log
  l59_w4a8_lmhead_c1_20260821T111454Z.log

VERDICT -> GO as lever. NO-GO as 1.10x
  score replacement. Leave LMHEAD=1 Up.
  Do not demote 25.0. Next: LMHEAD=1 on
  DSpark.

---

### 2026-08-21aah - LOOP 60: attach NOMTP lmhead32 27.0

CONTEXT -> 15m dual fire. NEXT PICK K8 on
  DSpark. Attach live :18082 while card 1
  restarts. DD PARKED. P2PACCESS=0.

CONFIG -> :18082 GRAPH=1 TP=1 NOMTP
  LMHEAD=1 g32
  SERVED=qwen3.8-27b-W4A8-gptq-lmhead32
  bench_code c1 vs 27.0 / 25.0

COMMAND ->
  ```
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18082/v1 \
    qwen3.8-27b-W4A8-gptq-lmhead32 1 256 3
  ```

RESULT -> Paris exact. 391 exact. c1
  avg=best 27.0 wall~9.5s. Log:
  l60_w4a8_lmhead_attach_20260821T113610Z.log

VERDICT -> GO. K8 NOMTP holds. Honesty
  25.0 unchanged.

---

### 2026-08-21aai - LOOP 61: DSpark+LMHEAD c1 33.8

CONTEXT -> NEXT PICK. Stack K8 on GRAPH=1
  DSpark k=7. Keep :18082. D14-D19 closed.
  DD PARKED. P2PACCESS=0.

CONFIG -> GRAPH=1 SPECTOK=7 MAXSEQS=1
  CAPSIZES=1 MAXLEN=4096 UTIL=0.90
  LMHEAD=1 LMHEAD_GROUP=32 PORT=18083
  DEVICE=1
  SERVED=qwen3.8-27b-W4A8-gptq-dspark7-lmhead32

COMMAND ->
  ```
  NAME=qwen38_w4a8_dspark bash \
    vllm/w4a8/serve_qwen38_w4a8_dspark.sh stop
  B70_GPU_LOCK_TIMEOUT=0 B70_AGENT=w4a8-l61-dspark-lmhead \
    ./bin/gpu-run --card 1 \
    env GRAPH=1 SPECTOK=7 MAXSEQS=1 CAPSIZES=1 \
      MAXLEN=4096 UTIL=0.90 LMHEAD=1 LMHEAD_GROUP=32 \
      PORT=18083 NAME=qwen38_w4a8_dspark DEVICE=1 \
      CARD=1 P2PACCESS=0 \
      SERVED=qwen3.8-27b-W4A8-gptq-dspark7-lmhead32 \
    bash vllm/w4a8/serve_qwen38_w4a8_dspark.sh start
  python3 -u vllm/nvfp4/bench_code.py \
    http://127.0.0.1:18083/v1 \
    qwen3.8-27b-W4A8-gptq-dspark7-lmhead32 1 256 3
  ```

RESULT -> HEALTHY 66s. RTN
  language_model.lm_head g32. apply hit
  M=7 then M=1. Paris exact. 391 exact.
  LRU bang=0. c1 avg=33.8 best=34.4
  wall~7.8s = 0.97x vs 34.7. drafts=397
  accepted=679 mean_len=2.71 tok_rate=24.4%.
  Logs:
  l61_w4a8_dspark_lmhead_graph1_20260821T113610Z.log
  l61_w4a8_dspark_lmhead_c1_20260821T113610Z.log

VERDICT -> GO as load/coherent. NO-GO as
  1.10x vs 34.7. Leave LMHEAD=1 Up. Do not
  demote 34.7 / 25.0.
