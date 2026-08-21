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
