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
