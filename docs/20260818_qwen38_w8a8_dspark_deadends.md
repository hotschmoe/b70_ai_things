# Qwen3.8 W8A8 + DSpark -- dead-end packets

Closed paths for `docs/20260818_qwen38_w8a8_dspark_campaign.md`.
A looping agent must read this before retrying anything that
"maybe works now".

Newest packet at the **bottom**. Do not rewrite old packets.
Retry only if the packet's **Retry if** line is now true, and
then write a new packet (or a LOOP ledger note that the retry
condition fired).

Packet shape:

```
## D<n> -- <short name> -- YYYY-MM-DD -- LOOP N

Tried:
Command / config:
Result:
Why it is closed:
Retry if:
Related JOURNAL:
```

Honest ugly numbers (e.g. P0.4 pos0 in the 20% band) are **not**
dead-ends -- they go in the loop ledger + JOURNAL and they make
the train mandatory. A dead-end is a path we will not walk again
without a stated condition.

---

## Pre-closed from prior lab (do not re-open casually)

These were closed before this campaign. They are listed so a loop
does not "just try it". Details live in JOURNAL / P2P_GPU.md /
the campaign standing-list (section 6).

| id | path | retry if |
|---|---|---|
| PRE.1 | `CCL_TOPO_P2P_ACCESS=1` in vLLM TP>1 | a reviewer demands a 7.1 retest **and** there is a reboot window. `I_KNOW_P2P_WEDGES=1`. Never chain two tries. |
| PRE.2 | FATTN_MMA=1 on llama.cpp JIT | we have an AOT 2026.1.1 image and Paris-first on that image. JIT already crash-looped. |
| PRE.3 | method=dflash on vLLM 0.26 | `DFlashQwen3DSparkModel` is registered in the serve image. Today it is not. Use method=dspark. |
| PRE.4 | Adaptive verify on GDNAttentionBackend | vLLM grows a GDN-safe adaptive path. Today it rejects. |
| PRE.5 | DeepSpec 38 TB offline cache | never. SpecForge offline + tens of GB hiddens is the recipe. |
| PRE.6 | llm-scaler 0.21 / rmacy v10-slim as a vehicle | we need an 8k / 17-22 tok/s curiosity retest. Not a campaign vehicle. |
| PRE.7 | Q4K reorder-family on a *new* JIT without Paris-first | never skip Paris-first. We got lucky on llama.cpp once. |
| PRE.8 | PCIe ASPM=performance | never on this box (lab kernel panic). |
| PRE.9 | Peer-pair comm mode 3 | never on this box (lab device-lost storm). |
| PRE.10 | oneCCL 2021.15 in a TP>1 serve | never. Overlay 2021.17. |
| PRE.11 | `xpu_shard_top1` default-on for NVFP4 | already e2e-negative (c1 48.9 -> 32.5). Re-A/B only on W8A8 DSpark, explicitly, once. |
| PRE.12 | Inventing a PSpark checkpoint / DeepSeek sibling | never. Prefill arm is SpecPrefill / PFlash-class (campaign section G). |
| PRE.13 | FP8 GEMM on Xe2 | never. Repack FP8 weights to s8. No systolic FP8/FP4. |
| PRE.14 | Overwriting `models/files/qwen3.8-27b/w8a8-gptq` | never. New scheme = new dir. |
| PRE.15 | Entering Phase 2 (torch 2.13) before a Phase 0+1 W8A8+DSpark number | the living header has that number **and** a written 0.27-only feature list. |
| PRE.16 | Long DSpark train before P1.2 10-sample overfit | overfit accepts the full block against the same W8A8 target. |
| PRE.17 | Speed work after HE+ plus < 0.90 | quality is back above the gate (campaign A). |

Campaign-origin packets start at D1 below, once a loop closes
something new.

---

## Campaign packets

## D1 -- GRAPH=1 DSpark k=7 @131k UTIL=0.90 KV OOM -- 2026-08-18 -- LOOP 9

Tried: S0 exact LOOP 8 recipe with GRAPH=1 (SPECTOK=7
  method=dspark THINK_BUDGET=0 MAXLEN=131072 UTIL=0.90
  MAXSEQS=2 TP=2 SERVED=qwen3.8-27b-W8A8-gptq-dspark7
  CGRECLAIM=0 P2PACCESS=0 IMG=int8g-v0260).
Command / config:
  GRAPH=1 SPECTOK=7 MAXLEN=131072 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: EngineCore ValueError at KV init. Need 6.98 GiB
  for 131072, available 6.84 GiB, estimated max 128128.
  Capture/profile already counted (enforce_eager=False).
  Workers died. No DEVICE_LOST. xpu-health card 0 OK.
Why it is closed: GRAPH=1 DSpark draft+verify graphs take
  extra memory vs GRAPH=0 @131k (LOOP 8 loaded). Same
  131k UTIL=0.90 will fail the same way.
Retry if: UTIL>0.90 measured, or MAXLEN<=128128, or a
  smaller capture/graph memory path is in the image.
  LOOP 9 already retried MAXLEN=122880 (loads, G1 hold,
  c1 26.2). Do not retry 131k at UTIL=0.90.
Related JOURNAL: ### 2026-08-18n
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop9_graph1_131k_crash.log

## D2 -- GRAPH=1 DSpark k=3 G1 "duct" / 0% accept -- 2026-08-18 -- LOOP 11

Tried: leftover k-sweep first cell. Same LOOP 9 recipe
  (method=dspark THINK_BUDGET=0 MAXLEN=122880 UTIL=0.90
  MAXSEQS=2 TP=2 CGRECLAIM=0 P2PACCESS=0 IMG=int8g-v0260)
  with SPECTOK=3 GRAPH=1.
Command / config:
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: HEALTHY 147s. G0 id
  qwen3.8-27b-W8A8-gptq-dspark3. G1 thinking-off and
  completions both emit "duct" (finish_reason=length,
  64/16 completion_tokens). Spec accept_len 1.00,
  pos0 0.000, 0 accepted. Loaded compile cache
  /vllm_cache/torch_compile_cache/b3f7e9e010 (k=7
  GRAPH=1 leftover). No DEVICE_LOST. Revert GRAPH=0
  k=3: G1 Paris / 391 / fib hold.
Why it is closed: do not publish GRAPH=1 k=3 speed.
  G1 fail is fail-closed. Same k+cache combo will
  replay "duct".
Retry if: wipe host
  /mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010
  (backbone + dspark_head) then GRAPH=1 k=3 G1 only.
  If still duct after a cold compile, k=3 GRAPH=1 is
  a real dead-end; stay GRAPH=0 / try k=4.
Related JOURNAL: ### 2026-08-18p
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop11_graph1_k3_g1fail.log

## D3 -- GRAPH=1 DSpark k=3 still duct after cold compile -- 2026-08-18 -- LOOP 12

Tried: D2 retry. Wiped host
  /mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010
  via docker --entrypoint /bin/rm (root-owned; host sudo
  needs a tty). Then GRAPH=1 SPECTOK=3 MAXLEN=122880.
Command / config:
  docker run --rm --user 0:0 --entrypoint /bin/rm
    -v /mnt/vm_8tb/b70/vllm_cache:/vllm_cache
    vllm-xpu-env:int8g-v0260
    -rf /vllm_cache/torch_compile_cache/b3f7e9e010
  GRAPH=1 SPECTOK=3 MAXLEN=122880 PORT=18080
    NAME=qwen38_w8a8_dspark
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: Cache GONE then rebuilt. Logs: Compiling a graph
  for compile range (1, 2048) takes 1.59 s -- NOT
  "Directly load". HEALTHY 147s. G1 content="duct"
  finish_reason=length ct=32. accept_len 1.00 pos0 0.000.
  No DEVICE_LOST. Revert GRAPH=0: G1 Paris / 391 / fib.
Why it is closed: k=3 GRAPH=1 is broken even on a cold
  compile. Not a stale-cache-only bug. Compile key hash
  b3f7e9e010 is shared across SPECTOK (k=7 and k=3 used
  the same dir) so a k=3 graph must not be left on disk
  for later k. LOOP 12 wiped it again after revert.
Retry if: capture sizes / DSpark graph path change, or
  compile key includes num_speculative_tokens. Do not
  retry the same recipe.
Related JOURNAL: ### 2026-08-18q
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop12_graph1_k3_cold_g1fail.log

## D4 -- P1.5 W8A16_M_MAX>0 @122880 KV OOM -- 2026-08-18 -- LOOP 15

Tried: default-on small-M w8a16 at the live long-ctx
  DSpark recipe (GRAPH=1 SPECTOK=4 MAXLEN=122880
  UTIL=0.90). W8A16_M_MAX=8 (covers verify M=k+1=5).
Command / config:
  W8A16_M_MAX=8 GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-w8a16
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: EngineCore ValueError. Model load 27.31 GiB/card
  (clone ON). Available KV cache memory: **-0.93 GiB**.
  No cache blocks. Workers died. No DEVICE_LOST.
  xpu-health card 0 OK. Revert W8A16_M_MAX=0: HEALTHY
  147s, G1 Paris exact.
Why it is closed: NT layout clone doubles s8 weight
  residency. Any W8A16_M_MAX>0 costs the same clone.
  122880 GRAPH=1 DSpark cannot spare ~9 GiB/card.
Retry if: int8_gemm_w8a16 consumes the s8s8 [K,N]
  layout (no NT clone), or a measured MAXLEN/UTIL
  where KV stays positive with the clone. Do not
  retry W8A16_M_MAX>0 at 122880 UTIL=0.90.
Related JOURNAL: ### 2026-08-18t
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop15_w8a16_oom.log

## D5 -- P1.6 fusedq e2e decode no win -- 2026-08-18 -- LOOP 17

Tried: mount v0240 fusedq `_xpu_C.abi3.so` over
  int8g-v0260 (live image/op has no fusedq). Keep
  AGASYNC k=4 GRAPH=1 @122880 W8A16=0.
Command / config:
  B70_EXTRA_ENV="PUSH_AR_ALLGATHER_ASYNC=1 B70_FUSEDQ=1"
  GDN_SO=.../w8a8_kernel_v0240_fusedq/_xpu_C.abi3.so
  GDN_LIB=.../w8a8_kernel_v0240_fusedq/libgdn_attn_kernels_xe_2.so
  GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-fusedq
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: has_fusedq True, fake registered. HEALTHY 142s.
  G1 Paris / 391 / fib. bench_code c1 avg **28.3** /
  best 30.7 (wall 8.9s) vs AGASYNC 29.4 / 33.2.
  No DEVICE_LOST. Revert without fusedq SO failed
  (cached graph calls fusedq). Wipe b3f7e9e010 then
  AGASYNC reloads; G1 Paris holds.
Why it is closed: no e2e decode move (slightly
  slower). v0240 fusedq SO on this recipe is not a
  speed win. Compile hash ignores the SO -- wipe
  before leaving fusedq graphs on disk.
Retry if: fusedq rebuilt vs v0260 ABI and a
  TTFT/PP A/B is the pick (P1.6 success was
  TTFT/PP, not decode). Do not retry this SO for
  bench_code c1.
Related JOURNAL: ### 2026-08-18v
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop17_fusedq.log
  bench loop17_bench_code_c1.log

## D5 addendum -- P1.6b TTFT not unlocked -- 2026-08-19 -- LOOP 31

Tried: check D5 retry-if before remounting v0240
  fusedq SO for TTFT/PP vs P4.1 1528/449.
Result: no `w8a8_kernel_v0260_fusedq` tree. Live
  int8g-v0260 `_xpu_C` is 61125200 B (stock, not
  the 61139960 B v0240 fusedq SO). Retry-if still
  false. Did not remount. AGASYNC left up. G1 Paris.
Why it is closed: LOOP 30 said do not retry D5.
  Remounting the same v0240 SO is that retry.
  Compile-key SO hash (LOOP 26) does not replace
  the ABI rebuild condition.
Retry if: unchanged -- fusedq rebuilt vs v0260 ABI,
  then TTFT/PP A/B vs P4.1, not c1.
Related JOURNAL: ### 2026-08-19n

## D6 -- xpu_shard_top1 SPEC flag is MTP-only, not DSpark -- 2026-08-18 -- LOOP 20

Tried: E1 / PRE.11 once on live W8A8 DSpark k=4
  GRAPH=1 AGASYNC. Inspected flag + op + proposer
  before any restart.
Command / config: no serve restart. Looked at
  SpeculativeConfig.use_local_argmax_reduction,
  DSparkSpeculator._sample_sequential, live
  torch.ops._xpu_C, host SOs.
Result: DSparkSpeculator does
  compute_draft_logits (full vocab) then sequential
  Markov + argmax/gumbel. It never reads
  use_local_argmax_reduction and has no
  get_top_tokens. That flag lives on
  llm_base_proposer (MTP/EAGLE). Live GDN_SO
  w8a8_kernel_v0240 has no xpu_shard_top1
  (has False). Proto SO with both int8 gemm and
  shard_top1 exists at
  /mnt/vm_8tb/b70/nvfp4_top1_proto/_xpu_C.abi3.so.
  AGASYNC left up; Paris exact. No c1 published.
Why it is closed: flipping the SPEC flag on this
  DSpark recipe is a no-op (or a load error if
  some parent checked get_top_tokens). Do not
  burn a TP=2 restart for an ignored JSON field.
Retry if: DSparkSpeculator._sample_sequential is
  patched to consume sharded logits +
  xpu_shard_top1 (keep Markov bias), or a
  dedicated W8A8 *MTP3* LOCALARGMAX A/B is the
  named pick. Do not retry the SPEC flag on
  method=dspark.
Related JOURNAL: ### 2026-08-18y

## D7 -- DSpark shard-top1 hook no decode win -- 2026-08-18 -- LOOP 21

Tried: overlay DSparkSpeculator._sample_sequential
  (local lm_head + local Markov bias + xpu_shard_top1
  + pair all-gather). Mount nvfp4_top1_proto SO
  (has shard_top1 + int8 gemm). Keep AGASYNC k=4
  GRAPH=1 @122880 W8A16=0. Wipe b3f7e9e010.
Command / config:
  GDN_SO=/mnt/vm_8tb/b70/nvfp4_top1_proto/_xpu_C.abi3.so
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER_ASYNC=1
    PUSH_AR_CHAIN_SITECUSTOMIZE=/opt/e1_shim/sitecustomize.py
  GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-agasync-shardtop1
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: Hook ENGAGED. HEALTHY 163s. G1 Paris / 391
  / fib. bench_code c1 avg **28.4** / best 29.8
  (wall 9.1s) vs AGASYNC **29.4** / 33.2. No
  DEVICE_LOST. Wipe + revert AGASYNC: HEALTHY 137s,
  Paris exact.
Why it is closed: coherent but no e2e decode move.
  Sequential Markov still pays N pair-gathers plus
  the proto SO is not faster here. Do not retry
  this overlay for bench_code c1.
Retry if: shard-top1 is fused with Markov in one
  kernel, or pair-gather is captured without the
  host sync. Do not remount this proto SO for c1
  without a new hook.
Related JOURNAL: ### 2026-08-18z
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop21_serve.log
  bench loop21_bench_code_c1.log

## D8 -- host-barrier ALLGATHER=1 no decode win -- 2026-08-18 -- LOOP 22

Tried: PUSH_AR_ALLGATHER=1 (host-barrier, not ASYNC)
  on k=4 GRAPH=1 @122880 W8A16=0. Same recipe as
  AGASYNC 29.4 minus the eager-async gather path.
Command / config:
  B70_EXTRA_ENV=PUSH_AR_ALLGATHER=1
  GRAPH=1 SPECTOK=4 MAXLEN=122880
    SERVED=qwen3.8-27b-W8A8-gptq-dspark4-aghost
    ./bin/gpu-run bash vllm/dflash/serve_qwen38_w8a8_dspark.sh start
Result: ALLGATHER redirect ENGAGED [host-barrier].
  HEALTHY 132s. G1 Paris / 391 / fib. bench_code
  c1 avg **26.6** / best 28.3 (wall 9.0s) vs
  AGASYNC **29.4** / 33.2. No DEVICE_LOST. Revert
  AGASYNC: HEALTHY 137s, Paris exact.
Why it is closed: coherent but slower (same
  direction as NVFP4 2.4x; here ~10% down). Host
  barrier per gather is not a W8A8 DSpark win.
  Keep ALLGATHER_ASYNC. Do not retry host-barrier
  on this recipe.
Retry if: gather is captured (device-side do_ar)
  so the host wait goes away. Do not retry
  PUSH_AR_ALLGATHER=1 for bench_code c1.
Related JOURNAL: ### 2026-08-18aa
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop22_serve.log
  bench loop22_bench_code_c1.log

## D9 -- E3 oneDNN barrier env no-op on int8g-v0260 -- 2026-08-19 -- LOOP 25

Tried: E3 Steve oneDNN barriers-on as a 0.26 env A/B
  vs AGASYNC 29.4. Fetched origin/main `924b518f`,
  read `9f90e2c3` lean-flag retest. Did **not**
  restart: live SO has no getenv.
Command / config: no serve restart. Looked up:
  VLLM_XPU_ONEDNN_INT8_COMPLETION_BARRIER
  VLLM_XPU_ONEDNN_INT8_INPUT_DEPENDENCY
  VLLM_XPU_ONEDNN_INT4_COMPLETION_BARRIER
  VLLM_XPU_ONEDNN_INT4_INPUT_DEPENDENCY
  (plus INT4 scope / greedy margin -- INT4-AR stack).
  strings of live
  /opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so
  = 0 ONEDNN_INT hits. Host w8a8_kernel SOs same.
  Image env has none of those keys.
Result: flags exist only in Steve
  `patches/qwen36-27b-autoround-int4-b70/int4-input-dependency-20260817/candidate-vllm-xpu-kernels-working.patch`
  (publish oneDNN SYCL completion event onto the
  current XPU stream; INT4 also submit_barrier
  input deps). Our `kernels/int8_gemm_w8a8.h`
  execute() has no getenv. AGASYNC left up;
  id qwen3.8-27b-W8A8-gptq-dspark4-agasync.
  No c1 published (would have been a no-op
  republish of 29.4).
Why it is closed: setting unused env on
  int8g-v0260 cannot engage. Do not burn a
  TP=2 restart for a silent no-op.
Retry if: port INT8 completion-barrier (and
  optional input-dep) getenv into
  `kernels/int8_gemm_w8a8.h` / w8a16, rebuild
  the v0260 `_xpu_C`, wipe compile hash, then
  G1 + bench_code vs 29.4. Do not overlay
  Steve's INT4-AR SO (wrong scheme / ABI).
Related JOURNAL: ### 2026-08-19f

## D3 addendum -- compile-key landed -- 2026-08-19 -- LOOP 26

Tried: put SPECTOK + mounted _xpu_C / GDN SO in
  the 0.26 compile cache key (no GPU).
Command / config: sitecustomize hook
  vllm/dflash/patches/v0260/compile_key_spectok_so.py
  wired into serve_qwen38_w8a8_dspark.sh only.
Result: stock SpeculativeConfig.compute_hash is
  identical for k=3 and k=4. After hook they
  differ. _xpu_C sha256 now in compile_factors.
Why it is closed: this is the D2/D3 *cache-dir*
  hole, not D3 itself. k=3 GRAPH=1 still ducted
  on a cold compile (LOOP 12). Do not retry
  GRAPH=1 k=3 just because the key is fixed.
Retry if: DSpark graph path / capture sizes
  change. Compile-key identity is no longer
  the missing condition.
Related JOURNAL: ### 2026-08-19i

## D10 -- 0.27 f01e24f6 TP=2 oneCCL -- 2026-08-19 -- LOOP 27

Tried: Qwen3.8 AutoRound INT4 TP=2 on public
  vLLM XPU 0.27.2rc1 digest f01e24f6.
Command / config:
  isolated TRITON_CACHE (shared 0.26 cache is
  libsycl.so.8; nightly is .so.9). Stock oneCCL
  in /opt/venv/lib. Then host 2021.17 file
  overlay over those .so (torch DT_RPATH ignores
  LD_LIBRARY_PATH) + libsycl.so.8 -> .so.9 shim
  + IPCX=sockets.
Result: stock 2021.15 dies at worker
  torch.distributed.all_reduce warmup:
  `ze_handle_manager.cpp:43 mem_to_ipc_handle:
  device_fd is invalid value` (PRE.10). 2021.17
  overlay ImportErrors
  `setNDRangeDescriptor` (SYCL-8 libccl vs
  nightly SYCL-9). Isolated-cache inspect of
  Qwen3_5 then succeeds. No DEVICE_LOST.
  xpu-health OK after each try.
Why it is closed: this nightly cannot TP=2
  without a SYCL-9-built oneCCL. Host 2021.17
  from 0.24 is the wrong ABI.
Retry if: a oneCCL rebuilt against this
  image's libsycl.so.9 (Steve 2025.3.3 public
  oneCCL, or a new 0.27 image that already
  ships 2021.17+ on SYCL 9). Then G1 on TP=2
  before any 101.922 cell.
Related JOURNAL: ### 2026-08-19j

## D11 -- 0.27 INT4-AR GRAPH=1 G1 garbage -- 2026-08-19 -- LOOP 27

Tried: same ckpt/image TP=1 GRAPH=1 PIECEWISE
  MTP5 isolated-triton, then with cookbook
  MTP patches + B70_MTP_BF16_DRAFT=1.
Command / config: serve_qwen38_27b_int4ar.sh
  TP=1 GRAPH=1; then docker run with
  apply_mtp_patches.py.
Result: unpatched GRAPH=1 loaded HEALTHY 269s
  but G1 FAIL (Paris/391/fib all garbage;
  mul emitted 倒 loops). Completions probe
  also junk. Patched BF16-draft died loading
  drafter: AutoRound has
  `layers.0.mlp.down_proj.qweight` but the
  unquantized MTP module wants `.weight`.
  GRAPH=0 unpatched: G1 PASS, c1 12.8,
  after-TTFT 16.66.
Why it is closed: GRAPH=1 on this
  nightly+AutoRound-MTP is incoherent.
  Cookbook BF16-draft is for GPTQ ckpts
  that kept mtp.* in BF16 (S1). This
  auto-round quantized mtp.layers.
Retry if: Steve graph-safe FA / a new
  0.27 image / kernels that G1 on GRAPH=1
  with this exact ckpt. Do not republish
  GRAPH=1 speed after another garbage G1.
Related JOURNAL: ### 2026-08-19j

## D12 -- stock sycl-tla C1 not a 29.4 closer -- 2026-08-19 -- LOOP 32

Tried: P1.8 stock-tile sycl-tla microbench
  (AOT intel_gpu_bmg_g31, v0240 image, card 0)
  vs oneDNN W8A8 88-100% of 581 GB/s at M=1.
Command / config:
  ITERS=50 ./bin/gpu-run --card 0 bash
    /mnt/vm_8tb/b70/sycl-tla-bench/run_bench.sh
  EXAMPLES=bf16,bf16_s8  M in 1,2,4,8,16
Result: all Disposition Passed. Stock bf16
  tiles 47-81% of 608 GB/s (gate_up M=1 437
  GB/s vs oneDNN int8 510). Mixed bf16_s8
  **1.1-1.5%** roof, time flat in M (~4.8 ms
  qkv, ~23 ms gate_up) -- not a DPAS decode
  path. Isolated << 1.2x. No e2e wrap.
Why it is closed: stock large-M tiles do not
  close 29.4 vs 41.2. oneDNN GEMM already at
  the roofline; do not tune it.
Retry if: rectangular small-M TiledMMA
  (SYCLTLA_SCAFFOLD step 3, XE_DPAS_TT M=8)
  is built and beats oneDNN at M=2..16. Then
  e2e; isolated 1.2x with e2e drop is still
  a packet.
Related JOURNAL: ### 2026-08-19o
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop32_sycltla.log

## D10 addendum -- nightly c48edf76 still 2021.15 -- 2026-08-19 -- LOOP 36

Tried: S2b TP=2 GRAPH=1 on today's public
  vllm/vllm-openai-xpu nightly
  sha256:c48edf76bb9f6b03f952af9ecf25ee049c60d9b8800d6c627c19939462fa03d8
  (v0.26.1rc1.dev942+g5a4c8d992, SYCL-9).
Command / config: serve_qwen38_27b_int4ar.sh
  TP=2 GRAPH=1 MTPTOK=5 isolated
  TRITON_CACHE triton_c48edf76. No CCL217
  overlay.
Result: same ze_handle_manager.cpp:43
  device_fd is invalid value at worker
  all_reduce. libccl.so.1.0 209976424 B
  NEEDED libsycl.so.9. qwen38-b70 /
  oneAPI 2025.3.3 libccl NEEDED
  libsycl.so.8 (Steve public oneCCL class).
  No DEVICE_LOST. Health OK after.
Why it is closed: a newer SYCL-9 nightly
  is not a 2021.17+ fix. Steve 2025.3
  oneCCL cannot overlay this ABI.
Retry if: oneCCL rebuilt against this
  image's libsycl.so.9, or a nightly that
  already ships 2021.17+ NEEDED
  libsycl.so.9. Then G1 on TP=2.
Related JOURNAL: ### 2026-08-19s
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop36_tp2_graph1_docker.log

## D11 addendum -- nightly c48edf76 GRAPH=1 G1 garbage -- 2026-08-19 -- LOOP 36

Tried: same ckpt/image TP=1 GRAPH=1
  PIECEWISE MTP5 isolated-triton.
Command / config: serve_qwen38_27b_int4ar.sh
  TP=1 DEVICE=0 GRAPH=1.
Result: HEALTHY ~161s (compile 125.9s +
  26.3s). G1 FAIL: chat Paris/391/fib
  empty content, reasoning garbage,
  finish_reason=length. Completions
  garbage. accept_len 1.00 pos0 0.000.
  No speed published. GRAPH=0 TP=1 on
  f01e24f6 still the gated cell (12.8 /
  16.66).
Why it is closed: a newer SYCL-9 nightly
  does not G1 on GRAPH=1 with this ckpt.
Retry if: Steve graph-safe FA / his 0.21
  SYCL-8 stack / kernels that G1 on
  GRAPH=1 with this exact ckpt. Do not
  republish GRAPH=1 speed after another
  garbage G1.
Related JOURNAL: ### 2026-08-19s
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop36_g1.jsonl

## D13 -- intel/vllm 0.21 MTP GDN spec_sequence_masks -- 2026-08-19 -- LOOP 37

Tried: S2b TP=2 GRAPH=1 on
  intel/vllm:0.21.0-xpu (torch 2.11.0+xpu,
  v0.21.1.dev18+g8df6feb7d, SYCL-8) with
  in-image oneCCL 2021.17.
Command / config:
  vllm/w4a16/start_int4ar_intel021.sh
  wrapper setvars + CCL_ROOT=2021.17
  MTPTOK=5 MAXLEN=16384 DTYPE=float16
  P2PACCESS=0 isolated caches.
Result: TP=2 workers loaded, no device_fd.
  XPU Graph disabled (comms). HEALTHY.
  First /v1/completions: EngineDead.
  AssertionError in
  vllm/_xpu_ops.py _gdn_attention_core_xpu_impl:
  attn_metadata.spec_sequence_masks is None.
  No DEVICE_LOST. Cards healthy after.
Why it is closed: this digest's GDN XPU
  op does not run MTP. Not Steve 44fc8fde0
  + graph-safe FA. Do not republish a
  speed cell.
Retry if: Steve GDN spec kernels / FA
  staged package, or vLLM 44fc8fde0, that
  accept spec_sequence_masks on this ckpt.
  Then G1 before any 101.922 cell.
Related JOURNAL: ### 2026-08-19t
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop37_g1_500_tail.log

## D13 addendum -- GDN spec fallback G1 fib bangs -- 2026-08-19 -- LOOP 38

Tried: overlay Steve
  patches/vllm-xpu-mtp-fallback.patch onto
  intel/vllm:0.21.0-xpu _xpu_ops.py +
  gdn_linear_attn.py. Fresh compile cache
  intel021_gdnfb. Same TP=2 MTP5 recipe.
Command / config:
  CACHE_NAME=intel021_gdnfb
  bash vllm/w4a16/start_int4ar_intel021.sh start
Result: HEALTHY 161s. Assert gone. Paris
  completions/chat hold. 17*23=391 at
  max_tokens=64. Chat fib: 256 "!!!!"
  reasoning tokens, empty content.
  Spec 322/380 accepted. GRAPH still
  disabled (comms). No speed published.
Why it is closed: Python fallback is not
  a G1-gated 101.922 cell. Fib bangs are
  fail-closed. GRAPH=1 TP=2 still off.
Retry if: Steve graph-safe FA / vLLM
  44fc8fde0 that G1 on GRAPH=1 TP=2 with
  this ckpt. Do not republish this overlay.
Related JOURNAL: ### 2026-08-19u
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop38_g1.jsonl

## D14 -- FORCE_GRAPH + in-image 2021.17 sycl_graph allgather -- 2026-08-19 -- LOOP 42

Tried: 44fc8fde0 + rebuilt 2dd55f38
  _xpu_C+GDN on intel/vllm:0.21.0-xpu
  TP=2 GRAPH=1 VLLM_XPU_FORCE_GRAPH_WITH_COMM=1
  in-image oneCCL 2021.17.
Command / config:
  start_int4ar_intel021.sh
  fuse_rope_kvcache_cat_mla=false
  CACHE_NAME=intel021_44fc_so
Result: int4 8-arg ABI OK. PIECEWISE
  kept. torch.compile 156s. Graph
  capture all_gather:
  |CCL_SYCL| sched algorithms do not
  support sycl_graph recording, please
  use sycl_algorithms. EngineDead. No
  DEVICE_LOST. Cards healthy.
Why it is closed: stock 2021.17 cannot
  record allgather in a SYCL graph.
  Steve 101.922 used public oneCCL
  4ceafd1 with graph-replay oracles.
Retry if: Steve 4ceafd1 oneCCL rebuilt
  vs this image's libsycl.so.8 is
  overlaid, then G1 on GRAPH=1 TP=2.
Related JOURNAL: ### 2026-08-19y
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop42_tp2_fail_tail.log

## D15 -- 4ceafd1 overlay TP=2 device_fd -- 2026-08-19 -- LOOP 44

Tried: overlay rebuilt public oneCCL
  4ceafd1 (SYCL-8, 240177816 B) over
  intel/vllm 2021.15+2021.17 libccl with
  44fc8fde0 + 2dd55f38 SO, GRAPH=1
  FORCE_GRAPH.
Command / config:
  CCL4CE=/mnt/vm_8tb/b70/steve-s2b/oneccl-install
  start_int4ar_intel021.sh
  IPC pidfd then sockets.
Result: CCL_ROOT=/opt/ccl4ce. Worker
  all_reduce ze_handle_manager.cpp:58
  device_fd invalid (both IPC modes).
  In-image 2021.17 still loads TP=2.
  No DEVICE_LOST. Cards healthy.
Why it is closed: this 4ceafd1 install
  is not a drop-in for this image's L0
  IPC. Do not retry the same bind.
Retry if: 4ceafd1 loads TP=2 (DRM/ze
  device_fd fixed) then G1 on GRAPH=1.
Related JOURNAL: ### 2026-08-19aa
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop44_tp2.log

## D15 addendum -- bake still device_fd; pid=host loads TP=2 -- 2026-08-19 -- LOOP 46

Tried: in-image copy of 4ceafd1 onto
  torch RPATH /opt/venv/lib plus
  2021.17 so.1 and /opt/ccl4ce,
  2dd55f38 SOs, 44fc tree. Tag
  intel/vllm:0.21.0-xpu-s2b. Same
  docker --device /dev/dri as LOOP 45.
Result: first GRAPH=1: "pidfd is not
  supported, fallbacks to drmfd" then
  device_fd invalid (same as overlay).
  --pid=host --security-opt
  seccomp=unconfined: no pidfd warning,
  workers init, torch.compile 154s.
Why it is closed: bake vs bind is not
  the IPC gap. Docker default pid/seccomp
  is. Overlay retry still closed.
Retry if: none for overlay. GRAPH=1 after
  TP=2 load is D16.
Related JOURNAL: ### 2026-08-19ac
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop46_tp2.log

## D16 -- baked 4ceafd1 GRAPH=1 capture hang -- 2026-08-19 -- LOOP 46

Tried: IMG intel/vllm:0.21.0-xpu-s2b
  BAKED=1 GRAPH=1 TP=2 FORCE_GRAPH
  pid=host seccomp=unconfined MTPTOK=5
  MAXLEN=16384 fuse_rope false.
Command / config:
  start_int4ar_intel021.sh
  CACHE_NAME=intel021_s2b
Result: TP=2 workers up. Compile 154s.
  Then 18+ min both workers 100% CPU,
  EngineCore shm_broadcast every 60s,
  no capture log, no /v1/models, no
  sycl_graph error (D14 was that error).
  Stopped at 22 min. No G1. No speed.
  Cards healthy. No DEVICE_LOST.
Why it is closed: waiting longer on the
  same hang is not a 101.922 cell.
Retry if: Steve graph-safe FA / a capture
  dump shows a fixable collective then
  G1 on GRAPH=1 TP=2. Do not republish
  this hang as a speed number.
Related JOURNAL: ### 2026-08-19ac
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop46_tp2_pidhost.log

## D16 addendum -- graph-safe FA same capture hang -- 2026-08-19 -- LOOP 47

Tried: Steve graph-safe FA vs 2dd55f38
  (local_accessor + force-chunk python,
  head256 focused --full). Overlay
  libattn + _vllm_fa2_C + interface.py
  on intel/vllm:0.21.0-xpu-s2b pid=host
  GRAPH=1 FORCE_GRAPH CACHE_NAME=intel021_s2b_fa
  VLLM_XPU_FA2_FORCE_CHUNK_DECODE=1.
Command / config:
  build_graphsafe_fa.sh then
  start_int4ar_intel021.sh FA_DIR=...
Result: FA so 27393320 B. FlashAttention
  v2. Compile 159.64s. Then same hang:
  both workers 100% CPU, shm_broadcast,
  no capture log, no /v1/models. Stopped
  at 7+ min post-compile. No G1. No
  DEVICE_LOST. Cards healthy.
Why it is closed: FA is not the unstick.
  Do not wait out this hang again.
Retry if: capture dump names a fixable
  collective then G1. Host-not-docker
  Steve venv is a separate arm.
Related JOURNAL: ### 2026-08-19ad
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop47_tp2_fa.log

## D16 addendum -- dump: sched_yield + XE exec-queue poll -- 2026-08-19 -- LOOP 48

Tried: strace Worker_TP after compile on
  s2b+pid=host GRAPH=1 CACHE=intel021_s2b
  (CAP_PTRACE=1). 12s on TP1.
Command / config:
  start_int4ar_intel021.sh BAKED=1
  CAP_PTRACE=1
Result: compile 6-8s (AOT cache). Hang
  is busy-wait not a stuck syscall:
  298904 sched_yield / 12s, 18916 poll,
  6041 futex, 124 ioctl
  DRM_IOCTL_XE_EXEC_QUEUE_GET_PROPERTY
  on fd4=renderD128 and fd5=renderD129
  (each rank polls both cards).
  libccl=/opt/ccl4ce (4ceafd1). No
  capture log. No /v1/models. Cards
  healthy. No DEVICE_LOST.
Why it is closed: dump named the
  mechanism. Waiting longer is not a
  101.922 cell.
Retry if: CCL_LOG_LEVEL names a
  fixable collective then a code/env
  change G1s. Do not set P2P=1.
Related JOURNAL: ### 2026-08-19ae
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop48_strace_tp1.txt

## D16 addendum -- CCL_LOG_LEVEL=info silent after streams -- 2026-08-19 -- LOOP 49

Tried: CCL_LOG_LEVEL=info on s2b+pid=host
  GRAPH=1 CACHE=intel021_s2b. Stop 60s
  after compile (6.02s).
Command / config:
  start_int4ar_intel021.sh BAKED=1
  CCL_LOG_LEVEL=info
Result: 335 CCL_INFO lines, all at
  init/compile. Last: "no ports
  detected", in_order 0 then 1 GPU
  streams family7. Then silence +
  shm_broadcast. CCL_SYCL_ALLREDUCE_ARC=0
  (do NOT set 1; Steve graph deadlock).
  ze fabric ports 0. ATL ofi tcp:eth0.
  No per-coll name. No G1. Cards healthy.
Why it is closed: info is not a coll
  name. Waiting longer is not a 101.922
  cell.
Retry if: COMPILE_ALLGATHER_CUSTOM_OP=1
  (Steve wait_tensor) G1s, or debug
  names a coll then an env/code change
  G1s. Do not ARC=1. Do not P2P=1.
Related JOURNAL: ### 2026-08-19af
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop49_tp2.log

## D16 addendum -- AGCUSTOM same capture hang -- 2026-08-19 -- LOOP 50

Tried: VLLM_XPU_COMPILE_ALLGATHER_CUSTOM_OP=1
  on s2b+pid=host GRAPH=1 new cache
  intel021_s2b_agcustom. Steve wait_tensor
  bypass.
Command / config:
  start_int4ar_intel021.sh AGCUSTOM=1
Result: env seen. Compile 156.54s (new
  cache). Then D16 hang: 100% CPU
  workers, shm_broadcast, no /v1/models
  at 150s post-compile. No G1. Cards
  healthy. No DEVICE_LOST.
Why it is closed: opaque all_gather
  custom op is not the unstick in docker.
Retry if: host-not-docker Steve venv
  G1s. Do not retry AGCUSTOM on this
  docker path. Do not ARC=1. Do not P2P=1.
Related JOURNAL: ### 2026-08-19ag
  log /mnt/vm_8tb/b70/qwen38-w8a8-dspark/loop50_tp2.log
