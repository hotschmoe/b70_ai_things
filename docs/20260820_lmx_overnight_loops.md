# LocalMaxxing overnight -- loop ledger

Standing feedback for `docs/20260820_lmx_overnight_plan.md`.
Newest loop at the **bottom**. Do not rewrite old loops.

Runtime status: `/mnt/vm_8tb/b70/lmx_overnight/STATUS` (not git).

---

## NEXT PICK (keep this line true)

Park. Overnight speed rows closed.
k1bar-pc1 UP ~90m G1 GO. Soak code
c1 24.8. Hold 31.9 not demoted.
Do not P2P. Do not DD.

---

## LOOP 0 -- 2026-08-20T0705Z -- contract + 2026.08.19 patch wire, no GPU

Picked: author overnight plan/ledger/dead-ends; copy cookbook
  2026.08.19 patches into `vllm/patches/cookbook/`; apply them
  from `apply_mtp_patches.py`; `DRAFT_INT4=1` on launch.sh;
  add `sweep_lmx_w1_draftint4.sh`. Start 30m durable scheduler.
Why this, not the other open row: LocalMaxxing review said
  draft-INT4 is the 27B W4A16 unlock (83.7 -> 112.7) and S1 on
  this box was 47.58 without that overlay. Ckpt and f01e24f6
  image are already local. Cards free.
GPU: none this fire (parent). Lease free both cards. DD PARKED.
Command: docs + patch copy + launch/apply edits.
Log: n/a
Result: contract LOOPING. Next pick W1 GPU.
Verdict: GO (plan + protocol). No number moved.
Changed beliefs: 30m is a check-in. W1 serve may span two fires.
  Do not start a second serve. Hold Ornith 34.9 and k1bar 31.9
  until something beats them on their own metric.
Next pick: W1 sweep on card 0.
Do not: start DD; P2P; emul NVFP4; demote W8A8; 4x B70; kill a
  healthy W1 serve at 30m.
Restore: DD PARKED. Cards free for W1.
JOURNAL: ### 2026-08-20au

## LOOP 0b -- 2026-08-20T0740Z -- add Pliny OBLITERATED Q8_0 to the loop

Picked: add P1. Fetch Q8_0 GGUF only (~29 GB). Write SYCL
  serve + sweep. Q8 fused doors ON, Q4K OFF. Manifest id
  qwen3.8-27b/obliterated-q8. No GPU serve until file complete.
Why this, not the other open row: operator asked to add and
  optimize the Pliny Q8 that just landed. W1 stays in queue.
GPU: none (fetch is CPU). Cards still free. DD PARKED.
Command:
  hf download OBLITERATUS/Qwen3.8-27B-OBLITERATED \
    Qwen3.8-27B-OBLITERATED-Q8_0.gguf \
    --local-dir models/files/qwen3.8-27b/obliterated-q8
  llamacpp/serve_qwen38_obliterated_q8.sh
  llamacpp/sweep_obliterated_q8.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1_hf_q8.log
Result: download started. Serve scripts landed. Next GPU
  is P1 when GGUF >= 25 GB, else W1.
Verdict: GO (added). No tok/s yet.
Changed beliefs: Q8_0 is weight-only (~W8A16), not our
  W8A8-INT8 XMX path. Historical B70 Q8_0 was slow vs Q4_K_M.
  Profile Q8_DOORS before writing new GEMV.
Next pick: P1 sweep if GGUF complete, else W1.
Do not: fetch whole OBLITERATUS repo; vLLM P2P; demote
  31.9/34.9; start DD; 4x B70.
Restore: DD PARKED.
JOURNAL: ### 2026-08-20av

## LOOP 2 -- 2026-08-20T0725Z -- P1 OBLITERATED Q8_0 2x SYCL G1 GO

Picked: fetch complete (27.7 GiB); serve 2x B70 llama.cpp
  SYCL Q8_DOORS=1 MMVQ pair/triple/quad; G1 + phase_bench.
Why this, not the other open row: operator-priority P1.
  First start failed: image ENV MODEL_SHA256 was the Q4_K_M
  pin. Cleared and retried.
GPU: both cards. Container qwen38_oblit_q8 :8010.
  Lease pid 237652 docker wait. DD PARKED.
Command:
  MODEL_SHA256= bash llamacpp/sweep_obliterated_q8.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1_phase_20260820T0722Z.json
Result: G1 Paris/391 PASS. median post_first **30.89**
  tok/s (n=5, early EOS ~20 tok). Warmup g128 **32.14**.
  Prefill proxy **569**. TTFT 2.02s. vs Q4_K_M 43.8
  (~1.36x slower, expected weight-only Q8). Coherent.
Verdict: GO. First live Pliny Q8 on this box.
Changed beliefs: SHA pin must be cleared vs 0xSero image
  ENV. COMM_DIRECT_Q8=3 still forbidden. Q8-DEDUP stats
  dump at process exit -- capture on stop.
Next pick: P1b Q8_DOORS A/B + longer gen.
Do not: vLLM P2P; stop a healthy serve for W1; fetch
  whole HF repo; COMM_DIRECT_Q8=3.
Restore: serve LEFT UP. DD PARKED.
JOURNAL: ### 2026-08-20ax

## LOOP 1 -- 2026-08-20T0715Z -- W1 draft-INT4 G1 GO 65.08

Picked: W1 3.8 GPTQ-Int4 MTP4 + cookbook 2026.08.19
  draft-INT4 on f01e24f6, 1x B70, G1 + phase_bench
  p512/g128 n=5 vs S1 47.58.
Why this, not the other open row: P1 GGUF was
  incomplete (~15.6/29 GB). W1 was the first
  unblocked GPU row; ckpt and image local.
GPU: card 0 via sweep gpu-run. Card 1 free.
  Container lmx_w1_d38. DD PARKED. P2P off.
Command:
  bash vllm/cookbook_campaign/sweep_lmx_w1_draftint4.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/w1_serve_20260820T0709Z.log
  G1 /mnt/vm_8tb/b70/lmx_overnight/w1_g1_20260820T0709Z.log
  bench /mnt/vm_8tb/b70/lmx_overnight/w1_phase_20260820T0709Z.json
Result: patches applied. MTP 0.85->0.22 GB, LM
  head 2.54->0.66 GB. Graph 5s. G1 Paris/391.
  median post_first **65.08** (52.84-81.91)
  vs S1 47.58. Spec 546/984, mean accept 2.22.
  No DEVICE_LOST. Serve stopped after bench.
Verdict: GO. New W1 hold 65.08. Not W8A8.
  Board 112.65 still ceiling.
Changed beliefs: dense 3.8 GPTQ + draft-INT4
  GRAPH-boots on float16 (unlike Ornith L65
  Half!=BF16). Overlay is a real decode lift
  on this box (+36.8% vs S1), still short of
  SergiioB 112.7.
Next pick: P1 if GGUF >=25 GB else O2.
Do not: publish 65.08 as W8A8; demote 31.9
  or 34.9; start DD; P2P; emul NVFP4; 4x B70.
Restore: DD PARKED. Cards free. W1 container
  removed.
JOURNAL: ### 2026-08-20aw

## LOOP 3 -- 2026-08-20T0750Z -- P1b script + ignore-eos; GPU blocked

Picked: P1b Q8_DOORS=0 vs 1 A/B +
  longer g128. Could not attach: LOOP 2
  Q8 serve was stopped by foreign
  sweep_p2p_71_retest.sh.
Why this, not the other open row:
  operator-priority P1. W1 DONE. O2
  needs our cards.
GPU: both cards HELD by foreign
  qwen38_w8a8_p2p71 then p2p71b.
  DD PARKED. P2P=1 in vLLM TP=2.
Command:
  bash llamacpp/sweep_obliterated_q8_doors.sh
  (lease-busy exit 8; no second serve)
Log: /mnt/vm_8tb/b70/lmx_overnight/STATUS
  p2p71_serve_20260820T0732Z.log
Result: P2P71 GRAPH=0 TP=2 P2P=1 hung
  15 min at encoder-cache / shm_broadcast
  (workers ~180% CPU). NOT HEALTHY 900s.
  No DEVICE_LOST in dmesg. xpu-health
  post HEALTHY. Then a second P2P=1
  start (p2p71b PUSH_AR=0) chained --
  standing no. P1b sweep + phase_bench
  --ignore-eos landed. GGUF 27702 MB.
Verdict: BLOCKED (foreign P2P chain).
  No Q8 A/B number. Hold 30.89 / 32.14
  stays. Health recovered after first
  hang; 7.1 did NOT cure H.13 hang.
Changed beliefs: P2P-in-vLLM-TP2 on
  7.1 still hangs at warmup all_reduce
  / profiling. Do not chain a third
  P2P start. P1b must hold lease for
  the whole A/B (no nested gpu-run gap).
Next pick: P1b doors A/B when lease
  free + xpu-health GO.
Do not: steal p2p71b; third P2P; DD;
  COMM_DIRECT_Q8=3; demote 31.9/34.9.
Restore: DD PARKED. Q8 serve DOWN.
JOURNAL: ### 2026-08-20ay

## LOOP 4 -- 2026-08-20T0808Z -- 7.1 P2P retest: hang stays, wedge gone

Picked: operator "send P2P". Kernel 7.1.0.
  I_KNOW_P2P_WEDGES=1. OneCCL P2P=1 TP=2 then
  P2P=0 TP=2 smoke.
Why this, not the other open row: operator
  override of PRE.1. 7.1 GuC wedge was cured;
  H.13 oneCCL P2P-in-serve was never retested.
GPU: both cards. DD PARKED.
Command:
  torch.xpu.can_device_access_peer True/True
  P2PACCESS=1 GRAPH=0 TP=2 W8A8 (PUSH_AR on, then off)
  then P2PACCESS=0 PUSH_AR=1 TP=2 GRAPH=0 smoke
Log: p2p71_serve_20260820T0732Z.log
  p2p71b_serve_20260820T0748Z.log
  p2p71c_p2p0_20260820T0805Z.log
Result: P2P=1 hung 900s both arms (shm_broadcast
  after load). NOT DEVICE_LOST. Health stayed
  GO. P2P=0 TP=2 HEALTHY **147s**, G1 Paris/391
  GO. That follow-up would have wedged on 7.0.
Verdict: GO (wedge cured). NO-GO (P2P-in-serve
  still hangs). Default stays P2PACCESS=0.
Changed beliefs: 7.1 fixed the *aftershock*
  (matmul + later TP=2 P2P=0). It did not make
  CCL_TOPO_P2P_ACCESS=1 boot. PUSH_AR L0-IPC
  remains the TP=2 comm. Do not retry P2P=1.
Next pick: P1b Q8 doors A/B. Cards free.
Do not: third P2P; DD; COMM_DIRECT_Q8=3.
Restore: DD PARKED. xpu-health GO. Serves stopped.
JOURNAL: ### 2026-08-20az

## LOOP 5 -- 2026-08-20T0809Z -- P1b Q8_DOORS A/B started

Picked: P1b Q8_DOORS=1 then 0 ignore-eos
  g128 n=5. Cards free after LOOP 4
  P2P teardown. xpu-health GO.
Why this, not the other open row:
  operator-priority P1. GGUF 27702 MB.
  W1 DONE. P2P closed PRE.1b. Do not
  do O2 while P1b is live.
GPU: both cards HELD pid 260538
  qwen38_oblit_q8 :8010 Q8_DOORS=1
  loading. DD PARKED. P2PACCESS=0.
Command:
  bash llamacpp/sweep_obliterated_q8_doors.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1b_sweep_20260820T0809Z.log
  STATUS P1B_STATUS=START doors=1
Result: preflight HEALTHY. Container
  up, loading Q8_0 GGUF (blk.64 MTP
  tensors unused, same as LOOP 2).
  Two loads will span fires.
Verdict: RUNNING. No A/B number yet.
  Hold 30.89 / 32.14 stays until
  ignore-eos benches land.
Changed beliefs: none on Q8. LOOP 4
  stands: do not retry vLLM P2P=1.
Next pick: attach P1b; parse benches
  if done; leave Q8_DOORS=1 up.
Do not: second serve; retry P2P; DD;
  COMM_DIRECT_Q8=3; demote 31.9/34.9.
Restore: DD PARKED. Sweep left running.
JOURNAL: ### 2026-08-20ba

## LOOP 6 -- 2026-08-20T0838Z -- P1b Q8_DOORS A/B GO 31.78 vs 31.56

Picked: attach P1b. Sweep DONE. Parse
  ignore-eos g128 n=5 doors 1 vs 0.
Why this, not the other open row:
  NEXT PICK was attach. Serve is ours.
GPU: both cards HELD pid 260538
  docker-wait qwen38_oblit_q8 :8010
  Q8_DOORS=1. DD PARKED. P2PACCESS=0.
Command:
  bash llamacpp/sweep_obliterated_q8_doors.sh
  (started LOOP 5; attach+parse)
Log: /mnt/vm_8tb/b70/lmx_overnight/p1b_sweep_20260820T0809Z.log
  p1b_doors_20260820T0809Z.json
Result: G1 Paris/391 both arms.
  doors=1 median post_first **31.78**
  (31.70-31.99) prefill 570.5 TTFT 2.03s
  warmup 31.93. doors=0 **31.56**
  (31.37-31.61) prefill 567.6.
  All runs 128 tok ignore-eos. vs Q4_K_M
  43.8 = 0.726x / 0.721x (~1.38x slower,
  not 4x). SYCL: mmq_q8_reorder=1
  PRIORITIZE_DMMV=0 mmvq_eff q8_0=13.
  Live G1 still Paris.
Verdict: GO. Doors are not the gap
  (+0.22 tok/s, +0.7%). Hold moves to
  ignore-eos **31.78**. Leave serve up.
Changed beliefs: LOOP 2 30.89 was
  early-EOS; matched g128 is 31.78.
  Already on reorder MMVQ. Next is
  COMM_DIRECT then 1x vs 2x, then
  DP4A2/GDN-quad not MMQ-on.
Next pick: P1c COMM_DIRECT_Q8=2 vs 0.
Do not: second serve; COMM_DIRECT=3;
  retry vLLM P2P; DD; demote 31.9/34.9.
Restore: DD PARKED. Q8_DOORS=1 UP.
JOURNAL: ### 2026-08-20bb

## LOOP 7 -- 2026-08-20T0920Z -- P1c COMM_DIRECT 2 vs 0 GO 31.99 vs 31.83

Picked: P1c COMM_DIRECT_Q8=2 vs 0.
  Q8_DOORS=1 both. Never 3.
Why this, not the other open row:
  NEXT PICK after doors A/B. Same ckpt.
GPU: both cards HELD pid 266919
  qwen38_oblit_q8 :8010 restored
  COMM=2. DD PARKED. P2PACCESS=0.
Command:
  bash llamacpp/sweep_obliterated_q8_comm.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1c_sweep_20260820T0910Z.log
  p1c_comm_20260820T0910Z.json
Result: G1 Paris/391 both arms.
  COMM=2 median post_first **31.99**
  (31.80-32.06) prefill 569.5 TTFT 2.02s.
  COMM=0 **31.83** (31.66-31.97)
  prefill 568.3. vs 43.8 = 0.730x /
  0.727x. vs P1b doors=1 31.78: +0.7%.
Verdict: GO. COMM_DIRECT is not the
  43.8 gap (+0.16 tok/s, +0.5%).
  Hold 31.99. Leave COMM=2 up.
Changed beliefs: TP2 llama.cpp USM
  sum is not the decode limiter.
  Next is 1x vs 2x, then DP4A2/GDN-quad
  if 1x does not close 43.8.
Next pick: P1d GPU_COUNT=1 vs 2.
Do not: second serve; COMM=3; retry
  vLLM P2P; DD; demote 31.9/34.9.
Restore: DD PARKED. Q8 COMM=2 UP.
JOURNAL: ### 2026-08-20bc

## LOOP 8 -- 2026-08-20T0950Z -- P1d 2x vs 1x GO 32.03 vs 17.93

Picked: P1d GPU_COUNT=2 vs 1. Same
  Q8_0, Q8_DOORS=1 COMM=2. Always
  restore 2x.
Why this, not the other open row:
  NEXT PICK after COMM A/B.
GPU: both cards HELD pid 273052
  restored 2x qwen38_oblit_q8 :8010.
  DD PARKED. P2PACCESS=0.
Command:
  bash llamacpp/sweep_obliterated_q8_gpus.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/p1d_sweep_20260820T0940Z.log
  p1d_gpus_20260820T0940Z.json
Result: G1 Paris/391 both. 1x DID boot
  (29GB Q8 on 1x B70). ignore-eos g128:
  2x **32.03** (31.80-32.09) prefill 566.
  1x **17.93** (17.91-17.95) prefill 573.
  Scale 1.79x. vs 43.8 = 0.731x / 0.409x.
Verdict: GO. 1x is slower, not the 43.8
  path. TP2 compute split is real.
  Hold **32.03**. Leave 2x up.
Changed beliefs: decode gap vs Q4_K_M
  is per-card MMVQ, not comm and not
  missing a second GPU. Next is DP4A2
  / GDN-quad, not MMQ-on, not P2P.
Next pick: P1e Q8_0 MMVQ kernel notes
  + optional llama-bench after STOP.
Do not: second serve; COMM=3; retry
  vLLM P2P; DD; demote 31.9/34.9.
Restore: DD PARKED. Q8 2x UP.
JOURNAL: ### 2026-08-20bd

## LOOP 9 -- 2026-08-20T1024Z -- P1e SG32=1 30.11; DP4A2/SG24 absent

Picked: P1e in-image MMVQ geometry.
  Lab DP4A2 x QUAD_SG24 is the 36.8
  Q8 stack; 0xSero JIT .so is not that
  binary. A/B GGML_SYCL_MMVQ_SG32=1
  (only SG geometry symbol present).
Why this, not the other open row:
  NEXT PICK was kernel. Serve was ours.
GPU: both cards HELD pid 279160
  restored SG32=0 2x qwen38_oblit_q8.
  DD PARKED. P2PACCESS=0.
Command:
  bash llamacpp/sweep_obliterated_q8_sg32.sh
  docker exec sha256sum libggml-sycl.so
Log: /mnt/vm_8tb/b70/lmx_overnight/p1e_sweep_20260820T1017Z.log
  p1e_phase_s1_20260820T1017Z.json
Result: libggml-sycl.so 258f4729 != lab
  DP4A2xSG24 e75b9603. No QUAD_SG24 /
  DP4A2 strings. SG32=1 forced on both
  B70s (log: SG32 q8_0 ne1=1 kernel).
  G1 Paris/391. ignore-eos **30.11**
  (30.04-30.49) vs hold 32.03 (-6%).
  Prefill 572. vs 43.8 = 0.687x.
  Restored SG32=0. Our 32.03/43.8=0.731
  matches lab Q8/Q4 36.8/49.7=0.74.
Verdict: NO-GO SG32=1. DP4A2/SG24
  BLOCKED on 0xSero JIT / oneAPI
  2026.1.1 AOT (UR enum). Hold 32.03.
Changed beliefs: Q8 vs Q4_K_M gap is
  the quant, not a missing DMMV. The
  13% to lab Q8 36.8 is AOT DP4A2xSG24,
  not an env door. P1 in-image profile
  complete.
Next pick: O2 GRAPH-safe INT4. Leave
  Q8 up until that fire STOPs it.
Do not: FATTN_MMA=1 JIT; COMM=3;
  retry vLLM P2P; DD; demote 31.9/34.9.
Restore: DD PARKED. Q8 2x SG32=0 UP.
JOURNAL: ### 2026-08-20be

## LOOP 10 -- 2026-08-20T1047Z -- O2 GRAPH INT4 G1 GO; code c1 21.7

Picked: O2 GRAPH-safe INT4. Stopped
  Q8 (benches exist, different ckpt).
Why this, not the other open row:
  NEXT PICK. P1 in-image done.
GPU: card 0 HELD pid 285957 ornith_o2
  :18080. Card 1 free. DD PARKED.
  P2PACCESS=0. KV_FP8=0.
Command:
  python3 vllm/nvfp4/test_int4_graph_dtype.py
  GRAPH=1 MTPTOK=3 B70_DRAFT_MTP_INT4=1
  bash vllm/nvfp4/serve_nvfp4_moe_35b.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/o2_g1_20260820T1047Z.log
  o2_code_20260820T1047Z.txt
Result: unit XPUGraph PASS (raw fp16,
  cast bf16, capture OK). Stale L65
  eagle_head cache python_error; wipe
  then GRAPH capture 6s. G1 Paris/391.
  bench_code c1 **21.7** vs 34.9.
  Dense MTP 78->20 MB. Experts still
  BF16 Triton.
Verdict: GO boot (L65 dummy_run cured).
  NO-GO speed vs 34.9. Hold 34.9.
Changed beliefs: opaque custom op +
  cache wipe unblocks GRAPH INT4.
  Speed wait is O3 routed experts.
Next pick: O3 pack MTP routed experts
  INT4. Leave ornith_o2 up.
Do not: demote 34.9/31.9; emul NVFP4
  G1; P2P; DD; KVDTYPE=bfloat16.
Restore: DD PARKED. Q8 DOWN. O2 UP.
JOURNAL: ### 2026-08-20bf

## LOOP 11 -- 2026-08-20T1133Z -- O3 expert INT4 GRAPH G1 GO; c1 21.1

Picked: O3 pack MTP routed experts
  INT4 (L65 miss, 1.5 GB BF16 Triton).
Why this, not the other open row:
  NEXT PICK. Same Ornith ckpt.
GPU: card 0 HELD ornith_o3 :18080.
  Card 1 free. DD PARKED. P2P=0.
Command:
  pack routed_experts w13/w2 INT4 NT
  GRAPH=1 MTPTOK=3 B70_DRAFT_MTP_INT4=1
  bash vllm/nvfp4/serve_nvfp4_moe_35b.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/o3_g1_20260820T1133Z.log
  o3_code_20260820T1133Z.txt
Result: E=256 w13=(256,1024,2048)
  w2=(256,2048,512). Packed 1688->435 MB.
  NT .contiguous() was wrong; store
  [N,K/8] and pass .t() view. SharedExperts
  must not be called from apply. GRAPH
  capture 5s. G1 Paris/391. bench_code
  c1 **21.1** vs O2 21.7 vs hold **34.9**.
Verdict: GO pack+boot. NO-GO speed.
  Slot INT4 loops did not beat Triton
  experts. Hold 34.9.
Changed beliefs: MTP experts were the
  VRAM (1.6 GB) not the 34.9 decode
  limiter. Next is NVFP4 apply kernel
  (O4) or measure 34.9 as O1.
Next pick: O4 or O1. Leave ornith_o3 up.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1.
Restore: DD PARKED. Q8 DOWN. O3 UP.
JOURNAL: ### 2026-08-20bg

## LOOP 12 -- 2026-08-20T1146Z -- O4 M=1 1D GEMV proto PASS; 2.3x oneDNN

Picked: O4 ESIMD M=1 1D block_load NVFP4
  GEMV proto (llm-scaler #491). Card 1
  only. Leave ornith_o3 up.
Why this, not the other open row:
  NEXT PICK. Overnight serve is O3 not
  Q8; do not steal. O1 needs the 34.9
  no-MTP ckpt. Kernel work can span.
GPU: card 0 HELD ornith_o3 :18080.
  Card 1 proto + oneDNN + unit. DD PARKED.
  P2P=0.
Command:
  bash vllm/nvfp4/proto_moe_m1/build.sh
  gpu-run --card 1 proto_moe_m1/run.sh
  bench_onednn_m1.py + test_fused_moe_apply.py
Log: /mnt/vm_8tb/b70/lmx_overnight/o4_m1_run_20260820T114532Z.log
  o4_m1_onednn_unit_20260820T114532Z.log
Result: AOT BMG-G31, 53 lsc_load, 0 dpas.
  GEMV max_rel 1.6e-7. Apply 1.5e-3.
  Isolated vs oneDNN M=1:
  up 0.019 vs 0.043 ms (2.26x, 84 vs 31 GB/s)
  down 0.011 vs 0.041 ms (3.73x)
  grouped 8x up 0.110 vs 0.320 ms (2.91x)
  apply 0.176 ms. Unit XPUGraph PASS.
  o3 still Paris-id, Up. Hold 34.9.
Verdict: GO proto+numerics. NO-GO e2e.
  84 GB/s WG=1 << Intel 528. Not wired.
Changed beliefs: 1D load is real vs
  copy_from (+22%) and vs oneDNN M=1
  (~2.3x) on Ornith expert shapes. The
  34.9 gap is still launch/occupancy
  + wiring, not another Python copy.
Next pick: O4b occupancy/WG + torch op,
  or O1 34.9 phase_bench. Leave o3 up.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; swap live serve .so.
Restore: DD PARKED. Q8 DOWN. O3 UP.
JOURNAL: ### 2026-08-20bh

## LOOP 13 -- 2026-08-20T1212Z -- O4b WG/SLM occupancy NO-GO

Picked: O4b occupancy. WG=16/32 and SLM
  broadcast of x vs LOOP 12 WG=1.
  Card 1 only. Leave ornith_o3 up.
Why this, not the other open row:
  NEXT PICK after O4 proto. Overnight
  serve is O3 not Q8. O1 needs 34.9
  no-MTP ckpt.
GPU: card 0 HELD ornith_o3 :18080.
  Card 1 proto A/B. DD PARKED. P2P=0.
Command:
  bash vllm/nvfp4/proto_moe_m1/build.sh
  gpu-run --card 1 proto_moe_m1/run.sh
Log: /mnt/vm_8tb/b70/lmx_overnight/o4b_run_20260820T121149Z.log
Result: compile 68 lsc_load, 0 dpas.
  GEMV max_rel 1.6e-7 all arms.
  up wg1 0.014 ms 109 GB/s;
  wg16 0.986x; slm16 0.967x; slm32 0.922x.
  down slm16 0.427x. grouped slm16 0.978x
  (0.069 vs 0.067 ms, 184 vs 188 GB/s).
  o3 still Up. Hold 34.9.
Verdict: NO-GO occupancy. WG=1 1D load
  stays. Not wired. Dead-end packet O4b.
Changed beliefs: 84-vs-528 is not WG=1
  on Ornith expert shapes. Next speed
  step is a torch op (keep 2.3x vs
  oneDNN) or O1 measurement.
Next pick: O4c torch op or O1. Leave o3.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; swap live serve .so.
Restore: DD PARKED. Q8 DOWN. O3 UP.
JOURNAL: ### 2026-08-20bi

## LOOP 14 -- 2026-08-20T1243Z -- O4c sidecar torch op GO 2.36x

Picked: O4c torch XPU op for WG=1 1D
  NVFP4 GEMV. Sidecar .so, current
  stream. Leave ornith_o3 up.
Why this, not the other open row:
  NEXT PICK. o3 is not Q8. O1 needs
  34.9 no-MTP ckpt.
GPU: card 0 HELD ornith_o3 :18080.
  Card 1 compile+unit. DD PARKED. P2P=0.
Command:
  bash vllm/nvfp4/proto_moe_m1/build_op.sh
  gpu-run --card 1 test_m1_gemv_op.py
  test_fused_moe_apply.py (M1_KERNEL off)
Log: /mnt/vm_8tb/b70/lmx_overnight/o4c_test_20260820T124242Z.log
Result: .so 102 KB. vs oneDNN:
  up 0.018 vs 0.043 ms (2.36x) cos 0.999996
  rel 4.6e-3. down 0.015 vs 0.041 (2.67x).
  XPUGraph capture OK, replay cos 1.0.
  Unit apply PASS default OFF. o3 Up.
  Wired env B70_NVFP4_MOE_M1_KERNEL +
  B70_NVFP4_M1_SO, default OFF.
Verdict: GO op+graph. e2e not measured.
  Hold 34.9. Do not swap live _xpu_C.
Changed beliefs: sidecar op keeps the
  isolated 2.3x without a GDN rebuild.
  Next is a serve restart with M1_KERNEL
  or O1 34.9 phase_bench.
Next pick: O4d e2e or O1. Leave o3 up.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; swap live serve .so.
Restore: DD PARKED. Q8 DOWN. O3 UP.
JOURNAL: ### 2026-08-20bj

## LOOP 15 -- 2026-08-20T1315Z -- O4d e2e M1_KERNEL 32.2 NO-GO

Picked: O4d e2e GRAPH no-MTP + sidecar
  M1_KERNEL=1 vs hold 34.9. Stopped o3
  (21.1 exists, same ckpt).
Why this, not the other open row:
  NEXT PICK. P1 in-image done. O1 needs
  the 34.9 recipe (M1_KERNEL off).
GPU: card 0 HELD ornith_o4d :18080.
  Card 1 free. DD PARKED. P2P=0.
Command:
  docker rm -f ornith_o3
  gpu-run --card 0 bash vllm/nvfp4/sweep_ornith_o4d.sh
  g1_probe + bench_code c1 256 n=3
Log: /mnt/vm_8tb/b70/lmx_overnight/o4d_g1_20260820T131312Z.log
  o4d_code_20260820T131312Z.txt
Result: m1k=1. m1_gemv N=1024 K=2048
  at GRAPH capture 4s. G1 Paris/391.
  bench_code c1 **32.2** (best 32.3) vs
  hold **34.9** (0.92x) vs L64 33.3.
Verdict: Wire GO. Speed NO-GO. Hold
  34.9. M1_KERNEL default OFF. Dead-end
  packet O4d. Leave o4d up.
Changed beliefs: isolated 2.36x vs
  oneDNN is not e2e GRAPH decode. Slot
  loop launches dominate. Next speed
  is fused layerlet, not another env.
Next pick: O1 34.9 phase_bench (restart
  hold recipe, M1_KERNEL off).
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; leave M1_KERNEL on.
Restore: DD PARKED. Q8 DOWN. O4d UP.
JOURNAL: ### 2026-08-20bk

## LOOP 16 -- 2026-08-20T1346Z -- O1 hold phase_bench 45.56

Picked: O1 LocalMaxxing C1 of the 34.9
  GRAPH no-MTP hold. Stopped o4d
  (32.2 exists). M1_KERNEL off.
Why this, not the other open row:
  NEXT PICK. P1 in-image done. O4d
  speed closed. Measurement not a
  kernel unlock.
GPU: card 0 HELD ornith_o1 :18080.
  Card 1 free. DD PARKED. P2P=0.
Command:
  docker rm -f ornith_o4d
  gpu-run --card 0 bash vllm/nvfp4/sweep_ornith_o1.sh
  phase_bench p512/g128 n=5 + --ignore-eos
  bench_code c1 256 n=3
Log: /mnt/vm_8tb/b70/lmx_overnight/o1_g1_20260820T134005Z.log
  o1_phase_ieos_20260820T134005Z.json
  o1_code_20260820T134005Z.txt
Result: m1k=0. GRAPH 5s. G1 Paris/391.
  phase_bench **45.57** (2/5 short EOS).
  ignore-eos g128 n=5/5 **45.56**.
  Prefill proxy 265. TTFT 4.34s.
  bench_code c1 **34.8** vs hold 34.9.
Verdict: GO. Hold 34.9 recovered. O1
  C1 is 45.56 post-first, not 34.9.
Changed beliefs: Ornith GRAPH no-MTP
  submit-shaped decode is ~45.6, below
  S1 47.58 / W1 65.08 (different ckpt).
  Prefill proxy 265 is the MoE prefill
  gap vs W1 1695.
Next pick: O4e fused layerlet (card 1)
  or W8. Leave o1 up.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; treat 45.56 as 34.9.
Restore: DD PARKED. Q8 DOWN. O1 UP.
JOURNAL: ### 2026-08-20bl

## LOOP 17 -- 2026-08-20T1414Z -- O4e fused layerlet 1.04x NO-GO

Picked: O4e fused up+silu+down one
  launch layerlet. Card 1 only.
  Leave ornith_o1 up.
Why this, not the other open row:
  NEXT PICK. o1 is not Q8. O4d retry-if
  was this layerlet. W8 waits.
GPU: card 0 HELD ornith_o1 :18080.
  Card 1 compile+run. DD PARKED. P2P=0.
Command:
  bash vllm/nvfp4/proto_moe_m1/build_layerlet.sh
  gpu-run --card 1 proto_moe_m1/run_layerlet.sh
  gpu-run --card 1 bench_onednn_layerlet.py
Log: /mnt/vm_8tb/b70/lmx_overnight/o4e_run_20260820T141353Z.log
  o4e_onednn_*.log
Result: WG=1024 abort (ESIMD WG<=64).
  WG=64 PASS rel 1.4e-3. fused 0.133 ms
  vs seq 0.137 (1.036x) vs eager oneDNN
  1.159 (8.71x). 21 lsc_load, 0 dpas.
  o1 still Up. Hold 34.9.
Verdict: Proto GO. Speed NO-GO vs slot
  GEMV. Dead-end packet O4e. Not wired.
Changed beliefs: extra launches are not
  the GRAPH 34.9 gap. ESIMD WG=64 makes
  fused N=1024 serial. 8.7x eager oneDNN
  still loses e2e like O4d 2.36x.
Next pick: W8 k1bar 31.9. Leave o1 up
  until that fire STOPs it.
Do not: demote 34.9/31.9; P2P; DD;
  emul NVFP4 G1; wire layerlet e2e.
Restore: DD PARKED. Q8 DOWN. O1 UP.
JOURNAL: ### 2026-08-20bm

## LOOP 18 -- 2026-08-20T1446Z -- W8 k1bar 28.8; hold 31.9 stays

Picked: W8 remeasure k1bar W8A8-gptq
  DSpark TP=2. Stopped o1 (benches
  exist, different ckpt).
Why this, not the other open row:
  NEXT PICK. P1 in-image done. O*
  slot/layerlet closed.
GPU: both cards HELD qwen38_w8a8_dspark
  :18080. DD PARKED. P2P=0.
Command:
  docker rm -f ornith_o1
  gpu-run bash vllm/dflash/sweep_lmx_w8_k1bar.sh
  g1_probe + bench_code + phase_bench
Log: /mnt/vm_8tb/b70/lmx_overnight/w8_g1_20260820T144133Z.log
  w8_code_20260820T144133Z.txt
  w8_phase_20260820T144133Z.json
Result: HEALTHY 213s. BARRIER rank0+1.
  AGASYNC ENGAGED. G1 Paris/391.
  bench_code c1 **28.8** best 31.5 vs
  hold **31.9** / 34.0. phase_bench
  **26.85** prefill 2321 TTFT 0.49s.
Verdict: Serve/G1 GO. vs 31.9 NO-GO.
  Do not demote. PREFIXCACHE=0 turned
  on MRV2 (hold was likely cache on).
Changed beliefs: k1bar submit-shaped
  C1 is ~27, not 31.9 bench_code.
  Prefill 2321 >> Ornith 265.
Next pick: W8b PREFIXCACHE=1 MRV2=0
  rematch, or park. Leave k1bar up.
Do not: demote 31.9/34.9; P2P; DD;
  emul NVFP4 G1.
Restore: DD PARKED. Q8 DOWN. O1 DOWN.
  W8 UP.
JOURNAL: ### 2026-08-20bn

## LOOP 19 -- 2026-08-20T1514Z -- W8b pc1 28.1; hold 31.9 stays

Picked: W8b PREFIXCACHE=1 B70_MRV2=0
  rematch of k1bar 31.9. Same ckpt.
Why this, not the other open row:
  NEXT PICK. P1 in-image done. O*
  closed. Isolate LOOP 18 MRV2.
GPU: both cards HELD qwen38_w8a8_dspark
  :18080. DD PARKED. P2P=0.
Command:
  gpu-run bash vllm/dflash/sweep_lmx_w8b_k1bar.sh
  g1_probe + bench_code + phase_bench
Log: /mnt/vm_8tb/b70/lmx_overnight/w8b_g1_20260820T150951Z.log
  w8b_code_20260820T150951Z.txt
  w8b_phase_20260820T150951Z.json
Result: HEALTHY 218s. prefix-caching ON.
  no V2 runner. BARRIER rank0+1. G1
  Paris/391. bench_code c1 **28.1**
  best 30.7 vs hold **31.9** / L18 28.8.
  phase_bench **28.08**.
Verdict: Isolation GO. vs 31.9 NO-GO.
  Do not demote. Park.
Changed beliefs: PREFIXCACHE/MRV2 is
  not the 31.9 miss. Current k1bar on
  this box is ~28 code / ~28 phase.
Next pick: park. Leave k1bar-pc1 up.
Do not: demote 31.9/34.9; P2P; DD.
Restore: DD PARKED. Q8 DOWN. W8b UP.
JOURNAL: ### 2026-08-20bo

## LOOP 20 -- 2026-08-20T1540Z -- attach k1bar-pc1 soak 24.8

Picked: attach live k1bar-pc1. G1 +
  bench_code soak. No second serve.
Why this, not the other open row:
  NEXT PICK park. Cards held. P1 Q8
  GGUF complete but lease busy. G1
  already GO; recheck after 29 min.
GPU: both cards HELD qwen38_w8a8_dspark
  :18080. DD PARKED. P2P=0.
Command:
  g1_probe + bench_code c1 256 n=3
Log: /mnt/vm_8tb/b70/lmx_overnight/w8c_g1_20260820T153930Z.log
  w8c_code_20260820T153930Z.txt
Result: G1 Paris/391. bench_code c1
  **24.8** best 25.9 vs W8b 28.1 vs
  hold 31.9. Leave up.
Verdict: Attach GO. Soak slower. Do
  not demote 31.9.
Changed beliefs: k1bar current-box
  decode is ~25-28, not a recovered
  31.9. Hold stays historical 31.9.
Next pick: park. Leave k1bar-pc1 up.
Do not: demote 31.9/34.9; P2P; DD;
  start Q8 while lease held.
Restore: DD PARKED. Q8 DOWN. W8b UP.
JOURNAL: ### 2026-08-20bp

## LOOP 21 -- 2026-08-20T1609Z -- attach k1bar-pc1 59m G1 GO

Picked: attach live k1bar-pc1. G1
  only. Parse logs. No second serve.
Why this, not the other open row:
  NEXT PICK park. Lease busy. P1 Q8
  GGUF complete but cards held.
GPU: both cards HELD qwen38_w8a8_dspark
  :18080. DD PARKED. P2P=0.
Command:
  g1_probe + docker logs scan
Log: /mnt/vm_8tb/b70/lmx_overnight/w8d_g1_20260820T160924Z.log
Result: G1 Paris/391. No DEVICE_LOST.
  Serve Up 59 min.
Verdict: Attach GO. Leave up. Park.
Changed beliefs: none. 59m soak still
  coherent. Hold 31.9 stays.
Next pick: park. Leave k1bar-pc1 up.
Do not: demote 31.9/34.9; P2P; DD;
  start Q8 while lease held.
Restore: DD PARKED. Q8 DOWN. W8b UP.
JOURNAL: ### 2026-08-20bq

## LOOP 22 -- 2026-08-20T1639Z -- attach k1bar-pc1 ~90m G1 GO

Picked: attach live k1bar-pc1. G1
  only. No second serve. No bench.
Why this, not the other open row:
  NEXT PICK park. Lease busy. P1 Q8
  GGUF complete but cards held.
GPU: both cards HELD qwen38_w8a8_dspark
  :18080. DD PARKED. P2P=0.
Command:
  g1_probe
Log: /mnt/vm_8tb/b70/lmx_overnight/w8e_g1_20260820T163919Z.log
Result: G1 Paris/391. No DEVICE_LOST.
  Serve Up ~90 min.
Verdict: Attach GO. Leave up. Park.
Changed beliefs: none. Hold 31.9 stays.
Next pick: park. Leave k1bar-pc1 up.
Do not: demote 31.9/34.9; P2P; DD;
  start Q8 while lease held.
Restore: DD PARKED. Q8 DOWN. W8b UP.
JOURNAL: ### 2026-08-20br
