# SGLang Qwen3.6-27B (qwen3_5 GDN) on 2x Arc B70 -- perf optimization campaign

## ============ FINAL HONEST SCOREBOARD (warm steady-state, IN=2048 OUT=128, 2026-06-27) ============
ALL numbers warm (discard the 1st bench after idle -- the B70 idle-downclocks; cold runs ~2x slow & misled earlier).
| driver                         | c1 decode t/s | c4 decode (agg) | TTFT ms | cards | vision | wedge-proof | quant |
|--------------------------------|---------------|-----------------|---------|-------|--------|-------------|-------|
| bf16 TP=2 (serve_sglang.sh)    | ~9.2          | 6.06/str (23.4) | ~660    | 2     | YES    | no (TP wedge risk) | bf16 |
| woq int4 TP=1 (sglang-xpu:woq) | ~9.44         | (single-card)   | ~920    | 1     | YES    | YES (single-card)  | int4 woqgemm |
| woq int4 + XPU CUDAGRAPH       | ~7.6 e2e (DEGRADES) | -         | regresses | 1   | YES    | -           | int4 woqgemm |
*** CORRECTION (do NOT trust the earlier "2.5x breakthrough" -- I over-claimed it from sglang's INTERNAL decode-log).
  XPU cudagraph (B70_XPU_CUDAGRAPH=1 + triton attn + model_runner xpu patch) genuinely RUNS -- graph capture works on
  this hybrid GDN model (a real engineering first; torch.xpu.XPUGraph + the GDN init_cuda_graph_state carry it). BUT
  the actual END-TO-END (bench2048) is ~7.6 t/s -- WORSE than eager 9.4 -- because it has the SAME torch-xpu graph-
  replay COMMAND-STREAM ACCUMULATION the journal already root-caused as a DEAD END (vLLM PIECEWISE: 26->7 over a soak;
  needs an upstream torch-xpu/L0 fix; recapture crashes; env knobs don't help). Observed: decode starts ~23.6 then
  DEGRADES to ~9.4 + severe stalls (1.3-2.7 t/s intervals), and the (ungraphed, triton-attn) PREFILL ALSO degrades
  (3098 -> 25-309 t/s = catastrophic TTFT). So it's CORRECT but NOT a usable/stable speedup. Kept opt-in + OFF by default.
TAKEAWAY: the practical STABLE ceiling for sglang-XPU serving is ~9.2-9.4 t/s (eager). cudagraph does NOT beat it stably
(torch-xpu graph dead-end, same as vLLM). The campaign's DURABLE wins: (a) CORRECTNESS (no GDN NaN; vLLM has it), (b)
auto_round_kernel.woqgemm wired into sglang, (c) int4 single-card == bf16-TP2 warm -> woq int4 DP=2 (wedge-proof+vision)
driver, (d) MTP characterized (works to the tree kernels), (e) XPU cudagraph WIRED (works, but hits the known torch-xpu
degradation -> awaits an upstream fix).
DAILY DRIVER PICK: woq int4 DP=2 (serve_dp2.sh) for UNATTENDED (wedge-proof+vision, ~9.4/replica) OR bf16 TP=2 for best
c4 aggregate (23.4). ALL CORRECT. Did NOT beat the eager ceiling stably: 4-bit-without-graph, GDN num_warps (cold
artifact), PP=2 (hangs), TP=2-woq (hangs), MTP (needs tree-kernel reimpl), cudagraph (torch-xpu accumulation dead-end).
## ====================================================================================================


Goal: make the *correct* SGLang serve (no GDN NaN, unlike vLLM) FAST enough to be a daily driver.
Bench = `sglang/bench2048.sh` (random IN=2048 OUT=128, ignore-eos; matches vLLM scripts/121).
Columns: decode_tps = 1000/TPOT (single-stream TG); prefill_tps (PP) = IN*1000/TTFT; TTFT ms.

## Reference points (from JOURNAL / prior vLLM work)
- vLLM int4 daily driver (single-card, GRAPH): ~30.8 tok/s decode -- but GDN-NaN-prone ("!!!!").
- The whole point of SGLang: CORRECT under mixed prefill+decode. We trade some speed for that;
  the campaign goal is to claw the speed back while staying correct.

## Results table (config -> result)

| # | config                              | conc | decode_tps | TTFT ms | prefill_tps | notes |
|---|-------------------------------------|------|-----------|---------|-------------|-------|
| 0 | bf16 TP=2 (baseline, CTX8192 MF.93) | 1    | 9.03      | 661     | 3098        | out_tok 8.51; the documented baseline |
| 0 | bf16 TP=2 (baseline)                | 4    | 8.18      | 974     | 2103        | aggregate out 24.92 tok/s (4 streams) |
| A | bf16 TP=2 +FLA_FAST +numdecode2     | 1    | 9.34      | 599     | 3419        | coherent OK; TTFT -9% prefill +10% (FLA_FAST helps) |
| A | bf16 TP=2 +FLA_FAST +numdecode2     | 4    | 5.12      | 983     | 2084        | c4 REGRESSED (numdecode2 hurts concurrency) |
| B | bf16 PP=2 (--tp 1 --pp-size 2)      | -    | BROKEN    | -       | -           | /health 200 but ALL gen requests time out -> 500 (scheduler deadlock on GDN); NO-GO |

## fp16 is BROKEN on XPU for this model -> use bf16-AWQ patch instead (2026-06-27)
Serving the unquantized bf16 model as `--dtype float16` LOADS fine but CRASHES on the first forward:
  causal_conv1d Triton kernel CompilationError: "Mismatched type for col0 between then block (bf16) and
  else block (fp16)" (gdn_backend -> causal_conv1d_triton.py:~105). The GDN conv_state cache is bf16 while
  fp16 acts make x fp16 -> the kernel's branch dtypes diverge -> Triton compile fails. (PP=2 hung similarly.)
SINCE AWQ pins act dtype fp16 (awq.py:104), this would block AWQ. FIX = keep the model in the VALIDATED bf16
GDN regime and patch AWQ to run bf16 (sglang/patches/): (1) awq.py get_supported_act_dtypes += bfloat16;
(2) awq_kernels.py apply casts awq_dequantize output (fp16) -> x.dtype before matmul. Weight stays 4-bit
(decode-bandwidth win preserved); GDN/conv never see fp16 -> no crash, inherits proven bf16 coherence.
Serve via serve_sglang.sh MOUNTS=(overlay both patched files) QUANT=awq EXTRA="--dtype bfloat16".

## CRITICAL FINDING: 4-bit does NOT speed up decode on Battlemage (bf16 XMX wins) [2026-06-27]
Measured on card 0 (down_proj GEMV, in=17408 out=5120, M=1 decode), repacked AWQ vs bf16:
  per-GEMV ms:  nonfused(awq_dequantize+matmul)=1.29   fused(awq_gemm_triton sk8)=0.67   bf16 matmul=0.30
The bf16-AWQ patch WORKS (awq_dequantize finite+correct on XPU, bf16 cast/matmul fine), BUT every 4-bit path
is SLOWER than bf16: non-fused materializes the full fp16 weight (extra VRAM write+read); the FUSED Triton
awq_gemm kernel (tl.dot, no materialization) is still 2.2x slower than bf16 -- Intel's oneDNN bf16 XMX GEMM is
extremely optimized and the generic Triton 4-bit kernel can't realize the 4x bandwidth saving at M=1.
=> On these B70s, QUANT IS A MEMORY LEVER, NOT A SPEED LEVER. AWQ only helps by fitting one card (DP=2 aggregate).
   A true decode speedup would need a hand-tuned XMX/DPAS 4-bit GEMV (SYCL) -- frontier, uncertain payoff. Parked.
Tools: sglang/awq_kernel_probe.py, sglang/awq_fused_probe.py, sglang/patches/{awq.py,awq_kernels.py} (bf16-AWQ, works).

CORRECTION: full bf16 27B VLM is ~55GB -> needs TP=2 (does NOT fit one card; mem usage 25.6 GiB was PER-CARD at
TP=2). So bf16 DP=2 is impossible. Only a <=~28GB (4-bit) model fits one card for DP=2/single-card.

## THE REAL PATH: auto_round_kernel.woqgemm -- the proven-fast XPU int4 GEMM [2026-06-27]
vLLM on THIS box hit ~30 t/s single-card int4 (journal scripts/121: "int4 NONE TP1 23.3 / captured 30.5") = 3x
the sglang bf16 TP=2 (9 t/s). vLLM's speed came from a REAL XPU int4 GEMM: `auto_round_kernel` (auto-round-lib
0.13.3, in vllm-xpu-env:v0230) exposes `woqgemm` / `woq_linear` / `woqgemm_s8` with an `xpu_lib` backend. The
Triton/AWQ kernels lose to bf16 XMX; woqgemm (the auto-round serving kernel) does NOT (proven 30 t/s).
CAMPAIGN REFRAME:
  vLLM int4  = FAST (30 t/s, woqgemm) but BROKEN (GDN NaN "!!!!")
  sglang bf16= CORRECT (no NaN) but SLOW (9 t/s, no fast 4-bit GEMM)
  TARGET     = sglang + woqgemm int4 = CORRECT *and* FAST (+ vision via requant)
PLAN: (1) microbench woqgemm vs bf16 in isolation (confirm 4x-ish decode win on the AutoRound int4 ckpt);
  (2) wire auto_round_kernel into sglang as a quant method (install auto-round-lib into sglang image; patch a
  WoqLinearMethod that calls woq_linear on the AutoRound int4 layers); (3) single-card -> DP=2; (4) vision via
  AutoRound requant. Input ckpt: Lorbus_Qwen3.6-27B-int4-AutoRound (arch Qwen3_5ForConditionalGeneration).

## CONFIRMED: auto_round_kernel WOQ int4 = 2.17x decode vs bf16 (sglang/woq_probe.py) [2026-06-27]
Microbenched auto_round_kernel.QuantLinearGPTQ on a real Lorbus int4 down_proj layer (in=17408 out=5120):
  per-GEMV (M=1, DECODE): woq(int4 wt)=0.141 ms   bf16=0.305 ms   => 2.17x FASTER, output finite.
  per-GEMM (M=512, prefill-ish): woq=1.362  bf16=0.767  => 0.56x (woq SLOWER -- compute-bound; decode-only win).
The kernel: QuantLinear.forward -> ark.woqgemm (int4 weight, fp16 compute -- "XMX int8 not supported on B70 with
oneAPI < 2026, fell back to fp16"; an oneAPI>=2026 upgrade would enable int8 XMX -> likely even faster). UNLIKE the
Triton AWQ kernel, woqgemm REALIZES the 4-bit bandwidth saving at M=1. This is vLLM's proven-30-t/s kernel.
=> THE PLAN: wire auto_round_kernel into sglang as an XPU quant method for the AutoRound int4 ckpt:
   (1) install auto-round-lib into sglang-xpu:bmg (verify .so loads vs sglang's torch 2.12+xpu);
   (2) custom WoqLinearMethod (create_weights qweight/qzeros/scales -> post_init repack -> apply woqgemm),
       routing only the quantized Linears (GDN/vision stay bf16 via the ckpt ignore list);
   (3) serve single-card (no all-reduce) -> bench decode (target ~20-30 t/s, correct via sglang GDN fix) -> DP=2;
   (4) vision via AutoRound requant of the full VLM.
Tradeoff noted: prefill slower; decode (the daily-driver bottleneck) much faster. Hybrid prefill=bf16/decode=woq
is a later refinement if TTFT matters.
INTEGRATION FEASIBLE (verified 2026-06-27): `pip install auto-round-lib` works in sglang-xpu:bmg (torch 2.12+xpu),
the XPU .so loads + runs on the B70 -> woq_probe in the sglang image = 0.136ms vs bf16 0.314ms = 2.31x decode,
finite. So fast(woqgemm) + correct(sglang GDN fix) + installable all hold. Remaining = the sglang quant-method
plumbing only.
NEXT (integration steps): study sglang/patches/awq.py (model quant method) -> write AutoRoundWoqConfig +
WoqLinearMethod (create_weights: qweight[in//8,out]/qzeros[in//g,out//8]/scales[in//g,out] buffers;
process_weights_after_loading: ark.repack_quantized_weight; apply: ark.woqgemm). Register quant_method
"auto-round" in sglang/srt/layers/quantization/__init__.py. Exclude GDN/visual/lm_head (ckpt ignore list).
Bake auto-round-lib into a derived image OR pip-install at serve start. Serve Lorbus int4 TP=1 -> bench + coherence.

## WOQ INTEGRATION WORKS (sglang serves int4 via woqgemm, COHERENT) [2026-06-27]
Built sglang-xpu:woq (= sglang-xpu:bmg + auto-round-lib + sglang/patches/woq_shim.py auto-imported via .pth).
The shim patches GPTQLinearScheme._init_kernel -> XPU WOQ kernel (auto_round_kernel.QuantLinearGPTQ/woqgemm),
and guards AutoRound's check_marlin_supported(device_capability=None) crash on XPU. Served Lorbus int4 TP=1:
  - 304 WOQ layers built (incl. fused qkv/gate_up), model loaded int4 (quant=auto-round bits=4), mem 17.34 GiB
    on ONE card (14.5 GiB free -> DP=2 viable), KV 99200 tok.
  - COHERENT: "why is the sky blue" -> correct Rayleigh-scattering answer (NOT garbage/!!!!). sglang GDN fix holds.
  | config            | conc | decode_tps | TTFT ms | prefill_tps |
  | woq int4 TP=1     | 1    | 4.68       | 1223    | 1674        | <- SLOWER than bf16 TP=2 (9.03)!
WHY SLOWER: TP=1 puts ALL the unquantized bf16 GDN (10.4 GiB, 43% of model) + lm_head on ONE card; that
weight bandwidth dominates and outweighs the faster int4 GEMVs. bf16 TP=2 splits GDN across 2 cards (13 GiB/card
parallel) vs woq TP=1's 15 GiB on one card. The woqgemm kernel IS faster per-GEMV (2.2x microbench) -- the loss
is the TP=1 topology, not the kernel.
NEXT: woq + TP=2 -- fast int4 GEMVs AND GDN split across both cards (~7.5 GiB/card) -> should beat bf16 TP=2.
Then: (a) if TP=2 wins -> ship; (b) quantize GDN too (note-1 "pack further") for fast single-card + DP=2 (correctness
risk -> validate); (c) graph capture is unavailable on XPU (eager overhead is a fixed tax on both bf16 and woq).

## woq TP=2 HANGS + the real bottleneck is EAGER OVERHEAD, not bandwidth [2026-06-27]
woq int4 TP=2 LOADS (8.7 GiB/card, KV 345408 tok = 4.8x bf16's!) but generation HANGS: a TP worker dies in
the first forward (gloo "Connection closed by peer") -> the recurring TP>1 forward fragility, but woq-specific
(bf16 TP=2 works). woqgemm x oneCCL-all-reduce interaction (RowParallel down_proj/o_proj) is the suspect.
KEY REFRAME (from the woq numbers): sglang-XPU decode is EAGER-OVERHEAD / TOPOLOGY bound, NOT weight-bandwidth:
  - woq TP=1 4.68 t/s (213ms/tok) but only ~33ms is weight bandwidth -> ~180ms is eager Python dispatch + slow
    GDN triton + per-op overhead over ~256 layers. Quantization (fewer bytes) can't fix an overhead-bound decode.
  - bf16 TP=2 9 t/s is ALSO eager-bound; ~9 t/s looks near sglang-XPU's eager ceiling for this 27B GDN model.
  - vLLM hit 30 t/s via CUDAGRAPH (captured 30.5 vs NONE 23.3) -- sglang-XPU has NO graph capture (dead end) and
    torch.compile is a no-op on XPU. So we CANNOT close the gap to vLLM via quant alone.
woq's real value = fits one card -> DP=2 (wedge-proof, +vision, big KV) but single-stream stays ~4.7 t/s. Not a
single-stream win over bf16 TP=2.

## THE lever that beats overhead-bound decode: MTP / speculative decode
MTP generates K tokens per forward -> amortizes the per-forward eager overhead (the actual bottleneck). vLLM's
w8a8+MTP hit 25.6 t/s (MTP ~doubled decode). Risk: sglang MTP is XPU-gated (EAGLE has no intel_xpu draft-attn +
no xpu graph-runner). Agent's possible escape: --speculative-algorithm NEXTN --speculative-draft-attention-backend
triton --speculative-num-steps 1. Qwen3.6 ships an MTP head (qwen3_5_mtp); mtp-graft ckpts exist. NEXT: try MTP on
the bf16 serve (most likely to amortize the eager overhead -> real single-stream win, keeps vision).
Secondary: debug the woq TP=2 hang (-> int4 TP=2 huge-KV) and woq DP=2 (wedge-proof daily driver).

## MTP / NEXTN spec-decode WORKS ON XPU (gates defeated) [2026-06-27]
Goal (user-chosen): MTP is the one lever that beats the eager/GDN-bound decode (runs the slow forward once per
K tokens). Built sglang/graft_mtp.py -> grafted the 15 BF16 mtp.* head weights from W4A16-mtp-graft onto Lorbus
int4 (which already has int4-LM + VISION) -> /models/Lorbus_Qwen3.6-27B-int4-mtp (symlink LM shards + 1 mtp shard
+ num_nextn_predict_layers=1). Served via woqgemm (sglang-xpu:woq) with the agent's escape flags:
  --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 2 --speculative-draft-attention-backend triton
RESULT: BOTH models loaded on XPU -- main (Qwen3_5ForConditionalGeneration int4 17.3 GiB) AND the draft
(Qwen3_5ForCausalLMMTP int4 5.0 GiB). The XPU spec-decode gates (intel_xpu draft-attn, xpu graph-runner) were
DEFEATED by num-steps=1 + draft-attn=triton -- no "EAGLE not supported" crash. So MTP on sglang-XPU is VIABLE.
First attempt OOM'd (draft+main=22.3 GiB leaves too little KV at mem-fraction 0.9) -> retrying MEMFRAC=0.95 CTX=4096.
Pending: coherence + accept rate + does MTP actually speed decode past the 4.68 woq-no-MTP baseline.
NEXT: if MTP accepts + speeds decode -> the fast+correct+vision daily driver. Then kernel optimization (GDN triton).

MTP XPU-gate whack-a-mole (each patched, in order encountered):
  1. draft-attn intel_xpu gate -> --speculative-draft-attention-backend triton (DODGED).
  2. multi-step graph-runner gate -> --speculative-num-steps 1 (DODGED).
  3. OOM (draft+main 22.3GiB single-card) -> --max-running-requests 4 (shrinks mamba ssm cache; FIXED).
  4. spec mamba state cache device="cuda" HARDCODED (memory_pool.py:472/514/549) -> patched device=device
     (sglang/patches/memory_pool.py, mounted). Got "intermediate_ssm_state_cache 1.41GB" allocated on XPU.
  5. torch.cuda.synchronize() in qwen3_5_mtp.py:136 set_embed_and_head + torch.cuda.{Stream,Event} in spec path
     -> woq_shim redirects torch.cuda.{synchronize,Stream,Event,current_stream,empty_cache} -> torch.xpu. Also
     pass --disable-cuda-graph (skip the draft cuda-graph capture, unavailable on XPU).
  6. spec-decode warmup forward HUNG (600s) at first -- but --skip-server-warmup + long first request showed the
     forward actually RUNS in 13s (cold JIT, NOT a hang). So the GDN draft/verify path is NOT a wall (refutes the
     research agent's prediction). The 600s "hang" was just the cold triton JIT exceeding the warmup timeout.
  7. THE REMAINING BLOCKER = 2 MISSING C++ kernels with no XPU build: `sgl_build_tree_kernel_efficient`
     (eagle_utils.py:217, NameError) + `verify_tree_greedy` (sgl_kernel, _is_cuda-gated). NPU has torch.ops.npu
     equivalents; XPU has neither. For our degenerate config (topk=1 steps=1 -> a 2-token CHAIN, not a tree) both
     reduce to simple tensor ops -> implementable in pure PyTorch and injectable via the shim.
STATUS: MTP/spec-decode on sglang-XPU is ~90% there -- loads + forward runs (13s) + 6 gates patched; blocked only
  on 2 tree kernels (build_tree + verify_greedy) that need a Python/XPU reimpl for the chain case. This IS the
  "kernel work" the user wants next. Expected payoff: num-steps=1 -> ~1.5-1.6x (accept~0.6) -> int4 ~4.68 -> ~7.5
  t/s single-stream (below bf16 TP2's 9) BUT single-card -> wedge-proof DP=2 ~15 aggregate. Worth completing.
NEXT: implement build_tree_kernel_efficient + verify_tree_greedy (chain/topk=1) in pure torch, inject via woq_shim.

MTP CLOSEOUT (2026-06-27, operator chose to pivot): the 2 tree ops are UNREGISTERED XPU wrappers (no .so,
torch.ops.sgl_kernel.{build_tree_kernel_efficient,verify_tree_greedy}=registered False) -> need a full pure-torch
reimpl. Operator decision: NOT worth it -- realistic payoff ~7.5 t/s single-stream is BELOW bf16 TP2's 9 (num-steps=1
drafts only 1 tok/forward; more needs the gated multi-step graph runner). MTP is LEFT WORKING-UP-TO-TREE-KERNELS
(graft + 6 gate patches all committed; reproducible via graft_mtp.py + the woq_shim cuda-redirect + memory_pool patch
+ the NEXTN escape flags). PIVOT -> optimize the GDN/linear-attn Triton kernel (the real decode bottleneck capping
bf16's 9 t/s AND every other config). The GDN compute (48 linear-attn layers' recurrent scan, triton FLA) is what
makes single-card decode slow; vLLM is faster because it has a compiled SYCL GDN kernel.

## [CORRECTION 2026-06-27] The "GDN num_warps win" was a COLD-BENCH ARTIFACT -- it does NOTHING warm.
MEASUREMENT BUG: the B70 GPU downclocks when idle, so the FIRST bench after idle runs ~2x slow (clock ramp + cold
triton JIT); the warm steady-state is reached by the 2nd bench. My single-bench numbers conflated cold and warm.
PROOF (6x c1 back-to-back, same serve, warm steady-state = runs 2-6):
  warps=1 (baseline): 9.47, 9.47, 9.44, 9.47, 9.40, 9.41  -> ~9.44 t/s
  warps=4 ("win")   : 9.44, 9.42, 9.48, 9.49, 9.47        -> ~9.45 t/s   == IDENTICAL.
So num_warps 1->4 gives NO warm decode improvement (the GDN recurrent kernel was NOT the warm bottleneck; 192
programs already saturate the GPU at num_warps=1). The earlier 4.68->7.92->9.60 deltas were ALL cold-vs-warm noise.
The B70_GDN_DECODE_WARPS knob is kept (env-tunable, harmless) but is a NO-OP warm; do NOT claim it as a speedup.
LESSON: ALWAYS warm the serve (discard the 1st bench) before recording a decode number. Re-measuring all configs warm.
REAL (good) finding hidden under the artifact: woq int4 TP=1 WARM decode = ~9.44 t/s -- single-card, already
competitive with bf16 TP=2, with NO kernel change. -> woq int4 DP=2 (wedge-proof + vision) ~9.4 single / ~18 aggregate
is a strong daily driver. The campaign's true levers remain: correct serving (achieved) + int4 single-card (fast warm).

## (superseded) GDN num_warps experiment log:
The GDN decode kernel `fused_recurrent_gated_delta_rule_packed_decode` hardcoded `num_warps=1` (grid=(NV=4,
B*HV=48)=192 programs, each 1 sub-group) -> OCCUPANCY-STARVED on Battlemage. The recurrence has no within-step
time dependency at decode (1 token), so num_warps only parallelizes the spatial head/V work -> safe to raise.
Made it tunable (B70_GDN_DECODE_WARPS env; sglang/patches/fused_recurrent.py).
MEASURED (woq int4 TP=1, IN=2048):
  warps=1 (baseline): decode 4.68 t/s  (TPOT 213.65 ms)
  warps=4           : decode 7.92 t/s  (TPOT 126.34 ms)  = 1.69x, COHERENT (Rayleigh answer)
This is the FIRST real decode speedup of the campaign, and it applies to EVERY config (bf16 too -- same kernel).
NEXT: sweep warps (8/16) for the peak; apply to bf16 TP=2 (the daily driver, baseline 9 t/s -> ?); bake the winner.

APPLIED to bf16 TP=2 (warps=4): decode 9.03 -> 9.44 (+4.5%), TTFT 661 -> 603, prefill 3098 -> 3394. COHERENT.
INSIGHT: the GDN-warps win is MUCH bigger single-card (woq TP=1 1.69x) than TP=2 (+4.5%) -- at TP=2 the GDN heads
are SPLIT across cards (grid (4,24) vs (4,48)) AND decode is also all-reduce+weight-bandwidth bound, so the GDN
recurrent kernel is a smaller fraction. => the GDN-warps optimization most benefits the SINGLE-CARD int4 path,
making woq int4 TP=1 (7.92, wedge-proof, vision) a strong DP=2 daily driver (~15.8 aggregate). A second occupancy-
starved kernel fused_gdn_gating (num_warps=1) is now also tunable -- testing combined + warps=8 on woq next.
Scoreboard (decode t/s, IN=2048 c1): bf16 TP2 w1=9.03 w4=9.44 | woq int4 TP1 w1=4.68 w4=7.92.

## (parked) AWQ track. See sglang/AWQ_RECIPE.md.
- De-risk #1 (fp16-through-GDN, the AWQ act dtype): re-serve UNQUANTIZED bf16 with `--dtype float16` + gdn_nan_repro.
  Isolates the fp16 question from quant before producing any checkpoint.
- De-risk #2 (AWQ XPU speed): repack text-only W4A16 -> AWQ (CPU), serve, bench vs bf16.
- Then production: AutoRound auto_awq full-VLM (vision-retaining).

## Image capability map (VERIFIED 2026-06-27, image sglang-xpu:bmg)
Verified by listing sgl_kernel `.so` files + `torch.ops.sgl_kernel` registration (decisive: registration,
not just `dir(sgl_kernel)` Python wrappers, which exist for unregistered ops):
- WORKING XPU quant GEMM: **AWQ only** -- `awq_dequantize` is the lone registered quant op (2 .so), dequant
  4-bit -> fp16 then native `torch.matmul`. Needs `--quantization awq --dtype float16` + an AutoAWQ-format ckpt.
- DEAD on XPU (no .so / not registered): `int8_scaled_mm`, `fp8_scaled_mm`, `qserve_w4a8_*`, compressed-tensors
  W4A16/W8A8-int8/W8A8-fp8 (WNA16 -> Marlin, CUDA-gated), GPTQ/Marlin. So our W4A16/W4A8/W8A8 compressed-tensors
  AND AutoRound int4 (auto_gptq) checkpoints CANNOT serve on this XPU build.
- MXFP4 W4A16 group-gemm kernels (`GroupGemmMxfp4W4A16Xe20`) EXIST but are MoE-group-gemm only (N/A to dense 27B).
- DEAD (need new kernels/port, NOT flags): MTP/spec-decode (NEXTN draft wired but EAGLE has no intel_xpu attn +
  no xpu graph-runner key), CUDA-graph decode capture (XPUAttentionBackend implements no graph methods),
  `--enable-torch-compile` (no-op on XPU; decode is EagerRunner).

## Levers, ranked
1. **AWQ W4A16 checkpoint** (`--quantization awq --dtype float16`) -- THE decode win (4-bit weight bandwidth),
   may enable TP=1 single-card -> no all-reduce -> DP=2. Needs: produce an AutoAWQ-format ckpt (retain vision +
   GDN exclusions) + validate fp16 GDN numerics (gdn_nan_repro). MAIN THRUST.
2. Pure-flag decode levers (no checkpoint, try-now): `--enable-linear-replayssm`,
   `--num-continuous-decode-steps 2`, env `FLA_USE_FAST_OPS=1`. (Mostly aggregate/batch wins; A/B for single-stream.)
3. `--pp-size 2` vs `--tp 2` A/B (avoid per-layer all-reduce; aggregate/KV win, not single-stream latency).
4. Prefix cache recovery for agentic TTFT: `--mamba-radix-cache-strategy no_buffer --page-size 1` +
   `--attention-backend triton` (drops intel_xpu XMX attn) + `--schedule-policy lpm`. Bigger change; later.
5. Custom dense int8/int4 GEMV SYCL kernel (frontier; the user-offered lever if AWQ underperforms).

## woq int4 driver VERIFIED CORRECT under mixed load (2026-06-27)
Ran contrib/gdn_nan_repro on the woq int4 serve (the agentic mixed prefill+decode load that makes vLLM emit "!!!!"):
  dd_loadprobe 8 anchors + 12 concurrent probes -> all OK, coherent (the exact backhoe/grading request that broke
  vLLM); dd_rawtokens logprobs under load -> real logprobs, NO 'nan'. So the int4 woqgemm path inherits sglang's
  GDN fix (no NaN). => woq int4 (sglang-xpu:woq) is a VERIFIED-CORRECT daily driver: int4 single-card, vision, ~9.44
  t/s warm, correct under the load that breaks vLLM. Serve:
    IMG=sglang-xpu:woq CKPT=/models/Lorbus_Qwen3.6-27B-int4-AutoRound SERVED=qwen36-27b-int4-woq TP=1 DEVICE=N \
      bash sglang/serve_sglang.sh start   # one per card for DP=2

## woq int4 DP=2 daily driver: PACKAGED + VERIFIED (2026-06-27)
sglang/serve_dp2.sh launches the wedge-proof driver: 2 single-card woq replicas (card0:30000, card1:30001) +
nginx round-robin proxy :18080 (sglang/dp_nginx.conf). VERIFIED: both replicas healthy + coherent simultaneously
(one per card), proxy /health 200, round-robin requests coherent ("capital of France"). No cross-card collective
-> cannot BCS-wedge. CORRECT (gdn_nan_repro clean under load) + VISION + ~9.44 t/s/replica warm.
  Launch: ./sglang/serve_dp2.sh start | status | stop
TWO DAILY DRIVERS, both CORRECT + VISION:
  1. woq int4 DP=2 (serve_dp2.sh) -- UNATTENDED: wedge-proof, ~9.44/replica, int4 = big KV, :18080.
  2. bf16 TP=2 (serve_sglang.sh)  -- ATTENDED: ~9.2 c1 / 23.4 c4 aggregate, both cards.

## *** BREAKTHROUGH: XPU CUDAGRAPH WORKS -- decode 9.4 -> 23.6 t/s (graph steps) *** [2026-06-27]
WIRED UP via woq_shim (opt-in B70_XPU_CUDAGRAPH=1) + a mounted model_runner.py patch:
  1. support_cuda_graph()->True on XPU. 2. add "xpu" to model_runner's IN-TREE decode-graph device list
  (model_runner.py:924) -- the OUT-OF-TREE path is a dead end (core still hardcodes torch.cuda.*; "future PR").
  3. torch.cuda.CUDAGraph->torch.xpu.XPUGraph + graph_pool_handle. 4. ADAPTER on torch.xpu.graph (backend calls
  self._device_module.graph(cuda_graph=...) but torch.xpu.graph wants graph(xpu_graph,pool,stream)). 5. --attention-
  backend triton (graph-capable; intel_xpu backend lacks graph methods).
RESULT: capture SUCCEEDS (all 7 bs=1..24 graphs), COHERENT, decode logs `cuda graph: True` at **23.6 t/s** (eager
  ~9.4 = 2.5x!). torch.xpu graph capture of this hybrid GDN model WORKS (GDN init_cuda_graph_state carries mamba state).
REFINED (OUT=512, warm): decode-batch log STEADY at ~23 t/s (cuda graph: True), only ONE 12.66 dip -> the stall is
  essentially ONE-TIME (capture/first-interval), not periodic. End-to-end bench: OUT=128 -> 8.45, OUT=512 -> 12.57
  (the one-time stall amortizes). So: MODEL decode (server-side) = ~23 t/s (2.5x eager 9.4); END-TO-END (client) =
  ~12.57 t/s at OUT=512 (+34% over eager 9.4). The 23-vs-12.57 gap = per-token SERVING overhead (detok/stream/
  scheduler, NOT graphed). TRADEOFF: TTFT REGRESSES ~920 -> ~1938 ms because cudagraph requires --attention-backend
  triton (the intel_xpu backend lacks graph capture/replay methods) + prefill isn't graphed. So cudagraph = a DECODE
  win with a PREFILL/TTFT cost. num-continuous-decode-steps=2 did NOT help (end-to-end ~11.5, same). The gap is per-token detok/stream overhead
  at stream_interval=1, not scheduler steps. NEXT: --stream-interval N (fewer streaming round-trips). Old NEXT: (a) (--num-continuous-decode-steps stacks with
  graph?), (b) recover TTFT (implement intel_xpu backend graph methods, OR keep intel_xpu for prefill + triton for
  decode). Honest verdict: first REAL decode speedup of the campaign (+34% end-to-end, 2.5x model), opt-in + correct.

## (superseded -- DONE above) XPU cudagraph FEASIBLE-IN-PRINCIPLE [2026-06-27]
The ~9.4 t/s warm ceiling is EAGER LAUNCH OVERHEAD (190+ triton/kernel launches per token, no graph replay).
vLLM beat it (captured graph 30.5 vs eager 23.3) via cudagraph. KEY DISCOVERY: torch.xpu 2.12 SUPPORTS graph
capture -- torch.xpu.{XPUGraph, graph, graph_pool_handle, make_graphed_callables} all exist. And sglang's
XPUAttentionBackend already has PARTIAL graph scaffolding (decode_cuda_graph_metadata, get_cuda_graph_seq_len_
fill_value). The blockers: (1) current_platform.support_cuda_graph() returns False on XPU (model_runner gates
decode-graph to cuda/musa/cpu/npu); (2) the graph runner uses torch.cuda.CUDAGraph (needs torch.xpu.XPUGraph);
(3) XPUAttentionBackend lacks init_forward_metadata_capture/replay_cuda_graph (static buffers for replay);
(4) the GDN/mamba recurrent state cache must be made graph-replay-safe (the hard part for this hybrid model).
=> A real but BOUNDED integration. If it works, decode ~9.4 -> ~25-30 t/s (like vLLM captured). This is THE next
big lever. NEXT: bounded probe -- patch support_cuda_graph(xpu)=True + torch.cuda.CUDAGraph->torch.xpu.XPUGraph in
the graph runner, see how far capture gets + what breaks (scope the effort, like the MTP probe).
