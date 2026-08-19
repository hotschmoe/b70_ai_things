# Qwen3.8-27B W8A8-INT8 + DSpark -- frontier campaign

**Created:** 2026-08-18
**Status:** LOOPING -- research locked, no GPU work started on this track
**Goal:** make 2x Intel Arc Pro B70 the place people point at for private
Qwen3.8-27B: W8A8-INT8 on native XMX, a *matched* DSpark (and a prefill
arm), FP8-class quality, vision + MTP retained, both newest vLLM *and*
sglang as first-class vehicles.

This file is the standing prompt for a **continual loop**. Agents re-read
it every iteration. Do not treat it as a one-shot plan. See **LOOP**
(section L) before doing anything else.

Dead-ends are first-class results. Log them. Do not hide them.

Standing policy this campaign *intentionally stretches*: AGENTS.md still
says sglang is the W8A8 research backend and vLLM is paused; here we
run **both** to the newest Intel-capable cut, and we will rebuild
kernels / torch / dispatch / comms rather than stop at the 2.12 ABI
lock. The lock is Phase 0, not the ceiling.

### Living header (every loop updates this block, nothing above section 0)

| field | value |
|---|---|
| Last loop | 38 (D13 GDN fallback: Paris/391 hold, fib bangs) |
| Last JOURNAL heading | `2026-08-19u` |
| Loop ledger | `docs/20260818_qwen38_w8a8_dspark_loops.md` |
| Dead-ends | `docs/20260818_qwen38_w8a8_dspark_deadends.md` |
| Next pick | Steve vLLM 44fc8fde0 + graph-safe FA |
| Blocked on | D13 addendum (fib bangs; GRAPH TP=2 disabled). 101.922 cell not measured. |
| HE+ (W8A8-gptq) | **0.957 / 0.927** (GRAPH=0 MTP3 @131k, thinking-off greedy) |
| HE+ (W4A16-autoround) | **0.963 / 0.915** (GRAPH=0 TP=1 MTP5 @16384, thinking-off greedy) |
| Best W8A8 `bench_code` c1 | 26.62 MTP3 @131k (pre-campaign, JOURNAL 2026-08-15c) |
| Best W8A8 DSpark `bench_code` c1 | **29.4** k=4 GRAPH=1 ALLGATHER_ASYNC @122880 (G1 hold; was 28.7 push-AR only) |
| Best W8A8 DSpark accept_len / pos0 | **2.46 / 0.62** (k=7 GRAPH=0); k=4 GRAPH=1 greedy 2.45 / 0.65, **prob 3.16 / 0.80** |
| DD | PARKED. Do not start. Cards belong to this campaign. :18080 is research. |

---

## L. LOOP -- read this every iteration

You are one iteration of a long campaign. Previous loops left you a
ledger and a JOURNAL tail. Subsequent loops will only be as good as
the handoff you write. Re-planning the campaign from scratch is a
failure mode, not diligence.

### L.0 Read order (do this before any edit or GPU touch)

1. This file, especially the living header and this section.
2. `docs/20260818_qwen38_w8a8_dspark_loops.md` -- last 3 loops, then
   the "NEXT PICK" line. That file is the feedback channel.
3. `docs/20260818_qwen38_w8a8_dspark_deadends.md` -- do not retry a
   closed packet unless the packet itself says the retry condition
   is now true.
4. JOURNAL.md, newest entries at the **bottom**. Search headings
   `2026-08-18` onward and any `LOOP` / `W8A8` / `DSpark` / `qwen38`
   tags. The ledger is the index; JOURNAL is the evidence.
5. `./bin/gpu-run --status` and `curl -s http://192.168.10.5:18080/v1/models`.
   DD is PARKED for this campaign window. :18080 is empty or a
   research serve. Never start `daily_driver_serve.sh`.
6. Only then pick work. One pick per loop unless the first pick is
   a 5-minute no-GPU edit that unblocks the real pick.

### L.1 What one loop is

A loop is **one verdict**, not one phase.

- Allowed: one P0/P1/P2/P3/P4 row, or one standing-list packet, or
  one no-GPU unblock (yaml id, serve-script clone, eval config).
- Not allowed: "do Phase 0", "train the draft", "rebuild torch 2.13",
  "rewrite the campaign", "improve the plan a bit first".
- Stop when you have config -> command -> result -> verdict, **or**
  you are blocked on lease / DD / health / a missing artifact.
- If the pick is a multi-hour GPU job (HE+, quant, train), start it
  under `gpu-run`, write a LOOP-STARTED ledger line with the log
  path and pid. Leave a research serve up if the next pick needs
  the same id. Do not start DD. Do not sit on the lease after the
  job ends.

### L.2 Where to write (so the next loop can start)

Write in this order at the end of the loop. Skip none.

| dest | what goes there |
|---|---|
| `JOURNAL.md` bottom | Full evidence. Heading `### YYYY-MM-DD<letter> - LOOP N: <one line>`. Body = CONTEXT / CONFIG / COMMAND / RESULT / VERDICT. Commands are copy-pasteable. Numbers have units, ctx, concurrency, served id. |
| `docs/20260818_qwen38_w8a8_dspark_loops.md` | One `## LOOP N` block. See L.3. This is what the next agent reads first. |
| `docs/20260818_qwen38_w8a8_dspark_deadends.md` | Only if you closed a path. Packet format is in that file. |
| This file, living header only | Bump Last loop / Last JOURNAL heading / Next pick / Blocked on / the three score rows if they moved. |
| `RESEARCH_TODO.md` campaign blurb | One-line status if Next pick or a north-star number changed. |

Do **not** rewrite sections 0-9 of this file to narrate the loop.
Do **not** rewrite old `scripts/NN_*.sh`. Copy to a new number.
New experiment scripts live under the backend root (`vllm/`, `sglang/`).

### L.3 Ledger block the next loop needs

Append this shape to the loop ledger. Keep it short. The JOURNAL
entry holds the long form.

```
## LOOP N -- YYYY-MM-DDThhmmZ -- <one-line pick>

Picked: P0.x / P1.x / unblock / packet
Why this, not the other open row: <one sentence, cite last loop>
GPU: lease holder / card / DD stopped? / port / served id
Command: <the actual command>
Log: <path>
Result: <the number or the error, one or two lines>
Verdict: GO / NO-GO / BLOCKED / DEAD-END
Changed beliefs: <what a future loop must not re-discover>
Next pick: <exact next row or unblock, and the first command>
Do not: <the tempting wrong follow-up>
Restore: DD stays down. xpu-health? lock released?
JOURNAL: ### YYYY-MM-DD<letter>
```

If you started a long job and are handing off mid-run, use
`Verdict: RUNNING` and fill Command / Log / pid / how to tell it
finished / what the next loop should do on success vs fail.

### L.4 How to pick (decision tree, every loop)

Honor the living-header **Next pick** unless one of these is true:

- The ledger says that pick is RUNNING and the log is still live
  -- then you monitor / finish / restore, you do not start a sibling.
- The last verdict BLOCKED and the blocker is now gone -- same pick.
- The last verdict DEAD-END or NO-GO -- take the "likely dead-end"
  branch in section 6 for that row, not a creative new idea.
- HE+ plus is measured and `< 0.90` -- **stop speed work**. Quality
  only (A.2 SQ, A.3 AutoRound, A.4 RukaRat). Write that in Next pick.
- Off-shelf DSpark pos0 is already within ~2% of RadixArk-on-FP8
  -- skip the long train (C); short 400-step polish only.
- Accept is ugly (~20% band) -- train is mandatory (C), not more
  k-sweeps.
- Accept is fine but c1 < W8A8 MTP3 -- kernels / verify-AR (D, E),
  not more training.
- Phase 0+1 do not yet have a coherent W8A8+DSpark number **and**
  a written list of 0.27-only features we need -- **do not enter
  Phase 2**.
- Phase 4 / "PSpark" is not a week-1-2 pick. Prefix-cache TTFT
  baseline (P4.1) is the only prefill number allowed early.

Default order while Phase 0 is open: P0.1 -> P0.2 -> P0.3 -> P0.4
-> P0.5. Week-1 concrete list in section 8 still stands. A no-GPU
unblock that is on that list (evals yaml id, serve-script clone)
may run before the GPU slot for P0.1.

### L.5 Standing facts that loops keep forgetting

Memorize these. They are how previous weeks got burned.

- Query `/v1/models` before trusting any number. Served ids encode
  method+scheme (`qwen3.8-27b-W8A8-gptq-mtp3`,
  `qwen3.8-27b-W8A8-gptq-dspark7`). Never a bare `qwen3-14b-w8a8`.
- Fail-closed: G1 (Paris / 17*23 / fib) or G5 (18/18, no "!!!!")
  fail => do not publish the speed number.
- `gpu-run` for every real GPU touch. `gpu-run --card N` for one
  card. Editing and compiling do not take the lease.
- DD is PARKED (operator 2026-08-18f). Do not run
  `vllm/daily_driver_serve.sh start`. Do not restore
  `hotschmoe-dd` between loops or after a job. Cards belong
  to this campaign until the operator says otherwise.
- Speed window (operator 2026-08-18m): do not start A.2-A.4
  requant. Quality floor is HE+ 0.957/0.927. G1 fail => do
  not publish speed; revert. GRAPH=1 short bench_code is
  allowed. Do not HE+ under GRAPH=1 CGRECLAIM=0. Do not
  overnight-train; accept is already in the NVFP4 band.
- SergiioB 3.8 cookbook (2026-08-18v):
  `docs/20260818_qwen38_sergiioB_cookbook.md`. 83.7 is 1x B70
  GPTQ-Int4 MTP4 on vLLM 0.27.2rc1, not our W8A8 TP=2 c1.
  Do not demote W8A8. Do not mix digest `f01e24f6` with the
  3.6 `2c427ef` or Nemotron `1da0a954`. S1 already smoked
  (47.58 on this box).
- Steve 3.8 INT4-AR (2026-08-19b / 19h YOLO):
  `docs/20260819_steve_qwen38_int4ar.md`. MTP5 **101.922**
  / MTP4 100.497 after-TTFT on 2x B70. Weights on disk.
  S2b is the next GPU fire on 0.27 `f01e24f6`, not
  int8g-v0260. Compile-key SPECTOK+SO landed LOOP 26
  on the 0.26 DSpark path.
- P2P=1 in vLLM TP>1 wedges the box. Recovery = reboot. Never
  chain two tries. `I_KNOW_P2P_WEDGES=1` required. oneCCL overlay
  is 2021.17; 2021.15 is broken.
- After any TP>1 teardown that threw DEVICE_LOST, run `xpu-health`
  on a single card before the next TP>1 start.
- method=dflash is unregistered on v0.26. method=dspark, V2,
  `THINK_BUDGET=0`. Adaptive verify is dead on GDN.
- Draft geometry is locked (section C). Do not invent a new arch.
- Capture target for a matched draft is the **live W8A8 serve**,
  not the FP8/NVFP4 DD. Draft stays BF16.
- Do not overwrite `models/files/qwen3.8-27b/w8a8-gptq`. SQ /
  AutoRound get new dirs.
- GDN / visual / mtp stay BF16. Ignore lists stay ignore lists.
- Xe2 rule: never write an FP8 GEMM. Repack FP8 weights to s8.
- Do not tune a GEMM already at 88-100% of the 581 GB/s roofline.
  Win is shape and fusion, not a faster large-M s8s8s32.
- sglang 0.5.6 stays the W8A8 *shelf* until a newer cut is
  measured faster-or-equal **and** coherent. 0.5.15 already lost
  once vs 0.5.6 (-6.1% c1).
- "PSpark" is our name for a speculative-prefill arm, not a
  DeepSeek/SpecForge sibling. Do not invent a fake checkpoint.
- ASCII only. No emoji, typographic arrows, or smart punctuation.
- We run locally on this box. Do not SSH. Repo is
  `/mnt/vm_8tb/github/b70_ai_things`. Runtime root `/mnt/vm_8tb/b70`.

### L.6 Gates you must run when the pick is a serve/bench

Reuse section 6. Minimum for any published speed: G0, G1, G3, and
G5 if you claim concurrent. Spec picks add G4. Quality artifacts
add G6. Ctx >= 200k adds G7.

Identity check is not optional. If `/v1/models` does not match the
planned id, the number is trash even if Paris is exact.

### L.7 Commit and push

On the host, commit when a loop has a verdict or a durable artifact
(script, yaml, ledger, dead-end packet). Do not rewrite history.
Do not leave the only copy of a result in a chat transcript.

### L.8 Stop-the-line

Stop the loop (and write BLOCKED / DEAD-END) if:

- cards are unhealthy and `xe-reset` wants a reboot
- you are about to start Phase 2 without a Phase 0+1 number
- you are about to start a long DSpark train without the 10-sample
  overfit gate (P1.2)
- HE+ plus `< 0.90` and the pick is a speed experiment
- someone asks for P2P=1 and you do not have a reboot window
- the pick needs both a kernel exploit-style attack and a fix --
  fix only; do not write attack payloads

Then hand off. The next loop exists so you do not have to hero.

---

## 0. Why this path (one screen)

Xe2 XMX is **INT8/INT4 DPAS**. There is no FP8/FP4 systolic path.
NVIDIA's 206 tok/s (5090, SGLang NVFP4 + DSpark) is "verify is cheap
on their tensor cores." Our M1 (NVFP4 + off-the-shelf DSpark k=7)
was **34.4 vs MTP3 41.2** -- accept was real (2.53), verify was not
cheap. The Intel-shaped inversion:

```
cheap INT8-XMX verify of a k-token block
  + a DSpark trained on *this* target's hidden states
  + vision and MTP kept
  = the thing nobody has shipped
```

No public Qwen3.8-27B **W8A8-INT8 + DSpark** exists. That is the map
pin. Q8 GGUF + DSpark is a dead product (llama.cpp has no DSpark).
Q8 as a vLLM target is W8A16-class and does not light XMX.

"PSpark" is **not** a DeepSeek/SpecForge sibling. Treat it as our
name for a **speculative-prefill** arm (SpecPrefill / PFlash-class).
Do not invent a fake checkpoint. See section 6.

---

## 1. North-star numbers (what "on the map" means)

Must beat, on this box, under concurrent load, with vision live:

| bar | today | campaign target |
|---|---|---|
| 3.8 W8A8 decode c1 (code harness) | 18.45 MTP-off / **26.62 MTP3** | **>= RadixArk MTP3 41.2**, stretch **>= Q4_K_M 43.8**, fantasy **>= 3.6 NVFP4 48.9** |
| 3.8 W8A8 HE+ (thinking-off) | **unmeasured** | **>= 0.970/0.927** (Q4_K_M) and within 2 plus-pts of a same-harness FP8 3.8 |
| Accept (matched DSpark) | 2.53 on NVFP4 / FP8-trained draft | **>= 3.3 mean** on reasoning (RadixArk-on-FP8 card); pos0 >= 60% |
| Ctx | 229k MTP-off / 131k MTP3 | **native 262144** with spec on |
| Vision | landscape probe correct | same + a 20-image smoke, no silent-zero tower |
| Concurrent | c4 stayed up, no 18/18 | **18/18 mixed prefill+decode**, no "!!!!" |
| Dual backend | vLLM 0.26 only | vLLM newest *that we can load kernels on* **and** sglang newest *that can run W8A8+DSpark on XPU* |

Publishables (any one of these is a map pin):

- First public 3.8 W8A8-INT8 + matched DSpark recipe on Arc B70
- Small-M INT8 DPAS verify kernel that makes k=7 cheaper than MTP3
- XPU SpecForge offline train recipe (port of rwmacy #769)
- sglang-XPU DSpark (today CUDA/Spark only)
- Honest dead-end packets (P2P=1, torch 2.13 topk SIGSEGV, adaptive
  verify on GDN, FATTN_MMA JIT, off-shelf draft on W8A8)

---

## 2. What is already on this box

### 2.1 Daily driver (restored 2026-08-18)

`hotschmoe-dd` on :18080, vLLM 0.26.0 NVFP4 Qwen3.6-27B TP=2, cal fp8
KV, MTP5, native 262144. Paris exact. Do **not** take this down for
Phase 0 editing/compiling. GPU work uses `gpu-run` and stops the DD
only for on-GPU quant / serve / bench.

### 2.2 3.8 W8A8 artifact (the campaign target)

- Path: `models/files/qwen3.8-27b/w8a8-gptq`
- GPTQ W8A8, compressed-tensors, `SMOOTHQUANT=0`, 512 samples
- Ignore: `lm_head`, `linear_attn`, `visual`, `mtp` (those stay BF16)
- Grafted: 333 visual + 15 `mtp.*` from official BF16
  (`models/graft_qwen38_w8a8.py`)
- Kernel already fires: `XPUInt8ScaledMMLinearKernel`
- Recipe: `vllm/w8a8/serve_qwen38_27b.sh` (not a shelf)
- Numbers (JOURNAL 2026-08-15b/c):

| config | ctx | c1 TG | c4 agg | PP | accept |
|---|---:|---:|---:|---:|---|
| MTP-off text | 229376 | 18.45 | 51.8 | 2574 | -- |
| MTP3 + vis | 131072 | **26.62** | 50.2 | 2216 | 2.0-2.65 |

No HE+. No `bench_code` c1. No 262k + spec. No 18/18.

### 2.3 DSpark already wired (vLLM 0.26)

- Readout fix: `vllm/dflash/patches/v0260/{dflash,utils}.py`
- Arch remap: `DSparkDraftModel` -> `Qwen3DSparkModel`
- M1: NVFP4 + rwmacy draft, method=dspark k=7, V2, fp8 KV, 262k,
  code **34.4** vs MTP3 **41.2**, accept_len 2.53, pos0 58.4%
- method=dflash is unregistered (`DFlashQwen3DSparkModel`)
- V2 rejects `thinking_token_budget` (THINK_BUDGET=0)
- Adaptive verify **dead** on `GDNAttentionBackend`
- Drafters on disk: RadixArk (FP8-trained) and rwmacy B70 (FP8 hiddens)

### 2.4 Kernels (shared source, per-backend ABI)

`kernels/`: `int8_gemm_w8a8` (prefill), `int8_gemm_w8a16` (decode /
small-M), fusedq, NVFP4 decompress GEMMs, `xpu_shard_top1`.
Built into `vllm-xpu-env:int8g-v0260` (torch **2.12**) and
`/mnt/vm_8tb/b70/w8a8_kernel/_xpu_C.abi3.so` (sglang 2.12).
sycl-tla small-M INT8 scaffold exists (`kernels/SYCLTLA_SCAFFOLD.md`)
and is **not** the optimized kernel yet.

Xe2 rule: never write an FP8 GEMM. Repack FP8 weights to s8.

### 2.5 Backends today

| | vLLM | sglang |
|---|---|---|
| Shelf W8A8 | 3.6 only, `:int8g-v0260` | 3.6 only, `sglang-xpu:mtp` (0.5.6) |
| Research | 0.26.0 / torch 2.12 | 0.5.15 / torch 2.12 |
| Newest name | public `vllm-openai-xpu:v0.27.1` torch **2.13** | 0.5.17 + main #31751 torch **2.13** |
| 3.8 DSpark | **works** (M1) | **CUDA/Spark cookbook only** |
| 3.8 W8A8 | recipe exists | **none** |

---

## 3. Community map (what we are stealing)

| source | takeaway |
|---|---|
| SGLang + Qwen, 5090 NVFP4+DSpark **206 tok/s** (166-182 field) | DSpark pays when verify is cheap. Not a number to photocopy. |
| B300 TP8 2.4T: DSpark 378 / accept 4.0 vs MTP 346 / 3.3 | Same story at scale. |
| DGX Spark 34-38; MiaAI: MTP **beat** DSpark | Off-shelf draft + slow verify = MTP wins. We already reproduced that (34.4 < 41.2). |
| RadixArk DSpark card | Trained on **FP8** hiddens. 1.36B, block 7, layers 4/16/28/40/52, Markov 256. Accept mean 3.39 (GSM8K 4.57, HE 3.47, chat ~2.7). |
| rwmacy SpecForge #769 + B70 train | XPU offline train **works**. Warm-start + 961 samples + 440 steps. Constraints: sdpa, anchors<=64, chunk<=16, XPU_GRAPH=0, workers=0, CPU offload. Claimed 72 isolated C1 **did not** reproduce on our 0.21 image (20-22). |
| Dolboyob77, 1x B70 vLLM, GPTQ-Int4 + RadixArk DSpark | 28 / MTP2 50 / DSpark greedy 42 / **probabilistic 52**. k=7 beat k=4/6. Adaptive verify refused. |
| SpecForge v0.3 | DSpark yamls for 3.6-27B, not 3.8. Capture = patched sglang 0.5.14 CUDA. XPU = offline only. |
| DeepSpec | 38 TB offline cache if you do it their way. Do not. |
| Steve lab DSpark7 on V4-Flash, 4x B70, **80.8** | XPU DSpark kernel lessons: sharded target top-1, persistent Markov, do not materialize full vocab. Steal after accept is real. |
| Steve lab Qwen3.8 INT4-AR TP=2 (2026-08-19) | vLLM/XPU AutoRound W4A16 MTP5 **101.922** all-25 / MTP4 **100.497** (better on sel-12). After-TTFT, pinned compile cache, LocalMaxxing APPROVED. Digest `docs/20260819_steve_qwen38_int4ar.md`. S2 later, not W8A8. |
| RukaRat 3.8 W8A8-INT8 imatrix | Only public true W8A8-INT8 3.8 besides ours. A/B later, do not replace identity until gated. |
| SergiioB QWEN38-VLLM-XPU (2026-08) | 1x B70 GPTQ-Int4 + BF16 MTP, vLLM `0.27.2rc1` digest `f01e24f6`, MTP4 p512/g128 **83.7** post-first, accept 93-96%. C1 ceiling. Digest + digest law in `docs/20260818_qwen38_sergiioB_cookbook.md`. Do not mix with 3.6 `2c427ef`. |
| No PSpark product | SpecPrefill (arxiv 2502.02789, vLLM #39060 stale). PFlash = llama.cpp fork. P-EAGLE = decode-side. |

---

## 4. Dual-backend strategy

We try **both newest stacks**. We do not pretend 2.13 is free.

```
Phase 0  -- prove W8A8+DSpark on the ABI we have (0.26 / 0.5.15, torch 2.12)
Phase 1  -- train a matched DSpark; kernel/dispatch/comm on 2.12
Phase 2  -- dedicated torch-2.13 rewrite: vLLM 0.27.1 AND sglang 0.5.17+#31751
Phase 3  -- sglang-XPU DSpark port (the real "newest sglang" prize)
Phase 4  -- speculative prefill ("PSpark") + publish
```

Phase 2 is a **kernel rewrite session**, not `pip install`. Budget:

- rebuild `_xpu_C.abi3.so`, GDN so, NVFP4 so, fake-op registrations
- oneAPI 2026.0 (sglang #31751)
- soak: `torch.topk` SIGSEGV, `nonzero`/`unique` empty-on-26.22
- oneCCL 2021.17 overlay (0.27 images will ship 2021.15 again)
- graph fake names may have moved (`XPUW8A8FP8LinearKernel` already renamed once)

**Do not enter Phase 2** until Phase 0+1 have a coherent W8A8+DSpark
number and a written list of 0.27-only features we actually need
(quantized Markov heads, newer V2). If 0.26 already serves the draft,
2.13 is optional.

sglang 0.5.6 stays the W8A8 *shelf* until 0.5.15 or 0.5.17-XPU is
measured faster-or-equal **and** coherent (already failed once:
0.5.15 vs 0.5.6 was -6.1% c1).

---

## 5. Workstreams

### A. Quality -- "FP8-level W8A8"

W8A8 is allowed to lose at most **2 plus-pts** vs a same-harness
3.8 FP8 (or vs Q4_K_M 0.970/0.927 if we never stand up FP8 3.8).

1. **HE+ now** on grafted W8A8 MTP3 (thinking-off, greedy, 164).
   Identity: `qwen3.8-27b-W8A8-gptq-mtp3`. Add to `evals/configs/models.yaml`.
2. Selective SmoothQuant on the 16 full-attn layers (auto-SQ throws
   on the hybrid). New artifact `w8a8-sqgptq`, do not overwrite gptq.
3. AutoRound W8A8 A/B (policy: A/B, not first artifact). Calibrate
   CPU/CUDA, never XPU.
4. RukaRat imatrix W8A8 A/B after our GPTQ has a HE+ number.
5. lm_head: keep BF16 (ignored today). INT8 lm_head is a later
   bandwidth experiment, quality-gated.
6. GDN / visual / mtp stay BF16 unless a dedicated session proves
   int8 GDN is quality-safe (3.6 taught us this is a landmine).
7. Same-prompt FP8 vs W8A8 vs Q4_K_M battery: Paris, 17*23, fib,
   HE+ fail lists, 20-image vision, 3.8k needle.

If HE+ plus is < 0.90, **stop speed work** and fix the quant. A
fast wrong model is not a map pin.

### B. Serve baseline -- W8A8 at 262k, both spec methods

1. `vllm/w8a8/serve_qwen38_27b.sh` @ native 262144.
   Levers: `KV_FP8=1` (M1's 262k trick), UTIL, MAXSEQS, embed-INT8
   if the 3.6 block-13 trick ports.
2. MTP3 vs MTP5 A/B (`bench_code` + accept). Record accept_len.
3. Off-shelf DSpark on W8A8 (RadixArk + rwmacy), method=dspark,
   **probabilistic** sample, k in {3,4,7}, THINK_BUDGET=0, P2P=0,
   oneCCL 2021.17. This is the "does accept collapse?" experiment.
4. 18/18 mixed-load + Paris + vision smoke.
5. Port the same CKPT to sglang 0.5.15 + `w8a8_shim.py` + NEXTN.
   DSpark on sglang-XPU is Phase 3, not this step.

Gate to continue: W8A8 MTP3 `bench_code` c1 measured; HE+ started;
DSpark-off-shelf accept table exists (even if it is 20% and ugly).

### C. Train our DSpark

**Geometry (do not invent a new arch):** 1.36B BF16, 5 layers,
hidden 5120, GQA 40/8, aux layers **4/16/28/40/52**, Markov rank
256, **block 7**. Warm-start `RadixArk/Qwen3.8-27B-DSpark`.

**Tool:** SpecForge `strategy: dspark` **offline** (PR #769 XPU
constraints). Not DeepSpec 38 TB. Not online sglang 0.5.14 capture.

**Capture target = the live W8A8 serve**, thinking mode = serve
mode. Draft stays BF16; only captured hiddens change.

Sequence:

0. Confirm readout fix is in the serve image (M1 already applied
   it for NVFP4). Without it every new head sticks at ~24% accept.
1. Tiny Linear + 1.36B fwd/bwd/AdamW smoke on card 0 (rwmacy did
   this; re-prove on *our* image).
2. Regen 1-2k prompts (ShareGPT / Nemotron-v2 / a code+math mix)
   from the W8A8 server, temp 0, save thinking if thinking is on.
3. Dump hiddens at layers 4/16/28/40/52. Budget tens of GB, not 38 TB.
4. Train: `SPECFORGE_DEVICE=xpu`, sdpa, `num_anchors<=64`,
   `objective_chunk_blocks<=16`, workers=0, XPU_GRAPH=0, CPU
   offload, lr **5e-5** (warm-start; 6e-4 is from-scratch CUDA),
   save every 16 steps.
5. 10-sample overfit gate: must accept the full block against the
   **same** W8A8 target. If this fails, stop and debug readout /
   layer ids / dtype before spending a night.
6. 400-1000 steps, export HF, remap `Qwen3DSparkModel`.
7. k sweep {3,4,7} x {greedy, probabilistic}. Gate vs W8A8 MTP3
   **and** vs RadixArk MTP3 41.2.

If off-shelf RadixArk accept on W8A8 is already within ~2% of
RadixArk-on-FP8 (satgeze-style "quant tax"), **skip the long
train** and only do a short 400-step polish. Only spend the night
if accept is in the 20% band rwmacy saw on a mismatched stack.

Served id must encode method: `qwen3.8-27b-W8A8-gptq-dspark7`.

### D. Kernels / XMX / dispatch (the differentiator)

Stay INT8. Every kernel has a microbench **and** an e2e serve A/B.

| item | why | dead-end if |
|---|---|---|
| Default-on small-M `int8_gemm_w8a16` for 3.8 long ctx | W8A16_M_MAX is 0 at 253k today; DSpark verify is M=k+1 | extra 9 GiB/card does not fit |
| Finish fusedq (`int8_gemm_w8a8_fusedq`) + regen `int8_gemm_kernel.patch` | 101 us quant is 10-35% of layer time | no e2e move |
| sycl-tla C1: VNNI16 + rectangular tiles, M<8 DPAS | makes k=7 verify cheap; arXiv 2508.06753 | isolated 1.2x, e2e c1 drops (like fused top-1) |
| Capture-persistent fake ops for DSpark verify shapes | V2 + int8 graph | Paris 0/3 |
| `B70_INT8_GRAPH_CLONE` on the DSpark path | int8 output invisible to captured AR | TP=2 GRAPH garbage |
| Fused GDN decode / ReplaySSM-on-XPU | NVIDIA's high-k GDN trick | CUDA-only; port is a season |
| INT4 XMX (W4A8) | later, after W8A8+DSpark is real | starting now |

Do **not** tune inner loops of a GEMM that is already 88-100% of
the 581 GB/s roofline (docs/kernel/23). The win is *shape and
fusion*, not a faster s8s8s32 for large M.

### E. Comms (TP=2 decode is ~43% collectives)

| try | tag |
|---|---|
| Push-AR / L0-IPC for **DSpark verify all-gather** (today decode stays on oneCCL) | DO THIS -- only remaining TP decode lever |
| Captured eager gather (phase-1 hard project) | later |
| oneCCL 2022 / newer BMG-aware, **outside** serve first | one A/B, reset between |
| Custom host-staged or L0-IPC ring | yes |
| `CCL_TOPO_P2P_ACCESS=1` in vLLM TP>1 | **DEAD-END, wedge, reboot.** Record if someone insists; never chain two tries. |
| Hardware P2P as fabric | not available (cross-die 1950X) |
| Steve's sharded target top-1 / persistent Markov | after matched draft exists |
| `xpu_shard_top1` default-on | already e2e-negative on NVFP4 (c1 48.9 -> 32.5). Re-A/B on W8A8 DSpark only |

### F. sglang-XPU DSpark (Phase 3)

This is the "newest sglang" prize, not a flag.

1. Port 3.8 W8A8 + NEXTN to 0.5.15 first (no DSpark).
2. Inventory CUDA DSpark: draft runner, confidence head, GDN verify,
   ReplaySSM. Map each op to XPU (sdpa / our int8 / triton-xpu).
3. Bring up `method=DSPARK` on XPU against the **same** matched
   draft as vLLM. Bit-exact accept vs vLLM on a 12-prompt suite.
4. Only then consider 0.5.17 + torch 2.13 (Phase 2 must be green).

Cookbook cells (fa3 / flashinfer / ReplaySSM) will not run. Write
our own XPU cell. That *is* the map pin.

### G. Prefill arm ("PSpark") -- Phase 4

Not a train-your-own DSpark twin.

Candidates, in order:

1. Prefix cache + chunked prefill (already on). Measure 262k TTFT
   on W8A8 before inventing anything.
2. **SpecPrefill** (training-free, prompt-token importance). vLLM
   #39060 stale; spike as a plugin. Dead-end if no XPU attn mask.
3. PFlash-style span scoring (llama.cpp). Port ideas, not the fork.
4. DFlash/DSpark **block draft of the prompt** -- only if 1-3 die
   and TTFT is still the user-visible pain. This would be a new
   train. Do not start it in Phase 0-1.

---

## 6. Test campaign (the part that makes dead-ends valuable)

Every GPU run: config -> command -> result -> verdict in JOURNAL.
New numbered scripts only (copy, do not rewrite). Served ids encode
method+scheme. Query `/v1/models` before trusting a number.

### Gate set (reuse everywhere)

- G0 identity: `/v1/models` matches the planned id
- G1 Paris exact + 17*23=391 + fib
- G2 vision landscape-vs-person (grafted tower)
- G3 `bench_code` c1 out=256 reps=3 (and c4 if PARALLEL allows)
- G4 accept table: pos0..k, accept_len, rate (vLLM metrics)
- G5 18/18 mixed prefill+decode, no bangs
- G6 HE+ 164 thinking-off (quality artifacts only)
- G7 190k needle if ctx >= 200k
- G8 after-TTFT conv99 (lab metric) when comparing to 49.7 / 43.8

Fail-closed: G1 or G5 fail => do not publish the speed number.

### Phase 0 -- week 1 (no ABI bump)

| # | experiment | success | likely dead-end |
|---|---|---|---|
| P0.1 | HE+ on W8A8-gptq MTP3 @131k | plus >= 0.90 | plus << Q4_K_M => SQ/AutoRound before DSpark |
| P0.2 | W8A8 @262k MTP-off, KV_FP8 A/B | native ctx + Paris | fp8 KV tanks plus => stay bf16 KV, cut ctx |
| P0.3 | W8A8 MTP3/5 @ longest ctx that fits | c1 > 26.62 | accept < 2.0 => graft/head bug |
| P0.4 | Off-shelf DSpark on W8A8, k=3/4/7 x greedy/prob | accept table | pos0 < 30% => train is mandatory (expected) |
| P0.5 | sglang 0.5.15 W8A8 3.8 NEXTN smoke | loads + Paris | shim/ABI/GDN => stay vLLM for DSpark |

### Phase 1 -- weeks 2-4

| # | experiment | success | likely dead-end |
|---|---|---|---|
| P1.1 | SpecForge XPU smoke (Linear + 1.36B step) | 1 step, no NaN | flex_attention / GRAPH=1 / anchors 512 |
| P1.2 | 10-sample overfit | full-block accept | readout / layer-id / dtype |
| P1.3 | 400-1000 step warm-start | pos0 >= 55% on W8A8 | data too small; scale prompts, do not change arch |
| P1.4 | k/sample sweep vs MTP3 | c1 >= MTP3 and G5 | 34-vs-41 again => kernel C, not more train |
| P1.5 | small-M w8a16 default-on | verify cheaper | OOM from layout clone |
| P1.6 | fusedq e2e | TTFT/PP up, HE+ flat | isolated only |
| P1.7 | L0-IPC / push-AR on verify gather | c1 up, no wedge | DEVICE_LOST => revert, reboot, packet |

### Phase 2 -- torch 2.13 (dedicated)

| # | experiment | success | likely dead-end |
|---|---|---|---|
| P2.1 | Rebuild all `.so` vs 2.13 | load in 0.27.1 **and** sglang-2.13 | topk SIGSEGV / empty nonzero |
| P2.2 | Paris + G5 on W8A8 no-spec | bit-match 0.26 within noise | graph fake-op miss |
| P2.3 | DSpark on 0.27.1 | >= Phase 1 c1 | quantized Markov only if it moves e2e |
| P2.4 | sglang 0.5.17 W8A8 NEXTN | >= 0.5.15 | already lost 0.5.15 vs 0.5.6 once |

### Phase 3-4 -- sglang DSpark + prefill

| # | experiment | success | likely dead-end |
|---|---|---|---|
| P3.1 | XPU DSpark bring-up vs same draft | accept bit-match vLLM | CUDA-only ops |
| P3.2 | ReplaySSM-on-XPU or reject | high-k cheaper | no XPU fold kernel |
| P4.1 | 262k TTFT baseline | number | -- |
| P4.2 | SpecPrefill spike | TTFT down, G1 hold | no XPU mask |

### Standing "try it, write the packet" list

These are allowed **once**, with reboot plan, then closed:

- P2P=1 in vLLM TP>1 (will wedge; only if a reviewer demands the
  7.1 retest). `I_KNOW_P2P_WEDGES=1`, never chain two tries.
- FATTN_MMA=1 on JIT (already crash-looped llama.cpp).
- method=dflash on v0.26 (unregistered).
- Adaptive verify on GDN (vLLM rejects).
- DeepSpec 38 TB cache.
- llm-scaler 0.21 / rmacy v10-slim as a vehicle.
- Enabling Q4K reorder-family on a *new* JIT without Paris-first
  (we got lucky on llama.cpp; do not assume).
- PCIe ASPM=performance (lab: kernel panic).
- Peer-pair comm mode 3 (lab: device-lost storm).

---

## 7. Risks and box law

- GPU lease: `gpu-run` for every real GPU touch. DD holds both cards
  until `daily_driver_serve.sh stop`.
- P2P=1 in vLLM TP>1 wedges the box. Recovery = reboot.
- oneCCL 2021.15 is broken; always 2021.17 overlay.
- Display-attached `xe` cannot `modprobe -r`. Box is now headless
  but keep the reboot reflex.
- V2 runner + thinking budget + mamba align-mode are landmines.
- Identity: never serve a bare `qwen3-14b-w8a8`-style id.
- ASCII in files, commits, terminal.
- Do not rewrite old `scripts/NN_*.sh`.

---

## 8. First two weeks (concrete)

Week 1 (DD stays up except the HE+/serve slots):

1. Add `qwen3.8-27b-W8A8-gptq` to `evals/configs/models.yaml`.
2. Stop DD, serve W8A8 MTP3, HE+ 164 + `bench_code` c1/c4, restore DD.
3. Write `vllm/dflash/serve_qwen38_w8a8_dspark.sh` (clone of M1,
   CKPT=w8a8-gptq). Off-shelf DSpark accept table. Restore DD.
4. Log P0.4 in JOURNAL + the loop ledger even if it is ugly.
   If it is a closed path, also packet it in
   `docs/20260818_qwen38_w8a8_dspark_deadends.md`.

Week 2:

5. If accept < 40% pos0: SpecForge XPU smoke + 10-sample overfit.
6. If accept is fine but c1 < MTP3: small-M w8a16 + verify-AR,
   not more training.
7. Selective SQ quant script (copy 150 -> new number), no GPU
   until week 1 quality is on disk.

Do not start Phase 2 or "PSpark" in week 1-2.

---

## 9. Pointers

- **Loop ledger (read every iteration):** `docs/20260818_qwen38_w8a8_dspark_loops.md`
- **Dead-end packets:** `docs/20260818_qwen38_w8a8_dspark_deadends.md`
- **Evidence:** `JOURNAL.md` newest-at-bottom; headings `2026-08-18` onward
- This box W8A8 3.8: `vllm/w8a8/serve_qwen38_27b.sh`, JOURNAL 2026-08-15b/c
- DSpark M1: `vllm/dflash/serve_qwen38_radixark_dspark.sh`, JOURNAL 17h
- Kernels: `kernels/README.md`, `kernels/SYCLTLA_SCAFFOLD.md`
- Faster-DD ancestor: `docs/20260703_faster_dd_plan.md` (DFlash/PFlash/C1)
- SpecForge: https://github.com/sgl-project/SpecForge (+ PR #769)
- DSpark paper: arXiv 2607.05147
- Lab DSpark7 (steal later): `b70-optimization-lab` V4-Flash packets
- P2P law: `docs/P2P_GPU.md`, CLAUDE.md
- SergiioB 3.8 vLLM XPU 4-mode: `docs/20260818_qwen38_sergiioB_cookbook.md`
  (source https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27/QWEN38-VLLM-XPU.md)
- 3.6 cookbook already measured here: `docs/COOKBOOK_CAMPAIGN.md`
- Steve 3.8 INT4-AR 100+: `docs/20260819_steve_qwen38_int4ar.md`
  (https://github.com/steveseguin/b70-optimization-lab HEAD `924b518`)
