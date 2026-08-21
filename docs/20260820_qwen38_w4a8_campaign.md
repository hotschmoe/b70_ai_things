# Qwen3.8-27B W4A8 FULL-SEND -- successor session standing prompt

**Created:** 2026-08-20
**Operator instruction:** paste this file path as the first thing the new
Grok session reads. Then follow section 3 (read order) before any GPU
touch. This is a **continual loop** campaign, not a one-shot plan.

ASCII only. Journal + commit + push each milestone. Do not rewrite old
numbered scripts; copy to a new number.

---

## PASTE BLOCK -- new session, first 60 seconds

You are starting the **Qwen3.8-27B W4A8 full-send** on 2x Arc Pro B70
(`b70s4dayz`, local `/mnt/vm_8tb/github/b70_ai_things`). Accuracy is
later. Extract every native Intel path. There is **no public 3.8 W4A8
to download**; we produce it.

1. Read this whole file, then `docs/20260820_qwen38_w4a8_loops.md`
   (NEXT PICK), then `docs/20260820_qwen38_w4a8_deadends.md`, then
   `docs/20260820_qwen38_w4a8_journal.md` bottom.
2. `./bin/gpu-run --status`. Both cards must be free or you wait.
   Do **not** start `vllm/daily_driver_serve.sh`. P2PACCESS=0.
3. **Day-1 parallel (the whole point of two cards):**
   - **Card 0:** `B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 env DATAFREE=1 bash scripts/151_quantize_qwen38_27b_w4a8.sh`
     (RTN W4A8+GDN-INT8 pipeline smoke). Then GPTQ (`DATAFREE=0`) as fire 2.
   - **Card 1:** kernel matrix + whacky ideas that do **not** need the
     3.8 W4A8 file (section 8.2). Use 3.8 *shapes* and the 3.6
     `w4a8-sqgptq` checkpoint as a stand-in.
4. One loop = one verdict. Write JOURNAL + ledger before picking again.
5. Fresh containers come **after** the first isolated kernel number and
   a loadable 3.8 artifact, not before. Do not boil the ocean on bake day.

Standing holds at write: Ornith NVFP4 GRAPH 34.9, W8A8 k1bar **31.9**,
W1 GPTQ-Int4+draft 65.08, llama.cpp Q4_K_M 43.8, Pliny Q8 32.03.
W8A8-gptq HE+ 0.957/0.927. Hot decode bytes on that file: 30.2 GiB
(58% INT8, 42% BF16). No-spec TP=2 18.45 = 49% of 37.5 DRAM roof.

A-E in `docs/20260820_b70_bw_campaigns.md` are **parked**. This window
jumps campaign F and **absorbs** A (GDN ints), B (fused quant), D
(VNNI16), E (DSpark) as W4A8-internal loops. Do not run A-E as a
separate serial stack while this campaign owns the cards.

---

## 1. Living header (update every loop; nothing above section 0)

| field | value |
|---|---|
| Last loop | 14 (bench_code c1 25.0 on GRAPH=1 GPTQ) |
| Last JOURNAL heading | `2026-08-21n` in `docs/20260820_qwen38_w4a8_journal.md` |
| Campaign journal | `docs/20260820_qwen38_w4a8_journal.md` |
| Loop ledger | `docs/20260820_qwen38_w4a8_loops.md` |
| Dead-ends | `docs/20260820_qwen38_w4a8_deadends.md` |
| 30m loop | ARMED `01a021be5649` |
| Next pick | Path H `B70_W4A8_HYBRID=1` GRAPH=1 A/B on GPTQ (stop `:18082` first). Do not demote 31.9. |
| Blocked on | nothing. GRAPH=1 GPTQ Up `:18082`. RTN GRAPH=0 `:18081`. |
| 3.8 W4A8 artifact | RTN :18081. GPTQ :18082 GRAPH=1. |
| Best W4A8 isolated | Path H w4a16 M=1 down_proj 0.079 ms / 565 GB/s / 97% of 581. |
| Best W4A8 e2e 3.8 | GRAPH=1 TP=1 GPTQ `bench_code` c1 **25.0** avg=best. Wall 128-gen was 24.5. GRAPH=0 ~6.3. |
| DSpark INT | not trained |
| DD | PARKED. `:18080` is research. |

---

## 2. Mission (one paragraph)

Make 2x Intel Arc Pro B70 the place people point at for **private
Qwen3.8-27B at 4-bit storage + integer compute**. Scheme is **W4A8**:
int4 weights in HBM, unpack-in-register to int8, Xe2 XMX **DPAS s8xs8**
(the Intel steal of NVIDIA's "4-bit file, tensor-core verify" shape).
Accuracy is a later campaign. Native Intel paths only: oneDNN int4/int8
ops, ESIMD DPAS, VNNI16 prepack, XPUGraph, PUSH_AR, GDN SYCL, DSpark
verify at M=2..8. If a DSpark **INT** drafter beats the FP8-trained
one on e2e tok/s, it becomes the default. Fresh research containers
for vLLM, sglang, and a llama.cpp SYCL kernel lab -- not three copies
of the same broken mount recipe.

North-star number to beat, same box, same `bench_code` c1: **65.08**
(W1 GPTQ-Int4+draft) and **43.8** (Q4_K_M). Spec roof is the prize;
no-spec % of DRAM roof is the honesty metric.

---

## 3. Read order (before any edit or GPU)

1. This file (living header + sections 8, 10, 13).
2. `docs/20260820_qwen38_w4a8_loops.md` -- last 3 loops + NEXT PICK.
3. `docs/20260820_qwen38_w4a8_deadends.md` -- do not retry a closed packet.
4. `docs/20260820_qwen38_w4a8_journal.md` **bottom**. Root JOURNAL.md
   is a one-line pointer only.
5. Skim, do not re-derive:
   - `sglang/W4A8_PLAN.md` + `sglang/W4A8_BUILD.md` (3.6 hybrid, ABI,
     calling convention).
   - `research/w4a8/AUTOROUND_W4A8_FEASIBILITY.md` (AutoRound cannot
     export W4A8).
   - `docs/kernel/23_b70_gemv_gemm_roofline.md` (581 GB/s, 101 us quant).
   - `vllm/nvfp4/INT4_DPAS_PIONEER.md` (native s4 DPAS 2.0x s8 MAC).
   - `kernels/SYCLTLA_SCAFFOLD.md` (VNNI16 / arXiv:2508.06753).
   - `docs/20260818_qwen38_w8a8_dspark_campaign.md` section on SpecForge
     XPU train (constraints only).
6. `./bin/gpu-run --status` and `curl -s http://192.168.10.5:18080/v1/models`.
7. Then pick **one** loop. Card-0 quant + card-1 kernels may run at the
   same time because they are two leases, but each card still has one
   pick.

---

## 4. Why W4A8 is the Intel steal (and what it is not)

Xe2 / Battlemage XMX = **INT8 / INT4 / INT2 + FP16 / BF16**. No FP8
XMX, no FP4 XMX. NVIDIA NVFP4 is a Blackwell file. On this box it is
`nvfp4_gemm_w4a16` (E2M1 decompress into a W4A16 matmul). Stealing the
*file* was already done (Unsloth / RadixArk / Ornith). Stealing the
*shape* means: fewer bytes in HBM + an integer GEMV/GEMM DPAS can eat.

| scheme | storage | acts | kernel we have | XMX DPAS? |
|---|---|---|---|---|
| W8A8 | int8 | int8 | `int8_gemm_w8a8` / `w8a16` | YES s8xs8 / s8xf16 |
| **W4A8** | **int4** | **int8** | **`int4_gemm_w4a8`** | **YES if unpack int4->int8 in register, then DPAS s8xs8** |
| W4A16 | int4 | bf16/fp16 | `int4_gemm_w4a16` | NO. Dequant to BF16, then BF16 GEMM |
| NVFP4 | E2M1 | usually f16 | `nvfp4_gemm_w4a16` | NO |
| NVFP4->int8 | E2M1 unpacked to s8 | int8 | `nvfp4_gemm_w4a8` (research) | YES, but still an NVFP4 file |
| W4A4 int | int4 | int4 | ESIMD proto only | YES s4xs4, 2.0x s8 MAC, accuracy later |

The "funky W4A16 translated to int8xint8" trick **is W4A8**. Do not
unpack int4 and *store* int8 (that is W8A8, 2x the bytes). Do not
leave acts in BF16 (W4A16).

**Best-for-Intel (BW first, accuracy later):**
W4A8 on MLP+attn (+ GDN Linears once 151 lands). Group 128. K is
already %32 (5120, 17408). Avoid N=24-class tiles (that is why NVFP4
W4A16 died on Marlin `size_n=24`). DPAS wants M tile 8/16; decode M=1
will stay the weak citizen; DSpark verify at M=2..8 is where XMX
actually shows up. VNNI16 prepack is loop K5.

### Three paths (name them in every ledger line)

- **Path H -- hybrid (3.6 proven).** Decode M==1 -> `int4_gemm_w4a16`
  (fp16 act, no act-quant). Prefill M>1 -> `int4_gemm_w4a8` (int8 act).
  sglang 3.6 GRAPH: 27.3 tok/s, HE+ 0.921/0.896 == woqgemm. This is
  the **baseline to beat**, not the end of the campaign.
- **Path X -- native XMX at all M.** Fused unpack + per-token int8
  quant + DPAS s8xs8 in one launch. Prize: M=1 still reads 0.5
  byte/weight AND does not pay the 101 us quant tax. If Path X loses
  M=1 to Path H, keep Path H for decode and use Path X for
  prefill / DSpark verify / c>1.
- **Path S -- native s4 DPAS.** Prefill TOPS showcase
  (`proto_int4/`, 2.0x s8 MAC). Not a serve until rotation exists.
  Measure it on card 1 so we know the ceiling.

---

## 5. Hardware map -- extract EVERY unit

Loop catalog (section 10) must eventually have a number for each row.

| unit | what it wants | 3.8 decode today | W4A8 play |
|---|---|---|---|
| HBM 608 GB/s/card (spec), ~581 measured | fewer bytes, high %roof | W8A8 30.2 GiB hot, 49% of 37.5 roof | int4 MLP+attn (~0.5x) + GDN INT8 (~0.5x of 10.4 GiB) |
| XMX DPAS s8xs8 | M tile 8/16, K%32, packed VNNI | almost idle at M=1 | prefill, DSpark verify M=2..8, c>1 |
| XMX DPAS s8 x f16 (`w8a16`) | no act-quant | W8A8 decode near roof | Path H cousin; useful if fused W4A8 loses M=1 |
| XMX DPAS s4xs4 | same tiles, 2x MAC | unused in serve | Path S microbench; not the 3.8 serve |
| VNNI16 prepack | reuse dequant weight ~8x | unused | K5; sycl-tla scaffold exists |
| EU / ALU | GDN conv1d k=4, RMSNorm, RoPE, act-quant reduce | 101 us serial reduce on K=17408 | fusedq (K6); keep conv BF16 |
| XPUGraph / PIECEWISE | captured decode | W8A8 GRAPH k1bar 31.9 | Path H is capturable (no data-dep quant at M=1); Path X must register_fake |
| Triton GDN / FA | already the 3.8 stack | works | do not regress; autotune warps later |
| oneDNN matmul | `int4_gemm_w4a{8,16}`, `int8_gemm_w8a{8,16}` | W8A8 path | first Path X vehicle |
| ESIMD `xmx::dpas` | native s4/s8/s2 | proto only | card-1 whacky; `INT4_DPAS_PIONEER.md` |
| SYCL joint_matrix | woqgemm int8 compute | gated "no matrix hardware" on oneAPI 2025.3 | re-probe once; if still dead, packet it |
| L0-IPC PUSH_AR | ~11 GB/s, P2PACCESS=0 | W8A8 production | TP=2 later; do not enable vLLM P2PACCESS=1 |
| DSpark verify | M=k+1 in {3,5,8} | FP8-trained draft, accept 2.45 | the XMX window; INT drafter is K17 |

Display-attached box: `xe` also drives the framebuffer. Cards may still
clock-skew. Pin card 0 for the long quant (script 150/151 default).
Measure clocks with `xpu-smi` / `intel_gpu_top` before calling a card
"slow". RESEARCH_TODO's "HEADLESS / cards symmetric" note is about the
DD being down, not about the display disappearing.

---

## 6. Model, shapes, byte budget

Qwen3.8-27B = `Qwen3_5ForConditionalGeneration`, same hybrid as 3.6:
64 layers, `full_attention_interval=4` -> **16 full-attn + 48 GDN**.
hidden 5120, intermediate 17408, head_dim 256, **24 q-heads**, 4 kv-heads,
`attn_output_gate=true`, vocab 248320, ctx 262144, MTP 1 layer.

**K%32 is already true** (5120, 17408). N for the fat GEMMs is also
friendly (5120, 6144, 8192, 16384, 17408, 34816). The traps:

- **Do not tile per-head with N=24** (24 q-heads). Marlin-class
  `size_n=24` is how NVFP4 W4A16 died. q_proj N = 24*256 = **6144**.
- **GDN `in_proj_ba` is N=96** (5120->96; 48/card at TP=2). That is
  the N=24-class cousin. Keep it BF16. 151 already ignores
  `in_proj_a` / `in_proj_b`.
- GDN fat projections: `in_proj_qkvz` 5120->16384, `out_proj`
  6144->5120. 151 sends those to **W8A8 channelwise**, not int4,
  because vLLM fuses `in_proj_qkvz=[qkv,z]` and we want a homogeneous
  int8 pair on a proven kernel (`XPUInt8ScaledMMLinearKernel`). A later
  loop may try GDN as int4 (true W4A8-all-Linear). First artifact is
  **W4A8 MLP+attn + W8A8 GDN Linears**.

### EXPECTED hot-byte budget (not measured on 3.8 W4A8 yet)

W8A8-gptq 3.8 census (prior session): 30.2 GiB hot, 58% INT8 / 42% BF16.
GDN `linear_attn` Linears = **10.36 GiB BF16**.

| piece | W8A8-gptq today | 151 first artifact (EXPECTED) |
|---|---|---|
| MLP + full-attn Linears | INT8 ~17.5 GiB | INT4 ~8.8 GiB |
| GDN in_proj_qkv / z / out_proj | BF16 10.36 GiB | INT8 ~5.2 GiB |
| GDN conv / A_log / dt_bias / in_proj_ba / norms | BF16/FP32 | unchanged |
| lm_head | BF16 ~2.54 GiB | BF16 first; INT4 is K8 |
| MTP + visual | BF16, not decode-hot the same way | ignored (grafted) |

No-spec DRAM roof at TP=2, 100% of 1216 GB/s, if hot bytes land ~18 GiB:
1216e9 / 18e9 ~ **67 tok/s**. If we also INT4 the GDN Linears and
lm_head, hot ~12-14 GiB -> **87-101 tok/s** roof. Realistic GEMV is
60-80% of that. Spec multiplies by accept_len if verify stays BW-bound
(weight walked once per step). **Census the 151 artifact before quoting
any of these as fact.**

3.6 W4A8-sqgptq (MLP-only, GDN BF16) was 25.8 GiB and could not hold
KV at 131k. That is why 151 exists. Do not serve a 3.8 clone of that
ignore list.

---

## 7. What already exists (do not rediscover)

### Weights on disk

- `models/files/qwen3.8-27b/bf16` -- source for 151. COMPLETE.
- `models/files/qwen3.8-27b/w8a8-gptq` -- HE+ 0.957/0.927. Sibling for
  preprocessor graft.
- `models/files/qwen3.8-27b/int4-autoround` -- W4A16, Steve's stack.
  GRAPH=1 hung (D16). Not W4A8.
- `models/files/qwen3.6-27b/w4a8-sqgptq` -- 3.6 CT W4A8, GDN still
  BF16, prepacked. **Card-1 stand-in.**
- Drafters: `qwen3.8-27b/dflash-drafter-fp8-b70` and
  `dflash-drafter-radixark`. FP8-trained. Accept on W8A8 ~2.45 greedy.
- **No `qwen3.8-27b/w4a8-*` directory.** HF search found none worth
  downloading. AutoRound cannot emit W4A8 (`bits==8` hard assert).

### Kernels / images / shims

- Shared source: `kernels/` (`int8_gemm_w8a8.h`, `w8a16`, fusedq notes,
  `nvfp4_gemm_w4a8.h`). `int4_gemm_w4a8` / `w4a16` are **upstream** in
  vllm-xpu-kernels (`XPU_SPECIFIC_KERNELS_ENABLED=ON`).
- Built .so (git-ignored): `/mnt/vm_8tb/b70/w4a8_kernel/_xpu_C.abi3.so`
  (sglang torch 2.12), `/mnt/vm_8tb/b70/w8a8_kernel_v0240/_xpu_C.abi3.so`
  (vLLM 2.12, has int4 ops + GDN).
- Images: `vllm-xpu-env:int8g-v0260`, `sglang-xpu:woq` / `:mtp` /
  `:woq-0515` / `:mtp-0515`.
- sglang shims: `sglang/patches/w4a8_shim.py`, `woq_shim.py`,
  `w4a8_actquant_triton.py`. Triton AQ: do **not** torch.compile it
  (inductor deadlock). `B70_XPU_W4A8=1`.
- vLLM: `XPUW4A8IntLinearKernel` is **upstream** on 0.25.1/0.26.0.
  Shelf: `rdy_to_serve/vllm/qwen36-27b-w4a8/` (prepack patch + optional
  `B70_W4A8_HYBRID`). 0.25.1 port unverified on GPU.
- sglang shelf: `rdy_to_serve/sglang/qwen36-27b-w4a8/` -- 3.6 Lorbus
  int4-AutoRound served as Path H, 27.3 tok/s.
- Calling convention (CONFIRMED, `sglang/W4A8_BUILD.md`):
  ```
  int4_gemm_w4a8(A_int8[M,K], A_scale[M,1] fp16, A_zp[M,1] i32,
                 B[K/8,N] i32 NT-view, B_scale[K/g,N], B_zp=tensor([8]),
                 group_size, g_idx=None, bias=None) -> fp16 [M,N]
  B MUST be NT (stride[0]==1). Do NOT .contiguous() the view away.
  Serve --dtype float16 (op emits fp16).
  ```
- Native s4 DPAS proto: `vllm/nvfp4/proto_int4/` (bit-exact, 2.0x).
- VNNI16 scaffold: `kernels/SYCLTLA_SCAFFOLD.md`, tree at
  `/mnt/vm_8tb/b70/sycl-tla`.
- Producer: `scripts/151_quantize_qwen38_27b_w4a8.sh` (copy of 149,
  3.8 paths, `int8g-v0260`, default DATAFREE=1 ->
  `models/files/qwen3.8-27b/w4a8-rtn-gdn`).
- Decode GEMV harness (W8A8, extend it): 
  `research/w8a8/decode_gemv/bench_decode_gemv.py`.
- DSpark serve: `vllm/dflash/serve_qwen38_w8a8_dspark.sh`. SpecForge
  XPU train: anchors<=64, chunk<=16, XPU_GRAPH=0, workers=0, CPU offload.

### llama.cpp honesty

llama.cpp SYCL is **weight-only GGUF + fp16 compute**. It will not
consume compressed-tensors W4A8 or our `_xpu_C` ops. Its job in this
campaign is (a) Q4_K_M 43.8 as the BW/control, (b) a SYCL GEMV / MMQ
playground for ideas we steal back into Path X, (c) **not** a W4A8
serve unless someone writes an int8-act GGML path (that is a named
whacky loop, not day 1).

---

## 8. Dual-card day-1 (do this before baking images)

`gpu-run --card N` locks one card. Default `gpu-run` locks **both**.
Quant must be `--card 0` with `ZE_AFFINITY_MASK=0`. Kernels `--card 1`
with `ZE_AFFINITY_MASK=1`. `B70_GPU_LOCK_TIMEOUT=0` for the long quant
(default 600s will abort the wait, not the job, if the lock is busy).

### 8.1 Card 0 -- produce the 3.8 W4A8 file

Fire 1 (pipeline smoke, accuracy-be-damned, **hours not a day**):

```
cd /mnt/vm_8tb/github/b70_ai_things
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 \
  env DATAFREE=1 CARD=0 \
  bash scripts/151_quantize_qwen38_27b_w4a8.sh
```

Expect: `models/files/qwen3.8-27b/w4a8-rtn-gdn`. Stage A on GPU, stage B
CPU prepack. Log under `results/logs/151_qwen38_w4a8_*.log`.

Fire 2 (calibrated, start as soon as fire 1 is load-gated OR in
parallel if you accept RTN as the kernel-serve stand-in):

```
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 \
  env DATAFREE=0 METHOD=gptq SMOOTHQUANT=selective SAMPLES=128 CARD=0 \
  bash scripts/151_quantize_qwen38_27b_w4a8.sh
```

Expect: `models/files/qwen3.8-27b/w4a8-gptq-gdn`. GPTQ SequentialPipeline
onloads one layer at a time; it cannot usefully split across two cards.

**Census before any speed claim:** tensor dtype/bytes by category
(`mlp`, `self_attn`, `linear_attn.in_proj_qkv|z|out_proj`,
`linear_attn` other, `lm_head`, `mtp`, `visual`). Fail closed if GDN
fat projections are still BF16 or if a blanket `re:.*linear_attn.*`
ignore survived.

**First serve (after census):** clone
`rdy_to_serve/vllm/qwen36-27b-w4a8/serve.sh` to a 3.8 research script
under `vllm/w4a8/` (not a shelf). `DTYPE=float16`, GRAPH=0 first,
Paris + fib, then GRAPH=1. Served id
`qwen3.8-27b-W4A8-rtn-gdn` / `...-W4A8-gptq-gdn`. Never a bare
`qwen3-27b-w4a8`.

If 151's two-group recipe throws on 3.8 names, fail closed and emit a
DATAFREE W4A8 **with** the old blanket GDN ignore as a backup artifact
(`w4a8-rtn-mlp`) so card-1 e2e is not blocked. Log it as a dead-end
packet on the GDN group, not as success.

### 8.2 Card 1 -- kernels while 151 runs (no 3.8 W4A8 required)

Goal of day-1 card 1: a **table** of ms / GB/s / INT8-TOPS for 3.8
shapes, M in `{1,2,4,8,16,32,64,256,2048}`, kernels:

1. bf16 `F.linear` (roof control)
2. `int4_gemm_w4a16` (Path H decode)
3. `int4_gemm_w4a8` + eager act-quant (Path X unfused)
4. `int4_gemm_w4a8` op-only (quant excluded -- honesty split)
5. `int8_gemm_w8a8` / `w8a16` / fusedq if the .so has them
6. native ESIMD s4 DPAS tile (`proto_int4/run_bench.sh`) at one fat
   shape if it fits the lease

Shapes (full-card, TP=1):

| name | K | N | notes |
|---|---|---|---|
| gate_up | 5120 | 34816 | fat MLP |
| down_proj | 17408 | 5120 | 101 us quant lives here |
| qkv+gate | 5120 | 14336 | 16 full-attn layers |
| o_proj | 6144 | 5120 | row-parallel at TP=2 is 3072 |
| gdn_qkvz | 5120 | 16384 | today BF16; 151 -> INT8 |
| gdn_out | 6144 | 5120 | |
| gdn_ba | 5120 | 96 | trap; do not XMX-hero this |
| lm_head | 5120 | 248320 | K8; maybe later |

Use image `vllm-xpu-env:int8g-v0260` or `sglang-xpu:woq` with
`B70_XPU_C_SO` set. Starting points:

- `sglang/int4_gemm_w4a8_probe.py`
- `sglang/w4a8_from_woq_probe.py`
- `research/w8a8/decode_gemv/bench_decode_gemv.py` (extend, do not
  rewrite; copy to `vllm/w4a8/bench_w4a8_shapes.py`)
- `vllm/nvfp4/proto_int4/run_bench.sh` (card 1, `ZE_AFFINITY_MASK=1`)

**Whacky ideas queue (card 1, when the matrix has one row):**

- Pad M=1 -> M=8 dummy rows to fill DPAS. Isolated only. e2e only if
  GB/s wins after counting the wasted FLOPs.
- VNNI16 prepack vs current NT int32 packing (K5).
- `woqgemm(..., compute_type=int8)` joint_matrix re-probe on runtime
  26.22. One shot. If still "no matrix hardware", dead-end packet.
- oneDNN verbose dump (`ONEDNN_VERBOSE=1`) on `int4_gemm_w4a8` at
  M=1 vs M=8: are we even in DPAS?
- Block-scale integer analog of NVFP4 (group-16 INT scales) using
  `kernels/nvfp4_gemm_w4a8.h` ideas on **synthetic int4**, not E2M1.
- Persistent kernel / software pipelined weight prefetch vs GDN conv.
- Dual-issue hypothesis: conv1d on EU, GEMM on XMX, same layer.
- llama.cpp SYCL MMQ / IQ4_XS kernel read: steal tile sizes, do not
  serve GGUF as W4A8.
- N-pad 96 -> 128 on `in_proj_ba` just to see if oneDNN stops
  falling off a cliff (not a serve change).

Write every whacky idea as `## LOOP N` with GO/NO-GO. Isolated >=1.10x
before any e2e.

---

## 9. Fresh containers (after first numbers, not instead of them)

Do **not** start here. ABI is why we have three backends.

Policy: **one shared kernel SOURCE** (`kernels/` + upstream
vllm-xpu-kernels onednn int4). **Per-backend ABI-specific .so**.
Research tags are mutable; shelf tags are sweep-gated and immutable.

### 9.1 vLLM research image -- `vllm-xpu-env:w4a8-v0260`

Start from `vllm-xpu-env:int8g-v0260` (`vllm/images/int8g/bake_v0260.sh`).
Bake, do not mount:

- `_xpu_C.abi3.so` with `int4_gemm_w4a{8,16}` + `int8_gemm_w8a{8,16}`
  + fusedq + GDN (`vllm/build_v0240_int8gdn_fusedq_so.sh` as the
  pattern; copy to a v0260 number, do not rewrite).
- `register_fake` for PIECEWISE on every new op.
- `compressed_tensors_w4a8_int.py` prepack skip
  (`VLLM_W4A8_PREPACKED`).
- Optional `B70_W4A8_HYBRID` Path H route.
- DSpark readout patches from `vllm/dflash/patches/v0260/`.

Do not rebase to torch 2.13 in this campaign unless a loop is *only*
"does 2.13 buy Path X". Current lock is torch 2.12 / int8g-v0260.

### 9.2 sglang research image -- `sglang-xpu:w4a8`

Start from `sglang-xpu:mtp` (vision + XPUGraph) or `:woq` (int4
woqgemm). Bake `sglang/W4A8_BUILD.md` `_xpu_C.abi3.so` +
`w4a8_shim.py` + Triton AQ + `woq_shim.py`. PREPEND
`/opt/intel/oneapi/compiler/2025.3/lib` on `LD_LIBRARY_PATH` or torch
loses the XPU device. Image default: `B70_XPU_W4A8=1`.

Prefer **sglang as the long-term serve** (AGENTS.md). vLLM is the
kernel/DSpark vehicle this window because DSpark + fusedq live there.
Run both to the same 3.8 artifact. Do not pick a winner until c1 +
coherence + TTFT exist on both.

### 9.3 llama.cpp lab -- `llamacpp-sycl:w4a8-lab` (optional)

Not a W4A8 server. Clone `llamacpp/build_sycl.sh` image, add a
standalone SYCL GEMV microbench binary that uses the same 3.8 shapes.
Steal tiles. Feed Path X. Q4_K_M 43.8 remains the control serve, not
this image.

### 9.4 What "best-in-class W4A8 container" means

A container that, on a cold box:

1. Loads `qwen3.8-27b-W4A8-gptq-gdn` without 28 GiB unpack transients.
2. Dispatches MLP/attn to `int4_gemm_w4a8` (M>1) and the winning
   decode op (H or X).
3. Dispatches GDN fat Linears to INT8 XMX.
4. Captures PIECEWISE/XPUGraph without fake-op holes.
5. Keeps vision + MTP tensors (even if MTP accept is later).
6. Has a `serve.sh` that preflights the .so, oneAPI lib path, and
   census.
7. Passes G1 concurrent prefill+decode (the "!!!!" class).

Shelf promotion (`rdy_to_serve/<backend>/qwen38-27b-w4a8/`) only after
`bin/serve-sweep --smoke` AND faster-or-equal + coherent vs the
current 3.8 holds. Untested "improvement" does not land. Exactly one
shelf config per (backend, model, quant).

---

## 10. Relentless loop catalog (K0-K19)

One K per loop unless the ledger says two cards, two Ks. Do not mix
a kernel rewrite and a GPTQ rerun in one verdict.

| id | loop | metric | done when |
|---|---|---|---|
| **K0** | Census + dispatch proof on 151 artifact | dtype table; `torch.ops._xpu_C.int4_gemm_w4a8` actually called | no silent BF16 fallback on MLP |
| **K1** | Isolated kernel matrix (section 8.2) | GB/s and TOPS vs 581 / ~250 INT8 TOPS | M=1 and M=8 rows for gate_up + down |
| **K2** | Unpack-in-register Path X (oneDNN first) | M=1 full (quant+gemm) vs Path H | Path X >= Path H or a split-M rule |
| **K3** | Hybrid Path H e2e on 3.8 | `bench_code` c1 GRAPH=0 then GRAPH=1 | number vs 31.9 / 43.8 / 65.08 |
| **K4** | DSpark / MTP verify M=2..8 XMX | isolated GEMM at M=4,8; then e2e accept | XMX visible in ONEDNN_VERBOSE or TOPS |
| **K5** | VNNI16 prepack (arXiv:2508.06753, sycl-tla) | isolated M=1,4,8 vs K1 | >=1.10x isolated or packet |
| **K6** | Fuse 101 us act-quant (`fusedq` / Triton AQ) | down_proj M=1 layer us | 101 us gone or fused into matmul |
| **K7** | GDN Linears: INT8 (151) then optional INT4 | census + c1 | 10.36 GiB not BF16 |
| **K8** | lm_head INT4 g32 via `int4_gemm_w4a16` | c1; greedy Paris | 3.6 held HE+ at this lever |
| **K9** | XPUGraph / MRV2 / launch | tpot, not just kernel BW | one A/B |
| **K10** | Prefill large-M s8 DPAS | TTFT / PP tok/s @ 2k and 8k | beat 3.6 hybrid +24% PP as a floor |
| **K11** | Path S s4xs4 TOPS (no serve) | TOPS vs 2.0x s8 atom | logged; W4A4 serve still later |
| **K12** | Tile / N-trap audit | N=96, N=24 pad, N%16 | no Marlin-class tile in Path X |
| **K13** | Group 32 vs 64 vs 128 (kernel, then one requant) | isolated then HE+ | 128 stays default unless kernel wins |
| **K14** | KV: keep auto/bf16 first; INT8 KV only if c1 lives | retrieval + c1 | do not repeat D18 emul+fp8-KV garbage |
| **K15** | TP=2 PUSH_AR, P2PACCESS=0 | c1 vs 1x; GB/s aggregate | never vLLM P2PACCESS=1 |
| **K16** | Concurrent c=2,4,8 (XMX M>1) | agg tok/s + G1 | Path X's real niche vs Path H |
| **K17** | DSpark INT vs FP (section 11) | e2e c1, not isolated accept | INT wins or FP stays |
| **K18** | Accuracy recovery (section 12) | HE+ vs 0.957/0.927 | **after** a speed number exists |
| **K19** | Bake `vllm-xpu-env:w4a8-v0260` + `sglang-xpu:w4a8` | cold-box serve.sh | smoke + coherence; shelf only if gated |

### Split-M rule (write this on the Path X vs H verdict)

If M=1 Path X < Path H: **decode stays H, verify/prefill/c>1 stays X**.
That is how 3.6 already won. The campaign fails only if Path X also
loses at M=8 and at prefill -- that would mean int4-unpack is
fundamentally not feeding DPAS.

---

## 11. DSpark INT variant

Today: FP8-trained 1.36B drafter on an INT8 (and soon W4A8) target.
Greedy accept ~2.45, pos0 ~0.62. RadixArk-on-FP8 card saw ~3.3. The
gap is **hidden-state mismatch**, not "INT is slower."

Two INT experiments, in order:

1. **Matched FP drafter, W4A8 hiddens.** SpecForge `strategy: dspark`
   offline, PR #769 XPU constraints (anchors<=64, chunk<=16,
   XPU_GRAPH=0, workers=0, CPU offload). 10-sample overfit gate
   before a night run. Same readout patches as M1
   (`vllm/dflash/DSPARK_RMACY.md`). This is "accept ~3" on a 4-bit
   body. Drafter stays BF16/FP8 *weights*; only the *training target*
   is W4A8.
2. **INT8 (or INT4) drafter weights.** The drafter is 1.36B and fits
   easily. If verify GEMMs are the leftover after a W4A8 target walk,
   an INT drafter can cut draft time. Train as (1) then GPTQ W8A8 the
   draft, **or** train already-quantized if SpecForge allows. Gate is
   **e2e `bench_code` c1**, not accept_len. A faster draft at 2.2
   accept can beat a slower draft at 3.0.

DSpark verify is **the** M=2..8 XMX window. Even if Path X loses
decode M=1, DSpark can still be why W4A8 beats Q4_K_M.

sglang DSpark is still CUDA/Spark in upstream notes. Do not block
vLLM DSpark on an sglang port. If an sglang draft path appears, A/B it
as its own loop.

Do not start a night train until K0 census is green and a GRAPH=0
W4A8 serve produces hidden states. Use the 10-sample overfit first.

---

## 12. Accuracy later (do not sneak it into K1-K11)

Operator: "accuracy be damned for now." Honor that. HE+ is **K18**.

When K18 opens:

- Bar: W8A8-gptq 0.957/0.927 and W4A16-autoround 0.963/0.915.
- 14B W4A8-RTN was 0.817 and was dominated. GPTQ+selective-SQ is the
  first recovery. Rotation (QServe/SpinQuant R1/R2 offline) is the
  second. Online Hadamard is kernel-gated (Table D,
  `docs/quant_methods.md`) -- do not start it to "fix" K1.
- GDN INT8 may cost more than GDN BF16. If G1 bangs or greedy
  degenerates, split: serve MLP W4A8 + GDN BF16 as a named artifact,
  do not silently ignore.
- Thinking-off greedy for HE+ (same as 3.8 W8A8).

W4A4 integer accuracy is **known broken** without rotation (cosine
0.796, SNR 4.1 dB on a 3.6 gate slice). Path S is TOPS only.

---

## 13. Loop protocol (copy of the W8A8 campaign, tighter)

A loop is **one verdict**, not one phase.

Allowed: one K-row, or card0-quant + card1-K1 in parallel (two
verdicts, two ledger blocks), or a no-GPU unblock (yaml id, serve
script clone).

Not allowed: "bake all three images first", "rewrite the campaign",
"improve the plan a bit", "start DD", "enable P2PACCESS=1 to see".

Stop when you have config -> command -> result -> verdict, **or**
you are blocked on lease / health / a missing artifact. Long GPU jobs:
start under `gpu-run`, write LOOP-STARTED with log path and pid.

### Where to write

| dest | what |
|---|---|
| `docs/20260820_qwen38_w4a8_journal.md` bottom | full CONTEXT / CONFIG / COMMAND / RESULT / VERDICT |
| JOURNAL.md bottom | one-line pointer: `### YYYY-MM-DD<letter> - LOOP N: see docs/20260820_qwen38_w4a8_journal.md` |
| `docs/20260820_qwen38_w4a8_loops.md` | `## LOOP N` block below |
| `docs/20260820_qwen38_w4a8_deadends.md` | only if a path is closed |
| this file, living header only | bump Last loop / Next pick / scores / 30m loop id |
| RESEARCH_TODO.md campaign blurb | one line if Next pick or a north-star number moved |

### 30m armed loop

A `/loop 30m` (scheduler, durable, fire-immediately then every 30m)
owns this campaign while the operator leaves it armed. One fire = one
verdict or a LOOP-STARTED handoff. If a card is HELD by this campaign,
ATTACH (read the log; do not start a second job on that card). If GPU
work will exceed 30m, start under `gpu-run`, write LOOP-STARTED with
log path and pid, commit+push, return. Commit and push at every
verdict or LOOP-STARTED. Do not sit idle for 25 minutes. Foreign
lease -> CPU-only. DD PARKED. P2PACCESS=0.

Do **not** rewrite sections 4-12 of this file to narrate a loop.

### Ledger shape

```
## LOOP N -- YYYY-MM-DDThhmmZ -- <one-line pick>

Picked: K? / unblock / packet
Why this, not the other open row: <one sentence>
GPU: card0 / card1 / DD stopped? / port / served id
Command: <actual command>
Log: <path>
Result: <number or error>
Verdict: GO / NO-GO / BLOCKED / DEAD-END
Changed beliefs: <what a future loop must not re-discover>
Next pick: <exact>
Do not: <tempting wrong follow-up>
Restore: DD stays down. xpu-health? lock released?
JOURNAL: ### YYYY-MM-DD<letter>
```

---

## 14. Standing NEVER / fail-closed

- Do not start `daily_driver_serve.sh`. DD stays parked.
- Do not set `CCL_TOPO_P2P_ACCESS=1` in a vLLM TP>1 serve.
- Do not rewrite `scripts/49`, `149`, `150`. 151 is the 3.8 copy.
- Do not treat NVFP4 HF cards as INT4 XMX.
- Do not serve a bare id `qwen3-27b-w4a8`.
- Do not torch.compile the sglang act-quant (hangs). Triton only.
- Do not reuse `research/w4a8/offline_prepack_w4a8.py` on the GDN INT8
  group (it packs on `.weight_scale` presence and will corrupt I8).
- Do not chain TP=2 worker-init crashes. `bin/xpu-health` between.
- Do not promote a shelf entry without smoke + coherence + a measured
  win.
- Do not start W4A4 serve. Path S is a microbench.
- Do not retry D16 GRAPH=1 hang as "the W4A8 recipe".
- Do not retry D18 emul+auto-fp8-KV G1 garbage.
- Do not mix Path H and Path X in one A/B without naming which op
  ran at M=1.
- ASCII only. Newest JOURNAL at the bottom.

G1 (concurrent prefill+decode) is the fail-closed for any e2e claim.
Paris + fib is the fail-closed for any "it loads."

---

## 15. First commands (copy-paste)

Status:

```
cd /mnt/vm_8tb/github/b70_ai_things
./bin/gpu-run --status
curl -s http://192.168.10.5:18080/v1/models || true
```

Card 0 -- RTN W4A8+GDN (fire 1):

```
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 0 \
  env DATAFREE=1 CARD=0 \
  bash scripts/151_quantize_qwen38_27b_w4a8.sh
```

Card 1 -- isolated int4 ops on 3.8 down_proj shape (example; replace
with the extended harness once copied):

```
B70_GPU_LOCK_TIMEOUT=0 ./bin/gpu-run --card 1 \
  docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
    --ipc=host --shm-size 16g -e ZE_AFFINITY_MASK=1 \
    -v /mnt/vm_8tb/b70:/mnt/vm_8tb/b70 \
    -v /mnt/vm_8tb/github/b70_ai_things:/mnt/vm_8tb/github/b70_ai_things \
    -e B70_XPU_C_SO=/mnt/vm_8tb/b70/w8a8_kernel_v0240/_xpu_C.abi3.so \
    vllm-xpu-env:int8g-v0260 \
    bash -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1;
      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH;
      python3 /mnt/vm_8tb/github/b70_ai_things/sglang/int4_gemm_w4a8_probe.py'
```

If that probe is 3.6-path hardcoded, copy it to
`vllm/w4a8/bench_w4a8_shapes.py` and add the table in 8.2. That copy
is a valid no-serve loop.

When 151 fire 1 finishes: census, then a GRAPH=0 vLLM smoke on card 1
(card 0 can already be on GPTQ fire 2).

---

## 16. Pointers

- This campaign: `docs/20260820_qwen38_w4a8_campaign.md`
- Campaign journal: `docs/20260820_qwen38_w4a8_journal.md`
- Loops: `docs/20260820_qwen38_w4a8_loops.md`
- Dead-ends: `docs/20260820_qwen38_w4a8_deadends.md`
- Parked A-E: `docs/20260820_b70_bw_campaigns.md`
- 3.6 W4A8 thesis: `sglang/W4A8_PLAN.md`
- 3.6 W4A8 ABI build: `sglang/W4A8_BUILD.md`
- AutoRound cannot W4A8: `research/w4a8/AUTOROUND_W4A8_FEASIBILITY.md`
- Roofline: `docs/kernel/23_b70_gemv_gemm_roofline.md`
- Native s4: `vllm/nvfp4/INT4_DPAS_PIONEER.md`
- VNNI16 scaffold: `kernels/SYCLTLA_SCAFFOLD.md`
- Quant registry: `docs/quant_methods.md` Tables A/D
- DSpark train notes: `vllm/dflash/DSPARK_RMACY.md`
- Producer: `scripts/151_quantize_qwen38_27b_w4a8.sh`
- AGENTS.md: lease, ASCII, compressed-tensors, sglang-primary,
  no P2P-in-vLLM-TP
