# QUANTS_TODO.md -- INT8 fast-path quant queue for the Qwen3.6 family (point a future agent here)

**Created:** 2026-06-21 - **Updated:** 2026-06-21 (expanded to ALL qwen3.6 family x {W8A8, W4A8}) - **Status:** QUEUED
**Goal:** W8A8 (AutoRound) + W4A8 (selective SmoothQuant + GPTQ) of EVERY qwen3.6-family model, to measure how far the
B70's INT8 fast paths (int8-XMX systolic GEMM) carry on real models. W8A8 = int8 w x int8 a (full int8 XMX);
W4A8 = int4 w x int8 a (int8-XMX via `int4_gemm_w4a8`). Both light the int8 systolic path; this queue produces the
checkpoints to benchmark it.
**Related:** [`docs/kernel/15_autoround_w4a8_w8a8_recipes.md`](docs/kernel/15_autoround_w4a8_w8a8_recipes.md) (method bible) -
[`docs/kernel/18_xpu_int8_moe_kernel.md`](docs/kernel/18_xpu_int8_moe_kernel.md) (the 35B int8-MoE-kernel dependency) -
[`MTP_TODO.md`](MTP_TODO.md) Playbook B (selective SmoothQuant) - [`docs/literature/07_w8a8_int8_recovery.md`](docs/literature/07_w8a8_int8_recovery.md) -
[`scripts/49_quantize_27b_w8a8.sh`](scripts/49_quantize_27b_w8a8.sh) (GPTQ) - [`scripts/59_autoround_2xpu.sh`](scripts/59_autoround_2xpu.sh) (multi-XPU AutoRound, PROVEN).

> **How to use:** work top-to-bottom (priority-ordered). Per item: SMOKE first (cheap toolchain check) -> FULL run ->
> VALIDATE gate -> tick the box + log the result in sec 6. Every GPU run goes through `scripts/gpu-run` (flock lease).
> Output dirs + served ids are method+scheme tagged (CLAUDE.md rule) -- NEVER a bare `qwen36-27b-w8a8`.
>
> **Scope note:** this is now > 24 h. The DENSE 27B + Qwable (both schemes) are servable on `:int8g` TODAY and fit a
> ~24-36 h window. The **35B-A3B MoE quants can be PRODUCED now but NOT served yet** -- int8-act MoE has no XPU kernel
> (see docs/kernel/18); produce them so they are ready the moment the Track-A bootstrap kernel lands.

---

## 0. The method decision (per your spec: W8A8 = AutoRound, W4A8 = SmoothQuant+GPTQ)

| scheme | method | why / how on the qwen3_5 hybrids |
|---|---|---|
| **W8A8** (int8 w + int8 a) | **AutoRound** (smoke-gate) > GPTQ-W8A8 fallback | AutoRound's llm_compressor exporter SUPPORTS INT8_W8A8; >= GPTQ accuracy, Intel's first-class path. AutoRound does its own activation-aware rounding (no SmoothQuant needed). AutoRound-on-XPU for the VLM/MoE is UNVERIFIED -> smoke first; GPTQ-W8A8 (`scripts/49` default) is the proven low-risk ship. |
| **W4A8** (int4 w + int8 a) | **selective SmoothQuant + GPTQ** | AutoRound CANNOT export W4A8 (hard `bits==8` assert, all versions -- kernel/15 sec 1) -> GPTQ is the weight method. **SmoothQuant IS wanted** (recovers the int8-activation fidelity, the ~-10pt W8A16->W8A8 drop). BUT naive all-layers SmoothQuant THROWS on the hybrid (only 16/64 layers carry self_attn q/k/v -> smooth-layer<->qkv pairing fails). So use **SELECTIVE SmoothQuant** (Playbook B): mappings ONLY on the 16 full-attn layers (`input_layernorm->{q,k,v}`, `v_proj->o_proj`) + the 64 MLP layers (`post_attention_layernorm->{gate,up}`); SKIP DeltaNet `linear_attn`. **PREREQ: this selective mapping is NOT in `scripts/49` yet (it defaults `SMOOTHQUANT=0`) -> implement it first (Q0).** |

The dense **Qwen3-14B** has no hybrid issue: ordinary all-layers SmoothQuant+GPTQ works, and `Qwen3-14B-W4A8-gptq`
(SmoothQuant+GPTQ@128, 0.872/0.835) ALREADY exists -> 14B W4A8 is done; 14B is only here as the W8A8-AutoRound
toolchain check + a SmoothQuant reference.

### [!] PATH STATUS -- these are NOT all turn-key. For each quant we must FIND / borrow from COMMUNITY / BUILD a path.

Every item needs TWO working paths, and on the **qwen3_5 VLM / qwen3_5_moe** arch most are UNPROVEN:
- a **PRODUCTION path** (a quant toolchain that actually emits a correct compressed-tensors checkpoint for this arch), and
- a **SERVING path** (an XPU kernel that runs that scheme so we can benchmark the int8 fast path).

For each, expect to do one of three things -- tagged per queue item below as `PATH: ...`:
- **FIND** -- a correct recipe likely exists; locate + VERIFY it works on our exact arch (smoke). (Quant toolchains move fast;
  re-check auto-round/llmcompressor releases at run time -- a newer version may have just added VLM/MoE support.)
- **COMMUNITY** -- check if someone already solved it (HF model cards for qwen3.6/qwen3_5 W8A8/W4A8 checkpoints; auto-round /
  llm-compressor / GPTQModel issues+examples for VLM & MoE quant; Intel llm-scaler). Adopt or adapt rather than reinvent.
- **BUILD** -- no path exists; create it ourselves (e.g. the selective-SmoothQuant mapping Q0; the int8 MoE serving kernel
  docs/kernel/18; any loader/exporter graft). Document the new path back into kernel/15 + this doc when it works.

Rule: before BUILDING, spend a short timebox on FIND + COMMUNITY -- the qwen3.6 family is popular, so a community W8A8/W4A8
checkpoint or a fixed exporter may already exist. Only build the path once find/community come up empty. Record which path
each quant used in the sec-6 log (so the next agent knows what is proven vs improvised).

## 1. Why this needs TWO B70s (the unlock)

The 27B / Qwable / 35B bf16 SOURCES (~54-70 GB params) do NOT fit one 32 GB card for the full-precision load.
- **AutoRound across both cards: `device_map="0,1"` -> xpu:0 + xpu:1. PROVEN** (`scripts/59`; full 0.6B quant ran with
  peak_vram on both cards). Use `ZE_AFFINITY_MASK=0,1` + `device_map="0,1"` (or `"auto"`) + `low_gpu_mem_usage=True`.
- GPTQ/llmcompressor: `device_map="auto"` with CPU offload (host has 125 GB RAM) or both XPUs.
- Quant tuning is compute/host-bound, NOT all-reduce-bound -> the interconnect (Gen3 x16, no P2P -- FINDINGS) does
  NOT hurt quant runs. Multi-card quant is fine.

---

## 2. Inventory (reviewed 2026-06-21)

**BF16 SOURCES:**
| model | path | arch | note |
|---|---|---|---|
| Qwen3-14B (dense, reference) | `/mnt/vm_8tb/specula-build/models/Qwen3-14B` | qwen3 | 8 shards ~28 GB. No vision/MTP/GDN. |
| Qwen3.6-27B (base) | `/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B` | qwen3_5 (VLM+GDN+MTP) | 15 shards ~72 GB. VLM graft to serve. |
| Qwen3.6-27B-Coder (Qwable) | `/mnt/vm_8tb/b70/models/DJLougen_Qwable-5-27B-Coder` | qwen3_5 | 15 shards ~60 GB. Same VLM handling. |
| Qwen3.6-35B-A3B MoE | `/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-35B-A3B` | qwen3_5_moe (256-exp/top-8 +vision +MTP) | DOWNLOADING (`scripts/63`, ~67/70 GB). **Produce quants now; SERVING gated on the int8 MoE kernel (docs/kernel/18).** |

**Quants that ALREADY exist (do NOT redo):** 14B `W4A8-gptq` (SmoothQuant+GPTQ, 0.872/0.835), `W8A8-gptq`, `W8A16`,
`W4A16-gptq` - 27B `W4A8-q-prepacked` (GPTQ, no SmoothQuant -> supersede with Q3), `Lorbus int4-AutoRound` (W4A16
daily driver), `W8A8-INT8-RTNtest` (RTN, bad). - Qwable: none. - 35B: `Intel int4-AutoRound` (W4A16, serves 56.8 t/s).

Disk: 6.3 TB free. Each int8 output ~14-35 GB. Not a constraint.

---

## 3. THE QUEUE (priority order; all qwen3.6 x {W8A8, W4A8})

Legend: [ ] todo - [~] running - [x] done.

### [x] Q0 -- PREREQ: add SELECTIVE SmoothQuant to `scripts/49` (needed for every W4A8 on a hybrid)  DONE 2026-06-21
- Build SmoothQuant mappings ONLY where pairing is clean (Playbook B): the 16 full-attn layers
  (`input_layernorm->{q_proj,k_proj,v_proj}`, `v_proj->o_proj`) + the 64 MLP layers
  (`post_attention_layernorm->{gate_proj,up_proj}`); SKIP DeltaNet `linear_attn` + the vision tower + MTP.
  For the **35B MoE**: the MLP mapping becomes `post_attention_layernorm->{experts.*.gate_proj, up_proj}` (router-aware).
- Wire a `SMOOTHQUANT=selective` (or `SQMAP=hybrid`) mode into `scripts/49` that passes explicit `mappings=[...]` to the
  `SmoothQuantModifier` instead of the auto-all-layers default (which throws). Smoke on the 27B (iters tiny) to confirm
  no `ValueError: got [all 64 input_layernorm]`. **This gates Q3/Q5/Q7.** (Dense 14B doesn't need it; all-layers works.)

### [x] Q1 -- Qwen3-14B  W8A8  (AutoRound)   DONE 2026-06-21 (toolchain VALIDATED; served + benched)
- Validates AutoRound-on-XPU + the llm_compressor W8A8 export cheaply before the 27B. **Out:** `models/Qwen3-14B-W8A8-autoround`.
- `device_map="xpu"`; ignore `lm_head` only. Serve `:int8g`, id `qwen3-14b-w8a8-autoround`. Est ~1-3 h. Recipe 4A.
- Compare vs existing `Qwen3-14B-W8A8-gptq` (the AutoRound-vs-GPTQ W8A8 datapoint).

### [ ] Q2 -- Qwen3.6-27B (base)  W8A8  (AutoRound)
- **Out:** `models/Qwen3.6-27B-W8A8-autoround`. `device_map="0,1"` + `low_gpu_mem_usage`. Ignore VLM list (sec 5).
- Serve: `w4a8/fix_27b_vlm_config.py` graft -> `:int8g`, id `qwen36-27b-w8a8-autoround`. Est ~4-8 h. Recipe 4A.
- Fallback if AutoRound-on-XPU flakes: GPTQ-W8A8 (`scripts/49` default).

### [ ] Q3 -- Qwen3.6-27B (base)  W4A8  (selective SmoothQuant + GPTQ)   [needs Q0]
- **Out:** `models/Qwen3.6-27B-W4A8-sqgptq`. Method: `scripts/49 SCHEME=W4A8 SMOOTHQUANT=selective`. Ignore VLM list.
- Serve: VLM graft + the 4304-dim odd-dim handling (kernel/15 sec 1) -> `:int8g`, id `qwen36-27b-w4a8-sqgptq`. Est ~1-3 h.

### [ ] Q4 -- Qwable-5-27B-Coder  W8A8  (AutoRound)
- Identical handling to Q2 (same qwen3_5 VLM). **Out:** `models/Qwable-5-27B-Coder-W8A8-autoround`. `device_map="0,1"`.
- Serve: graft -> `:int8g`, id `qwable-27b-w8a8-autoround`. Est ~4-8 h.

### [ ] Q5 -- Qwable-5-27B-Coder  W4A8  (selective SmoothQuant + GPTQ)   [needs Q0]
- **Out:** `models/Qwable-5-27B-Coder-W4A8-sqgptq`. Method as Q3. Serve graft + odd-dim -> id `qwable-27b-w4a8-sqgptq`. Est ~1-3 h.

### [ ] Q6 -- Qwen3.6-35B-A3B MoE  W8A8  (AutoRound)   <- PRODUCE now; SERVE gated on docs/kernel/18
- Wait for the bf16 download to finish (`scripts/63`). **Out:** `models/Qwen3.6-35B-A3B-W8A8-autoround`. `device_map="0,1"`.
- Ignore: `lm_head re:.*visual.* re:.*mtp.*` (+ keep the router/gate in bf16; quant only the expert + attn linears).
- **SMOKE HARD:** AutoRound MoE export is "limited" (kernel/15) -- confirm it emits a valid compressed-tensors W8A8 MoE
  before the full run. **Serving NOT possible yet** (no XPU int8 MoE kernel; build Track-A first, docs/kernel/18). Est ~6-12 h.

### [ ] Q7 -- Qwen3.6-35B-A3B MoE  W4A8  (selective SmoothQuant + GPTQ)   [needs Q0; SERVE gated on docs/kernel/18]
- **Out:** `models/Qwen3.6-35B-A3B-W4A8-sqgptq`. llmcompressor MoE GPTQ + the router-aware selective-SQ mapping (Q0).
- Serving gated on the int8 MoE kernel. Est ~3-6 h.

**Already done / not needed:** 14B W4A8 (SmoothQuant+GPTQ exists) - 14B W4A16/W8A16 (exist) - 35B/27B W4A16 (exist).

### Path status per item (FIND = verify existing recipe - COMMUNITY = adopt an existing checkpoint/fix - BUILD = create it)
| item | PRODUCTION path | SERVING path |
|---|---|---|
| Q0 selective-SQ | **BUILD** (llmcompressor `SmoothQuantModifier(mappings=...)`; FIND first -- check if llmcompressor ships a hybrid/MoE SQ example) | n/a (tooling) |
| Q1 14B W8A8 AR | **FIND** -- AutoRound W8A8 dense is known-good; just smoke | PROVEN (`:int8g` dense int8 kernel) |
| Q2 27B W8A8 AR | **FIND** (AutoRound-on-XPU VLM unverified) -> **COMMUNITY** (HF qwen3.6-27B W8A8?) -> fallback **PROVEN** GPTQ-W8A8 | FIND (VLM config graft, exists for our quants) |
| Q3 27B W4A8 SQ+GPTQ | **BUILD** (Q0) then GPTQ-W4A8 PROVEN-on-14B | **FIND** (4304 odd-dim VLM serve graft -- kernel/15) |
| Q4 Qwable W8A8 AR | **FIND** (as Q2; Qwable niche -> COMMUNITY unlikely) | FIND (VLM graft) |
| Q5 Qwable W4A8 SQ+GPTQ | **BUILD** (Q0) then GPTQ-W4A8 | **FIND** (odd-dim serve) |
| Q6 35B W8A8 AR | **FIND/COMMUNITY/BUILD** -- AutoRound MoE export is "limited"; verify, else borrow llm-compressor MoE path, else build the exporter graft | **BUILD** int8 MoE kernel (docs/kernel/18) |
| Q7 35B W4A8 SQ+GPTQ | **FIND/COMMUNITY/BUILD** -- llmcompressor MoE GPTQ + router-aware selective-SQ; verify the MoE GPTQ path first | **BUILD** int8 MoE kernel (docs/kernel/18) |

Most boxes are NOT "PROVEN" -- this queue is as much about finding/building the right PATHS as running them. Timebox
FIND+COMMUNITY before BUILD; log which path each quant actually used.

---

## 4. Recipes (route every run via `scripts/gpu-run`)

### 4A. AutoRound W8A8 (Q1/Q2/Q4/Q6)
Template: `docs/kernel/15` sec 2 (`QuantizationScheme(bits=8, group_size=-1, sym=True, act_bits=8, act_dynamic=True,...)`
+ `quantize_and_save(format="llm_compressor")`). Per item: pip-install `auto-round "transformers>=4.52" accelerate datasets`;
27B/Qwable/35B use `device_map="0,1"` + `ZE_AFFINITY_MASK=0,1` + `low_gpu_mem_usage=True` (kernel/15's `device_map="xpu"`
OOMs on the >32 GB load -- the 2-card map is the fix, proven `scripts/59`); `layer_config` forces the IGNORE modules to
`{"bits":16}`. SMOKE at `iters=50,nsamples=64` (confirm export config) THEN full `iters=200,nsamples=128,seqlen=2048`.

### 4B. selective SmoothQuant + GPTQ W4A8 (Q3/Q5/Q7)  [after Q0]
```bash
scripts/gpu-run env \
  SCHEME=W4A8 METHOD=gptq SMOOTHQUANT=selective DEVICE=xpu \
  SRC=/mnt/vm_8tb/b70/models/<SOURCE_DIR> \
  OUTNAME=<OUTPUT>-W4A8-sqgptq SAMPLES=512 SEQLEN=2048 \
  IGNORE="lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*" \
  bash scripts/49_quantize_27b_w8a8.sh
```
- `SMOOTHQUANT=selective` is the Q0 enhancement (explicit Playbook-B mappings; the auto/all-layers mode still throws on hybrids).
- `actorder=None` inside `scripts/49` (avoids the XPU gather device-lost). `down_proj` early+late at W4A16 is an optional rescue.
- 27B/35B load may OOM one card -> `device_map="auto"` (CPU offload) or both XPUs.
- GPTQ-W8A8 fallback for the W8A8 items: same command `SCHEME=W8A8 SMOOTHQUANT=selective`.

---

## 5. Universal gotchas (apply to EVERY item)
1. **Ignore list (regex):** `lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*` for the qwen3_5 27B/Qwable; for the 35B
   MoE add the router/gate (keep bf16). 14B = `lm_head` only. Keep the PARENT regex `re:.*linear_attn.*` (avoid #40252 zeroing).
2. **27B/Qwable/35B are VLMs:** apply `w4a8/fix_27b_vlm_config.py` to the output before serving. The 4304-dim vision fc2
   trips the group-128 int4 kernel -> keeping vision bf16 (ignore list) sidesteps it.
3. **MTP head + router stay BF16.** 4. **Calibration:** chat/instruction data, 512 samples (128 to iterate), SEQLEN=2048.
5. **VALIDATE GATE:** eval top-1 agreement / gsm8k vs bf16 (CPU-anchored ok) before trusting; `evals/` or `evals/gsm8k_probe.py`.
   Verify the served id encodes method+scheme (`-autoround` / `-sqgptq`) vs `evals/configs/models.yaml`.
6. **One model on the card at a time;** stop the daily driver (`./daily_driver_serve.sh stop`) before a quant/serve run.
7. **Smoke-then-full for AutoRound** (VLM/MoE export UNVERIFIED -- never burn hours before a 10-min smoke).
8. **35B (Q6/Q7) are PRODUCE-only until the int8 MoE kernel exists** (docs/kernel/18 Track A). Eval their accuracy offline;
   the int8-fast-path perf test waits on the kernel.

---

## 6. Results log (fill as you go)

| date | item | model / scheme / method | out dir | served id | accuracy (agree/gsm8k) | decode/prefill t/s | verdict |
|---|---|---|---|---|---|---|---|
| 0621 | Q0 | scripts/49 selective-SmoothQuant | (code, committed) | -- | -- | -- | DONE: builds per-layer maps by model inspection |
| 0621 | Q1 | 14B / W8A8 / autoround | models/Qwen3-14B-W8A8-autoround | qwen3-14b-w8a8-autoround | acc TBD (== gptq kernel) | dec 25.1(c1)/18.0(c8); ttft 347ms | DONE. W8A8 decode BW-bound (~half int4); lowest c1 TTFT. ctx2048 sweep saved |
| -- | Q2 | 27B-base / W8A8 / autoround | -- | -- | -- | -- | -- |
| -- | Q3 | 27B-base / W4A8 / sq+gptq | -- | -- | -- | -- | -- |
| -- | Q4 | Qwable / W8A8 / autoround | -- | -- | -- | -- | -- |
| -- | Q5 | Qwable / W4A8 / sq+gptq | -- | -- | -- | -- | -- |
| -- | Q6 | 35B-A3B / W8A8 / autoround | -- | -- | -- | (serve gated) | -- |
| -- | Q7 | 35B-A3B / W4A8 / sq+gptq | -- | -- | -- | (serve gated) | -- |
