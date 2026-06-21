# QUANTS_TODO.md -- AutoRound/GPTQ INT quantization queue (point a future agent here)

**Created:** 2026-06-21 - **Window:** a ~24 h block on the **2x B70** rig - **Status:** QUEUED (no GPU work done yet)
**Related:** [`docs/kernel/15_autoround_w4a8_w8a8_recipes.md`](docs/kernel/15_autoround_w4a8_w8a8_recipes.md) (the method bible) -
[`MTP_TODO.md`](MTP_TODO.md) Playbook B - [`scripts/49_quantize_27b_w8a8.sh`](scripts/49_quantize_27b_w8a8.sh) (GPTQ) -
[`scripts/59_autoround_2xpu.sh`](scripts/59_autoround_2xpu.sh) (multi-XPU AutoRound, PROVEN) - [`FINDINGS.md`](FINDINGS.md)

> **How to use this doc:** work the queue top-to-bottom (it is priority-ordered). For each item: SMOKE first
> (cheap toolchain check), then the FULL run, then the VALIDATE gate, then tick the box and note the result here.
> Every GPU run goes through `scripts/gpu-run` (the flock lease). Output dirs + served ids are method+scheme
> tagged (CLAUDE.md rule) -- NEVER a bare `qwen36-27b-w8a8`.

---

## 0. The method decision (answers "does AutoRound supersede SmoothQuant+GPTQ?")

**Short answer: yes for W8A8 / W4A16, NO for W4A8 -- they are complementary, not one-supersedes-all.**

| scheme | method to use | why |
|---|---|---|
| **W8A8** (int8 w + int8 act) | **AutoRound** (smoke-gate it) > GPTQ fallback | AutoRound export IS supported for INT8_W8A8; generally >= GPTQ accuracy and is Intel's first-class Arc path. But AutoRound-on-XPU for the qwen3_5 VLM is UNVERIFIED -> smoke first; GPTQ-W8A8 (`scripts/49` default) is the proven low-risk ship. |
| **W4A8** (int4 w + int8 act) | **GPTQ only** (AutoRound CANNOT) | AutoRound's llm_compressor exporter hard-asserts `bits==8` for any int8-dynamic-act scheme on EVERY released+unreleased version (0.13.1 + main 0.14.0-dev). W4 -> AssertionError at export. See kernel/15 sec 1. So W4A8 = GPTQ (`scripts/49 SCHEME=W4A8`). GPTQ-W4A8 already lands 0.872/0.835 on the 14B -> recovers ~all of AutoRound's hoped-for edge. |
| **W4A16** (int4 w + fp16 act) | AutoRound (preferred) | What we actually serve today (Lorbus/Intel int4 AutoRound). Not in this queue (the ask is w4a8/w8a8) but AutoRound is the method if revisited. |

**Caveat on "SmoothQuant+GPTQ":** for the **hybrid qwen3_5 27B** SmoothQuant is OFF anyway (`SMOOTHQUANT=0`) -- only
16/64 layers carry self_attn q/k/v so SmoothQuant's smooth-layer<->qkv pairing throws before any GPU work. So the
27B/Qwable W4A8/W8A8 GPTQ path is **GPTQ-only**, not SmoothQuant+GPTQ. SmoothQuant CAN apply to the **dense 14B**.

## 1. Why this needs TWO B70s (the unlock)

The 27B / Qwable bf16 SOURCES are ~54 GB of params (72 GB / 60 GB on disk) -> **do NOT fit one 32 GB card** for the
full-precision load that AutoRound/GPTQ need. The 2nd card is what makes quantizing them on-GPU viable:
- **AutoRound across both cards: `device_map="0,1"` -> xpu:0 + xpu:1. PROVEN this session** (`scripts/59`, full
  0.6B quant ran with peak_vram on both cards). Use `ZE_AFFINITY_MASK=0,1` + `device_map="0,1"` (or `"auto"`).
  Add `low_gpu_mem_usage=True` for headroom on the 27B (54 GB across 64 GB is tight with tuning state).
- The 14B (28 GB bf16) fits ONE card -> `device_map="xpu"` is fine (do it first as the cheap toolchain check).
- GPTQ (`scripts/49` / llmcompressor) can instead use `device_map="auto"` with **CPU offload** (host has 125 GB
  RAM, holds the 72 GB bf16) -- slower but robust; or both XPUs.
- NOTE: quant tuning is compute/host-bound, NOT all-reduce-bound, so the **x1 PCIe link does NOT hurt quant runs**
  (unlike TP serving -- see FINDINGS). Multi-card quant is fine on the current wiring.

---

## 2. Inventory (reviewed 2026-06-21)

**BF16 SOURCES we can quantize:**
| model | path | arch | note |
|---|---|---|---|
| Qwen3-14B | `/mnt/vm_8tb/specula-build/models/Qwen3-14B` | qwen3 (dense) | 8 shards, ~28 GB. No vision/MTP/GDN -> simplest ignore list. |
| Qwen3.6-27B (base) | `/mnt/vm_8tb/b70/models/Qwen_Qwen3.6-27B` | qwen3_5 (VLM+GDN+MTP) | 15 shards, ~72 GB. Needs VLM ignore list + config graft to serve. |
| Qwen3.6-27B-Coder (Qwable) | `/mnt/vm_8tb/b70/models/DJLougen_Qwable-5-27B-Coder` | qwen3_5 | 15 shards, ~60 GB, complete. Same VLM handling as the base 27B. |
| ~~Qwen3.6-35B-A3B MoE~~ | (no bf16 on host) | MoE | **EXCLUDED** -- no source AND no XPU int8 MoE kernel (kernel/15 sec 3). Stay W4A16-int4. |

**Quants that ALREADY exist (do NOT redo unless re-baselining):**
- 14B: `W4A16-gptq`, `W4A8-gptq` (+prepacked), `W8A16`, `W8A8-gptq` (+gptq512). [all GPTQ; no AutoRound yet]
- 27B: `W4A8-q-prepacked` (GPTQ), `Lorbus_..-int4-AutoRound` (W4A16, the served daily driver), `W8A8-INT8-RTNtest`
  (RTN -- parked/bad, do NOT serve), `W4A16` (does not serve -- XPUwNa16 odd-dim).
- Qwable: NONE yet.

Disk: 6.3 TB free. Each int8 output ~14-27 GB. Not a constraint.

---

## 3. THE QUEUE (priority order -- do top-down; ~24 h fits items 1-4, item 5 if time)

Legend: [ ] todo - [~] running - [x] done (record result inline).

### [ ] Q1 -- Qwen3-14B  W8A8  (AutoRound)   <- DO FIRST: cheapest toolchain validation
- **Why first:** smallest source (fits ONE card), dense (no VLM graft), so it validates the AutoRound-on-XPU
  toolchain + export at low cost. If AutoRound flakes, find out here, not 6 h into the 27B.
- **Method:** AutoRound INT8_W8A8. **Source:** specula `/Qwen3-14B`. **Out:** `models/Qwen3-14B-W8A8-autoround`.
- **device_map:** `"xpu"` (one card OK) or `"0,1"`. **Ignore:** `lm_head` only (dense -- no visual/mtp/linear_attn).
- **Serve check:** `:int8g`, served id `qwen3-14b-w8a8-autoround`. **Est:** ~1-3 h. Recipe: sec 4A.
- Compare accuracy vs the existing `Qwen3-14B-W8A8-gptq` (the AutoRound-vs-GPTQ W8A8 datapoint we want).

### [ ] Q2 -- Qwen3.6-27B (base)  W8A8  (AutoRound)   <- the headline int path
- **Why:** best-accuracy int8 checkpoint of the flagship 27B; only a bad RTN W8A8 exists today. Phase C target.
- **Method:** AutoRound INT8_W8A8. **Source:** `/Qwen_Qwen3.6-27B`. **Out:** `models/Qwen3.6-27B-W8A8-autoround`.
- **device_map:** `"0,1"` (BOTH cards -- 54 GB won't fit one) + `low_gpu_mem_usage=True`, `ZE_AFFINITY_MASK=0,1`.
- **Ignore (VLM):** `lm_head re:.*visual.* re:.*mtp.* re:.*linear_attn.*` (keep vision/MTP/GDN bf16).
- **Serve:** apply `w4a8/fix_27b_vlm_config.py` to the output, then `:int8g`, id `qwen36-27b-w8a8-autoround`.
- **Est:** ~4-8 h. Recipe: sec 4A. Fallback if AutoRound-on-XPU flakes: **GPTQ-W8A8** `scripts/49` default (proven).

### [ ] Q3 -- Qwable-5-27B-Coder  W8A8  (AutoRound)
- **Why:** fresh coder finetune, zero quants exist; the coding-agent server wants a good int8 of it.
- **Method/handling:** identical to Q2 (same qwen3_5 VLM arch). **Source:** `/DJLougen_Qwable-5-27B-Coder`.
- **Out:** `models/Qwable-5-27B-Coder-W8A8-autoround`. **device_map** `"0,1"`. **Ignore:** VLM list (as Q2).
- **Serve:** `fix_27b_vlm_config.py` graft, id `qwable-27b-w8a8-autoround`. **Est:** ~4-8 h.

### [ ] Q4 -- Qwable-5-27B-Coder  W4A8  (GPTQ)   <- AutoRound CANNOT do W4A8
- **Why:** the single-card int8-XMX coder path (W4A8 ~= 17 GB fits one card; best prefill/TTFT). Fresh model.
- **Method:** GPTQ (`scripts/49 SCHEME=W4A8 METHOD=gptq SMOOTHQUANT=0`). **Source:** `/DJLougen_Qwable-..`.
- **Out:** `models/Qwable-5-27B-Coder-W4A8-gptq`. **Ignore:** VLM list. **Serve graft + odd-dim:** see sec 4B + the
  27B-VLM 4304-dim caveat (kernel/15 sec 1). id `qwable-27b-w4a8-gptq`. **Est:** ~1-3 h.

### [ ] Q5 -- Qwen3.6-27B (base)  W4A8  (GPTQ)   <- optional, if time remains
- **Why:** a clean method-tagged W4A8 of the base 27B (the existing `W4A8-q-prepacked` is fine but undocumented
  provenance; a fresh `-gptq` is the auditable one). **Method:** GPTQ as Q4. **Source:** `/Qwen_Qwen3.6-27B`.
- **Out:** `models/Qwen3.6-27B-W4A8-gptq`. id `qwen36-27b-w4a8-gptq`. **Est:** ~1-3 h.

**NOT queued (already have / impossible):** 14B W4A8 (GPTQ exists, 0.872/0.835) - 14B W4A16/W8A16 (exist) -
any AutoRound-W4A8 (export blocked) - 35B-A3B anything-int8 (no source + no XPU kernel).

---

## 4. Recipes (copy-paste; route every run via `scripts/gpu-run`)

### 4A. AutoRound W8A8 (Q1/Q2/Q3)
Full template lives in **`docs/kernel/15` sec 2** (the exact `QuantizationScheme(bits=8, group_size=-1, sym=True,
act_bits=8, act_dynamic=True ...)` + `quantize_and_save(format="llm_compressor")`). Adapt per item:
- pip-install at runtime (no image bakes it): `pip install -q auto-round "transformers>=4.52" accelerate datasets`.
- **For the 27B/Qwable use `device_map="0,1"` + `ZE_AFFINITY_MASK=0,1` + `low_gpu_mem_usage=True`** (kernel/15 wrote
  `device_map="xpu"` -- that was pre-2nd-card and OOMs on the 72 GB load; the 2-card map is the fix, proven by
  `scripts/59`). For the 14B keep `device_map="xpu"`.
- `layer_config`: force the IGNORE modules to `{"bits": 16}` (lm_head; +visual/mtp/linear_attn for the 27B/Qwable).
- **SMOKE FIRST:** one pass at `iters=50, nsamples=64` -> confirm it exports `quant_method=compressed-tensors` with a
  W8A8 group + estimate wall-clock, THEN the full `iters=200, nsamples=128, seqlen=2048`. If AutoRound-on-XPU
  device-losts, fall back to `device_map="cpu"` (slow ~1-2 days) or to GPTQ-W8A8.

### 4B. GPTQ W4A8 (Q4/Q5)  -- and GPTQ-W8A8 fallback
```bash
scripts/gpu-run env \
  SCHEME=W4A8 METHOD=gptq DEVICE=xpu SMOOTHQUANT=0 \
  SRC=/mnt/vm_8tb/b70/models/<SOURCE_DIR> \
  OUTNAME=<OUTPUT_DIR_NAME> SAMPLES=512 SEQLEN=2048 \
  IGNORE="lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*" \
  bash scripts/49_quantize_27b_w8a8.sh
```
- `scripts/49` passes SCHEME through to the GPTQModifier; `actorder=None` inside (avoids the XPU gather device-lost).
- For GPTQ-W8A8 fallback: same command with `SCHEME=W8A8` (the script default), same IGNORE list.
- 27B load may OOM one card -> if so, run llmcompressor with `device_map="auto"` (CPU offload, 125 GB RAM) or both XPUs.

---

## 5. Universal gotchas (apply to EVERY item)

1. **Ignore list (name-robust regex):** `lm_head re:.*linear_attn.* re:.*visual.* re:.*mtp.*` for the qwen3_5 27B/
   Qwable (DeltaNet, vision tower incl. the 4304-dim fc2, MTP head, lm_head stay BF16). 14B = `lm_head` only. Keep
   the PARENT-module regex `re:.*linear_attn.*`; never enumerate leaf names (avoids vLLM #40252 silent-zeroing).
2. **27B/Qwable are VLMs:** must apply `w4a8/fix_27b_vlm_config.py` to the output before serving (wrapper-config
   graft + processor files), else the VLM won't build the BF16 ignore modules. The 4304-dim vision fc2 trips the
   group-128 int4 kernel -> keeping vision BF16 (ignore list) sidesteps it.
3. **MTP head MUST stay BF16** (`re:.*mtp.*`) -- quantizing it kills drafting if MTP is ever wired (MTP_TODO).
4. **Calibration data matched to the model** (chat/instruction data for instruct models); 512 samples final, 128 to
   iterate, SEQLEN=2048.
5. **VALIDATE GATE (before trusting any output):** eval top-1 agreement / gsm8k vs the bf16 (or the existing best)
   baseline using `evals/` (or `evals/gsm8k_probe.py` for a quick check). A quant that misses the gate is out. And
   per CLAUDE.md, **verify the served id** (`curl .../v1/models`) encodes the method (`-autoround` / `-gptq`),
   cross-checked against `evals/configs/models.yaml`.
6. **One model on the card at a time** -- each serve evicts the others off port 18080. Quant runs hold the gpu-run
   lease; if the daily driver is up, stop it first (`./daily_driver_serve.sh stop`).
7. **Smoke-then-full for AutoRound** (sec 4A) -- the AutoRound-on-XPU-for-VLM path is UNVERIFIED; never burn 6 h
   before a 10-min smoke confirms the export config.

---

## 6. Results log (fill as you go)

| date | item | model/scheme/method | out dir | served id | accuracy (agree/gsm8k) | decode t/s | verdict |
|---|---|---|---|---|---|---|---|
| -- | Q1 | 14B W8A8 autoround | -- | -- | -- | -- | -- |
| -- | Q2 | 27B W8A8 autoround | -- | -- | -- | -- | -- |
| -- | Q3 | Qwable W8A8 autoround | -- | -- | -- | -- | -- |
| -- | Q4 | Qwable W4A8 gptq | -- | -- | -- | -- | -- |
| -- | Q5 | 27B W4A8 gptq | -- | -- | -- | -- | -- |
