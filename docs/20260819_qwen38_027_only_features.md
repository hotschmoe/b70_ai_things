# Qwen3.8 W8A8 + DSpark -- 0.27-only feature list (PRE.15)

**Written:** 2026-08-19 (LOOP 24). No GPU. Notes already on disk.
**Gate:** campaign PRE.15 / section 4. Phase 2 (torch 2.13 ABI rewrite)
stays closed until this list exists **and** Phase 0+1 have a coherent
W8A8+DSpark number. Both are now true. This file is the list.
**It does not start Phase 2.**

Sources (do not re-fetch for this list):

- `docs/20260818_qwen38_w8a8_dspark_campaign.md` sections 2.5, 4, D/E,
  Phase 2 table
- `docs/20260818_qwen38_sergiioB_cookbook.md` (digest `f01e24f6`)
- `docs/20260819_steve_qwen38_int4ar.md` (HEAD `924b518` at ingest)
- LOOP 6 / 16-23 (KV_FP8 no-op; AGASYNC 29.4; compile hash
  `b3f7e9e010`; P4.1 TTFT)
- `kernels/SYCLTLA_SCAFFOLD.md`

Phase 0+1 number this list is gated on: W8A8+DSpark `bench_code` c1
**29.4** (k=4 GRAPH=1 ALLGATHER_ASYNC @122880, G1 hold). HE+
0.957/0.927. Off-shelf accept is already in the NVFP4 band.

Campaign law still stands: if 0.26 already serves the draft, 2.13 is
optional. 0.26 **does** serve the draft (`method=dspark`, V2,
THINK_BUDGET=0).

---

## 1. What is actually 0.27 / torch-2.13-only

These need a dedicated kernel rewrite (P2.1), not `pip install`.
Enter Phase 2 only if one of them is the remaining 29.4 -> 41.2 path.

| feature | where it lives | need it for W8A8+DSpark 41.2? |
|---|---|---|
| vLLM `0.27.1` / nightly `0.27.2rc1.dev77+gac7509e2b` + kernels `0.1.12.3` on torch **2.13** | SergiioB pin, digest `f01e24f6`. Public name `vllm-openai-xpu:v0.27.1`. | No. Their 83.7 is 1x B70 GPTQ-Int4 MTP4 post-first, not our TP=2 `bench_code`. S1 on this box was **47.58**. |
| sglang 0.5.17 + main #31751 + oneAPI **2026.0** on torch 2.13 | campaign 2.5 / Phase 2 | No. 0.5.15 already lost vs 0.5.6 shelf (-6.1% c1). DSpark on sglang-XPU is Phase 3. |
| Rebuild `_xpu_C.abi3.so`, GDN so, NVFP4 so, fake-op registrations vs 2.13 | campaign Phase 2 budget | Cost of entry, not a feature. Graph fake names may have moved (`XPUW8A8FP8LinearKernel` already renamed once). |
| Soak: `torch.topk` SIGSEGV; `nonzero`/`unique` empty-on-26.22 | campaign Phase 2 | Risk, not a win. |
| oneCCL **2021.17** overlay | 0.27 images ship **2021.15** again (PRE.10) | Mandatory if we ever load a 0.27 image. Not a speed feature. |
| Quantized Markov heads | campaign section 4 / P2.3 | Only if it moves e2e. Off-shelf accept is already 2.45-3.16 / pos0 0.65-0.80. Train is not forced. |
| Newer V2 runner | campaign section 4 | 0.26 V2 already runs `method=dspark`. No written 0.27 V2 delta that closes 29.4 vs 41.2. |
| Official `method=dflash` registry (`DFlashQwen3DSparkModel`) | SergiioB Nemotron page: 0.26.1.dev668, **186** C1. PRE.3 on our 0.26.0. | Not uniquely 0.27. Do not use it on 0.26.0. Not this speed window. |

SergiioB 0.27 cookbook knobs that are **not** 0.27-only (or that we
refuse):

| knob | verdict |
|---|---|
| `B70_MTP_BF16_DRAFT` | MTP draft keep-BF16. We already keep DSpark BF16. |
| 131k MTP boundary (`patch_mtp_nightly` + `patch_mtp_boundary`) | Already ported under `vllm/patches/cookbook/`. Re-hash if a 0.27 image is ever built. |
| `VLLM_XPU_ENABLE_XPU_GRAPH=1` | GRAPH=1 PIECEWISE already on 0.26 (`int8g-v0260`). |
| fp8 KV on 3.8 dense | Their 1x 32 GiB 131k trick. Our W8A8 3.8 serve has **no KV_FP8 hook** (LOOP 6). That is a 0.26 code change, not a 0.27 feature. |
| `--language-model-only` | Drops vision. Graft keeps the tower. Do not copy. |
| `--no-enable-prefix-caching` | Their speed recipe. P4.1 showed cache **hits** on this W8A8 DSpark GRAPH=1 GDN path. Do not copy. |
| tool parser `qwen3_xml` | Agent, not decode. |
| 230 W power cap | Host-level. Do not change mid-loop without an A/B. |

**Phase 2 verdict from this list:** do not enter. Remaining 29.4 vs
41.2 is verify-shape / comms / kernel work on the ABI we have.
Do not bump to chase 83.7 or Steve 101.9.

---

## 2. Steve notes that are 0.26 steals (not 0.27, not S2)

Steve 3.8 AutoRound INT4 MTP5 **101.922** / MTP4 **100.497** is S2
later (`docs/20260819_steve_qwen38_int4ar.md`). Do not serve that
scheme in this window. Three lessons apply to grafted W8A8-gptq now:

1. **Compile-cache identity.** Our GRAPH=1 hash `b3f7e9e010` ignores
   SPECTOK and the mounted `_xpu_C` SO (D2/D3/D5/D7). Wipe that dir
   before changing SPECTOK or GDN_SO. Longer fix: put SPECTOK + SO
   identity in the compile key. Same class of bug as his pinned
   cache as run identity.
2. **oneDNN barriers-on.** Lean-flag retest `9f90e2c`: dropping
   oneDNN barriers is slower and less deterministic. This is a
   decode lever on 0.26, not a 2.13 reason. Exact env is in that
   commit / the 2026-08-18 GDN-scratch note; local
   `/mnt/vm_8tb/b70/b70-optimization-lab` was stale at ingest
   (`03f98aaf`, no INT4-AR repro). Next GPU fire fetches that name
   then A/Bs vs 29.4.
3. **GDN scratch zero-init.** `vllm-xpu-kernels`
   `fix/gdn-scratch-zero-init` `0ab8205`. Residue / nondeterminism
   fix, not a decode win until we see residue. Do not rebuild
   kernels just to have it.

Do not photocopy 101.9 onto W8A8.

---

## 3. Leftover W8A8 speed notes (stay on 0.26)

Closed this window (do not retry unless the packet Retry-if is
true): D1-D8, P1.5, P1.6 decode, E1, E2, GRAPH=1 k=3, fusedq for
`bench_code` c1, host-barrier ALLGATHER, shard-top1 hook, W8A16
clone @122880.

Best standing recipe: `vllm/dflash/serve_qwen38_w8a8_dspark.sh`
SPECTOK=4 GRAPH=1 ALLGATHER_ASYNC MAXLEN=122880
SERVED=`qwen3.8-27b-W8A8-gptq-dspark4-agasync`. Keep AGASYNC.

Ranked leftover (one pick per fire):

| # | pick | why | first command |
|---|---|---|---|
| E3 | oneDNN barriers-on A/B vs 29.4 | Steve steal; only unmeasured 0.26 flag-class decode lever | fetch Steve lab, read `9f90e2c` / GDN-scratch note for the env, restart this AGASYNC recipe with it, G1, `bench_code` c1 |
| D9 | compile-key includes SPECTOK + SO | stops D2/D3 class "duct"; not a c1 win by itself | code on 0.26 compile cache key; wipe `b3f7e9e010` before any SPECTOK change until then |
| P1.6b | fusedq **TTFT/PP** A/B | D5 retry-if; decode already NO-GO 28.3 | remount v0240 fusedq SO, wipe hash, measure cold/warm TTFT vs P4.1 1528/449, not c1 |
| P1.8 | sycl-tla C1 VNNI16 / M<8 DPAS | campaign D: the verify-shape differentiator | read `kernels/SYCLTLA_SCAFFOLD.md`; microbench then e2e. Isolated 1.2x with e2e drop is a packet |
| B1 | KV_FP8 hook on W8A8 3.8 serve | P0.2 leftover; capacity (262k+spec or GRAPH=1 @131k), not c1 | add the hook to the W8A8 serve path, then KV_FP8=1 A/B G1 |
| P4.1b | 262k TTFT on MTP-off | P4.1 leftover; recipe is 122880 (D1) | stop DSpark, P0.2 262k serve, same cold/warm TTFT |
| G5 | 18/18 mixed prefill+decode | concurrent quality; not the 29.4 gap | GRAPH=1 short only; no HE+ under GRAPH=1 CGRECLAIM=0 |

Not this window: P4.2 SpecPrefill, P1.1-P1.4 train, S2 INT4-AR,
SergiioB GPTQ-Int4 as vehicle, Phase 2 ABI rewrite, method=dflash
on 0.26, P2P=1, W8A16_M_MAX>0 @122880, overwrite w8a8-gptq.

29.4 vs 41.2 is still verify cost on INT8-XMX, not prefill (P4.1
warm 2048 TTFT 449 ms already works) and not accept (pos0 65-80%).

---

## 4. Pins / digest law (do not mix)

| thing | pin |
|---|---|
| Our W8A8 research image | `vllm-xpu-env:int8g-v0260` torch 2.12 |
| SergiioB 3.8 cookbook image | `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` |
| 3.6 cookbook (retired for 3.8) | digest `2c427ef` |
| Nemotron cookbook | digest `1da0a954` -- do not apply grouped-topk / SSU |
| Steve INT4-AR | later S2; HEAD at ingest `924b518` |
| Compile hash to wipe on SPECTOK/SO change | `/mnt/vm_8tb/b70/vllm_cache/torch_compile_cache/b3f7e9e010` |

---

## 5. PRE.15 status

- Phase 0+1 coherent W8A8+DSpark number: **yes (29.4)**.
- Written 0.27-only feature list: **this file**.
- Enter Phase 2? **No.** List says 2.13 is optional and does not
  close 29.4 vs 41.2. Next pick is leftover 0.26 speed (E3).
