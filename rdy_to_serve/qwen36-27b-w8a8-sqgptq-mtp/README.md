# qwen36-27b-w8a8-sqgptq-mtp

Qwen3.6-27B **W8A8** (compressed-tensors INT8 weights x INT8 activations, SmoothQuant+GPTQ) + a **BF16 MTP
graft**, served **TP=2 + MTP spec=3** on 2x B70.

## [!!!] 2026-06-25 UPDATE -- DEFAULT IS NOW `CGMODE=NONE` (the captured PIECEWISE default CRASHED)

The "captured PIECEWISE" default below **CRASHES under sustained MTP load** on TP=2 (~16-20 min / ~20-28k tokens):
the MTP path's XPU graph REPLAY overflows the Level-Zero (NEO) command stream. Root-caused + reproduced in
campaign 120 (`docs/20260625_w8a8_27b_mtp_graph_campaign.md`, JOURNAL 2026-06-25). It is a software EngineDeadError
(GPUs stay healthy), distinct from the cumulative-TP2 DEVICE_LOST wedge.

**New default: `CGMODE=NONE`** (GRAPH=1, cudagraph_mode=NONE) -- keeps torch.compile/inductor but skips graph
replay, so there is no command-stream accumulation. Verified STABLE (soaked clean to 57,344 tokens, ~2.9x the
crash zone) and **~2x the enforce-eager fallback**:

| config (TP=2 + MTP3) | decode tok/s (c1) | stable? |
|---|---:|---|
| enforce-eager (`GRAPH=0`) | 12.78 | yes (slow fallback) |
| **`CGMODE=NONE` (new default)** | **25.39** | **yes -- soaked 57k** |
| `CGMODE=PIECEWISE` (old default) | 34.89 | NO -- crashes ~20-28k |
| PIECEWISE + drafter-eager | 36.08 | NO -- degrades/hangs (target replay still accumulates) |

So the recipe now defaults `CGMODE=NONE`; `CGMODE=PIECEWISE` (fast but crashes) and `GRAPH=0` (enforce-eager,
slow but stable) remain selectable. spec=3 is the MTP winner. The 2026-06-23 correction below still applies to
the PIECEWISE path's numerics.

## [!!!] 2026-06-23 CORRECTION -- the old "63 t/s / 3.4x captured" headline was GARBAGE

This recipe previously claimed "the fastest single-stream 27B config, ~63-64 tok/s, 3.4x". **That was a false
positive measured on degenerate output.** Two stacked bugs (full diagnosis: JOURNAL 2026-06-23, scripts/101-106):

### Bug A -- the garbage (FIXED, config-only, NO requant)
The checkpoint `config.json` `ignore` list had **336 enumerated leaf names with the wrong flat prefix**
`model.layers.N.linear_attn.*`. The actual checkpoint keys are VLM-**nested** `model.language_model.layers.N.*`,
so those ignore entries **matched nothing** -> the 48 GDN `linear_attn` layers (correctly stored BF16) were **not
exempted** -> vLLM built them as W8A8 int8 linears. A BF16 `[out,in]` weight silently shape-matches the int8 param
buffer and the `weight_scale` is absent -> 48 recurrent GDN layers of garbage -> the model emitted `!!!!` / `is is is`.
A degenerate body makes the BF16 MTP draft head (drafting the same garbage) and the target **agree on `!`** -> trivial
~98% accept, accept_len saturates at spec+1=5.9. The throughput bench used `ignore_eos` + token-count and **never read
the text** (the exact QUANTS_TODO Q8 trap).
- **Fix:** replace the ignore list with the regex form `["lm_head","re:.*linear_attn.*","re:.*visual.*","re:.*mtp.*"]`
  (the same form the W4A8 already uses). Tool: `scripts/104_fix_w8a8_ignore.py` (backs up to `config.json.ignore339.bak`).
- **Weights were always good** (dequant int8*scale vs bf16 base: per-channel cosine 0.97-0.9999). llmcompressor
  quantized correctly; only the SAVED config.json ignore serialization had the wrong prefix. So this is config-only.

### Bug B -- captured path garbage: ROOT-CAUSED + FIXED 2026-06-24 (recipe now defaults CAPTURED)
It was **never** a "captured int8 numerics" bug. Root cause: vLLM's piecewise `CUDAGraphWrapper` does a pure
`replay()` with **no input copy** -> every captured piece's inputs must sit at the **same device address** on replay
as at capture. The XPU TP collectives are all **out-of-place** (`all_reduce` returns `input_.clone()`;
all_gather/reduce_scatter return fresh `torch.empty()+.contiguous()`). The old recipe listed the 3 collectives in
`splitting_ops`, which **ejects** them to eager at a piecewise boundary; on XPU their fresh output does not reproduce
the capture-time address -> the next captured piece reads **stale** data -> garbage. (`use_inductor_graph_partition=true`
separately KeyErrors on the mixed W8A8+BF16-GDN region, so we use `IGP=false`.)
- **FIX = eject NOTHING.** `splitting_ops` = the attention/GDN custom ops only; all collectives stay CAPTURED.
  all_reduce + reduce_scatter record fine. The spec-verify `all_gather` (which oneCCL 2021.17 cannot graph-record)
  is replaced by an **all-reduce-of-padded all_gather** shim (`patches/sitecustomize.py`) so it too records and stays
  captured -- nothing is ejected, so no boundary is ever stale.
- Two facets: (a) ejecting collectives -> garbage body; (b) ejecting only all_gather -> coherent body but the
  captured spec-VERIFY gives **0% accept** (MTP becomes pure overhead). Plan (a)+(b) together = capture everything
  correctly -> coherent AND real accept.
- **=> recipe DEFAULTS TO CAPTURED (`GRAPH=1`, `IGP=false`, eject nothing, capture-safe all_gather shim).**
  `GRAPH=0` is still available (eager, ~10 t/s) as a fallback.

## Honest result (coherence-gated, temp=0 greedy, hard prompt; scripts/111)

| config | decode tok/s | accept | note |
|---|---|---|---|
| eager, MTP-off (pure body) | ~4.1 | - | TP=2 fully-eager int8 is launch/collective-bound |
| eager, MTP spec=5 | 10.43 | 36% | the old "only coherent" path |
| captured, MTP-off | 18.10 | - | coherent; capture alone ~4.5x over eager |
| captured, MTP spec=5, eject only all_gather | 9.63 | ~0% | coherent but verify dead -> no speedup |
| captured, MTP spec=5, capture-safe all_gather | 26.10 | 26% | coherent |
| captured, MTP spec=4, capture-safe all_gather | 30.56 | 37% | coherent |
| **captured, MTP spec=3, capture-safe all_gather (DEFAULT)** | **34.82** | **51%** | **WINNER -- coherent** |

Captured-MTP decode DECREASES with spec (51%->37%->26% accept): the 1-layer MTP head over-drafts past ~3 tokens,
so spec=3 wins. (The old "spec=5 / climbing 50/57/63" was an artifact of degenerate garbage where draft==target gave
fake ~98% accept at any spec.) Shipped default = **spec=3 = 34.82 t/s coherent** = 1.92x vs captured-no-MTP,
3.3x vs eager-MTP, 8.5x vs eager-no-MTP. Accept is drafter-limited on hard prompts; code/easy prompts accept higher.

## The non-obvious ingredients (still wired in serve.sh)

1. **GDN kernel mount** -- :int8g bakes GDN OFF; mount the GDN-enabled `_xpu_C.abi3.so` (+ sibling lib).
2. **Combined shim** (`patches/sitecustomize.py` on PYTHONPATH, = `scripts/110_csag_shim`) does TWO things:
   (a) forces ONLY the `Qwen3_5MultiTokenPredictor` drafter unquantized/BF16 (else 0% accept); and
   (b) replaces `XpuCommunicator.all_gather` with an all-reduce-of-padded equivalent so the spec-verify all_gather
   records into the SYCL graph (oneCCL's native allgather can't) -- this is THE Bug B fix that lets MTP capture.
3. **`splitting_ops` = attention/GDN ops ONLY (eject nothing) + `IGP=false`** -- set in serve.sh. Do NOT add the TP
   collectives to splitting_ops: ejecting them is exactly what corrupted the captured path (Bug B).

## Host dependency (not in repo)

- **Model:** `/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft` (35 GiB W8A8 + 15 BF16 `mtp.*` grafted;
  config.json ignore is now the regex form, original at `config.json.ignore339.bak`).
- **GDN kernel:** `/mnt/vm_8tb/b70/vllm-xpu-kernels/vllm_xpu_kernels/{_xpu_C.abi3.so,libgdn_attn_kernels_xe_2.so}`.

## Open work (Bug B -- recover a fast coherent path)

Fix the captured-int8 + BF16-GDN + TP=2 numerics so capture is coherent (would restore the ~5x capture speedup on
top of eager). Leads: per-token dynamic int8 quant buffer aliasing across piecewise-split-at-collective boundaries;
or the BF16 GDN linears inside captured pieces. 14B single-card captures fine -> isolate the TP=2 delta.
