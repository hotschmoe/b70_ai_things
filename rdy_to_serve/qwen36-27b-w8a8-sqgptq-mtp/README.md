# qwen36-27b-w8a8-sqgptq-mtp

Qwen3.6-27B **W8A8** (compressed-tensors INT8 weights x INT8 activations, SmoothQuant+GPTQ) + a **BF16 MTP
graft**, served **TP=2 + MTP spec=5** on 2x B70.

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

### Bug B -- the captured path is numerically BROKEN on this TP=2 hybrid (NOT fixed; recipe defaults EAGER)
With the GDN now correctly BF16, PIECEWISE graph capture is broken here:
- `use_inductor_graph_partition=true` -> `KeyError: weight_scale` at profile/capture (the partitioner packs a
  weight_scale placeholder for a region that MIXES W8A8 (has scale) + BF16 GDN (no scale) linears).
- `use_inductor_graph_partition=false` (legacy piecewise, lib.sh `IGP=false`) -> capture succeeds but the decode is
  **numerically garbage even WITHOUT MTP** (clean-cache confirmed). 14B W8A8 captures coherently single-card, so this
  is a **TP=2 + BF16-GDN-in-captured-pieces + custom-int8** capture-numerics bug, to be chased separately.
- **=> recipe DEFAULTS TO EAGER (`GRAPH=0`), the only coherent path.** `GRAPH=1` reproduces the broken captured path.

## Honest result (coherent, EAGER, temp=0 greedy)

| config | decode tok/s | accept | note |
|---|---|---|---|
| eager, MTP-off (pure body) | ~4.1 | - | TP=2 fully-eager int8 is launch/collective-bound |
| eager, MTP spec=5 | ~9.0-9.6 | ~48% (accept_len ~3.9) | **2.3x vs MTP-off**, coherent |
| captured (GRAPH=1) | (was "63") | - | **BROKEN -- garbage; do not use** |

So the coherent W8A8 27B TP=2 serve is **correct but slow (~9 t/s)**. The fast number requires the captured path,
which is currently broken. If you need a fast *coherent* 27B int8-activation serve today, the single-card
**W4A8** (`../qwen36-27b-w4a8`, captures coherently) is the better pick until Bug B is fixed.

## The non-obvious ingredients (still wired in serve.sh)

1. **GDN kernel mount** -- :int8g bakes GDN OFF; mount the GDN-enabled `_xpu_C.abi3.so` (+ sibling lib).
2. **MTP-BF16 shim** (`patches/sitecustomize.py` on PYTHONPATH) -- forces ONLY the `Qwen3_5MultiTokenPredictor`
   drafter unquantized/BF16. (Eager MTP works; the shim is correct.)
3. **`splitting_ops` / `IGP` knobs in ../_common/lib.sh** -- relevant only to the (currently broken) captured path.

## Host dependency (not in repo)

- **Model:** `/mnt/vm_8tb/b70/models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft` (35 GiB W8A8 + 15 BF16 `mtp.*` grafted;
  config.json ignore is now the regex form, original at `config.json.ignore339.bak`).
- **GDN kernel:** `/mnt/vm_8tb/b70/vllm-xpu-kernels/vllm_xpu_kernels/{_xpu_C.abi3.so,libgdn_attn_kernels_xe_2.so}`.

## Open work (Bug B -- recover a fast coherent path)

Fix the captured-int8 + BF16-GDN + TP=2 numerics so capture is coherent (would restore the ~5x capture speedup on
top of eager). Leads: per-token dynamic int8 quant buffer aliasing across piecewise-split-at-collective boundaries;
or the BF16 GDN linears inside captured pieces. 14B single-card captures fine -> isolate the TP=2 delta.
