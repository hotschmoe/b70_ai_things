# Qwen3.6-27B W4A16 (compressed-tensors, int4 weight / 16-bit act) -- TEXT-ONLY

Serves `Qwen3.6-27B-W4A16` (compressed-tensors `pack-quantized`) on ONE Intel Arc Pro B70. This is the
**compressed-tensors** 27B (our format-parity target -- the substrate for future W4A4 research). The
checkpoint is a LANGUAGE-MODEL-ONLY quant of the Qwen3.5 VL model (no vision tensors). Full debugging story:
`docs/kernel/22_compressed_tensors_w4a16_xpu.md`.

> NOTE: the int4 weight-only path here uses the stock XPU `int4_gemm_w4a16` kernel (verified correct). The
> AutoRound int4 27B (`../qwen36-27b-int4/`) decodes faster (different kernel) -- use that for the daily
> driver; use THIS one for compressed-tensors parity / W4A16 research.

## Run (on the GPU host, ONE card)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen36-27b-w4a16
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh     # one card; leaves the other free for experiments
bash serve.sh stop
```
Served id: `qwen36-27b-w4a16`.

## [!] Image: `vllm-xpu-env:v0230` + the text-only-hybrid load shim (`patches/`, on PYTHONPATH)
This checkpoint declares `Qwen3_5ForCausalLM` (text-only) with all weights under `model.language_model.*`,
but vLLM's registry only knows the VL `Qwen3_5ForConditionalGeneration` -> it built a (weightless) vision
tower and, worse, silently skipped EVERY weight (name mismatch) -> ran on random init. The shim fixes FIVE
things so it loads + computes correctly (pinned to vLLM 0.23):
1. registers the real text class `Qwen3_5ForCausalLM` -> no vision tower built;
2. `is_hybrid=True` marker -> the GDN/mamba KV-cache setup runs;
3. grafts the VL class's `get_mamba_state_{shape,dtype,copy}_from_config` (GDN state);
4. `supports_mrope=True` + a text-only `get_mrope_input_positions` (== the VL text path);
5. **`load_weights` remap `model.language_model.` -> `model.`** -- THE fix for the garbage output
   (without it all weights skip -> random init -> "!!!!").

verified: GREEN on :v0230, one card, 2026-06-23: skipped-weight warnings = 0; coherent gens ("...Paris is
the capital and most populous city of France"; "The ocean is a vast... covers more than 70% of the Earth's
surface"). Re-verify: `gpu-run --card 0 bash serve.sh`. (A full HumanEval+ pass is the natural next check.)
