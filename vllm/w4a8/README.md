# vllm/w4a8 -- Qwen3.8-27B W4A8 research (not a shelf)

Standing prompt: `docs/20260820_qwen38_w4a8_campaign.md`.
Ledger: `docs/20260820_qwen38_w4a8_loops.md`.

This directory is the vLLM-side working tree for the 3.8 W4A8 full-send.
Do not promote `rdy_to_serve/vllm/qwen38-27b-w4a8/` until the campaign's
K19 + smoke + coherence + a measured win.

3.6 shelf (baseline, different model): `rdy_to_serve/vllm/qwen36-27b-w4a8/`.
Producer: `scripts/151_quantize_qwen38_27b_w4a8.sh`.
Default first artifact: `models/files/qwen3.8-27b/w4a8-rtn-gdn` (DATAFREE=1).
  On disk 2026-08-21: 20.616 GiB, 256 int4-packed MLP/attn + 144 GDN I8, vision 333 + mtp 15 grafted.
  Config is `Qwen3_5ForCausalLM` (151 loaded the text model). Census: `results/logs/k0_census_w4a8_rtn_gdn_20260821.txt`.
Calibrated: `models/files/qwen3.8-27b/w4a8-gptq-gdn` (DATAFREE=0). Not produced yet.

New bench/serve scripts for 3.8 live here. Copy, do not rewrite, the 3.6
shelf serve.sh.

K1 isolated matrix: `bench_w4a8_shapes.py` + `run_k1_matrix.sh` (card 1,
3.8 shapes, 3.6 w4a8-sqgptq stand-in). Does not bake an image.
