# Qwen3.6-27B W4A16 (compressed-tensors) -- WIP, NOT ON THE SHELF YET

> [!] STATUS: serves HEALTHY but generates GARBAGE ("!!!!") -- a numerical bug in the compressed-tensors
> W4A16 XPU GEMM (`int4_gemm_w4a16` / `XPUwNa16`). DO NOT use for real inference yet. Full investigation:
> `docs/kernel/22_compressed_tensors_w4a16_xpu.md`.

This checkpoint is a TEXT-ONLY Qwen3.5 quant (`architectures=["Qwen3_5ForCausalLM"]`, no vision tensors).
The `patches/` shim makes vLLM load it text-only as a hybrid (GDN) model -- four structural blockers fixed
(vision-tower / mamba_block_size / mamba-state methods / M-RoPE); see the kernel doc. What remains is the
int4 linear kernel producing garbage.

## Run (on the GPU host, ONE card)
```bash
ssh root@192.168.10.5 && cd /mnt/vm_8tb/b70/rdy_to_serve/qwen36-27b-w4a16
/mnt/vm_8tb/b70/gpu-run --card 0 bash serve.sh        # loads + serves, but output is garbage (WIP)
bash serve.sh stop
```

## patches/ (the text-only-hybrid load shim, pinned to :v0230)
- `sitecustomize.py` -- registers the `Qwen3_5ForCausalLM` arch (on PYTHONPATH at startup).
- `qwen35_text_hybrid.py` -- the marker subclass: `is_hybrid=True`, grafted GDN-state classmethods,
  `supports_mrope`=True + text-only positions; (currently also force-wires the dequant kernel for debugging).
- `xpu_wna16_dequant.py` -- the dequant fallback kernel (correct but ~4x memory; does not fit one card for 27B).

verified: LOADS + serves HEALTHY on :v0230 (text-only hybrid, 2026-06-23); OUTPUT GARBAGE (int4 kernel bug).
NOT verified for correctness. Open: fix `int4_gemm_w4a16` layout (see kernel/22).
