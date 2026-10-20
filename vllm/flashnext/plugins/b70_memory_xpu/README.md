# B70 memory plugin (X2)

Default: inert. No core source edits, new quantization, or GPU work at import.

- `B70_MEMORY_FIX=1`: dense shard/parameter aliases; discard unused merged
  dense placeholders; reuse dequant scratch per stream; use native PyTorch
  XPU memory info and reservation-peak headroom for consistent budgeting.
- `B70_MEMORY_REPORT=/absolute/directory`: write exact storage ownership,
  allocator snapshots, per-operation profile memory, KV specs and graph
  estimate. Profiling hooks are removed before real requests/capture.

Install editable in the campaign runtime venv with `pip install --no-deps
--no-build-isolation -e <this directory>`. The general plugin entry point
loads automatically. Set flags in every vLLM worker environment.
Packed embeddings also require the one-line fork constructor change saved
in `../../patches/x2-packed-embedding.patch`.

`test_cpu.py` and `test_linear_cpu.py` run without device mounts. Real GPU
checks must use the repository lease/health wrappers. The shared dequant
arena assumes a returned view is consumed before the next dequant on that
stream, as existing dense/grouped/MoE callers do. It does not support
holding multiple independently reusable dequant weight views on one stream.

Evidence and limitations: `../../reports/1X2_kv_budget.md`. Short arithmetic probes do not
qualify FP8 quality at 200K or establish serving speed/stability.
The measured FP8 token/head candidate failed short coherence and is
rejected; keep BF16 while diagnosing its first divergent attention op.
