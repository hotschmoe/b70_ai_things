# P1 selected-expert dequant primitive

An explicit XPU custom op: `torch.ops.b70_p1.prefill_dequant_selected`.
No vLLM entry point and no automatic serving hook. It preserves C13's
quant decode, BF16 conversion and destination layout, while mapping the
source expert through an int32 selection vector instead of copying the
packed experts first. The selected indices must be in range; the caller
supplies the existing host-validated bucket plan. The current op does not
perform a device-to-host index-bounds check.

Native source is byte-identical to the measured isolated component-r1
build. The custom op and all native kernel templates are uniquely named
under b70_p1 / b70_p1_dequant. Dynamic ABI dependencies are the campaign's
pinned PyTorch XPU, C++11 ABI1, SYCL 2026.1 and shared decode.hpp source.
Rebuild against source when changing this ABI stack.

Build output stays at
`/mnt/vm_8tb/b70/build/vllm-flashnext/p1/_C.p1selected.so`.
From the campaign repository, a coordinated CPU build is:

```
B70_AGENT=P1 MAX_JOBS=4 bin/cpu-run bash vllm/flashnext/harness/p1_build.sh
```

The build forces disk scratch under p1/tmp (override P1_TMPDIR). After an
explicit physical-card1 slot is granted, the bounded component gate is:

```
TAG=component-r2 bash vllm/flashnext/harness/p1_primitive_window.sh
```

It checks actual traffic-snake card1 packs, all five real gate quant types,
both down types, 32 noncontiguous selected experts, and destination
modes0/1/2. Every bit must match F1 C13, including untouched sentinel
halves. Timing compares selected dequant with index_select plus dequant
at the same logical shape, using two warmups and median of five events.
No bmm or serving runs. Native pre/post health and source/SO identity are
recorded. The measured run took44s plus19s health each side. The primitive
proves component correctness and copy savings; no end-to-end prefill
speed or serving stability is qualified. See reports/3P1_prefill.md.

The next proposed serving candidate is a default-off branch in
b70_gguf_xpu/csrc/bucket_moe.cpp: preserve the host bucket plan and bmm
shapes, then call this primitive on gate/up/down original packs with
sel and expert_rows F/F/H, replacing their three index_selects. It needs
an explicit load and one experimental flag; no bucket/combine follow-on
is part of this plugin. That hook has not been installed or measured.

An isolated bucket-serving experiment is also prepared. The frozen-source
`generate_bucket_overlay.py` checks the original F1 source hash and exact
edit anchors, then emits a small native overlay into the disk build tree.
All duplicated kernels use namespace b70_p1_bucket, and the new op is
b70_p1::prefill_moe_bucketed. It preserves plan, bmm and combine and switches
only packed gather versus direct selected dequant. The same overlay binary
runs flag0 and flag1. Build via cpu-run/harness/p1_bucket_build.sh.

Before serving, the parent must grant the bounded card1 integration gate:
`TAG=bucket-r1 bash vllm/flashnext/harness/p1_bucket_gate_window.sh`.
It requires original-repeat, original-versus-overlay0 and original-versus-
overlay1 exact at real shapes/types. This gate is not yet measured.

The optional Python hook requires B70_P1_BUCKET_OVERLAY=1 and is installed
explicitly by harness/p1_api.py, including spawned EngineCore. It intercepts
only card1 native experts; B70_PREFILL_SELECTED_DQ defaults to0. No shared
model/plugin edit or automatic vLLM entrypoint is installed. The parent
schedules harness/p1_serve.sh twice with host flag0/1 and one frozen
P1_PREFILL_FIXTURES path. The wrapper also passes flags separately to the
container and enforces source/fixture/activation/content/lifecycle checks.
No full serving performance result exists yet.
