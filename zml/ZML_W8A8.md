# ZML W8A8 (int8) for qwen3.5-27b -- implementation handoff

Handoff brief for a long-running research + implementation + testing agent. Goal: bring up a TRUE W8A8
(int8 weights + int8 activations) serve path for qwen3.5-27b (`Qwen3_5ForConditionalGeneration`, our
qwen3.6-27b) in ZML on the dual Intel Arc Pro B70, developed and numerically validated FIRST on the XLA
**CPU** backend (no GPU, daily driver untouched), then ported to oneAPI for the B70 INT8-XMX payoff.

Status when written (2026-06-30): scaffolding only. zml itself is bf16/f16 today and has NO int8 linear
path; this is a greenfield implementation. Companion docs: `zml/W8A8_FEASIBILITY.md` (zml/XLA int8 API
source analysis -- READ FIRST for the exact `dot`/`convert`/scale API and the can-XLA-do-i8xi8->i32
verdict), `zml/REVIEW_intel_arch.md` (zml oneAPI status), `docs/intel_support_per_backend.md`,
`docs/patch_applicability_matrix.md`.

## Progress log (update as milestones land)

- **M0 -- DONE (2026-06-30).** `//examples/w8a8` (repo: `zml/examples/w8a8/`, synced to the build clone
  via `zml/apply_examples.sh`). CPU, zero library change. Two gates PASS: i32-accumulation bit-exact
  (2048/2048; acc values >> i16 range so an i8/i16 result dtype would have wrapped) + dequant rel_l2
  0.00717 vs an independent f32 reference. Proves zml expresses the W8A8 graph + i32 accumulation on CPU.
  See `zml/examples/w8a8/README.md` and JOURNAL 2026-06-30.
- **M1 -- DONE (2026-06-30).** `examples/llm/models/common_quant.zig` (reusable
  `QuantizedLinear`, a drop-in for `zml.nn.Linear`: same field/forward shape, adds a
  `weight_scale` tensor + optional bias, generic over the contracting-axis tag). Parity test
  `//examples/llm:quant_tests` vs the real `nn.Linear` fed dequantized weights: rel_l2 0.00701
  for both the bias and no-bias paths (< 0.02 tol). See JOURNAL 2026-06-30.
- **M2 -- DONE (2026-06-30).** `//examples/llm:quant_load_probe` loads a real full-attention
  q_proj (I8 `[12288,5120]` + BF16 weight_scale `[12288,1]`) from `w8a8-sqgptq` into
  QuantizedLinear via zml's stock safetensors + TensorStore reflection -- NO loader rewrite,
  and LAZY (only the bound projection streams, not the 35GB). Validated layers 3 & 7 (rel_l2
  ~0.0082 vs in-graph dequant ref, all finite); a GDN layer errors cleanly. Confirmed the
  artifact details: full-attn layers 3,7,11,...,63; weight_scale is `[out,1]` (handled by a
  trailing-singleton squeeze in QuantizedLinear). See JOURNAL 2026-06-30.
- M3 -- pending (wire into qwen3_5 full-attention layers; block-level parity).
- M4 -- pending, the go/no-go GPU perf gate (add `Tensor.dotGeneralAcc(.i32)`; measure INT8-XMX lowering).
- M5 -- pending (TP=2).

## 0. Why this is worth doing / why it's hard

- Standing project target: "W8A8 INT8 of ANY model" -- 8-bit weights preferred, B70 has INT8 XMX fast
  paths. Our sglang W8A8 (fused oneDNN int8 kernels + MTP) is the daily driver (decode ~25 t/s,
  HumanEval+ 0.970/0.933). The zml angle is a SECOND, compiler-driven int8 path: if XLA/oneAPI lowers
  `s8 dot_general` to INT8 XMX, we get int8 GEMM "for free" from the graph compiler, plus zml's
  compiler-visible collectives (no graph-break-around-collective cost -- see patch_applicability_matrix).
- THE hard hinge (resolve before building the model): does the **oneAPI PJRT plugin lower `s8 x s8 ->
  s32` `dot_general` to the B70 INT8 XMX/DP4A path**, or does it widen to bf16/f32 (correct but NO perf
  win -- then weight-only int8 is simpler and W8A8 buys nothing on zml)? CPU validates NUMERICS only;
  the XMX-lowering question must be MEASURED on the B70 (perf vs bf16). Treat M4 as the go/no-go gate.

## 1. The W8A8 scheme to implement (from our existing checkpoint -- this IS the spec)

Source of truth: `models/files/qwen3.6-27b/w8a8-sqgptq` (compressed-tensors, `version 0.17.1`,
`quant_method: compressed-tensors`, `format: int-quantized`). Scheme (config.json `quantization_config`):

- **Weights:** int8, `type=int`, `num_bits=8`, **symmetric**, `strategy=channel` (per-output-channel),
  `dynamic=false` (static, baked at quant time). Stored as `...weight` dtype **I8** `[out, in]` +
  `...weight_scale` dtype **BF16** `[out, 1]`.
- **Input activations:** int8, `num_bits=8`, **symmetric**, `strategy=token` (per-token),
  `dynamic=true` -> quantized AT RUNTIME from the bf16 activation (scale = max(abs(x)) over the in-dim /
  127, per token row). No zero-point (symmetric).
- **Output activations:** none (not quantized).
- **ignore:** `["lm_head", "re:.*linear_attn.*", "re:.*visual.*", "re:.*mtp.*"]`. So ONLY the
  full-attention q/k/v/o projections and the MLP gate/up/down are W8A8. The GDN linear-attention
  projections, the vision tower, the MTP head, and lm_head stay bf16. (This matches the qwen3_5 hybrid:
  full-attention layers every 4th; the 3/4 linear-attention/GDN layers keep bf16 projections.)
- SmoothQuant (`smoothing_strength: 0.8`, recipe.yaml) is ALREADY folded into the stored weights/scales
  (q/k/v share an input scale into input_layernorm; gate/up share post_attention_layernorm). The zml
  loader just consumes the final int8 weights + scales; do NOT re-smooth.

Tensor naming in the checkpoint (note `model.language_model.layers.N.` prefix):
`model.language_model.layers.0.mlp.down_proj.weight` (I8 [5120,17408]) +
`...down_proj.weight_scale` (BF16 [5120,1]). Same for q/k/v/o + gate/up.

### The W8A8 linear math (per quantized projection)
```
# weight (static): W_int8 [out,in], w_scale [out,1] (bf16)   -- from checkpoint
# activation x (bf16) [tokens, in]:
x_scale = reduce_max(abs(x), axis=in) / 127            # [tokens,1], per-token, dynamic
x_int8  = round(x / x_scale)  clamped to [-127,127]    # [tokens,in] i8
acc_i32 = dot_general(x_int8, W_int8^T) -> i32         # [tokens,out]  <-- the INT8-XMX target op
y_bf16  = acc_i32.to(f32) * x_scale * w_scale^T        # broadcast [tokens,1]*[1,out] -> [tokens,out]
y       = y_bf16 (+ bias if present, bf16)
```
This is the building block. M0/M1 below validate exactly this on CPU.

## 2. zml ground truth (verify against W8A8_FEASIBILITY.md)

- `zml/dtype.zig` DataType includes `i4, i8, u4, u8` + f8 variants -- int8 exists at the type level.
- `examples/llm/models/qwen3_5/model.zig` uses `zml.nn.Linear` for q/k/v/o, gate/up/down, lm_head, and
  the linear-attention in/out projections. A `QuantizedLinear` with the same forward signature drops in.
- zml runs bf16/f16 via XLA; loader streams per-shard into PJRT buffers (`zml/io.zig`
  DirectMemoryWriter). It does NOT consume compressed-tensors today -- the loader work is part of this.
- oneAPI multi-device TP=2 is PROVEN on the B70 (sharding example: PJRT enumerates both cards, Shardy
  num_partitions=2, collectives correct). A dense Llama TP=2 enumerates both cards but the first-run XLA
  compile exceeds a 600s smoke (use timeout ~1800s; XLA caches after).
- **VERDICT (W8A8_FEASIBILITY.md): true `i8 x i8 -> i32` IS expressible in zml** -- NOT weight-only-
  limited. The accumulation dtype is the StableHLO `dot_general` result type, and zml's low-level builder
  `dialects.stablehlo.dot_general` takes an explicit `result_type` (`mlir/dialects/stablehlo/stablehlo.zig:167-207`);
  s8 operands + s32 result is legal. Two concrete paths:
  - **CPU (M0-M3): ZERO library change.** Use `x_i8.convert(.i32).dot(w_i8.convert(.i32), .d)` -- the
    high-level wrapper sets result = operand = i32, giving genuine int32 accumulation. (`convert` =
    `zml/tensor.zig:1112`, `dot` = `:1200`.) The naive `i8.dot(i8)` is WRONG -- `Tensor.dotGeneral`
    hardcodes result dtype = operand dtype (`tensor.zig:1282`) so it would overflow in i8 (not a float
    fallback, just the wrong result dtype). So always convert i8->i32 before `.dot` on the CPU path.
  - **GPU INT8-XMX (M4): add a ~15-line `Tensor.dotGeneralAcc(out_dtype)` to `tensor.zig`** (mirror
    `dotGeneral` but set the result type to `.i32`) so TRUE s8 operands reach `dot_general` and the
    oneAPI compiler can pick a DPAS/INT8 kernel. Must live in tensor.zig (`mlirCtx()`/`currentBlock()`/
    `dialects` are private, `:4352/:4356`). The convert-to-i32 CPU trick would widen before the dot and
    lose the XMX win, so the helper is required for the GPU payoff.
  - Dequant ops all exist as public integer-capable `Tensor` methods: `abs` (:331), max-reduce (:3001),
    `div` (:1046), `mul` (:1041), `round` (:1185), `clamp` (:1191), `convert` (:1112), `broadcast`
    (:2262), `scalar` (:2179). Gotcha: i32 * f32 asserts equal dtypes (`binaryOp` :4308) -> `convert`
    the i32 accumulator to f32 before scaling. No existing quantized-matmul helper (only FP8 in the MoE
    Triton path, which returns `error.UnsupportedQuantization` for int8).

## 3. Implementation plan

M0. **CPU int8-dot microbench** (no model). New zml target modeled on `examples/sharding/main.zig`:
    construct random `W_int8`, `w_scale`, bf16 `x`; do the section-1 math; compare `y` to a pure-bf16
    `x @ dequant(W)` reference; report max-abs / relative error. Run on the CPU platform (no oneapi flag).
    ACCEPTANCE: error within int8-quant tolerance (per-channel+per-token symmetric ~<1-2% rel on random
    Gaussian). PROVES zml can express the W8A8 graph + that XLA accumulates i8 dot in i32 on CPU.
    READY: `W8A8_FEASIBILITY.md` has the full `//examples/w8a8` Zig sketch (QuantLinear vs RefLinear) +
    BUILD target. CPU path is ZERO library change (`convert(.i32).dot(...)`); start here.

M1. **QuantizedLinear struct** in zml (new file, e.g. `examples/llm/models/common_quant.zig` or inline):
    fields `weight: zml.Tensor` (i8 [out,in]), `weight_scale: zml.Tensor` (bf16 [out,1]), optional
    `bias`. `forward(x)` implements section 1. Unit-test its parity vs `zml.nn.Linear` on dequantized
    weights (CPU). ACCEPTANCE: per-call parity within tolerance.

M2. **Loader**: read the compressed-tensors int8 `weight` + `weight_scale` from
    `models/files/qwen3.6-27b/w8a8-sqgptq` into `QuantizedLinear` (the `model.language_model.layers.N.`
    naming; honor the ignore list -- GDN/vision/MTP/lm_head load as bf16 `nn.Linear`). Decide reuse vs
    re-quant: REUSE the existing int8 weights (SmoothQuant already baked) is strongly preferred over
    re-quantizing bf16. Confirm the bf16 `qwen3.6-27b/bf16` tokenizer/config drive the rest.
    NO LOADER REWRITE: `safetensors.zig:753` maps `I8->.i8`; `zml.io.load` (`io.zig:1074`) reflects over
    the model struct's Tensor fields, so a `QuantizedLinear{ weight:i8, weight_scale, bias, tag }` (bind
    via `createTensor`, mirror `initProj` `qwen3_5/model.zig:483`) loads automatically. CONFIRMED our
    artifact is UNPACKED I8 weight + BF16 weight_scale [out,1] (not the packed-int32 W4 layout zml can't
    parse); load weight_scale as bf16 then `convert(.f32)` for the dequant multiply.

M3. **Wire into qwen3_5**: in the model's `initProj`/layer build, choose `QuantizedLinear` for the
    full-attention q/k/v/o + MLP gate/up/down ONLY when the layer is full-attention AND the proj is not
    in the ignore list; everything else stays `nn.Linear` bf16. Run the 27B on CPU is impractical for
    full inference (54GB bf16-equivalent + slow), so validate M3 with a SMALL slice: a single
    full-attention block (or a tiny synthetic config) end-to-end on CPU, output parity vs the bf16
    block. ACCEPTANCE: block-level logit parity within tolerance.

M4. **oneAPI GPU (the go/no-go perf gate)**: PREREQ -- add the ~15-line `Tensor.dotGeneralAcc(.i32)`
    helper to `zml/tensor.zig` (see §2 / W8A8_FEASIBILITY.md) and switch `QuantizedLinear.forward` from
    the CPU `convert(.i32).dot` path to it, so TRUE s8 operands reach `dot_general` (the convert-i32 trick
    widens before the dot and forfeits INT8-XMX). Then build with `--@zml//platforms:oneapi=true`, run the W8A8
    qwen3_5 (text-only; vision/MTP not ported in zml) on ONE B70 first, then TP=2. MEASURE: (a)
    coherence (the "paris"/HumanEval gate), (b) decode t/s vs the bf16 zml run AND vs sglang W8A8 (~25
    t/s daily driver). INSPECT the compiled module / PJRT logs for whether `s8 dot_general` lowered to
    an INT8/DPAS kernel vs a widening fallback (the XMX question). If no XMX lowering -> W8A8 gives no
    perf win on zml; document and fall back to weight-only int8 or stop.

M5. **TP=2 W8A8** across both B70s (only after M4 single-card is coherent + the wedge discipline below).
    Compare to the sharding-proven collective path.

## 4. Validation methodology (CPU-first; this is the whole point of the request)

- ALL of M0-M3 run on the XLA CPU backend -- NO GPU, NO gpu-run lease, daily driver keeps serving.
  CPU build = default platform (do NOT pass `--@zml//platforms:oneapi=true`; `--@zml//platforms:cpu` is
  default true). `~/.local/bin/bazelisk run //examples/<target> -- <args>`.
- Numeric gates at each milestone (max-abs + relative error vs bf16 reference); a single repeated-token
  output is the classic broken-quant signature -- carry a coherence check (substring "paris" on the
  capital-of-France probe; degenerate-output detector like rdy_to_serve/_common gen-probe).
- Cross-check accuracy against the sglang W8A8 numbers (HumanEval+ 0.970/0.933) once M4 serves.

## 5. CRITICAL operational guardrails (the box is SHARED with the production daily driver)

- **Daily driver = the prod endpoint** (`http://192.168.10.5:18080`, sglang W8A8 TP=2 + Open WebUI :3000
  + Grafana :3001 + Prometheus :9090), systemd-managed: `b70-daily-driver.service` +
  `b70-dd-watchdog.service` (both enabled, boot auto-start). It holds BOTH cards.
- **CPU work (M0-M3) needs NO GPU** -- do all numeric dev on CPU while the daily driver runs. Only M4+
  need the GPU.
- **To use the GPU (M4+)**: bring the daily driver down with
  `cd /mnt/vm_8tb/github/b70_ai_things && ./vllm/daily_driver_serve.sh stop` (releases the lease; the
  dd-watchdog goes observe-only and will NOT fight you while b70_daily_0 is down). Run GPU work under
  `./bin/gpu-run` (holds the lease). Restore with
  `DD_API_KEY=$(cat /mnt/vm_8tb/b70/secrets/dd_api_key) ./vllm/daily_driver_serve.sh start`. No
  passwordless sudo for systemctl -- use the script, not `systemctl`. COORDINATE with the user before
  taking prod down.
- **No risky UNATTENDED TP=2** (M5): the W8A8 TP=2 BCS/oneCCL wedge recovers only by reboot
  (w8a8-mtp-enforce-eager-and-tp2-wedge). Run TP=2 attended; `bin/xpu-health` before/after, `bin/xe-reset`
  to recover.
- **bazel-under-gpu-run flock leak (KNOWN, will bite you)**: `bazelisk run` under the gpu-run flock
  spawns a 3h bazel daemon that INHERITS the lock fds and keeps the GPU lease HELD after the script
  exits, deadlocking the next gpu-run (incl. the daily-driver restore). The zml scripts now `bazelisk
  shutdown` + `set +e` at the end; a `timeout`-KILL still leaks it -> run `cd /mnt/vm_8tb/b70/zml &&
  ~/.local/bin/bazelisk shutdown` by hand and confirm `fuser /mnt/vm_8tb/b70/gpu.lock.{0,1}` is empty
  before restoring the daily driver. (TODO worth doing: a trap in the scripts.)
- **oneAPI runtime env**: `export CCL_TOPO_P2P_ACCESS=0` (oneapi.zig:33 defaults it from the wrong var ->
  garbage), `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
- **HF gating**: gated repos (meta-llama/*) 401 with our token; use non-gated mirrors (e.g.
  `unsloth/Llama-3.2-1B-Instruct`) for any reference model. Our qwen3.6 weights are already on disk
  under `models/files/` (no download).

## 6. Build / toolchain quick reference

- `~/.local/bin/bazelisk` (reads `.bazelversion` -> Bazel 9.1.1); zig at `~/.local/bin`. Workspace:
  `/mnt/vm_8tb/b70/zml` (git-ignored upstream clone, HEAD 89b0908c). Repo (this doc, scripts):
  `/mnt/vm_8tb/github/b70_ai_things/zml/`.
- CPU build/run: `cd /mnt/vm_8tb/b70/zml && bazelisk run //examples/<target> --config=release -- <args>`.
- oneAPI build/run: add `--@zml//platforms:cpu=false --@zml//platforms:oneapi=true`. Hermetic oneAPI PJRT
  plugin (amd64) is fetched by bazel; no system oneAPI needed.
- A bazel BUILD edit is needed to add a new example target (model on `examples/sharding/BUILD.bazel`).

## 7. References

- `zml/W8A8_FEASIBILITY.md` -- the zml/XLA int8 API source analysis + i8xi8->i32 verdict (READ FIRST).
- `zml/REVIEW_intel_arch.md` -- zml oneAPI status, TP layout, loader/attention/MoE notes to steal.
- `models/files/qwen3.6-27b/w8a8-sqgptq/{config.json,recipe.yaml}` -- the scheme + SmoothQuant recipe.
- `research/w8a8/` + `sglang/W8A8_BUILD.md` (if present) -- our proven sglang W8A8 (oneDNN int8 kernels)
  for accuracy/perf baselines and the int8-on-XPU lessons.
- `kernels/` -- the oneDNN int8 GEMM source (NOT reusable in zml/XLA, but documents the B70 int8-XMX
  shapes that win: docs/kernel int8 microbench).
- `docs/patch_applicability_matrix.md` -- why our int8 kernels/oneCCL code do NOT port to zml (XLA
  solves collectives in-graph; the int8 GEMM must come from XLA lowering, not our oneDNN .so).

## 8. Open questions / risks (rank-ordered)

1. (M4 gate) Does oneAPI PJRT lower `s8 dot_general` to INT8 XMX? If not, W8A8 has no perf upside on zml.
2. Per-token DYNAMIC activation quant inside the XLA graph -- cheap enough? (reduce_max + div + round +
   convert per linear; may or may not fuse well on oneAPI). Measure overhead vs the GEMM win.
3. Loader for compressed-tensors int8 + bf16 scales into zml's TensorStore (custom dtype/byte handling).
4. The qwen3_5 model in zml is text-only (no vision, no MTP) -- a W8A8 zml serve is a text/architecture
   experiment, NOT a drop-in daily-driver replacement (sglang W8A8 keeps vision+MTP). Scope accordingly.
5. Mixed precision in one TP=2 graph (bf16 GDN/vision + int8 full-attn/MLP) -- sharding + dtype interplay.
