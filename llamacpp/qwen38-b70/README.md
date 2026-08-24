# qwen38-b70 production build overlay

This directory owns local changes to the pinned 0xSero qwen38-b70 image while
preserving its two optimization-lab patches as immutable inputs.

Build order:

1. mndodd/llama.cpp commit `4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126`
2. `tp2-full-stack.patch`, SHA-256 `f21e9b557c3d024527ac98d5f189cf7ea72fa8c38a5faf2a22ee339fd1988998`
3. `q4k-increment.patch`, SHA-256 `0a27858525f6a402cf9c92d1b93daee0a80e2ffaef9137bf7bce784a549b58b6`
4. repo-owned `patches/quant-census.patch`
5. repo-owned `patches/quant-timing.patch`

`build_image.sh` verifies the first two patch digests, stages a temporary Docker
context, and builds `qwen38-b70:quant-timing` by default. It does not mount a GPU.
The new image must pass the normal coherence and matched-throughput gates before
it can replace `qwen38-b70:latest` or its pinned shelf image ID.

The new instrument is disabled by default. Set
`GGML_SYCL_QUANT_CENSUS=1` on a short, gracefully stopped evidence serve. At
exit it prints machine-readable `[QUANT-CENSUS]` rows for:

- the exact standard quantized MUL_MAT route after branch-local reorder;
- each actual per-device operation callback, keyed by quant type, algorithm,
  width, K, row slice, reorder state, and tensor-split state.

The `actual` rows count operation callback invocations, not individual SYCL
kernels submitted inside one callback. No event timing, queue barrier, or wait
is added. Graph-level fused MMVQ operations bypass this instrument; that is
explicit in `scope=standard_mul_mat`, and the production `LAB_DOORS=0` path
disables those fusions.

Parse a captured server log with:

```bash
python3 llamacpp/parse_quant_census.py server.log --write census.json
```

## Sampled device timing

`GGML_SYCL_QUANT_TIMING_SAMPLE=N` enables queue profiling and samples each
per-device quant callback about once every N calls. A stable hash rotates the
sample phase by quant, algorithm, shape, and device so the added barriers are
spread across the run. The default is `0`, which leaves queue construction and
the callback path unchanged.

Each sample submits a start barrier, runs the normal callback, then submits an
end barrier on the same in-order queue. It does not wait or query timestamps in
the request path. Graceful process exit reads completed event timestamps and
reports:

- callback device time as `command_start(end) - command_end(start)`;
- the two barriers' own device duration separately;
- incomplete and invalid timestamp counts rather than silently accepting them.

This interval includes every kernel or device copy submitted by that callback,
but excludes Q8 activation quantization performed before the callback, output
copies performed after it, collectives, and the boundary barriers themselves.
Concurrent submissions to the same queue would also fall inside the interval,
so use a single-request evidence arm.

Recommended first gate:

```bash
IMG=qwen38-b70:quant-timing IMG_ID=<candidate-image-id> \
GGML_SYCL_QUANT_CENSUS=1 GGML_SYCL_QUANT_TIMING_SAMPLE=128 \
GGML_SYCL_QUANT_TIMING_SKIP=4 GGML_SYCL_QUANT_TIMING_MAX=65536 \
NAME=qwen38_xl_quant_timing LAB_DOORS=0 ENABLE_MTP=0 \
./bin/gpu-run bash rdy_to_serve/llamacpp/qwen38-27b-ud-q4-k-xl/serve.sh start
```

Run a 256-token, temperature-zero, single-request decode after normal warmup,
then stop the container gracefully so the at-exit rows are emitted. Parse with:

```bash
python3 llamacpp/parse_quant_timing.py server.log --write timing.json
```

This is an evidence arm, not a throughput arm. Gate it by requiring nonzero
samples on both devices, zero `incomplete`, zero `invalid`, zero `dropped`, and
agreement between timing `calls_seen` and counts-only actual census cells. Then
repeat with sample periods 64 and 256: dominant type/algo/shape ordering should
be stable before using projected shares to choose kernel work.

## Automated XL qualification campaign

After building the candidate image, run the fail-closed six-arm campaign under
the external two-card lease:

```bash
./bin/gpu-run bash llamacpp/01_qwen38_ud_q4k_xl_quant_timing_campaign.sh full
```

The current-image ID is pinned to the production shelf ID. Optionally pin the
freshly built candidate too with `CANDIDATE_IMG_ID_EXPECTED=sha256:...`; either
image tag changing during the campaign is rejected before the next arm.

The campaign compares the timing-disabled candidate with the current image,
then runs counts-only and sample-period 64/128/256 evidence arms. All arms use
UD-Q4_K_XL, TP=2, MTP off, `LAB_DOORS=0`, temperature zero, one 32-token warmup,
and fixed 256-token decode requests. It records image IDs, code and patch
hashes, model identity, container configuration, raw logs, parsed evidence, and
the final gate report.

The runner does not call `gpu-run` itself: it verifies that its process tree
already owns both leases. It never tags an image, never changes the production
image pin, refuses to stop an unrelated endpoint, gracefully stops every arm,
and leaves port 18080 down. There is intentionally no restore action.

Use `metadata` for a GPU-free image/config manifest only:

```bash
bash llamacpp/01_qwen38_ud_q4k_xl_quant_timing_campaign.sh metadata
```

Run the CPU-only fixture test with:

```bash
python3 -m unittest discover -s llamacpp/tests -p 'test_*.py'
```
