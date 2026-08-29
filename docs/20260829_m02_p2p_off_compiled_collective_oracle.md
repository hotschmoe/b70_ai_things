# M02 P2P-off compiled collective oracle

Date: 2026-08-29

Status: passed. This is an isolated collective result, not a model-serving or
performance qualification.

## Configuration

- Host kernel: `7.1.0-070100-generic`.
- Host Compute Runtime package: `26.22.38646.4-0`.
- Image: `vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f`.
- vLLM: `0.27.2rc1.dev77+gac7509e2b`.
- PyTorch: `2.13.0+xpu`.
- oneCCL: `89438cc`; `libccl.so.1.0` SHA256
  `733980ab6a6eb15d2d3da0649b92052c64a9597ced48fe9188434face5298b35`.
- oneCCL kernels SHA256:
  `0d549c35a558f1b216cb7d1efeaa9f86d7596ffc47b383644e075290d314f0c9`.
- Image UMD: `libze_intel_gpu.so.1.15.39122`, SHA256
  `0759d9a4beb9746f8a49c3e8f9b98376b49d04508b26827b46c0c5d40b030eb0`.
- World size: two B70 ranks; `CCL_TOPO_P2P_ACCESS=0`.
- Dtype: BF16 with small integer-derived values for exact equality.
- Shapes: all-reduce `[1,5120]` and `[4,5120]`; all-gather input
  `[4,2560]`.
- Consumer: `collective_output * 2 + 1` immediately after completion.
- Source commits: base oracle `d516f79`; graph-safe all-gather route `d18ceec`.

The tracked oracle and launcher SHA256 values are
`1f2702b8826c169ddba504564c11b7a58cb1b36fc73774cb7f5b658db40e6123`
and `52fd9aa4fffaccd93644bc97dbdbc9ee191292b6ff9e7413ce5c97d1d040940c`.

## Negative boundary

The first lifetime used functional `all_gather_tensor` plus `wait_tensor`
inside compiled XPUGraph capture. Eager and compiled non-graph execution were
exact, and both all-reduce shapes were exact through graph replay. Both ranks
then rejected all-gather capture with:

```text
RuntimeError: wait method cannot be used for an event associated with a command graph.
```

This was a synchronized framework error, not a device loss or hang. The
container tore down and card plus compiled P2P-off collective post-health
passed. The mandated `bin/xe-reset --method rebind` then completed with the
same boot ID and clean card/collective health.

Disposition: functional completion is valid for compiled all-gather outside
XPUGraph but is not a graph-safe all-gather representation on this stack.

## Passing route

The second source revision keeps compiled all-gather opaque through a custom
op whose implementation issues direct preallocated oneCCL
`all_gather_into_tensor`. This prevents Inductor from lowering the graph body
to a functional collective plus an illegal event wait. All-reduce retains the
functional `wait_tensor` route because it captured and replayed correctly.

Each of three fresh process-group/container/compile-cache lifetimes ran, per
rank and per shape:

| Mode | Iterations | Completion route | Result |
| --- | ---: | --- | --- |
| Eager direct | 8 | Blocking direct collective | Exact |
| Compiled | 8 | Functional collective plus `wait_tensor` | Exact |
| Compiled XPUGraph replay | 16 | Functional wait for all-reduce; opaque direct custom op for all-gather | Exact |

Warmup and capture entry/return records bring each rank to 102 monotonically
numbered calls per lifetime. Rank signatures matched, no call remained open,
and all dependent-consumer comparisons were exact. Every fresh container
disappeared after use. Card and compiled P2P-off collective health passed
before the matrix and after each lifetime.

Accepted combined result SHA256 values:

- lifetime 1: `041b5d57729061b1650b8f36c6139488ff95edc01f4e50b45ff267485f26acf6`
- lifetime 2: `e74ac321adea9565217dd33b75d8db210993f200588ff02fe13f9f1b43746be8`
- lifetime 3: `f3c59fbab483bffe1a364b9187358074ff08bed18e9216fde80b525e68dabe26`
- sorted 15-file evidence manifest: `706efae9d72c77e643fb6ea6b7350af6d128ffb02aec52c645ec4ed3e1f7a8bd`

External evidence directory:
`/mnt/vm_8tb/b70/results/m02_p2p_off_collective/20260829T055808Z/`.

## Verdict

M02 passes for the required BF16 P2P-off shapes, direct and compiled execution,
replayed graphs, matched rank entry/return, dependent consumers, fresh
lifetimes, teardown, and health. The pass is route-specific: do not place the
functional all-gather event wait inside XPUGraph. Use the opaque direct
all-gather boundary for a later model port.

Proceed to M03 as an all-reduce completion-ownership A/B: blocking c10d versus
`async_op=True` plus `Work.wait()`, P2P off, with an immediate consumer. M02
does not establish that the async route is faster or safer in a model.
