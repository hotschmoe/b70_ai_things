# F03a Qwen3.8 official-FP8 shared-cache positive

Date: 2026-08-30

Status: passed. Reusing one compiled-artifact set made two fresh Work.wait
server processes exact on all 12 natural prompts. Fresh compilation, not
request state, oneCCL completion ownership, or later process initialization,
selects the target output.

## CONFIG

- Git harness identity: `f33b223`.
- Model: official `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Runtime and image: vLLM `0.27.2rc1.dev77+gac7509e2b`, PyTorch
  `2.13.0+xpu`, and local Work.wait image
  `sha256:dce80db0a1ad861145e88b1c565f29172641912dc75a3b50d08e370f7d58e291`.
- Installed communicator SHA256:
  `5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d`.
- TP2, P2P off, MTP0, XPU Graph off, deterministic Inductor on, FP16 target,
  KV dtype `auto`, one request, 1,024 context, and prefix caching off.
- Lifetime 1 created a new cache. After clean teardown and health, lifetime 2
  mounted that same cache and was required to use the same complete suite.
- Both containers had a 32 GiB memory limit with no additional swap allowance.
- Result directory:
  `/mnt/vm_8tb/b70/results/f03a_qwen38_fp8_neural/20260830T005000Z/`.

## COMMAND

Run the tracked wrapper through its self-acquired whole-box lease:

```text
STAMP=20260830T005000Z \
  bash vllm/fp8/qualify_qwen38_fp8_neural_f03a.sh
```

The wrapper verified all model and runtime bytes, ran pre-health, executed the
complete 12-prompt natural suite and independent canaries in each server,
gracefully tore down, and ran card plus compiled P2P-off collective health
after each lifetime.

## RESULT

All 12/12 complete raw output-token arrays matched across the two fresh server
processes. Cached prompt tokens were zero and both independent canary files
passed with SHA256
`f234e605954b061e7f902eb92dd96739722df5437cadd9b2aceed79b976e45f8`.

The cache-reuse evidence is explicit. Lifetime 1 reported 137.22 seconds in
`torch.compile`. Lifetime 2 reconstructed 21 standalone artifacts and 65
submodules on each rank, logged direct AOT loads for both rank models, and
reported 1.98 seconds total compile time. It therefore reused lifetime 1's
compiled model rather than recompiling into the same directory.

The two diagnostic class-balanced rates were 11.303540 and 12.081169 tok/s,
with an 11.692355 tok/s median and 6.651 percent spread. The arrays are
coherent, but the rate spread is too large for a stable speed claim. The
frozen target matched 8/12 prompts from F02 Work.wait attempt 1 and 6/12 from
attempt 2. That is expected evidence of compile-selected target identity, not
parity with an arbitrary earlier fresh compile.

The shared cache occupied about 302 MiB and contained 2,250 files after the
run. Its sorted relative-path/content manifest is persisted as
`cache-files.sha256` with SHA256
`ec1af4f6a06cc860da03e3bf7b359714efe6612e2b07d9083cb4cd30de19d64a`.
The result summary SHA256 is
`362c5b3ca2f5efaf53933cbf1e1f1723e1094b7de6c416907c6046af8024eabc`.

All pre/inter/post card and compiled P2P-off collective checks passed. Across
271 host samples, swap use remained zero, minimum MemAvailable was
113,127,392 KiB or 107.887 GiB, and container host-RAM use peaked near
7.718 GiB. Memory PSI `some` and `full` totals increased by only 34.646 and
34.576 milliseconds across the roughly 30-minute transaction. The kernel
journal and server logs had no configured OOM, fatal, hang, GPU-fault, reset,
or wedge marker.

## VERDICT

F03a passes. The local fresh-lifetime nondeterminism is a compile-artifact
selection problem. A pinned compiled cache is a valid deterministic target
control for subsequent local experiments; generating a new cache creates a
new target and must not be compared as though it were identical.

This does not authorize shelf promotion or direct-P2P serving. Proceed to F04
with the packed-RMS MTP1 overlay under P2P off, compare every speculative
output to this pinned MTP0 target, and apply the same shared-cache discipline.
Long-context, concurrency, and stable-speed gates remain open.
