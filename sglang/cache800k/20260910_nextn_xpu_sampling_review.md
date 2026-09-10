# NEXTN XPU sampling review, 2026-09-10

CONFIG -> Pinned SGLang main 2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed,
Qwen3.5 NEXTN (resolved to EAGLE), XPU, temperature 0.7. CPU/source only.
No image or build-tree mutation, GPU calls, serving, or kernel implementation.

COMMAND -> Inspect the pinned speculative worker, eagle_sample, ordinary sampler,
UNO target-only sampling, and speculative startup gates. Run
`python3 sglang/cache800k/nextn_sampling/test_source_semantics.py`.
Dry-run the adjacent source patch against the pinned tree with `patch -p1`.

RESULT -> Five CPU tests pass: actual AST verifier predicate; actual isolated
rejection-startup gate; a temperature-sampling counterexample; exact rational
enumeration of sampled-ID verification; and candidate fail-closed checks.
Source patch dry-run and Python compilation pass. A no-device, no-network R276
container (2 CPUs, 4 GiB, image 521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad)
executes the exact ordinary PyTorch helper source with torch 2.13.0+xpu on CPU:
40,000 draws give P(token1)=0.80815 versus expected 0.80668 at temperature 0.7;
request-row expansion, top-k1/top-p exclusion, and shape gates pass.
`test_torch_helpers.py SOURCE_ROOT HELPER_PATH` reproduces this with CPU Torch.
These are source/reference and CPU tensor checks, not XPU sampler or Triton
verifier execution.

VERDICT -> Stock pinned main does not preserve temperature-0.7 sampling through
NEXTN verification on XPU. A bounded source candidate is prepared; it is NOT
full sampling parity and is NOT GPU-qualified.

## Confirmed source behavior

- `eagle_utils.py:eagle_sample`, line 784, selects the argmax branch when
  `is_all_greedy or _is_cpu or _is_hip or _is_xpu`. On XPU, temperature, top-k,
  and top-p target sampling in the later stochastic branch are unreachable.
  Static logit bias, existing penalties, and grammar masks run before argmax.
- This affects verify continuations including the verifier's trailing token.
  It does not imply that every token in the response is greedy: the target
  prefill path in `eagle_worker_v2.py:1263` calls the ordinary target worker
  sampler and passes its sampled first token into draft extension.
- NEXTN resolves to EAGLE in `arg_groups/speculative_hook.py`. The rejection
  startup gate at line 923 accepts EAGLE/topk1/thresholds1 without checking XPU.
  Setting `--speculative-use-rejection-sampling` does not override the earlier
  XPU argmax branch. It can alter draft proposal sampling while target verify
  remains greedy. The CPU test executes this isolated gate, not full startup.
- `verify_tree_greedy_func` dispatches XPU to the existing Triton tree walker.
  Its input is target token IDs; argmax is a caller choice. The existing UNO
  branch already demonstrates sampled target IDs followed by this walker.
- Ordinary `layers/sampler.py` has PyTorch probability sampling helpers for
  categorical and top-k/top-p/min-p sampling. Reusing these avoids the CUDA
  `sgl_kernel` target-only verifier. The existing Triton linear rejection
  kernel is another future port option, but changing dispatch alone does not
  qualify its XPU imports, compilation, RNG, or collective behavior.

Source permalink:
https://github.com/sgl-project/sglang/blob/2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed/python/sglang/srt/speculative/eagle_utils.py#L784

## Deliberate bounded candidate

Files in `nextn_sampling/`:

- `xpu_target_sampling.py`: install as
  `sglang/srt/speculative/xpu_target_sampling.py` in a separate candidate.
- `sglang-main-xpu-target-sampling.patch`: applies only to pinned eagle_utils.
- `test_source_semantics.py`: CPU source/reference checks.

For non-greedy XPU, validate host metadata before mutating logits, expand each
request's temperature/top-k/top-p by verify width, sample every adjusted target
row with ordinary PyTorch helpers, and pass sampled IDs to the existing walker.
Use its existing mismatch/trailing-token and accepted-index contracts unchanged.
Broadcast predict/accept_index/accepted-count outputs from TP rank zero, matching
the existing stochastic branch. Preserve the existing greedy path, including
seed-42 deterministic controls. No device .item(), host copies, or new sync.

The candidate accepts only unseeded homogeneous non-greedy requests with draft
topk1 and ordinary temperature/top-k/top-p. It explicitly rejects seeded
non-greedy requests, mixed greedy/non-greedy batches, min-p, dynamic penalties
(including min-new-tokens), custom processors, logprob reporting, rejection flag,
and simulated acceptance. Existing static bias and verify grammar masks are
retained, but grammar compatibility still requires runtime qualification.

For independently sampled target rows, conditioning on an accepted draft prefix
leaves the next target sample distributed as the target conditional. A mismatch
emits that target sample and stops the speculative segment; a full match emits
the final target sample. Exact enumeration of a binary two-token reference
matches ordinary autoregressive probabilities. Acceptance may be lower than
p/q rejection sampling. This proof assumes correct conditional logits/masks,
independent target draws, and no relaxed dynamic-penalty discrepancy.

Important remaining work:

- Test actual PyTorch XPU helpers and existing Triton walker, accepted-count and
  bonus handling, numerical distribution, TP broadcast, graphs, and concurrency.
- Seeded sampling is not a trivial tensor repeat: ordinary sampler hashes seed,
  absolute position, and vocabulary column, uses FP64, and rejects seeded min-p.
  The ordinary
  sampler hashes the INPUT token position: decode uses forward_batch.positions;
  prefill uses seq_lens-1. A future chain port should match actual verify input
  positions, not blindly add one for the generated token. Sibling/shared-depth
  and TP RNG semantics require explicit tests.
- Existing verify penalties are documented as relaxed and frozen over all draft
  rows. Exact non-MTP dynamic-penalty parity needs prefix-aware corrections.
- Do not call Sampler.forward directly with request metadata and expanded verify
  rows: row counts differ and it repeats custom preprocessing.
- Unsupported requests currently fail in verification, after prefill; add an
  API admission gate before operational deployment to reject before streaming.

## Actual Pi evidence

The redacted manifest records temperature 0.7 for all three affected agents and
seeds 843240016, 1160881401, and 2120442760. It does not contain their actual API
wire payloads, so these may be harness seeds rather than forwarded API sampling
seeds. The current bounded candidate rejects the latter. The candidate checks both SamplingBatchInfo.sampling_seed and original request
sampling_params.sampling_seed: upstream only constructs the former when
`enable_deterministic` is true. The original request check prevents an explicit
seed silently slipping through a nondeterministic server configuration.

No explicit top-k,
top-p, min-p, or dynamic-penalty scalar was found in the parsed JSON evidence;
absence is not proof of effective API defaults.

Raw evidence:
`/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-nextn-sampling/`.

## Seeded XPU source assessment (not implemented)

`kernels/ops/sampling/murmur_hash.py` routes CPU to a native CPU op and all
non-CPU devices to Triton, with no XPU-specific rejection. It splits uint64 seed
into two uint32 words, mixes input position and vocabulary column with wrapping
32-bit operations, and returns uint32 hashes. `multinomial_with_seed` converts
to FP64 Gumbel noise and adds FP64 log probabilities. Neither XPU compilation
nor FP64 support/performance is established by this CPU review; require a leased
small arithmetic/hash and sampler oracle before enabling it on B70. The CPU op
is a useful bit-exact hash reference, but copying a CUDA-only binary is not a port.

Matching temperature/top-k/top-p distributions does not imply identical random
sequences across vLLM and SGLang. A seed port should first reproduce SGLang's own
non-MTP seeded sequence under unchanged input positions and filtering, then
state the remaining cross-backend reproducibility limit.

Independent review additionally interpreted the actual Triton walker AST with
CPU scalar operations: two request cases cover permuted retrieve indices,
complete draft acceptance plus final bonus, and immediate rejection in the
second request. This preserves the real pointer/index expressions while not
executing Triton compilation or GPU code. Evidence is in adjacent raw directory
`sglang-nextn-sampling-review/{walker_cpu.py,walker-results.json,REPORT.md}`.
The reviewer found no remaining blocking source issue within the explicitly
restricted unseeded chain scope after the original-request seed gate fix.
