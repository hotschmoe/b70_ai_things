# X1 -- Flash-Next MTP on vLLM

The qwen4_exp MTP drafter loads the Strata sidecar and runs under
FULL_DECODE_ONLY graphs. It is NOT qualified: full Flash-Next greedy
parity fails, and the no-MTP control itself fails exact repeatability.
The latest eventless, fixed-slot, uniform-Q8 control completes all
requests and its same-load repeat, then tears down with both cards
HEALTHY. The matched eventless MTP retry failed during startup; its corrected
retry was cancelled for the wave3 handoff before weight load. No
promotion or commit;
no daily-driver start or GPU reset.

## Weights and Strata behavior

The main ISTA IQ3_XXS GGUF has no MTP tensors. The DD selects a separate
sidecar, `/mnt/vm_8tb/b70/build/strata-ab/mtp-bf16/rt`:

| File | Bytes | Contents |
| --- | ---: | --- |
| dense.bin | 116099072 | Q8_0 dense projections, BF16 HC/router, F32 folded norms |
| dense.txt | 29 records | Runtime tensor layout |
| experts.bin | 707788800 | Relayout of Q2_0 expert codes/scales |
| ../mtp-q2_0.gguf | 889014272 tensor bytes | 31 original-layout MTP tensors |

The source inventory pins Qwen/Qwen3.8-Flash-Next revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`; mtp-manifest.json records
per-tensor hashes. The main model-lock json pins the target GGUF, not
this sidecar. References: `strata/flash-next/dd/hotschmoe-dd.json:43`,
Strata `src-ep/tools/mtp_rt.py:3`, mtp-inventory.json.

Strata `--spec 4 --mtp-max-t 4` means four verification rows: one known
target token plus up to THREE drafts. Concurrent batch MTP depth is two.
One decoder layer is reused across draft steps. Historical acceptance
examples are 12526/17457 (71.75%) and108/159 (67.92%); they are not a
matched average for the141 tok/s DD reference. References: Strata
`src-ep/src/program/generate.cpp:4495`, engine.log:383554/383614, and
`docs/20261016_night_lab.md:267`.

The FC stage uses pooled RMS over four HC streams, projects each stream
through fc_hidden, and broadcasts fc_embedding onto them. Mean-collapse
before projection is different math (`src-ep/src/core/mtp.cpp:906`).
The Unsloth eh_proj head is not the DD draft graph (G3 report). Strata
also restricts drafting to40525 vocabulary IDs in draft_vocab_en.bin;
this vLLM path currently evaluates the full248320-token head.

## Implementation and scope

Fork: `/mnt/vm_8tb/b70/build/vllm-flashnext/vllm`, b70/flashnext.
Other agents' dirty changes are preserved. Sources:

- `vllm/models/qwen4_exp/gguf/mtp_loader.py`: isolated Qwen4ExpMTP and
  Qwen3_5MoeMTP load, shape/payload audit, rt dense bytes, GGUF Q2_0
  experts, shared target vocab. Fused expert gate/up is split on CPU
  before upload, avoiding a redundant225MiB per EP rank (derived).
- `gguf/serve_loader.py`: early draft dispatch and PLE history install.
- `xpu/mtp.py`: packed vocab/FC quant methods; native wide FC retained;
  rt loading bypasses eh_proj. Folded norm gamma becomes deviation
  gamma-1 for the native modules.
- `gguf/mtp_ple.py`: position-indexed convolution/token history ring,
  sized history+verification width. Rejected future rows cannot destroy
  needed history. CPU preparation uses raw token IDs, explicit int64,
  positions, slots and query boundaries; captured forward has no new
  host reads. Long prefill, rejection, reorder and slot reuse are checked.
- `v1/worker/gpu/model_runner.py`: prepare PLE outside capture/replay;
  optional all-rank CPU readiness fence scoped to this PLE path.
- `harness/x1_serve_{proxy,flashnext}.sh`, corresponding window scripts,
  and x1_compare_greedy.py: identity, raw token gates, C1/C2, acceptance,
  RAM watchdog, leased health and teardown.

The Flash-Next recipe is BF16 activations, TP1/DP2/native EP, maxlen2048,
max2 sequences/rank, batch512, FA C7b overlay, no prefix caching,
FULL_DECODE_ONLY capture sizes1/2/3/4/6/8. Prefix-cache state transfer,
200K context and RAM eviction are not qualified here. Stable first served
ID is hotschmoe-dd; model/method/scheme aliases are registered in
`evals/configs/models.yaml`. Main target remains the DD ISTA IQ3_XXS.

CPU sidecar load/payload/wide-FC checks pass. CPU PLE history/reference
checks pass, including int32 input IDs. The real HC10240 convolution
captured/reordered/rejected test passes50 replays, conv-only0.095748ms;
this is not model ITL. Proxy block40 nextn weights load into the native
Qwen3.5 MTP head, not target layer zero. Requested reports G3, D2 and
Shisa overlay were inspected before implementation.

## Measured results

Raw evidence root: `/mnt/vm_8tb/b70/build/vllm-flashnext/x1`.
All rates below are diagnostics until their associated gates pass.
Effective latency is1000/decode_tok_s; MTP wire-chunk p50 is a separate
quantity because one chunk can contain multiple accepted tokens.

| Configuration | C1 effective ms/token | C2 effective ms/token per stream | Bench acceptance | Gate |
| --- | ---: | --- | ---: | --- |
| 35B proxy, MTP0 | 9.24 | 12.15 /13.35 | n/a | See EOS caveat |
| 35B proxy, MTP2 | 6.99 | 10.94 /11.86 | 114/148=77.03% | See EOS caveat |
| Original Flash-Next, MTP0 | 41.70 | 43.98 /43.98 | n/a | Exact repeat fails |
| Original Flash-Next, MTP2 | 30.27 | 32.89 /32.01 | 171/222=77.03% | Parity fails |
| Eventless uniform-Q8 Flash-Next, MTP0 | 43.24 | 45.78 /49.83 | n/a | Exact repeat fails |

Proxy: all367 compared tokens through configured EOS/96-token limit
match. Three forced96-token streams match completely; the fourth differs
at token82, after configured EOS248044 at token78. The forced continuation
gate actually returned3; the EOS-truncated comparison is derived, not a
fresh normal-EOS serving run. Identity/coherence/health/teardown passed.
C1 wire-chunk p50 was9.27ms off,18.29ms with MTP2. Aggregate draft
acceptance373/454=82.16%. Raw proxy logs/JSON and proxy EOS gate in root.

Original Flash-Next: target/draft FULL graphs captured; all30 draft
parameter slots loaded per EP rank. C1 wire-chunk p50 was40.23ms off,
85.20ms with MTP2; C2 was44.35/44.18ms off,83.37/83.53ms on. Aggregate
acceptance528/700=75.43%. C1 Paris differs at17; other three96-token
C1 outputs match. C2 Paris/Fibonacci differ at4/72, before EOS.
Two subsequent same-load MTP0 gates also fail, including Paris at4 and
C2 at2/72. Raw flash-*.json/log, flash-repeat-pass1/2.json.

Latest control: all six token requests, latency benchmark, and six repeat
requests complete in one load. Coherence mentions Paris but same=false.
C1 repeat differs at4; other three C1x96 match. Concurrent repeats differ
at1/1. Wire-chunk p50 is43.016ms C1,45.299/45.606ms C2. Both-card
post-health HEALTHY. Raw eventless-canonical/flash-*.json/log,
control-repeat-diff.json and window-eventless-control.out.

## Numerical and graph diagnosis

Actual rt HC/router BF16, packed Q8 dense, full-shape GDN three-row
verification versus sequential decode, and FA speculative versus
sequential at past16/64/128/512/1536 all pass sampled bit-exact checks.
These primitive checks do not prove the full model batch invariant.

Two concrete MoE differences were located:

1. Scheduled reduction summed grouped-expert plan order, which depends
   on other tokens. Actual mixed-quant probes differed up to1.49e-8 FP32;
   10/50 records also changed BF16 (11 elements). Unscheduled fallback
   passed50/50 exactly. `patches/x1_slot_reduce.patch` adds an inverse
   (token,router-slot) map to both builders and reduces slots in fixed
   order, shared last, for both output modes. CPU-only build128s/MAX4;
   separate `_C.slot.so` SHA256
   bd22ebbd40c4c6c4b0e2513be99306fe8927834e6ecdf9beeccdfd52bcda6426.
   All63 actual-weight numeric/captured-scratch checks pass exactly.
2. Dispatcher T>8 changed Q8 activation math to BF16 grouped math.
   New opt-in B70_MTP_CANONICAL_MOE=1 chunks scheduled Q8 at8 for every
   size, leaving small decode unchanged. At T7 versus T9/T17, the old
   dispatcher changes15216 BF16 elements,maxabs.0029296875; the new path
   is bit-exact and passes captured T17. This is a slower-prefill
   correctness baseline, not a prefill optimization.

Both initial full fixed-slot controls completed six requests then stalled
at PLE token D2H. The CPU readiness fence did not cure it. Both ranks
logged begin/done of periodic re-instantiation at500 replays; this is
correlation, not proof of cause. They were stopped, with both-card
post-health HEALTHY and no valid benchmark/repeat result. Raw slot/,
slot-fenced/, saved rank stacks and reclaim.log.

The optional `patches/x1_eventless.cpp` interposes only Torch's
XPUGraphImpl::replay using execute_graph. It preserves device guard,
capture checks, RNG seed/offset writes and rounded Philox advancement
from installed Torch08187d9e. Hidden RNG helpers required mirroring
public state fields. CPU build17s/MAX4; SHA256
b82405b40a13457c5d6e6b19c7a44d19988cf0b7e981cccb7ea181cb249a7cc3.
It is ABI-specific and must be rebuilt after Torch updates.

Retained probe-r3 passes2066 replays/RNG, uniform-Q8 eager/captured
checks, exit0 and pinned card1 health before/after. RSS grew39504KiB in
this small probe; no leak/stability claim. The full eventless control
completes beyond the earlier stall point, but exactness still fails.
Periodic re-instantiation is disabled only for this opt-in recipe.

## Evidence, failures and next work

Host: kernel7.1.0-070100, UMD26.22, loader1.28.2, Torch2.14.0+xpu
commit08187d9e, compiler oneAPI2026.1.1. Initial fork HEAD
0a6170da8ca9991972322fa1ca1d27674f0e75cd with shared dirty changes.
Original library `_C.o3.so` SHA256
be76661bd120aa2f6c7523ae3c08813ba56d6114660f3d7f31183c32e5e861eb
is unchanged. Latest exact source copies/manifests/diff are under
x1/eventless-canonical; shared diff is not a standalone upstream patch.
The longer chronological report is preserved as
x1/1X1_mtp-through-eventless-control.md. JOURNAL has stage records.

Failures retained: persistent SYCL cache startup segfault (cache off
fixed subsequent startup); int32 PLE history index error (fixed); first
shim hidden symbol error (fixed); deliberate queue abort after another
agent's failed collective; two stalled full controls above. Health
passed after teardown. No hardware reset was used.

A short-probe wrapper mistakenly leased card1 while invoking default
both-card health. Its termination raced acquisition; the orphaned owned
container was explicitly stopped. Retained R3 passes --card1 to health
and workload, with explicit mapped device0. No other agent was stopped.

Claude Fable5.1/Claude5.5 CLI consultations failed with unrecognized_model
and Not logged in. No responding model IDs/advice were obtained; raw
fable51.txt/claude55.txt. No invented review was used.

Next correctness work is per-layer target-state/logit capture on repeated
short prompts, comparing DP rank, prefill width, fresh/reused state and
graph replay. Then fix the first differing operation and rerun exact
C1/C2 on/off gates. Estimate1-2 engineering days including GPU windows;
this is a work estimate, not an unsupported-kernel claim. Only after
that gate: restricted-vocab drafting, batch draft depth and integration
with the measured O3/O4 decode optimizations. The present recipe does
not beat Strata's7.09ms C1 or10.9ms C2.

The first matched eventless MTP retry stopped before weight load: the API
waited on cold draft model-info inspection while its fresh subprocess
spun entirely in libze_intel_gpu.so (103 CPU perf samples, no syscalls).
The global preload reaches this CPU inspection child too. Shutdown then
faulted in SYCL queue destruction (exit139); both-card post-health
HEALTHY, no reset. The helper now primes hashed target/draft model-info
with the ordinary runtime before exporting the preload. Corrected retry
uses eventless-primed/ with the completed control JSON copied unchanged;
startup-* evidence and failed arm remain in eventless-canonical/.

Wave3 handoff: the20:28 ORCHESTRATOR directive replaces DP2 work
with O4 sidecar work; the later ownership note assigns sidecar MTP
to O5 and exact-repeat work to X4. X1 cancelled its corrected retry to
release the cards. Cancellation raced acquisition: compiled pre-health
started, then the owned container stopped exit143 in18s. No weights
loaded; model-info priming remains UNVALIDATED. No new acceptance or
latency result exists. eventless-primed/ and its window log retain this
abort. Final native post-health passes on BOTH cards (window exit1 because
ready0, not health failure). No X1 jobs or containers remain. X1
schedules no further GPU windows.

O5's current plugin already excludes mtp prefixes from sidecar expert
registration (`b70_ep_xpu/method.py:35`) and separates speculator graph
handling (`hooks.py:116`), keeping the drafter local on card0. X1 has
not exercised or qualified that new route. The X1 native-EP loader/PLE
results alone are not sidecar MTP evidence.

Final closeout20:44 UTC: both native card probes HEALTHY after the final
abort. Focused Ruff F/I, Python syntax/ASCII, shell syntax and diff
whitespace checks pass. No commit, DD start or reset was performed.
