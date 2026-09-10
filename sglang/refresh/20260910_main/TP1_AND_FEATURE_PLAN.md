# Current-main TP1 baseline and incremental feature gates

CONFIG -> Immutable bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf;
physical card1, relabel-r212 AutoRound GPTQ INT4 g128, FP16 compute/KV,
FP32 SSM, eager, MTP0, prefix off, overlap off. Context8192, chunk512, c4.
Explicit Triton full attention and Triton GDN; stable model name hotschmoe-dd.
Use a new cache with no seed from another image. HTTP deadline300 seconds,
startup deadline1200 seconds, bounded diagnostic job1500 seconds.

COMMAND -> Prepared only, after current-main pair-health/oracle campaign
finishes with successful post-health and the coordinator allocates card1:

```text
python3 sglang/cache800k/tp1_viability.py \
  --card 1 --port 18137 \
  --image sha256:bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf \
  --attention-backend triton \
  --health-probe /mnt/vm_8tb/b70/results/bang_isolation_20260910/health-repair/xpu-health \
  --out /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-tp1-fp16-triton-card1 \
  --job /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh/tp1-baseline-plan/job.json \
  --startup-timeout 1200
```

The lifecycle acquires `bin/gpu-run --card 1` internally and verifies its
inherited card1 lease. The job is the same 26-case diagnostic used for the
older source-port image: two exact canaries, eight serial cases and sixteen
requests across four c4 mixed batches. It checks /v1/models before requests.
This is a text/structure and repetition screen, not full model-quality or
exact-token equivalence. A cold-cache result is not a matched speed comparison.

RESULT -> No GPU launch. Exact installed ServerArgs parser accepts all baseline
flags. The old `--disable-cuda-graph` spelling is deprecated but supported;
source resolution maps it to disabled decode and prefill graphs. The actual
KV dtype resolver returns FP16 for auto plus FP16 model dtype without a KV
quantization override; the artifact is GPTQ and declares no FP8 KV override.
Native GDN and held aten INT4 pack/matmul XPU registrations are present.
The initial CPU checker omitted importing sgl_kernel before querying dispatch;
that checker failed after CLI parse. The corrected CPU-only check imports the
package and passes. Both records are preserved; this was not a backend failure.

VERDICT -> Concrete baseline launch plan and CPU CLI/native compatibility pass.
Model initialization, runtime resolved settings, coherence, c4, teardown and
strict post-health still require measurement. `tp1_baseline_plan.json` records
command, job, hashes and raw CPU checks. The baseline command and image are unchanged. Later opt-in lifecycle
extensions are described below.

## Incremental arms after the baseline

Each arm needs a distinct result directory and the same image, model, card,
context, chunk, c4, compute/SSM dtype and diagnostic requests. Record actual
resolved config and cache identity. Change one feature at a time; retain a
baseline control. Only reuse a compiler cache from this exact immutable image.

1. Prefix-only: existing lifecycle `--prefix-cache` enables explicit
   `extra_buffer`; keep eager/MTP0/FP16 KV. Triton supports page1; resolve and
   record page size, with an explicit page1 server argument in the reviewed
   future arm. Run the 26-case screen plus the existing 32-check concurrent
   tool-history probe. Require measured cache hits and no stale history/state.
   Older native-attention page128 prefix results do not qualify this arm.
2. Decode-graph-only: prefix off/MTP0/FP16 KV. Replace disabled graphs with
   `--cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled
   --cuda-graph-bs-decode 1 2 4`. Actual CLI parsing passes and the decode
   runner routes graph operations through its device module, including XPU.
   Require observed capture and replay, c1/c2/c4 coherence, clean teardown and
   post-health. Do not assume graph use from accepted flags. The automatic
   tc_piecewise prefill compatibility rules exclude XPU; that is not the next
   graph arm. The lifecycle now exposes this arm as `--decode-graph`.
3. MTP-only: prefix off/eager/FP16 KV. Start with
   `--speculative-algorithm NEXTN --speculative-num-steps 1
   --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
   --speculative-draft-model-path /model
   --speculative-draft-attention-backend triton`.
   CLI parsing passes. The artifact index contains29 embedded MTP tensors;
   the pinned MTP constructor preserves GPTQ quantization. Actual draft weight
   mapping, target/draft backend identity, memory fit and acceptance still need
   qualification. Run greedy temperature0 first, then increase to steps3 and
   draft tokens4 only after a clean one-step result. The lifecycle exposes
   these settings as `--mtp-steps 1` and `--mtp-steps 3`.
4. Combine individually qualified prefix and decode graph, then add qualified
   MTP. Run a matched control when each feature is added. Calibrated FP8 KV is a
   separate overlay/numeric/model-quality gate after fresh artifact provenance
   and the actual Triton oracle; changing the cache dtype alone is insufficient.

MTP limitation: the stock pinned XPU verifier takes an argmax branch even for
non-greedy requests. A greedy MTP pass does not qualify Pi temperature0.7 or
seeded sampling parity. The separate NEXTN sampling review and candidate remain
prerequisites for that behavior; bdc51 contains no sampling repair. Future
feature argv were CPU parsed only; they are not silently enabled in the
baseline defaults and no feature serving result is claimed. Opt-in graph/MTP
flags require explicit Triton attention before any lease or Docker invocation.
The 26-case job forces temperature0 and seed42 on every request, including
completions, so it is already a greedy MTP control. CPU argv checks verify the
unchanged baseline, draft budgets, feature isolation, stable identity and
rejection of unreviewed steps/backends. Lease, cleanup and cache-seed identity
code is unchanged.

## Large-collective prerequisite before TP2

The compiled health probe tests [4,5120] BF16 and cannot clear the prior
[2048,5120] FP16 subgroup hang. Before any TP2 full-model startup, repeat the
existing `sglang/cache800k/collective_control.py` numerical control in the new
immutable image with the intended server's exact P2P0/oneCCL environment and
matching mapped runtime-library hashes. It performs [4,5120], then three fresh
[2048,5120] FP16 SUMs in a non-world two-rank XCCL subgroup, using exactly
representable data and explicit producer/completion fences.

Require both ranks' entry, host return, completion fence return and exact
numerical pass for every sequence, plus owned teardown and strict per-card +
compiled pair post-health. A missing rank completion is a failed arm, even if
its peer returns. Use the new reviewed ownership helper, not the historical
collective_lifecycle.py: that older file imports its image from server.py and
uses the old shared health probe. Prepare any oneCCL env variant as a separate
arm after cleanup/recovery, never by toggling a live runtime.

Passing the fresh-process subgroup control is necessary but does not certify
loaded-model collectives. Capture the new model's actual profile row counts,
collective counts, per-rank entry/return and graph boundaries before expanding
the loaded-context oracle. Include any newly observed larger profile shape.
Only then attempt a bounded TP2 eager/MTP0 control; add prefix, graph and MTP
incrementally. No P2P1 full-model arm is proposed.
