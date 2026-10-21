# Q1 Torch exact packed-key selector

Measured original scratch directory:
/mnt/vm_8tb/b70/build/vllm-flashnext/q1/torch-r1
Profile: profile-r1 (2026-10-21 18:03-18:04UTC).
measured-source-sha256.json records every measured source hash. The port
changes only script paths/import search; selector.py is byte-identical.
No GPU rerun is needed for these source-location changes.

The selector maps exact native unsigned64 score/index keys to signed
int64 via sign-bit XOR, masks invalid cells below all valid keys, uses
upstream Torch topk2051 sorted descending, masks device widths, stably
sorts IDs ascending and preserves output beyond width. Fixed147456score
columns were tested. See reports/3Q1_qsa.md for numerical and timing data.

CPU66cases PASS; GPU126adversarial comparisons and all6populated real
shapes/canaries/changed-input capture/exact gated-output parity PASS.
Native pre/posthealth and source/identity/lifecycle checks PASS. Component
only: no serving/fidelity/pressure/liveness qualification.

A future independently authorized replay uses fresh output identity:

    Q1_PROFILE_TAG=profile-new bash vllm/flashnext/harness/q1_torch/window.sh

Physicalcard1/logical0,240s body cap, quiet CPU outside lease, mandatory
native pre/posthealth, source hashes and no-overwrite output. Results
remain under scratch q1/torch-r1 by default; Q1_TORCH_ROOT overrides it.
No custom library build. Do not silently change this measured control
when preparing new fused selectors. The first measured path showed4K
regression and16K/32K component gains, not a serving speedup claim.
