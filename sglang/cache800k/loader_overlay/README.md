# Fresh calibration review and isolated SGLang loader overlay

CONFIG -> Fresh calibration uses exact phase-fixed R276 image
0328900cf1f8f29f5a8e76ed21a3eff71ef54d0abf495a88fdb86e93e880b077,
TP2/P2P0/MTP3/FP16 KV/eager/prefix-off. Candidate base is SGLang triton-dense
bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf,
source 2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed. CPU-only work here.

COMMAND -> Existing verify_and_freeze.py freeze runs only after lifecycle exit0,
rehashes every current model file, compares pre/post inode/stat/content identity,
then invokes vllm/cache800k/freeze_scales.py. Run the additional read-only gate:

```text
python3 sglang/cache800k/loader_overlay/review_fresh.py \
  --root /mnt/vm_8tb/b70/results/bang_isolation_20260910/fresh-calibration-plan \
  --config models/files/qwen3.8-27b/int4-autoround-gptq-relabel-r212/config.json
```

RESULT -> NEW artifact SHA256:
`be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062`.
Independent strict final review passes all262 request IDs/order, corpus/prompt/text
hash binding, cancellation/error/finish/usage gates, pre/post health and clean
teardown evidence, complete positive finite target/MTP observations, raw record
hashes, full model provenance, and six role/TP/rank mapper plans. Both continuation
requests generated4096 tokens. There are258 length and4 stop finishes.

VERDICT -> Artifact accepted for a separate experimental loader candidate.
No calibrated serving, FP8 numeric, graph, MTP, or Pi sampling qualification is
implied. The existing tracked loader still pins the old digest and its provenance
checks remain intact. Only the prepared candidate copy accepts the new digest.

## Freeze gate review

The original freezer requires exit0, WORKLOADS_PASSED, matching eager/record/
prefix-off/FP16/MTP configuration, response count at least262, successful positive
completions, and finite nonnegative activation maxima. The outer fresh verifier
adds the exact image/TP/P2P/health-probe controls, full model pre/post hashing and
stat identity, expected34 layer/rank records, and response degeneration checks.

The additional gate closes the remaining acceptance gaps: counts alone could
allow duplicate or missing continuation IDs; it requires the exact reviewed
262-entry corpus and ordered response identity, SHA256 binding for every prompt
and text, no cancellation, stop/length finishes, positive integer usage, and
strict integer counts/ranks with positive finite Q/K/V maxima. It binds explicit
per-card and compiled-pair pre/post health logs and rejects recovery/forced-remove
markers. It does not edit the active run, old artifact, or freezer behavior.

Periodic activation snapshots still do not prove exhaustive final-step coverage.
The fresh artifact accurately retains that limitation. The extra gate consumes
verified pre/post model hashes; it does not pretend to perform another full
model rehash. Runtime verify_model_files remains unchanged and will rehash the
actual mounted weights before installing scales.

## Exact source overlay

The current-main wheel packaged in the base contains model_runner.py SHA256
6159576a0506b508da274097d603d9ec25e52c439d90915f2ed54c48c4857aae,
identical to reviewed source. The only changed/added Python files are:

- sglang/srt/model_executor/model_runner.py
- sglang/srt/model_executor/b70_calibrated_kv/__init__.py
- sglang/srt/model_executor/b70_calibrated_kv/scale_loader.py
- sglang/srt/model_executor/b70_calibrated_kv/scale_plan.py

Their installed root is /opt/venv/lib/python3.12/site-packages. The existing
sglang-main-scale-loader.patch adds one common runner call before kernel
precompile/profile and skips the generic loader only after successful custom
loading. Both target and MTP runners use this call; no Qwen model file patch is
needed. Strict inventories are target model.layers.{3,7,...,63}.attn and draft
model.model.layers.0.attn, with classes Qwen3_5ForConditionalGeneration and
Qwen3_5ForCausalLMMTP. Both prefill/decode must resolve to Triton.

prepare.py requires the passing review JSON AND an explicit reviewed digest,
checks the original runner hash, applies the patch with zero fuzz, and changes
only the artifact digest in a COPY of scale_loader.py. The old digest remains
rejected by that candidate. The data artifact is embedded at
/opt/b70/calibrated-kv/fresh-scales.json, for --quantization-param-path.
No sampling overlay, native wheel, driver, Torch, package install, or kernel
change is combined with this candidate.

## Build and identity workflow

prepare.py only writes a fresh context and commands; it never builds. A dedicated
local base tag is created from the exact base image ID and verified against that
ID. Dockerfile FROM cannot use a raw sha256 image ID with this BuildKit, which
tries a registry repository named sha256; the failed initial attempt is retained
as evidence. The second context uses the dedicated full-ID tag, with pull=false
and network=none. No active serving tag is updated.

The Dockerfile contains only FROM/LABEL/COPY. Compare base/candidate RootFS layer
prefixes and run native_identity.py in each without devices/network, boundedCPU/
memory. It hashes every ELF or static archive under /opt/venv/lib,
/usr/lib/x86_64-linux-gnu, and /opt/intel and records package versions. Require
exact native path/size/hash/package equality; do not waive a difference because
the edit was intended to be Python-only.

cpu_candidate_gate.py checks actual installed Python bytes, ordering before
precompile, rejection of the old digest, acceptance of the new digest before the
CPU-device refusal, and persistent scalar FP32 buffers/float mirrors for target
and MTP across TP1/rank0 and TP2/ranks0,1. It exposes no GPU devices and loads no
model weights. The image still needs required stack health and leased serving
qualification before use.

Raw evidence roots:
- fresh-calibration-plan/loader-final-review.json
- sglang-loader-fresh-overlay/ (failed raw-ID build retained)
- sglang-loader-fresh-overlay-v2/ (candidate context/build/CPU/native checks)

All are under /mnt/vm_8tb/b70/results/bang_isolation_20260910/.

## Measured candidate result

CONFIG -> Base bdc51...; only the four Python files and embedded reviewed JSON
are overlaid. No sampling patch.

COMMAND -> Build the prepared v2 context with no pull/network, then run
native_identity.py on both base and candidate and cpu_candidate_gate.py in the
candidate, all without GPU devices. Compare inspected RootFS layers.

RESULT -> Candidate image:
`sha256:f82a10b2c3d04f10b230ba299dced23455366f307d96bdce54a7143264b41a99`.
All3088 native ELF/archive paths and recorded packages are exactly equal;
identity JSON SHA256 is
`fd40e73e08da60332ac004baebef96d34f6f8256b6851959bf9dcfac4233b9a3`.
The base15 layers are unchanged and only2 COPY layers are added. Installed
source, precompile ordering, old-digest rejection, new-digest acceptance before
CPU-device refusal, and all six target/MTP TP/rank buffer plans pass.

VERDICT -> CPU/native identity gates pass; candidate is ready for parent-managed
leased health/serving qualification. This does not resolve the separate XPU
FP8 write/read numeric investigation or establish full sampling parity.
