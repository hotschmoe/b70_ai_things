# XPU FP8 cache read-policy candidate

CONFIG -> Main source 2f7393f0d245bfaa8ebe0b9d0533432e6b3bd9ed,
calibrated loader base f82a10b2c3d04f10b230ba299dced23455366f307d96bdce54a7143264b41a99.
The frozen patch is ../../cache800k/fp8_oracle/xpu-fp8-cache-half-compute.patch.
It changes only grouped non-MLA decode and standard extend when Q is FP16
and K/V are E4M3FN on XPU. Q/K/V/P dot operands use FP16 compute; existing
cache scales remain applied. Other dispatches retain upstream behavior.

COMMAND -> Copy the pinned source into a separate oracle-source tree, apply
the existing f82 loader overlay, then apply the reviewed patch with
`patch -p1 --batch --fuzz=0`. Copy only the two attention files into overlay/.
Tag the exact base ID with the unique alias used by Dockerfile; verify its
image ID before building. Build with `docker build --pull=false --network=none
--iidfile image.id .` in the raw build context. The .dockerignore excludes
all but Dockerfile and the two overlay files. No devices are exposed.
Run source_identity.py and native_identity.py inside the resulting image
with --network none --cpus 2 --memory 4g and no /dev/dri mount.

RESULT -> Runnable image
sha256:14ee7d0112b4e321ea7618a2b6c6f351b7428c2efe43deb3c2e51d128c7dd57d.
Exactly two files differ among 3772 installed SGLang Python sources; their
hashes match the reviewed patch. All 239 package identities and 3088 native
file hashes match f82 exactly. Base rootfs layers are an unchanged 17-layer
prefix followed by two COPY layers. Environment unchanged; labels added.
Both changed sources compile. CPU audit containers were removed.
The existing preflight ownership checks and oracle stage-order tests pass.
An initial unittest module invocation lacked the oracle directory on sys.path;
the supported direct test-script invocation passed without source changes.

VERDICT -> Source-only candidate ready for new-image GPU health and numerical
oracle testing. No GPU execution was performed by this build task. No claim
of model quality, graph/MTP/TP2 correctness, general attention support, native
vision execution, or full feature parity follows from these CPU checks.
The upstream bdc and f82 images and failed oracle evidence are preserved.

## Prepared health plan

`python3 sglang/refresh/20260910_fp8_read_policy/preflight_plan.py` prints the
plan only. The recorded --run command acquires both leases and reuses the
reviewed strict per-card/compiled pair P2P=0 ownership and recovery helper.
The output directory must not already exist. This does not run an attention
oracle or model server. No launch was performed here.

The exact matching oracle --source path is recorded in manifest.json and
preflight_plan.json. It includes both f82 loader additions and this patch.
Use that source with the same immutable image; an upstream source mount would
fail the oracle source-identity contract. Numerical oracle fixtures remain
independent of full-model calibration validation.

Raw evidence: /mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-read-policy-image
