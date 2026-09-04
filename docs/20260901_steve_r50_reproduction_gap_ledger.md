# Steve R50 independent reproduction gap ledger

Date: 2026-09-01 UTC

## Outcome

CONFIG -> Independently replay Steve Seguin's public Qwen3.8-27B FP8 W8A16
R50 graph-off MTP1 recipe from the latest public
`b70-optimization-lab` source, then distinguish missing publication inputs
from local configuration differences.

COMMAND -> Fetch latest `origin/main`, create a clean detached worktree at
`0e8c4c577d40674f0aceb3c5005c24f3d305f951`, run the public source-closure
verifier, run the final R50 builder preflight unchanged, inspect Git history
and GitHub code/release assets, probe both documented final registry tags,
build the complete R13 -> R15 -> R31 -> R49 -> R50 chain with the tracked
patch, compare installed files and ELF sections, and run the strict c1
benchmark under the two-card lease with pre/post health.

RESULT -> Public source closure passes, but the unmodified final builder fails
immediately with `split GDN patch digest mismatch`. The tracked patch is
`08a3de4f...`; the builder and every qualified R50/R55C result identify an
unpublished `40ca8c3f...` patch. The local build from the tracked patch matched
all six published host `.text`, `.rodata`, and `.data` hashes, but not either
published complete-library pair. Its strict launch matched Steve's arguments
except served ID, matched every selected environment value and cgroup bound,
passed model identity, workload, canaries, cache-zero, teardown, and health,
but measured only `17.203380 tok/s` versus `51.808087`.

VERDICT -> The public R50 recipe is not independently source-reproducible as
published. Do not attribute the performance miss to launch flags and do not
promote the locally rebuilt R50 candidate. Obtain the missing R50 patch or
final artifacts first, then use the exact artifact on this host to separate
artifact drift from the documented host-platform difference.

## Confirmed missing publication pieces

### 1. The R50 split-GDN patch used by the qualified images

Latest public state is internally inconsistent:

- tracked file
  `vllm-xpu-kernels-qwen38-gdn-split-serial-gates-r50-20260901.patch`:
  `08a3de4f26119c50a23be87004708508eb444fed168175fb65a565e9a90e4033`;
- `verify-public-source-closure.sh` requires that same `08a3de4f...` value;
- `build-mtp1-rebuilt-gdn-image.sh` requires
  `40ca8c3fc15fea1b7dda8d268761f0b1339eb821f5d8357b3da7600585fe750f`;
- the clean R55C result and all qualified R50 container labels also identify
  `40ca8c3f...`.

The patch exists at only one commit in public Git history and that blob hashes
to `08a3de4f...`. GitHub code search finds `40ca8c3f...` only as a referenced
hash in builders, labels, and result metadata, not as downloadable patch
content. A pristine latest-main checkout therefore passes source closure and
then fails the advertised one-command build before compilation.

Required resolution: publish the patch blob whose SHA-256 is `40ca8c3f...`,
or state that `08a3de4f...` is intended and update the builder, final binary
hashes, image contract, and qualification attribution consistently.

### 2. An accessible final R50/R55C artifact

Neither documented final image can be pulled anonymously:

- `neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50`;
- `neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-serial-fa-split-gdn-r50-reprocheck-r55c`.

Both registry manifest probes return `denied` and `unauthorized`. The public
GitHub releases contain the upstream R13 wheel and an older flash-next runtime
archive, but no final R50/R55C image, wheel, `_xpu_C.abi3.so`, or
`libgdn_attn_kernels_xe_2.so`.

The alternative `build-mtp1-split-gdn-image.sh` requires the two lab binaries
to already exist in `KERNEL_ARTIFACT_DIR`; it does not provide an artifact URL.
It expects the lab pair:

- `_xpu_C.abi3.so`: `f8013aff50f815b290cbec87d7926936c3fae9daacad6e1cf1f4c01ca60180ef`;
- `libgdn_attn_kernels_xe_2.so`:
  `32a13caab7d56e6b584b7396ff61b3755a60362e6647db26337b98fdbd0bb4ec`.

The clean R55C pair is also not downloadable:

- `_xpu_C.abi3.so`: `1632cafcf2afc0bc039dd49ebbb5eda4e62d626f4c20729aecd9e87874d1dc08`;
- `libgdn_attn_kernels_xe_2.so`:
  `2c343620d689409bfa371a8b4c3db680e4786f23bc092411e7d03140f1b2a355`.

Required resolution: publish either the clean R55C OCI archive/public registry
manifest or the two exact final libraries. The recorded Docker image ID
`sha256:41aec5da...` is a machine-local config ID, not a pullable registry
digest.

### 3. Device-code identity needed to diagnose clean rebuild drift

The public result records `.text`, `.rodata`, and `.data`, but Intel SYCL
device binaries are stored separately in the `OFFLOAD_DEVICE_CODE` ELF
section. The public result does not publish that section's hash. Therefore the
six advertised stable-section hashes cannot prove that an independent build
contains the qualified XPU device image.

The local tracked-patch build produced:

- whole `_xpu_C.abi3.so`: `771d2eb181a99e5fc4635461eaed5bf6a30a311f5d9de30239930ce0cd16e4cf`;
- `_xpu_C` `OFFLOAD_DEVICE_CODE`:
  `023a45d0ab8363dd3d4538f7c171ca88e29afa68d048c74086439572e6d8678b`;
- whole `libgdn_attn_kernels_xe_2.so`:
  `f574956d7d89d426d5b77dee66d2e802c92e546541a543bf1e880c88c1061399`;
- GDN `OFFLOAD_DEVICE_CODE`:
  `88bf2317b00c74afc9700f3ca3a05fb3c260d69c2277d9dd9eca84a6dad03db7`.

All six published host code/data section hashes matched. The 17.2 tok/s live
result shows why that is insufficient evidence.

Required resolution: publish `OFFLOAD_DEVICE_CODE`, `.dynstr`, and preferably
all-section hashes for the clean R55C pair, or provide the binaries so they can
be compared directly.

### 4. Complete clean-build toolchain receipt if source rebuild is the route

The public builder pins only the first `icpx --version` line. The R55C result
does not include the final wheel SHA-256, build log, full oneAPI package
manifest, linker/binutils identity, or all ELF section hashes. Those details
are not needed if the exact final artifact is published, but they are needed
to explain a whole-file/device-image mismatch in a source-only reproduction.

Required resolution: if Steve prefers source rebuild over distributing the
final artifact, publish the R55C build stdout/stderr, final wheel SHA-256,
`dpkg-query` or equivalent oneAPI component manifest, linker/binutils versions,
and hashes of every allocatable/offload ELF section.

## Known host boundary difference, not missing documentation

Steve documents the working host as Ubuntu 24.04.4, kernel
`7.0.0-28-generic`, Docker 29.1.3, `intel-opencl-icd
26.22.38646.7-1~24.04~ppa1`, and `libze1 1.28.6-1~24.04~ppa1`.

The local host is Ubuntu 26.04.1, kernel `7.1.0-070100-generic`, Docker
29.1.3, `intel-opencl-icd 26.22.38646.4-0`, and `libze1 1.28.2-2`. Project
policy keeps kernel 7.1. This is a confirmed unmatched platform boundary, but
the versions are already documented and should not be described as a missing
recipe input. Testing the exact R55C artifact here is the clean way to decide
whether it matters.

## Confirmed present and matched

Do not ask Steve to resend these unless he says they changed:

- latest public source and the complete R13 -> R49 overlay sources;
- official base image digest `f01e24f6...`;
- upstream kernel commit `1e90ffa672...` and durable R13 wheel;
- Qwen3.8 model revision `017b9c7a...` and all direct/ordinary file identities;
- oneAPI compiler banner `2026.1.1.20260724`;
- strict server arguments, deterministic compilation JSON, MTP1, graph-off,
  FlashAttention, FP8 W8A16, direct P2P, and cache-off settings;
- 9 GiB memory / 12 GiB memory-plus-swap serving bounds;
- Steve's strict benchmark and canary harnesses.

The local R55C-container comparison found zero differences across the 31
selected strict environment variables and identical cgroup limits. The only
CLI difference was the served-model display name.

## DM draft

Hey Steve - thanks again for the Sept 1 update. I did a clean independent
replay against latest main (`0e8c4c57`) and found a concrete publication gap.
Your `verify-public-source-closure.sh` passes and requires the tracked R50
split-GDN patch SHA `08a3de4f...`, but
`build-mtp1-rebuilt-gdn-image.sh`, the R55C result, and the qualified image
labels require `40ca8c3f...`. The `40ca...` patch content is not in Git history
or the public release assets, so the advertised one-command build fails at the
patch digest gate on a pristine checkout.

Could you please provide one of these, preferably option 1?

1. A public OCI archive/registry digest for the clean R55C image
   (`41aec5da...` locally), or its exact final `_xpu_C.abi3.so` and
   `libgdn_attn_kernels_xe_2.so`; or
2. The actual split-GDN patch blob with SHA `40ca8c3f...`, plus the R55C final
   wheel SHA/build log and full oneAPI component versions.

Could you also send the `OFFLOAD_DEVICE_CODE` section hashes for both R55C
libraries? The repo publishes `.text/.rodata/.data`, but the SYCL device image
lives in `OFFLOAD_DEVICE_CODE`. Our build from the tracked `08a3...` patch
matched all six published host sections but produced whole hashes
`771d2eb1...` / `f574956d...`, device-section hashes `023a45d0...` /
`88bf2317...`, and only `17.203 tok/s` under the strict suite. The launch itself
matched your R55C args, all 31 selected env settings, and cgroup limits; model
identity, cache-zero, canaries, and pre/post GPU health all passed.

Our host is newer than your documented boundary (Ubuntu 26.04.1/kernel 7.1
versus Ubuntu 24.04.4/kernel 7.0). If you can share the exact R55C artifact, we
can test it directly and cleanly separate artifact drift from the host/KMD
boundary. Thanks!

## Public references

- https://github.com/steveseguin/b70-optimization-lab/blob/main/packages/qwen38-27b-fp8-tp2-b70/README.md
- https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/build-mtp1-rebuilt-gdn-image.sh
- https://github.com/steveseguin/b70-optimization-lab/blob/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/verify-public-source-closure.sh
- https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-09-01-qwen38-fp8-clean-rebuild-r55c-result.json

## Resolution on 2026-09-04

Steve published the corrected `40ca8c3f...` patch, exact R55C libraries,
R139 extension and section hashes, and remote-bound manifests. A fresh
full-history clone now passes public source and remote-asset closure, and the
no-compiler chain builds through R156. The matched local R156 graph-off and
sizes `[1,2]` graph-on strict pair passed 12/12 complete arrays at 16.845797
and 49.675873 tok/s. The publication gap in this ledger is closed; see
`docs/20260904_steve_r187_independent_replay.md` for current evidence and
remaining qualification scope.
