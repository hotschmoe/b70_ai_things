# R276 native convolution contract provenance

CONFIG -> Actual repaired7b retains _xpu_C SHA271db0d4882124e21ac6a4d080bfeab303fbb08b9ec10e11f21d10fb0723998f. Steve fetched source pin ac2f723ec0eb150976c744f54cddb7c4e0d99c82 provides clean-clone R220/R221 replay proving bit identity of this binary. Kernel pin1e90ffa672ba02f17a909da11838a4c55b199783 plus hash-pinned R35/R50 GDN patches; oneDNN0e2a5b plus R137a/R137b/R221, sycl-tla cd763. R276 is Python-only _xpu_ops overlay on that native chain. The unrelated a397 source is not its provenance.

COMMAND -> Reconstructed exact GDN source from pinned Git archive and applied original hash-verified patches. Inspected causal_conv1d.hpp688-724/805-823 against worker get_conv_copy_spec. Inspected upstream5802a414d47855b01b63121bce3655795ef8dfa8 (#544 including#545); verified it is absent from shipped1e90 ancestry. No GPU work.

RESULT -> Shipped speculative native convolution reads first3 rows from accepted-1 state COLUMN and publishes first3 rows to every speculative COLUMN. It does not publish six-row rollback history in column0. Upstream5802 changes precisely this contract: column0, accepted-1 ROW, rolling history length Width-1+num_spec; derives effective length without treating padded/interleaved stride as state dimension. Existing producer/consumer mismatch matches independent parent-owned actual native publication measurements.

RESULT -> Native header fix applies unchanged to reconstructed source. Interface guard needs deliberate adaptation because R35 supports dynamic active width: after active num_spec_tokens calculation, require conv_state.size(1)>=Width-1+(num_spec_tokens-1). No serial-exact branch activation/change. Review patch r276-upstream5802-conv-contract.patch applies cleanly. Full build sources and command frozen; compiler2026.1.1-325 downloaded from Intel official apt HTTPS, every deb SHA checked against index, extracted into isolated directory, not installed on host. CPU version in7b is2026.1.1.20260724.

VERDICT -> Existing source fix is directly relevant; native candidate still requires rebuild, actual publication/math/copy oracles and strict image pre/post health before model inference. No claim that source inspection alone qualifies repaired serving. Trace/sync candidates remain parked NOTRUN.

Source URLs:
- https://github.com/steveseguin/b70-optimization-lab/blob/ac2f723ec0eb150976c744f54cddb7c4e0d99c82/experiments/qwen38-27b-b70/data/replay-w4a16-r220-r221-clean-clone-20260906T192527Z/replay-summary.json
- https://github.com/vllm-project/vllm-xpu-kernels/commit/5802a414d47855b01b63121bce3655795ef8dfa8
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html

REBUILD RECIPE

Use source-pins.json to export fresh tracked sources for kernel, oneDNN and sycl-tla. Apply the two vllm-xpu R35/R50 patches to kernel, three oneDNN R137a/R137b/R221 patches to oneDNN, then r276-upstream5802-conv-contract.patch to kernel. original-patch-shas.json and port-review-inputs.json pin every patch. No historical object, archive or shared library is an input.

Run fetch_toolchain.py --out <isolated-toolchain-dir> to retrieve the exact recorded packages, verify SHA256 and extract without host install. Mount its root/opt/intel/oneapi read-only into exact7b at /opt/intel/oneapi. build-command.json records the exact launched container paths, source/dependency mounts and flags; build.sh is the original pinned Steve CMake recipe. When relocating, change host source/output paths only. Keep no devices, no network; the scheduling-only resumed build uses8CPU40GiB and exact image/toolchain/native flags. Newly compiled _xpu_C is the only intended final image replacement; inspect dependencies and compare all other native/Python/package identities before qualification.

SCHEDULING UPDATE -> Parent authorized8jobs/8CPUs/40GiB after noting free host resources. Owned no-device firstbuild was stopped (exit137), its command/log/return preserved under rawattempt1-jobs4, and the exactsameNinja tree resumed. Sources, compiler and native compiler flags unchanged; existing completed objects retained.

FINAL CPU RESULT -> Buildexit0. Imaged55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067 contains native6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9. All2105othernativefiles/allpackages/all2227vLLMPythonfiles match7b. NewextensionNEEDED/RUNPATH unchanged; exportsremove0/add1torchCheckMsg helper. Rebuilt auxiliaryGDN/MHC export APIs match original, while bytes differ; originals deliberately retained. Patchedtranslationunit/objectproof establishes newconvkernel is in_xpu_C, not auxiliarylibrary. No-device/no-network CPUimport/schema check passes. Strictpairpreflight andactualnumeric/modelqualification remain required; no GPU results claimed here.
