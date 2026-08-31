# Neural.Download r31 image provenance and transfer ledger

Date: 2026-08-31

Status: the known runtime lineage has been reconstructed in publisher order
and its content manifest passes. The publisher OCI artifact is not public, so
the exact publisher image ID cannot be independently recreated or inspected.
This document corrects the incomplete parent lineage recorded in the 2026-08-29
port ledger.

## Where Steve published the evidence

The material is public, but it is split between a presentation page and a
large evidence repository:

- model page:
  `https://neural.download/models/qwen38-27b-fp8-vllm-tp2-asrock-b70.html`
- repository:
  `https://github.com/steveseguin/b70-optimization-lab`
- reproduction directory:
  `https://github.com/steveseguin/b70-optimization-lab/tree/main/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70`
- qualified r32 result using the r31 image:
  `https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-08-28-qwen38-fp8-mtp1-deterministic-r32.json`
- raw r32-A container inspection:
  `https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/qwen38-fp8-mtp1-deterministic-r32a/container-inspect.json`
- upstream vLLM source:
  `https://github.com/vllm-project/vllm/commit/ac7509e2b1db40fec2f03dde1ed4e9dfdc2338c9`
- upstream kernel build:
  `https://github.com/vllm-project/vllm-xpu-kernels/actions/runs/32798686770`

The repository is the important source. It contains the Dockerfiles, build and
launch scripts, patches, operator proof, model manifest, raw server logs,
canaries, performance JSON, comparison JSON, health results, and container
inspection. The web page classifies the package as a candidate portable
reproduction and still lists a clean-host replay and tested host installation
path as unfinished.

The r31 packet was introduced by repository commit
`6aab301f30912c87bfcc7b7982f2fab27eb1eca5`. The preserved local checkout is
`0948f7c2c2e21f0e8fcc444e319e5e8f5b83d0e7`; the fetched public `main` was
`3561c8d0f241403d1f92a2a5a99709e62a2b28fe` during this audit. The r31 files
and result remain present on that public revision.

No public registry copy, OCI archive, or r31 GitHub release was found. Pulling
`neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-rms-serial-r31` fails. The
only currently visible repository release is for a later, unrelated runtime.

## Corrected image lineage

The publisher evidence resolves this chain:

```
official f01e image
  -> exact 1e90ffa XPU-kernel wheel image, r13
    -> four-file deterministic vLLM overlay, r15
      -> two-copy packed MTP1 RMSNorm overlay, r31
```

| Stage | Publisher identity | Material change |
| --- | --- | --- |
| Official base | `sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f` | Pinned vLLM XPU image at vLLM `ac7509e2` |
| Kernel parent | tag `f01e-kernel-1e90-r13`; publisher ID recorded as `sha256:9403883cdbec3df988f486815f9dd528eb98baf0cc73d04ef3631ff0ac6a35b0` | Reinstalls exact `vllm-xpu-kernels` wheel from commit `1e90ffa` |
| Deterministic parent | `sha256:d19f802ba702a9cb94b155f807a4674a0100702aee838323372f740d7168e34e` | Copies four patched vLLM Python files |
| Final r31 | `sha256:ba42e928e69c60d1c9102df6ec1c0e998e9dd8463f74d5dc0a8b4bb45108fa9b` | Copies patched `layernorm.py` into both effective Python trees |

### The missing recipe edge

`build-deterministic-compiled-image.sh` defaults `BASE_IMAGE` directly to the
official `f01e` image. The displayed recipe does not show an override. However,
the raw r31 container labels inherit all three kernel-parent labels:

- base digest `f01e24f6...`;
- kernel head `1e90ffa...`;
- kernel wheel SHA256 `f3d99906...`.

Therefore the publisher r15 was built with the r13 kernel image supplied as an
external `BASE_IMAGE` override, or through an equivalent unstated step. The
nominal two build commands cannot produce the recorded lineage as written.
This is an omitted invocation detail, not proof that the published runtime
contents differ from the public inputs.

The earlier 2026-08-29 local build started r15 directly from official `f01e`.
Its four Python overlay hashes were correct, but its binary XPU-kernel parent
was not the publisher parent. Later local F05C/F10 work installed the `1e90ffa`
wheel after the Python overlays, yielding the known final executable files but
the reverse image-layer order. The ordered reconstruction fixes that provenance
error.

## Exact components

### Official base runtime

The locally pulled immutable base reports:

| Component | Version |
| --- | --- |
| Python | `3.12.3` |
| vLLM | `0.27.2rc1.dev77+gac7509e2b.xpu` |
| PyTorch | `2.13.0+xpu` |
| Triton XPU | `3.7.2` |
| oneCCL Python runtime | `2022.0.0` |
| DPC++/SYCL Python runtime | `2026.0.0` |
| Container `intel-opencl-icd` | `26.27.39122.11-0` |
| Container `libze-intel-gpu1` | `26.27.39122.11-0` |

The host driver and the container user-mode runtime are separate identities.
The project host remains on its required kernel 7.1 baseline; it must not be
downgraded merely to imitate Steve's host.

### Kernel wheel

- upstream repository: `vllm-project/vllm-xpu-kernels`;
- commit: `1e90ffa672ba02f17a909da11838a4c55b199783`;
- Actions run: `32798686770`;
- artifact: `vllm-xpu-kernels--20260825-014754`;
- wheel:
  `vllm_xpu_kernels-0.1.dev1+g1e90ffa67-cp38-abi3-manylinux_2_28_x86_64.whl`;
- wheel SHA256:
  `f3d999060c11ad6db5b4033d50d19c6b665492380075480d041ec4ee58fdfeb6`;
- installed package version: `0.1.dev1+g1e90ffa67`;
- installed `_xpu_C.abi3.so` SHA256:
  `ba911f7e7d0bae668f0039a3e443e1768c2010d239d2970d281a7dd01fcb5289`.

The GitHub Actions artifact was still present and unexpired during this audit.
A verified copy is retained outside the repository under the Steve reproduction
root. It is an ABI-specific binary and must not be copied into a refreshed
PyTorch, vLLM, or XPU runtime. Rebuild or port its source at the pinned commit.

### Python overlays

| Runtime file | SHA256 |
| --- | --- |
| `vllm/model_executor/kernels/linear/scaled_mm/xpu.py` | `7c36e4a8dab4bfc06b1d5be2d8466e8cdc94099dd5409424fecc6dd8ffc2c208` |
| `vllm/_xpu_ops.py` | `f3273ccfb41be44c3c02080c26df10e8b200060366b900d940803f4221224c59` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `7afb4de8b87d7f180d696f7cadad8b9d48d9ab7b706ae19616425c4f9456fb19` |
| `vllm/distributed/device_communicators/xpu_communicator.py` | `5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d` |
| site-package `vllm/model_executor/layers/layernorm.py` | `50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8` |
| workspace `vllm/model_executor/layers/layernorm.py` | `50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8` |

The four patches and their SHA256 values are:

| Patch | SHA256 | Mechanism |
| --- | --- | --- |
| FP8 block W8A16 | `5db7f1af1156f3490ca91d0d74a07aa2d0909e175eeb1ae23f2074c55c44ff8a` | Default-off dispatch to `_xpu_C.fp8_gemm_w8a16` without activation quantization |
| Deterministic GDN B/A and state | `cda7dd1e42a1e0fed2dd34f3936303cb038852a46d8d00786a1c2ebae326f8eb` | Fixed padded B/A projection shape and explicit recurrent-state mutation |
| Compiled GDN state and CCL wait | `8f8febcd0abc59bc9b69830827cd7607c00870414b17bd02cf32e2d879858ac8` | Compiler-visible persistent state plus explicit async all-reduce `Work.wait()` |
| Packed MTP1 RMSNorm | `ff5b4f33f5596efbad75112bdbbca2bbf81b6c84688476bfa1c9ec9e546c78c4` | For exactly two rows, replay each row through the original native RMSNorm and concatenate |

## Exact qualified launch boundary

The 51.918757 tok/s result was the median of two fresh qualified runs at
51.606902 and 52.230611 tok/s. Both passed 12/12 complete target-array checks,
12/12 repeat equality, independent canaries, and zero cached prompt tokens.

Its important settings were TP2, FP16 compute and KV, official block-FP8
weights, W8A16 activation path, MTP1, one sequence, 1,024 maximum model length,
1,024 maximum batched tokens, prefix cache off, deterministic Inductor, direct
P2P enabled, and XPU Graph disabled. The command still names `PIECEWISE` with a
size-one capture, but `VLLM_XPU_ENABLE_XPU_GRAPH=0` disables the XPU graph
mechanism. The container used 9 GiB RAM, 12 GiB RAM-plus-swap, 8 GiB shared
memory, host IPC, `/dev/dri`, and `SYS_PTRACE`.

This is a strict, short-context, single-active-request qualification. It is not
evidence for 262K active context, long prefill, high concurrency, or our kernel
7.1 queue-handoff safety. Current later c64 work in Steve's repository also
shows that the deterministic single-stream treatment does not automatically
generalize to large batches.

## What transfers elsewhere

| Component | Transfer value | Boundary |
| --- | --- | --- |
| W8A16 dispatch | High for the same block-FP8 layout and operator contract | Does not directly implement W8A8 INT8; port the dispatch idea and validate layout, scales, dtype, and op support |
| Fixed-shape GDN B/A | High for Qwen3.8 GDN determinism | Architecture and shape specific; do not apply to Ornith or another GDN implementation without tracing its exact state and projection contract |
| Compiler-visible recurrent state | Broadly useful | Re-register against the target vLLM cache API and prove mutation/capture behavior |
| Explicit collective `Work.wait()` | Generic correctness experiment | It can serialize the hot path; our later matched work found source-default c10d faster, so keep it as an A/B control rather than a universal optimization |
| Two-row RMSNorm replay | High for this MTP1 packed shape | Intentionally shape-specific; it should not be assumed correct or fast for MTP2/MTP8, batch concurrency, or other models |
| `1e90ffa` wheel | Valuable source reference | ABI-specific binary; port source and rebuild for every refreshed stack |
| Deterministic Inductor | Useful qualification control | Can reduce throughput and has not guaranteed batch invariance in later c64 tests |
| Graph disabled | Removes capture compatibility constraints | Gives up graph replay savings and does not itself explain or guarantee concurrency throughput |

For W8A8 INT8, the most transferable pieces are the integration discipline,
fixed-shape/state analysis, explicit operator dispatch, and qualification
method. The FP8 W8A16 GEMM itself is not an INT8 XMX kernel. An INT8 candidate
must separately prove its activation quantization, scale layout, accumulation,
epilogue, shape coverage, and dispatch. Decode may remain memory-sensitive,
but the runtime still pays GEMM, collective, scheduling, state, and launch
costs; FP8 and INT8 should not be assumed equal without matched measurement.

## Ordered local reconstruction

The tracked builder is
`vllm/fp8/build_qwen38_fp8_r31_ordered_repro.sh`. It:

1. requires Steve's packet commit in the local source ancestry;
2. verifies six build inputs and all four patch SHA256 values;
3. verifies the immutable official base;
4. verifies or downloads the exact kernel wheel;
5. builds the kernel image first;
6. supplies that image explicitly as the deterministic r15 `BASE_IMAGE`;
7. supplies the resulting r15 image explicitly to the r31 build; and
8. fails unless all seven known final runtime hashes match.

The 2026-08-31 no-GPU build produced:

| Local stage | Image ID |
| --- | --- |
| Kernel | `sha256:90746d6d1f1129bcbe20fb102c4091f1b712fc485742f614cccfedfc934e9708` |
| r15 | `sha256:82cac0986b5495bd1c005723654ddf7dfde3e94d7429f97eb5bf2ec2a8e5fc94` |
| r31 | `sha256:3e29569b6d15ff6da46c8e62b3325184ed86812aa8faefff962a4f6b44c648cc` |

The final image is
`b70-local/vllm-openai-xpu:qwen38-fp8-r31-ordered`, and its runtime manifest
passes. This establishes a content reconstruction from the public inputs. It
does not establish publisher OCI identity or runtime performance on this host.

## What only Steve can still provide

Exact OCI reproduction and complete provenance require the publisher to add:

- an `oci-archive` or `docker save` export of r13, r15, and r31 plus archive
  SHA256 values;
- `docker image inspect` and `docker history --no-trunc` for each image, not
  only inspection of a running r31 container;
- manifest, config, and every compressed/uncompressed layer digest;
- BuildKit/frontend versions, build args, timestamps, provenance settings,
  cache state, and the missing r15 `BASE_IMAGE` invocation;
- a full rootfs file manifest, package inventory, `pip freeze`, and SBOM;
- complete host `lscpu`, NUMA, PCIe topology, BIOS, firmware, power limit,
  frequency/governor, Docker daemon, kernel/xe, Level Zero, compute-runtime,
  and oneCCL identities; and
- a clean-host replay following the published installation procedure.

Until those exist, we can say "known runtime content reconstructed". We cannot
say "byte-identical publisher image recreated" or attribute the remaining
performance gap to a single hidden component.

## Verdict

The public repository contains enough evidence to recover every known code and
binary component that matters to r31 execution. The major previously missed
piece was the `1e90ffa` kernel-wheel parent before the deterministic Python
overlays. That ordering is now reproduced locally and content-verified. The
unavailable OCI layers and incomplete host/build metadata prevent exact image
identity and a fully controlled 51.918757 tok/s reproduction claim.
