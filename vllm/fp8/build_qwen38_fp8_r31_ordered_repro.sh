#!/usr/bin/env bash
set -euo pipefail

# Rebuild the publisher r31 runtime lineage in the order proven by the
# retained r32 container inspection:
#
#   official f01e -> XPU kernels 1e90 -> deterministic vLLM overlay -> r31
#
# The public deterministic-overlay helper defaults to official f01e.  That
# default is not the parent recorded in the publisher image labels, so this
# wrapper supplies the kernel image explicitly and validates every known
# input and final runtime file.

source_root=${STEVE_SOURCE_ROOT:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260829/source}
build_root=${BUILD_ROOT:-/mnt/vm_8tb/b70/steve-repro/qwen38-r31-ordered-20260831}
repro_dir=${source_root}/repro/qwen38-27b-fp8-vllm-tp2-asrock-b70
experiment_dir=${source_root}/experiments/qwen38-27b-b70

official_image=vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
official_id=sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f
kernel_image=${KERNEL_IMAGE:-b70-local/vllm-openai-xpu:f01e-kernel-1e90-r13-ordered}
r15_image=${R15_IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-r15-ordered}
r31_image=${R31_IMAGE:-b70-local/vllm-openai-xpu:qwen38-fp8-r31-ordered}

kernel_head=1e90ffa672ba02f17a909da11838a4c55b199783
wheel_name=vllm_xpu_kernels-0.1.dev1+g1e90ffa67-cp38-abi3-manylinux_2_28_x86_64.whl
wheel_sha=f3d999060c11ad6db5b4033d50d19c6b665492380075480d041ec4ee58fdfeb6
packet_commit=6aab301f30912c87bfcc7b7982f2fab27eb1eca5

for command_name in docker git sha256sum awk install; do
  command -v "${command_name}" >/dev/null || {
    printf '%s is required\n' "${command_name}" >&2
    exit 1
  }
done

[[ -d "${source_root}/.git" ]] || {
  printf 'missing Steve source checkout: %s\n' "${source_root}" >&2
  exit 1
}
git -C "${source_root}" merge-base --is-ancestor "${packet_commit}" HEAD || {
  printf 'Steve source does not contain pinned r31 packet commit %s\n' "${packet_commit}" >&2
  exit 1
}

check_input() {
  local expected=$1
  local path=$2
  [[ -f "${path}" ]] || { printf 'missing input: %s\n' "${path}" >&2; exit 1; }
  [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || {
    printf 'input digest mismatch: %s\n' "${path}" >&2
    exit 1
  }
}

check_input ccebb0231d8fa523661c36cc8edf28292bd13ef37bca6b6bed9353cf63561e55 \
  "${repro_dir}/build-mtp1-kernel-image.sh"
check_input 83f7b45d73af5b797b85f2d99905cf820165a8a5660e4914b254131aef3a7664 \
  "${experiment_dir}/docker/Dockerfile.fp8-kernel-1e90-r13"
check_input a7211406a2603db1ac499cd950fdc55d1a3cdf1d5656f39938d0355777cf9cf9 \
  "${repro_dir}/build-deterministic-compiled-image.sh"
check_input c610c0ad8b8519c103e37efbe89f088d70bde5d2c5fbf03ed382f27cb8f0986d \
  "${repro_dir}/Dockerfile.deterministic-compiled"
check_input 7e5edb640fa0a541111d2baba79aeda939c2fb0f613321c3087f2f5b668f75b8 \
  "${repro_dir}/build-mtp1-rmsnorm-serial-image.sh"
check_input ba69f4f608e2eeec0eb55436d5dd0989a3b1bf368f22b5883078f50a0258e27c \
  "${repro_dir}/Dockerfile.mtp1-rmsnorm-serial"
check_input 5db7f1af1156f3490ca91d0d74a07aa2d0909e175eeb1ae23f2074c55c44ff8a \
  "${experiment_dir}/patches/vllm-qwen38-fp8-block-w8a16-20260826.patch"
check_input cda7dd1e42a1e0fed2dd34f3936303cb038852a46d8d00786a1c2ebae326f8eb \
  "${experiment_dir}/patches/vllm-qwen38-xpu-deterministic-gdn-ba-state-20260828.patch"
check_input 8f8febcd0abc59bc9b69830827cd7607c00870414b17bd02cf32e2d879858ac8 \
  "${experiment_dir}/patches/vllm-qwen38-xpu-compiled-gdn-state-ccl-wait-20260828.patch"
check_input ff5b4f33f5596efbad75112bdbbca2bbf81b6c84688476bfa1c9ec9e546c78c4 \
  "${experiment_dir}/patches/vllm-qwen38-xpu-gemma-rmsnorm-mtp1-serial-exact-r30-20260828.patch"

docker image inspect "${official_image}" >/dev/null 2>&1 || {
  printf 'missing immutable official base: %s\n' "${official_image}" >&2
  exit 1
}
[[ "$(docker image inspect "${official_image}" --format '{{.Id}}')" == "${official_id}" ]] || {
  printf 'official base identity mismatch\n' >&2
  exit 1
}

mkdir -p "${build_root}/kernel" "${build_root}/r15" "${build_root}/r31"
artifact_dir=${build_root}/kernel/vllm-xpu-kernels-${kernel_head}
mkdir -p "${artifact_dir}"
if [[ ! -f "${artifact_dir}/${wheel_name}" ]]; then
  preserved_wheel=${KERNEL_WHEEL:-/mnt/vm_8tb/b70/steve-repro/qwen38-fp8-neural-20260829/kernel-artifacts/vllm-xpu-kernels-${kernel_head}/${wheel_name}}
  if [[ -f "${preserved_wheel}" ]]; then
    install -m 0644 "${preserved_wheel}" "${artifact_dir}/${wheel_name}"
  fi
fi

if ! docker image inspect "${kernel_image}" >/dev/null 2>&1; then
  BUILD_ROOT="${build_root}/kernel" IMAGE="${kernel_image}" \
    "${repro_dir}/build-mtp1-kernel-image.sh"
fi
[[ "$(docker image inspect "${kernel_image}" --format '{{ index .Config.Labels "neural.download.base.digest" }}')" == "${official_id}" ]]
[[ "$(docker image inspect "${kernel_image}" --format '{{ index .Config.Labels "neural.download.kernel.head" }}')" == "${kernel_head}" ]]
[[ "$(docker image inspect "${kernel_image}" --format '{{ index .Config.Labels "neural.download.kernel.wheel.sha256" }}')" == "${wheel_sha}" ]]

if ! docker image inspect "${r15_image}" >/dev/null 2>&1; then
  BUILD_ROOT="${build_root}/r15" BASE_IMAGE="${kernel_image}" IMAGE="${r15_image}" \
    "${repro_dir}/build-deterministic-compiled-image.sh"
fi
r15_id=$(docker image inspect "${r15_image}" --format '{{.Id}}')

if ! docker image inspect "${r31_image}" >/dev/null 2>&1; then
  BUILD_ROOT="${build_root}/r31" BASE_IMAGE="${r15_image}" \
    EXPECTED_BASE_IMAGE_ID="${r15_id}" IMAGE="${r31_image}" \
    "${repro_dir}/build-mtp1-rmsnorm-serial-image.sh"
fi

declare -A expected_files=(
  [/opt/venv/lib/python3.12/site-packages/vllm/model_executor/kernels/linear/scaled_mm/xpu.py]=7c36e4a8dab4bfc06b1d5be2d8466e8cdc94099dd5409424fecc6dd8ffc2c208
  [/opt/venv/lib/python3.12/site-packages/vllm/_xpu_ops.py]=f3273ccfb41be44c3c02080c26df10e8b200060366b900d940803f4221224c59
  [/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py]=7afb4de8b87d7f180d696f7cadad8b9d48d9ab7b706ae19616425c4f9456fb19
  [/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/xpu_communicator.py]=5ab2ea5d9e049e6b53e2d56d1e3419ce01d1988e8be5295bab1f912a7fdbf74d
  [/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/layernorm.py]=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8
  [/workspace/vllm/vllm/model_executor/layers/layernorm.py]=50cf5f4f9c72f679e4318cd3e3e021a844f59ac188a891d9a4f9638188f4bce8
  [/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so]=ba911f7e7d0bae668f0039a3e443e1768c2010d239d2970d281a7dd01fcb5289
)

mapfile -t runtime_hashes < <(
  docker run --rm --entrypoint sha256sum "${r31_image}" "${!expected_files[@]}"
)
for line in "${runtime_hashes[@]}"; do
  actual=${line%% *}
  path=${line##*  }
  [[ "${actual}" == "${expected_files[${path}]:-missing}" ]] || {
    printf 'final runtime digest mismatch: %s\n' "${path}" >&2
    exit 1
  }
done
[[ "${#runtime_hashes[@]}" -eq "${#expected_files[@]}" ]] || {
  printf 'final runtime manifest is incomplete\n' >&2
  exit 1
}

printf '%s\n' \
  "publisher-r31-id=sha256:ba42e928e69c60d1c9102df6ec1c0e998e9dd8463f74d5dc0a8b4bb45108fa9b" \
  "local-kernel-image=$(docker image inspect "${kernel_image}" --format '{{.Id}}')" \
  "local-r15-image=${r15_id}" \
  "local-r31-image=$(docker image inspect "${r31_image}" --format '{{.Id}}')" \
  "runtime-manifest=PASS" \
  "image=${r31_image}"
