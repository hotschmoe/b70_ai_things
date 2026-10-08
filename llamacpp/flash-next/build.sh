#!/usr/bin/env bash
# Build only: no GPU devices are passed into the compiler container.
set -euo pipefail
export LC_ALL=C
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$task_dir/../.." && pwd)
if [[ ${1:-} != --leased ]]; then
    exec "$project_root/bin/gpu-run" "$0" --leased "$@"
fi
shift
# gpu-run inherits both locked descriptors. Refuse direct inner invocation.
lock_base=${B70_GPU_LOCK:-/mnt/vm_8tb/b70/gpu.lock}
for pair in 8:0 9:1; do
    lease_fd=${pair%:*}
    lease_card=${pair#*:}
    [[ -e /proc/$$/fd/$lease_fd && /proc/$$/fd/$lease_fd -ef "$lock_base.$lease_card" ]] || {
        echo 'Missing inherited whole-host GPU lease.' >&2; exit 2;
    }
    flock -n "$lease_fd" || { echo 'Inherited GPU lease is not exclusive.' >&2; exit 2; }
done
[[ $# == 0 ]] || { echo 'Usage: build.sh (no arguments)' >&2; exit 2; }
source_dir=/mnt/vm_8tb/github/llama.cpp-flashnext
build_dir=/mnt/vm_8tb/b70/build/flashnext-llamacpp
source_revision=de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b
image_tag=b70/flashnext-llamacpp-build:oneapi2026.1-de7fa0a
[[ $(git -C "$source_dir" rev-parse HEAD) == "$source_revision" ]] || {
    echo 'Source revision differs from the reviewed pin.' >&2; exit 2;
}
[[ -z $(git -C "$source_dir" status --porcelain) ]] || {
    echo 'Source tree must be clean for the unpatched control build.' >&2; exit 2;
}
mkdir -p "$build_dir"
log_dir=$(mktemp -d "$build_dir/preparation-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
exec > >(tee "$log_dir/build.log") 2>&1
trap 'build_rc=$?; if (( build_rc != 0 )); then printf "RESULT build failed exit=%s\nVERDICT unqualified; inspect build.log\n" "$build_rc"; fi' EXIT
printf 'CONFIG source=%s revision=%s build=%s\n' "$source_dir" "$source_revision" "$build_dir"
printf 'COMMAND docker build; configure SYCL ON DNN ON F16 OFF; build server bench perplexity and correctness probes -j8\n'
docker build --progress=plain -t "$image_tag" "$task_dir"
image_id=$(docker image inspect --format '{{.Id}}' "$image_tag")
docker image inspect "$image_id" > "$log_dir/image-inspect.json"
printf '%s\n' "$source_revision" > "$log_dir/source-revision.txt"
git -C "$source_dir" status --porcelain > "$log_dir/source-status.txt"
sha256sum "$task_dir/Dockerfile" "$task_dir/build.sh" > "$log_dir/preparation-sha256.txt"
printf 'CONFIG image=%s\n' "$image_id"
# Deliberately no --device, --privileged, or host-driver bind mounts.
docker run --rm --network none --user "$(id -u):$(id -g)" \
    -e TASK_LOG_DIR="/build/$(basename "$log_dir")" \
    -v "$source_dir:/src:ro" -v "$build_dir:/build" "$image_id" '
set -eo pipefail
source /opt/intel/oneapi/setvars.sh >/dev/null
set -u
export LC_ALL=C
icpx --version > "$TASK_LOG_DIR/compiler-version.txt"
dpkg-query -W > "$TASK_LOG_DIR/packages.txt"
cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_SYCL=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_F16=OFF \
    -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_TESTS=ON
if grep -q "DNNL_DIR:PATH=DNNL_DIR-NOTFOUND" /build/CMakeCache.txt; then
    echo "oneDNN was not found; refusing a silently changed control" >&2
    exit 2
fi
cmake --build /build --target llama-server llama-bench llama-perplexity test-backend-ops test-quantize-fns -j8
cp /build/CMakeCache.txt "$TASK_LOG_DIR/CMakeCache.txt"
sha256sum /build/bin/llama-server /build/bin/llama-bench /build/bin/llama-perplexity /build/bin/test-backend-ops /build/bin/test-quantize-fns > "$TASK_LOG_DIR/binary-sha256.txt"
'
printf 'RESULT build completed; records=%s\n' "$log_dir"
printf 'VERDICT binaries built only; GPU compatibility, coherence, speed and teardown unqualified\n'
