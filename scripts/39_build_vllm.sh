#!/usr/bin/env bash
# Generic vLLM-XPU from-source builder. COMMIT=<tag/sha> IMGTAG=<image:tag>.
# Builds a separate tagged image so the working build stays as rollback.
set -uo pipefail
BUILD=/mnt/vm_8tb/b70/build
COMMIT="${COMMIT:-v0.23.0}"; IMGTAG="${IMGTAG:-vllm-xpu-env:v0230}"
mkdir -p "$BUILD"; cd "$BUILD"
[ -d vllm/.git ] || git clone https://github.com/vllm-project/vllm.git
cd vllm
echo "=== fetch + checkout $COMMIT ==="
git fetch --all --tags -q || true
git checkout "$COMMIT" 2>&1 | tail -2
git log -1 --oneline
ls docker/Dockerfile.xpu || { echo "no Dockerfile.xpu at $COMMIT"; exit 1; }
echo "=== docker build -t $IMGTAG (LONG; torch bump may reduce cache reuse) ==="
time docker build -f docker/Dockerfile.xpu -t "$IMGTAG" --shm-size=4g . 2>&1 | tail -40
echo "=== result ==="
docker images "$IMGTAG" --format '{{.Repository}}:{{.Tag}} {{.Size}}'
docker run --rm --entrypoint python "$IMGTAG" -c 'import vllm,transformers,torch; print("vllm",vllm.__version__,"tf",transformers.__version__,"torch",torch.__version__)' 2>&1 | tail -1
df -h /var/lib/docker | tail -1
echo "=== DONE ==="
