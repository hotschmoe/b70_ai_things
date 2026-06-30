#!/usr/bin/env bash
# Apply the zml W8A8 work (this repo's contribution) into the git-ignored upstream zml
# build clone, where bazel builds it. The repo is the source of truth; the clone at
# /mnt/vm_8tb/b70/zml is a throwaway checkout of upstream zml.
#
# The contribution is a single git patch, zml/patches/zml_w8a8.patch -- which is ALSO the
# PR-ready artifact (git am / git apply against a fresh zml checkout). It adds:
#   - examples/w8a8/                          (M0 CPU int8-dot microbench)
#   - examples/llm/models/common_quant.zig    (M1 reusable QuantizedLinear)
#   - examples/llm/models/quant_tests.zig     (M1 parity test)
#   - examples/llm/BUILD.bazel                (quant_tests target)
# Browsable copies of the new .zig files also live under zml/examples/ for review.
#
# Usage:
#   bash zml/apply_examples.sh            # apply to the default clone (idempotent)
#   ZML_CLONE=/path bash zml/apply_examples.sh
#
# Then build/run on CPU (no GPU):
#   cd "$ZML_CLONE"
#   ~/.local/bin/bazelisk run //examples/w8a8 --config=release
#   ~/.local/bin/bazelisk run //examples/llm:quant_tests --config=release
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZML_CLONE="${ZML_CLONE:-/mnt/vm_8tb/b70/zml}"
PATCH="$REPO_DIR/patches/zml_w8a8.patch"

[[ -f "$PATCH" ]] || { echo "ERROR: patch not found at $PATCH" >&2; exit 1; }
[[ -d "$ZML_CLONE/.git" ]] || { echo "ERROR: zml clone (git) not found at $ZML_CLONE (set ZML_CLONE=...)" >&2; exit 1; }

cd "$ZML_CLONE"

# Idempotent: if the patch already reverse-applies, it is already in the tree.
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "already applied in $ZML_CLONE -- nothing to do"
  exit 0
fi

if ! git apply --check "$PATCH" >/dev/null 2>&1; then
  echo "ERROR: patch does not apply cleanly to $ZML_CLONE (HEAD $(git rev-parse --short HEAD))." >&2
  echo "       The upstream clone may have diverged; regenerate the patch or reset the clone." >&2
  exit 1
fi

git apply "$PATCH"
echo "applied $PATCH into $ZML_CLONE"
