#!/usr/bin/env bash
# Sync the repo-canonical zml example sources into the git-ignored upstream zml build
# clone, where bazel builds them. The repo holds the contributable source of truth
# (zml/examples/<name>/); the clone at /mnt/vm_8tb/b70/zml is a throwaway checkout.
#
# Usage:
#   bash zml/apply_examples.sh            # copy all examples into the clone
#   ZML_CLONE=/path bash zml/apply_examples.sh
#
# Then build/run on CPU (no GPU):
#   cd "$ZML_CLONE" && ~/.local/bin/bazelisk run //examples/<name> --config=release
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZML_CLONE="${ZML_CLONE:-/mnt/vm_8tb/b70/zml}"

if [[ ! -d "$ZML_CLONE/examples" ]]; then
  echo "ERROR: zml clone not found at $ZML_CLONE (set ZML_CLONE=...)" >&2
  exit 1
fi

src="$REPO_DIR/examples"
[[ -d "$src" ]] || { echo "no examples to sync under $src"; exit 0; }

for d in "$src"/*/; do
  name="$(basename "$d")"
  dst="$ZML_CLONE/examples/$name"
  mkdir -p "$dst"
  cp -v "$d"main.zig "$d"BUILD.bazel "$dst/"
done

echo "synced examples into $ZML_CLONE/examples"
