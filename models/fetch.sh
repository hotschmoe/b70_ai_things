#!/usr/bin/env bash
# models/fetch.sh -- reprovision model weights on a fresh box from manifest.yaml.
#
#   bash models/fetch.sh            # download every source:hf entry into files/<id>
#   bash models/fetch.sh --list     # just print what would be fetched/skipped
#   ONLY=qwen3.6-27b/bf16 bash models/fetch.sh   # one model
#
# source:hf  -> huggingface-cli download <repo> --local-dir files/<id>
# source:custom -> quantized on the old box; NOT downloadable. Printed as a SKIP with the
#                  derived_from base so you know what to rebuild (see manifest TODO).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.yaml"
LIST_ONLY=0; [ "${1:-}" = "--list" ] && LIST_ONLY=1
ONLY="${ONLY:-}"

mapfile -t ROWS < <(python3 - "$MANIFEST" <<'PY'
import sys, yaml
m = yaml.safe_load(open(sys.argv[1]))
root = m.get("root", "files")
for e in m["models"]:
    s = e.get("source", {})
    files = s.get("files", e.get("files", []))
    if isinstance(files, str):
        files = [files]
    fields = [
        e["id"],
        s.get("type", "?"),
        s.get("repo", ""),
        e.get("derived_from", ""),
        root,
        s.get("revision", ""),
        ",".join(files),
    ]
    if any("|" in str(field) for field in fields):
        raise ValueError("manifest fetch fields may not contain '|'")
    print("|".join(str(field) for field in fields))
PY
)

[ ${#ROWS[@]} -eq 0 ] && { echo "no models parsed from $MANIFEST"; exit 1; }

for row in "${ROWS[@]}"; do
  IFS='|' read -r id type repo derived root revision files_csv <<<"$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && continue
  dst="$HERE/$root/$id"
  if [ "$type" = "hf" ]; then
    detail=""
    [ -n "$revision" ] && detail="$detail revision=$revision"
    [ -n "$files_csv" ] && detail="$detail files=$files_csv"
    echo ">> $id  <-  hf:$repo$detail"
    [ "$LIST_ONLY" = 1 ] && continue
    mkdir -p "$dst"
    download_args=(download "$repo")
    [ -n "$revision" ] && download_args+=(--revision "$revision")
    if [ -n "$files_csv" ]; then
      IFS=',' read -r -a exact_files <<<"$files_csv"
      download_args+=("${exact_files[@]}")
    fi
    download_args+=(--local-dir "$dst")
    if command -v hf >/dev/null 2>&1; then
      hf "${download_args[@]}" || { echo "FAILED: $id"; exit 1; }
    else
      huggingface-cli "${download_args[@]}" || { echo "FAILED: $id"; exit 1; }
    fi
  else
    echo "-- $id  SKIP (source:custom, quantized-on-device; rebuild from ${derived:-?} -- see manifest TODO)"
  fi
done
echo "done. custom quants (w4a16/w4a8) still need: bash models/graft_w4_complete.sh"
