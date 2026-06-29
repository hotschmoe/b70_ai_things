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
    print("\t".join([e["id"], s.get("type","?"), s.get("repo",""), e.get("derived_from",""), root]))
PY
)

[ ${#ROWS[@]} -eq 0 ] && { echo "no models parsed from $MANIFEST"; exit 1; }

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r id type repo derived root <<<"$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && continue
  dst="$HERE/$root/$id"
  if [ "$type" = "hf" ]; then
    echo ">> $id  <-  hf:$repo"
    [ "$LIST_ONLY" = 1 ] && continue
    mkdir -p "$dst"
    huggingface-cli download "$repo" --local-dir "$dst" || { echo "FAILED: $id"; exit 1; }
  else
    echo "-- $id  SKIP (source:custom, quantized-on-device; rebuild from ${derived:-?} -- see manifest TODO)"
  fi
done
echo "done. custom quants (w4a16/w4a8) still need: bash models/graft_w4_complete.sh"
