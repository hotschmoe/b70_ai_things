#!/usr/bin/env bash
# Download the Flash-Next dev-model weights onto /mnt/cache (NVMe).
# Run under bin/cpu-run. Resumable. Writes a manifest next to the weights
# and a one-line status file for the watcher.
#
#   Qwen/Qwen3.6-35B-A3B                         official BF16 safetensors
#   unsloth/Qwen3.6-35B-A3B-MTP-GGUF             UD-IQ3_XXS (best IQ3_XXS
#                                                with MTP), plus UD-IQ4_XS
#                                                and UD-Q4_K_M when each
#                                                file is under 25 GB.
set -euo pipefail

BASE=/mnt/cache/b70/devmodel
LOGDIR=/mnt/vm_8tb/b70/build/vllm-flashnext/logs
STATUS="$LOGDIR/a3-download.status"
MANIFEST="$BASE/MANIFEST.txt"
HF="${HF_BIN:-/home/hotschmoe/.local/bin/hf}"

OFFICIAL_REPO=Qwen/Qwen3.6-35B-A3B
OFFICIAL_REV=995ad96eacd98c81ed38be0c5b274b04031597b0
GGUF_REPO=unsloth/Qwen3.6-35B-A3B-MTP-GGUF
GGUF_REV=5bc3e238d916f48a861bac2f8a1990a0e9b7e98d
# Decimal GB, matching the Hugging Face size display.
MAX_COMPARE=$((25 * 1000 * 1000 * 1000))

OFFICIAL_DIR="$BASE/Qwen/Qwen3.6-35B-A3B"
GGUF_DIR="$BASE/unsloth/Qwen3.6-35B-A3B-MTP-GGUF"

mkdir -p "$LOGDIR" "$BASE" "$OFFICIAL_DIR" "$GGUF_DIR"
# Keep the hub cache on NVMe. The root filesystem must not absorb a 72 GB copy.
export HF_HOME="$BASE/.hf-home"
export HF_HUB_CACHE="$BASE/.hf-cache"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

if [[ ! -x "$HF" ]]; then
  HF=$(command -v hf || command -v huggingface-cli || true)
fi
if [[ -z "$HF" ]]; then
  echo "FAILED no hf cli" > "$STATUS"
  exit 1
fi

write_status() { printf '%s\n' "$*" > "$STATUS"; }
on_err() {
  write_status "FAILED line ${1:-?} rc ${2:-?}"
  echo "download_devmodel: FAILED line ${1:-?} rc ${2:-?}" >&2
}
trap 'on_err $LINENO $?' ERR

write_status "RUNNING start $(date -Is) hf=$HF"

python3 - <<'PY' > "$LOGDIR/a3-gguf-sizes.txt"
import json, urllib.request
repo = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
rev = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=1"
req = urllib.request.Request(url, headers={"User-Agent": "b70-a3"})
with urllib.request.urlopen(req, timeout=120) as r:
    rows = json.load(r)
want = (
    "Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf",
    "Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
)
by = {row.get("path"): row.get("size") for row in rows}
for name in want:
    size = by.get(name)
    print(f"{name}\t{size if size is not None else 'MISSING'}")
PY
echo "download_devmodel: remote sizes:" >&2
cat "$LOGDIR/a3-gguf-sizes.txt" >&2

declare -a GGUF_FILES=()
while IFS=$'\t' read -r name size; do
  [[ -z "$name" ]] && continue
  if [[ "$size" == "MISSING" || -z "$size" ]]; then
    echo "download_devmodel: missing remote file $name" >&2
    write_status "FAILED missing $name"
    exit 1
  fi
  case "$name" in
    *IQ3_XXS*)
      GGUF_FILES+=("$name")
      ;;
    *)
      if (( size < MAX_COMPARE )); then
        GGUF_FILES+=("$name")
      else
        echo "download_devmodel: skip $name size=$size >= $MAX_COMPARE" >&2
      fi
      ;;
  esac
done < "$LOGDIR/a3-gguf-sizes.txt"

write_status "RUNNING official ${OFFICIAL_REPO}@${OFFICIAL_REV}"
echo "download_devmodel: official BF16 -> $OFFICIAL_DIR" >&2
"$HF" download "$OFFICIAL_REPO" \
  --revision "$OFFICIAL_REV" \
  --local-dir "$OFFICIAL_DIR"

write_status "RUNNING gguf ${GGUF_REPO}@${GGUF_REV} ${GGUF_FILES[*]}"
echo "download_devmodel: gguf ${GGUF_FILES[*]} -> $GGUF_DIR" >&2
"$HF" download "$GGUF_REPO" \
  "${GGUF_FILES[@]}" \
  README.md \
  --revision "$GGUF_REV" \
  --local-dir "$GGUF_DIR"

{
  echo "downloaded $(date -Is)"
  echo "official_repo $OFFICIAL_REPO"
  echo "official_revision $OFFICIAL_REV"
  echo "gguf_repo $GGUF_REPO"
  echo "gguf_revision $GGUF_REV"
  echo "gguf_note unsloth dynamic UD quants, MTP head included (separate from the non-MTP GGUF repo)"
  echo "compare_limit_bytes $MAX_COMPARE"
  echo
  echo "official tree:"
  find "$OFFICIAL_DIR" -type f ! -path '*/.cache/*' -printf '%s %p\n' | sort -k2
  echo
  echo "gguf tree:"
  find "$GGUF_DIR" -type f ! -path '*/.cache/*' -printf '%s %p\n' | sort -k2
  echo
  echo "remote_sizes:"
  cat "$LOGDIR/a3-gguf-sizes.txt"
} > "$MANIFEST"

# Confirm the files we asked for exist and are at least the remote size.
python3 - "$GGUF_DIR" "$LOGDIR/a3-gguf-sizes.txt" "$OFFICIAL_DIR" <<'PY'
import os, sys
gguf_dir, sizes_path, official = sys.argv[1:]
bad = False
for line in open(sizes_path):
    name, size_s = line.rstrip("\n").split("\t")
    if "IQ3_XXS" not in name and "IQ4_XS" not in name and "Q4_K_M" not in name:
        continue
    path = os.path.join(gguf_dir, name)
    remote = int(size_s)
    if not os.path.isfile(path):
        # skipped on purpose when over the compare limit
        if remote >= 25_000_000_000 and "IQ3_XXS" not in name:
            print(f"SKIPPED {name} remote={remote}")
            continue
        print(f"MISSING {path}")
        bad = True
        continue
    got = os.path.getsize(path)
    print(f"OK {name} bytes={got} remote={remote}")
    if got != remote:
        print(f"SIZE_MISMATCH {name} local={got} remote={remote}")
        bad = True
n_st = 0
st_bytes = 0
for root, dirs, files in os.walk(official):
    if ".cache" in root:
        continue
    for fn in files:
        if fn.endswith(".safetensors"):
            n_st += 1
            st_bytes += os.path.getsize(os.path.join(root, fn))
print(f"OFFICIAL safetensors_files={n_st} bytes={st_bytes}")
if n_st < 26 or st_bytes < 60_000_000_000:
    print("OFFICIAL_SHORT")
    bad = True
index = os.path.join(official, "model.safetensors.index.json")
if not os.path.isfile(index):
    print("MISSING index")
    bad = True
else:
    text = open(index, encoding="utf-8").read()
    has_mtp = ("mtp" in text.lower()) or ("nextn" in text.lower())
    print(f"OFFICIAL_INDEX mtp_or_nextn={has_mtp} bytes={os.path.getsize(index)}")
sys.exit(1 if bad else 0)
PY

write_status "DONE $(date -Is)"
echo "download_devmodel: DONE" >&2
trap - ERR
exit 0
