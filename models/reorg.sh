#!/usr/bin/env bash
# models/reorg.sh -- ONE-TIME migration to the new models/files layout.
#
#   de-root      : everything ends up owned by hotschmoe (uid 1000), not root.
#   materialize  : copy REAL bytes into models/files/<family>/<scheme> -- no symlinks, no
#                  hardlinks. Resolves the old container-style "/models/..." symlinks (which
#                  dangle on the host) by rewriting /models -> /mnt/vm_8tb/b70/models and
#                  following multi-hop chains, so each dest dir is fully self-contained.
#   drop         : delete off-target + prior-gen + subsumed-intermediate model dirs.
#
# Run ONCE as root. DRY-RUN by default (prints the plan, touches nothing).
#   sudo bash models/reorg.sh             # dry run
#   sudo APPLY=1 bash models/reorg.sh     # execute
#
# W4A16 and W4A8 are NOT handled here -- their vision-stripped bases are consumed next by
# graft_w4_complete.sh, which writes the complete builds straight into models/files.
set -uo pipefail

SRCROOT=/mnt/vm_8tb/b70
DST=/mnt/vm_8tb/github/b70_ai_things/models/files
UID_KEEP=1000; GID_KEEP=1000          # hotschmoe:hotschmoe
APPLY="${APPLY:-0}"

say(){ printf '%s\n' "$*"; }
hr(){ printf -- '----------------------------------------------------------------------\n'; }

# srcrelpath | family/scheme   (srcrelpath is under /mnt/vm_8tb/b70)
COMPLETE=(
  "models/Qwen_Qwen3.6-27B|qwen3.6-27b/bf16"
  "models/Qwen_Qwen3.6-27B-FP8|qwen3.6-27b/fp8"
  "models/Lorbus_Qwen3.6-27B-int4-mtp|qwen3.6-27b/int4-autoround"
  "models_w8a8/Qwen3.6-27B-W8A8-sqgptq-vision-mtp|qwen3.6-27b/w8a8-sqgptq"
  "models/Qwen_Qwen3.6-35B-A3B|qwen3.6-35b-a3b/bf16"
  "models/Intel_Qwen3.6-35B-A3B-int4-AutoRound|qwen3.6-35b-a3b/int4-autoround"
  "models/Qwen3.6-35B-A3B-Quark-W8A8-INT8|qwen3.6-35b-a3b/quark-w8a8-int8"
)

# Off-target / prior-gen -- removed entirely (see manifest "DROPPED" note).
DROP=(
  models/google_gemma-4-12B-it
  models/Qwen_Qwen3-0.6B
  models/deepreinforce-ai_Ornith-1.0-35B
  models/Qwen3-14B-W4A16-gptq
  models/Qwen3-14B-W4A8-gptq-prepacked
  models/Qwen3-14B-W8A8-autoround
  models/Qwen3.6-27B-W4A16-awq-repack
)

# Intermediates whose bytes are now materialized into a canonical dest -- safe to remove
# AFTER the COMPLETE copies above succeed.
SUBSUMED=(
  models/Qwen3.6-27B-W8A8-sqgptq
  models/Qwen3.6-27B-W8A8-sqgptq-mtp-graft
  models/Qwen3.6-27B-W8A8-sqgptq-vision
  models/Lorbus_Qwen3.6-27B-int4-AutoRound
)
# NOTE: the W4A16/W4A8 *-mtp-graft dirs are intentionally NOT dropped here --
# graft_w4_complete.sh consumes them (LM + MTP delta) to build the complete w4a16/w4a8,
# then drops them itself.

materialize_py() {
  # $1 = abs source dir, $2 = abs dest dir
  APPLY="$APPLY" SRCROOT="$SRCROOT" python3 - "$1" "$2" <<'PY'
import os, sys, shutil
src, dst = sys.argv[1], sys.argv[2]
APPLY = os.environ.get("APPLY") == "1"
SRCROOT = os.environ["SRCROOT"]
SKIP_SUBSTR = (".bak", ".ignore", ".textonly", ".owner")
SKIP_EXACT = {"MTP_GRAFT_NOTES.txt"}
SKIP_DIRS = {"mtp_bf16_patch", ".cache", ".ipynb_checkpoints"}

def host_real(p):
    seen = 0
    while os.path.islink(p):
        seen += 1
        if seen > 32: raise RuntimeError("symlink loop at " + p)
        t = os.readlink(p)
        if t.startswith("/models"):                 # container path -> host path
            t = SRCROOT + t
        elif not os.path.isabs(t):
            t = os.path.join(os.path.dirname(p), t)
        p = t
    return p

def copy_dir(s, d):
    os.makedirs(d, exist_ok=True) if APPLY else None
    total = 0
    for name in sorted(os.listdir(s)):
        if name in SKIP_EXACT or any(x in name for x in SKIP_SUBSTR):
            continue
        sp = os.path.join(s, name)
        real = host_real(sp)
        if not os.path.exists(real):
            print(f"  !! MISSING target for {name} -> {real}"); sys.exit(3)
        if os.path.isdir(real):
            if name in SKIP_DIRS: continue
            total += copy_dir(real, os.path.join(d, name)); continue
        dp = os.path.join(d, name)
        sz = os.path.getsize(real)
        total += sz
        link = " (deref)" if os.path.islink(sp) else ""
        print(f"  + {name:42s} {sz/1e9:7.2f} GB{link}")
        if APPLY:
            shutil.copy2(real, dp)
    return total

t = copy_dir(src, dst)
print(f"  = {t/1e9:.1f} GB total")
PY
}

hr; say "MODE: $([ "$APPLY" = 1 ] && echo APPLY || echo DRY-RUN)   DST=$DST"; hr

# free-space check
avail_k=$(df -Pk "$(dirname "$DST")" | awk 'NR==2{print $4}')
say "free space at dest: $((avail_k/1024/1024)) GB (need ~330 GB transient)"; hr

say "[1/3] MATERIALIZE (de-root + no-links) -> models/files"
for pair in "${COMPLETE[@]}"; do
  rel="${pair%%|*}"; tgt="${pair##*|}"
  s="$SRCROOT/$rel"; d="$DST/$tgt"
  if [ ! -d "$s" ]; then say ">> SKIP $tgt (source missing: $s)"; continue; fi
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then say ">> SKIP $tgt (dest exists, non-empty)"; continue; fi
  say ">> $rel  ->  files/$tgt"
  materialize_py "$s" "$d" || { say "MATERIALIZE FAILED for $tgt"; exit 2; }
  if [ "$APPLY" = 1 ]; then
    chown -R "$UID_KEEP:$GID_KEEP" "$d"
    if find "$d" -type l | grep -q .; then say "!! symlinks remain in $d -- ABORT"; exit 4; fi
  fi
done
hr

say "[2/3] DROP off-target / prior-gen (deleted entirely)"
for rel in "${DROP[@]}"; do
  p="$SRCROOT/$rel"; [ -e "$p" ] || { say "   (already gone) $rel"; continue; }
  say "   rm -rf $rel   ($(du -sh "$p" 2>/dev/null | cut -f1))"
  [ "$APPLY" = 1 ] && rm -rf "$p"
done
hr

say "[3/3] DROP subsumed intermediates (bytes now live under files/)"
say "   guard: only if all COMPLETE dests exist"
ok=1
for pair in "${COMPLETE[@]}"; do
  d="$DST/${pair##*|}"
  if [ "$APPLY" = 1 ] && { [ ! -d "$d" ] || [ -z "$(ls -A "$d" 2>/dev/null)" ]; }; then ok=0; say "   !! dest missing: $d"; fi
done
if [ "$ok" = 1 ]; then
  for rel in "${SUBSUMED[@]}"; do
    p="$SRCROOT/$rel"; [ -e "$p" ] || { say "   (already gone) $rel"; continue; }
    say "   rm -rf $rel"
    [ "$APPLY" = 1 ] && rm -rf "$p"
  done
else
  say "   SKIPPED subsumed-drop (dest verification failed)"
fi
hr
# final de-root sweep: the intermediate files/ and files/<family>/ dirs were created by root
# (os.makedirs); chown the whole tree so NOTHING under files/ is root-owned.
if [ "$APPLY" = 1 ] && [ -d "$DST" ]; then chown -R "$UID_KEEP:$GID_KEEP" "$DST"; say "de-rooted: chown -R $UID_KEEP:$GID_KEEP $DST"; fi
hr
say "NOTE: kept in place for graft_w4_complete.sh (it consumes + drops them):"
say "      $SRCROOT/models/Qwen3.6-27B-W4A16{,-mtp-graft}"
say "      $SRCROOT/models/Qwen3.6-27B-W4A8-sqgptq-prepacked{,-mtp-graft}"
say "DONE ($([ "$APPLY" = 1 ] && echo APPLIED || echo dry-run -- re-run with APPLY=1 to execute))."
