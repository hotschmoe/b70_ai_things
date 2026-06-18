#!/usr/bin/env bash
# Run a local .sh script on the b70 Unraid box cleanly, with optional env vars.
# Bash equivalent of runremote.ps1 (for the Linux dev machine).
# Strips CRLF/BOM, prepends `export KEY=VALUE` for any KEY=VALUE args, base64-encodes
# (avoids any shell-quoting/encoding issues over ssh), decodes + executes remotely under bash.
#
# Usage:   scripts/runremote.sh scripts/foo.sh [KEY=VALUE ...] [host=<sshhost>]
# Example: scripts/runremote.sh scripts/49_quantize_27b_w8a8.sh METHOD=rtn host=b70
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.sh> [KEY=VALUE ...] [host=<sshhost>]" >&2
  exit 2
fi

SCRIPT_PATH="$1"; shift
SSH_HOST="b70"
EXPORTS=""

for a in "$@"; do
  if [[ "$a" =~ ^host=(.+)$ ]]; then
    SSH_HOST="${BASH_REMATCH[1]}"
    continue
  fi
  if [[ "$a" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
    key="${BASH_REMATCH[1]}"; val="${BASH_REMATCH[2]}"
    # single-quote the value, escaping embedded single quotes
    val="${val//\'/\'\\\'\'}"
    EXPORTS+="export ${key}='${val}'"$'\n'
    continue
  fi
  echo "warning: ignoring unrecognized arg '$a'" >&2
done

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "error: script not found: $SCRIPT_PATH" >&2
  exit 1
fi

# Read script, strip CRLF and a leading UTF-8 BOM.
raw="$(sed 's/\r$//' "$SCRIPT_PATH" | sed '1s/^\xEF\xBB\xBF//')"

# Insert exports after a shebang line if present, else at the top.
if [[ "$raw" == '#!'* ]]; then
  shebang="${raw%%$'\n'*}"
  rest="${raw#*$'\n'}"
  raw="${shebang}"$'\n'"${EXPORTS}${rest}"
else
  raw="${EXPORTS}${raw}"
fi

b64="$(printf '%s' "$raw" | base64 | tr -d '\n')"
if [[ ${#b64} -gt 100000 ]]; then
  echo "error: script too large for inline transport (${#b64} b64 chars)" >&2
  exit 1
fi

ssh "$SSH_HOST" "echo $b64 | base64 -d | bash -s"
