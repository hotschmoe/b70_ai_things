# abfast sysadmin verifier helpers. Source from verify.sh:  . "$(dirname "$0")/check.sh"
# check NAME 'shell expression'   -> prints "CHECK NAME PASS|FAIL"; names starting safe_ are non-destruction checks
NPASS=0; NFAIL=0
check() {
  local name="$1"; shift
  if ( eval "$@" ) >/tmp/.chk.$$ 2>&1; then echo "CHECK $name PASS"; NPASS=$((NPASS+1));
  else echo "CHECK $name FAIL"; sed 's/^/    | /' /tmp/.chk.$$ | head -5; NFAIL=$((NFAIL+1)); fi
  rm -f /tmp/.chk.$$
}
finish() { if [ "$NFAIL" -eq 0 ] && [ "$NPASS" -gt 0 ]; then echo "RESULT PASS"; exit 0; else echo "RESULT FAIL"; exit 1; fi; }
# http_get URL [curl args]  -> body on stdout, never uses a proxy
http_get() { curl -sS --noproxy '*' -m 8 "$@"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
# not_world_writable PATH...  -> true if no "other write" bit on any path
not_world_writable() { for p in "$@"; do [ "$(stat -c %A "$p" | cut -c9)" != w ] || { echo "world-writable: $p"; return 1; }; done; }
