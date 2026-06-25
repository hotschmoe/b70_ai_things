#!/usr/bin/env bash
# agentic-eval/serve/serve_config.sh <label> [start|stop|smoke]
#
# Thin dispatcher: maps a config label to its verified shelf recipe (rdy_to_serve/<dir>/serve.sh)
# and applies the eval's shared env overrides (PORT/MAXLEN/MAXSEQS/TOOLCALL). It does NOT acquire the
# GPU lease -- the caller (run/run_all.sh) holds ONE gpu-run lease around the whole config so the serve
# stays leased for the duration of the eval (serve.sh's `docker run -d` detaches; the lease must be
# held by a still-alive parent, not by the serve start which returns once healthy).
#
#   start  serve + wait healthy + coherence gen-probe, stay up   (default)
#   stop   graceful teardown + release (wedge-guarded for TP=2)
#   smoke  serve + probe + immediate teardown (serve-path sanity, no harnesses)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"

LABEL="${1:?usage: serve_config.sh <label> [start|stop|smoke]}"
ACTION="${2:-start}"
ae_set_config "$LABEL" || exit 2

SERVE="$REPO_ROOT/rdy_to_serve/$EVAL_SERVE_DIR/serve.sh"
[ -f "$SERVE" ] || { echo "serve_config.sh: missing recipe $SERVE" >&2; exit 2; }

# Shared overrides for comparability (see configs.sh). Exported so serve.sh/lib.sh pick them up.
export PORT="$EVAL_PORT" MAXLEN="$EVAL_MAXLEN" MAXSEQS="$EVAL_MAXSEQS"
export TOOLCALL="$EVAL_TOOLCALL" TOOLPARSER="$EVAL_TOOLPARSER" REASONPARSER="$EVAL_REASONPARSER"
export PREFIXCACHE="$EVAL_PREFIX_CACHE"
# Per-config serve-env overrides from configs.sh (e.g. 27b-w8a8 -> GRAPH=0 enforce-eager, the MTP+capture
# crash workaround; JOURNAL 2026-06-25). Applied for all actions so start/stop see the same recipe knobs.
[ -n "${EVAL_SERVE_ENV:-}" ] && { export ${EVAL_SERVE_ENV}; ae_log "per-config serve env: $EVAL_SERVE_ENV"; }

ae_log "serve $LABEL ($EVAL_ARCH/$EVAL_SCHEME, ${EVAL_CARDS} card(s), id=$EVAL_SERVED) action=$ACTION port=$EVAL_PORT maxlen=$EVAL_MAXLEN"
bash "$SERVE" "$ACTION"
rc=$?
# After a real start, double-check identity (cheap insurance against a stale container on the port).
if [ "$ACTION" = start ] && [ $rc -eq 0 ]; then
  ae_verify_identity || rc=1
fi
exit $rc
