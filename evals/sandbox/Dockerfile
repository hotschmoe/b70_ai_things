# Tier-1 code-execution sandbox (EvalPlus grader ONLY).
#
# The dangerous step of Tier 1 — EXECUTING model-generated code to grade pass@1 — runs in
# THIS container, never on the dev box. The orchestrator invokes it with:
#   --network none            (untrusted code gets no network)
#   --user <uid>:<gid>        (non-root; output files owned by you, not root)
#   a THROWAWAY cache copy     (can't poison your real ~/.cache/evalplus)
#   --memory / --pids-limit   (blast radius caps)
#
# Pinned to the SAME evalplus version the dev box uses for codegen+sanitize, so the
# host-side sanitize and the in-container evaluate always agree.
FROM python:3.11-slim
ARG EVALPLUS_VERSION=0.3.1
RUN pip install --no-cache-dir "evalplus==${EVALPLUS_VERSION}"
WORKDIR /work
# No ENTRYPOINT: the orchestrator passes the full `evalplus.evaluate ...` command.
