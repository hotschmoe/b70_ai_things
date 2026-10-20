#!/usr/bin/env bash
# Run one headless codex agent (gpt-6.1-sol, high effort) on a prompt file; log to /mnt/vm_8tb/b70/logs/grok-flashnext/<name>.codex.log
set -u
name=$1; prompt=$2; shift 2
cd /mnt/vm_8tb/github/b70_ai_things
exec codex exec -m "${CODEX_MODEL:-gpt-6.1-sol}" -c model_reasoning_effort="${CODEX_EFFORT:-high}" \
  --dangerously-bypass-approvals-and-sandbox -C /mnt/vm_8tb/github/b70_ai_things \
  -o "/mnt/vm_8tb/b70/logs/grok-flashnext/$name.codex.last.md" "$@" - < "$prompt" \
  > "/mnt/vm_8tb/b70/logs/grok-flashnext/$name.codex.log" 2>&1
