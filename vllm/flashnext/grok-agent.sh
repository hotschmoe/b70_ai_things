#!/usr/bin/env bash
# Run one headless grok agent on a prompt file; log to /mnt/vm_8tb/b70/logs/grok-flashnext/<name>.log
set -u
name=$1; prompt=$2; shift 2
cd /mnt/vm_8tb/github/b70_ai_things
exec grok --prompt-file "$prompt" --always-approve -m "${GROK_MODEL:-grok-4.7}" \
  --cwd /mnt/vm_8tb/github/b70_ai_things "$@" \
  > "/mnt/vm_8tb/b70/logs/grok-flashnext/$name.log" 2>&1
