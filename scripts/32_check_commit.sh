#!/usr/bin/env bash
# Check if the user's proven commit c51df4300 (vllm 0.20.2rc1.dev2) has NATIVE gemma4
# model support (vs the buggy Transformers-integration fallback in 3ca6ca2).
set -uo pipefail
cd /mnt/vm_8tb/b70/build/vllm
git fetch --all -q 2>&1 | tail -1 || true
echo "=== checkout c51df4300 ==="
git checkout c51df4300 2>&1 | tail -2
git log -1 --oneline
echo "=== native gemma model files at this commit ==="
ls vllm/model_executor/models/ | grep -i gemma || echo "(no gemma files?)"
echo "=== is Gemma4 registered natively? ==="
grep -riE "gemma4|Gemma4" vllm/model_executor/models/registry.py | head -5 || echo "(gemma4 NOT in registry -> still Transformers-integration fallback)"
echo "=== transformers pin in requirements ==="
grep -iE "transformers" requirements/*.txt 2>/dev/null | head -5
echo "=== DONE ==="
