#!/usr/bin/env bash
# Does llm-scaler b8.3.1's vLLM support the gemma4 architecture? (released 2026-06-03)
IMG="intel/llm-scaler-vllm:0.14.0-b8.3.1"
echo "=== gemma-related registered architectures ==="
docker run --rm --entrypoint python "$IMG" -c '
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print("gemma archs:", [a for a in archs if "gemma" in a.lower()])
' 2>&1 | grep -iE 'gemma|error' | tail -5
echo "=== gemma model files in image ==="
docker run --rm --entrypoint bash "$IMG" -c 'ls /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/ | grep -i gemma'
echo "=== DONE ==="
