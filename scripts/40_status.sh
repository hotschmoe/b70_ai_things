#!/usr/bin/env bash
# Quick status of a vLLM container. NAME (default vllm_qwen3), PORT (18080).
NAME="${NAME:-vllm_qwen3}"; PORT="${PORT:-18080}"
echo "=== state ==="; docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$NAME" 2>/dev/null
echo -n "=== health: "; curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && echo HEALTHY || echo "not ready"
echo "=== key log lines ==="
docker logs "$NAME" 2>&1 | grep -iE 'Resolved architecture|XPUFp8|Fp8LinearMethod|GDN|Selected|Model loading took|Available KV cache|GPU KV cache size|Maximum concurrency|Application startup complete|out of memory|No available memory|insufficient|KeyError|ValueError|RuntimeError|Traceback|Engine core' | grep -viE 'OperatorEntry|registered|dispatch|splitting_ops' | tail -16
