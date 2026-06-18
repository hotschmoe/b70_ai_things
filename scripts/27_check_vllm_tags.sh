#!/usr/bin/env bash
# List available intel/vllm Docker Hub tags (looking for a recent xpu tag with gemma4).
echo "=== intel/vllm tags ==="
curl -s 'https://hub.docker.com/v2/repositories/intel/vllm/tags?page_size=100' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(t["name"], t.get("last_updated","")[:10]) for t in d.get("results",[])]' 2>/dev/null \
  | sort || echo "(could not parse / repo not found)"
echo
echo "=== try: does intel/vllm exist at all? ==="
curl -s -o /dev/null -w "%{http_code}\n" 'https://hub.docker.com/v2/repositories/intel/vllm/'
echo "=== DONE ==="
