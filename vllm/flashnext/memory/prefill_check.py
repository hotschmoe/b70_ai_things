"""Two concurrent near-limit prefills verify reuse of profiled workspace."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import urllib.request

port, output = sys.argv[1:]
url = f"http://127.0.0.1:{port}/v1/completions"

def request(seed):
    # Exact token count; content is intentionally only a memory probe.
    prompt = [100 + (i*17 + seed) % 1000 for i in range(2047)]
    body = json.dumps(dict(model="hotschmoe-dd", prompt=prompt,
                           max_tokens=1, temperature=0, ignore_eos=True)).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    usage = result["usage"]
    assert usage["prompt_tokens"] == 2047 and usage["completion_tokens"] == 1, usage
    return dict(seed=seed, usage=usage, choices=result["choices"])

with ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(request, (1, 2)))
with open(output, "w", encoding="ascii") as f:
    json.dump(rows, f, indent=2)
print("CONCURRENT_2047_TOKEN_PREFILL_OK")
