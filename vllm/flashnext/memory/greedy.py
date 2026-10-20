"""Two deterministic identity-checked arithmetic probes; no speed claim."""
import json
import sys
import urllib.request

port, output = sys.argv[1:]
base = f"http://127.0.0.1:{port}"
with urllib.request.urlopen(base + "/v1/models") as response:
    ids = [item["id"] for item in json.load(response)["data"]]
assert ids[0] == "hotschmoe-dd" and "qwen3.8-flash-next-iq3-xxs" in ids, ids
rows = []
for prompt, expected in (("17 * 19", "323"), ("23 * 29", "667")):
    body = dict(model=ids[0], messages=[dict(role="user", content=f"Return only the integer: {prompt}.")],
                max_tokens=16, temperature=0, chat_template_kwargs=dict(enable_thinking=False))
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"].strip()
    rows.append(dict(prompt=prompt, expected=expected, text=text, match=text == expected))
with open(output, "w", encoding="ascii") as f:
    json.dump(dict(ids=ids, rows=rows), f, indent=2)
assert all(r["match"] for r in rows), rows
print("IDENTITY_GREEDY_OK")
