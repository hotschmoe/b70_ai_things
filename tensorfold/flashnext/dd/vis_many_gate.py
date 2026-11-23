#!/usr/bin/env python3
"""Many-images gate (2026-11-23, --vision-max-images 20 --vision-image-tokens 6000): one request with N distinct
generated PNGs must answer normally (N <= limit), and N = limit + 1 must be refused with HTTP 400. Prints only counts,
timings, finish reasons and the short answer.
Usage: vis_many_gate.py URL N [expect_refused]
"""
import base64, json, struct, sys, time, urllib.error, urllib.request, zlib

url, n = sys.argv[1].rstrip('/'), int(sys.argv[2])
expect_refused = len(sys.argv) > 3 and sys.argv[3] == "refused"


def png(w, h, rgb, box):
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            inside = box[0] <= x < box[0] + 40 and box[1] <= y < box[1] + 40
            row += bytes((255 - c for c in rgb)) if inside else bytes(rgb)
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


parts = []
for i in range(n):
    rgb = ((37 * i) % 256, (91 * i + 60) % 256, (151 * i + 120) % 256)
    data = png(320, 240, rgb, (10 + 13 * i % 260, 10 + 7 * i % 190))
    parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}})
parts.append({"type": "text", "text": "How many images did I send? Answer with one number."})
body = {"model": "hotschmoe-dd", "messages": [{"role": "user", "content": parts}], "max_tokens": 256, "temperature": 0, "reasoning_effort": "none"}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=170) as r:
        d = json.load(r)
    status = 200
except urllib.error.HTTPError as e:
    status, d = e.code, {"error": e.read()[:300].decode(errors="replace")}
except Exception as e:  # noqa: BLE001
    status, d = None, {"error": repr(e)[:200]}
dt = round(time.time() - t0, 2)
if expect_refused:
    ok = status == 400
    detail = {"n": n, "status": status, "s": dt, "error": str(d.get("error"))[:160]}
else:
    ch = (d.get("choices") or [{}])[0]
    ans = (ch.get("message") or {}).get("content") or ""
    ok = status == 200 and ch.get("finish_reason") in ("stop", "length")
    detail = {"n": n, "status": status, "s": dt, "finish": ch.get("finish_reason"), "prompt_tokens": d.get("usage", {}).get("prompt_tokens"), "answer": ans.strip()[:40], "error": str(d.get("error", ""))[:160]}
print("GATE vis_many", "PASS" if ok else "FAIL", json.dumps(detail))
sys.exit(0 if ok else 1)
