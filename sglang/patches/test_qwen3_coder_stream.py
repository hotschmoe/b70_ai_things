"""Offline validation: patched qwen3_coder streaming == baked == non-streaming.

Run INSIDE the sglang container (it imports the baked sglang detector):
    docker cp sglang/patches/qwen3_coder_detector.py b70_daily_0:/tmp/qwen3_coder_detector_patched.py
    docker cp sglang/patches/test_qwen3_coder_stream.py b70_daily_0:/tmp/test_qwen3_coder_stream.py
    docker exec b70_daily_0 python3 /tmp/test_qwen3_coder_stream.py    # exit 0 = all pass
This is the de-risk gate for the serve.sh mount: it asserts the patched detector's streamed
`arguments` are byte-identical to the baked detector AND to the non-streaming parse, across
14 cases x 11 chunk sizes. Exits non-zero on any mismatch.

Loads the BAKED detector from site-packages and the PATCHED detector from
/tmp/qwen3_coder_detector_patched.py, feeds identical raw model output through
both at many chunk sizes, and asserts per-tool concatenated `arguments` are
byte-identical to each other AND to the non-streaming detect_and_parse, and are
valid JSON. Also reports the silent-gap improvement on a large value.
"""
import importlib.util
import json
import sys

from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector as Baked
from sglang.srt.entrypoints.openai.protocol import Tool, Function

spec = importlib.util.spec_from_file_location(
    "patched_detector", "/tmp/qwen3_coder_detector_patched.py"
)
pmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pmod)
Patched = pmod.Qwen3CoderDetector


def mktool(name, props, required=None):
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": props,
                "required": required or [],
            },
        ),
    )


def tc(func, params):
    """Build one Qwen3.6 XML tool call. params: list of (name, value_str)."""
    s = "<tool_call>\n<function=%s>\n" % func
    for n, v in params:
        s += "<parameter=%s>\n%s\n</parameter>\n" % (n, v)
    s += "</function>\n</tool_call>"
    return s


def stream(DetClass, raw, tools, chunk):
    det = DetClass()
    per = {}       # tool_index -> {"name": str|None, "args": str}
    feeds_argbytes = []
    i = 0
    while i < len(raw):
        piece = raw[i : i + chunk]
        i += chunk
        res = det.parse_streaming_increment(piece, tools)
        ab = 0
        for c in (res.calls or []):
            d = per.setdefault(c.tool_index, {"name": None, "args": ""})
            if getattr(c, "name", None):
                d["name"] = c.name
            p = c.parameters or ""
            d["args"] += p
            ab += len(p)
        feeds_argbytes.append(ab)
    return per, feeds_argbytes


def nonstream(raw, tools):
    det = Baked()
    res = det.detect_and_parse(raw, tools)
    out = {}
    for idx, c in enumerate(res.calls or []):
        out[c.tool_index if c.tool_index is not None else idx] = {
            "name": c.name,
            "args": c.parameters or "",
        }
    return out


def max_silent_run(feeds):
    best = cur = 0
    for ab in feeds:
        if ab == 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


# --------------------------------------------------------------------------
# Test cases: (label, tools, raw)
# --------------------------------------------------------------------------
T_WRITE = [mktool("write_file", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])]
T_MIXED = [mktool("cfg", {"name": {"type": "string"}, "count": {"type": "integer"}, "enabled": {"type": "boolean"}, "opts": {"type": "object"}})]
T_UNTYPED = [mktool("u", {})]  # no property types

BIG = "\n".join('  <div class="r-%d" data-x=\'q\'>line %d & co</div>' % (i, i) for i in range(200))
TRICKY = 'has "quotes", back\\slash, tab\there, unicode é☃\U0001f600, <html> & </html>, newline\nmid'

CASES = [
    ("single-large-string", T_WRITE, tc("write_file", [("path", "index.html"), ("content", BIG)])),
    ("tricky-escapes", T_WRITE, tc("write_file", [("path", "a/b.txt"), ("content", TRICKY)])),
    ("empty-content", T_WRITE, tc("write_file", [("path", "e.txt"), ("content", "")])),
    ("null-literal", T_WRITE, tc("write_file", [("path", "n.txt"), ("content", "null")])),
    ("null-prefix-string", T_WRITE, tc("write_file", [("path", "n.txt"), ("content", "nullable value here")])),
    ("nul-short", T_WRITE, tc("write_file", [("path", "n.txt"), ("content", "nul")])),
    ("content-has-param-end", T_WRITE, tc("write_file", [("path", "x"), ("content", "before</parameter>after")])),
    ("content-has-param-prefix", T_WRITE, tc("write_file", [("path", "x"), ("content", "a<parameter=z>b")])),
    ("mixed-types", T_MIXED, tc("cfg", [("name", "svc"), ("count", "42"), ("enabled", "true"), ("opts", '{"a": 1, "b": [2, 3]}')])),
    ("untyped-passthrough", T_UNTYPED, tc("u", [("k", "12345"), ("s", "hello world")])),
    ("two-tool-calls", T_WRITE, tc("write_file", [("content", "first " + BIG[:300])]) + "\n" + tc("write_file", [("content", "second file body")])),
    ("normal-text-before", T_WRITE, "Sure, writing it now.\n" + tc("write_file", [("path", "p"), ("content", "body\nwith lines")])),
    ("leading-trailing-nl", T_WRITE, tc("write_file", [("content", "no-strip-internal\nkept")])),
    ("emoji-boundary", T_WRITE, tc("write_file", [("content", "x" * 30 + "\U0001f600" + "y" * 30)])),
]

CHUNKS = [1, 2, 3, 5, 7, 11, 12, 13, 16, 64, 100000]

fails = 0
checks = 0
for label, tools, raw in CASES:
    ref = nonstream(raw, tools)
    for chunk in CHUNKS:
        baked, _ = stream(Baked, raw, tools, chunk)
        patched, pfeeds = stream(Patched, raw, tools, chunk)
        # normalize keys to sorted tool order
        for d in (baked, patched, ref):
            pass
        # 1) patched == baked (per tool: name + args)
        ok = True
        keys = sorted(set(baked) | set(patched) | set(ref))
        for k in keys:
            b = baked.get(k, {"name": None, "args": "<MISSING>"})
            p = patched.get(k, {"name": None, "args": "<MISSING>"})
            r = ref.get(k, {"name": None, "args": "<MISSING>"})
            if p["args"] != b["args"] or p["name"] != b["name"]:
                ok = False
                print("FAIL [%s chunk=%s] tool %s patched!=baked\n  baked=%r\n  patch=%r" % (label, chunk, k, b, p))
            if p["args"] != r["args"]:
                ok = False
                print("FAIL [%s chunk=%s] tool %s patched!=nonstream\n  nonstr=%r\n  patch =%r" % (label, chunk, k, r, p))
            # valid JSON
            try:
                json.loads(p["args"])
            except Exception as e:
                ok = False
                print("FAIL [%s chunk=%s] tool %s patched args not valid JSON: %s\n  %r" % (label, chunk, k, e, p["args"]))
        checks += 1
        if not ok:
            fails += 1

# silent-gap demonstration on the large case at a realistic small chunk
_, bfeeds = stream(Baked, CASES[0][2], CASES[0][1], 7)
_, pfeeds = stream(Patched, CASES[0][2], CASES[0][1], 7)
print("\n--- silent-gap (single-large-string, chunk=7) ---")
print("baked   max consecutive zero-arg feeds: %d  (of %d feeds)" % (max_silent_run(bfeeds), len(bfeeds)))
print("patched max consecutive zero-arg feeds: %d  (of %d feeds)" % (max_silent_run(pfeeds), len(pfeeds)))

print("\n==== %d/%d chunk-configs passed, %d failed ====" % (checks - fails, checks, fails))
sys.exit(1 if fails else 0)
