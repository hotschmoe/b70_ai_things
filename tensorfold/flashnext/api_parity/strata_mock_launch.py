#!/usr/bin/env python3
"""Run Strata's real serve/server.py (the production HTTP layer) over a scripted MockEngine. CPU only.

    strata_mock_launch.py PORT [CONTEXT]

The production config (hotschmoe-dd-0216.json) supplies the tokenizer, names and reasoning defaults; the engine is a
MockEngine that picks its script by "@script=NAME" in the decoded prompt (scenarios.SCRIPTS). Not server_intel.py: no
Intel telemetry, nothing that opens a device. Never use the production port.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TERMINAL = Path("/mnt/vm_8tb/b70/build/strata-0216/src/terminal")
ENGINE = TERMINAL.parent / "engine"
for p in (str(TERMINAL), str(ENGINE), str(TERMINAL.parent / "tools")):
    sys.path.insert(0, p)

import scenarios  # noqa: E402
import serve.server as S  # noqa: E402

port = int(sys.argv[1])
if port in (18080, 18124):
    raise SystemExit("refusing a production port")
context = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
prod = json.loads(Path("/mnt/vm_8tb/github/b70_ai_things/strata/flash-next/dd/hotschmoe-dd-0216.json").read_text())
cfg = {k: prod[k] for k in ("model_name", "aliases", "reasoning_budget_tokens", "reasoning_effort", "tokenizer") if k in prod}
cfg["parallel"] = prod.get("parallel", 4)
cfg_path = Path(tempfile.mkdtemp(prefix="strata-mock-")) / "mock.json"
cfg_path.write_text(json.dumps(cfg))

_orig_init = S.MockEngine.__init__


def init(self, tokenizer, script, max_context=32768, **kw):
    _orig_init(self, tokenizer, script, max_context=context, **kw)
    self.calls = []


def generate(self, ids, max_new, sampling, cancel, embeddings=None):
    self.last_prompt = list(ids)
    self.last_embeddings = embeddings
    text = self.tok.decode(list(ids)) if hasattr(self.tok, "decode") else ""
    name = "default"
    at = text.rfind("@script=")
    if at >= 0:
        end = at + 8
        while end < len(text) and (text[end].isalnum() or text[end] in "_-"):
            end += 1
        name = text[at + 8:end]
    if name not in scenarios.SCRIPTS:
        name = "default"
    script = scenarios.SCRIPTS[name]
    delay = script.get("delay_ms", 0) / 1000.0
    body = scenarios.strata_text(name)
    if script.get("echo_request"):
        body = json.dumps({"max_new": max_new, "sampling": sampling or {}})
    toks = self.tok.encode(body, parse_special=True) + self.tok.encode(S.IM_END, parse_special=True)
    for t in toks[:max_new]:
        if cancel.is_set():
            return
        if delay:
            time.sleep(delay)
        yield t


S.MockEngine.__init__ = init
S.MockEngine.generate = generate

_svc_init = S.Service.__init__


def svc_init(self, *a, **kw):
    _svc_init(self, *a, **kw)
    self.model_arch = "qwen4exp"      # as main() sets for the strata engine: the model-card defaults and family
    self.set_family(None)


S.Service.__init__ = svc_init
sys.exit(S.main(["--engine", "mock", "--config", str(cfg_path), "--port", str(port), "--no-api-key"]))
