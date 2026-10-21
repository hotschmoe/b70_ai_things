"""Yield to already-running/queued higher-priority GPU lease wrappers."""
import os
import time
from pathlib import Path

priority = {"L1", "O4", "O5", "F1", "C7b", "H1e"}
while True:
    active = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(x.endswith((b"/bin/gpu-run", b"/harness/xpu-run.sh")) for x in cmd[:3]):
                continue
            env = (proc / "environ").read_bytes().split(b"\0")
            agent = next((x[10:].decode("ascii", "replace") for x in env if x.startswith(b"B70_AGENT=")), "")
            if agent in priority:
                active.add(agent)
        except (OSError, ProcessLookupError):
            pass
    if not active:
        print("X6_PRIORITY_GAP", flush=True)
        break
    print("X6_YIELD " + ",".join(sorted(active)), flush=True)
    time.sleep(20)
