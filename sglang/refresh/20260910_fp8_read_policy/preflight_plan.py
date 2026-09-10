#!/usr/bin/env python3
"""Print the reviewed plan; --run requires both inherited GPU leases."""
import importlib.util
import json
from pathlib import Path
import signal
import sys

HELPER = Path(__file__).resolve().parents[1] / "20260910_main/preflight.py"
IMAGE = "sha256:14ee7d0112b4e321ea7618a2b6c6f351b7428c2efe43deb3c2e51d128c7dd57d"
ROOT = Path("/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-fp8-read-policy-preflight")
SOURCE = ROOT.parent / "sglang-fp8-read-policy-image/oracle-source"

def main():
    if sys.argv[1:] != ["--run"]:
        if sys.argv[1:]:
            raise SystemExit("only --run is supported")
        print(json.dumps(dict(image=IMAGE, output=str(ROOT), oracle_source=str(SOURCE),
            command=["bin/gpu-run", sys.executable, str(Path(__file__).resolve()), "--run"],
            stages=["strict per-card health", "compiled pair P2P=0 health"],
            scope="health only; no serving or attention oracle executed"), indent=2))
        return 0
    spec = importlib.util.spec_from_file_location("read_policy_health", HELPER)
    health = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health)
    health.ROOT = ROOT
    health.IMAGES = [("sglang-fp8-read-policy", IMAGE)]
    def interrupted(signum, frame):
        if health.CLEANUP_PENDING:
            return
        raise KeyboardInterrupt("preflight interrupted by signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    return health.main()

if __name__ == "__main__":
    raise SystemExit(main())
