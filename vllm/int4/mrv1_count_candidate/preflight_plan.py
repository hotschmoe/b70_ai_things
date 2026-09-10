#!/usr/bin/env python3
"""Print the reviewed plan; --run requires both inherited GPU leases."""
import importlib.util
import json
from pathlib import Path
import signal
import sys

HELPER = Path(__file__).resolve().parents[3] / "sglang/refresh/20260910_main/preflight.py"
IMAGE = "sha256:7b107d0e675390fedbbdbe743f104c08072b6175ce8ccbbfdad64dc82bf90ad1"
ROOT = Path("/mnt/vm_8tb/b70/results/bang_isolation_20260910/mrv1-counts-phase-preflight")

def main():
    if sys.argv[1:] != ["--run"]:
        if sys.argv[1:]:
            raise SystemExit("only --run is supported")
        print(json.dumps(dict(image=IMAGE, output=str(ROOT),
            command=["bin/gpu-run", sys.executable, str(Path(__file__).resolve()), "--run"],
            stages=["strict per-card health", "compiled pair P2P=0 health"],
            scope="health only; no serving or attention oracle executed"), indent=2))
        return 0
    spec = importlib.util.spec_from_file_location("mrv1_phase_health", HELPER)
    health = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health)
    health.ROOT = ROOT
    health.IMAGES = [("vllm-phase-mrv1-counts", IMAGE)]
    def interrupted(signum, frame):
        if health.CLEANUP_PENDING:
            return
        raise KeyboardInterrupt("preflight interrupted by signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    return health.main()

if __name__ == "__main__":
    raise SystemExit(main())
