#!/usr/bin/env python3
"""Observe uptime, endpoint errors, RAM and per-card resident VRAM once a minute.

No generation or GPU queries: read HTTP metadata, cgroup memory, and DRM fdinfo.
The server continues to own the GPU lease. The observer never restarts it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request


def process_memory():
    rows = []
    seen = set()
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "comm").read_text().strip() != "strata":
                continue
            status = (proc / "status").read_text()
            rss = re.search(r"^VmRSS:\s+(\d+)", status, re.M)
            cards = []
            for fd in (proc / "fdinfo").iterdir():
                data = dict(line.split(":\t", 1) for line in fd.read_text().splitlines()
                            if line.startswith("drm-") and ":\t" in line)
                cid = data.get("drm-client-id")
                key = (data.get("drm-pdev"), cid)
                if not cid or key in seen:
                    continue
                seen.add(key)
                amount, unit = data.get("drm-resident-vram0", "0 KiB").split()
                cards.append(dict(pci=data.get("drm-pdev"), client=cid,
                                  resident_bytes=int(amount)*{"KiB":1024, "MiB":2**20, "GiB":2**30}[unit]))
            rows.append(dict(pid=int(proc.name), rss_bytes=int(rss[1])*1024 if rss else None, cards=cards))
        except (OSError, ValueError):
            continue
    memory = Path("/sys/fs/cgroup/memory.current")
    return dict(processes=rows, container_memory_bytes=int(memory.read_text()) if memory.exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18085")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.sample:
        print(json.dumps(process_memory()))
        return
    if args.output is None or args.config is None:
        parser.error("--output and --config are required")
    config = json.loads(args.config.read_text())
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    log = Path(config["log"])
    offset = log.stat().st_size if log.exists() else 0
    bad = re.compile(r"fatal|DEVICE_LOST|OUT_OF_RESOURCES|Traceback|(?<!no )\bERROR\b|\bHTTP[^\n]*\b5\d\d\b", re.I)
    script = Path(__file__).read_text()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as output:
        while True:
            row = dict(ts=time.time(), config_path=str(args.config), config_sha256=config_sha256, errors=[])
            for path in ["/health", "/v1/status", "/metrics"]:
                try:
                    with urllib.request.urlopen(args.base_url + path, timeout=10) as response:
                        row[path] = json.load(response)
                except (OSError, ValueError) as exc:
                    row["errors"].append(path + ": " + str(exc))
            try:
                result = subprocess.run(["docker", "exec", "-i", "strata-ab-serve",
                                         "/mnt/vm_8tb/b70/build/strata-ab/venv/bin/python",
                                         "-", "--sample"], input=script, capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip())
                row["memory"] = json.loads(result.stdout)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row["errors"].append("memory: " + str(exc))
            if log.exists():
                if log.stat().st_size < offset:
                    offset = 0
                with log.open(errors="replace") as engine_log:
                    engine_log.seek(offset)
                    lines = engine_log.readlines()
                    offset = engine_log.tell()
                row["engine_error_lines"] = [line.rstrip() for line in lines if bad.search(line)]
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
            if args.once:
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
