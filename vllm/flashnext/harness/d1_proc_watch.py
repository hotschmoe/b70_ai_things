#!/usr/bin/env python3
"""CPU-only /proc observer. No ptrace, signal, XPU import or metrics ioctl.

python3 d1_proc_watch.py --pid HOST_ENGINE_PID --out DIR --seconds 120
Run with the already-leased server owner's consent, outside latency timing.
"""

import argparse
import json
from pathlib import Path
import time


def read(path):
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        return "READ_ERROR:" + repr(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("seconds must be 1..300")
    root = Path(f"/proc/{args.pid}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("maps", "status", "stat", "limits"):
        (args.out / f"initial-{name}.txt").write_text(read(root / name), encoding="ascii", errors="backslashreplace")
    deadline = time.monotonic() + args.seconds
    identities = {}
    with (args.out / "proc.jsonl").open("a", encoding="ascii") as output:
        while root.exists() and time.monotonic() < deadline:
            started = time.monotonic()
            threads = []
            for task in sorted((root / "task").glob("*")):
                tid = int(task.name)
                if tid not in identities:
                    identities[tid] = {line.split(':',1)[0]: line.split(':',1)[1].strip()
                                       for line in read(task / "status").splitlines()
                                       if line.startswith(('Name:', 'NSpid:', 'NStgid:'))}
                threads.append(dict(tid=int(task.name), stat=read(task / "stat"),
                                    identity=identities[tid], wchan=read(task / "wchan"),
                                    syscall=read(task / "syscall")))
            row = dict(wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                       pid=args.pid, status=read(root / "status"), threads=threads,
                       fd_count=sum(1 for _ in (root / "fd").glob("*")))
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
            output.flush()
            time.sleep(max(0, 1 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
