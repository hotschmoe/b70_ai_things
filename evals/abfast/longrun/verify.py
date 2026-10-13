#!/usr/bin/env python3
"""Hidden-suite scoring of one submission tarball for a longrun task (no model involved).

  verify.py TASK_ID SUBMISSION_TAR|DIR   -> prints JSON score
The submission is extracted into a FRESH container (network none) that has NO target binaries
(/opt/target is empty); it is built with the task's build_cmd, then /hidden/check.py runs the
hidden milestones against it.
"""
import hashlib, io, json, os, sys, tarfile, time, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lr_common as C  # noqa: E402


def blob_scan(task, tar_bytes):
    """Flag submissions that smuggle the target binary (any file containing a 256-byte slice of it)."""
    flagged = []
    for b in task.get("target_bins", []):
        p = f"{C.LDATA}/{task['id']}/target/{b}"
        if not os.path.exists(p):
            continue
        d = open(p, "rb").read()
        if len(d) < 4096:
            continue
        needles = [d[o:o + 256] for o in (1024, len(d) // 2, len(d) - 1024)]
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
            for m in tf:
                if m.isfile() and m.size >= 256:
                    data = tf.extractfile(m).read()
                    if any(n in data for n in needles):
                        flagged.append(m.name)
    return flagged


def score(task, sub_tar, tag=None):
    import gzip
    if sub_tar[:2] == b"\x1f\x8b":
        sub_tar = gzip.decompress(sub_tar)
    tid = task["id"]
    tag = tag or uuid.uuid4().hex[:8]
    out = {"id": tid, "built": False, "milestones": [], "tests_passed": 0, "tests_total": 0}
    hid = f"{C.LDATA}/{tid}/hidden"
    flagged = blob_scan(task, sub_tar)
    box = C.LBox(f"abfast-lverify-{tid}-{tag}-{uuid.uuid4().hex[:6]}"[:100], task.get("image", C.IMAGE), net="none", mem="8g", cpus="4")
    try:
        box.put_tar(sub_tar, "/work/sub")
        box.put_tar(C.tar_dir(hid), "/hidden", user="0")
        box.put_tar(C.tar_dir(os.path.join(C.HERE, "lib")), "/hidden/lib", user="0")
        t0 = time.time()
        if flagged:
            out["cheat_flagged"] = flagged[:5]
            out["build_log"] = "submission contains target binary bytes; scored zero"
        else:
            rc, log = box.exec(f"cd /work/sub && {task['build_cmd']}", timeout=task.get("build_timeout", 900))
            out["build_rc"] = rc
            out["build_log"] = log[-1500:]
            out["built"] = rc == 0
        out["build_s"] = round(time.time() - t0, 1)
        total_ms = task.get("milestones_total")
        if out["built"] or task.get("check_even_if_build_fails"):
            t1 = time.time()
            rc, log = box.exec("cd /work/sub && PYTHONPATH=/hidden/lib python3 /hidden/check.py /work/sub",
                               timeout=task.get("check_timeout", 1800))
            out["check_rc"] = rc
            out["check_s"] = round(time.time() - t1, 1)
            res = None
            for line in log.splitlines():
                if line.startswith("RESULT "):
                    res = json.loads(line[7:])
            if res:
                out.update(res)
            else:
                out["check_log"] = log[-1500:]
        if not out["milestones"] and total_ms:
            out["milestones"] = [{"id": f"m{i+1}", "passed": 0, "total": 0, "ok": False} for i in range(total_ms)]
    finally:
        box.rm()
    ms = out["milestones"]
    out["milestones_passed"] = sum(1 for m in ms if m["ok"])
    out["milestones_total"] = len(ms)
    out["fraction"] = round(out["milestones_passed"] / max(len(ms), 1), 4)
    out["tests_fraction"] = round(out["tests_passed"] / max(out["tests_total"], 1), 4)
    return out


def main():
    tid, sub = sys.argv[1], sys.argv[2]
    task = [t for t in C.load_tasks() if t["id"] == tid][0]
    data = C.tar_dir(sub) if os.path.isdir(sub) else open(sub, "rb").read()
    r = score(task, data)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
