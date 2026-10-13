"""Shared constants/helpers for the abfast longrun tier (host side)."""
import json, os, subprocess, sys, tarfile, io, time

HERE = os.path.dirname(os.path.abspath(__file__))
ABFAST = os.path.dirname(HERE)
DATA = "/mnt/vm_8tb/b70/evals/abfast"
LDATA = f"{DATA}/longrun"          # per-task built artifacts: LDATA/<id>/{agent,target,hidden,ref_sub,ref_partial}
IMAGE = os.environ.get("ABFAST_LONG_IMAGE", "abfast-long:1")
TASKS_FILE = os.path.join(ABFAST, "tasks_longrun.json")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_tasks(path=TASKS_FILE):
    return json.load(open(path))["tasks"]


def tar_dir(path, arcprefix="."):
    """In-memory tar of a directory's contents (paths relative to it)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for root, dirs, files in os.walk(path):
            dirs.sort()
            for f in sorted(files) + [d for d in dirs if os.path.islink(os.path.join(root, d))]:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, path)
                tf.add(full, arcname=os.path.join(arcprefix, rel), recursive=False)
            for d in dirs:
                full = os.path.join(root, d)
                if not os.path.islink(full):
                    tf.add(full, arcname=os.path.join(arcprefix, os.path.relpath(full, path)), recursive=False)
    return buf.getvalue()


class LBox:
    """Throwaway container. net='host' for agents (endpoint access), 'none' for verifiers."""
    def __init__(self, name, image=IMAGE, net="host", mem="12g", cpus="6"):
        self.name = name
        r = sh(["docker", "run", "-d", "--name", name, "--network", net, "--memory", mem, "--cpus", cpus,
                "--pids-limit", "4096", "--cap-add", "SYS_PTRACE", "-e", "ZIG_GLOBAL_CACHE_DIR=/home/agent/.zigcache",
                image, "sleep", "infinity"])
        if r.returncode:
            raise RuntimeError(r.stderr)

    def exec(self, script, timeout=None, stdin=None, env=None, user=None, workdir=None):
        cmd = ["docker", "exec", "-i"] + (["-u", user] if user else []) + (["-w", workdir] if workdir else [])
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [self.name, "bash", "-c", script]
        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT"
        return p.returncode, (p.stdout + p.stderr).decode(errors="replace")

    def exec_bytes(self, script, timeout=None, user=None):
        cmd = ["docker", "exec"] + (["-u", user] if user else []) + [self.name, "bash", "-c", script]
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout

    def put_tar(self, data, dest, user=None):
        rc, o = self.exec(f"mkdir -p {dest} && tar -xf - -C {dest}", stdin=data, user=user)
        if rc:
            raise RuntimeError(o)

    def rm(self):
        sh(["docker", "rm", "-f", self.name])


def docker_job(script, binds=(), image=IMAGE, timeout=1800, net="none", user="1000:1000", workdir="/work"):
    """One-shot container for task BUILD steps (compile references, generate cases). binds: [(host, ctr, 'ro'|'rw')].
    Returns (rc, combined_output). Never used for agent runs."""
    cmd = ["docker", "run", "--rm", "--network", net, "--memory", "8g", "--cpus", "4", "--user", user, "-w", workdir]
    for h, c, mode in binds:
        cmd += ["-v", f"{h}:{c}:{mode}"]
    cmd += [image, "bash", "-c", script]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    return p.returncode, (p.stdout + p.stderr).decode(errors="replace")


def task_data(tid):
    """Fresh-or-existing LDATA/<id> with the standard subdirs."""
    base = f"{LDATA}/{tid}"
    for s in ("agent", "target", "hidden", "ref_sub", "ref_partial"):
        os.makedirs(f"{base}/{s}", exist_ok=True)
    return base
