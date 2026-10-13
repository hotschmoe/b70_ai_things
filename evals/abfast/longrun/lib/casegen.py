"""Case RECORDING helpers (run at task build time, inside the build container, against the reference).

  rec = Recorder(hidden_dir, ref_argv0)
  rec.add(ms, args, files={rel: bytes}, stdin=None, capture=[relpaths], cmp=None, modes=None, env=None)
Writes cases.jsonl-style list to hidden_dir/cases.json via rec.save().
"""
import json, os, shutil, tempfile
from mscore import run, put_blob, sha


class Recorder:
    def __init__(self, hidden, ref_argv0, timeout=20):
        self.hidden, self.ref, self.timeout, self.cases = hidden, list(ref_argv0), timeout, []

    def add(self, ms, args, files=None, stdin=None, capture=(), cmp=None, modes=None, env=None, name=None):
        files = files or {}
        d = tempfile.mkdtemp(prefix="rec")
        try:
            fh = {}
            for rel, data in files.items():
                p = os.path.join(d, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(data)
                os.chmod(p, (modes or {}).get(rel, 0o644))
                fh[rel] = put_blob(self.hidden, data)
            a = [x.replace("{DIR}", d) for x in args]
            e = dict(os.environ); e.update(env or {})
            rc, out, err = run(self.ref + a, stdin=stdin, timeout=self.timeout, cwd=d, env=e)
            if rc in (-9, -1):
                raise RuntimeError(f"reference failed on case {name or args}: {err[:200]!r}")
            outf = {}
            for rel in capture:
                p = os.path.join(d, rel)
                outf[rel] = put_blob(self.hidden, open(p, "rb").read()) if os.path.isfile(p) else None
            c = {"id": f"{ms}-{len(self.cases):04d}", "ms": ms, "args": args, "files": fh,
                 "stdin": put_blob(self.hidden, stdin) if stdin else None, "env": env, "modes": modes or None,
                 "expect": {"rc": rc, "stdout": put_blob(self.hidden, out), "stderr": put_blob(self.hidden, err),
                            "out_files": outf}}
            if cmp:
                c["cmp"] = cmp
            self.cases.append(c)
            return c
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def save(self, path=None):
        path = path or os.path.join(self.hidden, "cases.json")
        json.dump(self.cases, open(path, "w"))
        return len(self.cases)
