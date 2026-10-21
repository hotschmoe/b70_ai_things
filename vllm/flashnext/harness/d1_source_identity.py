#!/usr/bin/env python3
"""Snapshot/verify the dirty fork delta plus effective campaign sources/SOs."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

R = Path("/mnt/vm_8tb/github/b70_ai_things")
B = Path("/mnt/vm_8tb/b70/build/vllm-flashnext")
F = B / "vllm"


def git(*args):
    return subprocess.check_output(["git", "-C", str(F), *args])


def digest(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "verify"))
    parser.add_argument("manifest")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if args.mode == "snapshot":
        dirty = git("diff", "--name-only", "HEAD").decode().splitlines()
        dirty += git("ls-files", "--others", "--exclude-standard").decode().splitlines()
        paths = {F / name for name in dirty if (F / name).is_file()}
        campaign = R / "vllm/flashnext"
        extensions = {".py", ".cpp", ".h", ".hpp", ".sh", ".toml", ".env"}
        for directory in (campaign / "plugins", campaign / "harness"):
            paths.update(p for p in directory.rglob("*") if p.is_file()
                         and p.suffix in extensions)
        paths.update(p for p in (campaign / "localmaxxing").rglob("*")
                     if p.is_file() and p.suffix in {".py", ".txt", ".json"})
        for directory in (B / "c7b-fa-overlay/vllm_xpu_kernels",
                          B / "d1/fa-overlay/vllm_xpu_kernels"):
            for name in ("flash_attn_interface.py", "_vllm_fa2_C.abi3.so",
                         "libattn_kernels_xe_2.so"):
                paths.add(directory / name)
        paths.add(B / "d1/fa-overlay/d1_fa_plan.py")
        paths.add(R / "evals/configs/models.yaml")
        paths.add(campaign / "plugins/b70_o4_xpu/b70_o4_xpu/_C.so")
        paths.add(campaign / "plugins/b70_gguf_xpu/b70_gguf_xpu/libdequant_cpu.so")
        envs = {}
        for name in ("o4_recipe.env", "f1_prefill_delta.env"):
            for line in (campaign / "harness" / name).read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    envs[key] = value.strip()
        paths.update(Path(value) for key, value in envs.items()
                     if key.endswith("_SO") and Path(value).is_file())
        data = dict(head=git("rev-parse", "HEAD").decode().strip(),
                    diff_sha256=hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
                    untracked=sorted(set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())),
                    files={str(p): digest(p) for p in sorted(paths)})
        manifest.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
        print(json.dumps(dict(event="snapshot", files=len(paths), path=str(manifest))))
    else:
        data = json.loads(manifest.read_text())
        bad = [name for name, sha in data["files"].items()
               if not Path(name).is_file() or digest(name) != sha]
        if git("rev-parse", "HEAD").decode().strip() != data["head"]:
            bad.append("fork HEAD")
        if hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest() != data["diff_sha256"]:
            bad.append("fork tracked diff")
        if sorted(set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())) != data["untracked"]:
            bad.append("fork untracked list")
        print(json.dumps(dict(event="verify", bad=bad, files=len(data["files"]))))
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
