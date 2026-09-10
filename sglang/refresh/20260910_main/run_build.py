#!/usr/bin/env python3
"""CPU-only pinned SGLang build, bounded without exposing device nodes."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import shutil

HERE = Path(__file__).resolve().parent
RAW = Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-refresh')
IMAGE = 'intel/sglang-dev@sha256:1a13d3d2b0adc63422a643eb5c59524842577d99ca3935b4b8c2c432e739e127'
NAME = 'b70-sglang-main-cpu-build-20260910'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='build')
    parser.add_argument('--profile', choices=['full', 'dense', 'triton-dense'], default='full')
    parser.add_argument('--cpus', type=int, default=4)
    parser.add_argument('--memory-gib', type=int, default=24)
    parser.add_argument('--resume-work', type=Path)
    args = parser.parse_args()
    if not 1 <= args.cpus <= 16 or not 4 <= args.memory_gib <= 64:
        parser.error('build resource limit outside reviewed bounds')
    if not args.attempt.replace('-', '').replace('_', '').isalnum():
        parser.error('attempt must be an ASCII alphanumeric component')
    name = NAME + '-' + args.attempt
    out = RAW / args.attempt; out.mkdir(exist_ok=False)
    recipe = out / 'recipe-snapshot'; recipe.mkdir()
    for f in HERE.iterdir():
        if f.is_file():
            shutil.copy2(f, recipe / f.name)
    cmd = ['docker', 'run', '--name', name, '--cpus', str(args.cpus), '--memory', str(args.memory_gib) + 'g',
           '--memory-swap', str(args.memory_gib) + 'g', '--ulimit', 'core=0',
           '-v', str(RAW / 'sources') + ':/inputs:ro',
           '-v', str(recipe) + ':/recipe:ro', '-v', str(out) + ':/out',
           '-e', 'B70_BUILD_MOE=' + ('OFF' if args.profile in ('dense', 'triton-dense') else 'ON'),
           '-e', 'B70_BUILD_MLA=' + ('OFF' if args.profile in ('dense', 'triton-dense') else 'ON'),
           '-e', 'B70_BUILD_FMHA=' + ('OFF' if args.profile == 'triton-dense' else 'ON'),
           '-e', 'B70_BUILD_JOBS=' + str(args.cpus)]
    if args.resume_work:
        args.resume_work = args.resume_work.resolve()
        prior = json.loads((args.resume_work.parent / 'manifest.json').read_text())
        assert prior['profile'] == args.profile
        assert prior['sources'] == json.loads((RAW / 'sources.json').read_text())
        assert IMAGE in prior['command']
        cmd += ['-v', str(args.resume_work) + ':/out/work']
    cmd += ['--entrypoint', 'bash', IMAGE, '/recipe/build_inside.sh']
    manifest = dict(command=cmd, profile=args.profile, resume_work=str(args.resume_work),
                    sources=json.loads((RAW / 'sources.json').read_text()),
                    recipe_hashes={str(f.name):hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in HERE.iterdir() if f.is_file()},
                    host_kernel=subprocess.check_output(['uname','-r'],text=True).strip())
    (out / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    subprocess.run(['docker','image','inspect',IMAGE],stdout=(out/'base-image.json').open('w'),check=True)
    try:
        with (out / 'build.log').open('w') as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=5400).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        with (out / 'stop.log').open('w') as log:
            subprocess.run(['docker','stop','-t','20',name],stdout=log,stderr=subprocess.STDOUT,timeout=45)
    (out / 'exit.rc').write_text(str(rc)+'\n')
    # Retain the stopped owned container for inspect/commit after verification.
    return rc

if __name__ == '__main__':
    raise SystemExit(main())
