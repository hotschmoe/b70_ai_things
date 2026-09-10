#!/usr/bin/env python3
"""Owned, bounded new-image pair health; invoke with bin/gpu-run (both cards)."""
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO = pathlib.Path('/mnt/vm_8tb/github/b70_ai_things')
ROOT = pathlib.Path('/mnt/vm_8tb/b70/results/bang_isolation_20260910/sglang-main-triton-dense-preflight')
IMAGES = [('sglang-main-triton-dense', 'sha256:bdc51c5f083fdcbacd59a74bfeb8389fa6d62c8478e066ed5c62f9bffcfaa4cf')]
BASELINE = 'sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad'
DOCKER = shutil.which('docker')
LABEL = 'b70.preflight.owner'
CLEANUP_PENDING = False


def lease_identity():
    lock = os.environ.get('B70_GPU_LOCK', '/mnt/vm_8tb/b70/gpu.lock')
    rows = []
    for card, fd in ((0, 8), (1, 9)):
        expected = pathlib.Path(lock + '.' + str(card))
        actual = pathlib.Path('/proc/self/fd/' + str(fd)).resolve(strict=True)
        assert actual == expected.resolve(), (fd, actual, expected)
        assert os.fstat(fd).st_ino == expected.stat().st_ino
        rows.append(dict(card=card, fd=fd, path=str(actual),
                         owner=pathlib.Path(str(expected) + '.owner').read_text().strip()))
    return rows


def capture(command, timeout=30):
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout)


def cleanup_owned(owner, out):
    """Only delete our labeled containers; daemon uncertainty blocks reset."""
    records = []
    try:
        found = capture([DOCKER, 'ps', '-aq', '--filter', 'label=' + LABEL + '=' + owner])
        if found.returncode:
            raise RuntimeError('cannot list owned containers: ' + found.stderr)
        for cid in found.stdout.split():
            inspected = capture([DOCKER, 'inspect', cid])
            if inspected.returncode:
                # --rm can race this inspect; final listing must prove absence.
                records.append(dict(id=cid, inspect_rc=inspected.returncode))
                continue
            data = json.loads(inspected.stdout)[0]
            if data['Config'].get('Labels', {}).get(LABEL) != owner:
                raise RuntimeError('ownership mismatch: ' + cid)
            removed = capture([DOCKER, 'rm', '-f', cid], timeout=45)
            records.append(dict(id=cid, name=data['Name'], image=data['Image'],
                                remove_rc=removed.returncode, stderr=removed.stderr))
        remaining = capture([DOCKER, 'ps', '-aq', '--filter', 'label=' + LABEL + '=' + owner])
        if remaining.returncode or remaining.stdout.strip():
            raise RuntimeError('owned container removal unverified: ' + remaining.stdout + remaining.stderr)
        records.append(dict(verified_absent=True))
        ok = True
    except Exception as exc:
        records.append(dict(error=ascii(exc), verified_absent=False))
        ok = False
    out.write_text(json.dumps(records, indent=2) + '\n')
    return ok


def run(command, out, timeout, env=None, owned=False):
    global CLEANUP_PENDING
    owner = uuid.uuid4().hex if owned else None
    runenv = dict(os.environ if env is None else env)
    if owned:
        runenv['PATH'] = str(ROOT / 'docker-wrapper') + os.pathsep + runenv.get('PATH', '')
        runenv['B70_PREFLIGHT_OWNER'] = owner
    process = None
    rc = 1
    try:
        with out.open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=runenv, start_new_session=True)
            try:
                rc = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rc = 124
    finally:
        CLEANUP_PENDING = owned
        # An outer timeout must terminate the whole launcher group, not only bash.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            # A bash leader can exit while docker/timeout descendants linger.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        clean = cleanup_owned(owner, out.with_suffix('.cleanup.json')) if owned else True
        if not clean:
            rc = rc or 1
            (ROOT / 'CLEANUP_BLOCKED.json').write_text(json.dumps(
                dict(command=command, owner=owner, log=str(out),
                     verdict='Retaining both leases. No reset or next GPU probe until removal is verified.')) + '\n')
            while not cleanup_owned(owner, out.with_suffix('.cleanup.json')):
                time.sleep(5)
            (ROOT / 'CLEANUP_RESOLVED').touch()
        CLEANUP_PENDING = False
        out.with_suffix('.rc').write_text(str(rc) + '\n')
    return rc


def main():
    probe = ROOT.parent / 'health-repair/xpu-health'
    probe_sha = hashlib.sha256(probe.read_bytes()).hexdigest()
    assert probe_sha == '2a1488deeda1ede12dd1494fe3990a6104a584dadb6a7a1ba33815624911e9ba'
    leases = lease_identity()
    assert DOCKER, 'docker unavailable'
    ROOT.mkdir(exist_ok=False)
    (ROOT / 'health-probe-identity.json').write_text(json.dumps(dict(path=str(probe), sha256=probe_sha)) + '\n')
    (ROOT / 'lease-identity.json').write_text(json.dumps(leases, indent=2) + '\n')
    wrapperdir = ROOT / 'docker-wrapper'
    wrapperdir.mkdir()
    wrapper = wrapperdir / 'docker'
    wrapper.write_text('#!' + sys.executable + '\nimport os, sys\na = sys.argv[1:]\n'
        'if a and a[0] == "run":\n'
        '    a[1:1] = ["--label", ' + repr(LABEL + '=') + ' + os.environ["B70_PREFLIGHT_OWNER"]]\n'
        'os.execv(' + repr(DOCKER) + ', ["docker", *a])\n')
    wrapper.chmod(0o755)
    for label, image in IMAGES:
        out = ROOT / label
        out.mkdir()
        inspected = capture([DOCKER, 'image', 'inspect', image])
        if inspected.returncode:
            raise RuntimeError('image inspect failed: ' + inspected.stderr)
        data = json.loads(inspected.stdout)[0]
        if image.startswith('sha256:'):
            assert data['Id'] == image
        else:
            assert image in data.get('RepoDigests', [])
        (out / 'image-inspect.json').write_text(inspected.stdout)
        env = dict(os.environ, XPU_COLLECTIVE_HEALTH_CACHE=str(out / 'compiled-cache'))
        (out / 'identity.json').write_text(json.dumps(dict(
            requested_image=image, resolved_image=data['Id'],
            scope='Per-card and compiled pair health only; no model qualification.')) + '\n')
        for tool, extra, timeout in [
                ('xpu-health', [], 200),
                ('xpu-collective-health', ['--ccl-root', '/opt/venv', '--p2p', '0', '--timeout', '180'], 240)]:
            rc = run([str(ROOT.parent / 'health-repair/xpu-health' if tool == 'xpu-health' else REPO / 'bin' / tool), '--img', image, *extra],
                     out / (tool + '.log'), timeout, env, owned=True)
            if rc:
                # run() only returns after verified owned-container removal.
                recovery = run(['env', 'B70_XE_RESET_UNDER_LEASE=1', str(REPO / 'bin/xe-reset'),
                                '--method', 'rebind', '--no-probe'], out / 'recovery.log', 240)
                if recovery:
                    (out / 'FAIL.json').write_text(json.dumps(dict(tool=tool, rc=rc,
                        recovery=recovery, posthealth='skipped: reset failed')) + '\n')
                    return 1
                recovery_env = dict(env, XPU_COLLECTIVE_HEALTH_CACHE=str(out / 'baseline-recovery-compiled-cache'))
                health = run([str(ROOT.parent / 'health-repair/xpu-health'), '--img', BASELINE],
                             out / 'post-recovery-health.log', 200, recovery_env, owned=True)
                collective = 'skipped: per-card health failed'
                if not health:
                    collective = run([str(REPO / 'bin/xpu-collective-health'), '--img', BASELINE,
                                      '--p2p', '0', '--timeout', '180'],
                                     out / 'post-recovery-collective.log', 240, recovery_env, owned=True)
                (out / 'FAIL.json').write_text(json.dumps(dict(tool=tool, rc=rc,
                    recovery=recovery, posthealth=health, postcollective=collective)) + '\n')
                return 1
        (out / 'PASS').write_text('per-card and compiled pair health passed; no model qualification\n')
    (ROOT / 'PASS').write_text('immutable current-main image passed health preflight only\n')
    return 0


if __name__ == '__main__':
    def interrupted(signum, frame):
        if CLEANUP_PENDING:
            return  # Keep both leases until the owned GPU containers are absent.
        raise KeyboardInterrupt('preflight interrupted by signal ' + str(signum))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, interrupted)
    raise SystemExit(main())
