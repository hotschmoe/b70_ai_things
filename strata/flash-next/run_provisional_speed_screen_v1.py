#!/usr/bin/env python3
"""Diagnostic client with existing C140 owned supervisor and fresh source gates."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import c1_serve_controller_combined_v140_v3 as c1
import qualify_c1_serving_combined_v140_v3 as q

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'strata/flash-next/provisional-speed-screen-source-plan-v1.json'


def sources():
    plan = c1.read(PLAN)
    for row in plan['files']:
        assert c1.sha(ROOT/row['path']) == row['sha256'], 'Speed source closure changed: '+row['path']
    q.source_pins()
    return c1.sha(PLAN)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--one-card-receipt', type=Path)
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(ROOT/'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in ((8, 0), (9, 1)):
        assert os.path.samefile('/proc/self/fd/'+str(fd), '/mnt/vm_8tb/b70/gpu.lock.'+str(card))
    out = a.prepared.resolve()
    assert not (out/'speed-parent.json').exists() and not (out/'launch.json').exists()
    source_sha = sources()
    m = c1.validate_prepared(out)
    assert m['profile'] in ('one-card-segmented', 'two-card-segmented')
    lock_path = ROOT/'strata/flash-next/model-lock.json'
    lock = c1.read(lock_path)
    shards = [ROOT/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')]
    assert [r['path'] for r in m['model_shards']] == list(map(str, shards))
    report = {'schema': 1, 'scope': 'provisional speed diagnostic, not C140 qualification',
              'passed': False, 'started_epoch': time.time(), 'errors': [], 'commands': [],
              'source_sha256': c1.sha(Path(__file__)), 'prepared_sha256': c1.sha(out/'prepared.json'),
              'source_plan_sha256': source_sha,
              'correctness_qualified': False, 'shelf_qualified': False}
    (out/'speed-source-plan.snapshot.json').write_bytes(PLAN.read_bytes())
    child = None
    interrupted = False
    def interrupt(*args):
        nonlocal interrupted
        interrupted = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, interrupt)
    def pages(label):
        return q.upload_v2.guarded_pages(shards, out, 'speed-'+label)
    def run(command, name, timeout):
        row = {'command': command, 'started_epoch': time.time(), 'return_code': None, 'error': None}
        c1.write(out/(name+'.command.json'), command)
        try:
            with (out/(name+'.log')).open('w') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(8, 9))
                try:
                    deadline = time.monotonic()+timeout
                    poll = 0
                    while proc.poll() is None:
                        if interrupted or time.monotonic()>deadline:
                            raise RuntimeError('Interrupted/deadline '+name)
                        if name == 'speed-client':
                            poll += 1
                            pages('client-'+str(poll))
                        time.sleep(2)
                    row['return_code'] = proc.returncode
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            assert row['return_code'] == 0, name+' failed'
        except BaseException as e:
            row['error'] = type(e).__name__+': '+str(e)
            raise
        finally:
            row.update(finished_epoch=time.time(), path=str(out/(name+'.log')),
                       sha256=c1.sha(out/(name+'.log')),
                       command_file_sha256=c1.sha(out/(name+'.command.json')))
            report['commands'].append(row)
            c1.write(out/(name+'.receipt.json'), row)
        return row
    def health(stage):
        start = time.time()
        rows = [run([str(ROOT/'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', q.HEALTH], stage+'-strict', 180),
                run([str(ROOT/'bin/xpu-collective-health'), '--img', q.HEALTH, '--p2p', '0', '--timeout', '180'], stage+'-compiled', 210)]
        path = out/(stage+'-speed-health.json')
        c1.write(path, {'passed': True, 'cards': [0, 1], 'files': rows,
                        'started_epoch': start, 'finished_epoch': time.time(), 'health_image': q.HEALTH})
        return path
    def journal(stage):
        row = run(['journalctl', '-k', '--since', '@'+str(int(report['started_epoch'])), '--no-pager'], stage+'-speed-journal', 20)
        assert not re.search(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', Path(row['path']).read_text(), re.I)
    ctrl = ROOT/'strata/flash-next/c1_serve_controller_combined_v140_v3.py'
    try:
        pages('admission')
        pre = health('pre'); journal('pre')
        command = [sys.executable, str(ctrl), 'launch', '--prepared', str(out), '--pre-health', str(pre),
                   '--ready-deadline', '600', '--max-runtime', '7200']
        if a.one_card_receipt:
            command += ['--one-card-receipt', str(a.one_card_receipt.resolve())]
        c1.write(out/'speed-launch.command.json', command)
        with (out/'speed-launch.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, pass_fds=(8, 9))
        report['launch_started_epoch'] = time.time()
        deadline = time.monotonic()+620
        poll = 0
        while not (out/'launch.json').exists() or not c1.read(out/'launch.json').get('ready'):
            assert not interrupted and time.monotonic()<deadline and child.poll() is None
            poll += 1; pages('ready-'+str(poll)); time.sleep(2)
        run([sys.executable, str(ROOT/'strata/flash-next/provisional_speed_screen_v1.py'),
             '--prepared', str(out), '--output', str(out/'speed-client')], 'speed-client', 5000)
    except BaseException as e:
        report['errors'].append(type(e).__name__+': '+str(e))
    finally:
        # Signal requests do not cancel mandatory owned stop/post-health/source cleanup.
        interrupted = False
        try:
            if (out/'launch.json').exists():
                run([sys.executable, str(ctrl), 'stop', '--prepared', str(out)], 'speed-stop', 180)
            if child is not None:
                child.wait(timeout=180)
                terminal = c1.stop_receipt_checked(out)
                assert child.returncode == 0 and terminal['clean_exit'] is True and terminal['engine_clean_exit_proven'] is True, 'Normal original supervisor/stop required'
            report['terminal_epoch'] = time.time()
            pages('terminal')
        except BaseException as e:
            report['errors'].append('cleanup: '+type(e).__name__+': '+str(e))
            # Retain lease until original owned supervisor returns; never infer absence.
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait()
        post_finished = time.time()
        try:
            post = health('post'); journal('post')
            post_finished = c1.read(post)['finished_epoch']
            report['post_health_sha256'] = c1.sha(post)
        except BaseException as e:
            report['errors'].append('post-health: '+type(e).__name__+': '+str(e))
        try:
            pages('pre-full4')
            boundary = max(report.get('terminal_epoch', time.time()), post_finished)
            identity = q.upload_v2.full_buffered_identity(lock_path, lock, shards, out/'speed-post-full4.json', boundary)
            assert identity['passed'] is True
            pages('post-full4')
            report['post_full4_sha256'] = c1.sha(out/'speed-post-full4.json')
        except BaseException as e:
            report['errors'].append('post: '+type(e).__name__+': '+str(e))
        try:
            assert sources() == source_sha and c1.sha(out/'speed-source-plan.snapshot.json') == source_sha
            assert c1.sha(out/'prepared.json') == report['prepared_sha256']
        except BaseException as e:
            report['errors'].append('post-source: '+type(e).__name__+': '+str(e))
        report['passed'] = not report['errors'] and child is not None and child.returncode == 0
        report['finished_epoch'] = time.time()
        c1.write(out/'speed-parent.json', report)
    assert report['passed'], report['errors']


if __name__ == '__main__':
    main()
