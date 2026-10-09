#!/usr/bin/env python3
"""Parent-owned serving screen, stop and health around a prepared C1 profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import math
import c1_serve_controller_combined_v11 as c1
import run_source_upload_oracle_full_v2 as upload_v2
import source_page_watchdog_v3 as watchdog

ROOT = Path(__file__).resolve().parents[2]
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'



CONTROLLER_SHA = 'bb629910af8296c4016affa79be023af3d9c126a1c58e523363452b5a6931ade'
UPLOAD_RUNNER_SHA = '05f7ff786922916d83589657e0e3e20ea24198b6e84b3faa84d9611592bd32ec'
WATCHDOG_SHA = '45a793c86c3af0e488a28e77648126b3a59fb9d318b6ba224fb1685c725dbcb1'


def source_pins():
    assert c1.sha(Path(c1.__file__)) == CONTROLLER_SHA, 'Strict C1V11 controller changed'
    assert c1.sha(Path(upload_v2.__file__)) == UPLOAD_RUNNER_SHA, 'Fresh full4/page preservation helper changed'
    assert c1.sha(Path(watchdog.__file__)) == WATCHDOG_SHA, 'Both-page watchdog changed'


def finalizable(result, identity):
    return (result.get('owned_terminal') is True and result.get('post_health_passed') is True and
            result.get('launch_supervisor_exit_code') == 0 and identity.get('passed') is True and
            len(identity.get('rows', [])) == 4 and not any(key.endswith('error') for key in result))


def validate_final_source_proof(directory, prepared, candidate_final=None):
    """Metadata/source-content checks; no new model payload read or page probe."""
    directory = Path(directory).resolve(); final = candidate_final if candidate_final is not None else c1.read(directory / 'qualification.json')
    assert final.get('teardown_passed') is True and final.get('post_health_passed') is True and final.get('passed') is True and final.get('c1_parent_generation') == 11 and final.get('post_full4_source_qualified') is True, 'Parent strengthened C1 source qualification absent; legacy controller PASS refused'
    generation = prepared.get('combined_generation', {})
    assert generation.get('current_PLE_host_observer_source33') is True, 'Exact source33 host observer compiled source required'
    assert generation.get('semantic_current_PLE_gather_fix32') is True, 'Corrected current PLE model source32 proof required'
    assert generation.get('plan_sha256') == c1.COMBINED_PLAN_SHA and generation.get('source_count') == 63 and len(generation.get('header_payloads', [])) == 27 and len(generation.get('sdk_targets', [])) == 8 and generation.get('runtime_python_sources') == c1.PYTHON_SOURCE_SHA, 'Final source proof requires actual integrated63/27/8/six preparation'
    binding = final['c1_source_identity_proof']; path = directory / 'c1-source-identity-proof-v11.json'
    assert binding['path'] == str(path) and c1.sha(path) == binding['sha256'], 'Final C1 source proof changed/moved'
    proof = c1.read(path)
    assert proof['combined_generation'] == generation, 'Source proof integrated SDK source association differs'
    assert proof['passed'] is True and proof['parent_controller_sha256'] == final['c1_parent_controller_sha256'] == c1.sha(Path(__file__)) and proof['controller_sha256'] == CONTROLLER_SHA and proof['watchdog_sha256'] == WATCHDOG_SHA, 'C1 source-proof generation differs'
    assert proof['prepared_sha256'] == c1.sha(directory / 'prepared.json') and proof['engine_receipt_sha256'] == prepared['engine_receipt_sha256'] == final['engine_receipt_sha256'], 'C1 source proof not matched to actual preparation/new SDK'
    assert proof['upload_v2_post_identity_sha256'] == prepared['upload_lifecycle']['post_model_identity_sha256'] and prepared['upload_lifecycle']['post_full4_source_qualified'] is True, 'C1 source proof upload-v2 association differs'
    identity_path = directory / 'c1-post-model-identity-v11.json'
    assert proof['identity']['path'] == str(identity_path) and proof['identity']['sha256'] == c1.sha(identity_path), 'C1 final full4 identity changed'
    identity = c1.read(identity_path)
    epochs = [identity.get('started'),identity.get('finished'),proof.get('terminal_epoch'),proof.get('post_health_finished_epoch')]
    assert all(type(epoch) in (int,float) and math.isfinite(epoch) and epoch > 0 for epoch in epochs), 'C1 temporal proof invalid'
    assert proof['between_polls_identity_observed'] is False and proof['page_guard_calls'] > 0, 'C1 page polling scope false'
    assert identity['passed'] is True and len(identity['rows']) == 4 and identity['started'] >= max(proof['terminal_epoch'],proof['post_health_finished_epoch']) and identity['finished'] >= identity['started'], 'C1 final source scan incomplete/stale chronology'
    assert identity['after_terminal_and_post_health_epoch'] == max(proof['terminal_epoch'],proof['post_health_finished_epoch']), 'C1 final scan terminal/health bound differs'
    lock_path = ROOT / 'strata/flash-next/model-lock.json'; lock = c1.read(lock_path)
    expected = {str(ROOT / lock['destination'] / row['path']):row for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')}
    assert identity['lock_sha256'] == c1.sha(lock_path) and identity['model_revision'] == lock['revision'] and len(expected) == 4 and {row['path'] for row in identity['rows']} == set(expected), 'C1 final source roster/lock differs'
    for row in identity['rows']:
        want = expected[row['path']]
        assert row['passed'] is True and row['sha256'] == row['expected_sha256'] == want['sha256'] and row['bytes'] == want['size'] and row['stat_before'] == row['stat_after'] == c1.stat_signature(Path(row['path'])), 'C1 final publisher hash proof differs'
    health_path = directory / 'post-health.json'; health = c1.read(health_path)
    assert proof['post_health_sha256'] == final['post_health_sha256'] == c1.sha(health_path) and health['passed'] is True and health['finished_epoch'] == proof['post_health_finished_epoch'], 'C1 final actual health proof differs'
    assert len(health['files']) == 2, 'C1 strict+compiled-pair health artifacts required'
    for artifact in health['files']:assert c1.sha(artifact['path']) == artifact['sha256'], 'C1 health artifact changed'
    assert final['controller_qualification_sha256'] == c1.sha(directory / 'qualification-controller-v11.json'), 'C1 original controller result changed'
    third = next(row for row in identity['rows'] if '00003-of-00004' in row['path'])
    for name in ('known_pages_before_hash','known_pages_after_hash'):
        guard = proof[name]
        assert guard['passed'] is True and guard['path'] == third['path'] and guard['stat_before'] == guard['stat_after'] == third['stat_after'], 'C1 both-page path/stat proof differs'
        assert len(guard['rows']) == 2 and [(row['offset'],row['expected_sha256']) for row in guard['rows']] == list(watchdog.KNOWN_PAGES) and all(row['passed'] is True and row['bytes'] == 4096 and row['sha256'] == row['expected_sha256'] for row in guard['rows']), 'C1 exact BOTH page views required'
    assert proof['known_pages_before_hash']['epoch'] <= identity['started'] and proof['known_pages_after_hash']['epoch'] >= identity['finished'], 'C1 page guards must bracket full4 scan'
    if candidate_final is None:
        parent = c1.read(directory / 'parent-qualification.json')
        assert parent.get('passed') is True and parent.get('parent_generation') == 11 and parent.get('parent_controller_sha256') == proof['parent_controller_sha256'] and parent.get('owned_terminal') is True and parent.get('post_health_passed') is True and parent.get('launch_supervisor_exit_code') == 0 and not any(key.endswith('error') for key in parent), 'Completed C1 parent lifecycle/source qualification absent'
        assert parent['qualification_sha256'] == c1.sha(directory / 'qualification.json') and parent['source_identity_proof_sha256'] == binding['sha256'] and parent['engine_receipt_sha256'] == prepared['engine_receipt_sha256'], 'C1 parent/final/source-proof crossbinding differs'
        assert parent['post_model_identity']['sha256'] == proof['identity']['sha256'] and parent['post_model_identity']['passed'] is True and parent['owned_terminal_epoch'] == proof['terminal_epoch'] and parent['post_health_finished_epoch'] == proof['post_health_finished_epoch'] and parent['finished_epoch'] >= proof['known_pages_after_hash']['epoch'], 'C1 parent post-source chronology differs'
    return proof


def record_supervisor_exit(out, result, code):
    receipt = {'schema': 1, 'return_code': code, 'observed_epoch': time.time(),
        'parent_pid': os.getpid(), 'passed': type(code) is int and code == 0}
    (out / 'launch-supervisor-exit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    result['launch_supervisor_exit_code'] = code
    if not receipt['passed']:
        result['supervisor_error'] = 'Launch supervisor exited unexpectedly: ' + str(code)
    return receipt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--one-card-receipt', type=Path, help='Matched completed one-card qualification for two-card launch')
    p.add_argument('--leased', action='store_true')
    a = p.parse_args()
    if not a.leased:
        os.execv(str(ROOT / 'bin/gpu-run'), ['gpu-run', sys.executable, __file__, *sys.argv[1:], '--leased'])
    for fd, card in [(8, 0), (9, 1)]:
        assert os.path.samefile('/proc/self/fd/' + str(fd), '/mnt/vm_8tb/b70/gpu.lock.' + str(card))
    out = a.prepared.resolve()
    ctrl = ROOT / 'strata/flash-next/c1_serve_controller_combined_v11.py'
    source_pins()
    assert not (out / 'parent-qualification.json').exists() and not (out / 'qualification.json').exists(), 'Preserve completed C1 evidence; use new prepared directory'
    raw_prepared = c1.read(out / 'prepared.json')
    lock_path = ROOT / 'strata/flash-next/model-lock.json'; lock = c1.read(lock_path)
    shards = [ROOT / lock['destination'] / row['path'] for row in lock['files'] if row['path'].startswith('UD-Q4_K_XL/')]
    assert [row['path'] for row in raw_prepared['model_shards']] == [str(path) for path in shards], 'Prepared selected source paths differ'
    c1.metadata_admission_gate(raw_prepared)
    upload_v2.guarded_pages(shards, out, 'c1-before-prepared-validation')
    prepared = c1.validate_prepared(out)
    started = time.time()
    result = dict(scope='Bounded C1/API screen with new final buffered source identity, not full fidelity/latency/shelf qualification', passed=False, parent_generation=11, parent_controller_sha256=c1.sha(Path(__file__)), controller_sha256=CONTROLLER_SHA, watchdog_sha256=WATCHDOG_SHA, engine_receipt_sha256=prepared['engine_receipt_sha256'], prepared_sha256=c1.sha(out / 'prepared.json'), source_guard_poll_seconds=2, source_guard_between_polls_unobserved=True)
    guard_calls = 0
    def page_guard(label):
        nonlocal guard_calls
        guard_calls += 1
        return upload_v2.guarded_pages(shards, out, 'c1-' + label + '-' + str(guard_calls))
    child = None
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    for s in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(s, stop)

    def run(cmd, name, timeout=210, check=True):
        with (out / name).open('w') as f:
            if name == 'screen-client.log':
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + timeout
                    while process.poll() is None:
                        page_guard('screen-poll')
                        if stopped or time.monotonic() > deadline:raise RuntimeError('C1 screen interrupted/deadline')
                        time.sleep(2)
                    page_guard('screen-terminal')
                    done = process
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=30)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
            else:
                done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        if check and done.returncode:
            raise RuntimeError(name + ' failed ' + str(done.returncode))

    def health(stage):
        files = []
        for name, cmd, timeout in [
            (stage + '-health.log', [str(ROOT / 'vllm/int4/diagnostics/xpu_health_strict.sh'), '--img', HEALTH], 180),
            (stage + '-collective.log', [str(ROOT / 'bin/xpu-collective-health'), '--img', HEALTH, '--p2p', '0', '--timeout', '180'], 210)]:
            run(cmd, name, timeout)
            files.append(dict(path=str(out / name), sha256=hashlib.sha256((out / name).read_bytes()).hexdigest(), command=cmd))
        receipt = out / (stage + '-health.json')
        receipt.write_text(json.dumps(dict(passed=True, cards=[0, 1], files=files, finished_epoch=time.time()), indent=2) + '\n')
        result[stage + '_health_passed'] = True
        result[stage + '_health_finished_epoch'] = c1.read(receipt)['finished_epoch']
        return receipt

    def faults():
        done = subprocess.run(['journalctl', '-k', '--since', '@' + str(int(started)), '--no-pager'], capture_output=True, timeout=20)
        (out / 'qualification-kernel-journal.log').write_bytes(done.stdout + done.stderr)
        if done.returncode or re.search(r'Fault response|CAT error|GPU HANG|GPU coredump|Job .* timed out|GT.* reset failed', done.stdout.decode(errors='replace'), re.I):
            raise RuntimeError('GPU fault signature or unavailable journal')

    try:
        result['known_pages_admission'] = page_guard('admission')
        pre = health('pre')
        with (out / 'launch-supervisor.log').open('w') as log:
            launch_cmd = [sys.executable, str(ctrl), 'launch', '--prepared', str(out), '--pre-health', str(pre),
                          '--ready-deadline', '600', '--max-runtime', '1000']
            if a.one_card_receipt:
                launch_cmd += ['--one-card-receipt', str(a.one_card_receipt.resolve())]
            child = subprocess.Popen(launch_cmd, stdout=log, stderr=subprocess.STDOUT, pass_fds=(8, 9))
            deadline = time.monotonic() + 620
            while True:
                page_guard('readiness-poll')
                faults()
                if stopped or time.monotonic() > deadline:
                    raise RuntimeError('Readiness deadline or interruption')
                launch = out / 'launch.json'
                if launch.exists() and json.loads(launch.read_text()).get('ready'):
                    break
                if child.poll() is not None:
                    raise RuntimeError('Launch supervisor exited ' + str(child.returncode))
                time.sleep(2)
            run([sys.executable, str(ctrl), 'screen', '--prepared', str(out)], 'screen-client.log', 600)
    except Exception as e:
        result['error'] = str(e)
    finally:
        if (out / 'launch.json').exists() and not (out / 'stop.json').exists():
            try:
                run([sys.executable, str(ctrl), 'stop', '--prepared', str(out)], 'stop-client.log', 120)
            except Exception as e:
                result['stop_error'] = str(e)
        if child is not None:
            try:
                child.wait(timeout=120)
                record_supervisor_exit(out, result, child.returncode)
            except subprocess.TimeoutExpired:
                # Keep lease while the owned serving supervisor/container is live.
                child.terminate()
                child.wait()
                record_supervisor_exit(out, result, child.returncode)
                result['supervisor_error'] = 'Supervisor needed termination'
        if (out / 'launch.json').exists():
            name = json.loads((out / 'launch.json').read_text())['container']
            while True:
                try:
                    ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], timeout=20)
                    if not ids.strip():
                        break
                    obj = json.loads(subprocess.check_output(['docker', 'inspect', name], timeout=20))[0]
                    assert obj['Config']['Labels'].get('b70.c1.prepared') == hashlib.sha256((out / 'prepared.json').read_bytes()).hexdigest()
                    result['cleanup_error'] = 'Owned container needed extra removal'
                    run(['docker', 'rm', '-f', name], 'forced-container-remove.log', 60)
                except Exception as e:
                    result['cleanup_error'] = str(e)
                    time.sleep(2)
        result['owned_terminal'] = not any(key in result for key in ['stop_error','supervisor_error','cleanup_error']) and child is not None and child.poll() == 0
        result['owned_terminal_epoch'] = time.time()
        try: result['known_pages_after_terminal'] = page_guard('after-owned-terminal')
        except Exception as error: result['terminal_source_error'] = str(error)
        try:
            post = health('post')
            faults()
            if ((out / 'screen.json').exists() and (out / 'stop.json').exists() and result.get('owned_terminal') and
                    not any(key.endswith('error') for key in result)):
                before = page_guard('post-hash-before')
                boundary = max(result['owned_terminal_epoch'],result['post_health_finished_epoch'])
                identity_path = out / 'c1-post-model-identity-v11.json'
                identity = upload_v2.full_buffered_identity(lock_path,lock,shards,identity_path,boundary)
                result['post_model_identity'] = {'path':str(identity_path),'sha256':c1.sha(identity_path),'passed':identity['passed'],'started':identity['started'],'finished':identity['finished']}
                after = page_guard('post-hash-after')
                assert finalizable(result,identity), 'Final C1 requires fresh complete4 publisher hashes after terminal/post-health'
                proof_path = out / 'c1-source-identity-proof-v11.json'
                proof = {'schema':11,'combined_generation':prepared['combined_generation'],'passed':True,'parent_controller_sha256':c1.sha(Path(__file__)),'controller_sha256':CONTROLLER_SHA,'watchdog_sha256':WATCHDOG_SHA,
                         'prepared_sha256':c1.sha(out/'prepared.json'),'engine_receipt_sha256':prepared['engine_receipt_sha256'],
                         'upload_v2_post_identity_sha256':prepared['upload_lifecycle']['post_model_identity_sha256'],
                         'terminal_epoch':result['owned_terminal_epoch'],'post_health_finished_epoch':result['post_health_finished_epoch'],
                         'post_health_sha256':c1.sha(post),'identity':{'path':str(identity_path),'sha256':c1.sha(identity_path)},
                         'known_pages_before_hash':before,'known_pages_after_hash':after,'page_guard_calls':guard_calls,
                         'between_polls_identity_observed':False,'scope':'New complete4 buffered publisher identity after owned terminal/post-health; no continuous source/disk-health or full math proof'}
                c1.write(proof_path,proof)
                run([sys.executable, str(ctrl), 'finalize', '--prepared', str(out), '--post-health', str(post)], 'finalize.log', 60)
                raw_path = out / 'qualification-controller-v11.json'
                (out/'qualification.json').rename(raw_path)
                final = c1.read(raw_path)
                final.update(c1_parent_generation=11,c1_parent_controller_sha256=c1.sha(Path(__file__)),post_full4_source_qualified=True,
                             c1_source_identity_proof={'path':str(proof_path),'sha256':c1.sha(proof_path)},controller_qualification_sha256=c1.sha(raw_path))
                validate_final_source_proof(out,prepared,candidate_final=final)
                c1.write(out/'qualification.json',final)
                result.update(passed=True,post_full4_source_qualified=True,source_identity_proof_sha256=c1.sha(proof_path),qualification_sha256=c1.sha(out/'qualification.json'))
            else:
                result['post_identity_scope']='UNOBSERVED: incomplete/failed C1 terminal or screen gate; no dependent hash/final qualification'
        except Exception as e:
            result['post_error'] = str(e)
            try: result['post_error_source_pages_observation'] = page_guard('post-error-current-views')
            except Exception as error: result['post_error_source_pages_error'] = str(error)
        result['finished_epoch'] = time.time()
        result['page_guard_calls'] = guard_calls
        (out / 'parent-qualification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], receipt=str(out / 'parent-qualification.json'))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
