"""One-card OFF/ON expert-pair fixture with owned teardown and post-health."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = 'sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HEALTH = 'sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def require(ok, msg):
    if not ok:
        raise ValueError(msg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--build-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.fstat(9).st_ino == Path('/mnt/vm_8tb/b70/gpu.lock.1').stat().st_ino, 'Root card1 lease required')
    require(sha(args.build / 'receipt.json') == args.build_receipt_sha256, 'Exact completed compile required')
    compile_receipt = json.loads((args.build / 'receipt.json').read_bytes())
    require(compile_receipt['passed'] and compile_receipt['devices_exposed'] is False, 'NoDri compile must pass')
    binary = args.build / 'iq-pair-fixture'
    require(sha(binary) == compile_receipt['new_artifacts_sha256']['iq-pair-fixture'], 'Fresh fixture ELF changed')
    output = args.output
    require(output.is_absolute() and output.resolve() == output and not output.exists(), 'New canonical output required')
    output.mkdir()
    report = dict(schema=1, started_epoch=time.time(), passed=False, errors=[], arms={},
                  build_receipt_sha256=args.build_receipt_sha256, binary_sha256=sha(binary),
                  fixture_plan_sha256=compile_receipt['fixture_plan_sha256'], card=1,
                  expected_PCI='0000:44:00.0', synthetic_only=True, model_payload=False,
                  model_speed_qualified=False, original_fullmath_qualified=False)
    retirement = {'active': False, 'signal': None}
    def signal_handler(number, frame):
        retirement['signal'] = number
        if not retirement['active']:
            raise KeyboardInterrupt('Requested signal ' + str(number))
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    def retire(name, owner, arm):
        # Unknown Docker state must keep this process and its GPU lease alive.
        # Retry only exact owned identity; post-health begins after proven absence.
        retirement['active'] = True
        forced = False
        attempts = 0
        try:
            while True:
                attempts += 1
                try:
                    query = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=20)
                    if query.returncode != 0:
                        require('no such object: ' + name in query.stderr.decode().lower(), 'Docker state unknown')
                        return dict(exit_code=None, forced_cleanup=forced, container_removed=True,
                                    retirement_attempts=attempts, container_never_inspected=True)
                    row = json.loads(query.stdout)[0]
                    require(row['Config']['Labels'].get('b70.iqpair.owner') == owner and row['Config']['Image'] == RUNTIME,
                            'Owned container identity differs; retaining lease')
                    if row['State']['Running']:
                        forced = True
                        subprocess.run(['docker', 'stop', '--time', '30', name], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45)
                        continue
                    (output / (arm + '.terminal.json')).write_text(json.dumps(row, indent=2) + '\n', encoding='ascii')
                    subprocess.run(['docker', 'rm', name], check=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=20)
                    absence = subprocess.run(['docker', 'inspect', name], capture_output=True, timeout=20)
                    require(absence.returncode != 0 and 'no such object: ' + name in absence.stderr.decode().lower(),
                            'Exact owned container absence required')
                    return dict(exit_code=row['State']['ExitCode'], forced_cleanup=forced,
                                container_removed=True, retirement_attempts=attempts)
                except Exception as error:
                    report['errors'].append('Retirement attempt ' + str(attempts) + ': ' + str(error))
                    (output / (arm + '.retirement-pending.json')).write_text(
                        json.dumps(dict(attempts=attempts, error=str(error), lease_retained=True)) + '\n', encoding='ascii')
                    time.sleep(2)
        finally:
            retirement['active'] = False
    def run(command, label, timeout=180):
        (output / (label + '.command.json')).write_text(json.dumps(command, indent=2) + '\n', encoding='ascii')
        with (output / (label + '.stdout')).open('wb') as stdout, (output / (label + '.stderr')).open('wb') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
        require(result.returncode == 0, label + ' returned ' + str(result.returncode))
        return result
    try:
        run([str(ROOT / 'bin/xpu-health'), '--card', '1', '--img', HEALTH], 'pre-health')
        for flag, arm in [('0', 'OFF'), ('1', 'ON')]:
            name = 'b70-iq-pair-' + str(os.getpid()) + '-' + flag
            owner = str(os.getpid())
            command = ['docker', 'run', '--name', name, '--label', 'b70.iqpair.owner=' + owner,
                       '--network', 'none', '--cpus', '2', '--memory', '4g', '--memory-swap', '4g',
                       '--user', str(os.getuid()) + ':' + str(os.getgid()), '--device', '/dev/dri']
            for gid in sorted({Path('/dev/dri/card1').stat().st_gid, Path('/dev/dri/renderD129').stat().st_gid}):
                command += ['--group-add', str(gid)]
            command += ['-e', 'ZE_AFFINITY_MASK=1', '-e', 'ONEAPI_DEVICE_SELECTOR=level_zero:gpu',
                        '-e', 'B70_IQ_EXPECTED_PCI=0000:44:00.0', '-e', 'STRATA_EXPERT_PAIR_REUSE=' + flag,
                        '-e', 'SYCL_CACHE_PERSISTENT=0', '-v', str(args.build) + ':/harness:ro',
                        '-v', str(output) + ':/results', '--entrypoint', '/bin/bash', RUNTIME, '-lc',
                        'if [[ ${SETVARS_COMPLETED:-0} != 1 ]]; then source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; fi; '
                        'exec /harness/iq-pair-fixture ' + flag + ' --output /results/' + arm +
                        (' --compare /results/OFF' if flag == '1' else '')]
            forced = False
            try:
                run(command, arm, timeout=240)
                text = (output / (arm + '.stdout')).read_text(encoding='ascii')
                require(text.count('IQ_PAIR_CASE ') == 40 and text.count('IQ_PAIR_PHASE_LATENCY ') == 112,
                        'Complete fixture roster absent')
                require('IQ_PAIR_FIXTURE_PASS cases=40 changed_graphs=40 flag=' + flag in text,
                        'Fixture terminal PASS absent')
                if flag == '1':
                    require('original_OFF_bitwise=1 model_payload=0' in text, 'OFF/ON byte equality absent')
                report['arms'][arm] = dict(passed=True, cases=40, changed_graphs=40)
            finally:
                terminal = retire(name, owner, arm)
                report['arms'].setdefault(arm, {}).update(terminal)
                require(terminal['exit_code'] == 0 and not terminal['forced_cleanup'], 'Normal owned teardown required')
    except BaseException as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        try:
            run([str(ROOT / 'bin/xpu-health'), '--card', '1', '--img', HEALTH], 'post-health')
            report['post_health_passed'] = True
        except Exception as error:
            report['errors'].append('Post-health: ' + str(error))
        try:
            run(['journalctl', '-k', '--since', '@' + str(int(report['started_epoch'])), '--no-pager'], 'post-kernel')
            text = (output / 'post-kernel.stdout').read_text()
            require(not re.search(r'Fault response|Faulted Address|Engine reset|Timedout job|segfault|GPU HANG|CAT error', text, re.I),
                    'Fresh kernel fault signature')
            report['kernel_window_passed'] = True
        except Exception as error:
            report['errors'].append('Kernel: ' + str(error))
        report['binary_unchanged'] = sha(binary) == report['binary_sha256']
        report['passed'] = not report['errors'] and report['binary_unchanged'] and len(report['arms']) == 2
        report['finished_epoch'] = time.time()
        (output / 'controller.snapshot.py').write_bytes(Path(__file__).read_bytes())
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='ascii')
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
