#!/usr/bin/env python3
"""Run the pinned author's strict suite on four independently leased lifecycles."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from prepare_replica import IMAGE, SOURCE_COMMIT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE_COMMIT
    args.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    bench = args.source / 'repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/bench-w8a16-mtp1-strict.sh'
    compare = args.source / 'scripts/compare-strict-attempt-outputs.py'
    def parity(left, right):
        output = args.out / ('compare-' + left + '-vs-' + right + '.json')
        result = subprocess.run(['python3', str(compare), str(args.out / left / 'strict'),
                                 str(args.out / right / 'strict'), '--output', str(output)],
                                capture_output=True, text=True)
        (output.with_suffix('.log')).write_text(result.stdout + result.stderr)
        row = json.loads(output.read_text())['comparison']
        if row['exact_prompts'] != 12 or row['total_prompts'] != 12:
            raise RuntimeError('strict token parity failed: ' + output.name)
    for name, mtp in [('mtp0-a', 0), ('mtp0-b', 0), ('mtp4-a', 4), ('mtp4-b', 4)]:
        out = args.out / name
        model = f'qwen3.8-27b-AutoRound-INT4-W4A16-g128-r276-mtp{mtp}-fp16kv-strict'
        command = ['python3', str(repo / 'vllm/fp8/kv_campaign_server.py'),
                   '--preservation', str(args.config), '--out', str(out),
                   '--image', IMAGE, '--served-model', model, '--mtp', str(mtp),
                   '--memory-gib', '12', '--memory-swap-gib', '16', '--health-p2p-check']
        with (args.out / (name + '-runner.log')).open('w') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 660
                while not (out / 'jobs').is_dir():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('runner failed before queue creation: ' + name)
                    time.sleep(1)
                job = ['env', 'OUT_DIR=' + str(out / 'strict'), 'BASE_URL=http://127.0.0.1:18125',
                       'MODEL_NAME=' + model, 'PROFILE_LABEL=r276-int4-exact-replica',
                       'ATTEMPT_LABEL=' + name, 'bash', str(bench)]
                (out / 'jobs/01-strict.json').write_text(json.dumps({'command': job, 'timeout': 1800}) + '\n')
                (out / 'jobs/99-stop.json').write_text(json.dumps({'command': ['touch', str(out / 'STOP')], 'timeout': 10}) + '\n')
                rc = server.wait(timeout=3600)
            finally:
                if server.poll() is None:
                    (out / 'STOP').touch()
                    server.wait(timeout=1500)
        if rc or (out / 'exit.rc').read_text().strip() != '0':
            raise RuntimeError('lifecycle failed: ' + name)
        if (out / 'jobs/01-strict.done').read_text().strip() != '0':
            raise RuntimeError('strict quality/workload gate failed: ' + name)
        if name == 'mtp0-b':
            parity('mtp0-a', 'mtp0-b')
        if mtp == 4:
            parity(name, 'mtp0-a')
        if name == 'mtp4-b':
            parity('mtp4-a', 'mtp4-b')
        print('QUALIFIED STRICT LIFECYCLE ' + name, flush=True)
    (args.out / 'PASSED').touch()


if __name__ == '__main__':
    main()
