#!/usr/bin/env python3
"""CPU-only checks of actual C1 prerequisite rejection and prepared launch gate."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('c1_controller', HERE / 'c1_serve_controller.py')
c1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c1)
UPLOAD = Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f06-20261009/source-upload-v2/receipt.json')
ORACLES = [Path('/mnt/vm_8tb/b70/build/strata-source-upload-oracle-v2-v8ogvlno/receipt.json'),
           Path('/mnt/vm_8tb/b70/build/strata-source-upload-oracle-v2-e3pk0ph8/receipt.json')]


def main():
    actual = c1.read(UPLOAD)
    oracle = next(path for path in ORACLES if c1.sha(path) == actual['oracle_receipt_sha256'])
    proof = c1.upload_gate(UPLOAD, oracle, c1.ENGINE / 'receipt.json', c1.INTAKE)
    assert proof['oracle_schema'] == 2
    rejected = []
    with tempfile.TemporaryDirectory(prefix='b70-c1-negative-') as directory:
        directory = Path(directory)
        mutations = [
            ('failed coupled run', lambda r: r.update(passed=False)),
            ('missing post-health', lambda r: r.update(post_health_passed=False)),
            ('legacy metadata-free gate', lambda r: r.update(oracle_schema=1)),
            ('changed frozen snapshot', lambda r: r.update(oracle_plan_sha256='0'*64)),
            ('failed source readback', lambda r: r['cases'][0]['report'].update(source_and_probe_passed=False)),
            ('missing owner destructor', lambda r: r['cases'][0]['report'].update(all_owners_destructor_returned=False)),
            ('nonzero process exit', lambda r: r['cases'][0]['state'].update(ExitCode=139)),
        ]
        for name, mutate in mutations:
            report = copy.deepcopy(actual)
            mutate(report)
            path = directory / 'receipt.json'
            path.write_text(json.dumps(report))
            try:
                c1.upload_gate(path, oracle, c1.ENGINE / 'receipt.json', c1.INTAKE)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Acceptance gate failed to reject: ' + name)
        path = directory / 'prepared.json'
        path.write_text(json.dumps({'launch_allowed': False}))
        try:
            c1.validate_prepared(directory)
        except ValueError:
            rejected.append('incomplete draft cannot launch')
        else:
            raise AssertionError('Incomplete draft unexpectedly launches')
    print(json.dumps({'scope': 'CPU actual coupled receipt/ledger revalidation and eight rejection controls',
        'positive_actual_v2_gate': True, 'rejected': rejected, 'gpu_operations': False, 'passed': len(rejected) == 8}, sort_keys=True))


if __name__ == '__main__':
    main()
