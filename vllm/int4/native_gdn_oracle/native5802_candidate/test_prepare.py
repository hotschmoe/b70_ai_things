#!/usr/bin/env python3
"""CPU-only identity repin checks against the preserved actual oracle source."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import prepare


base = prepare.BASE_ROOT / 'inputs/oracle.py'
assert prepare.sha(base) == prepare.BASE_SHA
original = json.loads((prepare.BASE_ROOT / 'plan.json').read_text())
with tempfile.TemporaryDirectory(prefix='native5802-framework-') as tmp:
    root = Path(tmp)
    evidence = root / 'evidence.json'
    evidence.write_text('{"test_only":true}\n')
    command = [sys.executable, str(Path(prepare.__file__)), '--image', 'sha256:'+'a'*64,
               '--native-sha256', 'b'*64, '--pair-pass', str(root/'health/PASS'),
               '--pair-image-inspect', str(root/'health/candidate/image-inspect.json'),
               '--build-evidence', str(evidence), '--root', str(root/'plan'),
               '--output', str(root/'output')]
    subprocess.run(command, check=True, capture_output=True)
    plan = json.loads((root/'plan/plan.json').read_text())
    assert prepare.normalized(base.read_text()) == prepare.normalized((root/'plan/inputs/oracle.py').read_text())
    for key in ['check_names', 'geometry', 'steps', 'accepted_pattern', 'state_strides_bytes', 'prerequisite_outcomes']:
        assert plan[key] == original[key], key
    assert len(plan['check_names']) == 12 and plan['steps'] == 128
    assert plan['oracle_command'].count('sha256:'+'a'*64) == 1
    assert original['image'] not in plan['oracle_command']
    # Dry run succeeds without a health marker; activation must reject it.
    life = Path(__file__).with_name('lifecycle.py')
    subprocess.run([sys.executable, str(life), str(root/'plan/plan.json')], check=True, capture_output=True)
    rejected = subprocess.run([sys.executable, str(life), str(root/'plan/plan.json'), '--run'], capture_output=True)
    assert rejected.returncode != 0 and b'AssertionError' in rejected.stderr
    assert not (root/'output').exists()
    (root/'health/candidate').mkdir(parents=True)
    (root/'health/PASS').write_text('test only')
    (root/'health/candidate/image-inspect.json').write_text(json.dumps([{'Id':original['image']}]))
    rejected = subprocess.run([sys.executable, str(life), str(root/'plan/plan.json'), '--run'], capture_output=True)
    assert rejected.returncode != 0 and b'candidate pair health image mismatch' in rejected.stderr
    assert not (root/'output').exists()
assert prepare.sha(base) == prepare.BASE_SHA
print('PASS actual-source identity-only AST repin; all numerical gates unchanged; missing/wrong-image pair health rejected before GPU')
