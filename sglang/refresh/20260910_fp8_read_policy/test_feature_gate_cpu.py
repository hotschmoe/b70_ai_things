"""CPU-only prerequisite gates; no serving command is executed."""
import hashlib
import json
from pathlib import Path
import tempfile
import feature_gate

with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);prior=root/'prior';(prior/'viability').mkdir(parents=True)
    (prior/'exit.rc').write_text('0');(prior/'job.rc').write_text('0')
    (prior/'viability/OUTCOME.json').write_text(json.dumps(dict(passed=True,requests=26)))
    pair=root/'PAIR_PASS';pair.touch();numeric=root/'numeric.json'
    numeric.write_text(json.dumps(dict(passed=True,all_nine_retained=True,image='sha256:fixture')))
    source=root/'source';source.write_text('frozen')
    plan=dict(pair_preflight_pass=str(pair),read_policy_numeric_outcome=str(numeric),
              image='sha256:fixture',prior_viability_outputs=[str(prior)],
              frozen_files={str(source):hashlib.sha256(source.read_bytes()).hexdigest()},output=str(root/'out'))
    feature_gate.validate(plan)
    source.write_text('changed')
    try:feature_gate.validate(plan);raise RuntimeError('modified source accepted')
    except AssertionError:pass
    source.write_text('frozen');numeric.write_text(json.dumps(dict(passed=False,all_nine_retained=True,image='sha256:fixture')))
    try:feature_gate.validate(plan);raise RuntimeError('numeric failure accepted')
    except AssertionError:pass
    numeric.write_text(json.dumps(dict(passed=True,all_nine_retained=True,image='sha256:fixture')))
    (prior/'exit.rc').write_text('1')
    try:feature_gate.validate(plan);raise RuntimeError('failed lifecycle accepted')
    except AssertionError:pass
print('PASS frozen source, numeric gate and prior lifecycle failures block feature launch')
