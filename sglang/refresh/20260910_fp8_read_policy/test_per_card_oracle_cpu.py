"""CPU-only lifecycle controls, including expected-failure JSON retention."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch

import per_card_oracle as runner


def report(gate):
    rows = [dict(mode=mode, query_length=q, scale_case=label,
                 close_gate=gate, scale_sensitivity_gate='pass')
            for mode, q in [('decode', 1), ('extend', 1), ('extend', 4)]
            for label in ['correct', 'K_descaling_x2', 'V_descaling_x2']]
    return dict(attention_execution='actual Triton kernels', device='xpu',
                bytes='pass', untouched_slots='pass', cloned_inputs='pass',
                attention=rows, source_hashes={}, loader_source_sha256='fixture',
                attention_gate=gate)


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    lock = root / 'gpu.lock.0'
    lock.touch()
    fd = os.open(lock, os.O_WRONLY)
    os.dup2(fd, 8)
    if fd != 8:
        os.close(fd)
    probe = root / 'strict'; probe.write_text('CPU fixture only\n')
    prior = root / 'PAIR_PASS'; prior.touch()
    for gate in ('fail', 'pass', 'crash'):
        out = root / gate
        plan = dict(card=0, image='sha256:fixture', expectation=gate,
                    output=str(out), pair_preflight_pass=str(prior), frozen_files={},
                    health_probe=str(probe), health_sha256=runner.sha(probe),
                    oracle_command=['CPU_FIXTURE'], source_hashes={}, loader_sha256='fixture')
        stages = []
        def run(command, path, timeout, owned):
            assert owned
            stages.append(path.name)
            if path.name == 'oracle.log':
                assert timeout == 420
                path.write_text('fixture crash' if gate == 'crash' else json.dumps(report(gate)))
                return 2 if gate == 'crash' else int(gate == 'fail')
            assert command[-2:] == ['--card', '0']
            path.write_text('CPU fake health\n')
            return 0
        with patch.dict(os.environ, B70_GPU_LOCK=str(root/'gpu.lock')):
            with patch.object(runner.health, 'run', side_effect=run), patch.object(
                    runner.health, 'capture', return_value=NS(returncode=0, stdout='[{"Id":"sha256:fixture"}]')):
                rc = runner.execute(plan)
        assert stages == ['pre-health.log', 'oracle.log', 'post-health.log']
        assert rc == (0 if gate == 'pass' else 1)
        outcome = json.loads((out/'OUTCOME.json').read_text())
        assert outcome['post_health_rc'] == 0
        if gate != 'crash':
            assert len(json.loads((out/'oracle-result.json').read_text())['attention']) == 9
            assert outcome['attention_gate'] == gate
    os.close(8)
print('PASS all nine rows retained on exit1; candidate exit0; crash still post-health; selected-card checks only')
