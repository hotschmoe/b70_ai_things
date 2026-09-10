"""CPU lifecycle controls; all GPU helpers are mocked."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import lifecycle as runner
import oracle


def report(passed):
    return dict(execution='actual installed Triton copy kernels',
                worker_source_sha256=oracle.WORKER_SHA, model_source_sha256=oracle.MODEL_SHA,
                cases=[dict(case=c,mismatched_bytes=0 if passed else 1,source_untouched=True,
                            all_untargeted_bytes_covered=True,passed=passed) for c in oracle.cases()],passed=passed)


with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);lock=root/'gpu.lock.1';lock.touch();fd=os.open(lock,os.O_WRONLY);os.dup2(fd,9)
    if fd!=9:os.close(fd)
    probe=root/'probe';probe.write_text('CPU fixture');prior=root/'PAIR_PASS';prior.touch()
    for mode in ['pass','numeric_fail','crash','post_fail']:
        out=root/mode
        plan=dict(card=1,image='sha256:fixture',output=str(out),pair_preflight_pass=str(prior),
                  frozen_files={},health_probe=str(probe),health_sha256=runner.sha(probe),
                  oracle_command=['CPU_FIXTURE'],worker_source_sha256=oracle.WORKER_SHA,
                  model_source_sha256=oracle.MODEL_SHA,cases=oracle.cases())
        stages=[]
        def run(cmd,path,timeout,owned):
            assert owned;stages.append(path.name)
            if path.name=='oracle.log':
                assert timeout==420
                path.write_text('crash' if mode=='crash' else json.dumps(report(mode!='numeric_fail')))
                return 2 if mode=='crash' else int(mode=='numeric_fail')
            assert cmd[-2:]==['--card','1']
            path.write_text('CPU fake health')
            return int(mode=='post_fail' and path.name=='post-health.log')
        with patch.dict(os.environ,B70_GPU_LOCK=str(root/'gpu.lock')):
            with patch.object(runner.health,'run',side_effect=run),patch.object(runner.health,'capture',return_value=NS(returncode=0,stdout='[{"Id":"sha256:fixture"}]')):
                rc=runner.execute(plan)
        assert stages==['pre-health.log','oracle.log','post-health.log']
        assert rc==(0 if mode=='pass' else 1)
        outcome=json.loads((out/'OUTCOME.json').read_text())
        if mode!='crash':assert len(json.loads((out/'oracle-result.json').read_text())['cases'])==12
        if mode=='post_fail':assert 'no pair reset' in outcome['recovery']
    os.close(9)
print('PASS numeric failure retention, crash post-health, selected card only, failed post-health blocks success without reset')
