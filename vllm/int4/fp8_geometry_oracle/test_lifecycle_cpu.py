"""CPU-only full-context FP8 result and lifecycle gates."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
from unittest.mock import patch
import lifecycle as runner


def rows(q):
    return [dict(card=0,exact_representable_write=True,untouched_slots=True,distinct_kv_scales=True),
            dict(card=0,hybrid_layout=True,cache_stride=[1638400,512,256,1],length=23240,query_length=q,attention_relative_l2=.002),
            dict(card=0,length=23240,block_size=1600,block_table=list(range(16,1,-1)),write_k_relative_l2=.02,write_v_relative_l2=.02,attention_relative_l2=.002,k_and_v_read_scales_consumed=True,q_dtype='torch.float16')]


with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);lock=root/'gpu.lock.0';lock.touch();fd=os.open(lock,os.O_WRONLY);os.dup2(fd,8)
    if fd!=8:os.close(fd)
    prior=root/'PAIR_PASS';prior.touch();probe=root/'probe';probe.write_text('CPU fixture')
    for q,mode in [(1,'pass'),(4,'pass'),(1,'numeric_fail'),(4,'post_fail')]:
        out=root/(str(q)+mode)
        plan=dict(card=0,image='sha256:fixture',output=str(out),pair_preflight_pass=str(prior),frozen_files={},health_probe=str(probe),health_sha256=runner.sha(probe),oracle_command=['CPU_FIXTURE'],length=23240,block_size=1600,query_length=q)
        stages=[]
        def run(cmd,path,timeout,owned):
            assert owned;stages.append(path.name)
            if path.name=='oracle.log':
                assert timeout==420
                data=rows(q)[:2] if mode=='numeric_fail' else rows(q)
                path.write_text('\n'.join(json.dumps(x) for x in data))
                return int(mode=='numeric_fail')
            assert cmd[-2:]==['--card','0'];path.write_text('CPU fake health')
            return int(mode=='post_fail' and path.name=='post-health.log')
        with patch.dict(os.environ,B70_GPU_LOCK=str(root/'gpu.lock')):
            with patch.object(runner.health,'run',side_effect=run),patch.object(runner.health,'capture',return_value=NS(returncode=0,stdout='[{"Id":"sha256:fixture"}]')):
                rc=runner.execute(plan)
        assert stages==['pre-health.log','oracle.log','post-health.log']
        assert rc==(0 if mode=='pass' else 1)
        assert len(json.loads((out/'oracle-result.json').read_text()))==(2 if mode=='numeric_fail' else 3)
    os.close(8)
print('PASS q1/q4 geometry, partial numerical failure retained, selected post-health and no reset')
