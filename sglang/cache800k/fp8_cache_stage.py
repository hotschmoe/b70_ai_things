#!/usr/bin/env python3
"""Two bounded probes inside the serving lifecycle; require scale load and hits."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess


def loaded_target(log, artifact):
    marker='B70_SGLANG_KV_SCALES_LOADED '
    rows=[]
    for line in log.splitlines():
        if marker in line:
            rows.append(json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0])
    assert len(rows)==1, 'require one target loader receipt with MTP disabled'
    row=rows[0]
    assert row['artifact_sha256']==artifact
    assert row['role']=='target' and row['tp_rank']==0 and row['tp_size']==1
    assert row['backend']=='triton' and row['kv_dtype']=='fp8_e4m3'
    assert row['coverage']==16 and len(row['rows'])==16
    assert all(item['query_scale_applied'] is False for item in row['rows'])
    return row


def cached_tokens(text):
    total=0.0
    for line in text.splitlines():
        if not line.startswith('sglang:cached_tokens_total{'):
            continue
        labels=dict(re.findall(r'([a-zA-Z_][a-zA-Z_0-9]*)="([^"]*)"',line))
        if labels.get('model_name')!='hotschmoe-dd' or labels.get('cache_source') not in ('device','total'):
            continue
        value=float(line.rsplit(' ',1)[1])
        assert math.isfinite(value) and value>=0
        total+=value
    return total


def main():
    parser=argparse.ArgumentParser();parser.add_argument('plan',type=Path);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());root=Path(plan['server_output']);result={'passed':False,'stages':[]}
    try:
        receipt=loaded_target((root/'server.log').read_text(),plan['artifact_sha256'])
        (root/'calibrated-loader-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        for stage in plan['stages']:
            with (root/(stage['name']+'.log')).open('w') as log:
                completed=subprocess.run(stage['command'],stdout=log,stderr=subprocess.STDOUT,timeout=stage['timeout'])
            result['stages'].append(dict(name=stage['name'],rc=completed.returncode))
            assert completed.returncode==0, stage['name']+' failed'
            if stage['name']=='tiny-text':
                tiny=json.loads((root/'viability/OUTCOME.json').read_text())
                assert tiny['passed'] is True and tiny['requests']==26
        summary=json.loads((root/'tools/summary.json').read_text())
        assert summary['passed'] is True and summary['checks']==32
        assert summary['attempts']==32 and summary['bang_attempts']==0
        before=cached_tokens((root/'tools/metrics-before.txt').read_text())
        after=cached_tokens((root/'tools/metrics-after.txt').read_text())
        result.update(cached_tokens_before=before,cached_tokens_after=after,cached_token_delta=after-before)
        assert after>before, 'concurrent fixture did not demonstrate positive cache reuse'
        result['passed']=True
    except Exception as exc:
        result['error']=ascii(exc)
    (root/'fp8-cache-stage-outcome.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
