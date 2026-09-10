#!/usr/bin/env python3
"""Fail closed on exact warehouse semantics and observed prefix reuse."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_replay_quality import audit


def agent(root):
    results = json.loads((root / 'results.json').read_text())
    assert len(results) == 4 and {s['session'] for s in results} == set(range(4))
    evidence = []
    for session in results:
        assert session['passed'] is True and len(session['rows']) == 8
        assert len(session['attempts']) == 8 and all(not a['bang'] and not a.get('error') for a in session['attempts'])
        hits = []
        assert [(r['turn'],r['phase']) for r in session['rows']] == [(i,p) for i in range(4) for p in ('tool','answer')]
        for row in session['rows']:
            assert row['passed'] is True and not row.get('error')
            response = row['response']; choice, = response['choices']; msg = choice['message']
            assert choice['finish_reason'] in ('stop','tool_calls')
            assert not re.search(r'(.{1,80}?)\1{15,}', json.dumps(msg), re.S)
            assert response['usage']['completion_tokens'] > 0
            hit = response['usage']['prompt_tokens_details']['cached_tokens']
            assert type(hit) is int and hit >= 0
            hits.append(hit)
            if row['phase'] == 'answer':
                assert msg['content'].strip() == str(730 + session['session']*10 + row['turn'])
                assert not msg.get('tool_calls')
            else:
                call, = msg['tool_calls']
                assert call['function']['name'] == 'lookup_stock' and call['id']
                assert json.loads(call['function']['arguments']) == {'sku':f"part-{session['session']}-{row['turn']}"}
        assert any(hit > 0 for hit in hits), 'no actual prefix reuse in session'
        evidence.append(dict(session=session['session'],cached_tokens=hits))
    return dict(passed=True,checks=32,sessions=evidence)


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['agent','replay']);p.add_argument('root',type=Path);p.add_argument('--count',type=int)
    a=p.parse_args()
    try:
        if a.kind=='agent': result=agent(a.root)
        else:
            result=audit(a.root)
            assert result['requests']==a.count
            assert all(not r['bang'] and not r['error'] and not r['flags'] for r in result['rows']), 'replay requires quality review'
            result['passed']=True
    except Exception as exc:
        result=dict(passed=False,error=ascii(exc))
    (a.root/'strict-quality-gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
