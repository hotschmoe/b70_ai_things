"""Validate a reviewed completed100K receipt before freezing/launching200K."""
import hashlib
import json
from pathlib import Path

IMAGE='sha256:d55637b3353eaf470677627dc1627c3dda3ec6a6abb0298451aed34b73937067'
NATIVE='6717e3ec7fe3e7cdc26cd66b84aabdd076f211e5bed62b8711b7377fca8668d9'
SCALES='be02d915a8ac188341870cc9f642d77665b744235e142330a7a37b8f4c711062'
ROOT=Path('/mnt/vm_8tb/b70/results/bang_recurrence_testing_20260910/native5802-backend100k-v2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(path, expected_sha):
    path=Path(path)
    if not expected_sha or len(expected_sha)!=64 or digest(path)!=expected_sha:
        raise ValueError('Missing or changed reviewed100K receipt')
    r=json.loads(path.read_text())
    required={'passed':True,'backend_scope':'native5802-backend100k',
              'root':str(ROOT),'image':IMAGE,'native_sha256':NATIVE,
              'scale_sha256':SCALES,'lifecycle_passed':True,'tool_checks':192,
              'tiny_checks':24,'early_checks':3,'strict_pre_post_health':True,
              'known_numeric_array_quality_passed':False}
    for k,v in required.items():
        if type(r.get(k))!=type(v) or r.get(k)!=v:raise ValueError(('Receipt field',k))
    hashes=r.get('evidence_hashes')
    if not isinstance(hashes,dict) or not hashes:raise ValueError('Evidence hashes absent')
    # Receipt producer must include the complete reviewed backing evidence.
    mandatory={ROOT/'plan.json',ROOT/'frozen.json',ROOT/'plan.lifecycle-rc',
               ROOT/'run/WORKLOADS_PASSED',ROOT/'known-quality-negative.json'}
    for stage in ('pre','post'):
        mandatory.update(ROOT/'run'/f'{stage}-{tool}.log' for tool in ('xpu-health','xpu-collective-health'))
    plan=json.loads((ROOT/'plan.json').read_text())
    for job in plan['jobs']:mandatory.add(ROOT/'run/jobs'/(job['name']+'.done'))
    for kind in ('deterministic','sampled'):
        for i in (1,2,3):
            base=ROOT/'run'/f'{kind}-round{i}'
            mandatory.update(base/f for f in ('strict-quality-gate.json','results.json','summary.json'))
    mandatory.update(ROOT/'run'/f for f in ('tiny-prefill.json','early-history/strict-quality-gate.json','startup-host-trace-review.json'))
    if not {str(p) for p in mandatory}.issubset(hashes):raise ValueError('Incomplete backing evidence')
    for p,sha in hashes.items():
        if digest(p)!=sha:raise ValueError(('Evidence changed',p))
    if (ROOT/'plan.lifecycle-rc').read_text().strip()!='0':raise ValueError('Lifecycle failed')
    for job in plan['jobs']:
        if (ROOT/'run/jobs'/(job['name']+'.done')).read_text().strip()!='0':raise ValueError('Job failed')
    for p,sha in json.loads((ROOT/'frozen.json').read_text()).items():
        if digest(p)!=sha:raise ValueError(('Frozen input changed',p))
    return r
