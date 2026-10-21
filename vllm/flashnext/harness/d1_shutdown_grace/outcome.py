"""Reuse unchanged operational classifier and require actual shutdown30 proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
CORE=Path('/mnt/vm_8tb/github/b70_ai_things/vllm/flashnext/harness/d1_existing_route_q8f')
p=argparse.ArgumentParser();p.add_argument('--aout',type=Path,required=True);p.add_argument('--outer',type=Path,required=True);p.add_argument('--arm',choices=['control','routeq8f'],required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('d1_existing_outcome',CORE/'outcome.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
try:
    r=m.classify(a.aout,a.outer,a.arm)
    actual=json.loads((a.aout/'lmx/server-config.json').read_text()).get('shutdown_timeout')
    r['actual_shutdown_timeout']=actual
    if actual!=30:r['issues'].append('live shutdown_timeout is not30')
    try:armed=json.loads((a.aout/'teardown-armed.json').read_text())
    except Exception:armed={'first_sample_saved':False,'error':'missing/invalid handshake'}
    r['teardown_monitor_handshake']=armed
    if armed.get('first_sample_saved')is not True:r['issues'].append('teardown monitor first sample unavailable')
    try:r['teardown_proc_summary']=json.loads((a.aout/'teardown-proc.summary.json').read_text())
    except Exception:r['teardown_proc_summary']=None;r['issues'].append('teardown proc summary missing')
    raw=(a.aout/'window.log').read_text()
    for key in ['TEARDOWN_ARM_RC','TEARDOWN_WATCH_RC']:
        values=re.findall(r'\b'+key+r':(-?\d+)\b',raw)
        if not values or values[-1]!='0':r['issues'].append(key+' missing/failed')
    r['operational_complete']=not r['issues']
except Exception as e:r={'operational_complete':False,'issues':['incomplete grace evidence '+type(e).__name__],'qualification_pass':False}
r['scope']='finite grace discrimination only; original forced-teardown reject remains unchanged'
(a.outer/'outcome.json').write_text(json.dumps(r,indent=2)+'\n',encoding='ascii');print(json.dumps(r))
raise SystemExit(0 if r['operational_complete']else 1)
