#!/usr/bin/env python3
"""Actual source counters and explicit SDK/C1 lanes; no GGUF weight reads/GPU."""
import ast,hashlib,json,tempfile,re
from pathlib import Path
from unittest.mock import patch
import batch_numerical_proofs_v3 as p
from audit_batch_fidelity_coverage_v4 import coverage
HERE=Path(__file__).resolve().parent
ENGINE=Path('/mnt/vm_8tb/b70/build/strata-native-hc-engine-20261009T185206Z-d5q32zc7')

def bad(fn):
 try:fn()
 except (AssertionError,ValueError,KeyError,FileNotFoundError):return
 raise AssertionError('Wrong sentinel/source lane accepted')

def main():
 controls=0
 source=(ENGINE/'source/sycl/src/program/generate.cpp').read_text();assert 'int64_t reread_to = -1;' in source and 'const int64_t read_from = reread_to > 0 ? 0 : resume;' in source
 assert source.count('cache_prefix_ceiling>0 && o.prompt_cache > 0')==2 and 'if (!conversations.enabled() || !live_ok || live.empty()) return true;' in source
 header=(ENGINE/'source/include/strata/core/conversation_cache.hpp').read_text();assert 'bool enabled() const { return budget_ != 0 && slots_ != 0; }' in header
 for lane,generation in [('source29',8),('source31',9)]:
  c1,parent=p.providers(lane);assert c1.__name__.endswith('_v'+str(generation)) and parent.__name__.endswith('_v'+str(generation))
 actual=p.engine_binding(ENGINE,'source29');assert len(actual['patched_source_sha256'])==60
 bad(lambda:p.engine_binding(ENGINE,'source31'));controls+=1;bad(lambda:p.providers('old28'));controls+=1
 trace='SBF resume rid=1 slot=0 reused=0 read_from=0 reread_to=-1';policy={(1,'admission'):{'reused':0,'read_from':0,'reread_to':-1}};assert p.exact_producer_counters(trace,{1},policy)
 for field,value in [('reread_to','0'),('reread_to','1'),('reused','1'),('read_from','1')]:bad(lambda:p.exact_producer_counters(trace.replace(field+'='+('-1' if field=='reread_to' else '0'),field+'='+value),{1},policy));controls+=1
 latest=json.loads(Path('/mnt/vm_8tb/b70/build/strata-observer26-cpu-latest-v1.json').read_bytes());out=Path(latest['output']);raw=(out/'trace.log').read_text();assert hashlib.sha256(raw.encode()).hexdigest()==latest['trace_sha256'];fixed=re.sub(r'reread_to=\d+','reread_to=-1',raw);result=coverage(fixed,[11,12],[11,12],[(0,0,32),(1,32,48)],out/'capture');assert result['migration_vectors']==98
 bad(lambda:coverage(raw,[11,12],[11,12],[(0,0,32),(1,32,48)],out/'capture'));controls+=1
 for n in (2,4,6):
  spec=json.loads((HERE/('batch-numerical-case'+str(n)+'-source29-v1.json')).read_bytes());assert spec['slots']==n and spec['source_lane']=='source29' and len(spec['actual_counter_policy'])==n+1
  assert all(row['values']=={'reused':0,'read_from':0,'reread_to':-1} for row in spec['actual_counter_policy'])
  assert [row['logical_index'] for row in spec['actual_counter_policy'] if row['role']=='solo_migration']==[1]
  assert all(ids[0]!=target[0] for ids in spec['tokens']['warm'] for target in spec['tokens']['target']) and spec['api_token_ids']['target']==spec['tokens']['target']
 print(json.dumps({'passed':True,'negative_controls':controls,'actual_source_no_cache_no_reread_contract':'0/0/-1','synthetic_migration_vectors':98,'actual_source29_SDK_files':60,'future_source31_oldSDK_rejected_before_model_admission':True,'actual_model_weight_payload_reads':0,'actual_GPU_executions':0},ensure_ascii=True))
if __name__=='__main__':main()
