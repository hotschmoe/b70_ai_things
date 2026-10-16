#!/usr/bin/env python3
"""Proof is rejected-adapter identity + surviving load + actual LevelZero oracle success."""
from pathlib import Path
import json,re,sys
p=Path(sys.argv[1]);env=json.loads((p/'environment.json').read_text())
assert env['SYCL_UR_TRACE']=='1'
assert all(env[k]is None for k in ['SYCL_UR_USE_LEVEL_ZERO_V2','UR_LOADER_USE_LEVEL_ZERO_V2','UR_L0_USE_DRIVER_INORDER_LISTS'])
assert(p/'mode0/leaf.exit').read_text().strip()=='0'
assert json.loads((p/'mode0/closure.json').read_text())['status']=='PASS_ONE_MODE_TWO_QUEUE_ORDERING_ONLY'
s=(p/'mode0/leaf.stderr').read_text();lines=s.splitlines()
skips={v:[l for l in lines if 'Skipping L0 V'+str(v)+' adapter:'in l]for v in[1,2]}
rejected=[v for v in[1,2]if skips[v]]
assert len(rejected)==1,('HOLD ambiguous/no adapter rejection',skips)
r=rejected[0];active=3-r
assert all('Using default adapter version based on device IP version'in l for l in skips[r]),('HOLD forced/unknown reason',skips)
lib='libur_adapter_level_zero'+('_v2'if active==2 else'')+'.so'
loaded=[l for l in lines if lib in l and re.search(r'loaded adapter',l,re.I)]
assert loaded,('HOLD no surviving adapter load proof',lib)
assert not any('Skipping Level Zero adapter due to'in l for l in lines), 'HOLD unsupported driver/loader skip'
v={'status':'PASS_RUNTIME_DISCOVERY_ONLY','active_adapter_version':active,'rejected_adapter_version':r,'proof_default_device_IP_rejection':skips[r],'proof_surviving_adapter_loaded':loaded,'proof_LevelZero_execution':'original checksum-bound probe.cpp asserts both queues backend LevelZero, same context/device, strict8-depthA/B/A completed','queue_properties_source':['in_order'],'enable_profiling_source':False,'main_queue_profiling_not_measured':True,'knob_semantics_or_performance_not_measured':True,'no_driver_or_adapter_force':True}
(p/'comparison.json').write_text(json.dumps(v,indent=2)+'\n',encoding='ascii');print(json.dumps(v))
