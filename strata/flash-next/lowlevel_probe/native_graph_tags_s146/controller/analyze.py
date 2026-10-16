from pathlib import Path
import json,math,re,sys
p=Path(sys.argv[1]);lines=(p/'leaf.stdout').read_text().splitlines()
rows=[json.loads(x)for x in lines if x.startswith('{')]
native=json.loads((p/'removal.json').read_text())['native_exit']
assert (p/'leaf').is_dir() and not list((p/'leaf').iterdir()),'no fixture/snapshot output files'
assert not (p/'leaf.stderr').read_text().strip(),'unexpected native diagnostics'
if native==3:
 assert rows==[{'status':'UNSUPPORTED_S146_NATIVE_PROFILING_TAG','native_aspect':False,'fallback_attempted':False}]
 assert not any('fixture' in x.lower() or 'HOLD' in x for x in lines)
 summary={'status':'UNSUPPORTED_S146_NATIVE_PROFILING_TAG','native_exit':3,'normal_capability_stop':True,'fallback_attempted':False,'fixture_absent':True}
else:
 assert native==0 and len(rows)==10
 snap=[r for r in rows if r.get('type')=='snapshot'];times=[r for r in rows if r.get('type')=='native_tag_timing'];final=[r for r in rows if r.get('status')]
 assert len(snap)==1 and len(times)==8 and len(final)==1
 r=snap[0];assert r['pair']==2 and r['phase']==0 and r['kind']=='S146 reference' and r['complete_bytes_compared']==235339776 and r['digest_not_numeric_comparator']is True
 assert all(re.fullmatch('[0-9a-fA-F]{16}',r[k])for k in ['owner0_fnv1a64','owner1_fnv1a64'])
 assert [x['iteration']for x in times]==list(range(8)) and [x['arm']for x in times]==[0,1,1,0,0,1,1,0]
 assert [x['tags']for x in times]==[False,False,True,True,True,True,False,False]
 for x in times:
  assert all(type(x[k])in (int,float) and math.isfinite(x[k])and x[k]>0 for k in ['completed_host_chain_us','host_graph_submit_us'])
  assert x['complete_state_bytes_checked']==235339776 and all(x[k]is True for k in ['guards','readonly','finite','output_sentinel'])
  assert x['conv_calls_source']==x['state_calls_source']==(2 if x['arm']else 72)
  assert x['device_envelope_valid']is x['tags']
  assert all(type(x[k])is int and x[k]>=0 for k in ['device_envelope_ns','begin_command_end_ns','end_command_start_ns'])
  if x['tags']:assert x['end_command_start_ns']>x['begin_command_end_ns'] and x['device_envelope_ns']==x['end_command_start_ns']-x['begin_command_end_ns']
  else:assert x['device_envelope_ns']==x['begin_command_end_ns']==x['end_command_start_ns']==0
 assert final[0]=={'status':'PASS_S146_NATIVE_TAG_METHOD_ONLY','native_aspect':True,'queue_profiling':False,'tags_outside_capture':True,'samples':8,'samples_per_arm_tag_cell':2,'accepted_pair':[2,2],'S145_full_numeric_campaign_not_repeated':True,'per_kernel_occupancy_or_readiness_qualified':False,'serving_qualified':False}
 summary={'status':'PASS_S146_NATIVE_TAG_METHOD_ONLY','native_exit':0,'reference':snap[0],'timings':times,'final':final[0],'scope':'two outside-capture device tags bracket whole captured graph; not per-op service, occupancy, readiness causality, CPU/GPU clock join or serving gain; all eight retained, two per arm/tag cell'}
(p/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':summary['status'],'native_exit':native}))
