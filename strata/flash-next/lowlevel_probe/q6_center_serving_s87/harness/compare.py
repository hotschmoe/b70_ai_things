#!/usr/bin/env python3
"""S87 exact observable output/work/capture gates; no per-window trace inference."""
import hashlib,json,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def content(r):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content')or'' for c in r['chunks']if c.get('choices'))
inputs=read(D/'serving-inputs.json');assert inputs['same_binary']and inputs['arms']['0']==inputs['arms']['1']
for a in ['0','1']:
 with Path(inputs['arms'][a]['exe']).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==inputs['arms'][a]['sha256']
configs=[read(D/f'mode{a}/config.json')for a in range(2)];normalized=[]
for a,c in enumerate(configs):
 assert c['exe']==inputs['arms'][str(a)]['exe'] and c['log']==str(D/f'mode{a}/engine.log')
 assert c['env']['STRATA_Q6_CENTER']==str(a)and'STRATA_BATCH_COMMIT_OVERLAP'not in c['env']
 for k,v in [('STRATA_MMVQ_WIDE','1'),('STRATA_MMVQ_WIDE_MIN','0'),('STRATA_MMVQ_WIDE_MAXC','8'),('STRATA_Q6_PACKED','0'),('STRATA_HEAD_Q5K_RPW2','0'),('STRATA_BATCH_MTP_PLE_PREFETCH','0'),('STRATA_TRACE','1'),('STRATA_MMVQ_WIDE_K','1'),('STRATA_MMVQ_WIDE_K_RPW','1'),('STRATA_MMVQ_WIDE_RPW','1'),('STRATA_MMVQ_A2','1'),('STRATA_MMVQ_PF','0')]:assert c['env'][k]==v
 assert all(k not in c['env']for k in ['STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING','STRATA_PREFILL_SYNC','STRATA_VERIFY_TRACE','STRATA_VERIFY_EAGER'])
 n=json.loads(json.dumps(c));del n['log'];del n['env']['STRATA_Q6_CENTER'];normalized.append(n)
assert normalized[0]==normalized[1]
checks={'identity_image_and_effective_native_env':all(read(D/f'mode{a}/health-ready.json')['model']=='hotschmoe-dd'and read(D/f'mode{a}/container-ready.json')['Image']==inputs['image']and len(read(D/f'mode{a}/native-process-identity.json'))==1 for a in range(2))}
results=[]
for label in ['warm','repeat1','repeat2','repeat3']:
 rows=[[read(D/f'mode{a}/coding/{label}/coding-{i}.json')for i in range(2)]for a in range(2)];pairs=[read(D/f'mode{a}/coding/{label}/pair.json')for a in range(2)]
 c={'admission_overlap':all(p['admission_gate']['status']=='PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP'and p['client_overlap_ns']>0 for p in pairs),
 'request_equal':all(rows[0][i]['request']==rows[1][i]['request']for i in range(2)),
 'full_text_equal':all(rows[0][i]['text']==rows[1][i]['text']for i in range(2)),
 'SSE_content_equal':all(content(rows[0][i])==content(rows[1][i])==rows[0][i]['text']for i in range(2)),
 'usage_equal':all(rows[0][i]['usage']==rows[1][i]['usage']and rows[0][i]['usage']['completion_tokens']==1500 and rows[0][i]['usage']['prompt_tokens']==107 for i in range(2)),
 'native_draft_work_equal':pairs[0]['work_timings']==pairs[1]['work_timings'],
 'native_window_pair_solo_equal':all(pairs[0][k]==pairs[1][k]for k in ['native_windows','paired','solo']),
 'native_reported_rows_equal':pairs[0]['native_window_rows']==pairs[1]['native_window_rows'],
 'logged_capture_layouts_equal':pairs[0]['logged_capture_layouts']==pairs[1]['logged_capture_layouts'],
 'eligible_host_selection':all(pairs[a]['q6_host_selection_marker']==f'strata Q6 center: enabled={a} C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240'for a in range(2))}
 checks[label]=all(c.values())
 results.append({'label':label,'matched':checks[label],'checks':c,'pair_wall_s':[p['wall_s']for p in pairs],
 'aggregate_output_tps':[p['aggregate_output_tps']for p in pairs],'streams':[p['stream_metrics']for p in pairs],
 'native_work':[{'per_request':p['work_timings'],'windows':p['native_windows'],'paired':p['paired'],'solo':p['solo'],'reported_rows':p['native_window_rows'],'logged_new_captures':p['logged_capture_layouts']}for p in pairs]})
checks['functionals']=all(read(D/f'mode{a}/functional/verdict.json')['status']=='PASS'for a in range(2))
checks['health_teardown']=all(read(D/f'mode{a}/normal-stop.json')['lease_wrapper_exit']==0 and read(D/f'mode{a}/normal-stop.json')['container_removed']and all(v['exit']==0 for v in read(D/f'mode{a}/post-health.json').values())for a in range(2))
checks['full_CID_and_NAME_absence']=all(read(D/f'mode{a}/container-removal.json')['docker_ps_exit']==0 and read(D/f'mode{a}/container-removal.json')['full_cid_absent'] and read(D/f'mode{a}/container-removal.json')['name_absent'] and read(D/f'mode{a}/container-removal.json')['created_cid']==read(D/f'mode{a}/container-ready.json')['Id'] for a in range(2))
checks['final_diagnostics_complete']=all(read(D/f'mode{a}/aggregate-proof.json')['present_once']for a in range(2))
startup=[]
for a in range(2):
 log=(D/f'mode{a}/engine-final.log').read_text();offset=read(D/f'mode{a}/coding/warm/window-start.json')['byte_offset'];head=(D/f'mode{a}/engine-final.log').read_bytes()[:offset].decode()
 startup.append(re.findall(r'^strata verify: captured .*$',head,re.M))
 markers=re.findall(r'^strata Q6 center: .*$',log,re.M);checks[f'final_host_marker_{a}']=markers==[f'strata Q6 center: enabled={a} C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240']
checks['logged_startup_capture_equal']=startup[0]==startup[1]
matched=all(checks.values())
summary=None
if matched:
 summary={'pair_wall_median_s':[statistics.median(x['pair_wall_s'][a]for x in results[1:])for a in range(2)],
 'aggregate_output_tps_median':[statistics.median(x['aggregate_output_tps'][a]for x in results[1:])for a in range(2)],
 'per_stream_medians':[[{k:statistics.median(x['streams'][a][i][k]for x in results[1:])for k in ['ttft_s','completion_wall_s','decode_tps','whole_request_output_tps']}for i in range(2)]for a in range(2)]}
report={'status':'PASS_MATCHED_DISCRIMINATOR'if matched else'FAILED_OR_UNMATCHED','checks':checks,'same_binary':True,'pairs':results,'timed_summary':summary,
 'latency_scope':'client wall and SSE content-event gaps; TRACE1 CPU logging common, CPU ring/profile absent; SSE events can contain many tokens, not token ITL/device service',
 'capture_scope':'exact observed new-capture logs and aggregate native work only; not all per-window layouts/cache hits or independent GPU kernel dispatch',
 'reported_rows_scope':'native printed avg rows rounded to2decimals, not exact unprinted per-window row sequence',
 'configuration_normalization':['STRATA_Q6_CENTER0/1','arm-local log/container/service names'],'raw_native_token_ids_observed':False,'no_promotion_claim':True}
(D/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':report['status'],'checks':checks}));raise SystemExit(0 if matched else 2)
