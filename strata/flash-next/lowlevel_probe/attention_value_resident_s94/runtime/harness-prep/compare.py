#!/usr/bin/env python3
"""S94 same-binary prefill-only comparison; no summary for unmatched arms."""
import hashlib,json,math,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def content(r):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content')or'' for c in r['chunks']if c.get('choices'))
WORK=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
def work(r):
 assert all(k in r['timings'] and isinstance(r['timings'][k],(int,float)) and not isinstance(r['timings'][k],bool) and math.isfinite(r['timings'][k]) and r['timings'][k]>=0 for k in WORK)
 return {k:r['timings'][k] for k in WORK}
def row_checks(x,y):
 return {'request_equal':x['request']==y['request'],'full_text_and_SSE_equal':x['text']==y['text']==content(x)==content(y),'usage_equal':x['usage']==y['usage'],'native_work_equal':work(x)==work(y)}
inputs=read(D/'serving-inputs.json');assert inputs['same_binary'] and inputs['arms']['0']==inputs['arms']['1']
configs=[read(D/f'mode{a}/config.json')for a in range(2)];normalized=[]
fixed={'STRATA_PROMPT_ATTN_XMX':'0','STRATA_KV_ROT':'0','STRATA_QSA_PREFILL_TILED':'0','STRATA_Q6_CENTER':'0','STRATA_MMVQ_WIDE':'1','STRATA_MMVQ_WIDE_MIN':'0','STRATA_MMVQ_WIDE_MAXC':'8','STRATA_Q6_PACKED':'0','STRATA_HEAD_Q5K_RPW2':'0','STRATA_BATCH_MTP_PLE_PREFETCH':'0','STRATA_TRACE':'1','STRATA_MMVQ_WIDE_K':'1','STRATA_MMVQ_WIDE_K_RPW':'1','STRATA_MMVQ_WIDE_RPW':'1','STRATA_MMVQ_A2':'1','STRATA_MMVQ_PF':'0','STRATA_BATCH_PREFILL_QUANTUM':'0','STRATA_PREFILL_FIRST':'0'}
for a,c in enumerate(configs):
 assert c['exe']==inputs['arms'][str(a)]['exe'] and c['log']==str(D/f'mode{a}/engine.log') and c['env']['STRATA_ATTN_VALUE_PIPELINE']==str(a)
 with Path(c['exe']).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==inputs['arms'][str(a)]['sha256']
 assert all(c['env'][k]==v for k,v in fixed.items()) and 'STRATA_BATCH_COMMIT_OVERLAP' not in c['env']
 assert all(k not in c['env']for k in ['STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING','STRATA_PREFILL_SYNC','STRATA_VERIFY_TRACE','STRATA_VERIFY_EAGER','STRATA_ATTN_LANECELL'])
 n=json.loads(json.dumps(c));del n['log'];del n['env']['STRATA_ATTN_VALUE_PIPELINE'];normalized.append(n)
assert normalized[0]==normalized[1]
checks={'identity_image_and_native_process':all(read(D/f'mode{a}/health-ready.json')['model']=='hotschmoe-dd' and read(D/f'mode{a}/container-ready.json')['Image']==inputs['image'] and len(read(D/f'mode{a}/native-process-identity.json'))==1 for a in range(2))}
warm=[]
for i in range(2):
 x,y=[read(D/f'mode{a}/coding/warm/coding-{i}.json')for a in range(2)];c=row_checks(x,y);c['counts']=x['usage']['prompt_tokens']==y['usage']['prompt_tokens']==107 and x['usage']['completion_tokens']==y['usage']['completion_tokens']==256;c['retained_S16']=x['prefix_matches_s16']and y['prefix_matches_s16'];checks[f'warm_slot{i}']=all(c.values());warm.append(c)
pairs=[read(D/f'mode{a}/coding/warm/pair.json')for a in range(2)]
checks['warm_native_work_capture']=all(pairs[0][k]==pairs[1][k]for k in ['work_timings','native_windows','paired','solo','native_window_rows','logged_capture_layouts'])
checks['warm_value_marker_absent']=all('prefill attention value pipeline selected'not in(D/f'mode{a}/coding/warm/engine-segment.log').read_text()for a in range(2))
checks['warm_admission_overlap']=all(p['admission_gate']['status']=='PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP' and p['client_overlap_ns']>0 and p['paired']>0 for p in pairs)
results=[]
for case in ['p4096','p32768']:
 for repeat in range(3):
  name=f'{case}-{repeat}';rows=[read(D/f'mode{a}/prefill/{name}.json')for a in range(2)];verdicts=[read(D/f'mode{a}/prefill/{name}-verdict.json')for a in range(2)];c=row_checks(*rows)
  c['local_gates']=all(v['status']=='PASS_LOCAL_GATES'for v in verdicts)
  c['fresh_counts']=all(r['usage']['prompt_tokens']==(4059 if case=='p4096'else 32728) and r['usage']['completion_tokens']==256 and work(r)['cache_n']<=64 for r in rows)
  c['native_part_intervals_equal']=verdicts[0]['native_part_intervals']==verdicts[1]['native_part_intervals']
  c['marker_domain']=all((v['QSA_marker'] is None if case=='p4096' else v['QSA_marker']['enabled']==0 and v['QSA_marker']['first_active']==4097)for a,v in enumerate(verdicts))
  c['value_HOST_selection']=all((v['VALUE_marker']['count']==0 and not v['VALUE_marker']['selected'])if a==0 else(v['VALUE_marker']['count']==1 and v['VALUE_marker']['selected']and v['VALUE_marker']['first_emission_request']=='p4096-0')for a,v in enumerate(verdicts))
  checks[name]=all(c.values());results.append({'name':name,'matched':checks[name],'checks':c,'work':[work(r)for r in rows],'usage':[r['usage']for r in rows],'value_HOST_observations':[v['VALUE_marker']for v in verdicts],'metrics':[{k:r[k]for k in ['ttft_s','wall_s','client_decode_tps','content_event_gap_ms']}|{'native_prompt_tps':r['timings']['prompt_per_second'],'native_prompt_ms':r['timings']['prompt_ms'],'output_sha256':r['output_sha256']}for r in rows]})
checks['ON_first4K_selection']=read(D/'mode1/prefill/complete.json')['first_value_emission_request']=='p4096-0'
checks['functionals']=all(read(D/f'mode{a}/functional/verdict.json')['status']=='PASS'for a in range(2))
checks['prestop_running_not_OOM']=all(read(D/f'mode{a}/normal-stop.json')['status']=='PASS_NORMAL_TEARDOWN'and read(D/f'mode{a}/normal-stop.json')['prestop_state']['Running']is True and read(D/f'mode{a}/normal-stop.json')['prestop_state']['OOMKilled']is False for a in range(2))
checks['health_teardown']=all(read(D/f'mode{a}/normal-stop.json')['lease_wrapper_exit']==0 and read(D/f'mode{a}/normal-stop.json')['container_removed']and all(v['exit']==0 for v in read(D/f'mode{a}/post-health.json').values())for a in range(2))
checks['full_CID_and_NAME_absent']=all(read(D/f'mode{a}/container-removal.json')['docker_ps_exit']==0 and read(D/f'mode{a}/container-removal.json')['full_cid_absent']and read(D/f'mode{a}/container-removal.json')['name_absent']and read(D/f'mode{a}/container-removal.json')['created_cid']==read(D/f'mode{a}/container-ready.json')['Id']for a in range(2))
checks['final_diagnostics_complete']=all(read(D/f'mode{a}/aggregate-proof.json')['present_once']for a in range(2))
for a in range(2):
 log=(D/f'mode{a}/engine-final.log').read_text();markers=re.findall(r'^strata QSA prefill tiled: enabled=(0|1) nq=256 idx=4x128 R4 stride=50002 active=4097\.\.50001 first_active=(\d+) tile=8 key_wg=256 SG32$',log,re.M)
 checks[f'final_QSA_marker_{a}']=markers==[('0','4097')]
 value_marker='strata: prefill attention value pipeline selected (nq32 cap2051 INT8 page4)'
 checks[f'final_value_HOST_marker_{a}']=[x for x in log.splitlines()if 'prefill attention value pipeline selected'in x]==([value_marker]if a else[])
 checks[f'common_Q6_off_{a}']=re.findall(r'^strata Q6 center: .*$',log,re.M)==['strata Q6 center: enabled=0 C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240']
matched=all(checks.values());summary=None
if matched:
 summary={case:{k:[{'median':statistics.median(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-')),'range':[min(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-')),max(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-'))]}for a in range(2)]for k in ['native_prompt_tps','native_prompt_ms','ttft_s','wall_s']}for case in ['p4096','p32768']}
report={'status':'PASS_MATCHED_DISCRIMINATOR'if matched else'FAILED_OR_UNMATCHED','checks':checks,'same_binary':True,'warm_coherence_only':warm,'value_first_emission_requests':[read(D/f'mode{a}/prefill/complete.json')['first_value_emission_request']for a in range(2)],'prefill_requests':results,'timed_summary':summary,'primary_metric':'native prompt ms/tok-s with client TTFT; three fixed-order samples per shape, not independent significance/p95','normalization':['STRATA_ATTN_VALUE_PIPELINE0/1','arm-local log/container/service names'],'activation_scope':'per-request observed value HOST marker; first emission required at FIRST4K, ON once throughout all prefills; actual resident-mode0/full-width source domain, plus fixed QSA0 marker + actual native part bounds; C12288 subdivisions source-derived, no independent GPU dispatch/call counts','SSE_scope':'content events may contain multiple tokens; not token ITL','no_promotion_claim':True}
(D/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='ascii');print(json.dumps({'status':report['status'],'checks':checks}));raise SystemExit(0 if matched else 2)
