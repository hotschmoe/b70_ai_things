#!/usr/bin/env python3
"""S113 NEW fixed-policy discriminator; partial/mismatched evidence means NULL summary."""
import hashlib,json,math,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def content(r):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content')or''for c in r['chunks']if c.get('choices'))
WORK=['cache_n','prompt_n','predicted_n','draft_n','draft_n_accepted']
def work(r):
 assert all(k in r['timings']and isinstance(r['timings'][k],(int,float))and not isinstance(r['timings'][k],bool)and math.isfinite(r['timings'][k])and r['timings'][k]>=0 for k in WORK)
 return{k:r['timings'][k]for k in WORK}
def row_checks(x,y):
 return {'request_equal':x['request']==y['request'],'full_text_SSE_equal':x['text']==y['text']==content(x)==content(y),'usage_equal':x['usage']==y['usage'],'native_work_equal':work(x)==work(y),'HTTP_fixed_minp':x['request']['strata_tune']['spec_min_p']==y['request']['strata_tune']['spec_min_p']==0}
def save(x):(D/'comparison.json').write_text(json.dumps(x,indent=2)+'\n',encoding='ascii')
required=[D/'serving-inputs.json']
for arm in [0,1]:
 folder=D/f'mode{arm}';required += [folder/n for n in ['config.json','health-ready.json','container-ready.json','native-process-identity.json','warm4k/warm4k.json','warm4k/warm4k-verdict.json','warm4k/complete.json','prefill/complete.json','functional/verdict.json','normal-stop.json','container-removal.json','post-health.json','aggregate-proof.json','engine-final.log']]
 for case in ['p4096','p32768']:
  for repeat in range(3):required += [folder/'prefill'/f'{case}-{repeat}{suffix}.json'for suffix in ['', '-verdict']]
missing=[str(f)for f in required if not f.is_file()]
if missing:
 save({'status':'FAILED_OR_UNMATCHED','missing':missing,'timed_summary':None,'normalization':None,'scope':'No requests/retry/continuation authorized by this comparator.'});raise SystemExit(2)
inputs=read(D/'serving-inputs.json');assert inputs['same_binary']and inputs['arms']['0']==inputs['arms']['1']
configs=[read(D/f'mode{a}/config.json')for a in range(2)];normalized=[]
fixed={'STRATA_PROMPT_ATTN_XMX':'0','STRATA_KV_ROT':'0','STRATA_QSA_PREFILL_TILED':'0','STRATA_Q6_CENTER':'0','STRATA_Q6_A4':'0','STRATA_IQ2XS_MULTI':'0','STRATA_BATCH_MTP_PAIR_GRAPH':'0','STRATA_EXPERT_SPLIT':'0','STRATA_GU_LANES':'8','STRATA_MMVQ_WIDE':'1','STRATA_MMVQ_WIDE_MIN':'0','STRATA_MMVQ_WIDE_MAXC':'8','STRATA_Q6_PACKED':'0','STRATA_HEAD_Q5K_RPW2':'0','STRATA_BATCH_MTP_PLE_PREFETCH':'0','STRATA_TRACE':'1','STRATA_MMVQ_WIDE_K':'1','STRATA_MMVQ_WIDE_K_RPW':'1','STRATA_MMVQ_WIDE_RPW':'1','STRATA_MMVQ_A2':'1','STRATA_MMVQ_PF':'0','STRATA_BATCH_PREFILL_QUANTUM':'0','STRATA_PREFILL_FIRST':'0'}
for a,c in enumerate(configs):
 assert c['exe']==inputs['arms'][str(a)]['exe']and c['log']==str(D/f'mode{a}/engine.log')and c['env']['STRATA_ATTN_VALUE_PIPELINE']==str(a)
 with Path(c['exe']).open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==inputs['arms'][str(a)]['sha256']
 assert all(c['env'][k]==v for k,v in fixed.items())and'STRATA_BATCH_COMMIT_OVERLAP'not in c['env']
 assert all(k not in c['env']for k in ['STRATA_HOST_BOUNDARY','STRATA_VERIFY_PROFILE','STRATA_PREFILL_HOST_BOUNDARY','STRATA_PREFILL_TIMING','STRATA_PREFILL_SYNC','STRATA_VERIFY_TRACE','STRATA_VERIFY_EAGER','STRATA_ATTN_LANECELL'])
 for key,value in inputs['fixed_work_args'].items():assert c['args'].count(key)==1 and c['args'][c['args'].index(key)+1]==value
 n=json.loads(json.dumps(c));del n['log'];del n['env']['STRATA_ATTN_VALUE_PIPELINE'];normalized.append(n)
assert normalized[0]==normalized[1]
checks={'identity_image_native':all(read(D/f'mode{a}/health-ready.json')['model']=='hotschmoe-dd'and read(D/f'mode{a}/container-ready.json')['Image']==inputs['image']and len(read(D/f'mode{a}/native-process-identity.json'))==1 for a in range(2))}
results=[]
for name,relative,iswarm in [('warm4k','warm4k',True)]+[(f'{case}-{i}','prefill',False)for case in ['p4096','p32768']for i in range(3)]:
 rows=[read(D/f'mode{a}'/relative/(name+'.json'))for a in range(2)];v=[read(D/f'mode{a}'/relative/(name+'-verdict.json'))for a in range(2)];c=row_checks(*rows)
 c['local_gates']=all(z['status']=='PASS_LOCAL_GATES'for z in v)
 c['fresh_and_output']=all(work(r)['cache_n']<=64 and r['usage']['completion_tokens']==256 for r in rows)
 if not iswarm:c['timed_prompt_count']=all(r['usage']['prompt_tokens']==(4059 if name.startswith('p4096')else 32728)for r in rows)
 c['parts_and_captures']=all(v[0][k]==v[1][k]for k in ['native_part_intervals','capture_stats'])
 c['value_host_marker']=v[0]['VALUE_marker']['count']==0 and v[1]['VALUE_marker']['count']==1 and v[1]['VALUE_marker']['first_emission_request']=='warm4k'
 checks[name]=all(c.values());results.append({'name':name,'unmeasured_warm':iswarm,'matched':checks[name],'checks':c,'work':[work(r)for r in rows],'usage':[r['usage']for r in rows],'observed_capture_stats':[z['capture_stats']for z in v],'VALUE_HOST_observations':[z['VALUE_marker']for z in v],'metrics':[{k:r[k]for k in ['ttft_s','wall_s','client_decode_tps','content_event_gap_ms']}|{'native_prompt_tps':r['timings']['prompt_per_second'],'native_prompt_ms':r['timings']['prompt_ms'],'output_sha256':r['output_sha256']}for r in rows]})
checks['functionals']=all(read(D/f'mode{a}/functional/verdict.json')['status']=='PASS'for a in range(2))
checks['normal_and_health']=all(read(D/f'mode{a}/normal-stop.json')['status']=='PASS_NORMAL_TEARDOWN'and read(D/f'mode{a}/normal-stop.json')['lease_wrapper_exit']==0 and read(D/f'mode{a}/normal-stop.json')['prestop_state']['Running']is True and read(D/f'mode{a}/normal-stop.json')['prestop_state']['OOMKilled']is False and all(v['exit']==0 for v in read(D/f'mode{a}/post-health.json').values())for a in range(2))
checks['fullCID_NAME_absence']=all(read(D/f'mode{a}/container-removal.json')['docker_ps_exit']==0 and read(D/f'mode{a}/container-removal.json')['full_cid_absent']and read(D/f'mode{a}/container-removal.json')['name_absent']and read(D/f'mode{a}/container-removal.json')['created_cid']==read(D/f'mode{a}/container-ready.json')['Id']for a in range(2))
checks['normalQUIT_diag']=all(read(D/f'mode{a}/aggregate-proof.json')['present_once']for a in range(2))
startup=[]
for arm in [0,1]:
 log=(D/f'mode{arm}/engine-final.log').read_text();value=[x for x in log.splitlines()if'prefill attention value pipeline selected'in x]
 checks[f'final_value_HOST_{arm}']=value==(['strata: prefill attention value pipeline selected (nq32 cap2051 INT8 page4)']if arm else[])
 checks[f'QSA_OFF_{arm}']=re.findall(r'^strata QSA prefill tiled: enabled=(0|1).*first_active=(\d+) tile=8 key_wg=256 SG32$',log,re.M)==[('0','4097')]
 checks[f'unrelated_HOST_absent_{arm}']=not re.search(r'^strata (?:Q6 A4:|IQ2_XS multi:|MTP pair graph:)',log,re.M)
 center=re.findall(r'^strata Q6 center: .*$',log,re.M);checks[f'CENTER_OFF_if_reached_{arm}']=center in [[],['strata Q6 center: enabled=0 C6 A2 R1 SG32 PF0 shapes=6144x2560,2560x10240']]
 offset=read(D/f'mode{arm}/warm4k/warm4k-start.json')['byte_offset'];startup.append(re.findall(r'^strata verify: captured .*$',(D/f'mode{arm}/engine-final.log').read_bytes()[:offset].decode(),re.M))
checks['startup_captures_equal']=startup[0]==startup[1]
provenance=[read(D/f'mode{a}/native-process-identity.json')[0]for a in range(2)]
checks['loader_provenance']=all(x['proc_maps']and x['proc_maps_sha256']==hashlib.sha256(x['proc_maps'].encode()).hexdigest()and len(x['loaded_libstdcxx'])==1 for x in provenance)and all(provenance[0][k]==provenance[1][k]for k in ['loader_environment','loaded_libstdcxx'])
matched=all(checks.values());summary=None
if matched:
 summary={case:{k:[{'median':statistics.median(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-')),'range':[min(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-')),max(x['metrics'][a][k]for x in results if x['name'].startswith(case+'-'))]}for a in [0,1]]for k in ['native_prompt_tps','native_prompt_ms','ttft_s','wall_s']}for case in ['p4096','p32768']}
report={'status':'PASS_MATCHED_DISCRIMINATOR'if matched else'FAILED_OR_UNMATCHED','checks':checks,'same_binary':True,'requests':results,'timed_summary':summary,'normalization':['attention-value flag0/1','arm-local names/logs'],'scope':'NEW fixed-policy/4K-warm protocol; no S94 retrospective clearance or proof of its probability cause. Three fixed-order timed samples per shape, not independent replication.','activation_scope':'Once eligible value HOST marker by unmeasured4K warm, not GPU dispatch or all call counts.','SSE_scope':'Content events may contain multiple tokens, not token ITL.','no_cache_campaign_or_promotion':True};save(report);print(json.dumps({'status':report['status'],'checks':checks}));raise SystemExit(0 if matched else 2)
