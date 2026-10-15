#!/usr/bin/env python3
"""ONE bounded small API host-summary parse per arm; no raw trace."""
import argparse,collections,csv,hashlib,json,math,re
from pathlib import Path
P=Path(__file__).resolve().parent
API=re.compile(r'(?:ze|zet|zel|zex)[A-Z][A-Za-z0-9_]*')
HEADER=['Function','Calls','Time (ns)','Time (%)','Average (ns)','Min (ns)','Max (ns)']
def read(p):return json.loads(p.read_text())
def save(p,v):
 with p.open('x',encoding='ascii')as f:json.dump(v,f,indent=2,ensure_ascii=True);f.write('\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def emitted(r):return ''.join(c.get('choices',[{}])[0].get('delta',{}).get('content')or''for c in r['chunks']if c.get('choices'))
def family(name):
 if name in ['zeCommandListImmediateAppendCommandListsExp','zeCommandQueueExecuteCommandLists','zeCommandListAppendGraphExp','zeCommandListAppendGraphExt']:return 'graph_or_command_list_submission'
 if re.search(r'Execute|Submit|ImmediateAppend|AppendGraph',name):return 'UNREVIEWED_SUBMISSION_LIKE_API'
 if re.search(r'HostSynchronize|EventQueryStatus',name):return 'host_wait_or_status_poll'
 if name.startswith('zeCommandListAppendLaunch'):return 'kernel_append_immediate_or_recorded_unknown'
 if name.startswith('zeCommandListAppendMemory'):return 'memory_copy_fill_append'
 if name.startswith('zeCommandListAppend'):return 'other_command_append'
 if re.search(r'Graph|CommandList(?:Create|Close|Destroy|Reset|Clone|Finalize)',name):return 'graph_or_command_list_lifecycle'
 if re.search(r'MemAlloc|MemFree|MakeMemoryResident|EvictMemory',name):return 'memory_lifetime_residency'
 if name.startswith('zeKernel'):return 'kernel_setup_query'
 return 'other_host_API_retained'
def summary_file(path):
 raw=path.read_bytes();assert 0<len(raw)<=2_000_000
 api={};header=False;seen_header=0;section=0;backend=0;total_api=None;execution=None
 for lineno,line in enumerate(raw.decode('ascii').splitlines(),1):
  text=line.strip()
  if not text:header=False;continue
  cells=[s.strip()for s in next(csv.reader([line],skipinitialspace=True,strict=True))]
  if cells[0]=='Function':
   assert cells==HEADER,'HOLD malformed recognized API header'
   seen_header+=1;assert seen_header==1,'HOLD duplicate API table'
   header=True;continue
  if header:
   assert len(cells)==len(HEADER),'HOLD malformed recognized table row'
   name=cells[0];assert API.fullmatch(name)and name not in api,'HOLD malformed/duplicate API row'
   assert re.fullmatch('[0-9]+',cells[1])and int(cells[1])>0,'HOLD malformed/nonpositive Calls'
   assert all(re.fullmatch('[0-9]+',cells[i])for i in [2,4,5,6]),'HOLD malformed time fields'
   count,total,avg,lo,hi=[int(cells[i])for i in [1,2,4,5,6]]
   percent=float(cells[3]);assert math.isfinite(percent)and 0<=percent<=100.000001
   assert total//count==avg and 0<=lo<=avg<=hi and hi<=total,'HOLD inconsistent API row statistics'
   api[name]={'calls':count,'time_ns':total,'time_percent':percent,'average_ns':avg,'min_ns':lo,'max_ns':hi};continue
  if text=='=== API Timing Summary ===':section+=1;continue
  if text=='== L0 Backend ==':backend+=1;continue
  m=re.fullmatch(r'Total Execution Time \(ns\):\s+([0-9]+)',text)
  if m:
   assert execution is None;execution=int(m[1]);continue
  m=re.fullmatch(r'Total API Time for L0 backend \(ns\):\s+([0-9]+)',text)
  if m:
   assert total_api is None;total_api=int(m[1]);continue
  raise AssertionError(('HOLD unexpected summary line/section',lineno,text))
 assert section==seen_header==backend==1 and execution is not None and api and raw.endswith(b'\n\n')and total_api==sum(v['time_ns']for v in api.values()),'HOLD incomplete API summary'
 return {k:v['calls']for k,v in api.items()},{'path':str(path),'bytes':len(raw),'sha256':sha(raw)},{'API_statistics':api,'total_API_time_ns':total_api,'total_execution_time_ns':execution,'scope':'Instrumented overlapping host API statistics, not GPU time/model speed; total_execution_time_ns includes whole process startup/paused warm/teardown'}
def one(arm):
 out=P/f'mode{arm}';assert not(out/'api-counts.json').exists(),'One parse only; no automatic reparsing'
 save(out/'analysis-start.json',{'status':'ONE_BOUNDED_SMALL_HOST_SUMMARY_PARSE_STARTED_NO_RETRY'})
 closure=read(out/'closure.json');assert closure['status']=='PASS_NORMAL_QUIT_FULL_HEALTH_SUMMARY_UNPARSED'
 assert read(out/'collection.json')['status']=='PASS_PAIR_NATIVE_CLOSURE_AND_CONTROLS_SUMMARY_UNPARSED'
 assert read(out/'native-counter-closure.json')['status']=='PASS_IDLE_AND_EXACT_NATIVE_SUMMARY_CLOSURE'
 for action in ['resume','pause','stop']:assert read(out/f'unitrace-{action}.json')['exit']==0
 normal=read(out/'normal-stop.json');assert normal['status']=='PASS_NORMAL_TEARDOWN'and normal['lease_wrapper_exit']==0
 removal=read(out/'container-removal.json');assert removal['full_cid_absent']and removal['name_absent']and removal['docker_ps_exit']==0
 assert all(v['exit']==0 for v in read(out/'post-health.json').values())and set(read(out/'post-health.json'))=={'card','collective'}
 assert read(out/'aggregate-proof.json')['present_once']
 pid=read(out/'native-process-identity.json')[0]['pid']
 summary=Path(closure['summary']);assert summary.parent==out/'trace'and summary.stat().st_size==closure['summary_bytes']
 assert summary.name==f'timing.{pid}.csv'and set((out/'trace').iterdir())=={summary},'HOLD extra/misattributed host output'
 counts,summary_pin,statistics=summary_file(summary)
 assert counts.get('zeCommandListImmediateAppendCommandListsExp',0)>0,'HOLD expected actual main graph submission API absent'
 families={}
 for name,count in sorted(counts.items()):families.setdefault(family(name),{})[name]=count
 assert 'UNREVIEWED_SUBMISSION_LIKE_API'not in families,'HOLD unknown submission family; root source review required'
 for name in ['server.log','engine-final.log']:
  text=(out/name).read_text()
  assert not re.search(r'\[ERROR\]|DEVICE_LOST|lost .*events|dropped .*events|buffer overflow|truncated trace',text,re.I),'HOLD profiler/runtime diagnostics'
 result={'status':'PASS_BOUNDED_MAIN_HOST_API_SUMMARY_COUNTS_ONLY','summary':summary_pin,'all_host_API_statistics':statistics,'all_API_counts':dict(sorted(counts.items())),'known_submission_API_counts':{n:counts.get(n,0)for n in ['zeCommandListImmediateAppendCommandListsExp','zeCommandQueueExecuteCommandLists','zeCommandListAppendGraphExp','zeCommandListAppendGraphExt']},'API_families':families,'API_calls':sum(counts.values()),'native':read(out/'native-counter-closure.json')['counts'],'first_host_selected_key':read(out/'coding/profile/pair.json')['first_host_selected_T0_T1'],'collection_scope':'Exactly one resumed107/64 pair plus idle/native-summary closure; paused256 warm excluded by actual retained callback gates; admission/capture/solo/tail retained','collection_complete_evidence':['All three session controls exit0.','Byte-offset native summaries exactly match completed requests and remain stable while HTTP slots idle.','Normal QUIT/fullCID+name gone/wrapper0 and pinned per-card/collective posthealth0.','One process-specific complete host summary; strict table/count/time/header-total closure; source/tool/binary identities rechecked.'],'independent_API_event_trace':False,'instrumented_speed_qualified':False,'all_depth2_calls_observed':False,'all_replayed_keys_observed':False,'reconciled_5_times_bt_pair_claim':False,'limits':['Counts are the actual retained collector host-summary counts, without an independent per-event mirror.','Host API invocation counts, not GPU kernel dispatch/traffic/wait causality.','Kernel/memory append calls may record a graph or act on an immediate list; no argument handles are logged.','Structural summary closure and normal teardown do not establish uninstrumented scheduler equivalence.','bt_pair does not log max_steps for every call. Total delta cannot alone identify all depth2 graph savings.']}
 save(out/'api-counts.json',result);print(json.dumps({'arm':arm,'status':result['status'],'API_calls':sum(counts.values())}))

def compare():
 inputs=read(P/'serving-inputs.json');assert inputs['same_binary']and inputs['arms']['0']==inputs['arms']['1']
 results=[read(P/f'mode{a}/api-counts.json')for a in range(2)];assert all(x['status']=='PASS_BOUNDED_MAIN_HOST_API_SUMMARY_COUNTS_ONLY'for x in results)
 normalized=[]
 for arm in range(2):
  c=read(P/f'mode{arm}/config.json');assert c['env']['STRATA_BATCH_MTP_PAIR_GRAPH']==str(arm)and c['env']['STRATA_IQ2XS_MULTI']=='0'
  for k,v in inputs['fixed_work_args'].items():assert c['args'].count(k)==1 and c['args'][c['args'].index(k)+1]==v
  del c['log'];c['exe']=inputs['arms'][str(arm)]['exe'];del c['env']['STRATA_BATCH_MTP_PAIR_GRAPH'];normalized.append(c)
 assert normalized[0]==normalized[1]
 for phase,cap in [('warm',256),('profile',64)]:
  pairs=[read(P/f'mode{a}/coding/{phase}/pair.json')for a in range(2)]
  assert all(p['output_tokens_per_stream']==cap and p['admission_gate']['status']=='PASS_NATIVE_PROMPT_START_ORDER_AND_CLIENT_OVERLAP'and p['client_overlap_ns']>0 for p in pairs)
  assert all(pairs[0][k]==pairs[1][k]for k in ['work_timings','native_windows','paired','solo','native_window_rows','logged_capture_layouts'])
  assert pairs[0]['pair_graph_host_selection_marker']is None and pairs[1]['pair_graph_host_selection_marker']is not None
  for slot in range(2):
   x,y=[read(P/f'mode{a}/coding/{phase}/coding-{slot}.json')for a in range(2)]
   assert x['request']==y['request']and x['request']['strata_tune']['spec_min_p']==0
   assert x['text']==y['text']==emitted(x)==emitted(y)and x['usage']==y['usage']and x['usage']['completion_tokens']==cap and x['usage']['prompt_tokens']==107
  if phase=='profile':assert results[0]['native']==results[1]['native']
 provenance=[read(P/f'mode{a}/native-process-identity.json')[0]for a in range(2)]
 assert all(provenance[0][k]==provenance[1][k]for k in ['exe','sha256','loader_environment','loaded_libstdcxx','loaded_unitrace'])
 names=sorted(set(results[0]['all_API_counts'])|set(results[1]['all_API_counts'])|set(results[0]['known_submission_API_counts']))
 deltas={n:{'OFF':results[0]['all_API_counts'].get(n,0),'ON':results[1]['all_API_counts'].get(n,0),'ON_minus_OFF':results[1]['all_API_counts'].get(n,0)-results[0]['all_API_counts'].get(n,0)}for n in names}
 save(P/'comparison.json',{'status':'PASS_MATCHED_NATIVE_WORK_AND_OBSERVED_MAIN_API_COUNTS_ROOT_ATTRIBUTION_REQUIRED','all_API_deltas':deltas,'submission_API_deltas':{n:v for n,v in deltas.items()if family(n)=='graph_or_command_list_submission'},'native':results[0]['native'],'source_only_depth2_launch_calls':{'OFF':6,'ON':1},'eligible_depth2_population_observed':False,'5_times_bt_pair_reconciliation_allowed':False,'speed_qualified':False,'attribution':'Observed main collected API totals; prefill/admission/capture/solo/tail and all max_steps/replay keys remain unresolved. Root decides any further counter/source check before code changes.'})
 print('PASS matched source/native work and actual API totals; root attribution required; no speed claim.')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=['0','1','compare']);a=ap.parse_args()
 try:
  if a.arm=='compare':compare()
  else:one(a.arm)
 except Exception as e:
  hold=(P if a.arm=='compare'else P/f'mode{a.arm}')/'analysis-HOLD.json'
  if not hold.exists():save(hold,{'status':'HOLD_ROOT_REVIEW_NO_RETRY','type':type(e).__name__,'detail':str(e),'speed_qualified':False})
  raise
