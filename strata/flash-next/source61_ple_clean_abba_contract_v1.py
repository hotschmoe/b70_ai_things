"""Natural serial histories, recorded references and descriptive ABBA timings."""
import math
from pathlib import Path
import run_natural_p2048_native61_v1 as natural
from serial37_canonical_json_v3 import canonical
from source60_loaded_fairness_observations_v1 import gap_summary
c=natural.c;require=natural.require
B17=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f17-source37-20261010')
REFERENCES=(('natural-p2048-copy-engine-v2-public-v1.json','2aaff799a6c1f319c8cdfa5560377b9e1c4a4fbed3b95b9942855f05a9bdeed4','natural-p2048-copy-engine-v2-run-v1','beab962fadfa8de324462573a76726ea24643ccb70e6f915311bcc397a00263b'),('source56-hc-prefill-tile4-natural-v1-public-v1.json','4282bfdacff740a586cd2556333975d3a9409e6a456af51e1ae73986b0dde7f2','source56-hc-prefill-tile4-natural-v1-run-v1','33b5bc24fbd30b2ebcb37cb12f5bf40fbde0c0d2f085ea1b6f4145ebf7f1deb1'))
OFF_ENV=('STRATA_TRACE','STRATA_PREFILL_HOST60','STRATA_BATCH_PREFILL_LAYER_COOPERATE','STRATA_BATCH_PREFILL_SINGLE_WINDOW','STRATA_CRITICAL_PATH_TRACE','STRATA_FIDELITY_DIAG','STRATA_PREFIX_DIAG','STRATA_PREFIX_LIFECYCLE_DIAG','STRATA_BATCH_FIDELITY_DIAG','STRATA_FULL_CACHE_OBSERVER38','STRATA_COPY_ENGINE','STRATA_PF_GROUPED_F16','STRATA_PF_ESIMD_F16')

def reference_binding(expected):
 results=[]
 for filename,digest,dirname,report_sha in REFERENCES:
  public_path=B17/filename;root=B17/dirname
  require(c.sha(public_path)==digest and c.sha(root/'report.json')==report_sha,'Pinned recorded natural reference changed')
  public=c.read(public_path);report=c.read(root/'report.json');require(public['passed'] is True and report['passed'] is True and public['report_sha256']==report_sha,'Recorded natural reference did not close')
  for arm in public['arms']:
   phase=arm['phase'];p=root/phase/'recipe.json';require(c.sha(p)==report['artifacts'][phase+'/recipe.json'],'Recorded reference input recipe changed')
   rows=c.read(p)['requests'];views=arm['natural_observations'];require(len(rows)==len(views)==len(expected)==6,'Recorded six-role repeat census required')
   for want,prior,view in zip(expected,rows,views):
    require(want['ids']==prior['ids'] and want['role']==prior['role']==view['case_id'] and want['ordinal']==prior['ordinal']==view['ordinal'] and want['max_new']==prior['max_new']==512 and prior['fresh']==1,'Recorded reference authentic prompt association differs')
   results.append(views)
 return results

def requests(rows,expected):
 views=natural.requests(rows,expected)
 for reference in reference_binding(expected):
  for view,prior in zip(views,reference):require(view['output_ids']==prior['output_ids'] and view['done_signature'][3]==prior['finish'],'Current complete natural history differs from admitted52/56')
 return views

def matched_recipes(recipes):
 require(len(recipes)==4,'Four actual ABBA owners required')
 normalized=[]
 for recipe,mode in zip(recipes,('mmap','direct','direct','mmap')):
  argv=recipe['native_argv'];env=recipe['native_env']
  require(recipe['PLE_mode']==mode and argv.count('--ple-io')==argv.count('--ple-row-cache')==argv.count('--ple-inflight')==1 and argv[argv.index('--ple-io')+1]==mode and argv[argv.index('--ple-row-cache')+1]=='65536' and argv[argv.index('--ple-inflight')+1]=='256' and '--ple-sync-submit' not in argv,'Exact IO mode/cache/default worker/inflight required')
  require(argv[argv.index('--batch')+1]==argv[argv.index('--prompt-cache')+1]=='0' and argv[argv.index('--prefill')+1]=='512' and argv[argv.index('--max-context')+1]=='4096' and argv[argv.index('--spec')+1]=='2' and '--stop-eos' in argv and not any(x in argv for x in ('--mtp','--batch-mtp','--keep-canonical')),'Clean serial no-cache P2048/GEN512/noMTP scope required')
  require(all(env[k]=='0' for k in OFF_ENV) and env['STRATA_VERIFY_MIRROR_STAGE_T1']=='1' and 'STRATA_BATCH_DECODE_SHARE' not in env and 'SYCL_UR_TRACE' not in env and recipe['PTI_enabled'] is False and recipe['observer_enabled'] is False,'Clean baseline must disable instrument/cooperation/cap/copy/GPU trace')
  a=list(argv);a[a.index('--ple-io')+1]='PLE_MODE';normalized.append((a,env,recipe['requests'],recipe['expert_profile_binding'],recipe['natural_settings']))
 require(all(canonical(x)==canonical(normalized[0]) for x in normalized),'Only PLE mode may differ across ABBA arms')
 return True

def equivalence(*arms):
 require(len(arms)==4 and all(len(x)==6 for x in arms) and all(canonical(x)==canonical(arms[0]) for x in arms),'Four complete natural ABBA histories must match')
 return dict(bounded61_clean_PLE_ABBA_histories_passed=True,matched_arms_outputs_equal=True,recorded52_and56_histories_equal=True,scope='Four fresh owned serial processes, three useful P2048 roles with fresh repeats; no fullmath/concurrency/task-success/shelf claim',API_qualified=False,original_fullmath_qualified=False,task_completion_qualified=False,shelf_qualified=False,speed_promotion_qualified=False)

def observations(rows,recipe):
 requests(rows,recipe['requests']);result=[]
 for row,want in zip(rows,recipe['requests']):
  raw=row['raw'];done=raw['done'].split();pairs=[(int(line.split()[1]),stamp) for line,stamp in zip(raw['stdout'],raw['stdout_seconds']) if line.startswith('T ')]
  useful=[(token,stamp) for token,stamp in pairs if token not in (248044,248046)]
  require(useful and all(t not in (248044,248046) for t,_ in pairs[:-1]),'Useful natural output and terminating EOS required')
  require((done[5]=='length' and len(pairs)==512) or (done[5]=='stop' and pairs[-1][0] in (248044,248046)),'Natural EOS/length finish differs')
  pp=float(done[3]);decode=float(done[4]);require(math.isfinite(pp) and math.isfinite(decode) and pp>0 and decode>0,'Finite actual native PP/decode intervals required')
  gaps=gap_summary([stamp for _,stamp in useful]);gaps['token_receive_relative_seconds']=gaps.pop('token_receive_monotonic_seconds');gaps['clock_basis']='Original client monotonic receive minus request start; no cross-process clock alignment'
  result.append(dict(ordinal=row['ordinal'],case_id=want['role'],repeat=want['repeat'],process_position=want['process_temperature'],input_position=want['input_temperature'],input_tokens=2048,emitted_tokens=len(pairs),useful_non_EOS_tokens=len(useful),finish=done[5],task_completion_qualified=False,native_PP_ms=pp,native_PP_tok_s=2048*1000/pp,native_decode_ms=decode,native_reported_emitted_over_decode_tok_s=len(pairs)*1000/decode,native_decode_rate_denominator='emitted tokens / native DONE decode interval; may include first-token accounting; no sustained claim',root_client_TTFT_ms=useful[0][1]*1000,root_client_completion_ms=raw['stdout_seconds'][-1]*1000,root_token_gaps=gaps,output_ids=raw['output_ids']))
 return result
