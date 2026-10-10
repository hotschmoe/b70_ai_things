"""NEW positive-cache profile and authentic CPU fixture checkpoint admission."""
import json,hashlib,re
from pathlib import Path
from api_fresh_migration_contract_v1 import preflight as capability_profile,PROFILE_ENV,actual_startup

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE_PLAN=HERE/'api-cache-positive-source-plan-v2.json'
FIXTURE_ROOT=Path('/mnt/vm_8tb/b70/results/flashnext_udq4xl_20261008/f16-source35-20261009/cache-positive-tokenizer-fixture-v2')
CHECKPOINT_ARGS={'--prompt-cache-root':'2048','--prompt-cache-every':'16384','--turn-token':'248045','--tail-role-token':'-1'}

def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def option(args,k):
 require(args.count(k)==1 and args.index(k)+1<len(args),'Exactly one explicit checkpoint option required '+k);return args[args.index(k)+1]

def checkpoint_nonreuse(fixtures,args,env):
 lock_checkpoint_defaults(args,env)
 require(all(option(args,k)==v for k,v in CHECKPOINT_ARGS.items()),'Exact source-derived checkpoint defaults required')
 require('--prompt-cache-tail' not in args and 'STRATA_CACHE_MESSAGE_BOUNDARY' not in env,'Optional tail/message checkpoint routes mustbe absent including0')
 require(option(args,'--prefill')=='64','Exact fixed64 prefill source partial-state bound required')
 rows=[]
 for phase in ('warm','target'):
  for i,row in enumerate(fixtures[phase]):
   ids=row['ids'];require(row['logical_index']==i and 2<len(ids)<1984 and all(type(t) is int and 0<=t<248320 for t in ids),'Actual fixture integer tokens/roster invalid')
   turns=[j for j,t in enumerate(ids) if t==248045];require(len(turns)==3 and turns[0]==0,'Exact system/user/assistant template role boundaries required')
   last=turns[-1];first=turns[1];require(first<2048 and len(ids)+64<16384,'Root/periodic boundary couldintroduce an earlier checkpoint')
   # Source checkpoint_at(L) captures accepted tokens ids[0:L], excluding token L.
   accepted=[last,len(ids)-1];require(min(accepted)>0,'Positive complete prompt boundaries required')
   rows.append({'phase':phase,'logical_index':i,'ids':ids,'candidate_turn_token_indices':turns,'possible_prompt_checkpoint_accepted_lengths':accepted,'minimum_accepted_checkpoint_length':min(accepted),'possible_first_partial_live_accepted_length':64,'minimum_source_reusable_accepted_length':min(64,min(accepted)),'root_checkpoint_absent_below2048':True,'periodic_checkpoint_absent_below16384':True})
 comparisons=[]
 for i,left in enumerate(rows):
  for right in rows[i+1:]:
   lcp=next((j for j,(a,b) in enumerate(zip(left['ids'],right['ids'])) if a!=b),min(len(left['ids']),len(right['ids'])))
   require(lcp<min(left['minimum_source_reusable_accepted_length'],right['minimum_source_reusable_accepted_length']),'Shared actual checkpoint prefix couldinvalidate exact initialzero claim')
   comparisons.append({'left':[left['phase'],left['logical_index']],'right':[right['phase'],right['logical_index']],'actual_lcp_tokens':lcp,'minimum_source_reusable_accepted_tokens':min(left['minimum_source_reusable_accepted_length'],right['minimum_source_reusable_accepted_length'])})
 return {'prefix_convention':'checkpoint_at(L) holds exactly accepted ids[0:L]; token at indexL is excluded','rows':[{k:v for k,v in r.items() if k!='ids'} for r in rows],'pairwise_actual_LCP':comparisons,'initial_zero_counters_runtime_readback_still_required':True,'actual_initial_nonreuse_runtime_qualified':False}

def fixture_binding():
 plan=read(SOURCE_PLAN);pin=plan['actual_tokenizer_fixture'];require(Path(pin['root']).resolve()==FIXTURE_ROOT.resolve(),'Exact genuine fixture root required')
 for name,digest in pin['files'].items():require(sha(FIXTURE_ROOT/name)==digest,'Actual tokenizer fixture evidence changed '+name)
 fixture=read(FIXTURE_ROOT/'fixtures.json');seed=read(FIXTURE_ROOT/'seed.snapshot.json');require(fixture['source_seed_sha256']==sha(FIXTURE_ROOT/'seed.snapshot.json')==sha(HERE/'batch-cache-positive-case2-source-seed-v1.json') and seed==read(HERE/'batch-cache-positive-case2-source-seed-v1.json'),'Authentic exact seed/template message identity differs')
 sdk=read(HERE/'current-ple-prompt35-engine-build-plan-v1.json');expected=sdk['expected_patched_source_sha256'];require(fixture['source_file_sha256']=={('/'+name):expected[name] for name in ('tools/strata_tokenizer.py','serve/frontend.py')},'Authentic original tokenizer/frontend differs from consumed locked source35')
 require(fixture['tokenizer_file_sha256']==read(HERE/'batch-numerical-case2-source35-v2.json')['tokenizer_sha256'],'Actual CPU tokenizer/template bytes differ from original source35 case identity')
 require(fixture['actual_GPU_touch'] is False and fixture['actual_model_payload_read'] is False and fixture['initial_checkpoint_nonreuse_qualified'] is False,'Tokenization cannottransfer runtime/cache authority')
 receipt=read(FIXTURE_ROOT/'receipt.json');state=receipt['state'];require(receipt['return_code']==0 and state['ExitCode']==0 and not state['Running'] and not state['OOMKilled'] and receipt['removed'] is True and receipt['devices']==[] and receipt['device_requests'] is None and receipt['image']==pin['image'],'Actual noGPU tokenizer terminal/image/ownedremoval absent')
 command=read(FIXTURE_ROOT/'command.json');require(command==pin['command'] and '--device' not in command and '--gpus' not in command and not any('.gguf' in w for w in command),'Actual noGPU/noGGUF/tokenizer-only recipe differs')
 require(command[command.index('--network')+1]=='none' and '--read-only' in command,'Tokenizer isolated readonly profile differs')
 for phase in ('warm','target'):
  require([r['messages'] for r in fixture['fixtures'][phase]]==seed['messages'][phase] and [r['logical_index'] for r in fixture['fixtures'][phase]]==list(range(2)),'Actual template/literal role/messages fixture differs')
 return fixture,{'fixture_root':str(FIXTURE_ROOT),'fixture_sha256':sha(FIXTURE_ROOT/'fixtures.json'),'evidence_files':pin['files'],'image':pin['image'],'authentic_original_tokenizer_and_template_CPU_only':True,'actual_initial_nonreuse_runtime_qualified':False}

def profile(plan):
 n=plan['slots'];zero=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(n)]+[{'logical_index':1,'role':'solo_migration','values':{'reused':0,'read_from':0,'reread_to':-1}}];capability_profile(plan['args'],plan['env'],{'strata_fresh':True},zero,n)
 require(n==2,'Authentic current positive fixture covers only2; new4/6 fixtures required')
 require(plan['API_warm_request_policy']=={'strata_fresh':True} and type(plan['API_warm_request_policy']['strata_fresh']) is bool and plan['API_target_request_policy']=={'strata_fresh':False} and type(plan['API_target_request_policy']['strata_fresh']) is bool,'Explicit warmfresh and targetcached policies required')
 expected=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(n)]+[{'logical_index':1,'role':'solo_migration','values':{'reused':'prompt_minus_one','read_from':'prompt_minus_one','reread_to':-1}}];require(plan['actual_counter_policy']==expected,'Exact initial0/0/-1 + last-live promptminusone/promptminusone/-1 preregistration required')
 fixture,binding=fixture_binding();proof=checkpoint_nonreuse(fixture['fixtures'],plan['args'],plan['env']);require(plan['messages']=={k:[r['messages'] for r in fixture['fixtures'][k]] for k in ('warm','target')} and plan['api_token_ids']=={k:[r['ids'] for r in fixture['fixtures'][k]] for k in ('warm','target')},'Authentic CPU template/IDs differ from positive plan')
 return {'fixture':binding,'source_checkpoint_nonreuse':proof,'actual_cached_state_handoff_qualified':False,'positive_actual_counter_and_full49_readback_required':True}

def alias(cfg,n,diagnostic,lane='source35'):
 require(lane=='source35','Positive source35 only');topo='gpu0' if cfg['env']['ZE_AFFINITY_MASK']=='0' else 'gpu0-1-split32-16';return 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-source35-C113-fp16kv-ctx2048-prefill64-batch'+str(n)+'-pcache3-chain25-pub27-fresh0-cachedhandoff-'+topo+'-diag24'+('on' if diagnostic else 'off')

def registry_gate(name):
 path=ROOT/'evals/configs/models.yaml';ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',path.read_text(),re.M);require(ids.count(name)==1,'Separate positive-cache alias mustbe explicitly registered beforeprepare/GPU/modelscan');from registry_cache_positive_association_v2 import association
 return {'path':str(path),'sha256':sha(path),'served_model_id':name,'append_only_association':association(path),'scope':'actual newpositive alias/currentglobal digest; prior/global associations separate'}

# NEW phase-scoped reset check: initial cold admission is real and retained.
def last_live_handoff(trace,rid,job,stages):
 require(type(rid) is int and rid>0 and job['rid']==rid and job['role']=='solo_migration' and len(job['ids'])>=2 and job['position']==len(job['ids'])-1 and job['token']==job['ids'][-1],'Exact actual migrated fullhead prefix job required')
 rows=[(i,line,fields(line)) for i,line in enumerate(trace.splitlines())]
 begins=[r for r in rows if r[1].startswith('SBF migration_begin ') and int(r[2]['rid'])==rid];require(len(begins)==1,'Actual unique logical migration begin required');begin_i,_,begin=begins[0]
 closes=[r for r in rows if r[1].startswith('SBF slot_closed ') and int(r[2]['rid'])==rid and r[2]['slot']==begin['source_slot'] and r[2]['slotgen']==begin['source_slotgen']]
 require(len(closes)==1 and closes[0][0]<begin_i and closes[0][2]['cancelled']==closes[0][2]['completed']=='1' and closes[0][2]['pid']==begin['pid'] and closes[0][2]['enginegen']==begin['enginegen'] and closes[0][2]['requestgen']==begin['requestgen'],'Actual source owner/closed slot generation differs')
 restored=[r for r in rows if r[1].startswith('BCHAIN restored ') and int(r[2]['rid'])==rid and r[2]['slot']==begin['source_slot'] and r[2]['slotgen']==begin['source_slotgen']]
 require(len(restored)==1 and restored[0][0]>begin_i,'Actual matching closed-slot transfer record required');restore_i,_,restore=restored[0]
 expected=len(job['ids'])-1;require(int(restore['tokens'])==expected and restore['active_checkpoint']=='0' and int(restore['points'])>0 and int(restore['host_bytes'])>0 and expected>0,'Positive last-live full-chain strictprefix restore required, not checkpoint fallback')
 resumes=[r for r in rows if r[1].startswith('SBF resume ') and int(r[2]['rid'])==rid and r[2]['slot']=='6'];require(len(resumes)==1 and resumes[0][0]>restore_i,'Actual positive main-session resume after realtransfer required');_,_,resume=resumes[0]
 require(all(resume[k]==begin[k] for k in ('pid','enginegen','requestgen','slotgen')) and int(resume['reused'])==int(resume['read_from'])==expected and resume['reread_to']=='-1','Positive actual0-reread counter/source identity differs')
 require(not any(i>=begin_i and line.startswith('BCPUBLIC reset ') and int(f['rid'])==rid for i,line,f in rows),'Cached migrated RID mustnot claim zero reset/recompute')
 require(int(begin['prompt'])==len(job['ids']) and int(begin['main_session'])==1,'Actual migration prompt/main-session shape differs')
 return {'source_slot':int(begin['source_slot']),'source_slot_generation':int(begin['source_slotgen']),'pid':int(begin['pid']),'engine_generation':int(begin['enginegen']),'request_generation':int(begin['requestgen']),'actual_reused':expected,'actual_read_from':expected,'actual_reread_to':-1,'actual_full_chain_last_live_restore_observed':True,'initial_cold_reset_before_migration_permitted':True,'all_stage_transfer_source_guards_required':stages,'full_cached_fresh_math_qualified':False,'stale_model_or_session_negative_runtime_qualified':False,'scope':'real positive last-live handoff lineage/restore/counters; exact full49 freshness/control still required'}

def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))
def actual_phase_legs(events,calls,fresh):
 legs={call:[] for call in calls};wanted='fresh='+str(int(fresh))
 for e in events:
  if e['kind']!='native_send' or e.get('call') not in legs or not e['line'].startswith(('GEN ','BGEN ')):continue
  words=e['line'].split();require(set(fields(e['line']))<= {'seed','rid','slotgen','fresh'} and fields(e['line']).get('seed')=='1','Only actual greedy seed1/RID/fresh native keys admitted');require([w for w in words if w.startswith('fresh=')]==[wanted] and not any(w.startswith('pin=') for w in words),'Actual positive-cache phase native policy differs');legs[e['call']].append(e['line'])
 require(all(legs.values()),'Every actual phase call requires native leg')
 return {'fresh':fresh,'calls':sorted(calls),'actual_native_leg_counts':{str(k):len(v) for k,v in legs.items()},'actual_cached_state_handoff_qualified':False}

def lock_checkpoint_defaults(args,env):
 for key,value in CHECKPOINT_ARGS.items():
  if key in args:require(args.count(key)==1 and option(args,key)==value,'Inherited checkpoint override refused '+key)
 require('--prompt-cache-tail' not in args and 'STRATA_CACHE_MESSAGE_BOUNDARY' not in env,'Inherited optional checkpoint route refused including0')
 for key in ('STRATA_SPLIT_SMALL_OWN','STRATA_SPLIT_SMALL_MAX','STRATA_PREFILL_EQUAL'):require(key not in env,'Inherited partial-prefill geometry option refused including0: '+key)
 if '--short-read' in args:require(option(args,'--short-read')=='64','Inherited short-read boundary override refused')
