"""Cheap capability-on/fresh-recompute preflight, no model or device access."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
SOURCE_PLAN=HERE/'api-fresh-recompute-migration-source-plan-v1.json'
REQUEST_POLICY={'strata_fresh':True}
PROFILE_ENV={'STRATA_BATCH_FULL_STATE_CHAIN':'1','STRATA_BATCH_PUBLIC_PREFIX':'1','STRATA_BATCH_CHAIN_MIB':'8192','STRATA_BATCH_TRANSFER_MIB':'512'}

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def value(args,key):
 require(args.count(key)==1 and args.index(key)+1<len(args),'Exact API migration option required '+key);return args[args.index(key)+1]

def preflight(args,env,policy,counters,slots):
 require(type(slots) is int and slots in (2,4,6),'Exact API2/4/6 capacity required');require(value(args,'--batch')==str(slots) and value(args,'--batch-groups')=='1','Migration requiresactualprivate group1 slots')
 require(value(args,'--prompt-cache')=='3','Fresh migration requires genuine slot_cache1/cache3; cache0 impossible beforeGPU')
 require(all(env.get(k)==v for k,v in PROFILE_ENV.items()),'Exact source35 full chain/public profile and explicit8192/512MiB budgets required')
 require(value(args,'--adapt-every')=='0' and value(args,'--adapt-swaps')=='0' and '--no-prefill-borrow' in args and value(args,'--kv')=='fp16' and value(args,'--conversation-cache-mib')=='0','Static fixedKV/noBorrow/nonparked profile required')
 require(not any(k in args for k in ('--mtp','--pipeline-windows','--kv-grow','--adapt-async','--peer-device')),'Migration noMTP/nonpipeline/fixedKV/static profile required')
 require(env.get('STRATA_KV_GROW','0') in ('','0'),'Inherited elasticKV invalidates full chain startup')
 if '--layer-split' in args:require(value(args,'--layer-split')=='32' and value(args,'--split-device')=='1','Exact distinct-device32/16 one-extra-stage profile required')
 else:require('--split-device' not in args,'Split device without admitted layer split refused')
 for key in ('STRATA_VERIFY_EAGER','STRATA_CKPT_REREAD','STRATA_STATE_HASH'):require(key not in env,'Migration cannotuse presence toggle including0: '+key)
 require(env.get('STRATA_PARALLEL_SOLO','1')!='0','Actual slot->main scheduling mustnotbe disabled')
 require(policy==REQUEST_POLICY and type(policy['strata_fresh']) is bool,'Exact explicitfresh APIbody required for everyphase')
 require(len(counters)==slots+1 and {(r['logical_index'],r['role']) for r in counters}=={(i,'admission') for i in range(slots)}|{(1,'solo_migration')},'Complete declared initial+solo counter roster required')
 require(all(r['values']=={'reused':0,'read_from':0,'reread_to':-1} for r in counters),'Fresh-recompute exact0/0/-1 policy cannotbe relaxed')
 return {'slot_cache_capability_expected':1,'fresh_reset_each_native_leg':True,'actual_cached_state_handoff_qualified':False,'migration_scope':'actual logical slot->main lineage/body withcompleteprefix reevaluation; no cachedstate reuse','actual_counter_policy_unchanged':True,'native_geometry_and_transfer_budget_startup_still_required':True,'geometry_budget_runtime_qualified':False}

def alias(cfg,slots,diagnostic):
 topology='gpu0' if cfg['env']['ZE_AFFINITY_MASK']=='0' else 'gpu0-1-split32-16';return 'qwen3.8-flash-next-Unsloth-UD-Q4_K_XL-strata-source35-C113-fp16kv-ctx2048-prefill64-batch'+str(slots)+'-pcache3-chain25-pub27-fresh1-recompute-'+topology+'-diag24'+('on' if diagnostic else 'off')

def registry_gate(name):
 path=ROOT/'evals/configs/models.yaml';ids=re.findall(r'^\s*served_model_id:\s*(\S+)\s*$',path.read_text(),re.M);require(ids.count(name)==1,'NEW cache3/fresh migration detailedalias mustbe registered exactlyonce BEFOREGPU/fullmodelscan; sourceproposal doesnoteditregistry');from registry_append_association_v1 import association
 return {'path':str(path),'sha256':sha(path),'served_model_id':name,'append_only_association':association(path)}

def actual_startup(trace,slots):
 rows=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in trace.splitlines() if line.startswith('BCHAIN startup ')]
 require(len(rows)==1,'Actual current native geometry/budget startup record required')
 r=rows[0];require(int(r['slots'])==slots and int(r['cap'])==3 and int(r['chain_limit'])==8192*(1<<20) and int(r['transfer_limit'])==512*(1<<20),'Actual native chain capacity/limit differs')
 point=int(r['point_bound']);required=int(r['required']);require(point>0 and required==point*((slots+3)*3+3) and required<=int(r['chain_limit']),'Actual native geometry-bound startup budget failed')
 return r

def actual_fresh_legs(events,expected_calls):
 legs={call:[] for call in expected_calls}
 for e in events:
  if e['kind']!='native_send' or e.get('call') not in legs:continue
  line=e['line'];words=line.split()
  if not words or words[0] not in ('GEN','BGEN'):continue
  require([w for w in words if w.startswith('fresh=')]==['fresh=1'] and not any(w.startswith('pin=') for w in words),'Every actual API native leg mustcarry exactlyfresh1 and no pin')
  legs[e['call']].append(line)
 require(all(legs.values()),'Every actual warm/target API call needs real fresh native send')
 return {'calls':sorted(legs),'leg_counts':{str(k):len(v) for k,v in legs.items()},'all_actual_native_legs_fresh1':True,'actual_cached_state_handoff_qualified':False}

def actual_resets(trace,events,calls,stages):
 expected={}
 for e in events:
  if e['kind']!='native_send' or e.get('call') not in calls or not e['line'].startswith(('GEN ','BGEN ')):continue
  fields=dict(re.findall(r'(\w+)=([^ ]+)',e['line']));require('rid' in fields,'Actual native leg mustcarry real RID');rid=int(fields['rid']);expected[rid]=expected.get(rid,0)+1
 observed={}
 for line in trace.splitlines():
  if not line.startswith('BCPUBLIC reset '):continue
  f=dict(re.findall(r'(\w+)=([^ ]+)',line));rid=int(f['rid']);require(rid in expected and int(f['stages'])==stages and f['fresh']=='1' and f['pin_present']=='0' and f['lookup_ceiling']==f['read_from']=='0','Actual all-stage reset source record differs')
  observed[rid]=observed.get(rid,0)+1
 require(expected and observed==expected,'Every actual native leg requires matching all-stage reset record')
 return {'actual_reset_counts':{str(k):v for k,v in observed.items()},'all_stage_zero_resets_observed':True,'stage_count':stages}
