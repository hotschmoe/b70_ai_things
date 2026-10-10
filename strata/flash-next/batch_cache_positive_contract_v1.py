"""Source-only positive last-live slot-to-main handoff admission, no runtime proof."""
import re
from api_fresh_migration_contract_v1 import preflight as fresh_profile,REQUEST_POLICY,actual_startup

def require(ok,msg):
 if not ok:raise ValueError(msg)
def fields(line):return dict(re.findall(r'(\w+)=([^ ]+)',line))

def profile(args,env,slots,warm_policy,target_policy):
 zero=[{'logical_index':i,'role':'admission','values':{'reused':0,'read_from':0,'reread_to':-1}} for i in range(slots)]+[{'logical_index':1,'role':'solo_migration','values':{'reused':0,'read_from':0,'reread_to':-1}}]
 fresh_profile(args,env,REQUEST_POLICY,zero,slots)
 require(warm_policy=={'strata_fresh':True} and type(warm_policy['strata_fresh']) is bool,'Warm requests mustbe explicitly fresh and distinct')
 require(target_policy=={'strata_fresh':False} and type(target_policy['strata_fresh']) is bool,'Positive target keys mustpersist explicitfreshfalse through every native leg')
 return {'warm_fresh':True,'target_fresh':False,'migration_requires_actual_last_live_slot_restore':True,'checkpoint_only_or_recompute_handoff_qualified':False,'initial_root_nonreuse_requires_actual_distinct_prefix_fixture':True,'runtime_qualified':False}

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
 require(not any(line.startswith('BCPUBLIC reset ') and int(f['rid'])==rid for _,line,f in rows),'Cached migrated RID mustnot claim zero reset/recompute')
 require(int(begin['prompt'])==len(job['ids']) and int(begin['main_session'])==1,'Actual migration prompt/main-session shape differs')
 return {'source_slot':int(begin['source_slot']),'source_slot_generation':int(begin['source_slotgen']),'pid':int(begin['pid']),'engine_generation':int(begin['enginegen']),'request_generation':int(begin['requestgen']),'actual_reused':expected,'actual_read_from':expected,'actual_reread_to':-1,'actual_full_chain_last_live_restore_observed':True,'all_stage_transfer_source_guards_required':stages,'full_cached_fresh_math_qualified':False,'stale_model_or_session_negative_runtime_qualified':False,'scope':'real positive last-live handoff lineage/restore/counters; exact full49 freshness/control still required'}
