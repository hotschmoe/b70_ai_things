"""Preregistered complete shared-cache phase recipe and actual counter readback."""
import copy,json,re
from pathlib import Path
import full_cache_shared_admission_v2 as fixture
from full_cache_shared_http_client_v1 import validate_policy
require=fixture.require;read=fixture.read;sha=fixture.sha
PHASES=('warm','prime0','prime1','target_shared','independent','prefill_cancel0','prefill_cancel_peer','decode_cancel','eviction0','eviction1','eviction2','eviction3','eviction4','eviction5','pinned_after_eviction','fresh_shared')
MANDATORY=('independent_concurrent','shared_divergent_concurrent','history_continuation_and_fresh_replay','history_unpinned_source_selected_boundary','prefill_cancel_native_terminal','decode_cancel_survivor_and_solo','fresh_absent_pin','combined_fresh_pin_actual_control','positive_shared_pin_saved_and_reused','stale_request_owner','new_engine_empty_cache','changed_model_cache_refusal','diskcache_wrong_fingerprint_batch0','parallel_pair_persisted_501_refusal','exact_parked_victim','exact_checkpoint_victim','owned_cache_capacity_and_frees','physical_host_device_expert_residency')
RAW49_BYTES=48*10240*4+248320*4
ACTOR_BYTE_LIMIT=64<<20
def schedules(case):
 require(case['schema']=='full-cache-shared-CPU-case-v2' and case['slots']==2 and case['shared_boundary']['tokens']==272,'Exact authentic shared272 fixture required');n=case['shared_boundary']['tokens'];out=[]
 def add(name,phase,indices,bound,reused,cancel=None):
  source=case['phases'][phase];rows=[copy.deepcopy(source['rows'][i]) for i in indices];policy=copy.deepcopy(source['actual_HTTP_body_policy']);validate_policy(policy);require(all(reused<len(r['ids']) for r in rows),'Expected read boundary outside prompt');out.append({'name':name,'source_phase':phase,'logical_indices':indices,'rows':rows,'policy':policy,'max_new':[bound]*len(rows),'expected_reused':[reused]*len(rows),'expected_read_from':[reused]*len(rows),'expected_reread_to':[-1]*len(rows),'expected_new_prompt_tokens':[len(r['ids'])-reused for r in rows],'cancel_kind':cancel,'cancel_index':0 if cancel else None,'native_multirow_required':len(rows)==2 and bound>1,'actual_native_work_and_raw49_still_required':True})
 add('warm','warm',[0,1],32,0)
 add('prime0','prime_shared',[0],1,0)
 add('prime1','prime_shared',[0],1,n)
 add('target_shared','target_shared',[0,1],64,n)
 add('independent','independent',[0,1],64,0)
 # Sole pending admission makes its first real PP ownership unambiguous. The
 # peer comes afterward; this does not claim concurrent prefill cancellation.
 add('prefill_cancel0','prefill_cancel',[0],64,n,'prefill')
 add('prefill_cancel_peer','prefill_cancel',[1],64,n)
 add('decode_cancel','decode_cancel',[0,1],64,n,'decode')
 for i in range(6):add('eviction'+str(i),'eviction_unpinned',[i],64,0)
 add('pinned_after_eviction','target_shared',[0,1],64,n)
 add('fresh_shared','fresh_shared',[0,1],64,0)
 require(tuple(p['name'] for p in out)==PHASES,'Complete preregistered phase roster changed');return out
def scenario_schedule(case,name):
 """Entire fullscope remains required; source37 cannot dump all in one actor."""
 base={p['name']:p for p in schedules(case)};groups={
  'shared':('warm','prime0','prime1','target_shared','fresh_shared'),
  'independent':('warm','independent'),
  'cancellation':('warm','prime0','prime1','prefill_cancel0','prefill_cancel_peer','decode_cancel'),
  'eviction':('warm','prime0','prime1',*[('eviction'+str(i)) for i in range(6)],'pinned_after_eviction','fresh_shared')}
 require(name in groups,'Preregistered bounded scenario required');result=[copy.deepcopy(base[n]) for n in groups[name]]
 if name=='eviction':
  for p in result:
   if p['name'].startswith('eviction'):p['max_new']=[1];p['native_multirow_required']=False;p['declared_before_execution']='GEN1 authentic eviction owners retain complete prompt state; no decode or solo state assumed'
 for p in result:
  pin=p['policy'].get('strata_shared_prefix',{}).get('tokens');p['solo_reuse_policy']='fresh_zero' if p['policy']['strata_fresh'] else 'shared_pin272' if pin is not None else 'actual_consumed_prompt_minus_one'
 # Conservative maximum: admission perrequest; at most one selected later
 # perrequest if genuinely multirow; at most one solo perrequest if budget>1.
 maximum=sum(0 if p['name'] in ('warm','prefill_cancel0') else len(p['rows'])*(1+(1 if p['native_multirow_required'] else 0)+(1 if max(p['max_new'])>1 else 0)) for p in result)
 require(maximum*RAW49_BYTES<=ACTOR_BYTE_LIMIT,'Source37 static64MiB observer bound would be exceeded before GPU')
 return {'name':name,'phases':result,'conservative_maximum_raw49_groups':maximum,'maximum_source_observer_bytes':maximum*RAW49_BYTES,'source_observer_static_process_byte_limit':ACTOR_BYTE_LIMIT,'byte_limit_or_USM_budget_increased':False,'all_other_fullscope_requirements_still_mandatory':list(MANDATORY),'shared_state_transferred_between_actors':False,'full_cache_runtime_qualified':False}
def phase_events(events,first,last):
 require(type(first)is int and type(last)is int and 0<=first<last,'Actual phase sequence interval invalid');rows=[e for e in events if first<e['sequence']<=last];require(rows,'Actual phase trace empty');return rows
def calls_and_work(phase,events):
 begins=[e for e in events if e['kind']=='engine_begin'];ends=[e for e in events if e['kind']=='engine_end'];require(len(begins)==len(ends)==len(phase['rows']) and {e['call'] for e in begins}=={e['call'] for e in ends},'Exact actual phase call roster incomplete');result=[]
 for row,expected in zip(phase['rows'],phase['expected_reused']):
  matches=[e for e in begins if e['rendered_prompt']['messages']==row['messages']];require(len(matches)==1 and matches[0]['submitted_ids']==row['ids'],'Actual phase input/template token IDs differ');begin=matches[0];end=next(e for e in ends if e['call']==begin['call']);legs=[e for e in events if e['kind']=='native_send' and e.get('call')==begin['call'] and e['line'].startswith(('GEN ','BGEN '))];require(legs,'Actual phase native command absent')
  first=legs[0];words=first['line'].split();keys=dict(re.findall(r'(\w+)=([^ ]+)',first['line']));pin=phase['policy'].get('strata_shared_prefix',{}).get('tokens');require(keys.get('fresh')==str(int(phase['policy']['strata_fresh'])) and ('pin' not in keys if pin is None else keys.get('pin')==str(pin)),'Actual first-leg fresh/pin differs');rid=int(keys['rid']);done=[e for e in events if e['kind']=='native_receive' and e['line'].startswith('DONE ') and dict(re.findall(r'(\w+)=([^ ]+)',e['line'])).get('rid')==str(rid) and (len(legs)==1 or e['sequence']<legs[1]['sequence'])];require(len(done)==1,'Actual unique admissionDONE required');d=done[0]['line'].split();require(int(d[2])==len(row['ids']) and int(d[8])==expected and int(d[14])==len(row['ids'])-expected,'Actual expected reuse/new-token admission counters differ')
  result.append({'call':begin['call'],'rid':rid,'pid':begin['engine_pid'],'engine_generation':begin['engine_generation'],'slotgen':int(dict(re.findall(r'(\w+)=([^ ]+)',done[0]['line']))['slotgen']),'input_ids':row['ids'],'actual_reused':int(d[8]),'actual_new_prompt_tokens':int(d[14]),'native_first_command':first,'admission_done':done[0],'observer_end':end})
 return result
def full_scope_status(observed):
 require(type(observed)is dict and set(observed)<=set(MANDATORY),'Unknown fullcache requirement flags');missing=[name for name in MANDATORY if observed.get(name) is not True]
 return {'mandatory_requirements':list(MANDATORY),'remaining_unobserved':missing,'full_cache_runtime_qualified':False,'full_model_math_qualified':False,'latency_qualified':False,'missing_checkpoint_victim_cannot_be_waived':True}
