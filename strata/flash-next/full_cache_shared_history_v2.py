"""Own-response history derivation; tokenizer observations are input metadata only."""
import copy, hashlib, json, struct

FOLLOWUP = 'Explain one practical consequence of your answer in one clear sentence.'

def require(ok, message):
 if not ok: raise ValueError(message)

def digest(ids):
 require(type(ids) is list and ids and all(type(t) is int and 0 <= t < 248320 for t in ids), 'Exact nonempty accepted token IDs required')
 return hashlib.sha256(b''.join(struct.pack('<I', t) for t in ids)).hexdigest()

def canonical_sha(value):
 return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()

def own_response_messages(messages, client, native_begin, native_end, decoded):
 """The completed reply must belong to this actor, not a captured other actor."""
 require(type(messages) is list and messages and all(type(m) is dict and set(m) == {'role', 'content'} and type(m['content']) is str for m in messages), 'Exact own message input required')
 require(type(client) is dict and client['request']['messages'] == messages and client['done_received'] is True and client['cancel_requested'] is False and client['error'] is None, 'History requires a complete uncancelled own HTTP reply')
 require(native_end['kind'] == 'engine_end' and native_end['error'] is None and native_end['cancelled'] is False and type(native_end['engine_pid']) is int and native_end['engine_pid'] > 0 and type(native_end['call']) is int and type(native_end['rid']) is int, 'History requires actual completed owned native reply')
 require(native_begin['kind'] == 'engine_begin' and native_begin['rendered_prompt']['messages'] == messages and native_begin['rendered_matches_submitted'] is True and native_begin['rendered_prompt']['ids'] == native_begin['submitted_ids'] and native_begin['submitted_ids_sha256'] == digest(native_begin['submitted_ids']) and native_begin['call'] == native_end['call'] and native_begin['engine_pid'] == native_end['engine_pid'] and native_begin['engine_generation'] == native_end['engine_generation'], 'Own history begin/end/input association differs')
 require(native_end['generated_ids_sha256'] == digest(native_end['generated_ids']), 'Own generated ID association changed')
 text = ''; finished = False
 for event in client['events']:
  data = event['data']
  if data == '[DONE]':
   require(not finished, 'Duplicate HTTP terminal'); finished = True; continue
  require(not finished, 'HTTP data after terminal')
  row = json.loads(data); require(not row.get('error') and len(row.get('choices', [])) == 1, 'Own HTTP reply invalid')
  content = row['choices'][0].get('delta', {}).get('content')
  require(content is None or type(content) is str, 'Own reply content must be text')
  if content: text += content
 require(finished and text.strip(), 'Actual complete nonempty own reply required')
 require(type(decoded) is dict and set(decoded) == {'ids', 'text', 'engine_pid', 'call', 'independent_original_tokenizer_decode'} and decoded['ids'] == native_end['generated_ids'] and decoded['text'] == text and decoded['engine_pid'] == native_end['engine_pid'] and decoded['call'] == native_end['call'] and decoded['independent_original_tokenizer_decode'] is True, 'Actual independent decode must join HTTP text to own native IDs')
 derived = copy.deepcopy(messages) + [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': FOLLOWUP}]
 return {'messages': derived, 'messages_sha256': canonical_sha(derived), 'source_client_sha256': canonical_sha(client), 'source_native_begin_sha256': canonical_sha(native_begin), 'source_native_end_sha256': canonical_sha(native_end), 'source_independent_decode_sha256': canonical_sha(decoded), 'engine_pid': native_end['engine_pid'], 'call': native_end['call'], 'rid': native_end['rid'], 'assistant_text': text, 'followup_literal': FOLLOWUP, 'different_actor_state_borrowed': False}

def render_binding(request, observed, actual_pid):
 require(type(request) is dict and set(request) == {'messages', 'messages_sha256', 'template_kwargs'} and request['messages_sha256'] == canonical_sha(request['messages']) and request['template_kwargs'] == {'enable_thinking': False}, 'Exact prerequest history render contract required')
 require(type(observed) is dict and set(observed) == {'messages', 'messages_sha256', 'template_kwargs', 'tools', 'ids', 'ids_sha256', 'engine_pid', 'original_encode_prompt_called'} and observed['messages'] == request['messages'] and observed['messages_sha256'] == request['messages_sha256'] and observed['template_kwargs'] == request['template_kwargs'] and observed['tools'] is None and observed['engine_pid'] == actual_pid and observed['original_encode_prompt_called'] is True, 'Actual own original tokenizer render association differs')
 require(type(actual_pid) is int and actual_pid > 0 and observed['ids_sha256'] == digest(observed['ids']), 'Actual render token digest/owner differs')
 return copy.deepcopy(observed)

def pinned_history_phase(derivation, rendered, original_ids, boundary, fresh=False):
 """Boundary is fixed before execution, never inferred from resulting reuse."""
 require(type(fresh) is bool and type(boundary) is int and boundary == 272 and len(original_ids) > boundary and len(rendered['ids']) > boundary, 'Declared shared272 history boundary required')
 require(rendered['messages'] == derivation['messages'] and rendered['messages_sha256'] == derivation['messages_sha256'] and rendered['engine_pid'] == derivation['engine_pid'], 'History render must belong to its own prior actor')
 require(rendered['ids'][:boundary] == original_ids[:boundary] and rendered['ids_sha256'] == digest(rendered['ids']), 'Actual rendered history does not preserve preregistered shared prefix')
 reused = 0 if fresh else boundary
 return {'name': 'history_fresh' if fresh else 'history_pinned', 'rows': [{'messages': copy.deepcopy(rendered['messages']), 'ids': list(rendered['ids'])}], 'policy': {'strata_fresh': fresh, **({} if fresh else {'strata_shared_prefix': {'tokens': boundary}})}, 'max_new': [1], 'expected_reused': [reused], 'expected_read_from': [reused], 'expected_reread_to': [-1], 'expected_new_prompt_tokens': [len(rendered['ids']) - reused], 'cancel_kind': None, 'cancel_index': None, 'native_multirow_required': False, 'history_derivation': copy.deepcopy(derivation), 'history_render': copy.deepcopy(rendered), 'expected_counters_declared_before_native_request': True, 'unpinned_source_selected_history_qualified': False, 'full_cache_runtime_qualified': False}

HISTORY_PHASES=('history_pinned','history_unpinned','history_fresh','combined_fresh_pin')

def history_prototypes():
 """One request each, with no guessed output text, token IDs or token counts."""
 return [{'name':name,'rows':[{'origin':'actual_same_actor_prime1_reply_and_original_tokenizer','messages':None,'ids':None}], 'max_new':[1],'cancel_kind':None,'cancel_index':None,'native_multirow_required':False,'dynamic_own_history_required':True,'history_rendering_changes_engine_requests':False,'expected_reused_rule':('actual_rendered_prompt_minus_one_saved_before_repeat' if name=='history_unpinned' else 'zero' if name in ('history_fresh','combined_fresh_pin') else 'fixed_shared272'),'actual_native_work_and_raw49_still_required':True} for name in HISTORY_PHASES]

def resolve_history(prototype,derivation,rendered,original_ids,prior_inventory=None,prior_rid=None):
 name=prototype['name'];require(name in HISTORY_PHASES,'Declared history family required')
 expected=next(p for p in history_prototypes() if p['name']==name)
 require({k:v for k,v in prototype.items() if k not in ('expected_actual_RID_set','raw_capture_requested','solo_reuse_policy')}==expected,'Declared history prototype changed')
 fresh=name in ('history_fresh','combined_fresh_pin');phase=pinned_history_phase(derivation,rendered,original_ids,272,fresh);phase['name']=name
 resume=0 if fresh else 272
 turns=[i for i,t in enumerate(rendered['ids']) if i>resume and t==248045]
 require(turns,'Actual rendered own history must contain a real source turn after its admitted boundary')
 phase['expected_source_turn_boundary']=max(turns);phase['expected_source_turn_prefix_sha256']=digest(rendered['ids'][:max(turns)])
 if name=='combined_fresh_pin':phase['policy']['strata_shared_prefix']={'tokens':272}
 if name=='history_unpinned':
  # Source39 batch_chain_enabled saves the complete prompt leaf at n-1
  # (generate.cpp checkpoint_at(n-1)); a same-input unpinned repeat selects
  # that longest eligible prefix. Require its actual prior saved observation,
  # rather than substituting the resulting future reuse counter as an oracle.
  n=len(rendered['ids']);require(type(prior_rid)is int and prior_rid>0 and type(prior_inventory)is dict and prior_inventory['kind']=='checkpoint_inventory' and prior_inventory['phase']=='main_body_complete' and prior_inventory['rid']==prior_rid and prior_inventory['pid']==rendered['engine_pid'],'Actual earlier history checkpoint inventory required')
  matching=[p for p in prior_inventory['points'] if type(p['tokens'])is int and p['tokens']==n-1 and p['sha256_le32']==digest(rendered['ids'][:-1])]
  require(len(matching)==1,'Actual prior exact own prompt-minus-one checkpoint missing')
  turn=[p for p in prior_inventory['points'] if type(p['tokens'])is int and p['tokens']==phase['expected_source_turn_boundary'] and p['sha256_le32']==phase['expected_source_turn_prefix_sha256']]
  require(len(turn)==1,'Actual prior real history turn checkpoint missing')
  phase['required_prior_turn_boundary']=phase['expected_source_turn_boundary'];phase['required_prior_turn_prefix_sha256']=phase.pop('expected_source_turn_prefix_sha256');phase['expected_source_turn_boundary']=-1
  phase.update(policy={'strata_fresh':False},expected_reused=[n-1],expected_read_from=[n-1],expected_new_prompt_tokens=[1],prior_saved_leaf_inventory=copy.deepcopy(prior_inventory),prior_saved_leaf_RID=prior_rid,unpinned_source_selected_history_qualified=False)
 phase.update(expected_actual_RID_set=prototype['expected_actual_RID_set'],raw_capture_requested=prototype['raw_capture_requested'],solo_reuse_policy='fresh_zero' if fresh else 'actual_consumed_prompt_minus_one' if name=='history_unpinned' else 'shared_pin272',dynamic_own_history_required=True,source_prototype=copy.deepcopy(prototype))
 return phase
