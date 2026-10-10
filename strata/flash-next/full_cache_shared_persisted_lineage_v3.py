"""Actual batch0 native session + HTTP response associations; no state inputs."""
import hashlib,json,re,stat
from pathlib import Path
from full_cache_shared_history_v2 import require
from full_cache_shared_persisted_v2 import wrong_model_header,fingerprint_refusal

TILE=1<<20

def sha_stream(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while True:
   chunk=stream.read(TILE)
   if not chunk:break
   h.update(chunk)
 return h.hexdigest()

def negative_file(original,negative):
 """ROOT ONLY: derives a new file from this actor's actual saved session."""
 original=Path(original);negative=Path(negative);require(original.is_file() and not original.is_symlink() and original.parent.resolve()==negative.parent.resolve() and not negative.exists() and original!=negative,'Exact same-actor original/new session file paths required');before=original.stat();original_sha=sha_stream(original)
 with original.open('rb') as source:
  header=source.read(64);new_header,derivation=wrong_model_header(header,before.st_size)
  with negative.open('xb') as target:
   target.write(new_header)
   while True:
    chunk=source.read(TILE)
    if not chunk:break
    target.write(chunk)
 require(original.stat()==before and sha_stream(original)==original_sha and negative.stat().st_size==before.st_size,'Original session changed during bounded negative derivation')
 with original.open('rb') as a,negative.open('rb') as b:
  require(a.read(64)==header and b.read(64)==new_header,'Consumed session header changed')
  while True:
   aa,bb=a.read(TILE),b.read(TILE);require(aa==bb,'Negative changed original state payload/trailer')
   if not aa:break
 return {'original_path':str(original.resolve()),'negative_path':str(negative.resolve()),'original_sha256':original_sha,'negative_sha256':sha_stream(negative),'bytes':before.st_size,'original_header_hex':header.hex(),'negative_header_hex':new_header.hex(),'derivation':derivation,'original_session_rewritten':False,'native_checksum_agreement_observed':False,'actual_restore_execution_observed':False}

def session_binding(events,http,expected,actual_owner):
 """Original FIFO session method and original POST both have to match."""
 require(type(expected)is dict and set(expected)=={'action','filename','expected_saved_tokens'} and expected['action'] in ('save','restore_valid','restore_wrong_model') and type(expected['expected_saved_tokens'])is int and expected['expected_saved_tokens']>0 and re.fullmatch('[a-zA-Z0-9_-]+\\.bin',expected['filename']),'Explicit saved-prefix/session action contract required')
 action='save' if expected['action']=='save' else 'restore';require(http['request']=={'method':'POST','path':'/slots/0?action='+action,'body':{'filename':expected['filename']}} and http['error'] is None,'Actual original persisted HTTP request differs')
 begin=[e for e in events if e['kind']=='session_begin'];end=[e for e in events if e['kind']=='session_end'];require(len(begin)==len(end)==1,'Actual unique original session method begin/end required');begin,end=begin[0],end[0]
 require(all(type(e['engine_pid'])is int and e['engine_pid']==actual_owner['engine_pid'] and type(e['engine_generation'])is int and e['engine_generation']==actual_owner['engine_generation'] for e in (begin,end)) and type(begin['call'])is int and begin['call']==end['call'] and begin['sequence']<end['sequence'],'Actual session incarnation/control call differs')
 native_path='/results/sessions/'+expected['filename'];require(begin['action']==action and begin['path']==native_path and end['action']==action and end['path']==native_path,'Original native session path/action differs');sends=[e for e in events if e['kind']=='native_send'];require(len(sends)==1 and sends[0]['line']==('SAVE ' if action=='save' else 'RESTORE ')+native_path and sends[0]['call']==begin['call'] and sends[0]['engine_pid']==actual_owner['engine_pid'] and begin['sequence']<sends[0]['sequence']<end['sequence'],'Original native persisted command ownership differs')
 receives=[e for e in events if e['kind']=='native_receive'];require(receives and all(type(e['engine_pid'])is int and e['engine_pid']==actual_owner['engine_pid'] and sends[0]['sequence']<e['sequence']<end['sequence'] for e in receives),'Actual FIFO session receive owner/order differs');terminals=[e for e in receives if e['line'].startswith(('SAVED ','RESTORED ','SERR '))];require(len(terminals)==1,'Actual unique native persisted terminal required');terminal=terminals[0]['line']
 if expected['action']=='restore_wrong_model':
  require(terminal=='SERR invalid 0 session file: saved with another model (model fingerprint differs)' and end['error']=={'kind':'invalid','published':False,'message':'session file: saved with another model (model fingerprint differs)'},'Actual wrong-fingerprint native refusal must be unpublished');fingerprint_refusal(http['status'],http['body'],False)
 else:
  words=terminal.split();require(len(words)==4 and words[0]==('SAVED' if action=='save' else 'RESTORED') and int(words[1])==expected['expected_saved_tokens'] and int(words[2])>80 and end['error'] is None,'Actual native saved/restored prefix extent differs');result=end['result'];require(result['tokens']==int(words[1]) and result['bytes']==int(words[2]) and result['ms']==float(words[3]),'Original session parser result changed');body=http['body'];require(http['status']==200 and body['id_slot']==0 and body['filename']==expected['filename'] and body['n_saved' if action=='save' else 'n_restored']==int(words[1]) and body['n_written' if action=='save' else 'n_read']==int(words[2]),'Actual HTTP/native persisted result association differs')
 return {'actual_FIFO_native_and_HTTP_session_joined':True,'native_terminal':terminal,'actual_incarnation':dict(actual_owner),'requires_saved_file_and_prepost_full49_current_parent':True,'complete_persisted_or_full_cache_qualified':False}


def saved_files(actor_root,derivation,saved_tokens,consumed_prefix):
 """Root reader checks exact original/new files; no original file rewrite."""
 from full_cache_shared_history_v2 import digest
 root=Path(actor_root).resolve();original=root/'sessions/valid.bin';negative=root/'sessions/wrong.bin';require(derivation['original_path']==str(original) and derivation['negative_path']==str(negative) and original.is_file() and negative.is_file() and not original.is_symlink() and not negative.is_symlink(),'Actual same-actor persisted files required')
 require(type(saved_tokens)is int and saved_tokens==len(consumed_prefix) and digest(consumed_prefix),'Actual SAVED boundary must be original consumed PCL prefix')
 so,sn=original.stat(),negative.stat();require(so.st_size==sn.st_size==derivation['bytes'] and sha_stream(original)==derivation['original_sha256'] and sha_stream(negative)==derivation['negative_sha256'],'Original persisted file extent/digest changed')
 with original.open('rb') as a,negative.open('rb') as b:
  header=a.read(64);changed=b.read(64);computed,proof=wrong_model_header(header,so.st_size);require(header.hex()==derivation['original_header_hex'] and changed==computed and changed.hex()==derivation['negative_header_hex'] and proof==derivation['derivation'],'Exact independently rederived model-only negative header differs')
  while True:
   aa,bb=a.read(TILE),b.read(TILE);require(aa==bb,'Actual wrong-model file changed original payload/trailer')
   if not aa:break
 require(original.stat()==so and negative.stat()==sn and sha_stream(original)==derivation['original_sha256'] and sha_stream(negative)==derivation['negative_sha256'],'Persisted files changed during original-byte recollection')
 require(derivation['original_session_rewritten'] is False and derivation['native_checksum_agreement_observed'] is False and derivation['actual_restore_execution_observed'] is False,'Negative-file derivation cannot import runtime claims')
 return {'original_consumed_prefix_sha256_le32':digest(consumed_prefix),'original_files_recollected':True,'original_saved_file_rewritten':False,'native_execution_and_current_parent_still_required':True}

def refusal_state_unchanged(before,after,session_end,expected_prefix):
 """Own same-incarnation before/after full49, not another actor's state."""
 from full_cache_shared_raw49_v2 import compare49
 a,b=before['source_lineage'],after['source_lineage'];require(a['actual_native_PCL_pid']==b['actual_native_PCL_pid']==session_end['engine_pid'] and a['actual_engine_generation']==b['actual_engine_generation']==session_end['engine_generation'] and a['HTTP_end_sequence']<session_end['sequence']<b['native_send_sequence'] and b['actual_native_PCL_request_ordinal']==a['actual_native_PCL_request_ordinal']+1 and before['input_ids']==after['input_ids']==expected_prefix,'Actual own prefix/state before/after refusal lineage differs')
 require(session_end['action']=='restore' and session_end['error']=={'kind':'invalid','published':False,'message':'session file: saved with another model (model fingerprint differs)'},'Only actual unpublished native wrong-model refusal can precede control')
 result=compare49(before,after);require(result['all49_bitwise_equal'] is True,'Wrong-model refusal changed actual own full49 response/state control')
 return {'actual_own_before_after_49':result,'no_different_actor_cached_state_borrowed':True,'fresh_control_current_parent_and_public_identity_still_required':True,'full_cache_runtime_qualified':False}
