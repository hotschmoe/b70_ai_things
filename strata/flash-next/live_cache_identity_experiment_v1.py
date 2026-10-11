"""Preregistered actual live-owner refusal stages; root executor still separate."""
from live_cache_identity_namespace_v1 import namespace,namespace_probes,require,KEYS

def schedule(owner):
 original=namespace(owner)
 return {'schema':1,'owner_namespace':original,'phases':[
  {'name':'warm_owned_native','actual_native_requests':2,'fresh':True},
  {'name':'prime_original_cold','actual_native_requests':1,'expected_reused':0},
  {'name':'prime_original_shared272','actual_native_requests':1,'expected_reused':272},
  *[{'name':'foreign_request_'+p['component'],'expected_native_requests':0,**p}for p in namespace_probes(original)],
  *[{'name':'loaded_owner_'+k,'operation':'attempt_owned_component_update','expected_native_requests':0,'original_object_and_metadata_restored_after_refusal':True}for k in KEYS],
  {'name':'live_authoritative_namespace_rebind','expected_native_requests':0,'old_native_must_still_be_alive':True,'expected_refusal':'LIVE_NAMESPACE_REBIND_REFUSED'},
  {'name':'actual_native_restart_attempt','expected_native_requests':0,'original_restart_close_invoked':False,'expected_refusal':'LIVE_NAMESPACE_REBIND_REFUSED'},
  {'name':'repeat_original_after_all_refusals','actual_native_requests':1,'expected_reused':272,'all49_vs_own_prime_and_independently_fresh_required':True},
  {'name':'retire_original_native','normal_original_EOF_and_owned_removal_required':True},
  {'name':'fresh_new_incarnation','new_load_identity_and_empty_cache_admission_required':True,'old_cached_state_allowed':False}],
 'actual_weight_file_modification_required':False,'source40_native_model_fingerprint_cache_key_qualified':False,'actual_different_weights_loaded':False,'changed_loaded_identity_lifecycle_qualified':False,'actual_source_or_metadata_file_mutation_is_not_loaded_model_mutation':True,'full_cache_runtime_qualified':False}

def refusal_binding(probe,guard_rows,before,after):
 """Source-shaped negative receipt must have no actual native request activity."""
 require(type(guard_rows)is list and len(guard_rows)==1,'Exactly one original refusal marker required');row=guard_rows[0]
 require(row['kind']=='live_identity_prepare_refused' and row['original_prepare_invoked']is False and row['native_request_authorized']is False and row['loaded_owner_replaced']is False,'Refusal must precede original render/native/cache path')
 require(probe['expected_refusal'] in row['error'] and row['requested_namespace']==probe['requested_namespace'],'Original requested foreign namespace/refusal differs')
 require(before==after and type(before)is dict and type(before['native_pid'])is int and type(before['native_generation'])is int and type(before['native_begin_count'])is int and type(before['native_terminal_count'])is int,'Same exact actual incarnation/native request counters required across refusal')
 return {'frontend_refusal_before_cache_qualified':True,'different_weights_loaded':False,'native_cache_fingerprint_key_qualified':False,'full_cache_runtime_qualified':False}
