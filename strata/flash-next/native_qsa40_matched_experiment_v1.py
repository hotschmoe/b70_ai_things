"""Preregistered source40 matched observer experiment; no serving executor."""
from pathlib import Path
import hashlib
import native_layer3_qsa_observer_contract_v1 as observer
from serial37_canonical_json_v3 import canonical
require=observer.require
HERE=Path(__file__).resolve().parent
ENGINE_PLAN=HERE/'native-layer3-qsa-observer-engine-build-plan-v2.json'
ENGINE_SHA='d87740b5cbd5310895b057541ae5537034b993e38bb20e6248350a7452dbc2e6'
IDS=list(observer.IDS)
ABSENT=('STRATA_VERIFY_EAGER','STRATA_ROPE_TABLE','STRATA_NO_NORM_ROPE','STRATA_ATTN_LANECELL')
TARGET_KEYS=('STRATA_QSA3_TARGET','STRATA_QSA3_TARGET_DIR','STRATA_QSA3_TARGET_BINDING_SHA256')
# Standalone own normalization outputs have no internal fused-native witness.
DEVICE_TARGETS={
 'q_RoPE.f32':'q_fused_RoPE','k_RoPE.f32':'k_fused_RoPE',
 'iq_RoPE.f32':'indexer_query_fused_RoPE','attention.f32':'attention','gated.f32':'gated',
 'K_pool_windows.f16':'K_physical_page0_after_resolve',
 'V_pool_windows.f16':'V_physical_page0_after_resolve'}
UNOBSERVED=('q_normalized.f32','k_normalized.f32','iq_normalized.f32',
 'internal_norm_argument','internal_rsqrt','internal_attention_score','internal_softmax')

def request_roster():
 return [{'ordinal':i,'ids':list(IDS),'fresh':True,'pin':None,'max_new':1,
          'expected_windows':[list(w)for w in observer.WINDOWS]}for i in range(1,5)]

def admit_roster(rows):
 require(canonical(rows)==canonical(request_roster()),'Exact four fresh prefix4 requests, absent PIN, MAX1 and2/1/1 required')
 return {'native_first_head_only':True,'natural_continuation_qualified':False}

def arm_environment(base,on,binding,directory):
 require(type(on)is bool and len(binding)==64 and all(c in '0123456789abcdef'for c in binding), 'Typed arm and exact lowercase planSHA required')
 require(Path(directory).is_absolute(),'Actual absolute owned capture directory required')
 require(not any(k in base for k in ABSENT),'Eager/alternate math environment must be ABSENT including0')
 require(all(base.get(k,'0')=='0'for k in TARGET_KEYS),'Baseline native QSA observer must be OFF')
 env=dict(base)
 for k in TARGET_KEYS:env.pop(k,None)
 # P30/SFD are ON in both arms so the sole tested observer delta is QSA3.
 env.update({'STRATA_FIDELITY_DIAG':'1','STRATA_FIDELITY_DIAG_ACTIVATIONS':'1',
             'STRATA_PREFIX30':'1','STRATA_PREFIX30_DIR':'/results/p30',
             'STRATA_FIDELITY_DIAG_DIR':'/results/captures',
             'STRATA_FIDELITY_DIAG_ARM':'/results/ARM','STRATA_PREFIX30_BINDING_SHA256':binding,
             'STRATA_PREFIX_DIAG':'1','STRATA_PREFIX_LIFECYCLE_DIAG':'1',
             'STRATA_PREFIX_DIAG_ARM':'/results/ARM','STRATA_ARTIFACT_IDENTITY_SHA256':binding,
             'STRATA_PLE_INPUT33':'1','STRATA_PLE_INPUT33_DIR':'/results/ple-input',
             'STRATA_PLE_INPUT33_BINDING_SHA256':binding})
 if on:env.update({'STRATA_QSA3_TARGET':'1','STRATA_QSA3_TARGET_DIR':directory,
                   'STRATA_QSA3_TARGET_BINDING_SHA256':binding})
 return env

def admit_arm_pair(left,right,binding,directory):
 require(all(k not in left for k in TARGET_KEYS),'OFF observer has no target variables')
 expected=dict(left);expected.update({'STRATA_QSA3_TARGET':'1','STRATA_QSA3_TARGET_DIR':directory,
                                     'STRATA_QSA3_TARGET_BINDING_SHA256':binding})
 require(canonical(right)==canonical(expected),'Only exact defaultOFF observer activation may differ between arm environments')
 return True

def recipe(cards):
 require(cards in ([0],[0,1])and all(type(c)is int for c in cards),'Exact one/pair topology required')
 paired=len(cards)==2
 return {'schema':1,'engine_plan_sha256':ENGINE_SHA,'required_parent_generation':1403,
  'required_upload_generation':3,'cards':cards,'requests':request_roster(),
  'stages':{str(k):list(v)for k,v in ({0:(0,32),1:(32,48)}if paired else{0:(0,48)}).items()},
  'ON_owner_frames':12,'ON_nonowner_zero_frames':12 if paired else 0,
  'OFF_QSA_frames':0,'head_all48_lastrow_bitwise_pairs':196,
  'wholeprefix_P30_phase_bitwise_pairs':4*4*48*3,
  'QSA_fields':observer.fields(),'device_target_mapping':dict(DEVICE_TARGETS),
  'projection_target_mapping':{'attn_q.weight':'q_full_projected','attn_k.weight':'k_projected',
   'attn_v.weight':'v_projected','indexer.k_proj.weight':'indexer_key_projected',
   'indexer.q_proj.weight':'indexer_query_projected','attn_output.weight':'output_projected'},
  'indexer_state_layout':{'tail':{'offset':0,'bytes':1536},'dead':{'offset':1536,'bytes':512},
   'pooled':{'offset':2048,'bytes':1024},'block_position':{'offset':3072,'bytes':4}},
  'unobserved_native_values':list(UNOBSERVED),'captures_are_math_inputs':False,
  'original_reference_math_changed':False,'source37_runtime_proof_transferred':False,
  'internal_fused_normalization_or_score_witness_claimed':False,
  'actual_observer_runtime_qualified':False,'full_model_math_qualified':False,'runtime_ready':False}

def candidate_binding(root):
 """Only root preparation invokes current full C140/model/source admission."""
 import c140_baseline_admission_v3 as baseline
 require(hashlib.sha256(ENGINE_PLAN.read_bytes()).hexdigest()==ENGINE_SHA,'Exact corrected source40 plan required')
 prepared,proof=baseline.finalized_binding(root)
 generation=prepared['combined_generation']
 require(generation['consumer_generation']==1403 and generation['source_count']==67
         and generation['default_off_native_layer3_QSA_targets40']is True,
         'Fresh corrected C1401403/source40 association required')
 return {'prepared':prepared,'current_C140_proof':proof,'old_C113_or_C137_admission_transferred':False,
         'native_target_capture_or_original_math_qualified':False}
