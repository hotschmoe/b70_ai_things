"""First-row HC vector localization; original weights/embedding only."""
import numpy as np
import json
from pathlib import Path
import full48_owned_hc_fma_refinement_v1 as refined
from full48_owned_composition_storage_v2 import RouteAwareFull48OwnedComposition as old
import owned_hc35_f32_block_control_v2 as hc
require=refined.require
TARGET_FIELDS={'normalized':'attn_hc_normalized','low':'attn_hc_low','gate':'attn_hc_gate','inject':'attn_hc_inject','mixed':'attn_mixed'}
def source_binding():
 plan=json.loads((Path(__file__).parent/'owned-first-hc-fidelity-source-plan-v1.json').read_text())
 for n,w in plan['files'].items():require(refined.bulk.sha(Path(__file__).resolve().parents[2]/n)==w,'FirstHC localization source changed '+n)
 return plan['files']

def contract(token_ids):
 ids=list(token_ids);require(len(ids)==1 and type(ids[0]) is int and 0<=ids[0]<248320,'Actual prefix1 first-row token only');return ids

def compute_owned(model,token_ids):
 """No target/nativestate argument. Both candidates start from original embedding."""
 ids=contract(token_ids);embedding=refined.finite(model.p.rows('token_embd.weight',ids)[0],(2560,));residual=np.broadcast_to(embedding,(4,2560)).copy();candidate=model.hc_read(residual,'blk.0.hc_attn_');prior=old.hc_read(model,residual,'blk.0.hc_attn_');scaled=np.asarray(candidate['down']/4,dtype='<f4');low=np.asarray([hc.host.f32(float(x)*hc.sigmoid_candidate(float(x))) for x in scaled],dtype='<f4');candidate['low']=low;scaled_old=np.asarray(prior['down']/4,dtype='<f4');prior['low']=np.asarray(scaled_old*old.sigmoid(scaled_old),dtype='<f4')
 return {'ids':ids,'embedding_owned':embedding,'residual_owned':residual,'candidate':candidate,'prior':prior,'captured_inputs_used':False,'captured_state_or_routes_used':False,'full_model_math_qualified':False,'native_window_rounding_qualified':False,'scope':'Owned first HC only; prefix1 T1 native target, not T2 GDN proof'}

def conditional_gdn(model,native_mixed):
 """EXPLICIT conditional captured-input seam. Never feeds original fullmodel."""
 mixed=refined.finite(native_mixed,(2560,));gdn=model.gdn[0];zero=gdn.initial_state();state,detail=gdn.mixer(mixed,zero,'verifier')
 return {'captured_inputs_used':True,'captured_states_used':False,'scope':'Conditional native HC mixed -> independent original GDN; NOT independent fullmodel','full_model_math_qualified':False,'numeric_pass_claim':False,'tolerance_gate':None,'output':detail['output'],'state':state,'detail':detail}
