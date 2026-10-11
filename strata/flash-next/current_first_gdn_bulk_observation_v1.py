"""Output-only first-layer observation of existing independent own arithmetic."""
import ast,copy
from pathlib import Path
import numpy as np
import prefill_first_gdn_owned_storage_v1 as original
from independent_q8_1_activation_v1 import encode
from current_full48_own_state_bulk_v1 import require
EXTRA=('alpha','beta_pre','conv','normalized','core','decay')
def observed_mixer():
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse(Path(original.__file__).read_bytes())) if isinstance(n,ast.FunctionDef)and n.name=='mixer'))
 require(isinstance(node.body[-1],ast.Return),'Exact original mixer return required')
 # Only retain existing local results after their computation. No math/dispatch edits.
 additions=ast.parse("details['own_intermediates']={'alpha':alpha,'beta_pre':beta_pre,'conv_pre_silu':conv,'core_rms_normalized':normalized,'core_scaled':core}").body
 node.body[-1:-1]=additions;namespace=dict(vars(original));exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[retain-existing-locals-only]','exec'),namespace);return namespace['mixer'],node

def independent_first(model):
 require(model.p.actual_source is True,'Independent original provider required');embedding=np.asarray(model.p.rows('token_embd.weight',[248045])[0],dtype='<f4');residual=np.broadcast_to(embedding,(4,2560)).copy();hc=model.hc_read(residual,'blk.0.hc_attn_');gdn=model.gdn[0];zero=gdn.initial_state();require(not np.any(zero['recurrent'])and not np.any(zero['conv']),'Own first GDN initial state must be zero');after,detail=observed_mixer()[0](gdn,hc['mixed'],{k:v.copy()for k,v in zero.items()},'verifier');written=model.hc_write(residual,detail['output'],hc['inject'])
 fields={'residual_input':residual,'attn_hc_normalized':hc['normalized'],'attn_hc_low':hc['low_silu'],'attn_hc_gate':hc['gate'],'attn_hc_inject':hc['inject'],'attn_mixed':hc['mixed'],'gdn_state_before':zero['recurrent'].transpose(1,0,2),'gdn_conv_before':zero['conv'].T,'gdn_qkv':detail['qkv'],'gdn_z':detail['z'].reshape(-1),'gdn_decay_beta':np.stack([detail['log_decay'],detail['beta']]),'gdn_normalized_qkv':np.concatenate([detail[k].reshape(-1)for k in('q','k','v')]),'gdn_output_gated':detail['normalized_gated_mathematical_F32'].reshape(-1),'gdn_block_output':detail['output'],'gdn_state_after':after['recurrent'].transpose(1,0,2),'gdn_conv_after':after['conv'].T,'residual_after_attn':written}
 extras={'HC_square_sum':hc['square_sum'],'HC_rms_argument':hc['rms_argument'],'HC_rms_scale':hc['rms_scale'],'GDN_alpha':detail['own_intermediates']['alpha'],'GDN_beta_pre':detail['own_intermediates']['beta_pre'],'GDN_conv_pre_silu':detail['own_intermediates']['conv_pre_silu'],'GDN_conv_silu':detail['conv_silu'],'GDN_core_scaled':detail['core_scaled'],'GDN_core_rms_normalized':detail['own_intermediates']['core_rms_normalized']}
 packets={name:encode(np.asarray(fields[key],dtype='<f4').tobytes(),1,count)for name,key,count in(('attn_input_q81','attn_mixed',2560),('gdn_output_q81','gdn_output_gated',6144))};return {'fields':fields,'own_intermediates':extras,'packets':packets,'captured_inputs_used':False,'original_arithmetic_changed':False,'native_intrinsics_qualified':False}
