from postboot_original_identity_reader_v1 import identity_admission as historical_identity
"""Root-only original model work; native arrays are comparison targets only."""
import hashlib,time
from pathlib import Path
import numpy as np
import owned_first_hc_fidelity_v2 as first
import full48_owned_hc_device_rs_v3 as model_module
import postboot_hc35_bulk_host_v1 as bulk
from independent_q8_1_activation_v1 import encode
from explore_owned_layer0_gdn_num10_v1 import load_target,FOUNDATION
from postboot_original_native_targets_v1 import num10_admission,prefix_binding,native_prefix
from full48_owned_composition_storage_v2 import derive_schedule
from original_tensor_rows_postboot_v1 import BoundOriginalFull48RowsPostboot as BoundOriginalFull48Rows
from explore_full48_original_hc_fma_v1 import save_owned_and_compare
from explore_full48_original_prefix1_v4 import identity_admission,save_array,sha,read,write,require,compare_observation
class OriginalWork:
 def __init__(self,first_root,model_identity,bulk_root,prefix_root,output,*,model_association):
  self.model_association=model_association
  self.first_root=Path(first_root).resolve();self.identity=Path(model_identity).resolve();self.bulk_root=Path(bulk_root).resolve();self.prefix_root=Path(prefix_root).resolve() if prefix_root else None;self.output=Path(output).resolve();self.report={'captured_inputs_used':False,'captured_states_or_routes_used':False,'ULP_adjustment_or_lookup_used':False,'full_model_math_qualified':False,'tolerance_gate':None};self.host=bulk.finalized_binding(self.bulk_root,self.model_association)['original_binding'];parent,child,self.first_plan,self.first_binding=num10_admission(self.first_root,self.model_association);self.first_boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],read(self.first_root/'post-health.json')['finished_epoch']);self.first_source=Path(self.first_plan['engine_root'])/'source';self.first_env=read(Path(self.first_plan['prepared'])/'server-config.json')['env'];self.first_source_binding=model_module.prior.source_binding(self.first_source)
  here=Path(__file__).parent;root=here.parents[1];lock=read(here/'model-lock.json');shards=[root/lock['destination']/r['path'] for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];self.report['original_identity_pre']=historical_identity(self.identity,here/'model-lock.json',shards,self.first_boundary,model_association=self.model_association,original_binding=read(self.output/'work-report.json')['original_identity_pre'])['original_binding'];self.report['original_identity_path']=str(self.identity);self.report['host_binding']=self.host;self.report['first_native_binding']=self.first_binding;self.report['first_source_binding']=self.first_source_binding
  if self.prefix_root:
   parent,child,self.prefix_plan,self.prefix_binding=prefix_binding(self.prefix_root,self.model_association);self.prefix_source=Path(self.prefix_plan['engine_root'])/'source';self.prefix_env=read(Path(self.prefix_plan['prepared'])/'server-config.json')['env'];require(model_module.prior.source_binding(self.prefix_source)==self.first_source_binding,'Original first/prefix HC source contracts differ');boundary=max(parent['child_terminal_epoch'],child['finished_epoch'],self.prefix_binding['posthealth_finished']);historical_identity(self.identity,here/'model-lock.json',shards,boundary,model_association=self.model_association,original_binding=read(self.output/'work-report.json')['original_identity_pre']);self.report['prefix_native_binding']=self.prefix_binding
 def make_model(self,plan,env,source,device):
  provider=BoundOriginalFull48Rows(FOUNDATION,self.identity,model_association=self.model_association);model=model_module.OwnedHcDeviceRsReference(provider,sha(self.identity),plan['args'],env,source,self.bulk_root,device,self.output/'rs-operands');return model,provider
 def __call__(self,device):
  raise ValueError('Historical postboot reader cannot produce new mathematics or runtime evidence')
 def recheck(self,report):
  from serial37_canonical_json_v3 import canonical
  require(canonical(report['original_identity_pre'])==canonical(self.report['original_identity_pre']) and report['original_identity_path']==str(self.identity) and report['host_binding']==self.host and report['first_native_binding']==self.first_binding and report['first_source_binding']==self.first_source_binding,'Original current work prerequisite changed');_,targets,packets,binding=load_target(self.first_root,self.first_plan,1,self.first_binding['requests']);require(binding==report['first_target_binding'],'Original first target source changed');gate=report['first_gate'];norm=np.frombuffer(Path(report['first_arrays']['candidate_normalized']['path']).read_bytes(),dtype='<f4').reshape(4,2560);packet=(self.output/'first-candidate-mixed.q81').read_bytes();require(gate['native_normalized_sha256']==hashlib.sha256(np.asarray(targets['attn_hc_normalized'],dtype='<f4').tobytes()).hexdigest() and gate['native_packet_sha256']==hashlib.sha256(packets['attn_input_q81']).hexdigest() and gate['normalized_bytes_equal']==(norm.tobytes()==np.asarray(targets['attn_hc_normalized'],dtype='<f4').tobytes()) and gate['full_Q81_packet_bytes_equal']==(packet==packets['attn_input_q81']) and gate['passed'] is (gate['normalized_bytes_equal'] and gate['full_Q81_packet_bytes_equal']),'Actual normalized/complete packet first gate changed');require(report['prefix4_attempted'] is (self.prefix_root is not None and gate['passed'] is True),'Requested prefix4 must follow exact first gate')
  if report['prefix4_attempted']:
   require(self.prefix_root is not None and gate['passed'] is True,'Actual prefix4 requires admitted targets and first gate');schedule=derive_schedule(self.prefix_plan['prefixes']['4'],self.prefix_plan['args'],self.prefix_env);_,native,binding=native_prefix(self.prefix_root,self.prefix_plan,4,schedule,self.model_association);require(binding==report['prefix4_targets'] and len(report['prefix4']['comparisons'])==577,'Exact current prefix4 target/comparison roster changed')
   for key,target in native.items():
    value=np.frombuffer(Path(report['prefix4']['arrays'][key]['path']).read_bytes(),dtype='<f4').reshape(target.shape);require(compare_observation(value,target)==report['prefix4']['comparisons'][key],'Original prefix comparison changed '+key)
  return {'first_gate_recomputed':True,'prefix4_current_577_targets_recomputed':report['prefix4_attempted'],'full_model_math_qualified':False}
