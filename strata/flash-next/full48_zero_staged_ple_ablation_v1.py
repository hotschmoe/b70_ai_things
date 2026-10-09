"""Explicit MODIFIED reference: discard original table embedding at PLE projection.
This is a counterfactual localization experiment, never an exact model oracle.
"""
import numpy as np
from full48_owned_composition_storage_v1 import Full48OwnedComposition,OwnedPle
PROVENANCE={'kind':'MODIFIED_REFERENCE_ZERO_STAGED_PLE_EMBEDDING','actual_zero_native_PLE_measured':False,'actual_exact_original_model_reference':False,'original_PLE_table_lookup_retained':True,'original_PLE_projection_input_replaced_with_zero':True,'captured_values_used_as_reference_feed':False,'native_math_or_model_qualification':False,'purpose':'Counterfactual for missing current host PLE gather; numerical proximity alone cannot prove actual staged bytes were zero'}

class ZeroStagedPle(OwnedPle):
 def project(self,role,embedding):return super().project(role,np.zeros(2560))
 def advance(self,token,layer_residual):
  result=super().advance(token,layer_residual);result['original_table_embedding_F32']=result['embedding_F32'];result['embedding_F32']=np.zeros(2560);result['reference_ablation']=dict(PROVENANCE);return result

class ZeroStagedPleComposition(Full48OwnedComposition):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.ple=ZeroStagedPle(self.p,self.projector,self.identity)
 def tokens(self,ids):
  result=super().tokens(ids);result['reference_ablation']=dict(PROVENANCE);result['actual_exact_original_model_reference']=False;return result
