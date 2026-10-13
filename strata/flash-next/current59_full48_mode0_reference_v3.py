"""Current59 own-state scalar shared gate shape correction; V1/V2 frozen."""
import re
import numpy as np
from current59_full48_mode0_reference_v2 import ConsumerProjector as PriorProjector,Current59Full48OwnState as PriorState,OwnedPle,require
from current59_full48_mode0_reference_v1 import bf16_source_rows
class ConsumerProjector(PriorProjector):
 def project(self,name,x,weight='original',activation='f32',expert=None):
  context={'role':name,'original_shape':list(self.p.shape(name)),'original_kind':self.kind(name),'weight_storage':weight,'activation_storage':activation,'expert':expert}
  self.last_project_role=context
  try:
   if re.fullmatch(r'blk\.\d+\.ffn_gate_inp_shexp\.weight',name):
    require(tuple(context['original_shape'])==(2560,)and context['original_kind']=='F32'and weight=='bf16'and activation=='f32'and expert is None,'Exact original shared scalar gate vector/consumer required')
    operand=np.asarray(x,dtype='<f4');require(operand.shape==(2560,)and np.isfinite(operand).all(),'Own finite scalar gate operand required')
    # Original rank1 tensor has exactly one logical row, not an inferred matrix.
    rows=np.asarray(self.p.rows(name,[0]),dtype='<f4');require(rows.shape==(1,2560),'Original scalar gate single-row provider shape differs')
    value=bf16_source_rows(rows,operand,self.backend,'BF16_MMVF_pair2_block256')
    require(value.shape==(1,),'One scalar gate result required')
    self.record(name,'BF16_MMVF_scalar_nout1_pair2_block256',context['original_kind'],'RAW_F32_own','original_F32_narrowed_BF16_RNE',['device_FMA_gradual_semantics_unqualified','downstream_native_sigmoid_intrinsic_unqualified'])
    return value.astype(np.float64)
   return super().project(name,x,weight,activation,expert)
  except BaseException as exc:
   exc.reference_role_context=context
   raise
class Current59Full48OwnState(PriorState):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.projector=ConsumerProjector(self.p,self.rows_backend)
  for collection in(self.gdn,self.qsa,self.ffn):
   for model in collection.values():model.projector=self.projector
  self.ple=OwnedPle(self.p,self.projector,self.identity)
