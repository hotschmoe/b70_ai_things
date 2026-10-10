"""Exact OWN post-PLE HC operand publication; arithmetic inherited unchanged."""
import hashlib
from pathlib import Path
import full48_owned_hc_device_rs_v1 as original
from full48_owned_hc_device_rs_v1 import prior,hc,finite,require
class OwnedHcDeviceRsReference(original.OwnedHcDeviceRsReference):
 def __init__(self,provider,identity,args,env,source_root,bulk_build_root,device_rs,operand_root,tile_bytes=64<<20):
  self.operand_root=Path(operand_root).resolve();self.operand_root.mkdir(parents=True,exist_ok=True)
  super().__init__(provider,identity,args,env,source_root,bulk_build_root,device_rs,tile_bytes)
 def resolve_owned_rs(self,residual,stem):
  matrix=finite(residual,(4,2560));raw=hc.raw(matrix);sequence=len(self.device_rs.records)+1;path=self.operand_root/f'own-rs-seq{sequence:04d}.f32'
  with path.open('xb') as output:output.write(raw)
  result=super().resolve_owned_rs(matrix,stem);row=self.owned_rms_records[-1];require(row['device_sequence']==sequence and row['residual_sha256']==hashlib.sha256(raw).hexdigest(),'Own HC operand/RS ordinal changed');row['owned_operand']={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'shape':[4,2560],'encoding':'LE_F32','role':stem,'sequence':sequence,'origin':'exact independent HC operand at RS call; after owned PLE when applicable','captured_operand_used':False};return result
