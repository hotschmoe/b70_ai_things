"""Real role/shape header metadata with analytical SYNTHETIC original row values.
No original payload access. Fast matmul is equivalent to this sparse rows algebra.
"""
import json
from types import SimpleNamespace
import numpy as np
from full48_reference_inventory_fixture_v1 import INVENTORY,inventory_contract
from native_storage_first_gdn_estimate_v1 import f32,bf16_rne
from prefill_first_gdn_owned_storage_v1 import f16_rne

class SyntheticOriginalRoles:
 actual_source=False
 def __init__(self,inventory_path=INVENTORY):
  self.inventory=inventory_contract(inventory_path);data=json.loads(inventory_path.read_text());tensors={t['name']:(None,dict(t),None) for file in data['files'] if '/UD-Q4_K_XL/' in file['path'] for t in file['tensors']};self.reader=SimpleNamespace(tensors=tensors);self.row_reads=0;self.max_rows_bytes=0;self.synthetic_fast_products=0
 def shape(self,name):return tuple(self.reader.tensors[name][1]['shape_ggml_order'])
 def kind(self,name):return self.reader.tensors[name][1]['type']
 def coefficients(self,name,indices):
  indices=np.asarray(list(indices),dtype=np.int64);width=self.shape(name)[0];layer=int(name.split('.')[1]) if name.startswith('blk.') else 0;col=(indices*13+layer)%width;value=(indices%3+1)/2048
  if self.kind(name)=='F32':value=value+.00000023
  return col,value
 def rows(self,name,indices):
  indices=list(indices);width=self.shape(name)[0];self.row_reads+=len(indices);self.max_rows_bytes=max(self.max_rows_bytes,len(indices)*width*8)
  if name=='token_embd.weight':
   cols=np.arange(width);return np.stack([((cols*3+index*7)%31-15)/1024 for index in indices]).astype(np.float64)
  if name=='per_layer_token_embd.weight':
   levels=np.asarray([-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]);return np.stack([levels[(np.arange(width)+index)%16]/2048 for index in indices])
  if len(self.shape(name))==1:
   if name.endswith('ssm_a'):return np.full((len(indices),width),-.5)
   if name.endswith('ssm_dt.bias'):return np.zeros((len(indices),width))
   if 'norm' in name:return np.ones((len(indices),width))
  if name.endswith('conv1d.weight'):return np.tile(np.array([.0625,.125,.25,.5]),(len(indices),1))
  out=np.zeros((len(indices),width));cols,values=self.coefficients(name,indices);out[np.arange(len(indices)),cols]=values;return out
 def synthetic_matmul(self,name,indices,x,weight):
  self.synthetic_fast_products+=1;cols,values=self.coefficients(name,indices)
  if weight=='f16':values=f16_rne(values)
  elif weight=='bf16':values=bf16_rne(values)
  elif weight!='original':raise ValueError('Synthetic weight storage differs')
  return values*x[cols]
