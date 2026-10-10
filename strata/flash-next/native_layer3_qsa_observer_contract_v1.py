"""Pure CPU quota/lineage admission for proposed native QSA targets only."""
import os, re
import hashlib
import numpy as np
from pathlib import Path
HEADER=Path(__file__).with_name('layer3_qsa_target_observer_v1.hpp')
IDS=(248045,8678,198,15666)
WINDOWS=(('prompt_verifier',0,2),('prompt_verifier',2,1),('verifier',3,1))

def require(ok,msg):
 if not ok:raise ValueError(msg)

def fields():
 rows=re.findall(r'\{"([A-Za-z0-9_]+)",(\d+),"([^"]+)",(true|false)\}',HEADER.read_text())
 require(len(rows)==29 and len({r[0]for r in rows})==29,'Exact29 native field contract required')
 return {name:{'bytes':int(size),'encoding':enc,'rows':flag=='true'} for name,size,enc,flag in rows}

def quota(owner,rows):
 require(type(owner)is bool and type(rows)is int and rows in(1,2),'Typed owner/rows required')
 return {name:row['bytes']*(rows if row['rows']else 1) for name,row in fields().items()}if owner else{}

def nonce(pid,request,stage,position,rows):
 require(all(type(v)is int for v in(pid,request,stage,position,rows))and 0<pid<2**32 and 1<=request<=4 and stage in(0,1)and 0<=position<=3 and rows in(1,2),'Exact current native nonce coordinates required')
 return(pid<<32)|(request<<24)|(position<<16)|(rows<<8)|(stage+1)

def admit(frame,expected_binding):
 require(type(frame['schema'])is int and frame['schema']==1 and frame['layer']==3 and frame['binding_sha256']==expected_binding,'Current source binding/schema/layer required')
 owner=frame['owner'];stage=frame['stage'];lb=frame['lb'];le=frame['le']
 require(type(owner)is bool and all(type(v)is int for v in(stage,lb,le))and stage in(0,1) and (lb,le)in((0,48),(0,32),(32,48))and owner==(lb<=3<le)and(owner==(stage==0)),'Actual stage inventory and layer3 ownership required')
 route,p0,rows=frame['route'],frame['first_position'],frame['rows']
 require(type(p0)is int and type(rows)is int and(route,p0,rows)in WINDOWS,'Exact wholeprefix2/1/1 windows required')
 require(frame['nonce']==nonce(frame['pid'],frame['request'],stage,p0,rows)and frame['device_nonce_observed']is owner,'Actual current graph nonce/zero-quota source acknowledgement required')
 require(frame['captures_are_math_inputs']is False and frame['full_model_math_qualified']is False and frame['internal_norm_argument_or_rsqrt_observed']is False and frame['internal_attention_score_softmax_observed']is False and frame['forward_state_is_committed_state']is False and frame['unused_padding_is_native_math_target']is False,'Capture scope cannot become native arithmetic or committed-state authority')
 expected=quota(owner,rows);actual=frame['fields']
 require(len(actual)==len(expected)and len({f['name']for f in actual})==len(actual)and{name['name']for name in actual}==set(expected),'Complete unique actual native field quota required')
 for item in actual:
  require(type(item['bytes'])is int and item['bytes']==expected[item['name']]and item['encoding']==fields()[item['name']]['encoding'],'Exact native field extent/encoding required')
 if owner:
  require(type(frame['n_pages'])is int and frame['n_pages']>0 and frame['n_slots']==frame['n_pages']and frame['max_cells']==2048,'Actual resident allocation bounds required')
  require(frame['indexer_before_active_pooled_rows']==(p0//4+1 if p0 else 0)and frame['indexer_after_active_pooled_rows']==(p0+rows)//4+1,'Actual forward active pooled range differs')
 return {'native_field_quota_admitted':True,'capture_targets_only':True,'native_math_or_logical_pool_view_qualified':False}

def logical_page0(page_before,page_resolved):
 require(type(page_before)is int and type(page_resolved)is int and page_before==page_resolved==0,'Actual page mapping must prove0 before logical cell comparison')
 return {'physical_page0_logical_view_admitted':True,'identity_mapping_inferred_from_geometry':False}

def recollect(frame_path,expected_binding):
 """Future root reader. Tests use synthetic bytes; source author reads no captures."""
 from serial37_canonical_json_v3 import read_unique
 path=Path(frame_path).resolve();before=path.stat();metadata_raw=path.read_bytes();frame=read_unique(path)
 require(path.read_bytes()==metadata_raw and path.stat()==before,'Actual metadata changed during recollection')
 admission=admit(frame,expected_binding);raw_fields={};bindings={}
 for item in frame['fields']:
  name=item['name'];target=path.parent/item['file']
  require(Path(item['file']).name==item['file'] and target.is_file()and not target.is_symlink(),'Confined regular native field required')
  stat=target.stat();raw=target.read_bytes();require(len(raw)==item['bytes']and target.read_bytes()==raw and target.stat()==stat,'Exact consumed native field bytes/stat required')
  raw_fields[name]=raw;bindings[name]={'path':str(target),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if frame['owner']:
  page_before=int(np.frombuffer(raw_fields['page_before'],dtype='<i4')[0]);page_resolved=int(np.frombuffer(raw_fields['page_resolved'],dtype='<i4')[0]);logical_page0(page_before,page_resolved)
  steps=np.frombuffer(raw_fields['step'],dtype='<i4').reshape(frame['rows'],4)
  selected=np.frombuffer(raw_fields['selected_ids_capacity4'],dtype='<i4').reshape(frame['rows'],4)
  for row,step in enumerate(steps):
   position=frame['first_position']+row;width=int(step[3])
   require(1<=width<=4 and np.array_equal(step,np.asarray([position,position+1,(position+1)//4,position+1],dtype='<i4')),'Actual step/active selection width must match bounded native window')
   require(np.array_equal(selected[row,:width],np.arange(position+1,dtype='<i4')),'Actual selected IDs differ from own bounded all-cell target roster')
  for name,raw in raw_fields.items():
   spec=fields()[name]
   if spec['encoding'].startswith('LE_F32'):require(np.isfinite(np.frombuffer(raw,dtype='<f4')).all(),'Actual active F32 target must be finite')
   if name.startswith('K_physical')or name.startswith('V_physical'):
    active=frame['first_position']+(frame['rows']if name.endswith('after_resolve')else 0)
    value=np.frombuffer(raw,dtype='<f2').reshape(2,4,256)
    require(np.isfinite(value[:,:active]).all(),'Actual active logical FP16 cell must be finite')
 return {**admission,'metadata_sha256':hashlib.sha256(metadata_raw).hexdigest(),'raw_fields':bindings,
         'physical_page0_logical_view_admitted':frame['owner'],'selection_padding_used_as_math_target':False,
         'actual_owned_runtime_or_full_model_qualification':False}

def whole_prefix(frames,expected_binding,paired):
 require(type(paired)is bool and len(frames)==(6 if paired else 3),'Complete wholeprefix/stage quota required')
 stages={0:(0,32),1:(32,48)}if paired else{0:(0,48)}
 roster={(stage,route,pos,rows)for stage in stages for route,pos,rows in WINDOWS}
 actual=set();owners=set()
 for frame in frames:
  admit(frame,expected_binding);key=(frame['stage'],frame['route'],frame['first_position'],frame['rows'])
  require(key in roster and key not in actual and(frame['lb'],frame['le'])==stages[frame['stage']],'Exact unique native stage/window inventory required')
  actual.add(key);owners.add((frame['pid'],frame['request']))
 require(actual==roster and len(owners)==1,'Native owner/request cannot be borrowed across wholeprefix')
 return {'wholeprefix_three_windows_all_stages_complete':True,'stage1_nonowner_zero_quota_explicit':paired,
         'native_math_or_runtime_qualified':False}

class FailureModel:
 """Synthetic descriptor/free effects, not a C++/SYCL runtime proof."""
 def __init__(self,directory,allocate):
  self.fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);self.released=False;self.success_markers=0
  try:self.resource=allocate()
  except BaseException:os.close(self.fd);self.fd=-1;raise
 def release(self,free):
  if self.released:return
  free(self.resource);self.released=True;self.success_markers+=1
 def close(self,free):
  try:self.release(free)
  finally:
   if self.fd>=0:os.close(self.fd);self.fd=-1

class Recorder:
 """Synthetic row/nonce effects of the proposed C++ record protocol."""
 @staticmethod
 def create(enabled,owner,allocator):
  require(type(enabled)is bool and type(owner)is bool,'Typed startup flag/owner required')
  if not enabled:return None
  return Recorder(owner,allocator)
 def __init__(self,owner,allocator):
  self.owner=owner;self.resource=allocator()if owner else None;self.sealed=set();self.current=None
 def begin_capture(self,rows):
  quota(self.owner,rows);self.rows=rows;self.seen=set()
 def copy(self,name,first,count):
  require(self.owner and name in fields()and all(type(v)is int for v in(first,count))and first>=0 and count>0 and first+count<=self.rows,'Actual owner/copy bounds required')
  expected=fields()[name]
  require(expected['rows']or(first,count)==(0,1),'State/model fields have single-window quota')
  slots=[(name,i)for i in(range(first,first+count)if expected['rows']else[0])]
  require(not any(s in self.seen for s in slots),'Duplicate recorded field/row')
  self.seen.update(slots)
 def seal(self):
  expected={(n,i)for n,f in fields().items()for i in(range(self.rows)if f['rows']else[0])}if self.owner else set()
  require(self.seen==expected,'Complete actual recorded field roster required');self.sealed.add(self.rows)
 def begin(self,pid,request,stage,route,pos,rows):
  require((route,pos,rows)in WINDOWS and self.owner==(stage==0),'Exact native window/stage ownership required')
  self.current=(nonce(pid,request,stage,pos,rows),rows)
 def returned(self,marker):
  require(self.current is not None and self.current[1]in self.sealed and(marker==self.current[0]if self.owner else marker is None),'Fresh sealed device marker or explicit nonowner acknowledgement required')
  self.current=None;return {'source_record_admitted':True,'actual_SYCL_or_model_execution':False}
