"""Original native targets only; explicit historical source admission."""
from explore_full48_original_hc_fma_v1 import *
def native_prefix(run_root,plan,prefix,schedule,model_association):
 _,_,_,admitted=__import__('postboot_p30_historical_reader_v1').finalized_binding(run_root,model_association)['original_result'];root=Path(run_root).resolve();directory=root/'child/p30_on';requests_path=directory/'requests.json';rows=read(requests_path)
 require([row['prefix'] for row in rows]==[1,2,4,8],'Exact finalized native four-prefix roster differs');row=next(row for row in rows if row['prefix']==prefix);ids=row['raw']['ids'];require(ids==plan['prefixes'][str(prefix)]==schedule['accepted_ids'] and len(ids)==prefix and row['raw']['fresh']==1,'Actual fresh accepted prefix differs')
 pid=int(row['meta']['logits'][0]['pid']);ordinal=int(row['meta']['logits'][0]['request']);values={};bindings={};seen_windows=set()
 frames=[frame for frame in admitted['capture']['frames'] if frame['binding']['request']==ordinal]
 stages=row['meta']['coverage']['required_stage_ranges'];expected_windows={(window['position'],window['rows'],stage) for window in schedule['windows'] for stage in map(int,stages)}
 for frame in frames:
  data=frame['binding'];position=data['first_position'];count=data['rows'];stage=data['stage'];key=(position,count,stage);require(key in expected_windows and key not in seen_windows,'Native window/stage route differs from config-derived schedule');seen_windows.add(key)
  window=next(window for window in schedule['windows'] if window['position']==position and window['rows']==count);route='prompt_verifier' if window['normal_dispatch']=='prompt_verify' else 'verifier'
  require(data['pid']==pid and data['gen_ids']==ids and data['schema']==2 and data['route']==route and data['normal_dispatch']==window['normal_dispatch'] and data['nonce']==window_nonce(pid,ordinal,position,count,route),'Native source35 route/current window differs from independently derived config')
  for field in frame['fields']:
   path=Path(field['path']);require(path.resolve().parent==(directory/'p30').resolve() and not path.is_symlink() and sha(path)==field['sha256'] and field['bytes']==count*40960,'Native full matrix raw path/SHA/extent differs')
   raw=path.read_bytes();require(len(raw)==field['bytes'] and hashlib.sha256(raw).hexdigest()==field['sha256'],'Native matrix changed during read');matrix=np.frombuffer(raw,dtype='<f4').reshape(count,4,2560);require(np.isfinite(matrix).all(),'Native nonfinite matrix')
   for offset in range(count):
    key='p%d_l%d_%s'%(position+offset,field['layer'],field['phase']);require(key not in values,'Duplicate native row/layer/phase');values[key]=matrix[offset].copy();bindings[key]={**field,'row_offset':offset,'position':position+offset,'source_rows':count,'source_dispatch':data['normal_dispatch']}
 require(seen_windows==expected_windows and set(values)=={'p%d_l%d_%s'%(position,layer,phase) for position in range(prefix) for layer in range(48) for phase in ('input','attention','ffn')},'All native prefix rows/all48/three phases required')
 head=row['meta']['logits'][0];path=Path(head['path']);require(path.resolve().parent==(directory/'captures').resolve() and not path.is_symlink() and sha(path)==head['sha256'] and path.stat().st_size==993280,'Current native SFD fullhead binding differs');raw=path.read_bytes();require(len(raw)==993280 and hashlib.sha256(raw).hexdigest()==head['sha256'],'Native fullhead changed during read');values['head']=np.frombuffer(raw,dtype='<f4').copy();bindings['head']=head;require(np.isfinite(values['head']).all(),'Nonfinite actual fullhead')
 return ids,values,{'requests_path':str(requests_path),'requests_sha256':sha(requests_path),'native_vectors':bindings,'prefix':prefix,'nativePID':pid,'request':ordinal,'comparison_vector_count':144*prefix+1,'native_inputs_or_routes_used_for_own_computation':False}

def num10_admission(root,model_association):
 from postboot_num10_historical_reader_v1 import finalized_binding
 return finalized_binding(root,model_association)['original_result']
def prefix_binding(root,model_association):
 from postboot_p30_historical_reader_v1 import finalized_binding
 return finalized_binding(root,model_association)['original_result']
