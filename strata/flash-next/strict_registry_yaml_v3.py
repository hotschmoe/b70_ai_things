"""Reject duplicate YAML keys; exact original semantic rows/topkeys and append order."""
import yaml
class UniqueLoader(yaml.SafeLoader):pass
def mapping(loader,node,deep=False):
 loader.flatten_mapping(node);out={}
 for key_node,value_node in node.value:
  key=loader.construct_object(key_node,deep=deep)
  if key in out:raise ValueError('Duplicate registry YAML key: '+str(key))
  out[key]=loader.construct_object(value_node,deep=deep)
 return out
UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
def parse(raw):return yaml.load(raw,Loader=UniqueLoader)
def semantic_append(old,baseline_delta,batch_delta,current):
 original=parse(old);base_rows=parse(baseline_delta);batch_rows=parse(batch_delta);actual=parse(current)
 if type(original) is not dict or type(actual) is not dict or type(base_rows) is not list or type(batch_rows) is not list:raise ValueError('Exact oldmapping + list-entry-only append fragments required')
 if len(original['models'])!=149 or len(base_rows)!=4 or len(batch_rows)!=12 or len(actual['models'])!=165:raise ValueError('Actual semantic registry rowcounts mustbe149+4+12=165')
 if list(actual)!=list(original) or {k:v for k,v in actual.items() if k!='models'}!={k:v for k,v in original.items() if k!='models'}:raise ValueError('Every old top-level key/order/value mustremain exact')
 if actual['models']!=original['models']+base_rows+batch_rows:raise ValueError('Every old model row/order/content and exact4+12 semantic append required')
 ids=[row['served_model_id'] for row in actual['models']]
 if len(set(ids))!=165:raise ValueError('Actual model alias uniqueness violated')
 return {'original_semantic_models':149,'C137_semantic_models_added':4,'batch37_semantic_models_added':12,'actual_semantic_models':165,'every_old_top_key_order_value_and_model_row_preserved':True,'duplicate_key_rejected':True,'runtime_or_math_proof_transferred':False}
