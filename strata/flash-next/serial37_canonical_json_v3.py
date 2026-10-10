"""Exact JSON-domain equality, preserving fields and rejecting key collisions.
Only tuple/list and integer JSON object-key representation changes are allowed.
"""
import json
from pathlib import Path

def unique_object(pairs):
 out={}
 for key,value in pairs:
  if key in out:raise ValueError('Duplicate serialized JSON object key '+key)
  out[key]=value
 return out

def finite_constant(value):raise ValueError('Nonfinite JSON constant '+value)

def types(value):
 if isinstance(value,dict):
  for key,child in value.items():
   if type(key) not in (str,int):raise ValueError('Only string/integer JSON object keys supported')
   types(child)
 elif type(value) in (tuple,list):
  for child in value:types(child)
 elif value is not None and type(value) not in (str,bool,int,float):raise ValueError('Unsupported JSON value type')

def canonical(value):
 types(value)
 encoded=json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(',',':'))
 normalized=json.loads(encoded,object_pairs_hook=unique_object,parse_constant=finite_constant)
 return json.dumps(normalized,ensure_ascii=True,allow_nan=False,sort_keys=True,separators=(',',':'))

def read_unique(path):return json.loads(Path(path).read_bytes(),object_pairs_hook=unique_object,parse_constant=finite_constant)

def matches_saved(actual,path):return canonical(actual)==canonical(read_unique(path))
