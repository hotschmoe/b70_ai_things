"""Explicit frontend live-owner namespace; not a stock native cache key."""
import hashlib,json,stat
from pathlib import Path
KEYS=('model_sha256','tokenizer_sha256','template_sha256','source_sha256')
FIELD='strata_identity_namespace'
def require(ok,message):
 if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def namespace(value):
 require(type(value)is dict and set(value)==set(KEYS),'Exact full loaded identity namespace required')
 require(all(type(value[k])is str and len(value[k])==64 and all(c in '0123456789abcdef'for c in value[k])for k in KEYS),'Exact namespace hex hashes required')
 return dict(value)
def admit(owner,request):
 loaded=namespace(owner);require(request is None or type(request)is dict,'Original request object required')
 explicit=request is not None and FIELD in request;supplied=request[FIELD]if explicit else None
 candidate=namespace(supplied)if explicit else loaded
 changed=[k for k in KEYS if candidate[k]!=loaded[k]]
 require(not changed,'LIVE_IDENTITY_NAMESPACE_REFUSED '+','.join(changed))
 return {'loaded_namespace':loaded,'loaded_namespace_sha256':digest(loaded),'explicit_namespace_supplied':explicit,'native_cache_key_model_fingerprint_observed':False,'other_weights_or_tokenizer_loaded':False}
def watched_bytes(rows):
 require(type(rows)is list and 1<=len(rows)<=32 and len({r['path']for r in rows})==len(rows),'Exact declared watched current-byte roster required')
 require(all(type(r['bytes'])is int and 0<=r['bytes']<=64<<20 for r in rows) and sum(r['bytes']for r in rows)<=128<<20,'Bounded metadata/source/receipt/ELF bytes required; no whole model shards per request')
 for row in rows:
  path=Path(row['path']);require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),'Original watched regular file required');before=path.stat();require(before.st_size==row['bytes'],'LIVE_IDENTITY_CURRENT_BYTES_REFUSED declared extent '+str(path));raw=path.read_bytes();after=path.stat();require(before==after and len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'] and path.read_bytes()==raw and path.stat()==after,'LIVE_IDENTITY_CURRENT_BYTES_REFUSED '+str(path))
 return True

def namespace_probes(owner):
 original=namespace(owner);result=[]
 for key in KEYS:
  foreign=dict(original);foreign[key]=('0'if original[key][0]!='0'else'1')+original[key][1:]
  result.append({'component':key,'requested_namespace':foreign,'loaded_other_weights':False,'expected_refusal':'LIVE_IDENTITY_NAMESPACE_REFUSED '+key})
 return result
