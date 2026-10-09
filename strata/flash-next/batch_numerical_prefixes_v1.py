#!/usr/bin/env python3
"""Actual consumed-prefix jobs and strict supplied-prefix serial raw comparisons."""
import hashlib,json,math,re,struct
from pathlib import Path

def token_sha(ids):return hashlib.sha256(b''.join(struct.pack('<I',t) for t in ids)).hexdigest()

def serial_jobs(trace,roster):
 jobs={};vectors={}
 for line in trace.splitlines():
  if not line.startswith('SBF vector '):continue
  f=dict(re.findall(r'(\w+)=([^ ]+)',line));rid=int(f['rid']);layer=int(f['layer']);phase=f['phase'];role='admission' if phase.startswith('admission_') else 'later' if phase.startswith('batch_step_') else 'solo_migration' if phase.startswith('solo_migration_') else None
  if role is None:raise ValueError('Unknown supplied observation role')
  key=(rid,role);position=int(f['pos']);token=int(f['token'])
  # Initial native BGEN controls; API migration uses its exact GEN submissions
  # separately rather than pretending old BGEN prompt length proves that history.
  if role=='solo_migration':raise ValueError('Migration needs actual API/native segment GEN prefix binding')
  ids=roster.prefix(rid,position,token)
  current={'rid':rid,'role':role,'ids':ids,'ids_sha256_le32':token_sha(ids),'position':position,'token':token,'max_new':1,'fresh_serial':True,'original_own_state_math_reference':False}
  if key in jobs and jobs[key]!=current:raise ValueError('Cross-stage/cross-layer consumed input prefix differs')
  jobs[key]=current
  if (key,layer) in vectors:raise ValueError('Duplicate per-request raw layer/head')
  vectors[key,layer]=f
 for key in jobs:
  if set(l for (k,l) in vectors if k==key)!=set(range(-1,48)):raise ValueError('Every serial job requires actual full48/head coverage')
 return {'jobs':list(jobs.values()),'supplied_consumed_prefix_only':True,'full_model_math_qualified':False}

def compare_raw(batch_path,serial_path):
 a=Path(batch_path).read_bytes();b=Path(serial_path).read_bytes()
 if len(a)!=len(b) or len(a)%4:raise ValueError('Matched raw byte geometry differs')
 x=[v[0] for v in struct.iter_unpack('<f',a)];y=[v[0] for v in struct.iter_unpack('<f',b)]
 if not all(math.isfinite(v) for v in x+y):raise ValueError('Nonfinite native/serial values')
 err=[u-v for u,v in zip(x,y)];nmse=sum(v*v for v in err)/max(1e-30,sum(v*v for v in y));maximum=max(map(abs,err),default=0)/max(1e-6,max(map(abs,y),default=0));first=next((i for i,(u,v) in enumerate(zip(struct.iter_unpack('<I',a),struct.iter_unpack('<I',b))) if u!=v),None)
 return {'bitwise_equal':a==b,'first_different_float':first,'nmse':nmse,'max_normalized':maximum,'acceptance':'exact byte equality; metrics diagnostic only, never tolerance widening','batch_sha256':hashlib.sha256(a).hexdigest(),'serial_sha256':hashlib.sha256(b).hexdigest()}

def compare_all(batch_vectors,serial_vectors):
 if set(batch_vectors)!=set(serial_vectors):raise ValueError('Matched exact RID/role/layer roster differs')
 rows={str(k):compare_raw(batch_vectors[k],serial_vectors[k]) for k in batch_vectors}
 return {'passed':all(r['bitwise_equal'] for r in rows.values()),'comparisons':rows,'original_own_state_fp64_qualified':False,'full_model_math_qualified':False,'scope':'actual supplied full consumed prefixes, native serial/batch raw equality only'}
