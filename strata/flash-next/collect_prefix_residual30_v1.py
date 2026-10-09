#!/usr/bin/env python3
"""Strict source-bound all48 prefix residual coverage; no fullmodel proof."""
import hashlib,json,math,re,argparse
from pathlib import Path
import numpy as np

PHASES=('input','attention','ffn');D=10240;CAP=47185920

def collect(directory,requests,stages,binding,producer_log=None):
 directory=Path(directory);result=[];used=set();by_request={request:[] for request in requests}
 if not requests or len(requests)>4 or not re.fullmatch('[0-9a-f]{64}',binding):raise ValueError('Required bounded request/source roster absent')
 layers=[layer for lo,hi in stages.values() for layer in range(lo,hi)]
 if sorted(layers)!=list(range(48)) or any(not 0<=lo<hi<=48 for lo,hi in stages.values()):raise ValueError('Expectedstage roster mustcoverall48 exactlyonce')
 for path in sorted(directory.glob('p30-*.json')):
  if path.is_symlink():raise ValueError('P30 metadata symlink')
  data=json.loads(path.read_text());request=data['request']
  if request not in requests:raise ValueError('Unmatched P30 request')
  ids=requests[request];expected=stages.get(data['stage'])
  if data['schema']!=1 or expected!=(data['lb'],data['le']) or data['gen_ids']!=ids or data['binding_sha256']!=binding or data['native_hc_source'] is not True or data['full_model_math_qualified'] is not False:raise ValueError('P30 stage/source/input binding differs')
  if data['nonce']!=(((data['pid']&0xffffffff)<<32)|request):raise ValueError('P30 current request nonce differs')
  n=data['rows'];p=data['first_position'];route=data['route']
  if not 1<=len(ids)<=8 or not 1<=n<=8 or p<0 or p+n>len(ids) or data['row_floats']!=D or data['source_extent_bytes']!=n*D*4:raise ValueError('P30 entirematrix/source extent differs')
  if route=='verifier':
   if n!=1 or p!=len(ids)-1:raise ValueError('P30 verifier tokenboundary differs')
  elif route=='prefill':
   if p+n>len(ids)-1:raise ValueError('P30 prefill includes verifier token')
  else:raise ValueError('Unknown P30 actual route')
  fields=data['fields'];want={(layer,phase) for layer in range(*expected) for phase in PHASES};got={(field['layer'],field['phase']) for field in fields}
  if len(fields)!=len(want) or got!=want:raise ValueError('P30 required3phase/allstage layers absent/duplicated')
  bindings=[]
  for field in fields:
   file=directory/Path(field['file']).name
   if file.is_symlink() or str(file) in used or field['bytes']!=n*D*4 or field['encoding']!='LE_F32[rows,4,2560]':raise ValueError('P30 fieldfile/extent/layout differs')
   raw=file.read_bytes()
   if len(raw)!=field['bytes'] or not np.isfinite(np.frombuffer(raw,dtype='<f4')).all():raise ValueError('P30 truncated/nonfinite entirematrix')
   used.add(str(file));bindings.append({**field,'path':str(file),'sha256':hashlib.sha256(raw).hexdigest()})
  by_request[request].append((data,bindings));result.append({'metadata':str(path),'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'binding':data,'fields':bindings})
 for request,ids in requests.items():
  frames=by_request[request]
  if not frames:raise ValueError('Required complete P30 request entirely absent')
  if len({data['pid'] for data,_ in frames})!=1:raise ValueError('P30 mixed producer PID')
  coverage={stage:set() for stage in stages}
  for data,_ in frames:
   for pos in range(data['first_position'],data['first_position']+data['rows']):
    if pos in coverage[data['stage']]:raise ValueError('Duplicated P30 row/stage')
    coverage[data['stage']].add(pos)
  if any(positions!=set(range(len(ids))) for positions in coverage.values()):raise ValueError('P30 required earlierprefill/lastverifier/stage rows incomplete')
 if producer_log is not None:
  log=Path(producer_log).read_text();begins={};resumes={};markers=[]
  for line in log.splitlines():
   match=re.fullmatch(r'SFD request pid=(\d+) request=(\d+) tokens=(\d+) ids=([0-9,-]+) shape=.*',line)
   if match:
    pid,ordinal,tokens=map(int,match.groups()[:3]);ids=[int(token) for token in match[4].split(',')];key=(pid,ordinal)
    if key in begins or tokens!=len(ids):raise ValueError('Duplicate/invalid SFD request binding')
    begins[key]=ids
   match=re.fullmatch(r'SFD resume pid=(\d+) request=(\d+) reused=(-?\d+) evaluated_prompt_pending=1',line)
   if match:
    pid,ordinal,reused=map(int,match.groups());key=(pid,ordinal)
    if key in resumes:raise ValueError('Duplicate SFD resume binding')
    resumes[key]=reused
   match=re.fullmatch(r'P30 frame pid=(\d+) request=(\d+) stage=(\d+) route=(prefill|verifier) rows=(\d+) p0=(\d+) metadata=(.*)',line)
   if match:
    pid,ordinal,stage=map(int,match.groups()[:3]);route=match[4];rows,p0=map(int,match.groups()[4:6]);markers.append((pid,ordinal,stage,route,rows,p0,Path(match[7]).name))
  expected=[]
  for frame in result:
   data=frame['binding'];key=(data['pid'],data['request'])
   if begins.get(key)!=data['gen_ids'] or resumes.get(key)!=0:raise ValueError('Actual SFD source/PID/token/fresh binding absent/differs')
   expected.append((data['pid'],data['request'],data['stage'],data['route'],data['rows'],data['first_position'],Path(frame['metadata']).name))
  if sorted(markers)!=sorted(expected):raise ValueError('Actual P30 frame producer/collector roster differs')
 return {'passed':True,'requests':len(requests),'producer_log_crossbinding_verified':producer_log is not None,'frames':result,'coverage':'all source48 layers x3phases xactual prefixrows, perexpected stage','full_model_math_qualified':False,'scope':'native residual localization only; state/ownreference/lifecycle stillseparate'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--expected-requests',type=Path,required=True);p.add_argument('--expected-stages',type=Path,required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--producer-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserveevidence; usenewoutput')
 requests={int(key):value for key,value in json.loads(a.expected_requests.read_text()).items()};stages={int(key):tuple(value) for key,value in json.loads(a.expected_stages.read_text()).items()};result=collect(a.directory,requests,stages,a.binding_sha256,a.producer_log);result.update(producer_log_sha256=hashlib.sha256(a.producer_log.read_bytes()).hexdigest(),expected_requests_sha256=hashlib.sha256(a.expected_requests.read_bytes()).hexdigest(),expected_stages_sha256=hashlib.sha256(a.expected_stages.read_bytes()).hexdigest(),collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'passed':True,'requests':result['requests'],'full_model_math_qualified':False}))
if __name__=='__main__':main()
