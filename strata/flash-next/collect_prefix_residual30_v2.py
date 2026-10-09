#!/usr/bin/env python3
"""Strict source35 normal-dispatch all48 prefix residual coverage; no fullmodel proof."""
import hashlib,json,math,re,argparse
from pathlib import Path
import numpy as np

PHASES=('input','attention','ffn');D=10240;CAP=47185920
ROUTES={'prefill':(1,'prefill'),'prompt_verifier':(2,'prompt_verify'),'verifier':(3,'target_verify')}

def window_nonce(pid,request,pos,rows,route):
 return ((pid&0xffffffff)<<32)|(request<<24)|(pos<<16)|(rows<<8)|ROUTES[route][0]

def collect(directory,requests,stages,binding,producer_log=None):
 directory=Path(directory);result=[];used=set();by_request={request:[] for request in requests}
 if producer_log is None or directory.is_symlink():raise ValueError('Actual source35 producer log and nonsymlink root required')
 if not requests or len(requests)>4 or any(type(key) is not int or not 1<=key<=4 or len(ids) not in (1,2,4,8) or any(type(token) is not int or not 0<=token<248320 for token in ids) for key,ids in requests.items()) or not re.fullmatch('[0-9a-f]{64}',binding):raise ValueError('Required bounded request/source roster absent')
 layers=[layer for lo,hi in stages.values() for layer in range(lo,hi)]
 if sorted(layers)!=list(range(48)) or any(not 0<=lo<hi<=48 for lo,hi in stages.values()):raise ValueError('Expectedstage roster mustcoverall48 exactlyonce')
 for path in sorted(directory.glob('p30-*.json')):
  if path.is_symlink():raise ValueError('P30 metadata symlink')
  data=json.loads(path.read_text());request=data['request']
  if request not in requests:raise ValueError('Unmatched P30 request')
  ids=requests[request];expected=stages.get(data['stage'])
  if data['schema']!=2 or expected!=(data['lb'],data['le']) or data['gen_ids']!=ids or data['binding_sha256']!=binding or data['native_hc_source'] is not True or data['full_model_math_qualified'] is not False:raise ValueError('P30 stage/source/input binding differs')
  if data['route'] not in ROUTES or data['normal_dispatch']!=ROUTES[data['route']][1] or data['row_coverage_complete'] is not True or not 1<=request<=4 or data['pid']<=0:raise ValueError('Source35 exact normal-dispatch/request/row coverage binding differs')
  if data['nonce']!=window_nonce(data['pid'],request,data['first_position'],data['rows'],data['route']):raise ValueError('P30 current request/window nonce differs')
  n=data['rows'];p=data['first_position'];route=data['route']
  if len(ids) not in (1,2,4,8) or not 1<=n<=8 or p<0 or p+n>len(ids) or data['row_floats']!=D or data['source_extent_bytes']!=n*D*4:raise ValueError('P30 entirematrix/source extent differs')
  if route=='verifier':
   if n!=1 or p!=len(ids)-1:raise ValueError('P30 verifier tokenboundary differs')
  elif route=='prompt_verifier':
   if n not in (1,2) or p+n>len(ids)-1:raise ValueError('P30 earlier actual prompt verifier window differs')
  elif route=='prefill':
   if p+n>len(ids)-1:raise ValueError('P30 prefill includes verifier token')
  else:raise ValueError('Unknown P30 actual route')
  fields=data['fields'];want={(layer,phase) for layer in range(*expected) for phase in PHASES};got={(field['layer'],field['phase']) for field in fields}
  if len(fields)!=len(want) or got!=want:raise ValueError('P30 required3phase/allstage layers absent/duplicated')
  bindings=[]
  for field in fields:
   file=directory/Path(field['file']).name
   if '..' in Path(field['file']).parts or file.is_symlink() or str(file) in used or field['bytes']!=n*D*4 or field['encoding']!='LE_F32[rows,4,2560]':raise ValueError('P30 fieldfile/extent/layout differs')
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
  log=Path(producer_log).read_text();begins={};resumes={};markers=[];selects={};returns={};marker_lines={};begin_lines={};resume_lines={}
  for line_number,line in enumerate(log.splitlines(),1):
   match=re.fullmatch(r'SFD request pid=(\d+) request=(\d+) tokens=(\d+) ids=([0-9,-]+) shape=.*',line)
   if match:
    pid,ordinal,tokens=map(int,match.groups()[:3]);ids=[int(token) for token in match[4].split(',')];key=(pid,ordinal)
    if key in begins or tokens!=len(ids):raise ValueError('Duplicate/invalid SFD request binding')
    begins[key]=ids;begin_lines[key]=line_number
   match=re.fullmatch(r'SFD resume pid=(\d+) request=(\d+) reused=(-?\d+) evaluated_prompt_pending=1',line)
   if match:
    pid,ordinal,reused=map(int,match.groups());key=(pid,ordinal)
    if key in resumes:raise ValueError('Duplicate SFD resume binding')
    resumes[key]=reused;resume_lines[key]=line_number
   match=re.fullmatch(r'P30 frame pid=(\d+) request=(\d+) stage=(\d+) route=(prefill|prompt_verifier|verifier) rows=(\d+) p0=(\d+) metadata=(.*)',line)
   if match:
    pid,ordinal,stage=map(int,match.groups()[:3]);route=match[4];rows,p0=map(int,match.groups()[4:6]);markers.append((pid,ordinal,stage,route,rows,p0,Path(match[7]).name));marker_lines[(pid,ordinal,stage,route,rows,p0)]=line_number
   match=re.fullmatch(r'P30 prompt_select pid=(\d+) request=(\d+) stage=(\d+) pos=(\d+) T=(\d+) ids=([0-9,-]+)',line)
   if match:
    pid,ordinal,stage,p0,rows=map(int,match.groups()[:5]);key=(pid,ordinal,stage,p0,rows)
    if key in selects:raise ValueError('Duplicate actual prompt verifier selection')
    selects[key]=(line_number,[int(token) for token in match[6].split(',')])
   match=re.fullmatch(r'P30 prompt_returned pid=(\d+) request=(\d+) stage=(\d+) pos=(\d+) T=(\d+) rc=(\d+)',line)
   if match:
    pid,ordinal,stage,p0,rows,rc=map(int,match.groups());key=(pid,ordinal,stage,p0,rows)
    if key in returns or rc!=0:raise ValueError('Duplicate/failed actual prompt verifier return')
    returns[key]=line_number
  expected=[];prompt_expected=set()
  for frame in result:
   data=frame['binding'];key=(data['pid'],data['request'])
   if begins.get(key)!=data['gen_ids'] or resumes.get(key)!=0:raise ValueError('Actual SFD source/PID/token/fresh binding absent/differs')
   expected.append((data['pid'],data['request'],data['stage'],data['route'],data['rows'],data['first_position'],Path(frame['metadata']).name))
   if data['route']=='prompt_verifier':
    pkey=(data['pid'],data['request'],data['stage'],data['first_position'],data['rows']);prompt_expected.add(pkey)
    selected=selects.get(pkey);returned=returns.get(pkey);mark=marker_lines.get((data['pid'],data['request'],data['stage'],data['route'],data['rows'],data['first_position']))
    if selected is None or returned is None or not begin_lines[key]<resume_lines[key]<selected[0]<returned<mark or selected[1]!=data['gen_ids'][data['first_position']:data['first_position']+data['rows']]:raise ValueError('Actual normal prompt verifier select/run-return/frame/token association differs')
  if set(selects)!=prompt_expected or set(returns)!=prompt_expected:raise ValueError('Exact prompt verifier window roster absent/foreign')
  if sorted(markers)!=sorted(expected):raise ValueError('Actual P30 frame producer/collector roster differs')
 return {'passed':True,'requests':len(requests),'producer_log_crossbinding_verified':producer_log is not None,'frames':result,'coverage':'all source48 layers x3phases xactual prefixrows, perexpected stage','full_model_math_qualified':False,'normal_dispatch_source_crossbinding_verified':producer_log is not None,'GPU_graph_handle_source_row_association_observed':False,'scope':'source35 actual normal-dispatch residual localization only; GPU graph-handle/source-row association, state/ownreference/lifecycle still separate'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--expected-requests',type=Path,required=True);p.add_argument('--expected-stages',type=Path,required=True);p.add_argument('--binding-sha256',required=True);p.add_argument('--producer-log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserveevidence; usenewoutput')
 requests={int(key):value for key,value in json.loads(a.expected_requests.read_text()).items()};stages={int(key):tuple(value) for key,value in json.loads(a.expected_stages.read_text()).items()};result=collect(a.directory,requests,stages,a.binding_sha256,a.producer_log);result.update(producer_log_sha256=hashlib.sha256(a.producer_log.read_bytes()).hexdigest(),expected_requests_sha256=hashlib.sha256(a.expected_requests.read_bytes()).hexdigest(),expected_stages_sha256=hashlib.sha256(a.expected_stages.read_bytes()).hexdigest(),collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii')
 print(json.dumps({'passed':True,'requests':result['requests'],'full_model_math_qualified':False}))
if __name__=='__main__':main()
