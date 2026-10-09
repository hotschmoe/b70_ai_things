#!/usr/bin/env python3
"""Exact host PLE rows/embedding with actual publish->graph submission chronology."""
import argparse,hashlib,json,re,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(directory,requests,binding,producer_log,bounds,device=0):
 directory=Path(directory);log=Path(producer_log);text=log.read_text();frames=[];events=[];sfd={};resumes={}
 if directory.is_symlink() or not requests or len(requests)>4 or not re.fullmatch('[0-9a-f]{64}',binding):raise ValueError('Actual bounded PLEinput source roster required')
 for number,line in enumerate(text.splitlines(),1):
  if line.startswith('PLE_INPUT33 '):
   pairs=dict(re.findall(r'(\w+)=([^ ]+)',line));event=line.split()[1]
   if event not in ('published','submit_begin','submit_returned','terminal'):raise ValueError('Unknown actual PLEinput event')
   events.append((number,event,pairs))
  if line.startswith('SFD request '):
   f=dict(re.findall(r'(\w+)=([^ ]+)',line));key=(int(f['pid']),int(f['request']))
   if key in sfd:raise ValueError('Duplicate actual SFD request')
   sfd[key]=(number,[int(i) for i in f['ids'].split(',')])
  if line.startswith('SFD resume '):
   f=dict(re.findall(r'(\w+)=([^ ]+)',line));key=(int(f['pid']),int(f['request']))
   if key in resumes:raise ValueError('Duplicate actual SFD resume')
   resumes[key]=(number,int(f['reused']))
 seen=set();used=set()
 for path in sorted(directory.glob('ple-input33-*.json')):
  if path.is_symlink():raise ValueError('Metadata symlink refused')
  f=json.loads(path.read_text());request=f['request'];pid=f['pid'];key=(pid,request);ids=requests.get(request)
  if request in seen or ids is None or len(ids) not in (1,2,4,8) or f['gen_ids']!=ids or f['schema']!=1 or f['binding_sha256']!=binding:raise ValueError('Actual request/metadata/source binding differs')
  if f['stage']!=0 or f['device']!=device or (f['lb'],f['le'])!=bounds or f['reused']!=0 or f['epoch']!=(((pid&0xffffffff)<<32)|request) or not f['source_exact'] or f['AR'] is not False or not (f['no_host'] or f['device_plan']) or f['prelaunch_host_publication_only'] is not True or f['full_model_math_qualified'] is not False:raise ValueError('Actual native current route/epoch/stage differs')
  T=f['T'];pos=f['pos']
  if T!=1 or pos!=len(ids)-1 or f['window_tokens']!=ids[pos:pos+T] or f['prev']!=[ids[pos-2] if pos>=2 else -1,ids[pos-1] if pos>=1 else -1] or sfd.get(key,(None,None))[1]!=ids or resumes.get(key,(None,None))[1]!=0:raise ValueError('Actual fresh source window/previous tokens differ')
  if f['embedding_encoding']!='LE_F32[T,2560]' or f['row_ids_encoding']!='LE_U32[T,16]':raise ValueError('Raw source storage encoding differs')
  fields={}
  for stem,bytes_count in [('embedding',T*2560*4),('row_ids',T*16*4)]:
   filename=f[stem+'_file'];rawpath=directory/filename
   if Path(filename).name!=filename or rawpath.is_symlink() or str(rawpath) in used or f[stem+'_bytes']!=bytes_count or rawpath.stat().st_size!=bytes_count or sha(rawpath)!=f[stem+'_sha256']:raise ValueError('Exclusive raw source field/path/size/SHA differs')
   used.add(str(rawpath));fields[stem]={'path':str(rawpath),'bytes':bytes_count,'sha256':sha(rawpath)}
  emb=np.frombuffer(Path(fields['embedding']['path']).read_bytes(),dtype='<f4');rows=np.frombuffer(Path(fields['row_ids']['path']).read_bytes(),dtype='<u4')
  if not np.isfinite(emb).all() or rows.tolist()!=f['row_ids'] or len(rows)!=16*T or np.any(rows>=320001536):raise ValueError('Actual source rows/nonfinite embedding differ')
  matches=[(number,event,pairs) for number,event,pairs in events if int(pairs['pid'])==pid and int(pairs['request'])==request]
  if [event for _,event,_ in matches]!=['published','submit_begin','submit_returned','terminal']:raise ValueError('Actual prelaunch publication/submission events absent/duplicate/out-of-order')
  if not sfd[key][0]<resumes[key][0]<matches[0][0]<matches[1][0]<matches[2][0]<matches[3][0]:raise ValueError('Actual SFDbegin/resume/publication/submission/terminal chronology differs')
  for number,event,pairs in matches:
   if any(int(pairs[k])!=f[k] for k in ('stage','device','epoch','pos','T')):raise ValueError('Actual producer chronology identity differs')
   if event=='published' and pairs['metadata']!=path.name:raise ValueError('Actual published metadata filename differs')
   if event=='submit_returned' and int(pairs['rc'])!=0:raise ValueError('Actual graph submit failed')
   if event=='terminal' and (int(pairs['cancelled'])!=0 or int(pairs['generated'])!=1 or pairs['finish'] not in ('length','stop')):raise ValueError('Actual matching SFDrequest did not finish normally')
  seen.add(request);frames.append({'metadata':str(path),'metadata_sha256':sha(path),'binding':f,'fields':fields,'producer_lines':[n for n,_,_ in matches]})
 if seen!=set(requests) or len(events)!=4*len(requests) or len({f['binding']['pid'] for f in frames})!=1:raise ValueError('Exact complete source33 process/request/event roster differs')
 return {'schema':1,'passed':True,'frames':frames,'producer_log_sha256':sha(log),'source_binding_sha256':binding,'actual_prelaunch_host_input_observed':True,'actual_graph_submission_order_observed':True,'actual_matching_SFD_request_terminal_observed':True,'GPU_consumed_PLE_input_observed':False,'full_model_math_qualified':False,'scope':'Current host row IDs and F32 staged embedding after successful source32 gather, before actual graph submission; actual GPU input consumption/PLE math remains separate.'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('directory','requests','producer-log','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--binding-sha256',required=True);p.add_argument('--lb',type=int,default=0);p.add_argument('--le',type=int,required=True);p.add_argument('--device',type=int,default=0);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve source proof; newoutput required')
 result=collect(a.directory,{int(k):v for k,v in json.loads(a.requests.read_text()).items()},a.binding_sha256,a.producer_log,(a.lb,a.le),a.device);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='ascii');print(json.dumps({'passed':True,'frames':len(result['frames']),'GPU_consumed_PLE_input_observed':False}))
if __name__=='__main__':main()
