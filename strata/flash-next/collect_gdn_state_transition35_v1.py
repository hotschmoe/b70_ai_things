#!/usr/bin/env python3
"""Read-only recollection of synthetic GDN35 raw vectors. Not parent final qualification."""
import argparse,array,json,math,re,sys
from pathlib import Path
from prepare_gdn_state_transition35_v1 import sha,require
S=128;HV=48;C=10240;V=S*HV;ST=S*HV*S;CV=C*3

def collect(raw,log):
 raw=Path(raw);text=Path(log).read_text();rows=[];files=[]
 def read(name,n):
  path=raw/(name+'.f32');require(path.is_file() and not path.is_symlink() and path.stat().st_size==n*4,'Raw component missing/changed shape: '+name)
  data=path.read_bytes();f=array.array('f');f.frombytes(data)
  if sys.byteorder!='little':f.byteswap()
  require(all(math.isfinite(x) for x in f),'Nonfinite raw component: '+name);files.append({'path':str(path),'bytes':len(data),'sha256':sha(path)});return data
 h=read('t1-normalized-qkv',3*C);y=read('t1-output',3*V)
 states=[bytes(ST*4)]+[read('t1-state-after-'+str(n),ST) for n in range(1,4)]
 convs=[bytes(CV*4)]+[read('t1-conv-after-'+str(n),CV) for n in range(1,4)]
 for keep in range(3):
  label='t2-keep'+str(keep)
  components=[('forward-state-unmodified','forward-state',ST,states[0]),('forward-conv-unmodified','forward-conv',CV,convs[0]),('forward-normalized-qkv','forward-normalized-qkv',2*C,h[:2*C*4]),('forward-output','forward-output',2*V,y[:2*V*4]),('accepted-state','committed-state',ST,states[keep]),('accepted-conv','committed-conv',CV,convs[keep]),('carry-state','carry-state',ST,states[keep+1]),('carry-conv','carry-conv',CV,convs[keep+1]),('carry-output','carry-output',V,y[keep*V*4:(keep+1)*V*4])]
  for component,suffix,n,want in components:
   actual=read(label+'-'+suffix,n);matches=re.findall(r'^GDN35_COMPARE case='+label+r' component='+component+r' words=(\d+) differing=(\d+) first=(\d+) max_abs=([^ ]+) bitwise=([01])$',text,re.M)
   require(len(matches)==1 and int(matches[0][0])==n,'Actual comparison log missing/duplicate/shape differs')
   bitwise=actual==want;require((matches[0][-1]=='1')==bitwise,'Actual log/raw bitwise claim differs')
   require(not bitwise or int(matches[0][1])==0,'Claimed bitwise log has differing words')
   rows.append({'case':label,'component':component,'floats':n,'bitwise':bitwise})
 negative=read('negative-modified-state-output',V);rejected=negative!=y[2*V*4:];require(rejected and re.findall(r'^GDN35_NEGATIVE modified_state_output_detected=(\d+)$',text,re.M)==['1'],'Modified-state negative unobserved')
 results=re.findall(r'^GDN35_RESULT comparisons=(\d+) failures=(\d+) synthetic_component_passed=([01]) model_math_qualified=0 all_owned_allocations_freed=1$',text,re.M)
 require(len(results)==1 and int(results[0][0])==27,'Actual component terminal summary missing/duplicate')
 count=sum(not row['bitwise'] for row in rows);require(int(results[0][1])==count and (results[0][2]=='1')==(count==0),'Terminal summary/raw comparison mismatch')
 require('GDN35_ERROR' not in text,'Native component failure retained')
 return {'schema':1,'passed':count==0,'scope':'read-only synthetic currentSYCL component raw parity; parent health/teardown/USM/identity still required','comparisons':rows,'files':files,'negative_modified_state_detected':rejected,'log_sha256':sha(log),'model_math_qualified':False,'native_real_row3_inputs_observed':False,'owned_allocations_free_independently_qualified':False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Output must be new');r=collect(a.raw,a.log);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':r['passed'],'comparisons':len(r['comparisons'])}));return 0 if r['passed'] else 1
if __name__=='__main__':sys.exit(main())
