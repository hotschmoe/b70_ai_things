#!/usr/bin/env python3
"""Readonly PID1 mapped backing-file census; never process memory or model bytes."""
import argparse,hashlib,json,os,re,stat
from pathlib import Path
LIB='/pti/lib/libunitrace_tool.so'
def require(ok,message):
 if not ok:raise ValueError(message)
def signature(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def process(text):
 tail=text[text.rfind(')')+2:].split();return {'pid':int(text.split(' ',1)[0]),'start_ticks':int(tail[19])}
def select(maps,lib=LIB):
 rows=[]
 for line in maps.splitlines():
  fields=line.split(None,5)
  if len(fields)<6:continue
  path=fields[5]
  if Path(path.replace(' (deleted)','')).name=='libunitrace_tool.so':
   require(path==lib,'Foreign/deleted mapped injection path refused');region=fields[0].split('-');device=fields[3].split(':');require(len(region)==len(device)==2,'Actual mapping layout required');rows.append({'start':int(region[0],16),'end':int(region[1],16),'permissions':fields[1],'offset':int(fields[2],16),'device_major':int(device[0],16),'device_minor':int(device[1],16),'inode':int(fields[4]),'original_container_path':path})
 require(rows and any('x'in r['permissions']for r in rows),'Actual executable injected mapping missing');return rows
def capture(output,expected_sha256,phase,proc=Path('/proc/1'),library=Path(LIB)):
 require(phase in('after_READY','after_last_DONE_before_QUIT'),'Declared live census boundary required');output=Path(output);require(not output.exists(),'New census output required');output.mkdir(parents=True)
 before=process((proc/'stat').read_text());require(before['pid']==1 and before['start_ticks']>0,'Actual native PID1 identity required');maps=(proc/'maps').read_text();rows=select(maps,str(library));require(library.is_file()and not library.is_symlink()and library.resolve()==library,'Exact mapped regular backing file required');sig=signature(library);require(all(r['inode']==sig[1]and r['device_major']==os.major(sig[0])and r['device_minor']==os.minor(sig[0])for r in rows),'Actual mapping dev/inode differs from backing file');require(sig[2]<=64<<20,'Bounded mapped ELF backing file required')
 payload=library.read_bytes();require(payload[:4]==b'\x7fELF'and hashlib.sha256(payload).hexdigest()==expected_sha256 and signature(library)==sig,'Actual fresh mapped ELF backing bytes differ');(output/'mapped-injection.elf').write_bytes(payload);(output/'proc1.maps').write_text(maps,encoding='ascii');after=process((proc/'stat').read_text());require(after==before and select((proc/'maps').read_text(),str(library))==rows,'Native process/mappings changed during census')
 value={'schema':1,'passed':True,'phase':phase,'process':before,'observer_pid':os.getpid(),'original_container_path':str(library),'mapped_segments':rows,'backing_file_stat':sig,'sha256':expected_sha256,'bytes':len(payload),'exported_file':'mapped-injection.elf','maps_file':'proc1.maps','process_memory_bytes_read':False,'LevelZero_handle_proof':False,'versioned_symbol_origin_queried':False};(output/'receipt.json').write_text(json.dumps(value,indent=2)+'\n',encoding='ascii');return value
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--phase',required=True);a=p.parse_args();capture(a.output,a.expected_sha256,a.phase)
if __name__=='__main__':main()
