#!/usr/bin/env python3
"""CPU-only supplemental CURRENT-image file census over original raw maps paths."""
import argparse,hashlib,json,os,stat,time
from pathlib import Path
IMAGE='sha256:c388186da30785b302c9c76c0ce8ca5e5351c9783c628177f6f7b9eed2f4ad17'
HELPER_SHA='c2c34be0ef0df97d2dad298c76be8d1be2a3001c3b02828fb899824013a7aa49'
def require(ok,message):
 if not ok:raise ValueError(message)
def file_binding(path,cap):
 path=Path(path);resolved=path.resolve(strict=True);before=resolved.stat();require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=cap,'Bounded resolved regular file required');h=hashlib.sha256();magic=b''
 total=0
 with resolved.open('rb')as f:
  while True:
   block=f.read(1<<20)
   if not block:break
   total+=len(block);require(total<=cap,'Current file grew beyond byte bound')
   if not magic:magic=block[:4]
   h.update(block)
 after=resolved.stat();fields=lambda x:[x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns];require(fields(before)==fields(after),'Current image file changed during census');return {'path':str(path),'resolved':str(resolved),'bytes':before.st_size,'sha256':h.hexdigest(),'stat5':fields(after),'ELF_magic':magic.hex()}
def mapped_paths(text):
 require(type(text)is str and len(text.encode())<=4<<20,'Bounded raw maps required');paths=set()
 for line in text.splitlines():
  row=line.split(None,5)
  if len(row)<6:continue
  path=row[5]
  if '.so' not in Path(path).name and path!='/helper/half37-native-bundle':continue
  require(path.startswith('/') and not path.endswith(' (deleted)') and '\x00' not in path and '..' not in Path(path).parts,'Exact persistent mapped ELF path required');paths.add(path)
 require(paths and len(paths)<=128 and '/helper/half37-native-bundle' in paths,'Actual mapped helper/bounded library roster required');return sorted(paths)
def inside(before,after,helper):
 started=time.time();before=Path(before);after=Path(after);helper=Path(helper);require(str(helper)=='/helper/half37-native-bundle','Exact mounted new helper path required')
 originals={str(p):file_binding(p,4<<20)for p in (before,after)};texts=[]
 for p in (before,after):
  with p.open('rb')as f:raw=f.read((4<<20)+1)
  require(len(raw)<=4<<20 and hashlib.sha256(raw).hexdigest()==originals[str(p)]['sha256'],'Bounded original maps bytes changed');texts.append(raw.decode())
 paths=mapped_paths(texts[0])+mapped_paths(texts[1]);paths=sorted(set(paths));rows={};total=0
 for path in paths:
  row=file_binding(path,256<<20);require(row['ELF_magic']=='7f454c46','Every mapped library/helper must be ELF');total+=row['bytes'];require(total<=1<<30,'Supplemental library byte quota exceeded');rows[path]=row
 require(rows[str(helper)]['sha256']==HELPER_SHA,'Actual mounted helper identity differs')
 for p in (before,after):require(file_binding(p,4<<20)==originals[str(p)],'Original raw maps changed during census')
 return {'schema':1,'started_epoch':started,'finished_epoch':time.time(),'image_required':IMAGE,'maps_bindings':originals,'current_image_file_rows':rows,'helper_sha256':HELPER_SHA,'passed':True,'actual_GPU_touch':False,'actual_model_payload_read':False,'new_replay_library_file_paths_rejoined':True,'runtime_live_process_bytes_observed':False,'transient_or_unmapped_libraries_observed':False,'historical_JIT_or_direct_launch_module_proven':False,'scope':'Supplemental current immutable-image file bytes for saved mapped paths; not a retroactive process-memory/code witness.'}
def recipe(root,compile_root):
 root=Path(root).absolute();compile_root=Path(compile_root).absolute();source=Path(__file__).resolve()
 argv=['docker','run','--name','b70-half37-bundle-library-census-'+str(os.getpid()),'--label','b70.half37.bundle-census.source='+hashlib.sha256(source.read_bytes()).hexdigest(),'--network','none','--user','1000:1000','--memory','2g','--memory-swap','2g','--cpus','2','--pids-limit','128','--entrypoint','/opt/b70-c1-python/bin/python']
 for src,dst in [(source,'/harness/census.py'),(root/'half37.before','/inputs/before'),(root/'half37.after','/inputs/after'),(compile_root/'half37-native-bundle','/helper/half37-native-bundle')]:argv+=['-v',str(src)+':'+dst+':ro']
 argv += ['-e','PYTHONDONTWRITEBYTECODE=1',IMAGE,'/harness/census.py','--inside','--before','/inputs/before','--after','/inputs/after','--helper','/helper/half37-native-bundle'];return argv
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true');p.add_argument('--before',required=True);p.add_argument('--after',required=True);p.add_argument('--helper',required=True);a=p.parse_args();require(a.inside,'Root-owned CPU-only inside mode required');print(json.dumps(inside(a.before,a.after,a.helper),sort_keys=True,allow_nan=False))
