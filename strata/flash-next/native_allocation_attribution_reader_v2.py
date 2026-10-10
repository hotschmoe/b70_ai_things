"""Bounded consumed-log SHA/stat/reread and exact future placement admission."""
import hashlib,json
from pathlib import Path
from native_allocation_attribution_contract_v2 import analyze,require,typed
MAX_LINE=16384;MAX_TOTAL=64<<20

def unique(pairs):
 out={}
 for k,v in pairs:require(k not in out,'Duplicate serialized event key');out[k]=v
 return out

def recollect(path,owners,source_generation,expected_log_sha256):
 path=Path(path);require(path.is_file() and not path.is_symlink() and path.absolute()==path.resolve(),'Original regular confined log required');require(type(expected_log_sha256)is str and len(expected_log_sha256)==64,'Explicit saved original log SHA required');before=path.stat();require(before.st_size<=MAX_TOTAL,'Original total log bound');require(type(owners)is list and 2<=len(owners)<=3,'Host plus one/two native UUID owner scopes required');groups={};items={}
 for item in owners:
  key=typed(item['owner']);require(key not in groups,'Duplicate original owner scope');groups[key]=[];items[key]=item
 require(sum(item['owner']['device_uuid'] is None for item in owners)==1,'Exact host domain required');digest=hashlib.sha256();total=0;count=0
 with path.open('rb') as f:
  fd=Path('/proc/self/fd/'+str(f.fileno())).stat();require((fd.st_dev,fd.st_ino,fd.st_size,fd.st_mtime_ns,fd.st_ctime_ns)==(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns),'Actual opened log descriptor differs')
  while True:
   line=f.readline(MAX_LINE+1)
   if not line:break
   require(len(line)<=MAX_LINE,'Bounded original line required before JSON decode');digest.update(line);total+=len(line);require(total<=MAX_TOTAL,'Bounded actual total original log bytes')
   if line.startswith(b'ALLOC_ATTR_ERROR'):raise ValueError('Actual producer failure')
   if not line.startswith(b'ALLOC_ATTR '):continue
   require(line.endswith(b'\n'),'Complete final original event newline required');count+=1;require(count<=196608,'Bounded owner event total');row=json.loads(line[len(b'ALLOC_ATTR '):],object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')));key=typed(row['owner']);require(key in groups,'Foreign process/device/source owner');groups[key].append(row)
 require(path.stat()==before and total==before.st_size and digest.hexdigest()==expected_log_sha256,'Exact original consumed log SHA/stat/extent required')
 again=hashlib.sha256()
 with path.open('rb')as f:
  while True:
   block=f.read(1<<20)
   if not block:break
   again.update(block)
 require(path.stat()==before and again.hexdigest()==expected_log_sha256,'Original log changed after consumed byte pass');require(all(groups.values()),'Every declared owner event scope must be present');reports=[]
 for key,item in items.items():
  rows=groups[key];require({row['role']for row in rows}==set(item['expected_roles']) and all(row['stage'] in item['expected_stages']for row in rows),'Exact declared stage/API role roster required');reports.append(analyze(rows,item['owner'],item['expected_placements'],source_generation,item['expected_host_roles']))
 return {'original_log_binding':{'path':str(path),'sha256':expected_log_sha256,'bytes':total,'consumed_bytes_directly_SHA_bound':True,'stat_read_reread_guarded':True},'scoped_owner_reports':reports,'native_producer_runtime_qualified':False,'physical_residency_qualified':False,'whole_process_transient_peak_qualified':False,'full_cache_goal_qualified':False}
