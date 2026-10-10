"""Isolated CPU-only worker for exact frozen USM parser bytes, never model input."""
import base64,hashlib,json,os,resource,stat,sys
# Kernel bound applies during every write to regular output sinks.
MAX_OUTPUT=32<<20
resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_OUTPUT,MAX_OUTPUT))
from pathlib import Path
PARSER_SHA='71714821736fe98790b80f85d1ab72ac59f58ee6336298721f0aaee2753ca665'
MAX_INPUT=192<<20
PARSE_COUNTS={'positive':0,'negative':0}
INPUT_SHA=None

def require(ok,message):
 if not ok:raise ValueError(message)
def unique(pairs):
 out={}
 for key,value in pairs:require(key not in out,'Duplicate JSON key');out[key]=value
 return out
def stat5(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def runtime():
 paths={str(Path(sys.executable).resolve())};modules={}
 for name,module in sorted(sys.modules.items()):
  filename=getattr(module,'__file__',None)
  if filename:
   p=Path(filename).resolve()
   if p.is_file():paths.add(str(p));modules[name]=str(p)
   cached=getattr(module,'__cached__',None)
   if cached and Path(cached).is_file():paths.add(str(Path(cached).resolve()))
 # Actual mapped library paths only; no host-wide PID census or environment dump.
 for line in Path('/proc/self/maps').read_text().splitlines():
  fields=line.split(maxsplit=5)
  if len(fields)==6 and fields[5].startswith('/'):
   p=Path(fields[5]);require(p.is_file() and not str(p).endswith(' (deleted)'),'Worker mapped file is not current regular source');paths.add(str(p.resolve()))
 rows=[]
 for path in sorted(paths):
  p=Path(path);before=stat5(p);require(before[2]<=128<<20 and stat.S_ISREG(p.stat().st_mode),'Bounded actual worker runtime file required');raw=p.read_bytes();after=stat5(p);require(before==after and len(raw)==before[2],'Worker runtime changed during byte read');rows.append({'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'stat_before':before,'stat_after':after})
 return {'python_executable':str(Path(sys.executable).resolve()),'python_version':sys.version,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'dont_write_bytecode':sys.flags.dont_write_bytecode,'module_files':modules,'mapped_and_module_files':rows,'output_file_limit_bytes':resource.getrlimit(resource.RLIMIT_FSIZE)[0],'complete_external_runtime_environment_qualified':False}
def process_identity():
 text=Path('/proc/self/stat').read_text();end=text.rfind(')');parts=text[end+2:].split();require(end>=0 and len(parts)>=20,'Worker process stat malformed');return {'pid':os.getpid(),'parent_pid':os.getppid(),'start_ticks':int(parts[19])}

def negatives(text,owners,checked,parse):
 # Deliberate port: unchanged original three mutations, positive already checked.
 require(checked['passed'],'positive trace required before negative controls')
 lines=text.splitlines();indices=[i for i,line in enumerate(lines) if '<--- urUSMFree(' in line]
 require(indices,'free events missing');remove=lines[:];remove.pop(indices[-1]);duplicate=lines[:];duplicate.insert(indices[-1]+1,lines[indices[-1]]);failed=lines[:];failed[indices[-1]]=failed[indices[-1]].replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN')
 out={}
 for name,rows in [('missing_free',remove),('double_free',duplicate),('failed_free',failed)]:PARSE_COUNTS['negative']+=1;out[name]=not parse('\n'.join(rows),owners)['passed']
 return out
def main():
 global INPUT_SHA
 require(sys.flags.isolated==1 and sys.flags.no_site==1 and sys.flags.dont_write_bytecode==1,'Worker requires fresh isolated no-site/no-pyc process')
 raw=sys.stdin.buffer.read(MAX_INPUT+1);require(len(raw)<=MAX_INPUT,'Bounded worker input required');INPUT_SHA=hashlib.sha256(raw).hexdigest();request=json.loads(raw,object_pairs_hook=unique);source=base64.b64decode(request['parser_source_base64'],validate=True);require(hashlib.sha256(source).hexdigest()==PARSER_SHA,'Exact frozen original parser source required');namespace={'__name__':'frozen_logical_free_source'};exec(compile(source,'<immutable-original-usm-parser>','exec'),namespace)
 before=runtime();result=None
 if request['mode']=='parse':
  require(type(request['require_owners'])is bool and type(request['expected_owner_count'])is int and request['expected_owner_count']>=0,'Exact typed parser parameters required');data=base64.b64decode(request['log_base64'],validate=True);text=data.decode('utf-8').replace('\r\n','\n').replace('\r','\n');logical=json.loads(base64.b64decode(request['logical_base64'],validate=True));PARSE_COUNTS['positive']+=1;checked=namespace['parse_trace'](text,request['require_owners']);require(checked['passed'],'Chronological context-matched logical-free trace failed');negative=negatives(text,request['require_owners'],checked,namespace['parse_trace']);require(all(negative.values()),'Logical-free negative mutation was not detected')
  require(logical.get('passed') and set(logical.get('negative_controls',{}))=={'missing_free','double_free','failed_free'} and all(logical['negative_controls'].values()),'Logical-free negative controls incomplete');require(all(logical.get(k)==v for k,v in checked.items()),'Saved logical-free evidence differs from raw trace');require(logical['counts']['owners']==request['expected_owner_count'] and not logical['live'],'Owned allocation/scratch coverage differs or live ledger nonempty');result={'checked':checked,'negative_controls':negative,'original_upload_logical_subset_passed':True,'positive_parse_count':1,'negative_parse_count':3,'outer_source_model_health_or_runtime_gates_proven':False}
 else:require(request['mode']=='probe','Unsupported immutable worker mode')
 after=runtime();require(before==after,'Worker runtime source/library closure changed');print(json.dumps({'schema':1,'passed':True,'input_sha256':hashlib.sha256(raw).hexdigest(),'parser_source_sha256':PARSER_SHA,'runtime':before,'result':result,'parse_counts':PARSE_COUNTS,'process':process_identity()},sort_keys=True,allow_nan=False))
if __name__=='__main__':
 try:main()
 except Exception as exc:print(json.dumps({'schema':1,'passed':False,'error':type(exc).__name__+': '+str(exc),'input_sha256':INPUT_SHA,'parser_source_sha256':PARSER_SHA,'parse_counts':PARSE_COUNTS,'process':process_identity()},allow_nan=False));sys.exit(1)
