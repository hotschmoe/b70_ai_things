"""Live semantic admission bracketed by complete current bytes, no saved PASS."""
import hashlib,os,time
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
from operation_pack_hash_witness_v2 import current_row,stat5
from source_page_watchdog_v3 import guard

def require(ok,msg):
 if not ok:raise ValueError(msg)
MAX_TOTAL_BYTES=32<<30
MAX_FILE_BYTES=4<<30
MAX_FILES=30000
MAX_JSON_BYTES=64<<20
def original_path(value):
 path=Path(value).absolute()
 require(path==path.resolve()and not any(p.is_symlink()for p in(path,*path.parents)),'Original evidence/source path alias refused before resolution')
 return path

PREFIXES=(Path('/mnt/vm_8tb/b70/results'),Path('/mnt/vm_8tb/b70/build'))

def roots_and_files(plan,source_files):
 if 'prepared_sha256'in plan:
  require(original_path(original_path(plan['prepared'])/'prepared.json').stat().st_size<=MAX_JSON_BYTES,'Bounded authenticated prepared JSON before read')
  require(hashlib.sha256(original_path(original_path(plan['prepared'])/'prepared.json').read_bytes()).hexdigest()==plan['prepared_sha256'],'Authenticated current prepared metadata required before traversal')
  require(original_path(original_path(plan['own_producer_root'])/'report.json').stat().st_size<=MAX_JSON_BYTES,'Bounded authenticated own report before read')
  require(hashlib.sha256(original_path(original_path(plan['own_producer_root'])/'report.json').read_bytes()).hexdigest()==plan['own_producer_binding']['report_sha256'],'Authenticated own producer report required before traversal')
 if 'prepared_sha256'in plan:
  import c1_serve_controller_combined_v140_v3 as baseline
  require(original_path(plan['prepared']).is_relative_to(PREFIXES[0])and original_path(plan['own_producer_root']).is_relative_to(PREFIXES[0])and original_path(plan['engine_root']).is_relative_to(PREFIXES[1])and original_path(plan['pack'])==baseline.PACK.resolve(),'Exact qualified results/SDK/pack namespaces before byte traversal required')
 roots={original_path(plan[key])for key in('prepared','engine_root','pack','own_producer_root')};files={original_path(p)for p in source_files};seen=set();pending=[plan]
 while pending:
  value=pending.pop()
  if isinstance(value,dict):
   for key,item in value.items():
    if isinstance(item,str)and item.startswith('/'):
     path=Path(item).absolute()
     if any(path.is_relative_to(root)for root in PREFIXES):
      path=original_path(path)
      if path.is_dir()and(key in('root','prepared')or key.endswith('_root')):roots.add(path.resolve())
      elif path.is_file():files.add(path.resolve())
    elif isinstance(item,(list,dict)):pending.append(item)
  elif isinstance(value,list):pending.extend(v for v in value if isinstance(v,(list,dict)))
 # Follow original report JSON references before the semantic predicate, using
 # bounded metadata only. It cannot import a semantic result or bypass gates.
 changed=True
 while changed:
  changed=False
  candidates={report for root in roots for report in root.glob('*.json')}|{path for path in files if path.suffix=='.json'}
  for report in sorted(candidates):
   report=original_path(report)
   if report in seen:continue
   require(report.stat().st_size<=MAX_JSON_BYTES,'Referenced JSON exceeds explicit64MiB bound before parse: '+str(report)+' bytes='+str(report.stat().st_size))
   seen.add(report);pending=[read_unique(report)]
   while pending:
    value=pending.pop()
    if isinstance(value,dict):
     for key,item in value.items():
      if isinstance(item,str)and item.startswith('/'):
       path=Path(item).absolute()
       if any(path.is_relative_to(prefix)for prefix in PREFIXES):
        path=original_path(path)
        if path.is_dir()and(key in('root','prepared')or key.endswith('_root'))and path not in roots:roots.add(path);changed=True
        elif path.is_file()and path not in files:files.add(path);changed=True
      elif isinstance(item,(list,dict)):pending.append(item)
    elif isinstance(value,list):pending.extend(v for v in value if isinstance(v,(dict,list)))
  require(len(roots)<=128 and len(files)<=MAX_FILES and len(seen)<=MAX_FILES,'Bounded complete historical metadata root/file queue required')
 return sorted(roots),sorted(files)

class ByteEpoch:
 def __init__(self,plan,source_files,model_paths,max_seconds=21600):
  self.owner=os.getpid();self.started=time.monotonic();self.max_seconds=max_seconds;self.phase='entry';self.roots,self.files=roots_and_files(plan,source_files);self.model_paths=[original_path(p)for p in model_paths];self.model_stats=[stat5(p)for p in self.model_paths];self.boundaries=[];self.initial=self.fresh('entry')
 def layout(self):
  files=set(self.files);links=[]
  for root in self.roots:
   require(root.is_dir()and not root.is_symlink(),'Exact original evidence root required')
   for path in root.rglob('*'):
    if '.git'in path.relative_to(root).parts:continue
    require(not path.is_symlink(),'SDK/evidence symlink refused before whole read')
    if path.is_file():files.add(original_path(path))
  require(len(files)<=MAX_FILES and not any(path in files for path in self.model_paths),'Model payload cannot enter metadata/SDK/pack byte epoch')
  sizes=[path.stat().st_size for path in files];require(all(0<=size<=MAX_FILE_BYTES for size in sizes)and sum(sizes)<=MAX_TOTAL_BYTES,'Preregistered perfile/total byte budget before entry read');return sorted(files),sorted(links,key=lambda r:r['path'])
 def fresh(self,label):
  require(os.getpid()==self.owner and time.monotonic()-self.started<=self.max_seconds,'Exact live epoch owner/deadline required');start=time.time();files,links=self.layout();rows=[current_row(path)for path in files];require(self.layout()==(files,links),'Evidence file/link roster changed during complete read');require([stat5(p)for p in self.model_paths]==self.model_stats,'Original model stat changed');pages=guard(self.model_paths[2]);require(pages['passed'],'Original known pages changed');view={'rows':rows,'links':links}
  if hasattr(self,'initial'):require(canonical(view)==canonical(self.initial),'Current complete evidence/SDK/pack bytes changed across semantic bracket')
  self.boundaries.append({'label':label,'owner_pid':self.owner,'started_epoch':start,'finished_epoch':time.time(),'view':view,'model_stats':self.model_stats,'known_pages':pages});return view
 def ready(self):
  require(self.phase=='entry','Exactly one complete semantic admission per process');self.fresh('ready');self.phase='ready';return self.boundaries[-1]
 def predevice(self):
  require(self.phase=='ready','Actual current semantic READY required');self.fresh('predevice');self.phase='sealed';return self.boundaries[-1]
 def finish(self):
  require(self.phase=='sealed','Actual complete-byte predevice seal required');self.fresh('postoperation');self.phase='closed';return {'schema':1,'owner_pid':self.owner,'roots':[str(p)for p in self.roots],'files':[str(p)for p in self.files],'boundaries':self.boundaries,'saved_semantic_PASS_imported':False,'stat_only_evidence_or_pack_validation':False,'model_guard_is_stat_and_known_pages':True,'whole_model_fresh_hash_in_epoch':False,'between_boundaries_mutation_unobserved':True}

def binding(directory,plan,pid,ready,seal):
 directory=Path(directory).resolve();saved=read_unique(directory/'byte-epoch.json');require(type(saved['schema'])is int and saved['schema']==1 and saved['owner_pid']==pid and type(saved['owner_pid'])is int and saved['saved_semantic_PASS_imported']is False and saved['stat_only_evidence_or_pack_validation']is False and saved['model_guard_is_stat_and_known_pages']is True and saved['whole_model_fresh_hash_in_epoch']is False,'Exact live complete-byte witness scope required');rows=saved['boundaries'];require([b['label']for b in rows]==['entry','ready','predevice','postoperation'],'Actual four complete-byte boundaries required');require(hashlib.sha256((directory/'byte-ready.json').read_bytes()).hexdigest()==ready['byte_READY_sha256']and canonical(read_unique(directory/'byte-ready.json'))==canonical(rows[1]),'Original READY byte boundary differs');require(hashlib.sha256((directory/'byte-seal.json').read_bytes()).hexdigest()==seal['current_byte_seal_sha256']and canonical(read_unique(directory/'byte-seal.json'))==canonical(rows[2]),'Original predevice byte boundary differs')
 import native_qsa40_runtime_v7 as runtime
 source_files=[runtime.ROOT/p if not Path(p).is_absolute()else Path(p)for p in read_unique(runtime.PLAN)['files']]+[runtime.PLAN];roots,files=roots_and_files(plan,source_files);require(saved['roots']==[str(p)for p in roots]and saved['files']==[str(p)for p in files],'Actual current authenticated evidence/SDK/pack/source roster differs');previous=0
 for row in rows:
  require(row['owner_pid']==pid and type(row['started_epoch'])in(int,float)and type(row['finished_epoch'])in(int,float)and previous<=row['started_epoch']<=row['finished_epoch'],'Actual complete-byte boundary ordering differs');previous=row['finished_epoch'];require(canonical(row['view'])==canonical(rows[0]['view']),'Recorded entry/semanticREADY/predevice/post bytes differ')
 require(rows[1]['finished_epoch']<=ready['manifest_completed_epoch']and rows[2]['finished_epoch']<=seal['finished_epoch'],'Actual semantic/current-byte completion order differs');lock=read_unique(runtime.HERE/'model-lock.json');model=[runtime.ROOT/lock['destination']/x['path']for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')];require([stat5(p)for p in model]==rows[-1]['model_stats'],'Original model currentstat differs');require(guard(model[2])['passed'],'Original current knownpages failed')
 # Reconstruct exact current file/link roster, then reread every byte. This
 # readonly consumer imports no witness as a new live epoch.
 obj=object.__new__(ByteEpoch);obj.roots=roots;obj.files=files;obj.model_paths=model;current_files,links=obj.layout();view={'rows':[current_row(p)for p in current_files],'links':links};require(canonical(view)==canonical(rows[-1]['view'])and obj.layout()==(current_files,links),'Actual current raw evidence/SDK/pack bytes differ from original witness');return {'actual_complete_bytes_recollected':True,'saved_witness_imported_as_live_epoch':False}


def metadata_preflight(prepared,own):
 # Root-only cheap metadata/file-layout gate. No SDK, pack, model or capture
 # payload bytes are read; alias/size/roster refusal precedes full semantics.
 prepared=original_path(prepared);own=original_path(own);prep=original_path(prepared/'prepared.json');report=original_path(own/'report.json');require(prep.stat().st_size<=MAX_JSON_BYTES and report.stat().st_size<=MAX_JSON_BYTES,'Bounded exact preparation JSON before read');m=read_unique(prep)
 plan={'prepared':str(prepared),'own_producer_root':str(own),'engine_root':str(original_path(Path(m['engine_receipt']).parent)),'pack':str(original_path(m['pack'])),'prepared_sha256':hashlib.sha256(prep.read_bytes()).hexdigest(),'own_producer_binding':{'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest()}}
 roots,files=roots_and_files(plan,[]);obj=object.__new__(ByteEpoch);obj.roots=roots;obj.files=files;obj.model_paths=[];layout,links=obj.layout();return {'roots':[str(p)for p in roots],'explicit_files':len(files),'layout_files':len(layout),'layout_bytes':sum(p.stat().st_size for p in layout),'links':links,'full_payload_bytes_read':False}
