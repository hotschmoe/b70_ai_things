"""Bounded readonly sparse raw protocol receipt; never loads whole JSON trace."""
import hashlib,json
from pathlib import Path
PREFIXES=('T ','BT ','DONE ','BDONE ','BADM ','SBF batch_event ','SBF resume ','SBF migration_','BCHAIN restored ','BCPUBLIC reset ','PCL phase_begin ')
SEND=('GEN ','BGEN ','STOP','BSTOP ','QUIT')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(p):
 s=Path(p).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def collect(child):
 child=Path(child).resolve();report=json.loads((child/'report.json').read_bytes());path=child/'api-native-trace.jsonl';before=signature(path);selected=[];digest=hashlib.sha256();selected_bytes=0
 with path.open('rb') as f:
  for number,line in enumerate(f,1):
   digest.update(line)
   if not any(s.encode() in line for s in (*PREFIXES,*SEND,'engine_begin','engine_end')):continue
   row=json.loads(line);kind=row['kind'];body=row.get('line','');keep=kind in ('engine_begin','engine_end') or kind=='native_receive' and body.startswith(PREFIXES) or kind=='native_send' and body.startswith(SEND)
   if not keep:continue
   selected_bytes+=len(line)
   if len(selected)>=1024 or selected_bytes>8<<20:raise ValueError('Bounded source protocol receipt exceeded')
   selected.append({'original_line':number,'raw_line_sha256':hashlib.sha256(line).hexdigest(),'record':row})
 if signature(path)!=before or digest.hexdigest()!=report['artifact_bindings']['api-native-trace.jsonl']['sha256'] or before[2]!=report['artifact_bindings']['api-native-trace.jsonl']['bytes']:raise ValueError('Original trace changed or actual producer trace binding differs')
 begins=[r['record'] for r in selected if r['record']['kind']=='engine_begin'];ends=[r['record'] for r in selected if r['record']['kind']=='engine_end']
 if len(begins)!=4 or len(ends)!=4 or {r['call'] for r in begins}!={r['call'] for r in ends}:raise ValueError('Exact actual4call roster required')
 return {'schema':1,'source_sha256':sha(__file__),'original_child':str(child),'original_report_sha256':sha(child/'report.json'),'original_trace_sha256':digest.hexdigest(),'original_trace_stat5':before,'sparse_protocol_records':selected,'original_error_preserved':report['error'],'original_collection_qualified':False,'underlying_EOS_cause_established':False,'matched_fresh235_control_still_required':True,'GPU_touch':False,'model_payload_read':False,'original_writes':False}
