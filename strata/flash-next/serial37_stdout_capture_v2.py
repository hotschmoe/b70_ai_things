"""Owned supervisor stdout capture; completion evidence is collected after join."""
import time
from pathlib import Path
from batch_numerical_proofs_v40 import sha,require

def forward(stream,log,emit,status):
 status['started_epoch']=time.time()
 try:
  for line in stream:log.write(line);log.flush();emit(line)
  status['eof']=True
 except BaseException as exc:status['error']=type(exc).__name__+': '+str(exc)
 finally:status['completed_epoch']=time.time()

def completed(reader,status,path):
 retired=reader is not None and not reader.is_alive();row=dict(status)
 row.update(reader_retired=retired,finished_epoch=time.time(),path=str(Path(path).resolve()),sha256=sha(path) if retired and Path(path).is_file() else None)
 row['passed']=retired and row.get('eof') is True and row.get('error') is None and type(row.get('started_epoch')) in (int,float) and type(row.get('completed_epoch')) in (int,float) and row['started_epoch']<=row['completed_epoch']<=row['finished_epoch']
 return row

def binding(root,parent):
 import math
 root=Path(root).resolve();row=parent['child_stdout_capture'];path=root/'child-supervisor.log'
 require(row['passed'] is True and row['reader_retired'] is True and row['eof'] is True and row['error'] is None and row['path']==str(path) and row['sha256']==sha(path),'Actual supervisor stdout EOF/error/log closure differs')
 require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ('started_epoch','completed_epoch','finished_epoch')) and parent['child_started_epoch']<=row['started_epoch']<=row['completed_epoch']<=row['finished_epoch']<=parent['leaf_logs_epoch']<=parent['post_health_finished_epoch'],'Actual supervisor EOF/capture chronology differs')
 return row
