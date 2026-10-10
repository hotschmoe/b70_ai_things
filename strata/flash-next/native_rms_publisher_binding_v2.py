"""Strict original ordered publisher and page-bracket admission, read-only."""
import math
from pathlib import Path
import hashlib
from source_page_watchdog_v3 import KNOWN_PAGES

def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stat(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def epochs(values):require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values),'Finite positive actual source epochs required')

def publisher_binding(root,report,lock,lock_sha,repository_root):
 root=Path(root).resolve();expected=[r for r in lock['files'] if r['path'].startswith('UD-Q4_K_XL/')];paths=[Path(repository_root)/lock['destination']/r['path'] for r in expected]
 require(len(expected)==len(paths)==4,'Exact ordered original publisher4 required');proof=report['post_full4'];require(proof['passed'] is True and proof['lock_sha256']==lock_sha and proof['model_revision']==lock['revision'] and len(proof['rows'])==4,'Actual full4 lock/revision/complete roster differs')
 epochs([proof['started'],proof['finished'],report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch'],report['finished_epoch']]);boundary=max(report['GPU_terminal_epoch'],report['post_health']['finished_epoch'],report['post_journal']['finished_epoch'])
 require(proof['after_terminal_and_post_health_epoch']==boundary and proof['started']>=boundary and proof['finished']>=proof['started'] and report['finished_epoch']>=proof['finished'],'Actual fresh complete4 afterterminal/health/journal bound differs')
 for row,want,path in zip(proof['rows'],expected,paths):
  require(row['path']==str(path) and row['passed'] is True and row['bytes']==want['size'] and row['sha256']==row['expected_sha256']==want['sha256'] and row['stat_before']==row['stat_after']==stat(path),'Actual ordered publisher byte/hash/path/currentstat differs')
 for key in ('before_post_full4_pages','post_pages'):
  guard=report[key];epochs([guard['epoch']]);require(guard['passed'] is True and guard['path']==str(paths[2]) and guard['stat_before']==guard['stat_after']==proof['rows'][2]['stat_after'] and len(guard['rows'])==2,'Actual both-page source/path/stat/roster differs')
  require([(r['offset'],r['expected_sha256']) for r in guard['rows']]==list(KNOWN_PAGES),'Exact knownpage offsets/expected hashes differ')
  for row in guard['rows']:
   preserved=Path(row['preserved_path']);require(not preserved.is_symlink() and preserved.resolve().parent==root and row['passed'] is True and row['bytes']==4096 and preserved.stat().st_size==4096 and sha(preserved)==row['sha256']==row['expected_sha256'],'Actual preserved source knownpage proof changed/escaped')
 require(report['before_post_full4_pages']['epoch']<=proof['started']<=proof['finished']<=report['post_pages']['epoch']<=report['finished_epoch'],'Actual knownpages must bracket complete publisher4')
 return {'complete_ordered_publisher4_current':True,'actual_full4_after_terminal_health_journal':True,'preserved_pages_bracket_full4':True}
