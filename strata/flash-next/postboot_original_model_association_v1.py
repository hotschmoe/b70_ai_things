"""Named current publisher-byte association; original stat gates stay false."""
import hashlib,math
from pathlib import Path
from serial37_canonical_json_v3 import canonical,read_unique
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def current_stat(path):
 s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def identity_rows(receipt,lock,lock_sha):
 expected=[x for x in lock['files']if x['path'].startswith('UD-Q4_K_XL/')];require(len(expected)==4 and receipt['passed']is True and receipt['lock_sha256']==lock_sha and receipt['model_revision']==lock['revision']and len(receipt['rows'])==4,'Exact complete original publisher4 lock/revision required');require(all(type(receipt[k])in(int,float)and math.isfinite(receipt[k])and receipt[k]>0 for k in('started','finished'))and receipt['started']<=receipt['finished'],'Actual finite original identity epochs required');rows=[]
 for row,want in zip(receipt['rows'],expected):
  path=ROOT/lock['destination']/want['path'];require(row['path']==str(path)and row['passed']is True and type(row['bytes'])is int and row['bytes']==want['size']and row['sha256']==row['expected_sha256']==want['sha256'],'Exact original ordered publisher bytes/hash/path required');before=row['stat_before'];after=row['stat_after'];require(type(before)is list and len(before)==5 and all(type(v)is int for v in before)and before==after and before[2]==want['size'],'Original typed stable stat5 required');rows.append(row)
 return rows

def association(historical_path,current_path,*,current_expected_sha256,current_stat_reader=current_stat):
 historical_path=Path(historical_path).absolute();current_path=Path(current_path).absolute()
 for path in(historical_path,current_path):require(path==path.resolve()and path.is_file()and not path.is_symlink()and path.stat().st_size<=16<<20,'Original bounded nonalias identity receipt required')
 current_raw=current_path.read_bytes();old_raw=historical_path.read_bytes();require(hashlib.sha256(current_raw).hexdigest()==current_expected_sha256,'Exact actual postboot full4 receipt required');old=read_unique(historical_path);new=read_unique(current_path);lock=read_unique(HERE/'model-lock.json');lock_sha=sha(HERE/'model-lock.json');before=identity_rows(old,lock,lock_sha);after=identity_rows(new,lock,lock_sha);require(old['finished']<=new['started'],'Fresh postboot fullhash must follow original source proof');mapping=[]
 for left,right in zip(before,after):
  want=current_stat_reader(right['path']);require(want==right['stat_before']==right['stat_after'],'Current modelstat differs from actual new fullhash');require(left['stat_after'][1:]==right['stat_after'][1:]and left['stat_after'][0]!=right['stat_after'][0],'ONLY st_dev may change in this explicit reboot association');mapping.append({'path':left['path'],'publisher_sha256':left['sha256'],'historical_stat5':left['stat_after'],'current_stat5':right['stat_after'],'changed_index':0})
 require(len({row['historical_stat5'][0]for row in mapping})==len({row['current_stat5'][0]for row in mapping})==1,'Exact coherent original/remounted device mapping required');require(historical_path.read_bytes()==old_raw and current_path.read_bytes()==current_raw,'Original identity receipt changed during association')
 return {'schema':1,'historical_identity_path':str(historical_path),'historical_identity_sha256':hashlib.sha256(old_raw).hexdigest(),'current_identity_path':str(current_path),'current_identity_sha256':current_expected_sha256,'current_lock_sha256':lock_sha,'current_model_revision':lock['revision'],'mapping':mapping,'original_bytes_requalified_by_actual_full4':True,'old_current_stat_gate_passed':False,'old_report_modified':False,'historical_GPU_health_transferred':False,'current_GPU_or_model_runtime_qualified':False,'full_model_math_qualified':False}

def historical_stat(association,path,recorded):
 require(type(recorded)is list and len(recorded)==5 and all(type(v)is int for v in recorded),'Exact historical typed model stat5 required');matches=[row for row in association['mapping']if row['path']==str(Path(path).absolute())];require(len(matches)==1 and matches[0]['historical_stat5']==recorded and current_stat(path)==matches[0]['current_stat5'],'Named st_dev association cannot waive other stat changes');return {'historical_stat_associated_to_current_same_publisher_bytes':True,'old_current_stat_gate_passed':False}
