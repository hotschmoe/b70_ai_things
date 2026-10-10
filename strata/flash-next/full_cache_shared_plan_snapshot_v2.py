"""Captured source39 plan bytes across parent exclusion and child boundaries."""
import hashlib,json,stat
from pathlib import Path
from serial37_canonical_json_v3 import unique_object,finite_constant,canonical

def require(ok,message):
 if not ok:raise ValueError(message)

class Snapshot:
 def __init__(self,path,expected=None):
  unresolved=Path(path);require(not unresolved.is_symlink() and stat.S_ISREG(unresolved.stat().st_mode),'Original admitted plan must be regular and nonsymlink')
  self.path=unresolved.resolve();before=self.path.stat();self.raw=self.path.read_bytes();after=self.path.stat()
  require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and len(self.raw)==after.st_size,'Plan changed during admission read')
  self.sha256=hashlib.sha256(self.raw).hexdigest();require(expected is None or expected==self.sha256,'Actual prelease/child admitted byte digest differs')
  self.plan=json.loads(self.raw,object_pairs_hook=unique_object,parse_constant=finite_constant);self.value=canonical(self.plan)
  self.verify()
 def verify(self):
  require(not self.path.is_symlink() and self.path.is_file() and self.path.read_bytes()==self.raw and canonical(self.plan)==self.value,'Actual captured plan bytes or value changed')
  return self.sha256
 def write(self,path):
  self.verify();target=Path(path);require(not target.exists(),'Fresh admitted plan snapshot required');target.write_bytes(self.raw)
  require(target.read_bytes()==self.raw,'Written captured plan differs');return str(target.resolve())

def write_runtime(path,plan,raw):
 require(type(raw)is bytes and canonical(json.loads(raw,object_pairs_hook=unique_object,parse_constant=finite_constant))==canonical(plan),'Actual captured child plan bytes required')
 Path(path).write_bytes(raw)
