"""Operation-owned admitted bytes; never reread live config into trusted plan."""
import copy,hashlib,json
from pathlib import Path
from serial37_canonical_json_v3 import unique_object,finite_constant,canonical

def require(ok,message):
 if not ok:raise ValueError(message)
class Snapshot:
 def __init__(self,path):
  self.path=Path(path).resolve();self.raw=self.path.read_bytes();self.sha256=hashlib.sha256(self.raw).hexdigest();self.plan=json.loads(self.raw,object_pairs_hook=unique_object,parse_constant=finite_constant);self.value=canonical(self.plan)
 def verify(self,plan=None):
  require(self.path.read_bytes()==self.raw,'Original admitted plan bytes changed');require(canonical(self.plan if plan is None else plan)==self.value,'In-memory admitted plan changed');return self.sha256
 def write(self,path):
  self.verify();Path(path).write_bytes(self.raw);return str(Path(path).resolve())
def write_snapshot(path,plan,raw):
 require(raw is not None and canonical(json.loads(raw,object_pairs_hook=unique_object,parse_constant=finite_constant))==canonical(plan),'Exact admitted plan bytes must match runtime recipe');Path(path).write_bytes(raw)
