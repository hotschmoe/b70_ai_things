"""Child-only extra READY full-byte observation; inherited epoch semantics intact."""
from pathlib import Path
import hashlib
from operation_pack_hash_witness_v2 import PackEpoch,require
from operation_sdk37_byte_witness_v1 import SDK37Epoch
from sdk37_witness_scope_v1 import scope_metadata
from sdk49_operation_scope_v1 import roots
from serial37_canonical_json_v3 import read_unique
class ReadyPackEpoch(PackEpoch):
 def ready_seal(self):
  self._owned();require(self.phase=='admission' and len(self.boundaries)==1,'Exactly one READY pack observation before device seal');self._fresh('ready_complete_byte_recheck');return self.boundaries[-1]
class ReadySDK37Epoch(SDK37Epoch):
 def ready_seal(self):
  self._owned();require(self.phase=='admission' and len(self.boundaries)==1,'Exactly one READY SDK observation before device seal');self._fresh('ready');return self.boundaries[-1]
def pack_for_prepared(directory,max_seconds=10800):
 import c1_serve_controller_combined_v137 as c
 p=read_unique(Path(directory)/'prepared.json');require(Path(p['pack']).resolve()==c.PACK.resolve() and Path(p['pack_receipt']).resolve()==c.INTAKE.resolve(),'Exact original pack paths required');receipt=Path(p['pack_receipt']);raw=receipt.read_bytes();require(hashlib.sha256(raw).hexdigest()==p['pack_receipt_sha256'],'Current pack receipt changed');v=read_unique(receipt);require(receipt.read_bytes()==raw,'Pack receipt changed during roster read');roster=[]
 for name,value in v['RESULT']['files'].items():
  path=Path(name);require(not path.is_absolute() and '..'not in path.parts,'Invalid pack member path');roster.append((c.PACK/path,value['sha256']))
 return ReadyPackEpoch(roster,max_seconds)
def sdk_for_plan(plan,max_seconds=10800):
 roster,receipts=scope_metadata(roots(plan['topology_baselines']['onecard_root'],plan['topology_baselines']['pair_root'],plan['prepared']));return ReadySDK37Epoch(roster,receipts,max_seconds)
