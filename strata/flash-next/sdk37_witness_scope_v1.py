"""Explicit shared SDK/oracle scope across declared baseline prepared files.
Root-only epoch construction reads real executables; source/CPU tests use mocks.
No prepared-field or SHA helper is monkeypatched or imported as saved trust.
"""
from pathlib import Path
from serial37_canonical_json_v3 import canonical
from operation_sdk37_byte_witness_v1 import SDK37Epoch,metadata_roster,fresh_receipts,expected,require

def scope_metadata(prepared_paths):
 paths=[Path(p).resolve() for p in prepared_paths];require(1<=len(paths)<=4 and len(set(paths))==len(paths),'Bounded exact distinct declared prepared roots required');roster=None;receipts={}
 for path in paths:
  current,bindings=metadata_roster(path)
  if roster is None:roster=current
  else:require(canonical(expected(current))==canonical(expected(roster)),'Declared baselines must share exact SDK37/oracle corpus')
  for source,digest in bindings:
   source=Path(source).resolve();require(source not in receipts or receipts[source]==digest,'Declared receipt SHA disagreement');receipts[source]=digest
 require(2<=len(receipts)<=8,'Complete shared metadata input union exceeds bound');return roster,list(receipts.items())

def create_epoch(prepared_paths,max_seconds=900):
 roster,receipts=scope_metadata(prepared_paths);return SDK37Epoch(roster,receipts,max_seconds)

def require_prepared(epoch,prepared_path):
 require(isinstance(epoch,SDK37Epoch),'Explicit fresh current-process SDK37Epoch required');epoch._owned();path=Path(prepared_path).resolve();declared={Path(p).resolve():digest for p,digest in epoch.receipts};require(path in declared,'Prepared file outside explicit operation SDK scope');roster,bindings=metadata_roster(path)
 require(all(Path(p).resolve() in declared and declared[Path(p).resolve()]==digest for p,digest in bindings),'Current baseline receipt not in complete declared SDK scope');epoch.require_current_roster(roster,epoch.receipts);return True

def require_engine(epoch,engine):
 require(isinstance(epoch,SDK37Epoch),'Explicit fresh current-process SDK37Epoch required');epoch._owned();engine=Path(engine).resolve();require(all(Path(epoch.expected['SDK37:'+name]['path'])==engine/'build'/name for name in __import__('operation_sdk37_byte_witness_v1').TARGETS),'Engine outside exact operation SDK scope');epoch.require_current_roster([(role,item['path'],item['sha256']) for role,item in epoch.expected.items()],epoch.receipts);return True

def require_oracle(epoch,engine_receipt,oracle_receipt):
 require_engine(epoch,Path(engine_receipt).parent);declared={Path(p).resolve() for p,d in epoch.receipts};oracle_receipt=Path(oracle_receipt).resolve();require(oracle_receipt in declared and Path(epoch.expected['source37-upload-oracle']['path'])==oracle_receipt.parent/'source-upload-oracle','Oracle outside exact operation shared receipt scope');return True
