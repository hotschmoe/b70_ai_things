#!/usr/bin/env python3
"""Root owned five-case upload41; original strict lifecycle/source gates kept."""
import ast,copy,hashlib,sys
from pathlib import Path
import run_source_upload_oracle_full_v2 as original
from source41_sdk_binding_v1 import binding,require,qualification_source_binding
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native-source-upload-plan-full-source41-v1.json'
ORIGINAL_SHA='05f7ff786922916d83589657e0e3e20ea24198b6e84b3faa84d9611592bd32ec'
def derived():
 path=Path(original.__file__);require(hashlib.sha256(path.read_bytes()).hexdigest()==ORIGINAL_SHA,'Frozen full390 lifecycle changed');source=path.read_text();require(source.count('native-source-upload-plan-full-v6.json')==1,'Exact original default plan point required');source=source.replace('native-source-upload-plan-full-v6.json','native-source-upload-plan-full-source41-v1.json').replace('receipt = dict(runner_generation=2,','receipt = dict(runner_generation=41,');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main');namespace={**vars(original),'__file__':__file__};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[fresh41-full390-runtime]','exec'),namespace);return namespace['main']
def main():
 qualification_source_binding();require('--oracle-plan'in sys.argv and Path(sys.argv[sys.argv.index('--oracle-plan')+1]).resolve()==PLAN,'Explicit new41 upload plan required');require('--oracle-receipt'in sys.argv,'Fresh41 oracle receipt required');from admit_source_upload41_v1 import admission
 admission(Path(sys.argv[sys.argv.index('--oracle-receipt')+1]));before=binding();rc=derived()();require(binding()==before,'Current41 source/ABI/header proof changed during upload');return rc
if __name__=='__main__':raise SystemExit(main())
