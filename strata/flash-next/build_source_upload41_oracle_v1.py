#!/usr/bin/env python3
"""Root fresh41 header/library compile, deliberate private frozen-builder port."""
import ast,copy,hashlib,sys
from pathlib import Path
import build_source_upload_oracle_full as original
from source41_sdk_binding_v1 import binding,SDK,require,qualification_source_binding
HERE=Path(__file__).resolve().parent
PLAN=HERE/'native-source-upload-plan-full-source41-v1.json'
ORIGINAL_SHA='5dc477c1a85b9e58deb154e4fb63ce2129e30859e7ed1f60ddacd3e0be34db15'
def derived():
 path=Path(original.__file__);require(hashlib.sha256(path.read_bytes()).hexdigest()==ORIGINAL_SHA,'Frozen full390 builder changed');source=path.read_text();require(source.count('native_source_upload_gpu_oracle_full.cpp')==1 and source.count('native-source-upload-plan-full.json')==1,'Exact original oracle/plan source points required');source=source.replace('native_source_upload_gpu_oracle_full.cpp','native_source_upload41_gpu_oracle_v1.cpp').replace('native-source-upload-plan-full.json','native-source-upload-plan-full-source41-v1.json');node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='main');namespace={**vars(original),'__file__':__file__};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[fresh41-full390-build]','exec'),namespace);return namespace['main']
def main():
 qualification_source_binding();require('--engine-receipt'in sys.argv and Path(sys.argv[sys.argv.index('--engine-receipt')+1]).resolve()==SDK/'receipt.json','Actual new41 receipt required before compiler');require('--plan'in sys.argv and Path(sys.argv[sys.argv.index('--plan')+1]).resolve()==PLAN,'Explicit new41 upload plan required');before=binding();rc=derived()();require(binding()==before,'Current41 source/ABI/header proof changed during compile');return rc
if __name__=='__main__':raise SystemExit(main())
