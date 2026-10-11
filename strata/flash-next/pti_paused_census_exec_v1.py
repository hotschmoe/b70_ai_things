"""Same original protected census functions; explicit fresh paused-list tool mount."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import pti_census_owned_exec_v4 as old
EXPECTED_HOST_ROOT='/mnt/vm_8tb/b70/build/pti-unitrace-paused-regular-c388-build-v1/install'
EXPECTED_BIND_ROOT='/b70/build/pti-unitrace-paused-regular-c388-build-v1/install'
EXPECTED_FILESYSTEM_SOURCE=old.EXPECTED_FILESYSTEM_SOURCE
namespace={**vars(old),**globals()}
nodes=[copy.deepcopy(n)for n in ast.parse(Path(old.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name in('command','run')]
exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),__file__+'[fresh-tool-mount-only-protected-census]','exec'),namespace)
command=namespace['command'];run=namespace['run']
