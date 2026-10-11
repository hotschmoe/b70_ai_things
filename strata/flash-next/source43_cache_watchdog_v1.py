"""Exact three active known buffered views; no fullhash waiver or repair."""
import ast,copy
from pathlib import Path
import source_page_watchdog_v3 as original
from source41_model_cli_v2 import EXTRA_PAGE
KNOWN_PAGES=(*original.KNOWN_PAGES,EXTRA_PAGE)
namespace={**vars(original),'KNOWN_PAGES':KNOWN_PAGES}
nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_text()).body if isinstance(n,ast.FunctionDef)]
exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),'[same-readonly-three-page-functions]','exec'),namespace)
for name in('signature','inspect_pages','guard','preserve'):globals()[name]=namespace[name]
