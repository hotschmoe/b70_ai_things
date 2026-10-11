"""Source40 exact readline-LF association; immutable original raw records."""
import ast,copy,hashlib
from pathlib import Path
import eos_waiter_terminal_adjudication40_v1 as original
from eos_waiter_terminal_source40_v2 import source_guard
from serial37_canonical_json_v3 import canonical
ORIGINAL_SHA='cc931dc5db8cf80600f6fba5cd71397f71fbfb2b7cfd9226b1908cad50a51d3f'
ORIGINAL_POINT="line=gets[-1]['line'];require(type(line)is str and line.startswith('BDONE ') and line==terminal['line'],'Waiter must actually consume exact native terminal last')"
REPLACEMENT="""queued_line=gets[-1]['line'];require(type(queued_line)is str and queued_line.endswith('\\n')and '\\n'not in queued_line[:-1]and '\\r'not in queued_line,'Exact original readline single LF required');line=queued_line[:-1];require(line.startswith('BDONE ')and type(terminal['line'])is str and '\\n'not in terminal['line']and '\\r'not in terminal['line']and line==terminal['line'],'Waiter must actually consume exact native terminal last')"""
def derived_waiter():
 raw=Path(original.__file__).read_bytes();original.require(hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA,'Frozen original waiter source changed');source=raw.decode('ascii');original.require(source.count(ORIGINAL_POINT)==1,'Exact original terminal source point changed');tree=ast.parse(source.replace(ORIGINAL_POINT,REPLACEMENT));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Terminals');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='waiter_terminal_binding');namespace=dict(vars(original));exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[source40-exact-readline-LF]','exec'),namespace);return namespace['waiter_terminal_binding']
class Terminals(original.Terminals):
 waiter_terminal_binding=derived_waiter()
 def association(self,call):return source_guard(super().association(call))
def adjudicate(events,observer_proof):
 before=canonical(events);node=next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='adjudicate');namespace={**vars(original),'Terminals':Terminals};exec(compile(ast.fix_missing_locations(ast.Module([copy.deepcopy(node)],type_ignores=[])),'[source40-readline-LF-adjudication]','exec'),namespace);value=namespace['adjudicate'](events,observer_proof)
 for association in value['actual_API_terminal_associations']:source_guard(association)
 original.require(canonical(events)==before,'Original raw terminal records cannot be rewritten')
 return {**value,'actual_source40_EOS_precedence_verified':True,'actual_readline_single_LF_verified':True,'frozen_waiter_adjudicator_bytes_changed':False}
