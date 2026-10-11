"""Prefix1 raw numeric port preserving the four-byte PCL transport prefix."""
import ast,copy
from pathlib import Path
from types import SimpleNamespace
import native_qsa41_targets_v1 as original
from source43_num10_zero_case_v1 import IDS

def port():
    node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='numeric'));counts={'prompt_length':0,'PCL_prefix_slice':0}
    class PromptLength(ast.NodeTransformer):
        def visit_Subscript(self,n):
            if isinstance(n.value,ast.Name)and n.value.id=='line'and isinstance(n.slice,ast.Slice)and isinstance(n.slice.lower,ast.Constant)and type(n.slice.lower.value)is int and n.slice.lower.value==4 and n.slice.upper is None and n.slice.step is None:
                counts['PCL_prefix_slice']+=1;return n
            return self.generic_visit(n)
        def visit_Constant(self,n):
            if type(n.value)is int and n.value==4:counts['prompt_length']+=1;return ast.copy_location(ast.Constant(1),n)
            return n
    node=PromptLength().visit(node);original.require(counts=={'prompt_length':7,'PCL_prefix_slice':1},'Exact seven prompt lengths and one unchanged PCL slice required')
    namespace={**vars(original),'qsa':SimpleNamespace(IDS=tuple(IDS))};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[source43-prefix1-PCL-corrected-numeric]','exec'),namespace);return namespace['numeric'],counts,node

def numeric(raw,directory,stages):
    original.require(stages==[(0,48)],'Exactly original43 one-card48 layers required');return port()[0](raw,Path(directory),stages)
