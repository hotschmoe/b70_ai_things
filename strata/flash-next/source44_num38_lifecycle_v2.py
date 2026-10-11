"""Extent-only private NUM38 port of the immutable UR owner/free grammar."""
import ast,copy
from pathlib import Path
import audit_layer0_capture_lifecycle_v2 as original

FRAME_BYTES=7048204
TARGET_BYTES=7048460

def extent():
    from source44_num38_fields_v2 import field_contract
    rows=field_contract()
    if len(rows)!=38 or sum(r['bytes']for r in rows)!=FRAME_BYTES or FRAME_BYTES+256!=TARGET_BYTES:
        raise ValueError('Actual NUM38 field storage plus256 control extent differs')
    return FRAME_BYTES,TARGET_BYTES

def ports(frame_bytes=FRAME_BYTES,target_bytes=TARGET_BYTES):
    if type(frame_bytes)is not int or type(target_bytes)is not int or(frame_bytes,target_bytes)!=(FRAME_BYTES,TARGET_BYTES):
        raise ValueError('Preregistered NUM38 extent required')
    nodes=[copy.deepcopy(n)for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name in('audit_text','live_blob_bindings')]
    counts={7023048:0,7023304:0}
    class Extent(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value)is int and n.value in counts:
                counts[n.value]+=1
                return ast.copy_location(ast.Name(id='FRAME_BYTES_44'if n.value==7023048 else'TARGET_BYTES_44',ctx=ast.Load()),n)
            if type(n.value)is str:
                return ast.copy_location(ast.Constant(n.value.replace('complete33field','complete38field').replace('single7023304B','single7048460B')),n)
            return n
    nodes=[Extent().visit(n)for n in nodes]
    if counts!={7023048:1,7023304:2}:raise ValueError('Exact original allocation extent sites required')
    namespace={**vars(original),'FRAME_BYTES_44':frame_bytes,'TARGET_BYTES_44':target_bytes,'_ported_AST':nodes}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes,type_ignores=[])),'[NUM38-extent-only-UR-port]','exec'),namespace)
    return namespace

def live_blob_bindings(*args,**kwargs):
    return ports(*extent())['live_blob_bindings'](*args,**kwargs)

def audit_text(text):return ports(*extent())['audit_text'](text)
