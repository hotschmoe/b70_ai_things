"""Source-bound final original CPU terminal placement, never a measurement substitute."""
import ast,math
from pathlib import Path
def require(ok,msg):
 if not ok:raise ValueError(msg)
def source_contract(path):
 tree=ast.parse(Path(path).read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');body=next(n for n in main.body if isinstance(n,ast.Try)).body
 def contains(n,name):return any(isinstance(q,ast.Call) and isinstance(q.func,ast.Attribute) and q.func.attr==name for q in ast.walk(n))
 own=[i for i,n in enumerate(body) if contains(n,'compute_owned')];seam=[i for i,n in enumerate(body) if isinstance(n,ast.If) and contains(n,'conditional_gdn')]
 marks=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='report' and isinstance(t.slice,ast.Constant) and t.slice.value=='computation_terminal_epoch' for t in n.targets)]
 require(len(own)==len(seam)==len(marks)==1 and own[0]<seam[0]<marks[0],'Final original CPU terminal must follow ownedHC AND optionalconditionalGDN')
 terminal=[i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='terminal' for t in n.targets)]
 require(len(terminal)==1 and seam[0]<terminal[0]<marks[0] and isinstance(body[terminal[0]].value,ast.Call),'Actual terminal clock placement differs')
 return {'source_contract':'original CPU terminal after all ownedHC/optional GDN computation','source_order_validated':True,'actual_execution_observed':False}
def observed(report):
 fields=['owned_HC_terminal_epoch','computation_terminal_epoch']+(['conditional_GDN_terminal_epoch'] if report['conditional_captured_inputs_requested'] else [])
 require(all(type(report[k]) in (int,float) and math.isfinite(report[k]) and report[k]>0 for k in fields),'Finite actual computation epochs required')
 require(all(report['computation_terminal_epoch']>=report[k] for k in fields),'Final CPU terminal predates model arithmetic')
 return {k:report[k] for k in fields}
