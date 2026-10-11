"""Actual44 NUM38 targets with independent FP64 and conditional core lanes."""
import ast,copy,hashlib
from pathlib import Path
from types import FunctionType
import source43_num10_original_reference_v3 as software
import source44_num38_fields_v2 as fields
import source44_num38_core_comparator_v1 as core
from serial37_canonical_json_v3 import read_unique

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN=HERE/'source44-num38-original-cli-source-plan-v1.json'
require=software.require
SHAPES={**software.SHAPES,'gdn_core_scaled':(48,128),'gdn_rms_square_sum':(48,),'gdn_rms_argument':(48,),'gdn_rms_scale':(48,),'gdn_rsqrt128':(1,)}

def source_binding():
    before=hashlib.sha256(PLAN.read_bytes()).hexdigest();record=read_unique(PLAN)
    for name,want in record['files'].items():require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,'NUM38 reference source changed '+name)
    require(hashlib.sha256(PLAN.read_bytes()).hexdigest()==before,'NUM38 reference plan changed');return before

def array_binding(value):
    a=software.np.asarray(value,dtype='<f4');require(software.np.isfinite(a).all(),'Finite owned diagnostic required')
    return {'shape':list(a.shape),'encoding':'LE_F32','bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def conditional(model,values):
    result=software.conditional(model,values)
    node=copy.deepcopy(next(n for n in ast.parse(Path(core.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='conditional_core'))
    ret=node.body[-1];require(isinstance(ret,ast.Return)and isinstance(ret.value,ast.Dict),'Exact frozen conditional core return required')
    ret.value.keys.append(ast.Constant('owned_F32_core'));ret.value.values.append(ast.Call(func=ast.Name(id='array_binding',ctx=ast.Load()),args=[ast.Name(id='own',ctx=ast.Load())],keywords=[]))
    namespace={**vars(core),'array_binding':array_binding}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[NUM38-conditional-core-output-binding]','exec'),namespace)
    result['native_normalized_qkv_to_cpu_F32_core']=namespace['conditional_core'](values['gdn_normalized_qkv'],values['gdn_decay_beta'][1],values['gdn_state_before'].astype('<f4').tobytes(),float(values['gdn_rsqrt128'][0]),values['gdn_core_scaled'])
    result['native_normalized_qkv_to_cpu_F32_core']['captured_input_bindings']={n:array_binding(values[n])for n in('gdn_normalized_qkv','gdn_decay_beta','gdn_state_before','gdn_rsqrt128')}
    return result

def ports():
    namespace={**vars(software),'PLAN':PLAN,'SHAPES':SHAPES,'field_contract':fields.field_contract,'source_binding':source_binding,'independent_first':core.primary_owned,'conditional':conditional}
    original=ast.parse(Path(software.__file__).read_bytes())
    target=copy.deepcopy(next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=='load_targets'));counts={43:0,33:0}
    class Target(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value)is int and n.value in counts:
                counts[n.value]+=1;return ast.copy_location(ast.Constant({43:44,33:38}[n.value]),n)
            if type(n.value)is str:return ast.copy_location(ast.Constant(n.value.replace('Actual43','Actual44').replace('NUM10','NUM38').replace('raw33','raw38')),n)
            return n
    target=Target().visit(target);require(counts=={43:1,33:2},'Exact old target generation/quota sites required')
    enclosed=copy.deepcopy(next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name=='enclosed'))
    class Labels(ast.NodeTransformer):
        def visit_Constant(self,n):return ast.copy_location(ast.Constant(n.value.replace('current43','current44').replace('Actual43','Actual44')),n)if type(n.value)is str else n
    enclosed=Labels().visit(enclosed)
    exec(compile(ast.fix_missing_locations(ast.Module([target,enclosed],type_ignores=[])),'[NUM38-reference-original-closure-port]','exec'),namespace)
    namespace['evaluate']=FunctionType(software.evaluate.__code__,namespace,argdefs=software.evaluate.__defaults__)
    actual_loader=namespace['load_targets']
    def checked_targets(binding,rows):
        require(all(type(r['layer0']['frame'].get('numerical_source_generation'))is int and r['layer0']['frame']['numerical_source_generation']==44 and r['layer0']['frame'].get('gdn_core_output_observed')is True for r in rows),'Actual44 NUM38 frame provenance required')
        return actual_loader(binding,rows)
    namespace['load_targets']=checked_targets
    return namespace

_PORTS=ports();load_targets=_PORTS['load_targets'];enclosed=_PORTS['enclosed'];metrics=software.metrics
