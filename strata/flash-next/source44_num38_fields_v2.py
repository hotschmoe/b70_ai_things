"""NUM38 field reader; old collection grammar is software, not runtime proof."""
import ast,copy,json,re
from pathlib import Path
import layer0_numerical_qualification_v10 as software
import source44_sdk_binding_v1 as sdk

CONTRACT='sycl/include/strata/core/layer0_numerical_contract.hpp'
EXTRA=(('gdn_core_scaled','LE_F32[48,128]',24576),('gdn_rms_square_sum','LE_F32[48]',192),('gdn_rms_argument','LE_F32[48]',192),('gdn_rms_scale','LE_F32[48]',192),('gdn_rsqrt128','LE_F32[1]',4))

def field_contract():
    wanted=json.loads(sdk.PLAN.read_bytes())['expected_patched_source_sha256'][CONTRACT]
    source=sdk.SDK/'source'/CONTRACT
    sdk.require(sdk.sha(sdk.PLAN)==sdk.PLAN_SHA and sdk.sha(source)==wanted,'Actual44 field contract changed')
    rows=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',source.read_text())
    sdk.require(len(rows)==38 and tuple((n,e,int(b))for n,e,b in rows[-5:])==EXTRA,'Exact NUM38 output-only field roster required')
    sdk.require(sdk.sha(source)==wanted,'Actual44 field contract changed during read')
    return [{'name':n,'encoding':e,'bytes':int(b),'observed':True,'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up'if i in(25,29)else'actual_buffer'}for i,(n,e,b)in enumerate(rows)]

def port():
    node=copy.deepcopy(next(n for n in ast.parse(Path(software.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='collect_frame'))
    counts={33:0,31:0}
    class Counts(ast.NodeTransformer):
        def visit_Constant(self,n):
            if type(n.value)is int and n.value in counts:
                counts[n.value]+=1;return ast.copy_location(ast.Constant({33:38,31:36}[n.value]),n)
            return n
    node=Counts().visit(node)
    sdk.require(counts=={33:1,31:1},'Exact old field quota/source-count port required')
    from source44_num38_lifecycle_v2 import live_blob_bindings
    namespace={**vars(software),'field_contract':field_contract,'live_blob_bindings':live_blob_bindings}
    exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),'[source44-NUM38-field-reader]','exec'),namespace)
    return namespace['collect_frame']

def collect_frame(raw,directory,binding):
    result=port()(raw,Path(directory),binding)
    frame=result['frame']
    sdk.require(type(frame.get('numerical_source_generation'))is int and frame['numerical_source_generation']==44 and frame.get('gdn_core_output_observed')is True,'Actual44 observer provenance required')
    sdk.require(result['source_value_fields']==36 and result['derived_value_fields']==2,'Exact NUM38 source/derived counts required')
    return result
