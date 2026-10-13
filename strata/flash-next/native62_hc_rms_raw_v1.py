"""Current33 raw software grammar, corrected omission formatter and combinedSDK."""
import ast,copy,re
from pathlib import Path
from types import FunctionType,SimpleNamespace
import run_native62_hc_rms_fidelity_v1 as actor
import source59_mode0_fidelity_raw_v3 as prior
import source59_literal1_numeric_lpdisabled_v1 as numeric
c=actor.c;require=actor.require

def field_contract():
 sdk=actor.parent.fresh;require(sdk is not None and sdk.SDK is not None,'Actual combinedSDK must be configured');p=sdk.SDK/'source/sycl/include/strata/core/layer0_numerical_contract.hpp';want=c.read(sdk.PLAN)['expected_patched_source_sha256']['sycl/include/strata/core/layer0_numerical_contract.hpp'];require(c.sha(p)==want,'Actual NUM33 header changed');rows=re.findall(r'\{"([^"]+)","([^"]+)",(\d+)\}',p.read_text());require(len(rows)==33 and sum(int(r[2])for r in rows)==7023048,'Exactly NUM33 original source quota required');return [{'name':n,'encoding':e,'bytes':int(b),'observed':True,'provenance':'DERIVED_gpu_native_exp_from_actual_gate_up'if i in(25,29)else'actual_buffer'}for i,(n,e,b)in enumerate(rows)]
_namespace={**vars(prior),'actor':actor,'field_contract':field_contract,'numeric_software':SimpleNamespace(port=lambda:(numeric.port(),None,None))}
_namespace['frame']=FunctionType(prior.frame.__code__,_namespace,prior.frame.__name__,prior.frame.__defaults__,prior.frame.__closure__)
node=copy.deepcopy(next(n for n in ast.parse(Path(prior.__file__).read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='collect'));count=0;generation=0
for n in ast.walk(node):
 if isinstance(n,ast.ImportFrom)and n.module=='source41_model_cli_contract_v1'and [x.name for x in n.names]==['consume']:n.module='native_qsa41_targets_v1';count+=1
 if isinstance(n,ast.Constant)and type(n.value)is int and n.value==59:n.value=62;generation+=1
require(count==1 and generation==2,'Exact consume correction/two source generation labels required');exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[actual62-existing33-omission-raw]','exec'),_namespace)
collect=_namespace['collect']
