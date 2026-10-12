"""Source55 generation-only raw inspector port; unchanged instance/clock/Chrome joins."""
import argparse,ast,copy,json,types
from pathlib import Path
import inspect_source52_p2048_profile_v1 as original
import analyze_source52_selected_kernel_unions_v1 as aggregate

def family(name):
 n=name.lower().replace('_','')
 if '::hcq8tokenreusekernel' in n:return 'HC_Q8_projection'
 if '::hcprojectionkernel<false' in n:return 'HC_F32_projection'
 for needle,label in [('::hcnorm','HC_norm'),('::hcsilu','HC_silu'),('::hcmix','HC_mix'),('::hcpending','HC_pending_write'),('::hcstandalonewrite','HC_residual_write')]:
  if needle in n:return label
 return aggregate.family(name)
def detail(instances,devices):
 fn=aggregate.detail;scope={**fn.__globals__,'family':family}
 return types.FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)(instances,devices)
def inspect(directory):
 directory=Path(directory).resolve();r=json.loads((directory/'recipe.json').read_bytes())
 original.require(r['native_env']['STRATA_HC_Q8_TOKEN_TILE']=='2' and r['native_env']['STRATA_HC_PREFILL_COALESCE']=='1' and r['native_env']['STRATA_COPY_ENGINE']=='0','Exact fresh55 selected HC config required')
 node=copy.deepcopy(next(n for n in ast.parse(Path(original.__file__).read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='inspect'));count=0
 for n in ast.walk(node):
  if isinstance(n,ast.Compare)and ast.unparse(n)=="recipe['engine_generation'] == 52":n.comparators[0]=ast.Constant(55);count+=1
 original.require(count==1,'Exactly one inspector generation predicate port required')
 original.require(isinstance(node.body[-1],ast.Return)and isinstance(node.body[-1].value,ast.Dict),'Original inspector result changed')
 node.body.insert(len(node.body)-1,ast.parse('kernel_detail=detail(instances,devices)').body[0]);ret=node.body[-1].value;ret.keys.append(ast.Constant('kernel_detail'));ret.values.append(ast.Name(id='kernel_detail',ctx=ast.Load()))
 scope={**vars(original),'detail':detail};exec(compile(ast.fix_missing_locations(ast.Module([node],type_ignores=[])),__file__+'[actual55-generation-only]','exec'),scope)
 result=scope['inspect'](directory);result.update(source_generation=55,physical_idle_qualified=False,prefill_decode_temporal_split_qualified=False,selected_CPU_command_gap_metrics=None,host_device_alignment_qualified=False,kernel_names_prove_role_or_arithmetic=False);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();original.require(not a.output.exists(),'New external observed-only output required');a.output.write_text(json.dumps(inspect(a.directory),indent=2)+'\n',encoding='ascii')
